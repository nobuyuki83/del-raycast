use del_msh_core::io_svg::svg_outline_path_from_shape;
use image::Pixel;
use rayon::string;

#[derive(Debug, Clone, Default)]
struct TriangleMesh {
    vtx2xyz: Vec<f32>,
    tri2vtx: Vec<usize>,
    vtx2nrm: Vec<f32>,
    reflectance: [f32; 3],
    spectrum: Option<([f32; 3], bool)>,
}

impl TriangleMesh {
    fn normal_at(&self, pos: &[f32; 3], i_tri: usize) -> [f32; 3] {
        assert!(i_tri < self.tri2vtx.len() / 3);
        let iv0 = self.tri2vtx[i_tri * 3];
        let iv1 = self.tri2vtx[i_tri * 3 + 1];
        let iv2 = self.tri2vtx[i_tri * 3 + 2];
        let p0 = arrayref::array_ref![self.vtx2xyz, iv0 * 3, 3];
        let p1 = arrayref::array_ref![self.vtx2xyz, iv1 * 3, 3];
        let p2 = arrayref::array_ref![self.vtx2xyz, iv2 * 3, 3];
        let bc = del_geo_core::tri3::to_barycentric_coords(p0, p1, p2, pos);
        let n0 = arrayref::array_ref![self.vtx2nrm, iv0 * 3, 3];
        let n1 = arrayref::array_ref![self.vtx2nrm, iv1 * 3, 3];
        let n2 = arrayref::array_ref![self.vtx2nrm, iv2 * 3, 3];
        let n = [
            bc[0] * n0[0] + bc[1] * n1[0] + bc[2] * n2[0],
            bc[0] * n0[1] + bc[1] * n1[1] + bc[2] * n2[1],
            bc[0] * n0[2] + bc[1] * n1[2] + bc[2] * n2[2],
        ];
        del_geo_core::vec3::normalized(&n)
    }
}

fn parse_pbrt_file(
    file_path: &str,
) -> anyhow::Result<(Vec<TriangleMesh>, f32, [f32; 16], (usize, usize))> {
    let scene = pbrt4::Scene::from_file(file_path)?;
    let (camera_fov, transform_cam_glbl2lcl, img_shape) =
        del_raycast_core::parse_pbrt::hoge(&scene);
    let mut materials: Vec<[f32; 3]> = vec![];
    for material in scene.materials {
        match material {
            pbrt4::types::Material {
                name,
                attributes,
                reflectance,
                ..
            } => {
                materials.push(reflectance.get_rgb());
            }
            _ => {}
        }
    }
    let mut lights: Vec<([f32; 3], bool)> = vec![];
    for area_light in scene.area_lights {
        match area_light.clone() {
            pbrt4::types::AreaLight::Diffuse {
                filename,
                two_sided,
                spectrum,
                scale,
            } => {
                let spectrum =
                    del_raycast_core::parse_pbrt::spectrum_from_light_entity(&area_light).unwrap();
                lights.push((spectrum, two_sided));
            }
            _ => {}
        }
    }
    let mut shapes: Vec<TriangleMesh> = vec![Default::default(); scene.shapes.len()];
    for (i_shape, shape_entity) in scene.shapes.iter().enumerate() {
        let (material_idx, light_idx, tri2vtx, vtx2xyz, normal) =
            del_raycast_core::parse_pbrt::trimesh3_from_shape_entity(&shape_entity, file_path)
                .unwrap();
        shapes[i_shape].vtx2xyz = vtx2xyz.clone();
        shapes[i_shape].tri2vtx = tri2vtx.clone();
        shapes[i_shape].vtx2nrm = normal.clone();
        if let Some(light_idx) = light_idx {
            // dbg!(i_shape, light_idx);
            shapes[i_shape].spectrum = Some(lights[light_idx]);
        }
        shapes[i_shape].reflectance = materials[material_idx];
    }
    Ok((shapes, camera_fov, transform_cam_glbl2lcl, img_shape))
}

fn intersection_ray_against_trimeshs(
    ray_org: &[f32; 3],
    ray_dir: &[f32; 3],
    trimeshs: &[TriangleMesh],
) -> Option<(f32, usize, usize)> {
    let mut t_trimsh_tri = Option::<(f32, usize, usize)>::None;
    for (i_trimesh, trimesh) in trimeshs.iter().enumerate() {
        let Some((t, i_tri)) = del_msh_core::trimesh3_search_bruteforce::first_intersection_ray(
            &ray_org,
            &ray_dir,
            &trimesh.tri2vtx,
            &trimesh.vtx2xyz,
        ) else {
            continue;
        };
        match t_trimsh_tri {
            None => {
                if t > 0. {
                    t_trimsh_tri = Some((t, i_trimesh, i_tri));
                }
            }
            Some((t_min, _, _)) => {
                if t < t_min && t > 0. {
                    t_trimsh_tri = Some((t, i_trimesh, i_tri));
                }
            }
        }
    }
    t_trimsh_tri
}

fn sampling_light(
    tri2cumsumarea: &[f32],
    r2: [f32; 2],
    triangle_mesh: &TriangleMesh,
) -> ([f32; 3], [f32; 3]) {
    let (i_tri_light, r1, r2) =
        del_msh_core::sampling::sample_uniformly_trimesh(&tri2cumsumarea, r2[0], r2[1]);
    let (p0, p1, p2) = del_msh_core::trimesh3::to_corner_points(
        &triangle_mesh.tri2vtx,
        &triangle_mesh.vtx2xyz,
        i_tri_light,
    );
    let light_pos = del_geo_core::tri3::position_from_barycentric_coords(
        &p0,
        &p1,
        &p2,
        &[1. - r1 - r2, r1, r2],
    );
    let light_nrm = triangle_mesh.normal_at(&light_pos, i_tri_light);
    let light_pos = del_geo_core::vec3::axpy(1.0e-3, &light_nrm, &light_pos);
    (light_pos, light_nrm)
}

fn radiance_material<RNG>(
    ray0_org: &[f32; 3],
    ray0_dir: &[f32; 3],
    i_depth: u32,
    trimeshs: &Vec<TriangleMesh>,
    rng: &mut RNG,
) -> [f32; 3]
where
    RNG: rand::Rng,
{
    let Some((t1, i_trimsh1, i_tri1)) =
        intersection_ray_against_trimeshs(&ray0_org, &ray0_dir, &trimeshs)
    else {
        return [0f32; 3]; // the primal ray does not hit anything..
    };
    let pos1 = del_geo_core::vec3::axpy(t1, &ray0_dir, &ray0_org);
    let nrm1 = trimeshs[i_trimsh1].normal_at(&pos1, i_tri1);
    let emittance1 = if let Some((spectrum1, two_sided1)) = trimeshs[i_trimsh1].spectrum {
        if two_sided1 || (del_geo_core::vec3::dot(&nrm1, &ray0_dir) < 0.) {
            // the primal ray hit light
            spectrum1
        } else {
            [0f32; 3] // backside of the light does not emit light
        }
    } else {
        [0f32; 3] // this triangle mesh does not emit light
    };
    let reflectance1 = &trimeshs[i_trimsh1].reflectance.to_owned();
    let nrm1 = if del_geo_core::vec3::dot(&nrm1, &ray0_dir) > 0. {
        [-nrm1[0], -nrm1[1], -nrm1[2]]
    } else {
        nrm1
    };
    if i_depth > 65 {
        return emittance1;
    }
    let ray1_dir: [f32; 3] = del_raycast_core::sampling::hemisphere_cos_weighted(
        &nalgebra::Vector3::<f32>::new(nrm1[0], nrm1[1], nrm1[2]),
        &[rng.gen::<f32>(), rng.gen::<f32>()],
    )
    .into();
    let ray1_org = del_geo_core::vec3::axpy(1.0e-3, &nrm1, &pos1);
    let iradiance1 = radiance_material(&ray1_org, &ray1_dir, i_depth + 1, trimeshs, rng);
    let tmp0 = del_geo_core::vec3::element_wise_mult(&reflectance1, &iradiance1);
    del_geo_core::vec3::add(&tmp0, &emittance1)
}

fn mse_rgb_error_map(
    target_file: String,
    ground_truth: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
) {
    let mut error =
        vec![image::Rgb([0f32; 3]); (ground_truth.width() * ground_truth.height()) as usize];
    let cal_pixel_mse = |i_pix: usize, pix: &mut image::Rgb<f32>| {
        let ih = i_pix / ground_truth.width() as usize;
        let iw = i_pix % ground_truth.width() as usize;
        let truth = ground_truth.get_pixel(iw as u32, ih as u32).0;
        let output = img[(iw as u32, ih as u32)].0;
        // dbg!(truth, output);
        let mse = ((truth[0] as f32 - output[0] as f32) / 256.0).powf(2.0)
            + ((truth[1] as f32 - output[1] as f32) / 256.0).powf(2.0)
            + ((truth[2] as f32 - output[2] as f32) / 256.0).powf(2.0);
        let err = (mse as f32).sqrt() / (3.0_f32).sqrt();

        *pix = image::Rgb([err; 3]);
    };
    use rayon::prelude::*;
    error
        .par_iter_mut()
        .enumerate()
        .for_each(|(i_pix, pix)| cal_pixel_mse(i_pix, pix));
    use image::codecs::hdr::HdrEncoder;
    let file = std::fs::File::create(target_file).unwrap();
    let enc = HdrEncoder::new(file);
    let _ = enc.encode(
        &error,
        ground_truth.width() as usize,
        ground_truth.height() as usize,
    );
}

fn rmse_error(
    ground_truth: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
) -> f32 {
    let mut up = 0.0;
    let mut down = 0.0;
    let mut cal_pixel_mse = |i_pix: usize| {
        let ih = i_pix / ground_truth.width() as usize;
        let iw = i_pix % ground_truth.width() as usize;
        let truth = ground_truth.get_pixel(iw as u32, ih as u32).0;
        let output = img[(iw as u32, ih as u32)].0;
        // dbg!(truth, output);
        let mse = ((truth[0] as f32 - output[0] as f32) / 256.0).powf(2.0)
            + ((truth[1] as f32 - output[1] as f32) / 256.0).powf(2.0)
            + ((truth[2] as f32 - output[2] as f32) / 256.0).powf(2.0);
        let norm =
            (truth[0] as f32).powf(2.0) + (truth[1] as f32).powf(2.0) + (truth[2] as f32).powf(2.0);
        up += mse;
        down += norm;
    };
    for ih in 0..ground_truth.height() as usize {
        for iw in 0..ground_truth.width() as usize {
            cal_pixel_mse(ih * ground_truth.width() as usize + iw);
        }
    }
    up / down
}

fn main() -> anyhow::Result<()> {
    let pbrt_file_path = "asset/cornell-box/scene-v4.pbrt";
    let (trimeshs, camera_fov, transform_cam_glbl2lcl, img_shape) =
        parse_pbrt_file(pbrt_file_path)?;
    {
        let mut tri2vtx: Vec<usize> = vec![];
        let mut vtx2xyz: Vec<f32> = vec![];
        for trimesh in trimeshs.iter() {
            let trimesh_vtx2xyz =
                del_msh_core::vtx2xyz::transform(&trimesh.vtx2xyz, &transform_cam_glbl2lcl);
            del_msh_core::uniform_mesh::merge(
                &mut tri2vtx,
                &mut vtx2xyz,
                &trimesh.tri2vtx,
                &trimesh_vtx2xyz,
                3,
            );
        }
        del_msh_core::io_obj::save_tri2vtx_vtx2xyz(
            "target/02_cornell_box.obj",
            &tri2vtx,
            &vtx2xyz,
            3,
        )?;
    }
    let transform_cam_lcl2glbl =
        del_geo_core::mat4_col_major::try_inverse(&transform_cam_glbl2lcl).unwrap();
    use itertools::Itertools;
    // Get the area light source
    let i_trimesh_light = trimeshs
        .iter()
        .enumerate()
        .find_or_first(|(_i_trimesh, trimsh)| trimsh.spectrum.is_some())
        .unwrap()
        .0;
    let tri2cumsumarea = del_msh_core::sampling::cumulative_area_sum(
        &trimeshs[i_trimesh_light].tri2vtx,
        &trimeshs[i_trimesh_light].vtx2xyz,
        3,
    );
    let &area_light = tri2cumsumarea.last().unwrap();
    let ground_truth = image::open("asset/cornell-box/TungstenRender.png")
        .unwrap()
        .to_rgb8();
    assert!(ground_truth.dimensions() == (img_shape.0 as u32, img_shape.1 as u32));
    {
        // computing depth image
        let mut img = vec![image::Rgb([0f32; 3]); img_shape.0 * img_shape.1];
        let shoot_ray = |i_pix: usize, pix: &mut image::Rgb<f32>| {
            let ih = i_pix / img_shape.0;
            let iw = i_pix % img_shape.0;
            let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray(
                iw,
                ih,
                img_shape,
                camera_fov,
                transform_cam_lcl2glbl,
            );
            // compute intersection below
            let mut t_min = f32::INFINITY;
            let mut color_buf = [0.0, 0.0, 0.0];
            for trimesh in trimeshs.iter() {
                let Some((t, _i_tri)) =
                    del_msh_core::trimesh3_search_bruteforce::first_intersection_ray(
                        &ray_org,
                        &ray_dir,
                        &trimesh.tri2vtx,
                        &trimesh.vtx2xyz,
                    )
                else {
                    continue;
                };
                if t < t_min {
                    t_min = t;
                    color_buf = trimesh.reflectance;
                }
            }
            let v = t_min * 0.05;
            *pix = image::Rgb([v; 3]);
        };
        use rayon::prelude::*;
        img.par_iter_mut()
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        use image::codecs::hdr::HdrEncoder;
        let file = std::fs::File::create("target/02_cornell_box_depth.hdr").unwrap();
        let enc = HdrEncoder::new(file);
        let _ = enc.encode(&img, img_shape.0, img_shape.1);
    }
    {
        // computing reflectance image
        let mut img = vec![image::Rgb([0f32; 3]); img_shape.0 * img_shape.1];
        let shoot_ray = |i_pix: usize, pix: &mut image::Rgb<f32>| {
            let ih = i_pix / img_shape.0;
            let iw = i_pix % img_shape.0;
            let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray(
                iw,
                ih,
                img_shape,
                camera_fov,
                transform_cam_lcl2glbl,
            );
            let Some((_t, i_trimsh, _i_tri)) =
                intersection_ray_against_trimeshs(&ray_org, &ray_dir, &trimeshs)
            else {
                return;
            };
            *pix = image::Rgb(trimeshs[i_trimsh].reflectance);
        };
        use rayon::prelude::*;
        img.par_iter_mut()
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        let file1 = std::fs::File::create("target/02_cornell_box_color.hdr").unwrap();
        use image::codecs::hdr::HdrEncoder;
        let enc = HdrEncoder::new(file1);
        let _ = enc.encode(&img, img_shape.0, img_shape.1);
    }
    println!("---------------------light sampling---------------------");
    for i in 0..30 {
        // light sampling
        let num_sample = 8 + 2 * i;
        let shoot_ray = |i_pix: usize, pix: &mut image::Rgb<f32>| {
            use rand::Rng;
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64);
            let ih = i_pix / img_shape.0;
            let iw = i_pix % img_shape.0;
            let mut l_o = [0., 0., 0.];
            for _i_sample in 0..num_sample {
                let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray(
                    iw,
                    ih,
                    img_shape,
                    camera_fov,
                    transform_cam_lcl2glbl,
                );
                let Some((ray_t, i_trimsh_hit, i_tri_hit)) =
                    intersection_ray_against_trimeshs(&ray_org, &ray_dir, &trimeshs)
                else {
                    return;
                };
                if trimeshs[i_trimsh_hit].spectrum.is_some() {
                    *pix = image::Rgb(trimeshs[i_trimsh_hit].spectrum.unwrap().0);
                    return;
                }
                use del_geo_core::vec3;
                let hit_pos = vec3::axpy(ray_t, &ray_dir, &ray_org);
                let hit_nrm = trimeshs[i_trimsh_hit].normal_at(&hit_pos, i_tri_hit);
                let hit_nrm = if vec3::dot(&hit_nrm, &ray_dir) > 0. {
                    [-hit_nrm[0], -hit_nrm[1], -hit_nrm[2]]
                } else {
                    hit_nrm
                };
                let hit_pos = vec3::axpy(1.0e-3, &hit_nrm, &hit_pos);
                let (light_pos, light_nrm) = sampling_light(
                    &tri2cumsumarea,
                    [rng.gen(), rng.gen()],
                    &trimeshs[i_trimesh_light],
                );
                //
                let uvec_from_hit_to_light = vec3::normalized(&vec3::sub(&light_pos, &hit_pos));
                let cos_theta_hit = vec3::dot(&hit_nrm, &uvec_from_hit_to_light);
                let cos_theta_light = -vec3::dot(&light_nrm, &uvec_from_hit_to_light);
                if cos_theta_light < 0. {
                    continue;
                } // backside of light
                if let Some((_t, i_trimsh, _i_tri)) =
                    intersection_ray_against_trimeshs(&hit_pos, &uvec_from_hit_to_light, &trimeshs)
                {
                    if i_trimsh != i_trimesh_light {
                        continue;
                    }
                } else {
                    continue;
                };
                let l_i = trimeshs[i_trimesh_light].spectrum.unwrap().0;
                let pdf = 1.0 / area_light;
                let r2 = del_geo_core::edge3::squared_length(&light_pos, &hit_pos);
                let reflectance = trimeshs[i_trimsh_hit].reflectance;
                let li_r = vec3::element_wise_mult(&l_i, &reflectance);
                let tmp = cos_theta_hit * cos_theta_light / (r2 * pdf * num_sample as f32);
                l_o = vec3::axpy(tmp, &li_r, &l_o);
            }
            *pix = image::Rgb(l_o);
        };
        use rayon::prelude::*;
        let mut img = vec![image::Rgb([0f32; 3]); img_shape.0 * img_shape.1];
        img.par_iter_mut()
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        let target_file =
            "target/02_cornell_box_light_sampling".to_string() + &(&i + 1).to_string() + ".hdr";
        let file1 = std::fs::File::create(target_file.clone()).unwrap();
        use image::codecs::hdr::HdrEncoder;
        let enc = HdrEncoder::new(file1);
        let _ = enc.encode(&img, img_shape.0, img_shape.1);
        let output = image::open(&target_file).unwrap().to_rgb8();
        let target_file = "target/02_cornell_box_light_sampling".to_string()
            + &(&i + 1).to_string()
            + "_error_map.hdr";
        mse_rgb_error_map(target_file, &ground_truth, &output);
        let err = rmse_error(&ground_truth, &output);
        println!("num_sample: {}, mse: {}", num_sample, err);
    }
    println!("---------------------material sampling---------------------");
    for i in 0..30 {
        // material sampling
        let num_sample = 8 + 2 * i;
        let shoot_ray = |i_pix: usize, pix: &mut image::Rgb<f32>| {
            use rand::Rng;
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64);
            let ih = i_pix / img_shape.0;
            let iw = i_pix % img_shape.0;
            let mut l_o = [0., 0., 0.];
            for _i_sample in 0..num_sample {
                let (ray0_org, ray0_dir) = del_raycast_core::cam_pbrt::cast_ray(
                    iw,
                    ih,
                    img_shape,
                    camera_fov,
                    transform_cam_lcl2glbl,
                );
                let Some((t1, i_trimsh1, i_tri1)) =
                    intersection_ray_against_trimeshs(&ray0_org, &ray0_dir, &trimeshs)
                else {
                    continue; // the primal ray does not hit anything..
                };
                let pos1 = del_geo_core::vec3::axpy(t1, &ray0_dir, &ray0_org);
                let nrm1 = trimeshs[i_trimsh1].normal_at(&pos1, i_tri1);
                if let Some((spectrum1, two_sided1)) = trimeshs[i_trimsh1].spectrum {
                    // the primal ray hit light
                    if two_sided1 || (del_geo_core::vec3::dot(&nrm1, &ray0_dir) < 0.) {
                        l_o[0] += spectrum1[0];
                        l_o[1] += spectrum1[1];
                        l_o[2] += spectrum1[2];
                        continue;
                    }
                }
                let nrm1 = if del_geo_core::vec3::dot(&nrm1, &ray0_dir) > 0. {
                    [-nrm1[0], -nrm1[1], -nrm1[2]]
                } else {
                    nrm1
                };
                let ray1_dir: [f32; 3] = del_raycast_core::sampling::hemisphere_cos_weighted(
                    &nalgebra::Vector3::<f32>::new(nrm1[0], nrm1[1], nrm1[2]),
                    &[rng.gen::<f32>(), rng.gen::<f32>()],
                )
                .into();
                let ray1_org = del_geo_core::vec3::axpy(1.0e-3, &nrm1, &pos1);
                let Some((t2, i_trimsh2, i_tri2)) =
                    intersection_ray_against_trimeshs(&ray1_org, &ray1_dir, &trimeshs)
                else {
                    continue;
                };
                let Some((spectrum2, two_sided2)) = trimeshs[i_trimsh2].spectrum else {
                    continue;
                };
                let pos2 = del_geo_core::vec3::axpy(t2, &ray1_dir, &ray1_org);
                let nrm2 = trimeshs[i_trimsh2].normal_at(&pos2, i_tri2);
                if !two_sided2 && (del_geo_core::vec3::dot(&nrm2, &ray1_dir) > 0.) {
                    continue;
                }
                l_o[0] += spectrum2[0] * trimeshs[i_trimsh1].reflectance[0];
                l_o[1] += spectrum2[1] * trimeshs[i_trimsh1].reflectance[1];
                l_o[2] += spectrum2[2] * trimeshs[i_trimsh1].reflectance[2];
            }
            (*pix).0[0] = l_o[0] / num_sample as f32;
            (*pix).0[1] = l_o[1] / num_sample as f32;
            (*pix).0[2] = l_o[2] / num_sample as f32;
        };
        let mut img = vec![image::Rgb([0f32; 3]); img_shape.0 * img_shape.1];
        use rayon::iter::IndexedParallelIterator;
        use rayon::iter::IntoParallelRefMutIterator;
        use rayon::iter::ParallelIterator;
        img.par_iter_mut()
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        let target_file =
            "target/02_cornell_box_material_sampling".to_string() + &(&i + 1).to_string() + ".hdr";
        let file2 = std::fs::File::create(&target_file).unwrap();
        use image::codecs::hdr::HdrEncoder;
        let enc = HdrEncoder::new(file2);
        let _ = enc.encode(&img, img_shape.0, img_shape.1);
        let output = image::open(target_file).unwrap().to_rgb8();
        let target_file = "target/02_cornell_box_pt_nee_material_error_map".to_string()
            + &(&i + 1).to_string()
            + ".hdr";
        mse_rgb_error_map(target_file, &ground_truth, &output);
        let err = rmse_error(&ground_truth, &output);
        println!(
            "num_sample: {}, mse: {}",
            num_sample, err
        );
    }
    println!("---------------------MIS sampling---------------------");
    for i in 0..30 {
        // Multiple importance sampling on both light and BSDF, use balance heuristic to get the weight for different sampling strategies
        let num_sample_light = 8 + 2 * i;
        let num_sample_bsdf = 8 + 2 * i;
        let c_light = num_sample_light as f32 / (num_sample_light + num_sample_bsdf) as f32;
        let c_bsdf = num_sample_bsdf as f32 / (num_sample_light + num_sample_bsdf) as f32;
        let shoot_ray = |i_pix: usize, pix: &mut image::Rgb<f32>| {
            use del_geo_core::vec3;
            use rand::Rng;
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64);
            let ih = i_pix / img_shape.0;
            let iw = i_pix % img_shape.0;
            let mut l_o = [0., 0., 0.];
            for _i_sample in 0..num_sample_light {
                let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray(
                    iw,
                    ih,
                    img_shape,
                    camera_fov,
                    transform_cam_lcl2glbl,
                );
                let Some((ray_t, i_trimsh_hit, i_tri_hit)) =
                    intersection_ray_against_trimeshs(&ray_org, &ray_dir, &trimeshs)
                else {
                    continue; // the primal ray does not hit anything..
                };
                if trimeshs[i_trimsh_hit].spectrum.is_some() {
                    // the primal ray hit light
                    l_o[0] += trimeshs[i_trimsh_hit].spectrum.unwrap().0[0];
                    l_o[1] += trimeshs[i_trimsh_hit].spectrum.unwrap().0[1];
                    l_o[2] += trimeshs[i_trimsh_hit].spectrum.unwrap().0[2];
                    continue;
                }
                let hit_pos = vec3::axpy(ray_t, &ray_dir, &ray_org);
                let hit_nrm = trimeshs[i_trimsh_hit].normal_at(&hit_pos, i_tri_hit);
                let hit_nrm = if vec3::dot(&hit_nrm, &ray_dir) > 0. {
                    [-hit_nrm[0], -hit_nrm[1], -hit_nrm[2]]
                } else {
                    hit_nrm
                };
                let hit_pos = vec3::axpy(1.0e-3, &hit_nrm, &hit_pos);
                let (light_pos, light_nrm) = sampling_light(
                    &tri2cumsumarea,
                    [rng.gen(), rng.gen()],
                    &trimeshs[i_trimesh_light],
                );
                let uvec_from_hit_to_light = vec3::normalized(&vec3::sub(&light_pos, &hit_pos));
                let cos_theta_hit = vec3::dot(&hit_nrm, &uvec_from_hit_to_light);
                let cos_theta_light = -vec3::dot(&light_nrm, &uvec_from_hit_to_light);
                if cos_theta_light < 0. {
                    continue;
                } // backside of light
                if let Some((_t, i_trimsh, _i_tri)) =
                    intersection_ray_against_trimeshs(&hit_pos, &uvec_from_hit_to_light, &trimeshs)
                {
                    if i_trimsh != i_trimesh_light {
                        continue;
                    }
                } else {
                    continue;
                };
                let l_i = trimeshs[i_trimesh_light].spectrum.unwrap().0;
                let pdf_light = 1.0 / area_light;
                let pdf_bsdf = 1.0 / (2.0 * std::f32::consts::PI);
                let r2 = del_geo_core::edge3::squared_length(&light_pos, &hit_pos);
                let reflectance = trimeshs[i_trimsh_hit].reflectance;
                let li_r = vec3::element_wise_mult(&l_i, &reflectance);
                let tmp = cos_theta_hit * cos_theta_light
                    / (r2 * (pdf_light * c_light + pdf_bsdf * c_bsdf) as f32);
                l_o = vec3::axpy(tmp, &li_r, &l_o);
            }
            for _i_sample in 0..num_sample_bsdf {
                let (ray0_org, ray0_dir) = del_raycast_core::cam_pbrt::cast_ray(
                    iw,
                    ih,
                    img_shape,
                    camera_fov,
                    transform_cam_lcl2glbl,
                );
                let Some((t1, i_trimsh1, i_tri1)) =
                    intersection_ray_against_trimeshs(&ray0_org, &ray0_dir, &trimeshs)
                else {
                    continue; // the primal ray does not hit anything..
                };
                let pos1 = vec3::axpy(t1, &ray0_dir, &ray0_org);
                let nrm1 = trimeshs[i_trimsh1].normal_at(&pos1, i_tri1);
                if let Some((spectrum1, two_sided1)) = trimeshs[i_trimsh1].spectrum {
                    // the primal ray hit light
                    if two_sided1 || (vec3::dot(&nrm1, &ray0_dir) < 0.) {
                        l_o[0] += spectrum1[0];
                        l_o[1] += spectrum1[1];
                        l_o[2] += spectrum1[2];
                        continue;
                    }
                }
                let nrm1 = if vec3::dot(&nrm1, &ray0_dir) > 0. {
                    [-nrm1[0], -nrm1[1], -nrm1[2]]
                } else {
                    nrm1
                };
                let ray1_dir: [f32; 3] = del_raycast_core::sampling::hemisphere_cos_weighted(
                    &nalgebra::Vector3::<f32>::new(nrm1[0], nrm1[1], nrm1[2]),
                    &[rng.gen::<f32>(), rng.gen::<f32>()],
                )
                .into();
                let ray1_org = vec3::axpy(1.0e-3, &nrm1, &pos1);
                let Some((t2, i_trimsh2, i_tri2)) =
                    intersection_ray_against_trimeshs(&ray1_org, &ray1_dir, &trimeshs)
                else {
                    continue;
                };
                let Some((spectrum2, two_sided2)) = trimeshs[i_trimsh2].spectrum else {
                    continue;
                };
                let pos2 = vec3::axpy(t2, &ray1_dir, &ray1_org);
                let nrm2 = trimeshs[i_trimsh2].normal_at(&pos2, i_tri2);
                if !two_sided2 && (vec3::dot(&nrm2, &ray1_dir) > 0.) {
                    continue;
                }
                l_o[0] += spectrum2[0] * trimeshs[i_trimsh1].reflectance[0];
                l_o[1] += spectrum2[1] * trimeshs[i_trimsh1].reflectance[1];
                l_o[2] += spectrum2[2] * trimeshs[i_trimsh1].reflectance[2];
            }
            (*pix).0[0] = l_o[0] / (num_sample_light + num_sample_bsdf) as f32;
            (*pix).0[1] = l_o[1] / (num_sample_light + num_sample_bsdf) as f32;
            (*pix).0[2] = l_o[2] / (num_sample_light + num_sample_bsdf) as f32;
        };
        let mut img = vec![image::Rgb([0f32; 3]); img_shape.0 * img_shape.1];
        use rayon::iter::IndexedParallelIterator;
        use rayon::iter::IntoParallelRefMutIterator;
        use rayon::iter::ParallelIterator;
        img.par_iter_mut()
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        let target_file = format!("target/02_cornell_box_mis_{}.hdr", i);
        let file2 = std::fs::File::create(&target_file).unwrap();
        use image::codecs::hdr::HdrEncoder;
        let enc = HdrEncoder::new(file2);
        let _ = enc.encode(&img, img_shape.0, img_shape.1);
        let output = image::open(target_file).unwrap().to_rgb8();
        let target_file = format!("target/MIS_error_map_{}.hdr", i);
        mse_rgb_error_map(
            target_file,
            &ground_truth,
            &output,
        );
        let err = rmse_error(&ground_truth, &output);
        println!(
            "num_sample_light: {}, num_sample_bsdf: {}, mse: {}",
            num_sample_light, num_sample_bsdf, err
        );
    }
    println!("---------------------path tracer---------------------");
    for i in 0..30 {
        // path tracing sampling material
        let num_sample = 8 + 2 * i;
        let shoot_ray = |i_pix: usize, pix: &mut image::Rgb<f32>| {
            use rand::Rng;
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64);
            let ih = i_pix / img_shape.0;
            let iw = i_pix % img_shape.0;
            let mut l_o = [0., 0., 0.];
            for _i_sample in 0..num_sample {
                let (ray0_org, ray0_dir) = del_raycast_core::cam_pbrt::cast_ray(
                    iw,
                    ih,
                    img_shape,
                    camera_fov,
                    transform_cam_lcl2glbl,
                );
                let rad = radiance_material(&ray0_org, &ray0_dir, 0, &trimeshs, &mut rng);
                l_o[0] += rad[0];
                l_o[1] += rad[1];
                l_o[2] += rad[2];
            }
            (*pix).0[0] = l_o[0] / num_sample as f32;
            (*pix).0[1] = l_o[1] / num_sample as f32;
            (*pix).0[2] = l_o[2] / num_sample as f32;
        };
        let mut img = vec![image::Rgb([0f32; 3]); img_shape.0 * img_shape.1];
        use rayon::iter::IndexedParallelIterator;
        use rayon::iter::IntoParallelRefMutIterator;
        use rayon::iter::ParallelIterator;
        img.par_iter_mut()
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        let target_file =
            "target/02_cornell_box_pt_material".to_string() + &(&i + 1).to_string() + ".hdr";
        let file2 = std::fs::File::create(&target_file).unwrap();
        use image::codecs::hdr::HdrEncoder;
        let enc = HdrEncoder::new(file2);
        let _ = enc.encode(&img, img_shape.0, img_shape.1);
        let output = image::open(&target_file).unwrap().to_rgb8();
        let target_file = "target/02_cornell_box_pt_material".to_string()
            + &(&i + 1).to_string()
            + "_error_map.hdr";
        mse_rgb_error_map(target_file, &ground_truth, &output);
        let err = rmse_error(&ground_truth, &output);
        println!("num_sample: {}, mse: {}", num_sample, err);
    }
    println!("---------------------NEE tracer---------------------");
    for i in 0..30 {
        // path tracing next event estimation
        let num_sample_light = 8 + 2 * i;
        let num_sample_material = 8 + 2 * i;
        let c_light = num_sample_light as f32 / (num_sample_light + num_sample_material) as f32;
        let c_bsdf = num_sample_material as f32 / (num_sample_light + num_sample_material) as f32;
        let shoot_ray = |i_pix: usize, pix: &mut image::Rgb<f32>| {
            use rand::Rng;
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64);
            let ih = i_pix / img_shape.0;
            let iw = i_pix % img_shape.0;
            let mut l_brdf = [0., 0., 0.];
            for _i_sample in 0..num_sample_material {
                let (ray0_org, ray0_dir) = del_raycast_core::cam_pbrt::cast_ray(
                    iw,
                    ih,
                    img_shape,
                    camera_fov,
                    transform_cam_lcl2glbl,
                );
                let rad = radiance_material(&ray0_org, &ray0_dir, 0, &trimeshs, &mut rng);
                l_brdf[0] += rad[0] / num_sample_material as f32;
                l_brdf[1] += rad[1] / num_sample_material as f32;
                l_brdf[2] += rad[2] / num_sample_material as f32;
            }
            let mut l_light = [0., 0., 0.];
            for _i_sample in 0..num_sample_light {
                let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray(
                    iw,
                    ih,
                    img_shape,
                    camera_fov,
                    transform_cam_lcl2glbl,
                );
                let Some((ray_t, i_trimsh_hit, i_tri_hit)) =
                    intersection_ray_against_trimeshs(&ray_org, &ray_dir, &trimeshs)
                else {
                    return;
                };
                if trimeshs[i_trimsh_hit].spectrum.is_some() {
                    *pix = image::Rgb(trimeshs[i_trimsh_hit].spectrum.unwrap().0);
                    return;
                }
                use del_geo_core::vec3;
                let hit_pos = vec3::axpy(ray_t, &ray_dir, &ray_org);
                let hit_nrm = trimeshs[i_trimsh_hit].normal_at(&hit_pos, i_tri_hit);
                let hit_nrm = if vec3::dot(&hit_nrm, &ray_dir) > 0. {
                    [-hit_nrm[0], -hit_nrm[1], -hit_nrm[2]]
                } else {
                    hit_nrm
                };
                let hit_pos = vec3::axpy(1.0e-3, &hit_nrm, &hit_pos);
                let (light_pos, light_nrm) = sampling_light(
                    &tri2cumsumarea,
                    [rng.gen(), rng.gen()],
                    &trimeshs[i_trimesh_light],
                );
                //
                let uvec_from_hit_to_light = vec3::normalized(&vec3::sub(&light_pos, &hit_pos));
                let cos_theta_hit = vec3::dot(&hit_nrm, &uvec_from_hit_to_light);
                let cos_theta_light = -vec3::dot(&light_nrm, &uvec_from_hit_to_light);
                if cos_theta_light < 0. {
                    continue;
                } // backside of light
                if let Some((_t, i_trimsh, _i_tri)) =
                    intersection_ray_against_trimeshs(&hit_pos, &uvec_from_hit_to_light, &trimeshs)
                {
                    if i_trimsh != i_trimesh_light {
                        continue;
                    }
                } else {
                    continue;
                };
                let l_i = trimeshs[i_trimesh_light].spectrum.unwrap().0;
                let pdf = 1.0 / area_light;
                let r2 = del_geo_core::edge3::squared_length(&light_pos, &hit_pos);
                let reflectance = trimeshs[i_trimsh_hit].reflectance;
                let li_r = vec3::element_wise_mult(&l_i, &reflectance);
                let tmp = cos_theta_hit * cos_theta_light / (r2 * pdf * num_sample_light as f32);
                l_light = vec3::axpy(tmp, &li_r, &l_light);
            }
            (*pix).0[0] = c_light * l_light[0] + c_bsdf * l_brdf[0];
            (*pix).0[1] = c_light * l_light[1] + c_bsdf * l_brdf[1];
            (*pix).0[2] = c_light * l_light[2] + c_bsdf * l_brdf[2];
        };
        let mut img = vec![image::Rgb([0f32; 3]); img_shape.0 * img_shape.1];
        use rayon::iter::IndexedParallelIterator;
        use rayon::iter::IntoParallelRefMutIterator;
        use rayon::iter::ParallelIterator;
        img.par_iter_mut()
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        let target_file =
            "target/02_cornell_box_pt_nee_material".to_string() + &(&i + 1).to_string() + ".hdr";
        let file2 = std::fs::File::create(&target_file).unwrap();
        use image::codecs::hdr::HdrEncoder;
        let enc = HdrEncoder::new(file2);
        let _ = enc.encode(&img, img_shape.0, img_shape.1);
        let output = image::open(&target_file).unwrap().to_rgb8();
        let target_file = "target/02_cornell_box_pt_nee_material_error_map".to_string()
            + &(&i + 1).to_string()
            + ".hdr";
        mse_rgb_error_map(target_file, &ground_truth, &output);
        let err = rmse_error(&ground_truth, &output);
        println!(
            "num_sample_light: {}, num_sample_material: {}, mse: {}",
            num_sample_light, num_sample_material, err
        );
    }
    Ok(())
}
