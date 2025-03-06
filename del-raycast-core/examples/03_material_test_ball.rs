use del_geo_core::mat4_col_major;
use del_geo_core::vec3;
use del_geo_core::vec3::Vec3;
use del_msh_core::search_bvh3::TriMeshWithBvh;
use del_raycast_core::textures::Texture;

struct Shape {
    vtx2xyz: Vec<f32>,
    tri2vtx: Vec<usize>,
    uvs: Vec<f32>,
    transform: [f32; 16],
    bvhnodes: Vec<usize>,
    bvhnode2aabb: Vec<f32>,
    material_index: Option<usize>,
}

// return scale factor t, shape index and triangle index of intersection
fn intersect_bvh(
    ray_org: &[f32; 3],
    ray_dir: &[f32; 3],
    shapes: &[Shape],
) -> Option<(f32, usize, usize)> {
    // compute intersection
    let mut t_min = f32::INFINITY;
    let mut shape_i: usize = usize::MAX;
    let mut tri_i: usize = usize::MAX;
    for (s_i, trimesh) in shapes.iter().enumerate() {
        let ti = del_geo_core::mat4_col_major::try_inverse(&trimesh.transform).unwrap();
        let ray_org = del_geo_core::mat4_col_major::transform_homogeneous(&ti, ray_org).unwrap();
        let ray_dir = del_geo_core::mat4_col_major::transform_direction(&ti, ray_dir);

        let Some((t, i_tri)) = del_msh_core::search_bvh3::first_intersection_ray(
            &ray_org,
            &ray_dir,
            &TriMeshWithBvh {
                tri2vtx: &trimesh.tri2vtx,
                vtx2xyz: &trimesh.vtx2xyz,
                bvhnodes: &trimesh.bvhnodes,
                bvhnode2aabb: &trimesh.bvhnode2aabb,
            },
            0,
            f32::INFINITY,
        ) else {
            continue;
        };
        if t < t_min {
            shape_i = s_i;
            tri_i = i_tri;
            t_min = t;
        }
    }

    if t_min == f32::INFINITY {
        None
    } else {
        Some((t_min, shape_i, tri_i))
    }
}

// get uv coordinates of position of a triangle tri_i
fn get_tri_uv(tri_i: usize, pos: &[f32; 3], shape: &Shape) -> [f32; 2] {
    let vertices = &shape.tri2vtx[tri_i..tri_i + 3];

    let f: Vec<[f32; 3]> = vertices
        .iter()
        .map(|&vert| {
            assert!(
                vert * 3 + 3 <= shape.vtx2xyz.len(),
                "vertex index out of bounds"
            );
            let v = &shape.vtx2xyz[vert * 3..(vert * 3 + 3)];
            [v[0], v[1], v[2]]
        })
        .collect();

    let bary = del_geo_core::tri3::to_barycentric_coords(&f[0], &f[1], &f[2], pos);

    let vert_uvs: Vec<[f32; 2]> = vertices
        .iter()
        .map(|&vert| {
            assert!(vert * 2 + 2 <= shape.uvs.len(), "uv index out of bounds");
            let uv = &shape.uvs[vert * 2..(vert * 2 + 2)];
            [uv[0], uv[1]]
        })
        .collect();

    let mut acc = [0.; 2];
    vert_uvs.iter().enumerate().for_each(|(i, vert_uv)| {
        acc[0] += vert_uv[0] * bary[i];
        acc[1] += vert_uv[1] * bary[i];
    });

    acc
}

fn parse() -> anyhow::Result<(
    Vec<Shape>,
    del_raycast_core::parse_pbrt::Camera,
    pbrt4::Scene,
)> {
    let path_file = "asset/material-testball/scene-v4.pbrt";
    let scene = pbrt4::Scene::from_file(path_file)?;
    let mut shapes: Vec<Shape> = vec![];
    let camera = del_raycast_core::parse_pbrt::camera(&scene);
    for shape_entity in scene.shapes.iter() {
        let transform = shape_entity.transform.to_cols_array();
        let (_, _, tri2vtx, vtx2xyz, _, uvs) =
            del_raycast_core::parse_pbrt::trimesh3_from_shape_entity(shape_entity, path_file)
                .unwrap();
        let bvhnodes = del_msh_core::bvhnodes_morton::from_triangle_mesh(&tri2vtx, &vtx2xyz, 3);
        let bvhnode2aabb = del_msh_core::bvhnode2aabb3::from_uniform_mesh_with_bvh(
            0,
            &bvhnodes,
            Some((&tri2vtx, 3)),
            &vtx2xyz,
            None,
        );
        let shape = Shape {
            tri2vtx,
            vtx2xyz,
            uvs,
            transform,
            bvhnodes,
            bvhnode2aabb,
            material_index: shape_entity.material_index,
        };
        shapes.push(shape);
    }
    Ok((shapes, camera, scene))
}

fn main() -> anyhow::Result<()> {
    dielectric_sphere();

    let (shapes, camera, scene) = parse()?;
    {
        let mut tri2vtx: Vec<usize> = vec![];
        let mut vtx2xyz: Vec<f32> = vec![];
        for trimesh in shapes.iter() {
            let t = del_geo_core::mat4_col_major::mult_mat_col_major(
                &camera.transform_world2camlcl,
                &trimesh.transform,
            );
            let trimesh_vtx2xyz =
                del_msh_core::vtx2xyz::transform_homogeneous(&trimesh.vtx2xyz, &t);
            del_msh_core::uniform_mesh::merge(
                &mut tri2vtx,
                &mut vtx2xyz,
                &trimesh.tri2vtx,
                &trimesh_vtx2xyz,
                3,
            );
        }
        del_msh_core::io_obj::save_tri2vtx_vtx2xyz(
            "target/03_material_test_ball.obj",
            &tri2vtx,
            &vtx2xyz,
            3,
        )?;
    }

    let transform_cam_lcl2glbl =
        del_geo_core::mat4_col_major::try_inverse(&camera.transform_world2camlcl).unwrap();
    let img_shape = (540, 360);

    {
        // computing depth image
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref![pix, 0, 3];
            let iw = i_pix % img_shape.0;
            let ih = i_pix / img_shape.0;

            let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray_plus_z(
                (iw, ih),
                (0., 0.),
                img_shape,
                camera.camera_fov,
                transform_cam_lcl2glbl,
            );

            let Some((t, _i_shape_entity, _tri_i)) = intersect_bvh(&ray_org, &ray_dir, &shapes)
            else {
                return;
            };
            let v = (t - 1.5) * 0.8;
            *pix = [v; 3];
        };

        let mut img_out = vec![0f32; img_shape.0 * img_shape.1 * 3];
        use rayon::prelude::*;
        img_out
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        del_canvas::write_hdr_file("target/03_material_test_ball.hdr", img_shape, &img_out)?;
    }

    {
        // computing reflectance image
        let materials = del_raycast_core::parse_pbrt::parse_material(&scene);
        let textures = del_raycast_core::parse_pbrt::parse_texture(&scene);
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            let pix = arrayref::array_mut_ref![pix, 0, 3];
            let iw = i_pix % img_shape.0;
            let ih = i_pix / img_shape.0;

            let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray_plus_z(
                (iw, ih),
                (0., 0.),
                img_shape,
                camera.camera_fov,
                transform_cam_lcl2glbl,
            );

            let Some((t, i_shape_entity, tri_i)) = intersect_bvh(&ray_org, &ray_dir, &shapes)
            else {
                return;
            };

            let shape = &shapes[i_shape_entity];

            let i_material = shape.material_index.unwrap();

            assert!(
                i_material < materials.len(),
                "{} {}",
                i_material,
                materials.len()
            );

            let reflectance = match &materials[i_material] {
                del_raycast_core::material::Material::Diff(mat) => {
                    if mat.reflectance_texture != usize::MAX {
                        let tex = &textures[mat.reflectance_texture];
                        match tex {
                            Texture::Checkerboard(tex) => {
                                let pos = del_geo_core::vec3::axpy(t, &ray_dir, &ray_org);
                                let uv = get_tri_uv(tri_i, &pos, shape);
                                del_raycast_core::textures::sample_checkerboard(
                                    &uv, tex.uscale, tex.vscale, &tex.tex1, &tex.tex2,
                                )
                            }
                        }
                    } else {
                        mat.reflectance
                    }
                }
                del_raycast_core::material::Material::Cond(mat) => mat.reflectance,
                del_raycast_core::material::Material::CoaDiff(mat) => mat.reflectance,
                _ => {
                    panic!("No reflectance of Material");
                }
            };
            *pix = reflectance;
        };

        let mut img_out = vec![0f32; img_shape.0 * img_shape.1 * 3];
        use rayon::prelude::*;
        img_out
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        del_canvas::write_hdr_file(
            "target/03_material_test_ball_color.hdr",
            img_shape,
            &img_out,
        )?;
    }

    Ok(())
}

/// render a sphere with dielectric material
fn dielectric_sphere() {
    use arrayref::array_mut_ref;
    use del_geo_core::vec3::Vec3;
    let res_path = "target/test_dielectric.hdr";
    let env_map_path = "asset/material-testball/textures/envmap.pfm";
    let (tex_shape, tex_data) = {
        match del_raycast_core::io_pfm::PFM::read_from(env_map_path) {
            Ok(pfm) => ((pfm.w, pfm.h), pfm.data),
            Err(msg) => {
                panic!("{}", msg);
            }
        }
    };

    let camera_fov = 20.0;
    // let transform_cam_lcl2glbl = del_geo_core::mat4_col_major::from_translate(&[0., 0., -5.]);
    let transform_world2camlcl: [f32; 16] = [
        0.721367, -0.373123, -0.583445, -0., -0., 0.842456, -0.538765, -0., -0.692553, -0.388647,
        -0.60772, -0., 0.0258668, -0.29189, 5.43024, 1.,
    ];
    let transform_camlcl2world =
        del_geo_core::mat4_col_major::try_inverse(&transform_world2camlcl).unwrap();
    let transform_env = [
        -0.386527, 0., 0.922278, 0., -0.922278, 0., -0.386527, 0., 0., 1., 0., 0., 0., 0., 0., 1.,
    ];

    let transform_env: [f32; 16] = {
        let m = nalgebra::Matrix4::<f32>::from_column_slice(&transform_env);
        let m = m.try_inverse().unwrap();
        // let transform_env = del_geo_core::mat4_col_major::try_inverse(&transform_env).unwrap();
        m.as_slice().try_into().unwrap()
    };
    let sphere_cntr = [0.15, 0.50, 0.16];
    let img_shape = (640, 360);
    {
        let shoot_ray = |i_pix: usize, pix: &mut [f32]| {
            use rand::Rng;
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_pix as u64);
            let pix = array_mut_ref![pix, 0, 3];
            let transforms = (
                &transform_world2camlcl,
                &transform_world2camlcl,
                &transform_env,
            );
            let tex_info = (&tex_shape, &tex_data);
            let nsamples = 64;
            let mut rad = [0.; 3];
            for _i_sample in 0..nsamples {
                let dxdy = (
                    del_raycast_core::sampling::tent(rng.random::<f32>()),
                    del_raycast_core::sampling::tent(rng.random::<f32>()),
                );
                let (ray_org, ray_dir) = del_raycast_core::cam_pbrt::cast_ray_plus_z(
                    (i_pix % img_shape.0, i_pix / img_shape.0),
                    dxdy,
                    img_shape,
                    camera_fov,
                    transform_camlcl2world,
                );

                rad.add_in_place(&pt_dielectric(
                    &sphere_cntr,
                    &ray_dir,
                    &ray_org,
                    transforms,
                    tex_info,
                    &mut rng,
                ));
            }
            *pix = rad.scale(1. / nsamples as f32);
        };
        use rayon::iter::IndexedParallelIterator;
        use rayon::iter::ParallelIterator;
        use rayon::prelude::ParallelSliceMut;
        let mut img = vec![0f32; img_shape.0 * img_shape.1 * 3];
        img.par_chunks_mut(3)
            .enumerate()
            .for_each(|(i_pix, pix)| shoot_ray(i_pix, pix));
        let _ = del_canvas::write_hdr_file(res_path, img_shape, &img);
    }
}

fn pt_dielectric<RNG>(
    sphere_cntr: &[f32; 3],
    ray_dir_ini: &[f32; 3],
    ray_org_ini: &[f32; 3],
    transforms: (&[f32; 16], &[f32; 16], &[f32; 16]),
    tex_info: (&(usize, usize), &Vec<f32>),
    rng: &mut RNG,
) -> [f32; 3]
where
    RNG: rand::Rng,
{
    use del_raycast_core::material;

    let max_depth = 65;
    let w2c = transforms.0;
    let c2w = transforms.1;
    let transform_env = transforms.2;
    let tex_shape = tex_info.0;
    let tex_data = tex_info.1;
    let mut ray_org: [f32; 3] = ray_org_ini.to_owned();
    let mut ray_dir: [f32; 3] = ray_dir_ini.to_owned();

    let mut ior = 1.5; // index of refraction of glass
    let uroughness = 1e-4; // small value <1e-3 for smooth suface
    let vroughness = 1e-4;

    let mut rad_out = [0.; 3];
    let mut throughput = [1.; 3];

    for i_depth in 0..max_depth {
        let hit = intersect_sphere_with_normal(0.7, sphere_cntr, &ray_org, &ray_dir);
        if hit.is_none() {
            let nrm = vec3::normalize(&ray_dir);
            let env = mat4_col_major::transform_homogeneous(transform_env, &nrm).unwrap();
            let tex_coord = del_geo_core::uvec3::map_to_unit2_equal_area(&env);

            let color = del_canvas::texture::nearest_integer_center::<3>(
                &[
                    tex_coord[0] * tex_shape.0 as f32,
                    tex_coord[1] * tex_shape.1 as f32,
                ],
                tex_shape,
                tex_data,
            );

            let contribution = throughput.element_wise_mult(&color);
            rad_out = vec3::add(&rad_out, &contribution);
            break;
        }
        let (hit_pos, hit_nrm) = hit.unwrap();

        let entering = vec3::dot(&ray_dir, &hit_nrm) < 0.0;
        let hit_nrm = if entering {
            hit_nrm
        } else {
            //ray exiting sphere, normal point to sphere's center
            hit_nrm.scale(-1.0)
        };

        let wo = vec3::normalize(&ray_dir);
        ior = if entering { 1. / ior } else { ior };
        if let Some((wi, brdf, pdf)) = material::sample_brdf_dielectric(
            &wo,
            &[0.243117, 0.059106, 0.000849],
            &[ior; 3],
            uroughness,
            vroughness,
            rng,
        ) {
            let cos_hit = wi.dot(&hit_nrm).abs();
            throughput = throughput.element_wise_mult(&brdf.scale(cos_hit / pdf));

            if i_depth > 2 {
                // russian roulette
                let &russian_roulette_prob = throughput
                    .iter()
                    .max_by(|&a, &b| a.partial_cmp(b).unwrap())
                    .unwrap();
                if rng.random::<f32>() < russian_roulette_prob {
                    throughput = vec3::scale(&throughput, 1.0 / russian_roulette_prob);
                } else {
                    break; // terminate ray
                }
            }

            ray_dir = wi;
            ray_org = vec3::axpy(1e-4, &wi, &hit_pos); //offset
        } else {
            // internal reflection
            let reflected = vec3::mirror_reflection(&ray_dir, &hit_nrm);
            ray_dir = reflected;
            ray_org = vec3::axpy(1e-4, &reflected, &hit_pos);
        };
    }
    rad_out
}

/// get hit position and normal
fn intersect_sphere_with_normal(
    radius: f32,
    sphere_cntr: &[f32; 3],
    ray_org: &[f32; 3],
    ray_dir: &[f32; 3],
) -> Option<([f32; 3], [f32; 3])> {
    let hit = del_geo_core::sphere::intersection_ray(radius, sphere_cntr, ray_org, ray_dir);
    if hit.is_none() {
        None
    } else {
        let hit_pos = vec3::axpy::<f32>(hit.unwrap(), ray_dir, ray_org);
        let hit_nrm = {
            let nrm_dir = vec3::sub(&hit_pos, sphere_cntr);
            vec3::normalize(&nrm_dir)
        };

        Some((hit_pos, hit_nrm))
    }
}
