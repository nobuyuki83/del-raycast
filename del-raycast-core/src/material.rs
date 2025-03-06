#[derive(Debug)]
pub enum Material {
    None,
    Diff(DiffuseMaterial),
    Cond(ConductorMaterial),
    CoaDiff(CoatedDiffuse),
}

#[derive(Debug)]
pub struct DiffuseMaterial {
    pub reflectance: [f32; 3],
    pub reflectance_texture: usize, // valid if != usize:MAX
}

#[derive(Debug)]
pub struct ConductorMaterial {
    pub uroughness: f32,
    pub vroughness: f32,
    pub reflectance: [f32; 3],
    pub k: [f32; 3],
    pub eta: [f32; 3],
}

#[derive(Debug)]
pub struct DielectricMaterial {
    pub uroughness: f32,
    pub vroughness: f32,
    pub remaproughness: bool,
    pub eta: [f32; 3],
}

#[derive(Debug)]
pub struct CoatedDiffuse {
    pub uroughness: f32,
    pub vroughness: f32,
    pub reflectance: [f32; 3],
    pub remaproughness: bool,
}

pub fn sample_brdf_diffuse<RNG>(reflectance: &[f32; 3], rng: &mut RNG) -> ([f32; 3], [f32; 3], f32)
where
    RNG: rand::Rng,
{
    let ray_dir_next =
        crate::sampling::hemisphere_zup_cos_weighted(&[rng.random::<f32>(), rng.random::<f32>()]);
    use del_geo_core::vec3::Vec3;
    let brdf = reflectance.scale(std::f32::consts::FRAC_1_PI);
    let cos_hit = ray_dir_next[2].clamp(f32::EPSILON, 1f32);
    let pdf = cos_hit * std::f32::consts::FRAC_1_PI;
    (ray_dir_next, brdf, pdf)
}

pub fn eval_brdf_diffuse(reflectance: &[f32; 3]) -> [f32; 3] {
    del_geo_core::vec3::scale(reflectance, 1. / std::f32::consts::PI)
}

// --------------------------

pub fn microfacet_beckmann_roughness_to_alpha(roughness: f32) -> f32 {
    roughness
}

pub fn microfacet_beckmann_sample(alpha: f32, xi: &[f32; 2]) -> [f32; 3] {
    let phi = xi[1] * std::f32::consts::PI * 2f32;
    let tan_theta_sq = -alpha * alpha * (1f32 - xi[0]).ln();
    let cos_theta = 1f32 / (1f32 + tan_theta_sq).sqrt();
    let r = (1f32 - cos_theta * cos_theta).max(0f32).sqrt();
    [phi.cos() * r, phi.sin() * r, cos_theta]
}

pub fn microfacet_beckman_g1(alpha: f32, v: &[f32; 3], m: &[f32; 3]) -> f32 {
    use del_geo_core::vec3;
    if vec3::dot(v, m) * v[2] <= 0f32 {
        return 0f32;
    }
    let cos_theta_sq = v[2] * v[2];
    let tan_theta = ((1f32 - cos_theta_sq).max(0f32).sqrt() / v[2]).abs();
    let a = 1f32 / (alpha * tan_theta);
    if a < 1.6f32 {
        (3.535f32 * a + 2.181f32 * a * a) / (1f32 + 2.276f32 * a + 2.577f32 * a * a)
    } else {
        1f32
    }
}

pub fn microfacet_distribution_g(alpha: f32, i: &[f32; 3], o: &[f32; 3], m: &[f32; 3]) -> f32 {
    let g1i = microfacet_beckman_g1(alpha, i, m);
    let g1o = microfacet_beckman_g1(alpha, o, m);
    g1i * g1o
}

fn microfacet_beckmann_d(alpha: f32, m: &[f32; 3]) -> f32 {
    if m[2] <= 0f32 {
        return 0f32;
    }
    let alpha_sq = alpha * alpha;
    let cos_theta_sq = m[2] * m[2];
    let tan_theta_sq = (1f32 - cos_theta_sq).max(0f32) / cos_theta_sq;
    let cos_theta_qu = cos_theta_sq * cos_theta_sq;
    let d =
        std::f32::consts::FRAC_1_PI * (-tan_theta_sq / alpha_sq).exp() / (alpha_sq * cos_theta_qu);
    if d < f32::EPSILON {
        0f32
    } else {
        d
    }
}

/// the pdf of microfacet normal `m`
pub fn microfacet_beckmann_pdf(alpha: f32, m: &[f32; 3]) -> f32 {
    microfacet_beckmann_d(alpha, m) * m[2]
}

/// From "PHYSICALLY BASED LIGHTING CALCULATIONS FOR COMPUTER GRAPHICS" by Peter Shirley
/// <http://www.cs.virginia.edu/~jdl/bib/globillum/shirley_thesis.pdf>
pub fn fresnel_conductor_reflectance(eta: f32, k: f32, cos_theta_i: f32) -> f32 {
    let cos_theta_isq = cos_theta_i * cos_theta_i;
    let sin_theta_isq = (1f32 - cos_theta_isq).max(0f32);
    let sin_theta_iqu = sin_theta_isq * sin_theta_isq;

    let inner_term = eta * eta - k * k - sin_theta_isq;
    let a_sq_plus_bsq = (inner_term * inner_term + 4f32 * eta * eta * k * k)
        .max(0f32)
        .sqrt();
    let a = ((a_sq_plus_bsq + inner_term) * 0.5f32).max(0f32).sqrt();

    let rs = ((a_sq_plus_bsq + cos_theta_isq) - (2f32 * a * cos_theta_i))
        / ((a_sq_plus_bsq + cos_theta_isq) + (2f32 * a * cos_theta_i));
    let rp = ((cos_theta_isq * a_sq_plus_bsq + sin_theta_iqu)
        - (2f32 * a * cos_theta_i * sin_theta_isq))
        / ((cos_theta_isq * a_sq_plus_bsq + sin_theta_iqu)
            + (2f32 * a * cos_theta_i * sin_theta_isq));
    0.5f32 * (rs + rs * rp)
}

fn fresnel_conductor_reflectance_rgb(eta: &[f32; 3], k: &[f32; 3], cos_theta_i: f32) -> [f32; 3] {
    [
        fresnel_conductor_reflectance(eta[0], k[0], cos_theta_i),
        fresnel_conductor_reflectance(eta[1], k[1], cos_theta_i),
        fresnel_conductor_reflectance(eta[2], k[2], cos_theta_i),
    ]
}

pub fn sample_brdf_rough_conductor<RNG>(
    wi: &[f32; 3],
    reflectance: &[f32; 3],
    eta: &[f32; 3],
    k: &[f32; 3],
    roughness: f32,
    rng: &mut RNG,
) -> Option<([f32; 3], [f32; 3], f32)>
where
    RNG: rand::Rng,
{
    use del_geo_core::vec3::Vec3;
    if wi[2] < 0f32 {
        return None;
    }
    let sample_roughness: f32 = roughness;
    let alpha = microfacet_beckmann_roughness_to_alpha(roughness);
    let sample_alpha = microfacet_beckmann_roughness_to_alpha(sample_roughness);
    // sampling microfacet normal
    let m = microfacet_beckmann_sample(sample_alpha, &[rng.random::<f32>(), rng.random::<f32>()]);
    assert!(!m[0].is_nan() && !m[1].is_nan() && !m[2].is_nan());
    // microfacet normal PDF
    let m_pdf = microfacet_beckmann_pdf(sample_alpha, &m);
    assert!(!m_pdf.is_nan());
    //
    let wi_dot_m = wi.dot(&m);
    let wo = m.scale(2f32 * wi_dot_m).sub(wi);
    if wi_dot_m <= 0f32 || wo[2] <= 0f32 {
        return None;
    }
    // the masking shadow function
    let g = microfacet_distribution_g(alpha, wi, &wo, &m);
    assert!(!g.is_nan());
    let d = microfacet_beckmann_d(alpha, &m);
    assert!(!d.is_nan());
    let pdf = m_pdf * 0.25f32 / wi_dot_m; // compute `pdf of wo` from `pdf of m`
    let f = fresnel_conductor_reflectance_rgb(eta, k, wi_dot_m);
    let brdf = f
        .scale(g * d * 0.25f32 / (wi[2] * wo[2]))
        .element_wise_mult(reflectance); // 0.00281
    Some((wo, brdf, pdf))
}

pub fn eval_brdf_rough_conductor(
    wi: &[f32; 3],
    wo: &[f32; 3],
    reflectance: &[f32; 3],
    eta: &[f32; 3],
    k: &[f32; 3],
    roughness: f32,
) -> [f32; 3] {
    use del_geo_core::vec3::Vec3;
    if wi[2] <= 0f32 || wo[2] <= 0f32 {
        return [0f32; 3];
    }
    let alpha = microfacet_beckmann_roughness_to_alpha(roughness);
    let m = wi.add(wo).normalize();
    let wi_dot_m = wi.dot(&m);
    // the masking shadow function
    let g = microfacet_distribution_g(alpha, wi, wo, &m);
    let d = microfacet_beckmann_d(alpha, &m);
    let f = fresnel_conductor_reflectance_rgb(eta, k, wi_dot_m);
    f.scale(g * d * 0.25f32 / (wi[2] * wo[2]))
        .element_wise_mult(reflectance)
}

#[allow(unused_variables)]
pub fn sample_brdf_dielectric<RNG>(
    wo: &[f32; 3],
    reflectance: &[f32; 3],
    etas: &[f32; 3],
    uroughness: f32,
    vroughness: f32,
    rng: &mut RNG,
) -> Option<([f32; 3], [f32; 3], f32)>
where
    RNG: rand::Rng,
{
    let constant_etas = {
        let mut equals = true;
        for i in 1..3 {
            if etas[i] != etas[i - 1] {
                equals = false;
                break;
            }
        }
        equals
    };

    let (pdf_spectral, eta) = {
        let mut pdf = 1.;
        let mut eta = etas[0];
        if !constant_etas {
            pdf = 1. / 3.; // boost selected
            eta = etas[rng.random_range(0..3)];
        }
        (pdf, eta)
    };

    let smooth = |x: f32, y: f32| x.max(y) < 1e-3;
    // specular
    if eta == 1. || (smooth(uroughness, vroughness)) {
        let cos_theta_i = wo[2];
        let r = fresnel_dielectric_reflectance(eta, cos_theta_i);
        // transmission
        let t = 1. - r;
        // enhacement todo: choose sampling reflection or transmission
        let pr = r;
        let pt = t;

        let rdm = rng.random::<f32>();

        if rdm < (pr / (pr + pt)) {
            let wo = [-wo[0], -wo[1], wo[2]];
            let brdf = r / cos_theta_i.abs(); // perfect specular brdf
            let pdf = pr / (pr + pt);
            Some((wo, [brdf; 3], pdf * pdf_spectral))
        } else {
            if let Some((wt, eta)) = refract(wo, &[0., 0., 1.], eta) {
                let mut brdf = t / cos_theta_i.abs();
                // https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF#eq:transmitted-radiance-change
                brdf /= eta * eta;
                let pdf = pt / (pr + pt);
                return Some((wt, [brdf; 3], pdf * pdf_spectral));
            }
            None // can not be refracted
        }
    } else {
        // TODO: roughness sampling
        todo!();
    }
}

#[allow(unused_variables)]
pub fn eval_brdf_dielectric(
    wi: &[f32; 3],
    wo: &[f32; 3],
    reflectance: &[f32; 3],
    etas: &[f32; 3],
    uroughness: f32,
    vroughness: f32,
) -> [f32; 3] {
    let smooth = |x: f32, y: f32| x.max(y) < 1e-3;
    let debug = true;
    // specular
    if debug || (smooth(uroughness, vroughness)) {
        return [0.; 3];
    }

    todo!()
}

/// return refracted direction, eta(could be updated)
pub fn refract(wi: &[f32; 3], n: &[f32; 3], eta: f32) -> Option<([f32; 3], f32)> {
    use del_geo_core::vec3::Vec3;
    let (cos_theta_i, eta, n) = {
        let mut cos_theta = wi.dot(n);
        let mut eta = eta;
        let mut n = *n;
        // switch directions
        if cos_theta < 0. {
            eta = 1. / eta;
            cos_theta = -cos_theta;
            n.scale_in_place(-1.);
        }
        (cos_theta, eta, n)
    };

    let cos_theta_t = {
        let sqr_sin_theta_i = (1. - cos_theta_i * cos_theta_i).sqrt();
        let sqr_sin_theta_t = sqr_sin_theta_i / (eta * eta);
        if sqr_sin_theta_t >= 1. {
            return None;
        }
        (1. - sqr_sin_theta_t).sqrt()
    };

    let wt = wi
        .scale(-1. / eta)
        .add(&n.scale(cos_theta_i / eta - cos_theta_t));

    Some((wt, eta))
}

// <https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission.html#FrDielectric >
pub fn fresnel_dielectric_reflectance(mut eta: f32, mut cos_theta_i: f32) -> f32 {
    cos_theta_i = cos_theta_i.clamp(-1., 1.);
    // potential filp
    if cos_theta_i < 0. {
        eta = 1. / eta;
        cos_theta_i = -cos_theta_i;
    }

    let cos_theta_t = {
        let sqr_sin_theta_i = (1. - cos_theta_i * cos_theta_i).sqrt();
        let sqr_eta = eta * eta;
        (1. - sqr_sin_theta_i / sqr_eta).sqrt()
    };

    let r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    let r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);

    (r_parl * r_parl + r_perp * r_perp) / 2.
}

pub fn sample_brdf<RNG>(
    mat: &Material,
    obj_nrm: &[f32; 3],
    ray_in_outward_world: &[f32; 3],
    rng: &mut RNG,
    min_roughness: f32,
) -> Option<([f32; 3], [f32; 3], f32)>
where
    RNG: rand::Rng,
{
    use del_geo_core::mat3_col_major;
    use del_geo_core::vec3;
    debug_assert!((vec3::norm(obj_nrm) - 1f32).abs() < 1.0e-5);
    debug_assert!((vec3::norm(ray_in_outward_world) - 1f32).abs() < 1.0e-5);
    let transform_objlcl2world = mat3_col_major::transform_lcl2world_given_local_z(obj_nrm);
    let transform_world2objlcl = mat3_col_major::transpose(&transform_objlcl2world);
    let (ray_out_objlcl, brdf, pdf) = match mat {
        Material::Diff(a) => sample_brdf_diffuse(&a.reflectance, rng),
        Material::Cond(b) => {
            let ray_in_objlcl =
                mat3_col_major::mult_vec(&transform_world2objlcl, ray_in_outward_world);
            sample_brdf_rough_conductor(
                &ray_in_objlcl,
                &b.reflectance,
                &b.eta,
                &b.k,
                b.uroughness.max(min_roughness),
                rng,
            )?
        }
        Material::CoaDiff(_) => {
            eprintln!("Not implement sample CoaDiff");
            return None;
        }
        Material::None => return None,
    };
    debug_assert!((vec3::norm(&ray_out_objlcl) - 1f32).abs() < 1.0e-5);
    debug_assert!(ray_out_objlcl[2] >= 0f32);
    assert!(!brdf[0].is_nan() && !brdf[1].is_nan() && !brdf[2].is_nan());
    let ray_out_world = mat3_col_major::mult_vec(&transform_objlcl2world, &ray_out_objlcl);
    Some((ray_out_world, brdf, pdf))
}

pub fn eval_brdf(
    mat: &Material,
    obj_nrm: &[f32; 3],
    ray_in_outward_normalized: &[f32; 3],
    ray_out: &[f32; 3],
    minimum_roughness: f32,
) -> [f32; 3] {
    use del_geo_core::mat3_col_major;
    use del_geo_core::vec3;
    assert!(
        (vec3::norm(obj_nrm) - 1f32).abs() < 1.0e-5,
        "{}",
        vec3::norm(obj_nrm)
    );
    assert!(
        (vec3::norm(ray_in_outward_normalized) - 1f32).abs() < 1.0e-5,
        "{}",
        vec3::norm(ray_in_outward_normalized)
    );
    let transform_objlcl2world = mat3_col_major::transform_lcl2world_given_local_z(obj_nrm);
    let transform_world2objlcl = mat3_col_major::transpose(&transform_objlcl2world);
    match mat {
        Material::Diff(a) => eval_brdf_diffuse(&a.reflectance),
        Material::Cond(b) => {
            let ray_in_objlcl =
                mat3_col_major::mult_vec(&transform_world2objlcl, ray_in_outward_normalized);
            assert!((vec3::norm(&ray_in_objlcl) - 1.).abs() < 1.0e-3);
            let ray_out_objlcl = mat3_col_major::mult_vec(&transform_world2objlcl, ray_out);
            assert!((vec3::norm(&ray_out_objlcl) - 1.).abs() < 1.0e-3);
            eval_brdf_rough_conductor(
                &ray_in_objlcl,
                &ray_out_objlcl,
                &b.reflectance,
                &b.eta,
                &b.k,
                b.uroughness.max(minimum_roughness),
            )
            // eval_brdf_diffuse(&b.reflectance)
        }
        Material::CoaDiff(_) => {
            eprintln!("Not implement eval CoaDiff");
            [0f32; 3]
        }

        Material::None => [0f32; 3],
    }
}
