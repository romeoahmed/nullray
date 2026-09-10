struct DifferentialSky {
  mu: vec3f,
  phi: vec3f,
  energy: f32,
  valid: bool,
}

fn differential_sky_failure() -> DifferentialSky {
  return DifferentialSky(vec3f(0.0), vec3f(0.0), 0.0, false);
}

/**
 * Positive-C polar motion, including the derivatives of its phase and modulus.
 */
struct DifferentialPolar {
  phase: vec3f,
  frequency: vec3f,
  m: vec3f,
  amplitude2: vec3f,
  complement: vec3f,
  signed_root: vec3f,
  coefficient: vec3f,
  enabled: bool,
  jacobi: DifferentialJacobiPlan,
}

fn d_polar_motion(status: ptr<function, bool>, a: f32, e: vec3f, l: vec3f, c: vec3f,
    theta: f32, p_theta: vec3f) -> DifferentialPolar {
  let ae2 = d_scale(status, d_mul(status, e, e), a * a);
  let b = d_sub(status, d_add(status, d_mul(status, l, l), c), ae2);
  let frequency2 = d_sqrt(status, d_add(status, d_mul(status, b, b), d_scale(status, d_mul(status, ae2, c), 4.0)));
  var z = d_constant(status, 1.0);
  if (l.x != 0.0) {
    if (b.x >= 0.0) { z = d_div(status, d_scale(status, c, 2.0), d_add(status, b, frequency2)); }
    else { z = d_div(status, d_sub(status, frequency2, b), d_scale(status, ae2, 2.0)); }
  }
  let frequency = d_sqrt(status, frequency2);
  let m = d_div(status, d_mul(status, ae2, z), frequency2);
  let cn = d_div(status, d_constant(status, cos(theta)), d_sqrt(status, z));
  let dn = d_sqrt(status, d_add(status, d_sub(status, d_constant(status, 1.0), m), d_mul(status, m, d_mul(status, cn, cn))));
  let sn = d_div(status, d_scale(status, p_theta, -sin(theta)), d_mul(status, d_sqrt(status, d_mul(status, z, frequency2)), dn));
  let phase = d_incomplete_f(status, d_atan2(status, sn, cn), m);
  let root = d_mul(status, l, d_sqrt(status, d_div(status, z, d_add(status, c, d_mul(status, ae2, z)))));
  let complement = d_mul(status, root, root);
  let coefficient = d_sqrt(status, d_add(status, d_constant(status, 1.0), d_div(status, d_mul(status, m, complement), z)));
  return DifferentialPolar(phase, frequency, m, z, complement, root, coefficient, true, d_prepare_jacobi(status, m));
}

/**
 * The signed root avoids differentiating abs(L) at zero angular momentum.
 * Discrete unwrapping affects values; away from an endpoint pole it has no derivative.
 */
fn d_polar_base(status: ptr<function, bool>, motion: DifferentialPolar, events: EquatorEvents, l: f32, end: vec3f) -> vec3f {
  let initial = d_evaluate_jacobi(status, motion.jacobi, motion.phase);
  let terminal = d_evaluate_jacobi(status, motion.jacobi, d_add(status, motion.phase, d_mul(status, motion.frequency, end)));
  let first = d_atan2(status, initial.sn, d_mul(status, motion.signed_root, initial.cn));
  let last = d_atan2(status, terminal.sn, d_mul(status, motion.signed_root, terminal.cn));
  let gradient = d_sub(status, last, first);
  // Sky azimuth is periodic. Keep its value and derivatives on the same
  // differentiated path; integer full turns do not affect a celestial direction.
  if (l != 0.0) { return d_scale(status, gradient, -1.0); }
  let sign_l = select(1.0, -1.0, l < 0.0);
  let unwrapped = sign_l * (polar_primitive(events, motion.complement.x,
    events.phase + events.frequency * end.x, l == 0.0) - events.azimuth_phase);
  return d_scale(status, vec3f(unwrapped, gradient.yz), -1.0);
}

/**
 * Differentiate the regular exterior azimuth rate, including both the moving
 * path and its moving endpoint. Positive-C motion removes the analytic polar
 * primitive before quadrature; the remaining direct branch retains its guards.
 */
fn d_sky_rate(status: ptr<function, bool>, radial: DifferentialQuartic, polar: DifferentialQuartic,
    a: f32, q: f32, e: vec3f, l: vec3f, time: vec3f, motion: DifferentialPolar) -> vec3f {
  let u = d_evaluate_quartic(status, radial, time);
  let u2 = d_mul(status, u, u);
  let delta = d_add(status, d_sub(status, d_constant(status, 1.0), d_scale(status, u, 2.0)), d_scale(status, u2, a * a + q * q));
  if (delta.x <= 0.0) { return d_fail(status); }
  var polar_rate = vec3f(0.0);
  if (motion.enabled) {
    let j = d_evaluate_jacobi(status, motion.jacobi, d_add(status, motion.phase, d_mul(status, motion.frequency, time)));
    // With w²=1+m(1-z)/z, split L/(1-z cn²) into the exact
    // unscaled polar primitive plus Lm/[zw(w+dn)]. The residual and
    // its L derivative are smooth through L=0, including pole phases.
    let denominator = d_mul(status, d_mul(status, motion.amplitude2, motion.coefficient),
      d_add(status, motion.coefficient, j.dn));
    polar_rate = d_div(status, d_mul(status, l, motion.m), denominator);
  } else if (any(l != vec3f(0.0))) {
    let mu = d_evaluate_quartic(status, polar, time);
    let sin2 = d_sub(status, d_constant(status, 1.0), d_mul(status, mu, mu));
    if (sin2.x <= 0.0) { return d_fail(status); }
    polar_rate = d_div(status, l, sin2);
  }
  let radial_rate = d_div(status, d_scale(status, d_sub(status, d_scale(status, d_mul(status, e, u), 2.0),
    d_mul(status, d_add(status, d_scale(status, e, q * q), d_scale(status, l, a)), u2)), a), delta);
  return d_scale(status, d_add(status, radial_rate, polar_rate), -1.0);
}

fn d_warped_sky_rate(status: ptr<function, bool>, radial: DifferentialQuartic, polar: DifferentialQuartic,
    a: f32, q: f32, e: vec3f, l: vec3f, end: vec3f, fraction: f32, motion: DifferentialPolar) -> vec3f {
  let angle = 1.5707963268 * fraction;
  let sine = sin(angle);
  let rate = d_sky_rate(status, radial, polar, a, q, e, l, d_scale(status, end, sine * sine), motion);
  return d_scale(status, d_mul(status, rate, end), 3.1415926536 * sine * cos(angle));
}

/**
 * The endpoint gradient is constant during quadrature. Integrate its linear
 * determinant contraction alongside the azimuth components to avoid subtracting
 * two large accumulated gradient products in the convergence estimate.
 */
fn d_sky_area_rate(status: ptr<function, bool>, mu: vec3f, rate: vec3f) -> f32 {
  if (!d_product_safe(mu.y, rate.z) || !d_product_safe(mu.z, rate.y)) { return d_fail(status).x; }
  return d_checked(status, vec3f(mu.y * rate.z - mu.z * rate.y, 0.0, 0.0)).x;
}

/** Prepared screen derivatives shared by the lanes integrating one sky ray. */
struct DifferentialSkyPath {
  radial: DifferentialQuartic,
  polar: DifferentialQuartic,
  energy: vec3f,
  momentum: vec3f,
  carter: vec3f,
  end: vec3f,
  mu: vec3f,
  launch: vec3f,
  motion: DifferentialPolar,
  valid: bool,
}
fn differential_sky_path_failure() -> DifferentialSkyPath {
  var path: DifferentialSkyPath;
  return path;
}

/**
 * One semi-analytic ray carrying the derivatives of its physical screen inputs.
 * The arrival derivative follows u(t_b)=0: dt_b/dp = (partial u/partial p)/E,
 * since the backward escaping radial velocity at infinity is -E.
 */
fn prepare_differential_sky(frame: Frame, pixel: vec2f, expected_branch: f32) -> DifferentialSkyPath {
  var derivative_valid = true;
  let status = &derivative_valid;
  let screen = (pixel + 0.5 + frame.viewport.zw - frame.viewport.xy / 2.0) / frame.viewport.y * frame.observer.z;
  let dx = vec3f(screen.x, frame.observer.z / frame.viewport.y, 0.0);
  let dy = vec3f(screen.y, 0.0, frame.observer.z / frame.viewport.y);
  var source: array<vec3f, 3>;
  for (var axis = 0u; axis < 3u; axis++) {
    source[axis] = d_add(status, d_sub(status, d_constant(status, frame.camera_forward[axis]), d_scale(status, dy, frame.camera_up[axis])), d_scale(status, dx, frame.camera_right[axis]));
  }
  let norm = d_sqrt(status, d_add(status, d_add(status, d_mul(status, source[0], source[0]), d_mul(status, source[1], source[1])), d_mul(status, source[2], source[2])));
  for (var axis = 0u; axis < 3u; axis++) { source[axis] = d_div(status, source[axis], norm); }
  let a = frame.space.x;
  let q = frame.space.y;
  let r = frame.observer.x;
  let theta = frame.observer.y;
  if (theta == 1.5707963268 && source[1].x == 0.0) { return differential_sky_path_failure(); }
  let axis = theta == 0.0 || theta == 3.1415926536;
  let sine = select(sin(theta), 0.0, axis);
  let cosine = cos(theta);
  var launch_phi = d_constant(status, frame.observer.w);
  if (axis && (source[1].x != 0.0 || source[2].x != 0.0)) {
    launch_phi = d_add(status, launch_phi, d_atan2(status, source[2], d_scale(status, source[1], sign(cosine))));
  }
  let sigma = r * r + a * a * cosine * cosine;
  let delta = r * r - 2.0 * r + a * a + q * q;
  let big_a = (r * r + a * a) * (r * r + a * a) - a * a * delta * sine * sine;
  let lapse = sqrt(sigma * delta / big_a);
  let dragging = a * (2.0 * r - q * q) / big_a;
  let l = d_scale(status, source[2], -sqrt(big_a / sigma) * sine);
  let e = d_add(status, d_constant(status, lapse), d_scale(status, l, dragging));
  var p_theta = d_scale(status, source[1], -sqrt(sigma));
  if (axis) { p_theta = d_scale(status, d_sqrt(status, d_add(status, d_mul(status, source[1], source[1]), d_mul(status, source[2], source[2]))), -sign(cosine) * sqrt(sigma)); }
  let e2 = d_mul(status, e, e);
  let c = d_add(status, d_sub(status, d_scale(status, d_mul(status, source[1], source[1]), sigma), d_scale(status, e2, a * a * cosine * cosine)),
    d_scale(status, d_mul(status, source[2], source[2]), big_a / sigma * cosine * cosine));
  let l_ae = d_sub(status, l, d_scale(status, e, a));
  let k = d_add(status, d_mul(status, l_ae, l_ae), c);
  let quadratic = d_sub(status, d_sub(status, d_scale(status, e2, a * a), d_mul(status, l, l)), c);
  let c4 = d_scale(status, d_add(status, d_scale(status, c, a * a), d_scale(status, k, q * q)), -1.0);
  let radial = d_prepare_quartic(status, array<vec3f, 5>(e2, d_constant(status, 0.0), quadratic, d_scale(status, k, 2.0), c4),
    d_constant(status, 1.0 / r), d_scale(status, source[0], -sqrt(delta * sigma) / (r * r)));
  if (!(*status) || e.x <= 0.0) { return differential_sky_path_failure(); }
  let primal = d_primal_quartic(radial);
  let horizon = 1.0 / (1.0 + sqrt(1.0 - a * a - q * q));
  let destination = radial_destination(vec3f(e2.x, quadratic.x, 2.0 * k.x), c4.x, primal, horizon);
  if (destination != 1u) { return differential_sky_path_failure(); }
  let arrival = quartic_arrival(primal, 0.0, -e.x);
  if (arrival.status != 1u) { return differential_sky_path_failure(); }
  let time = (arrival.lower + arrival.upper) / 2.0;
  if ((arrival.upper - arrival.lower) * sqrt(abs(c.x) + l.x * l.x + a * a * e.x * e.x) > 4e-5) { return differential_sky_path_failure(); }
  let events = equator_events(a, e.x, l.x, c.x, theta, p_theta.x);
  if (events.status == 2u) { return differential_sky_path_failure(); }
  var motion = DifferentialPolar(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0),
    vec3f(0.0), vec3f(0.0), vec3f(0.0), false, d_prepare_jacobi(status, d_constant(status, 0.0)));
  if (events.status == 1u && !axis) { motion = d_polar_motion(status, a, e, l, c, theta, p_theta); }
  var polar = DifferentialQuartic();
  if (!motion.enabled) {
    polar = d_prepare_quartic(status, array<vec3f, 5>(c, d_constant(status, 0.0), quadratic, d_constant(status, 0.0), d_scale(status, e2, -a * a)),
      d_constant(status, cosine), d_scale(status, p_theta, sine));
  }
  if (!(*status)) { return differential_sky_path_failure(); }
  if (l.x == 0.0 && !motion.enabled) {
    if (any(l.yz != vec2f(0.0)) || events.status != 1u) { return differential_sky_path_failure(); }
    launch_phi = d_add(status, launch_phi, d_constant(status, -(polar_primitive(events, 0.0,
      events.phase + events.frequency * time, true) - events.azimuth_phase)));
  }
  var next = events.first;
  var branch = 1.0;
  let primal_plan = prepare_jacobi_pair(primal.elliptic.m,primal.elliptic.complement);
  for (var index = 0u; index < 64u && events.status == 1u && next <= arrival.upper; index++) {
    if (next >= arrival.lower) { return differential_sky_path_failure(); }
    let hit = evaluate_quartic_prepared(primal, next, primal_plan);
    if (!hit.valid || (hit.value >= 1.0 / frame.space.w && hit.value <= 1.0 / frame.space.z)) { return differential_sky_path_failure(); }
    branch += 1.0;
    next += events.spacing;
  }
  if ((events.status == 1u && next <= arrival.upper) || branch != expected_branch) { return differential_sky_path_failure(); }
  let endpoint = d_evaluate_quartic(status, radial, d_constant(status, time));
  let shift = d_div(status, vec3f(0.0, endpoint.yz), d_constant(status, e.x));
  let end = d_add(status, d_constant(status, time), shift);
  var mu = vec3f(0.0);
  if (motion.enabled) {
    let j = d_evaluate_jacobi(status, motion.jacobi, d_add(status, motion.phase, d_mul(status, motion.frequency, end)));
    mu = d_mul(status, d_sqrt(status, motion.amplitude2), j.cn);
  } else {
    mu = d_evaluate_quartic(status, polar, end);
  }
  if (!(*status) || abs(mu.x) >= 1.0) { return differential_sky_path_failure(); }
  if (motion.enabled) { launch_phi = d_add(status, launch_phi, d_polar_base(status, motion, events, l.x, end)); }
  if (!(*status)) { return differential_sky_path_failure(); }
  return DifferentialSkyPath(radial, polar, e, l, c, end, mu, launch_phi, motion, true);
}
