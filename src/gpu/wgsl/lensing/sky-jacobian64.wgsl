/** A fixed orthonormal screen basis travels with the prepared derivative path. */
struct PreparedSkyBeam {
  path: DifferentialSkyPath,
  basis: vec4u,
  refined: bool,
}

/** High-precision derivative preparation; the transport evaluator remains f32. */
fn prepare_differential_sky64(frame: Frame, pixel: vec2f, expected_branch: f32, observer_frame: Observer64) -> PreparedSkyBeam {
  return prepare_differential_sky_precise(frame,array<Soft64,2>(soft64_number(pixel.x),soft64_number(pixel.y)),expected_branch,observer_frame);
}

/** Preserve continuous image positions through the differentiated launch. */
fn prepare_differential_sky_precise(frame: Frame, pixel: array<Soft64,2>, expected_branch: f32, observer_frame: Observer64) -> PreparedSkyBeam {
  var derivative_valid = true;
  let status = &derivative_valid;
  let lanes = array<vec4f,6>(frame.viewport,frame.observer,frame.space,frame.camera_forward,frame.camera_up,frame.camera_right);
  var bits: array<u32,24>;
  for (var lane = 0u; lane < 6u; lane++) {
    for (var component = 0u; component < 4u; component++) { bits[lane*4u+component] = bitcast<u32>(lanes[lane][component]); }
  }
  let sine = d64_constant(observer_frame.sine);
  let cosine = d64_constant(observer_frame.cosine);
  let photon = d64_screen_launch_precise(bits,pixel,observer_frame);
  let a64 = d64_from_f32_bits(bits[8]);
  let q64 = d64_from_f32_bits(bits[9]);
  let aa = d64_multiply(a64,a64);
  let qq = d64_multiply(q64,q64);
  let ae = d64_multiply(a64,photon.energy);
  let a2e2 = d64_multiply(ae,ae);
  let difference = d64_subtract(photon.angular_momentum,ae);
  let k64 = d64_add(d64_multiply(difference,difference),photon.carter);
  let quadratic64 = d64_subtract(d64_subtract(a2e2,d64_multiply(photon.angular_momentum,photon.angular_momentum)),photon.carter);
  let zero = d64_constant(vec2u(0u));
  let coefficients = array<Dual64,5>(d64_multiply(photon.energy,photon.energy),zero,quadratic64,
    d64_scale(k64,1),d64_negate(d64_add(d64_multiply(aa,photon.carter),d64_multiply(qq,k64))));
  let radial64 = d64_prepare_quartic(coefficients,
    d64_divide(d64_constant(soft64_number(1.0)),d64_from_f32_bits(bits[4])),photon.inverse_radial_velocity);

  let gradient = radial64.elliptic.parameter;
  let length = soft64_sqrt(soft64_add(soft64_multiply(gradient.dx,gradient.dx),soft64_multiply(gradient.dy,gradient.dy)));
  if (!soft64_valid(length)) { return PreparedSkyBeam(); }
  var basis = array<Soft64,2>(soft64_number(1.0),vec2u(0u));
  if (!soft64_zero(length)) {
    basis[0] = soft64_divide(gradient.dx,length);
    basis[1] = soft64_divide(gradient.dy,length);
  }
  let radial = d64_round_quartic(status,basis,radial64);
  let e = d64_round(status,basis,photon.energy);
  let l = d64_round(status,basis,photon.angular_momentum);
  let c = d64_round(status,basis,photon.carter);
  let p_theta = d64_round(status,basis,photon.polar_velocity);
  let a = frame.space.x;
  let q = frame.space.y;
  let theta = frame.observer.y;
  let axis = theta == 0.0 || theta == 3.1415926536;
  if (theta == 1.5707963268 && soft64_zero(photon.source[1].value)) { return PreparedSkyBeam(); }
  var launch_phi = d_constant(status,frame.observer.w);
  if (axis && (!soft64_zero(photon.source[1].value) || !soft64_zero(photon.source[2].value))) {
    launch_phi = d_add(status,launch_phi,d_atan2(status,d64_round(status,basis,photon.source[2]),
      d_scale(status,d64_round(status,basis,photon.source[1]),sign(cos(theta)))));
  }
  if (!(*status) || e.x <= 0.0) { return PreparedSkyBeam(); }
  let primal = d_primal_quartic(radial);
  let horizon64 = soft64_divide(soft64_number(1.0),soft64_add(soft64_number(1.0),soft64_sqrt(soft64_subtract(soft64_subtract(soft64_number(1.0),aa.value),qq.value))));
  let scalar_coefficients = array<Soft64,5>(coefficients[0].value,coefficients[1].value,coefficients[2].value,coefficients[3].value,coefficients[4].value);
  let destination = soft64_radial_destination(scalar_coefficients,radial64.values[0].value,radial64.values[1].value,horizon64);
  if (destination != 1u) { return PreparedSkyBeam(); }
  let arrival = quartic_arrival(primal, 0.0, -e.x);
  if (arrival.status != 1u) { return PreparedSkyBeam(); }
  let time = (arrival.lower + arrival.upper) / 2.0;
  if ((arrival.upper - arrival.lower) * sqrt(abs(c.x) + l.x * l.x + a * a * e.x * e.x) > 4e-5) { return PreparedSkyBeam(); }
  let events = equator_events(a, e.x, l.x, c.x, theta, p_theta.x);
  if (events.status == 2u) { return PreparedSkyBeam(); }
  var motion = DifferentialPolar(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0),
    vec3f(0.0), vec3f(0.0), vec3f(0.0), false, d_prepare_jacobi(status, d_constant(status, 0.0)));
  if (events.status == 1u && !axis) { motion = d_polar_motion(status, a, e, l, c, theta, p_theta); }
  // General polar transport is needed only when the analytic representation is disabled.
  var polar = DifferentialQuartic();
  if (!motion.enabled) {
    let polar64 = d64_prepare_quartic(array<Dual64,5>(photon.carter,zero,quadratic64,zero,d64_negate(a2e2)),
      cosine,d64_multiply(sine,photon.polar_velocity));
    polar = d64_round_quartic(status,basis,polar64);
  }
  if (!(*status)) { return PreparedSkyBeam(); }
  if (l.x == 0.0 && !motion.enabled) {
    if (any(l.yz != vec2f(0.0)) || events.status != 1u) { return PreparedSkyBeam(); }
    launch_phi = d_add(status, launch_phi, d_constant(status, -(polar_primitive(events, 0.0,
      events.phase + events.frequency * time, true) - events.azimuth_phase)));
  }
  var next = events.first;
  var branch = 1.0;
  let primal_plan = prepare_jacobi_pair(primal.elliptic.m,primal.elliptic.complement);
  for (var index = 0u; index < 64u && events.status == 1u && next <= arrival.upper; index++) {
    if (next >= arrival.lower) { return PreparedSkyBeam(); }
    let hit = evaluate_quartic_prepared(primal, next, primal_plan);
    if (!hit.valid || (hit.value >= 1.0 / frame.space.w && hit.value <= 1.0 / frame.space.z)) { return PreparedSkyBeam(); }
    branch += 1.0;
    next += events.spacing;
  }
  if ((events.status == 1u && next <= arrival.upper) || branch != expected_branch) { return PreparedSkyBeam(); }
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
  if (!(*status) || abs(mu.x) >= 1.0) { return PreparedSkyBeam(); }
  if (motion.enabled) { launch_phi = d_add(status, launch_phi, d_polar_base(status, motion, events, l.x, end)); }
  if (!(*status)) { return PreparedSkyBeam(); }
  return PreparedSkyBeam(DifferentialSkyPath(radial, polar, e, l, c, end, mu, launch_phi, motion, true),vec4u(basis[0],basis[1]),true);
}

/** Use the same critical-root work-selection policy as primary and boundary endpoints. */
fn prepare_sky_beam(frame: Frame, pixel: vec2f, expected_branch: f32, observer_frame: Observer64) -> PreparedSkyBeam {
  let screen = (pixel+0.5+frame.viewport.zw-frame.viewport.xy/2.0)/frame.viewport.y;
  let direction = normalize(frame.camera_forward.xyz-screen.y*frame.observer.z*frame.camera_up.xyz
    +screen.x*frame.observer.z*frame.camera_right.xyz);
  if (needs_radial_precision(frame,direction)) { return prepare_differential_sky64(frame,pixel,expected_branch,observer_frame); }
  return PreparedSkyBeam(prepare_differential_sky(frame,pixel,expected_branch),vec4u(soft64_number(1.0),vec2u(0u)),false);
}

/** Rotate evaluated derivatives back before returning them to physical screen consumers. */
fn d64_restore_gradient(basis: vec4u, along: f32, across: f32) -> vec2f {
  let first = soft64_number(along);
  let second = soft64_number(across);
  let dx = soft64_subtract(soft64_multiply(basis.xy,first),soft64_multiply(basis.zw,second));
  let dy = soft64_add(soft64_multiply(basis.zw,first),soft64_multiply(basis.xy,second));
  return bitcast<vec2f>(vec2u(soft64_to_f32_bits(dx),soft64_to_f32_bits(dy)));
}
