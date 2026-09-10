/** Six vec4 lanes shared with the 96-byte host uniform; lengths use M = 1. */
struct Frame {
  viewport: vec4f,
  observer: vec4f,
  space: vec4f,
  camera_forward: vec4f,
  camera_up: vec4f,
  camera_right: vec4f,
}

/** Endpoint tag/payload and disk emission time relative to the observer epoch. */
struct RayEndpoint {
  value: vec4f,
  relative_time: f32,
}

/**
 * Preserve the failing stage instead of mapping exhausted or ambiguous work to capture.
 */
fn unresolved(stage: u32) -> vec4f {
  return vec4f(f32(stage), 0.0, 0.0, -2.0);
}

/**
 * Resolve ordered source events after a local ZAMO launch. Keep the launch
 * expressions in this pure function: returning their intermediates through a
 * separate struct changes f32 rounding on the retained critical launch fixture.
 */
fn trace_ray(frame: Frame, source: vec3f) -> RayEndpoint {
  let a = frame.space.x;
  let q = frame.space.y;
  let r = frame.observer.x;
  let theta = frame.observer.y;
  let axis = theta == 0.0 || theta == 3.1415926536;
  let sin_theta = select(sin(theta), 0.0, axis);
  let mu = cos(theta);
  var launch_phi = frame.observer.w;
  if (axis && length(source.yz) > 0.0) {
    launch_phi += atan2(source.z, sign(mu) * source.y);
  }
  let sigma = r * r + a * a * mu * mu;
  let delta = r * r - 2.0 * r + a * a + q * q;
  let big_a = (r * r + a * a) * (r * r + a * a) - a * a * delta * sin_theta * sin_theta;
  let lapse = sqrt(sigma * delta / big_a);
  let dragging = a * (2.0 * r - q * q) / big_a;
  let l = -source.z * sqrt(big_a / sigma) * sin_theta;
  let e = lapse + dragging * l;
  let p_theta = select(-source.y, -sign(mu) * length(source.yz), axis) * sqrt(sigma);
  // Cancel sin(theta) analytically in L² cot²(theta), including either pole.
  let c = source.y * source.y * sigma - a * a * e * e * mu * mu
    + source.z * source.z * big_a / sigma * mu * mu;
  if (theta == 1.5707963268 && source.y == 0.0) {
    return RayEndpoint(unresolved(1u), 0.0);
  }
  let k = (l - a * e) * (l - a * e) + c;
  let quadratic = a * a * e * e - l * l - c;
  let radial = prepare_quartic(vec4f(e * e, 0.0, quadratic, 2.0 * k), -a * a * c - q * q * k,
    1.0 / frame.observer.x, -source.x * sqrt(delta * sigma) / (r * r));
  let equator = equator_events(a, e, l, c, theta, p_theta);
  if (equator.status == 2u) { return RayEndpoint(unresolved(2u), 0.0); }
  // Positive-C motion already has a complete Jacobi cosine representation.
  // Reuse it for endpoints and transport instead of preparing a second quartic.
  var polar = QuarticPath();
  polar.x = mu;
  if (equator.status != 1u) {
    polar = prepare_quartic(vec4f(c, 0.0, quadratic, 0.0), -a * a * e * e, mu, sin_theta * p_theta);
  }
  let horizon_inverse = 1.0 / (1.0 + sqrt(1.0 - a * a - q * q));
  let destination = radial_destination(vec3f(e * e, quadratic, 2.0 * k), -a * a * c - q * q * k, radial, horizon_inverse);
  if (destination == 0u) { return RayEndpoint(unresolved(14u), 0.0); }
  if (destination == 1u && e <= 0.0) { return RayEndpoint(unresolved(9u), 0.0); }
  let boundary_value = select(horizon_inverse, 0.0, destination == 1u);
  let boundary_velocity = select(abs(e + (a * a * e - a * l) * horizon_inverse * horizon_inverse), -abs(e), destination == 1u);
  let arrival = quartic_arrival(radial, boundary_value, boundary_velocity);
  if (arrival.status != 1u) { return RayEndpoint(vec4f(15.0, arrival.lower, arrival.upper, -2.0), 0.0); }
  let boundary_time = (arrival.lower + arrival.upper) / 2.0;
  // All candidate disk crossings share this radial modulus and its AGM descent.
  let radial_plan = prepare_jacobi_pair(radial.elliptic.m,radial.elliptic.complement);
  var next_crossing = equator.first;
  var disk_order = 0u;
  for (var event = 0u; event < 64u && equator.status == 1u && next_crossing <= arrival.upper; event++) {
    let time = next_crossing;
    if (time >= arrival.lower) { return RayEndpoint(unresolved(16u), 0.0); }
    let order = disk_order;
    disk_order++;
    next_crossing += equator.spacing;
    let hit = evaluate_quartic_prepared(radial, time, radial_plan);
    if (!hit.valid) { return RayEndpoint(unresolved(6u), 0.0); }
    if (hit.value >= 1.0 / frame.space.w && hit.value <= 1.0 / frame.space.z) {
      let radius = 1.0 / hit.value;
      let frequency = disk_frequency(a, q, e, l, radius);
      let mapped = integrate_transport(radial, polar, a, q, e, l, c, time, true, equator);
      if (frequency.x <= 0.0) { return RayEndpoint(unresolved(7u), 0.0); }
      if (!mapped.valid) { return RayEndpoint(unresolved(8u), 0.0); }
      return RayEndpoint(vec4f(radius, launch_phi + mapped.value.x, frequency.x, -3.0 - f32(order)), mapped.value.y);
    }
  }
  if (equator.status == 1u && next_crossing <= arrival.upper) { return RayEndpoint(unresolved(12u), 0.0); }
  if (destination == 1u) {
    if ((arrival.upper - arrival.lower) * sqrt(abs(c) + l * l + a * a * e * e) > 4e-5) {
      return RayEndpoint(vec4f(15.0, arrival.lower, arrival.upper, -2.0), 0.0);
    }
    let mapped = integrate_transport(radial, polar, a, q, e, l, c, boundary_time, false, equator);
    var sky_mu = Evaluation(0.0, false);
    if (equator.status == 1u) {
      let j = jacobi(equator.phase + equator.frequency * boundary_time, equator.m);
      sky_mu = Evaluation(sqrt(equator.amplitude2) * j.y, true);
    } else { sky_mu = evaluate_quartic(polar, boundary_time); }
    if (!mapped.valid) { return RayEndpoint(unresolved(10u), 0.0); }
    if (!sky_mu.valid || abs(sky_mu.value) > 1.0) { return RayEndpoint(unresolved(11u), 0.0); }
    var branch = 1.0;
    if (equator.status == 1u && boundary_time >= equator.first) {
      branch += 1.0 + floor((boundary_time - equator.first) / equator.spacing);
    }
    // Positive alpha identifies the sky equator-crossing branch.
    return RayEndpoint(vec4f(sky_mu.value, launch_phi + mapped.value.x, e, branch), 0.0);
  }
  return RayEndpoint(vec4f(0.0), 0.0);
}

/**
 * Pixel offsets are relative to the current jittered sample, in physical image pixels.
 */
fn screen_ray(frame: Frame, pixel: vec2f) -> RayEndpoint {
  let screen = (pixel + 0.5 + frame.viewport.zw - frame.viewport.xy / 2.0) / frame.viewport.y;
  let direction = normalize(frame.camera_forward.xyz - screen.y * frame.observer.z * frame.camera_up.xyz
    + screen.x * frame.observer.z * frame.camera_right.xyz);
  return trace_ray(frame, direction);
}
