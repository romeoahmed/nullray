// Validation-only reference: bounded radial scan before direct arrival inversion.
fn scan_reference(source: vec3f, pixel: vec2i) -> vec4f {
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
    // An exact ray in the opaque zero-thickness surface needs an overlap decision.
    return unresolved(1u);
  }
  let k = (l - a * e) * (l - a * e) + c;
  let quadratic = a * a * e * e - l * l - c;
  let radial = prepare_quartic(vec4f(e * e, 0.0, quadratic, 2.0 * k), -a * a * c - q * q * k,
    1.0 / r, -source.x * sqrt(delta * sigma) / (r * r));
  let polar = prepare_quartic(vec4f(c, 0.0, quadratic, 0.0), -a * a * e * e, mu, sin_theta * p_theta);
  let equator = equator_events(a, e, l, c, theta, p_theta);
  if (equator.status == 2u) { return unresolved(2u); }
  let horizon_inverse = 1.0 / (1.0 + sqrt(1.0 - a * a - q * q));
  let step = 1.0 / (32.0 * sqrt(abs(c) + l * l + a * a * e * e + 1.0));
  var next_crossing = equator.first;
  var disk_order = 0u;
  for (var index = 1u; index <= 768u; index++) {
    let end = f32(index) * step;
    let ru = evaluate_quartic(radial, end);
    if (!ru.valid) { return unresolved(3u); }
    var boundary_time = end;
    let boundary = ru.value <= 0.0 || ru.value >= horizon_inverse;
    if (boundary) {
      let boundary_value = select(horizon_inverse, 0.0, ru.value <= 0.0);
      var lower = end - step;
      var upper = end;
      let initial = evaluate_quartic(radial, lower);
      if (!initial.valid) { return unresolved(4u); }
      for (var refine = 0u; refine < 16u; refine++) {
        let middle = (lower + upper) / 2.0;
        let sample = evaluate_quartic(radial, middle);
        if (!sample.valid) { return unresolved(5u); }
        if ((sample.value - boundary_value) * (initial.value - boundary_value) > 0.0) {
          lower = middle;
        } else { upper = middle; }
      }
      boundary_time = (lower + upper) / 2.0;
    }
    for (var event = 0u; event < 64u && equator.status == 1u && next_crossing <= boundary_time; event++) {
      let time = next_crossing;
      let order = disk_order;
      disk_order++;
      next_crossing += equator.spacing;
      let hit = evaluate_quartic(radial, time);
      if (!hit.valid) { return unresolved(6u); }
      if (hit.value >= 1.0 / frame.space.w && hit.value <= 1.0 / frame.space.z) {
        let radius = 1.0 / hit.value;
        let frequency = disk_frequency(a, q, e, l, radius);
        let mapped = integrate_transport(radial, polar, a, q, e, l, c, time, true, equator);
        if (frequency.x <= 0.0) { return unresolved(7u); }
        if (!mapped.valid) { return unresolved(8u); }
        textureStore(emission_time, pixel, vec4f(mapped.value.y, 0.0, 0.0, 0.0));
        return vec4f(radius, launch_phi + mapped.value.x, frequency.x, -3.0 - f32(order));
      }
    }
    if (ru.value <= 0.0) {
      if (e <= 0.0) { return unresolved(9u); }
      let mapped = integrate_transport(radial, polar, a, q, e, l, c, boundary_time, false, equator);
      let sky_mu = evaluate_quartic(polar, boundary_time);
      if (!mapped.valid) { return unresolved(10u); }
      if (!sky_mu.valid || abs(sky_mu.value) > 1.0) { return unresolved(11u); }
      var branch = 1.0;
      if (equator.status == 1u && boundary_time >= equator.first) {
        branch += 1.0 + floor((boundary_time - equator.first) / equator.spacing);
      }
      // Positive alpha identifies the equator-crossing branch. Other tags carry radiance.
      return vec4f(sky_mu.value, launch_phi + mapped.value.x, e, branch);
    }
    if (ru.value >= horizon_inverse) { return vec4f(0.0); }
    if (equator.status == 1u && next_crossing <= end) { return unresolved(12u); }
  }
  return unresolved(13u);
}
