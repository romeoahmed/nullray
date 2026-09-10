/** Nominal material-coordinate change across a transfer cell, before the vertical warp. */
fn cloud_span(inner: f32, r: f32, radius: f32, age: f32, omega: f32, omega_derivative: f32, span: vec4f) -> vec3f {
  let radial = 7 * span.x / r;
  let angular = span.y - (3 * sqrt(inner / r) / r + omega_derivative * age) * span.x - omega * span.w;
  return vec3f(abs(radial), radius * abs(angular), 0.5 * abs(span.z));
}

/** Local span including a worst-case derivative of the added vertical lattice warp. */
fn cloud_footprint(extent: vec3f) -> f32 {
  // Quintic interpolation has maximum slope 1.875 and lattice values span two units.
  // The sqrt(2) factor bounds planar L1 variation after rotation from the radial basis.
  let warp_span = 2.25 * (0.2545584412 * (extent.x + extent.y) + 0.25 * extent.z);
  return length(vec3f(extent.xy, extent.z + warp_span));
}

/**
 * Density, large-cloud corrugation and thermal structure, periodic in azimuth and time cohorts.
 * Span contains signed changes in r, longitude, H-normalized height and emission time across a cell.
 * Frequency attenuation filters the nominal line footprint; it is not a lensing-beam derivative.
 */
fn disk_cloud(space: vec2f, inner: f32, r: f32, phi: f32, height: f32, time: f32, omega: f32, span: vec4f) -> vec3f {
  let root = sqrt(inner - space.y * space.y);
  let lifetime = 12.566370614359172 * (inner * inner + space.x * root) / root;
  let age = time / lifetime;
  let epoch = floor(age);
  let blend = smoothstep(0.0, 1.0, fract(age));
  let logarithm = log(r / inner);
  let cohort_age = lifetime * fract(age);
  let phase = phi + 6 * sqrt(inner / r) - omega * cohort_age;
  let next_phase = phase + omega * lifetime;
  // Radius, azimuth and height span independent coordinates; vertical structure is not a radial shift.
  let annulus = 4 + 7 * logarithm;
  var position = vec3f(annulus * cos(phase), annulus * sin(phase), 0.5 * height + 17 * epoch);
  var next = vec3f(annulus * cos(next_phase), annulus * sin(next_phase), 0.5 * height + 17 * (epoch + 1));
  let radial_root = sqrt(r - space.y * space.y);
  let denominator = r * r + space.x * radial_root;
  let omega_derivative = (r * r / (2 * radial_root) - 2 * r * radial_root) / (denominator * denominator);
  let extent = cloud_span(inner, r, annulus, cohort_age, omega, omega_derivative, span);
  let next_extent = cloud_span(inner, r, annulus, cohort_age - lifetime, omega, omega_derivative, span);
  let large_scale = vec3f(0.18, 0.18, 0.25);
  let large_weight = 1 - smoothstep(0.5, 2.0, length(extent * large_scale));
  let next_large_weight = 1 - smoothstep(0.5, 2.0, length(next_extent * large_scale));
  let warp = large_weight * field_noise(position * large_scale);
  let next_warp = next_large_weight * field_noise(next * large_scale);
  let large_cloud = mix(warp, next_warp, blend);
  position.z += 0.6 * warp;
  next.z += 0.6 * next_warp;
  // Filter each cohort before blending so a retired cohort cannot change the surviving footprint.
  var footprint = cloud_footprint(extent);
  var next_footprint = cloud_footprint(next_extent);
  var sum = 0.0;
  var weight = 0.5;
  for (var octave = 0u; octave < 6u; octave++) {
    if (footprint >= 2 && next_footprint >= 2) { break; }
    var value = 0.0;
    var next_value = 0.0;
    if (footprint < 2) { value = (1 - smoothstep(0.5, 2.0, footprint)) * field_noise(position); }
    if (next_footprint < 2) { next_value = (1 - smoothstep(0.5, 2.0, next_footprint)) * field_noise(next); }
    sum += weight * mix(value, next_value, blend);
    position *= 2;
    next *= 2;
    footprint *= 2;
    next_footprint *= 2;
    weight *= 0.65;
  }
  return vec3f(0.4 * large_cloud + 0.6 * sum, large_cloud, 0.2 * large_cloud + 0.8 * sum);
}
