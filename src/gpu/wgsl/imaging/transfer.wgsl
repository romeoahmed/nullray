/** Detector-normalized plasma cutoff; the separable radial profile is regular away from Σ=0. */
fn plasma_cutoff(g: KerrGeometry, time: f32) -> f32 {
  let r2 = g.radius * g.radius;
  if (optical_frame.heating.z == 0 || g.radius <= optical_frame.heating.x || g.radius >= optical_frame.heating.y) {
    return optical_frame.plasma.x * (r2 / (r2 + optical_frame.plasma.y)) / g.sigma;
  }
  let geometry = medium_geometry(optical_frame.space.xy, g.radius, g.direction, g.chart);
  return medium_potential(optical_frame.space.xy, geometry, optical_frame.plasma.xy, optical_frame.heating, time).cutoff;
}

/**
 * Evaluate a shifted thermal source in color or narrowband brightness units.
 * Plasma mode returns Planck-corrected Rayleigh–Jeans temperature / 6500 K;
 * the detector refractive factor is applied later. Input temperature already includes redshift.
 */
fn source_spectrum(temperature: f32) -> vec4f {
  if (optical_frame.plasma.w == 0) { return blackbody_radiance(temperature); }
  if (!(temperature > 0)) { return vec4f(0, 0, 0, 1); }
  let quantum_temperature = optical_frame.plasma.z;
  let x = quantum_temperature / temperature;
  if (x > 80) { return vec4f(0, 0, 0, 1); }
  var correction = 1 - x / 2 + x * x / 12;
  if (x > 0.01) { correction = x / (exp(x) - 1); }
  return vec4f(vec3f(temperature * correction / 6500), 1);
}

/**
 * Append an accepted state while inspection storage has capacity.
 * A full buffer stops recording only; tracing retains its independent work budget.
 */
fn record_ray(path: KerrOrbit, block: vec3i) {
  let count = inspected_ray.summary.x;
  if (u32(count) >= arrayLength(&inspected_ray.points)) { return; }
  inspected_ray.points[count] = RayPathPoint(
    vec4f(path.state.radial.x, normalize(path.state.direction).z, path.state.radial.z, f32(path.inverse)),
    vec4i(block, i32(path.chart) * select(1, 2, path.bifurcation.z != 0)));
  inspected_ray.summary.x = count + 1;
}

/** Milne intensity and polarized intensity, normalized to preserve hemispheric thermal flux. */
fn disk_atmosphere(mu: f32) -> vec2f {
  let count = arrayLength(&atmosphere_table);
  let coordinate = clamp(mu, 0.0, 1.0) * f32(count - 1u);
  let lower = min(u32(coordinate), count - 2u);
  return mix(atmosphere_table[lower].xy, atmosphere_table[lower + 1u].xy, coordinate - f32(lower));
}

/** Sample the prepared disk temperature profile in logarithmic radius; the table has at least two entries. */
fn disk_temperature(r: f32) -> f32 {
  let count = arrayLength(&disk_profile);
  let coordinate = clamp(log(r / optical_frame.space.z) / log(optical_frame.space.w / optical_frame.space.z), 0.0, 1.0) * f32(count - 1u);
  let lower = min(u32(coordinate), count - 2u);
  return mix(disk_profile[lower], disk_profile[lower + 1u], coordinate - f32(lower));
}
