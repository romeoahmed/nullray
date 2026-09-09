@group(0) @binding(2) var<storage, read> blackbody_table: array<vec4f>;

/**
 * CIE-integrated linear RGB, log T in [100, 1e6] K, Y(6500 K) = 1.
 * Alpha distinguishes a resolved visible tail from an unsupported temperature.
 */
fn blackbody_radiance(temperature: f32) -> vec4f {
  if (!(temperature >= 0.0 && temperature <= 1000000.0)) { return vec4f(0.0); }
  // Below 100 K the visible tail is negligible at the declared normalization.
  if (temperature < 100.0) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let size = arrayLength(&blackbody_table);
  let coordinate = log(temperature / 100.0) / log(10000.0) * f32(size - 1u);
  let lower = min(u32(coordinate), size - 2u);
  return vec4f(mix(blackbody_table[lower].rgb, blackbody_table[lower + 1u].rgb, coordinate - f32(lower)), 1.0);
}

/**
 * Deterministic phases for a prescribed emissivity field, not physical turbulence samples.
 */
fn disk_phase(cell: vec4u, mode: vec4u, epoch: f32) -> vec4f {
  var cohort = bitcast<u32>(epoch);
  if (epoch == 0.0) { cohort = 0u; }
  var seed = cell * 747796405u + mode * 2891336453u + cohort;
  seed ^= seed << vec4u(13u);
  seed ^= seed >> vec4u(17u);
  seed ^= seed << vec4u(5u);
  seed *= 277803737u;
  return vec4f(seed >> vec4u(8u)) * (6.283185307179586 / 16777216.0);
}

/** Zero-mean bounded appearance modes at radius r, measured in M, and azimuthal phase. */
fn disk_structure(r: f32, inner: f32, phase: f32, epoch: f32) -> f32 {
  // Four independent physical appearance modes use native vector arithmetic.
  let modes = vec4u(3u, 7u, 13u, 23u);
  let coordinate = log(r / inner) * vec4f(3.0, 6.0, 12.0, 24.0);
  let cell = vec4i(floor(coordinate));
  let blend = smoothstep(vec4f(0.0), vec4f(1.0), fract(coordinate));
  let angle = vec4f(modes) * phase;
  // Convex interpolation bounds the mode and keeps its azimuthal mean exactly zero.
  let waves = mix(sin(angle + disk_phase(bitcast<vec4u>(cell), modes, epoch)),
    sin(angle + disk_phase(bitcast<vec4u>(cell + 1), modes, epoch)), blend);
  return dot(vec4f(0.45, 0.3, 0.15, 0.1), waves);
}

/** Nonnegative flux multiplier; time is the relative retarded epoch in M, base_epoch a cohort index. */
fn disk_modulation_epoch(r: f32, inner: f32, phi: f32, time: f32, base_epoch: f32, omega: f32, contrast: f32) -> f32 {
  // Smoothly replace advected structures after a finite coordinate-time coherence interval.
  // Both cohorts use the retarded emission epoch; their ages stay within ±24 M.
  let offset = floor(time / 24.0);
  let epoch = base_epoch + offset;
  let age = time - 24.0 * offset;
  let weight = smoothstep(0.0, 24.0, age);
  let first = disk_structure(r, inner, phi - omega * age, epoch);
  let second = disk_structure(r, inner, phi - omega * (age - 24.0), epoch + 1.0);
  return 1.0 + contrast * mix(first, second, weight);
}
