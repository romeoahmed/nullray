/**
 * Neutral Kerr–Newman circular emitter, orientation s=+1:
 * (observed/emitted frequency, coordinate angular velocity).
 * With u=1/r and v=u sqrt(u(1-q²u)), Ω=v/(1+av). Cancelling
 * the common (1+av) in u^t(E-ΩL) avoids rebuilding three metric components.
 */
fn disk_frequency(a: f32, q: f32, e: f32, l: f32, r: f32) -> vec2f {
  if (r <= q * q) { return vec2f(-1.0, 0.0); }
  let u = 1.0 / r;
  let charge = q * q * u;
  if (charge >= 1.0) { return vec2f(-1.0, 0.0); }
  let v = u * sqrt(u * (1.0 - charge));
  let denominator = 1.0 + a * v;
  let norm = 1.0 - (3.0 - 2.0 * charge) * u + 2.0 * a * v;
  let energy = e + (a * e - l) * v;
  if (denominator <= 0.0 || norm <= 0.0 || energy <= 0.0) { return vec2f(-1.0, 0.0); }
  return vec2f(sqrt(norm) / energy, v / denominator);
}

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
