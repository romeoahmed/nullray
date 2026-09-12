@group(0) @binding(2) var<storage, read> blackbody_table: array<vec4f>;

/**
 * Interpolate CIE-integrated linear sRGB on a log-spaced 100–1,000,000 K table.
 * Y(6500 K) = 1. Nonnegative temperatures below 100 K resolve as negligible
 * visible light; unsupported temperatures return zero alpha, not physical darkness.
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
