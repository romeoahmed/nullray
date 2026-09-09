/**
 * Inspection colors are derived from endpoint data; failed geometry keeps zero
 * coverage so photography cannot turn an unresolved ray into a successful sample.
 */
fn diagnostic_endpoint(sample: vec4f, mode: f32) -> vec4f {
  if (sample.w == -2.0) { return vec4f(0.0); }
  var color = vec3f(0.0);
  if (mode == 1.0 && (sample.w > 0.0 || sample.w <= -3.0)) {
    if (sample.z <= 0.0) { return vec4f(0.0); }
    // log2(1/E) = -log2(E), without an overflowing reciprocal.
    let log_shift = select(log2(sample.z), -log2(sample.z), sample.w > 0.0);
    let shift = clamp(log_shift, -1.0, 1.0);
    let hue = select(vec3f(0.8, 0.04, 0.015), vec3f(0.025, 0.25, 0.8), shift >= 0.0);
    color = mix(vec3f(0.3), hue, abs(shift));
  } else if (mode == 2.0) {
    if (sample.w == -3.0) { color = vec3f(0.1); }
    else if (sample.w == -4.0) { color = vec3f(0.02, 0.35, 0.6); }
    else if (sample.w <= -5.0) { color = vec3f(0.8, 0.3, 0.02); }
  }
  return vec4f(color, 1.0);
}
