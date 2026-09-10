@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
/**
 * Exposure gain, content peak relative to reference white, and scattered-light fraction.
 */
@group(0) @binding(2) var<uniform> display: vec4f;
@group(0) @binding(3) var bloom: texture_2d<f32>;

struct Vertex {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex fn vertex(@builtin(vertex_index) index: u32) -> Vertex {
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return Vertex(vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0), uv);
}

/**
 * Columns map linear sRGB through XYZ D65 into linear Display P3.
 * Signed working RGB must be transformed before destination-gamut clipping.
 */
const srgb_to_p3 = mat3x3f(
  vec3f(0.82246197, 0.03319420, 0.01708263),
  vec3f(0.17753803, 0.96680580, 0.07239744),
  vec3f(0.0, 0.0, 0.91051993)
);

/** Map signed linear sRGB to nonnegative P3 and roll highlights toward the content peak. */
fn display_radiance(rgb: vec3f, exposure: f32, content_peak: f32) -> vec3f {
  let radiance = max(srgb_to_p3 * rgb, vec3f(0.0)) * exposure;
  let peak = max(max(radiance.r, radiance.g), radiance.b);
  // One scale preserves RGB ratios while rolling highlights toward the content peak.
  return radiance * (content_peak / (content_peak + peak));
}

/**
 * Display P3 uses the sRGB transfer curve, extended above one for HDR.
 * Both select operands are safe because display_radiance is nonnegative.
 */
fn encode_display_p3(linear: vec3f) -> vec3f {
  return select(12.92 * linear, 1.055 * pow(linear, vec3f(1.0 / 2.4)) - 0.055,
    linear > vec3f(0.0031308));
}

@fragment fn fragment(input: Vertex) -> @location(0) vec4f {
  let sample = textureSample(scene, scene_sampler, input.uv);
  let scattered = textureSample(bloom, scene_sampler, input.uv).rgb;
  // Diagnostics enter only at presentation, after scattering resolved light.
  let rgb = mix(sample.rgb, scattered, display.z) + (1.0 - sample.a) * vec3f(0.65, 0.015, 0.24);
  return vec4f(encode_display_p3(display_radiance(rgb, display.x, display.y)), 1.0);
}
