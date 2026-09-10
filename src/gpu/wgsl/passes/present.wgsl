@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
/**
 * Tone: exposure, content peak, scattered-light fraction and missing-sample diagnostic switch.
 * White: P3 camera gains and the photographic shoulder switch.
 */
struct Display {
  tone: vec4f,
  white: vec4f,
}
@group(0) @binding(2) var<uniform> display: Display;
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

/**
 * Achromatic highlight compression, adapted from the Khronos PBR Neutral shoulder.
 * Copyright 2024 The Khronos Group, Inc.; Apache-2.0 (see NOTICE.md and licenses/).
 * Modified for emissive P3/HDR: no Fresnel offset, later desaturation, explicit content peak.
 */
fn photographic_shoulder(radiance: vec3f, content_peak: f32) -> vec3f {
  let relative = radiance / content_peak;
  let peak = max(max(relative.r, relative.g), relative.b);
  if (peak <= 0.7) { return radiance; }
  // This branch has a positive denominator and joins the linear range with unit slope.
  let compressed = 1 - 0.09 / (peak - 0.4);
  let whitening = 0.08 * (peak - compressed);
  return content_peak * compressed * (relative / peak + whitening) / (1 + whitening);
}

/** Map signed linear sRGB into P3 before clipping and camera calibration. */
fn display_radiance(rgb: vec3f, exposure: f32, content_peak: f32) -> vec3f {
  let radiance = max(srgb_to_p3 * rgb, vec3f(0.0)) * display.white.rgb * exposure;
  if (display.white.w > 0) { return photographic_shoulder(radiance, content_peak); }
  // Scientific false-color and monochrome views retain their simple channel-wise scale.
  return radiance * (content_peak / (vec3f(content_peak) + radiance));
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
  // Missing radiance keeps its zero contribution; the separate coverage report retains its weight.
  let rgb = mix(sample.rgb, scattered, display.tone.z) + display.tone.w * (1.0 - sample.a) * vec3f(0.65, 0.015, 0.24);
  return vec4f(encode_display_p3(display_radiance(rgb, display.tone.x, display.tone.y)), 1.0);
}
