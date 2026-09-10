@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var detail: texture_2d<f32>;
@group(0) @binding(3) var<uniform> blend: vec4f;

struct Vertex {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex fn vertex(@builtin(vertex_index) index: u32) -> Vertex {
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return Vertex(vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0), uv);
}

/**
 * Convolve the piecewise-linear source with [1, 2, 1]/4 on each axis.
 * At fractional texel phase f the four coefficients are
 * [1-f, 2-f, 1+f, f]/4. Pair adjacent coefficients into one bilinear
 * lookup each: four texture operations instead of nine, including odd sizes.
 */
fn filtered(uv: vec2f) -> vec3f {
  let size = vec2f(textureDimensions(source));
  let position = uv * size - 0.5;
  let base = floor(position);
  let phase = fract(position);
  let left = 3.0 - 2.0 * phase;
  let right = 1.0 + 2.0 * phase;
  let lo = (base - 0.5 + (2.0 - phase) / left) / size;
  let hi = (base + 1.5 + phase / right) / size;
  let a = textureSampleLevel(source, linear_sampler, lo, 0.0).rgb;
  let b = textureSampleLevel(source, linear_sampler, vec2f(hi.x, lo.y), 0.0).rgb;
  let c = textureSampleLevel(source, linear_sampler, vec2f(lo.x, hi.y), 0.0).rgb;
  let d = textureSampleLevel(source, linear_sampler, hi, 0.0).rgb;
  let weight = right * 0.25;
  return mix(mix(a, b, weight.x), mix(c, d, weight.x), weight.y);
}

@fragment fn downsample(input: Vertex) -> @location(0) vec4f {
  return vec4f(filtered(input.uv), 1.0);
}

@fragment fn upsample(input: Vertex) -> @location(0) vec4f {
  let fine = textureSampleLevel(detail, linear_sampler, input.uv, 0.0).rgb;
  return vec4f(mix(filtered(input.uv), fine, blend.x), 1.0);
}
