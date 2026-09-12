requires readonly_and_readwrite_storage_textures;

@group(0) @binding(0) var sample_image: texture_2d<f32>;
@group(0) @binding(1) var history: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(2) var output_image: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> sample_index: vec4u;

/** Update the mean for sequential samples 0–63; sample zero starts a new history. */
@compute @workgroup_size(8, 8)
fn accumulate(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= textureDimensions(sample_image))) { return; }
  let pixel = vec2i(id.xy);
  let sample = textureLoad(sample_image, pixel, 0);
  // Final radiation coverage includes geometry and source failures. A primary
  // endpoint alone cannot describe the resolved weight of a refined pixel.
  let unresolved = sample.a == 0.0;
  var value = vec4f(sample.rgb, 1.0 - sample.a);
  if (unresolved) { value = vec4f(0.0, 0.0, 0.0, 1.0); }
  if (sample_index.x > 0u) {
    let previous = textureLoad(history, pixel);
    value = previous + (value - previous) / f32(sample_index.x + 1u);
  }
  // Each invocation owns one texel. Ordered passes provide inter-frame visibility;
  // no invocation reads another invocation's writes within this pass.
  // History alpha stores missing weight; the resolved output below stores its complement.
  textureStore(history, pixel, value);
  // Match exploration: resolved light and coverage, without diagnostic color in radiance.
  textureStore(output_image, pixel, vec4f(value.rgb, 1.0 - value.a));
}
