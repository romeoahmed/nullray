@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var output_image: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var emission_time: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> observer_frame: Observer64;

/** Store one primary endpoint per texel; emission time is meaningful only for disk hits. */
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= textureDimensions(output_image))) { return; }
  let endpoint = screen_ray_refined(frame, vec2f(id.xy), observer_frame);
  textureStore(output_image, vec2i(id.xy), endpoint.value);
  if (endpoint.value.w <= -3.0) {
    textureStore(emission_time, vec2i(id.xy), vec4f(endpoint.relative_time, 0.0, 0.0, 0.0));
  }
}
