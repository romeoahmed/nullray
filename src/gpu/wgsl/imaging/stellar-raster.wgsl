/** Confirmed point images; each contributes through a unit-integral detector tent. */
struct StellarImage {
  screen: vec4f,
  radiance: vec4f,
  parameters: vec4f,
  convergence: vec4f,
}
@group(0) @binding(21) var<storage, read> stellar_images: array<StellarImage>;
@group(0) @binding(22) var<storage, read> stellar_counts: array<u32>;
@group(0) @binding(23) var<storage, read_write> stellar_heads: array<atomic<u32>>;
/** Each image owns four nodes: next pointer, image index, tent weight, reserved. Zero ends a list. */
@group(0) @binding(24) var<storage, read_write> stellar_links: array<vec4u>;

@compute @workgroup_size(64)
fn bin_stellar_images(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= min(stellar_counts[0], arrayLength(&stellar_images))) { return; }
  let image = stellar_images[id.x];
  if (image.radiance.w == 0.0 || !stellar_search_owns(image.screen.xy + image.screen.zw)) { return; }
  // Subtract detector jitter from the fractional lane before choosing the support.
  let fraction = image.screen.zw - stellar_view.viewport.zw;
  let shift = floor(fraction);
  let origin = vec2i(image.screen.xy + shift);
  let remainder = fraction - shift;
  for (var y = 0u; y < 2u; y++) {
    for (var x = 0u; x < 2u; x++) {
      let pixel = origin + vec2i(i32(x), i32(y));
      if (any(pixel < vec2i(0)) || any(pixel >= vec2i(stellar_view.viewport.xy))) { continue; }
      let weight = (1.0 - abs(remainder.x - f32(x))) * (1.0 - abs(remainder.y - f32(y)));
      if (!(weight > 0.0)) { continue; }
      let node = 4u * id.x + 2u * y + x;
      let next = atomicExchange(&stellar_heads[u32(pixel.y) * u32(stellar_view.viewport.x) + u32(pixel.x)], node + 1u);
      stellar_links[node] = vec4u(next, id.x, bitcast<u32>(weight), 0u);
    }
  }
}

/** Ordered passes publish complete lists before spectral resolve reads them. */
fn stellar_detector_flux(pixel: vec2i) -> vec3f {
  var color = vec3f(0.0);
  var node = atomicLoad(&stellar_heads[u32(pixel.y) * u32(stellar_view.viewport.x) + u32(pixel.x)]);
  loop {
    if (node == 0u) { break; }
    let link = stellar_links[node - 1u];
    color += stellar_images[link.y].radiance.rgb * bitcast<f32>(link.z);
    node = link.x;
  }
  return color;
}
