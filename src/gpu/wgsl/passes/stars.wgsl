@group(0) @binding(13) var composite_image: texture_storage_2d<rgba16float, write>;
@group(0) @binding(14) var foreground: texture_2d<f32>;
@group(0) @binding(15) var arrivals: texture_2d<f32>;
@group(0) @binding(16) var transmissions: texture_2d<f32>;
@group(0) @binding(17) var domains: texture_2d<i32>;

/** A local source chart excludes domain jumps, image-order jumps and large spherical chords. */
fn arrival_neighbor(pixel: vec2i, center: vec4f, domain: vec4i, order: f32) -> vec4f {
  let size = vec2i(textureDimensions(arrivals));
  if (any(pixel < vec2i(0)) || any(pixel >= size)) { return vec4f(0); }
  let endpoint = textureLoad(arrivals, pixel, 0);
  if (endpoint.w <= 0 || any(textureLoad(domains, pixel, 0) != domain)
    || textureLoad(transmissions, pixel, 0).y != order) { return vec4f(0); }
  let difference = endpoint.xyz - center.xyz;
  // A 1/8-radian chord bounds the neglected spherical sagitta below 0.8%.
  if (dot(difference, difference) > 0.015625) { return vec4f(0); }
  return vec4f(difference, 1);
}

fn arrival_difference(pixel: vec2i, offset: vec2i, center: vec4f, domain: vec4i, order: f32) -> vec3f {
  let next = arrival_neighbor(pixel + offset, center, domain, order);
  let previous = arrival_neighbor(pixel - offset, center, domain, order);
  let left = -previous.xyz;
  let right = next.xyz;
  if (next.w > 0 && previous.w > 0) {
    let sum = left + right;
    let change = left - right;
    if (dot(left, right) > 0 && dot(change, change) <= 0.25 * dot(sum, sum)) { return sum / 2; }
    return select(right, left, dot(left, left) < dot(right, right));
  }
  if (next.w > 0) { return right; }
  return left;
}

/** Flux-normalized stellar filtering over a locally linear lensed detector footprint. */
@compute @workgroup_size(8, 8)
fn composite_stars(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(composite_image);
  if (any(id.xy >= size)) { return; }
  let pixel = vec2i(id.xy);
  let base = textureLoad(foreground, pixel, 0);
  let arrival = textureLoad(arrivals, pixel, 0);
  let transfer = textureLoad(transmissions, pixel, 0);
  if (base.a == 0 || arrival.w <= 0 || transfer.x == 0 || optical_frame.sampling.w != 0 || optical_frame.plasma.w != 0) {
    textureStore(composite_image, pixel, base);
    return;
  }
  let n = normalize(arrival.xyz);
  var axis = vec3f(0, 0, 1);
  if (abs(n.z) > 0.9) { axis = vec3f(0, 1, 0); }
  let east = normalize(cross(axis, n));
  let north = cross(n, east);
  let domain = textureLoad(domains, pixel, 0);
  var dx = arrival_difference(pixel, vec2i(1, 0), arrival, domain, transfer.y);
  var dy = arrival_difference(pixel, vec2i(0, 1), arrival, domain, transfer.y);
  let pixel_angle = 2 * optical_frame.sampling.z / f32(size.y);
  // Isolated source pixels use the unlensed detector footprint; this is an explicit boundary approximation.
  if (dot(dx, dx) == 0) { dx = east * pixel_angle; }
  if (dot(dy, dy) == 0) { dy = north * pixel_angle; }
  let x = vec2f(dot(dx, east), dot(dx, north));
  let y = vec2f(dot(dy, east), dot(dy, north));
  var total = vec3f(0);
  let coefficients = textureSampleGrad(sky_spectra, sky_sampler, n, dx, dy).rgb / 1024;
  let temperatures = vec3f(4500, 6500, 12000);
  for (var population = 0u; population < 3u; population++) {
    let spectrum = blackbody_radiance(temperatures[population] / arrival.w);
    if (spectrum.a == 0) { textureStore(composite_image, pixel, vec4f(0)); return; }
    total += coefficients[population] * spectrum.rgb;
  }
  let stellar = stellar_radiance(n, east, north, x, y, arrival.w);
  if (stellar.a == 0) { textureStore(composite_image, pixel, vec4f(0)); return; }
  total += stellar.rgb;
  let intensity = base.rgb + optical_frame.appearance.w * transfer.x * total;
  let peak = max(max(abs(intensity.r), abs(intensity.g)), abs(intensity.b));
  var storage = intensity;
  if (peak > 65504) { storage *= 65504 / peak; }
  textureStore(composite_image, pixel, vec4f(storage, base.a));
}
