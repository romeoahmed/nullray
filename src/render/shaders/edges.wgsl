@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var optical_map: texture_2d<f32>;
@group(0) @binding(2) var edge_index: texture_storage_2d<r32uint, write>;
@group(0) @binding(3) var<storage, read_write> edge_count: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> edge_pixels: array<vec2u>;
@group(0) @binding(5) var edge_samples: texture_storage_2d<rgba32float, write>;
@group(0) @binding(6) var edge_times: texture_storage_2d<r32float, write>;

@group(0) @binding(7) var primary_times: texture_2d<f32>;

/**
 * A geometry-only work indicator for the bounded disk modes, not a radiance error bound.
 */
fn disk_variation(a: vec4f, b: vec4f, delay: f32) -> bool {
  let spin = frame.space.x;
  let charge2 = frame.space.y * frame.space.y;
  if (min(a.x, b.x) <= charge2 || min(a.z, b.z) <= 0.0) { return true; }
  let ar = sqrt(a.x - charge2);
  let br = sqrt(b.x - charge2);
  let ad = a.x * a.x + spin * ar;
  let bd = b.x * b.x + spin * br;
  if (min(ad, bd) <= 0.0) { return true; }
  let aw = ar / ad;
  let bw = br / bd;
  let longitude = abs(atan2(sin(b.y - a.y), cos(b.y - a.y)));
  let phase = longitude + 24.0 * abs(bw - aw) + max(abs(aw), abs(bw)) * abs(delay);
  let radial = abs(log(b.x / a.x));
  // Cohort replacement contributes at most 2 * max(smoothstep') / 24 = 1/8 per M.
  return 7.7 * phase + 22.05 * radial + abs(delay) / 8.0 > 1.0 || abs(log(b.z / a.z)) > 0.03;
}

fn needs_edge(id: vec3u) -> bool {
  let size = vec2i(textureDimensions(optical_map));
  if (any(vec2i(id.xy) >= size)) { return false; }
  let pixel = vec2i(id.xy);
  let center = textureLoad(optical_map, pixel, 0);
  let branch = center.w;
  var center_time = 0.0;
  if (branch <= -3.0) { center_time = textureLoad(primary_times, pixel, 0).x; }
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      if (x == 0 && y == 0) { continue; }
      let neighbor = pixel + vec2i(x, y);
      if (any(neighbor < vec2i(0)) || any(neighbor >= size)) { continue; }
      let sample = textureLoad(optical_map, neighbor, 0);
      if (sample.w != branch) { return true; }
      if (branch <= -3.0 && disk_variation(center, sample,
          textureLoad(primary_times, neighbor, 0).x - center_time)) {
        return true;
      }
    }
  }
  return false;
}

@compute @workgroup_size(8, 8)
fn find_edges(@builtin(global_invocation_id) id: vec3u) {
  let edge = needs_edge(id);
  let count = subgroupAdd(u32(edge));
  let offset = subgroupExclusiveAdd(u32(edge));
  var first = 0u;
  if (subgroupElect() && count != 0u) {
    first = atomicAdd(&edge_count, count);
  }
  let index = subgroupBroadcastFirst(first) + offset;
  if (any(id.xy >= textureDimensions(optical_map))) { return; }
  let pixel = vec2i(id.xy);
  textureStore(edge_index, pixel, vec4u(0u));
  if (!edge) { return; }
  if (index >= arrayLength(&edge_pixels)) {
    // Queue exhaustion remains visible; a center sample is not completed refinement.
    textureStore(edge_index, pixel, vec4u(0xffffffffu));
    return;
  }
  edge_pixels[index] = id.xy;
  textureStore(edge_index, pixel, vec4u(index + 1u));
}

@compute @workgroup_size(64)
fn sample_edges(@builtin(global_invocation_id) id: vec3u) {
  let index = id.x / 16u;
  if (index >= min(atomicLoad(&edge_count), arrayLength(&edge_pixels))) { return; }
  let sample = vec2u(id.x % 4u, (id.x % 16u) / 4u);
  let offset = (vec2f(sample) + 0.5) / 4.0 - 0.5;
  let endpoint = screen_ray(frame, vec2f(edge_pixels[index]) + offset);
  let columns = textureDimensions(edge_samples).x / 4u;
  let destination = vec2i(vec2u(index % columns, index / columns) * 4u + sample);
  textureStore(edge_samples, destination, endpoint.value);
  textureStore(edge_times, destination, vec4f(endpoint.relative_time));
}
