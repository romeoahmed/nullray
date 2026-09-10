/**
 * Translate a packed quarter-pixel boundary sample into the physical screen frame.
 */
@group(0) @binding(7) var<storage, read> boundary_pixels: array<vec2u>;
@group(0) @binding(8) var<storage, read> boundary_count: array<u32>;

fn needs_edge_beam(id: vec3u) -> bool {
  let size = textureDimensions(optical_map);
  if (any(id.xy >= size)) { return false; }
  let pixel = vec2i(id.xy);
  let columns = size.x / 4u;
  let boundary = (id.y / 4u) * columns + id.x / 4u;
  if (boundary >= min(boundary_count[0], arrayLength(&boundary_pixels))) { return false; }
  let center = textureLoad(optical_map, pixel, 0);
  if (center.w <= 0.0) { return false; }
  let origin = vec2i((id.xy / 4u) * 4u);
  let direction = celestial_direction(center);
  let stencil = celestial_patch_beam(optical_map, pixel, origin, center);
  let dx = stencil.dx.xyz;
  let dy = stencil.dy.xyz;
  return !celestial_has_area(direction, dx, dy);
}

@compute @workgroup_size(8, 8)
fn find_edge_beams(@builtin(global_invocation_id) id: vec3u) {
  let candidate = needs_edge_beam(id);
  let index = reserve_repair(candidate);
  if (any(id.xy >= textureDimensions(optical_map))) { return; }
  textureStore(beam_index, vec2i(id.xy), vec4u(0u));
  if (candidate && index < arrayLength(&repair_pixels)) {
    repair_pixels[index] = id.xy;
  }
}

@compute @workgroup_size(32)
fn repair_edge_beams(@builtin(workgroup_id) id: vec3u, @builtin(local_invocation_index) lane: u32) {
  if (lane == 0u) {
    repair_ready = id.x < min(atomicLoad(&repair_count.candidates), arrayLength(&repair_pixels));
    if (repair_ready) {
      let pixel = repair_pixels[id.x];
      let columns = textureDimensions(optical_map).x / 4u;
      let boundary = (pixel.y / 4u) * columns + pixel.x / 4u;
      let physical = vec2f(boundary_pixels[boundary]) + (vec2f(pixel % 4u) + 0.5) / 4.0 - 0.5;
      let center = textureLoad(optical_map, vec2i(pixel), 0);
      repair_prepared = prepare_sky_beam(frame,physical,center.w,repair_observer_frame);
    }
  }
  finish_repair(id.x, lane);
}
