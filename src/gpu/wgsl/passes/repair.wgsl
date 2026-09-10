@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var optical_map: texture_2d<f32>;
@group(0) @binding(2) var beam_index: texture_storage_2d<r32uint, write>;
@group(0) @binding(3) var<storage, read_write> repaired_beams: array<SkyBeam>;
struct RepairStatistics {
  candidates: atomic<u32>,
  rays: atomic<u32>,
  repaired: atomic<u32>,
}
@group(0) @binding(4) var<storage, read_write> repair_count: RepairStatistics;
@group(0) @binding(6) var<uniform> repair_observer_frame: Observer64;
@group(0) @binding(5) var<storage, read_write> repair_pixels: array<vec2u>;
/**
 * All lanes participate, including padding and rejected pixels. No fixed
 * subgroup width or subgroup-uniform-control-flow extension is assumed.
 */
fn reserve_repair(candidate: bool) -> u32 {
  let count = subgroupAdd(u32(candidate));
  let offset = subgroupExclusiveAdd(u32(candidate));
  var first = 0u;
  if (subgroupElect() && count != 0u) {
    first = atomicAdd(&repair_count.candidates, count);
  }
  return subgroupBroadcastFirst(first) + offset;
}

fn needs_beam(pixel: vec2i) -> bool {
  if (any(vec2u(pixel) >= textureDimensions(optical_map))) { return false; }
  let center = textureLoad(optical_map, pixel, 0);
  if (center.w <= 0.0) { return false; }
  let direction = celestial_direction(center);
  let stencil = celestial_beam(optical_map, pixel, center);
  let dx = stencil.dx.xyz;
  let dy = stencil.dy.xyz;
  return !celestial_has_area(direction, dx, dy);
}

@compute @workgroup_size(8, 8)
fn find_beams(@builtin(global_invocation_id) id: vec3u) {
  let candidate = needs_beam(vec2i(id.xy));
  let index = reserve_repair(candidate);
  if (any(id.xy >= textureDimensions(optical_map))) { return; }
  textureStore(beam_index, vec2i(id.xy), vec4u(0u));
  if (candidate && index < arrayLength(&repair_pixels)) {
    repair_pixels[index] = id.xy;
  }
}

/** One workgroup owns one ray; all collective calls follow uniform path decisions. */
var<workgroup> repair_ready: bool;
var<workgroup> repair_prepared: PreparedSkyBeam;
/** Lane zero prepares and stores the physical ray; all lanes integrate its shared path. */
fn finish_repair(index: u32, lane: u32) {
  let ready = workgroupUniformLoad(&repair_ready);
  if (!ready) { return; }
  let prepared = workgroupUniformLoad(&repair_prepared);
  let sky = evaluate_prepared_sky(frame,prepared,lane);
  if (lane == 0u) {
    let beam = beam_from_prepared_sky(sky,prepared);
    let pixel = vec2i(repair_pixels[index]);
    let center = textureLoad(optical_map, pixel, 0);
    atomicAdd(&repair_count.rays, 1u);
    repaired_beams[index] = beam;
    if (beam.dx.w <= 0.0 || beam.dy.w <= 0.0) { return; }
    if (!celestial_has_area(celestial_direction(center), beam.dx.xyz, beam.dy.xyz)) { return; }
    atomicAdd(&repair_count.repaired, 1u);
    textureStore(beam_index, pixel, vec4u(index + 1u, 0u, 0u, 0u));
  }
}

@compute @workgroup_size(32)
fn repair_beams(@builtin(workgroup_id) id: vec3u, @builtin(local_invocation_index) lane: u32) {
  if (lane == 0u) {
    repair_ready = id.x < min(atomicLoad(&repair_count.candidates), arrayLength(&repair_pixels));
    if (repair_ready) {
      let pixel = repair_pixels[id.x];
      let center = textureLoad(optical_map, vec2i(pixel), 0);
      repair_prepared = prepare_sky_beam(frame,vec2f(pixel),center.w,repair_observer_frame);
    }
  }
  finish_repair(id.x, lane);
}
