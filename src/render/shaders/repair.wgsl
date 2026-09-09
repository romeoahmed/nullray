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
@group(0) @binding(5) var<storage, read_write> repair_pixels: array<vec2u>;
/**
 * One differentiated physical ray supplies both screen derivatives.
 * Keep failures explicit; no neighboring branch is substituted for the center.
 */
fn differential_beam(pixel: vec2f, branch: f32) -> SkyBeam {
  var derivative_valid = true;
  let status = &derivative_valid;
  let sky = differential_sky(frame, pixel, branch);
  if (!sky.valid) { return SkyBeam(vec4f(0.0), vec4f(0.0)); }
  let radius = d_sqrt(status, d_mul(status, d_sub(status, d_constant(status, 1.0), sky.mu), d_add(status, d_constant(status, 1.0), sky.mu)));
  let x = d_mul(status, radius, d_cos(status, sky.phi));
  let y = d_mul(status, radius, d_sin(status, sky.phi));
  if (!(*status)) { return SkyBeam(vec4f(0.0), vec4f(0.0)); }
  return SkyBeam(vec4f(x.y, y.y, sky.mu.y, 1.0), vec4f(x.z, y.z, sky.mu.z, 1.0));
}

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

fn repair_differential_beam(index: u32, physical_pixel: vec2f) {
  let pixel = vec2i(repair_pixels[index]);
  let center = textureLoad(optical_map, pixel, 0);
  let beam = differential_beam(physical_pixel, center.w);
  atomicAdd(&repair_count.rays, 1u);
  // Every geometry update replaces previous successful and failed results.
  repaired_beams[index] = beam;
  if (beam.dx.w <= 0.0 || beam.dy.w <= 0.0) { return; }
  if (!celestial_has_area(celestial_direction(center), beam.dx.xyz, beam.dy.xyz)) { return; }
  atomicAdd(&repair_count.repaired, 1u);
  textureStore(beam_index, pixel, vec4u(index + 1u, 0u, 0u, 0u));
}

@compute @workgroup_size(32)
fn repair_beams(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= min(atomicLoad(&repair_count.candidates), arrayLength(&repair_pixels))) { return; }
  repair_differential_beam(id.x, vec2f(repair_pixels[id.x]));
}
