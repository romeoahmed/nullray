enable subgroups;

@group(0) @binding(0) var history: texture_2d<f32>;
@group(0) @binding(1) var<uniform> sample_count: vec4u;
@group(0) @binding(2) var<storage, read_write> counts: array<vec2u>;
var<workgroup> totals: array<atomic<u32>, 2>;

/**
 * Reduce in small integer blocks; CPU summation avoids a full-image u32 overflow.
 */
@compute @workgroup_size(16, 16)
fn coverage(@builtin(global_invocation_id) id: vec3u,
  @builtin(local_invocation_index) local: u32,
  @builtin(workgroup_id) group: vec3u) {
  let size = textureDimensions(history);
  var value = vec2u(0u);
  if (all(id.xy < size)) {
    // Recover integer sixteenth-sample units. Exploration's 4×4 quadrature
    // and whole photographic samples share this exact weight lattice.
    let failed = u32(round(textureLoad(history, vec2i(id.xy), 0).a * f32(sample_count.x * 16u)));
    value = vec2u(select(0u, 1u, failed > 0u), failed);
  }
  // Workgroup storage starts at zero. Integer subgroup reduction is exact;
  // elect one writer per subgroup without assuming its size or lane mapping.
  let sum = subgroupAdd(value);
  if (subgroupElect()) {
    atomicAdd(&totals[0], sum.x);
    atomicAdd(&totals[1], sum.y);
  }
  workgroupBarrier();
  if (local == 0u) {
    counts[group.y * ((size.x + 15u) / 16u) + group.x] = vec2u(atomicLoad(&totals[0]), atomicLoad(&totals[1]));
  }
}
