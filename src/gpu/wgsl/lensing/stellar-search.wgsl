/** Positive radial offsets are logarithmically spaced outside the vacuum critical curve. */
struct StellarSearchSettings {
  columns: u32,
  rows: u32,
  maximum_log_offset: f32,
  minimum_log_offset: f32,
}
struct StellarSearchSample {
  endpoint: vec4f,
  screen: vec4f,
}
/** A candidate still requires physical-ray refinement and a flux/ownership decision. */
struct StellarImageSeed {
  parameters: vec4f,
  screen: vec4f,
}
@group(0) @binding(0) var<uniform> search_frame: Frame;
@group(0) @binding(1) var<uniform> search_observer: Observer64;
@group(0) @binding(3) var<storage, read> critical_columns: array<array<Soft64,4>>;
@group(0) @binding(4) var<storage, read_write> search_samples: array<StellarSearchSample>;
@group(0) @binding(5) var<storage, read> star_nodes: array<StarNode>;
@group(0) @binding(6) var<storage, read_write> image_seeds: array<StellarImageSeed>;
/** Candidates, failed projections/endpoints, mixed parents, failed inversions, overflow, failed refinement, mixed children. */
@group(0) @binding(7) var<storage, read_write> search_statistics: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> search_settings: StellarSearchSettings;

/** One slot per coarse cell; only mixed parents enter the midpoint queue. */
@group(0) @binding(13) var<storage, read_write> mixed_stellar_cells: array<u32>;
/** Five new samples per parent: bottom, left, center, right, and top. */
@group(0) @binding(14) var<storage, read_write> mixed_stellar_samples: array<StellarSearchSample>;
/** Queue slot plus one; zero marks an unsplit coarse cell. */
@group(0) @binding(15) var<storage, read_write> mixed_stellar_slots: array<u32>;

/** Exact unsigned 64-bit coarse-cell offsets followed by one sum per 256-cell block. */
@group(0) @binding(16) var<storage, read_write> stellar_seed_offsets: array<vec2u>;
struct StellarSeedAllocation {
  count: u32,
  base: u32,
  writing: bool,
}

fn record_stellar_search_failure(allocation: ptr<function,StellarSeedAllocation>, counter: u32) {
  if (!(*allocation).writing) { atomicAdd(&search_statistics[counter],1u); }
}

fn sample_stellar_point(half_column: u32, logarithm: f32) -> StellarSearchSample {
  let critical = critical_columns[half_column];
  var sample = StellarSearchSample(vec4f(0.0,0.0,0.0,-2.0),vec4f(0.0));
  if (soft64_zero(critical[3])) {
    atomicAdd(&search_statistics[1],1u);
    return sample;
  }
  let pixel = critical_pixel(search_frame,array<Soft64,3>(critical[0],critical[1],critical[2]),exp(logarithm));
  if (!pixel.valid) {
    atomicAdd(&search_statistics[1],1u);
    return sample;
  }
  var frame = search_frame;
  frame.viewport = vec4f(frame.viewport.xy,pixel.jitter);
  sample.endpoint = screen_ray_refined(frame,pixel.base,search_observer).value;
  sample.screen = vec4f(pixel.base,pixel.jitter);
  if (sample.endpoint.w == -2.0) { atomicAdd(&search_statistics[2],1u); }
  return sample;
}

@compute @workgroup_size(64)
fn sample_critical_strip(@builtin(global_invocation_id) id: vec3u) {
  let count = search_settings.columns * (search_settings.rows + 1u);
  if (id.x >= count) { return; }
  let column = id.x % search_settings.columns;
  let row = id.x / search_settings.columns;
  let logarithm = mix(search_settings.maximum_log_offset,search_settings.minimum_log_offset,
    f32(row)/f32(search_settings.rows));
  search_samples[id.x] = sample_stellar_point(2u*column,logarithm);
}

@compute @workgroup_size(1)
fn prepare_mixed_stellar_cells() {
  let count = min(atomicLoad(&search_statistics[3]),arrayLength(&mixed_stellar_cells));
  stellar_dispatch[4] = (5u*count+63u)/64u;
  stellar_dispatch[5] = 1u;
  stellar_dispatch[6] = 1u;
}

@compute @workgroup_size(64)
fn sample_mixed_stellar_cells(@builtin(global_invocation_id) id: vec3u) {
  let count = min(atomicLoad(&search_statistics[3]),arrayLength(&mixed_stellar_cells));
  if (id.x >= 5u*count) { return; }
  let cell = mixed_stellar_cells[id.x/5u];
  let offset = array<vec2u,5>(vec2u(1u,0u),vec2u(0u,1u),vec2u(1u),vec2u(2u,1u),vec2u(1u,2u))[id.x%5u];
  let column = (2u*(cell%search_settings.columns)+offset.x)%(2u*search_settings.columns);
  let row = 2u*(cell/search_settings.columns)+offset.y;
  let logarithm = mix(search_settings.maximum_log_offset,search_settings.minimum_log_offset,
    f32(row)/f32(2u*search_settings.rows));
  mixed_stellar_samples[id.x] = sample_stellar_point(column,logarithm);
}

fn retain_stellar_seed(cell: StellarPatch, origin: vec2f, span: f32, source_index: u32, branch: f32,
    screens: array<vec4f,4>, allocation: ptr<function,StellarSeedAllocation>) {
  let node = star_nodes[source_index];
  let roots = stellar_patch_roots(cell, node.lower.xyz);
  if (!roots.valid) { record_stellar_search_failure(allocation,4u); return; }
  for (var index = 0u; index < roots.count; index++) {
    let uv = roots.positions[index];
    let value = mix(mix(cell.lower_left,cell.lower_right,uv.x), mix(cell.upper_left,cell.upper_right,uv.x),uv.y);
    if (!(dot(value.xyz,node.lower.xyz) > 0.0)) { continue; }
    let destination = (*allocation).base + (*allocation).count;
    (*allocation).count++;
    if (!(*allocation).writing || destination >= arrayLength(&image_seeds)) { continue; }
    let coordinate = origin + span*uv;
    let phase = 6.283185307179586 * coordinate.x / f32(search_settings.columns);
    let logarithm = mix(search_settings.maximum_log_offset,search_settings.minimum_log_offset,
      coordinate.y / f32(search_settings.rows));
    // This interpolated position is only a work locator. A curved strip requires
    // a fresh critical projection and physical-ray refinement before radiation.
    let bottom = mix(screens[0].xy + screens[0].zw, screens[1].xy + screens[1].zw, uv.x);
    let top = mix(screens[2].xy + screens[2].zw, screens[3].xy + screens[3].zw, uv.x);
    image_seeds[destination] = StellarImageSeed(vec4f(phase,logarithm,f32(source_index),branch),
      vec4f(mix(bottom,top,uv.y),origin));
  }
}

fn stellar_cell_has_one_branch(values: array<StellarSearchSample,4>) -> bool {
  return all(vec3f(values[1].endpoint.w,values[2].endpoint.w,values[3].endpoint.w) == vec3f(values[0].endpoint.w));
}

fn search_stellar_cell(values: array<StellarSearchSample,4>, origin: vec2f, span: f32, allocation: ptr<function,StellarSeedAllocation>) {
  let first = values[0];
  let second = values[1];
  let third = values[2];
  let fourth = values[3];
  let branch = first.endpoint.w;
  if (!(branch > 0.0)) { return; }
  let cell = StellarPatch(vec4f(celestial_direction(first.endpoint),first.endpoint.z),
    vec4f(celestial_direction(second.endpoint),second.endpoint.z),
    vec4f(celestial_direction(third.endpoint),third.endpoint.z),
    vec4f(celestial_direction(fourth.endpoint),fourth.endpoint.z));
  query_stellar_patch(cell,origin,span,branch,array<vec4f,4>(first.screen,second.screen,third.screen,fourth.screen),allocation);
}

fn query_stellar_patch(cell: StellarPatch, origin: vec2f, span: f32, branch: f32, screens: array<vec4f,4>, allocation: ptr<function,StellarSeedAllocation>) {
  let lengths = vec4f(length(cell.lower_left.xyz),length(cell.lower_right.xyz),length(cell.upper_left.xyz),length(cell.upper_right.xyz));
  if (any(lengths <= vec4f(0.0))) { record_stellar_search_failure(allocation,4u); return; }
  let directions = array<vec3f,4>(cell.lower_left.xyz/lengths.x,cell.lower_right.xyz/lengths.y,
    cell.upper_left.xyz/lengths.z,cell.upper_right.xyz/lengths.w);
  let sum = directions[0]+directions[1]+directions[2]+directions[3];
  if (!(length(sum) > 0.0)) { record_stellar_search_failure(allocation,4u); return; }
  let axis = normalize(sum);
  let alignment = min(min(dot(axis,directions[0]),dot(axis,directions[1])),min(dot(axis,directions[2]),dot(axis,directions[3])));
  if (!(alignment > 0.0)) { record_stellar_search_failure(allocation,4u); return; }
  let radius = max(max(length(directions[0]-axis),length(directions[1]-axis)),
    max(length(directions[2]-axis),length(directions[3]-axis)))+9.5367431640625e-7;
  let radius2 = radius * radius;
  var source_index = 0u;
  loop {
    if (source_index >= arrayLength(&star_nodes)) { break; }
    let node = star_nodes[source_index];
    if (node.lower.w >= 0.0) {
      let offset = clamp(axis,node.lower.xyz,node.upper.xyz) - axis;
      if (dot(offset,offset) > radius2) { source_index = u32(node.lower.w); continue; }
    } else if (node.upper.x > 0.0) {
      let offset = node.lower.xyz - axis;
      if (dot(offset,offset) <= radius2 && dot(axis,node.lower.xyz) > 0.0) {
        retain_stellar_seed(cell,origin,span,source_index,branch,screens,allocation);
      }
    }
    source_index += 1u;
  }
}

@compute @workgroup_size(64)
fn classify_stellar_cells(@builtin(global_invocation_id) id: vec3u) {
  let columns = search_settings.columns;
  if (id.x >= columns*search_settings.rows) { return; }
  let column = id.x%columns;
  let row = id.x/columns;
  let next = (column+1u)%columns;
  let values = array<StellarSearchSample,4>(search_samples[row*columns+column],
    search_samples[row*columns+next],search_samples[(row+1u)*columns+column],search_samples[(row+1u)*columns+next]);
  if (!stellar_cell_has_one_branch(values)) {
    // The queue is sized to the complete coarse grid, so this append cannot overflow.
    let slot = atomicAdd(&search_statistics[3],1u);
    mixed_stellar_cells[slot] = id.x;
    mixed_stellar_slots[id.x] = slot+1u;
    return;
  }
  mixed_stellar_slots[id.x] = 0u;
}

fn search_mixed_stellar_cell(cell: u32, slot: u32, allocation: ptr<function,StellarSeedAllocation>) {
  let columns = search_settings.columns;
  let column = cell%columns;
  let row = cell/columns;
  let next = (column+1u)%columns;
  let base = 5u*slot;
  let values = array<StellarSearchSample,9>(search_samples[row*columns+column],
    mixed_stellar_samples[base],search_samples[row*columns+next],
    mixed_stellar_samples[base+1u],mixed_stellar_samples[base+2u],mixed_stellar_samples[base+3u],
    search_samples[(row+1u)*columns+column],mixed_stellar_samples[base+4u],search_samples[(row+1u)*columns+next]);
  for (var y = 0u; y < 2u; y++) {
    for (var x = 0u; x < 2u; x++) {
      let at = 3u*y+x;
      let corners = array<StellarSearchSample,4>(values[at],values[at+1u],values[at+3u],values[at+4u]);
      if (!stellar_cell_has_one_branch(corners)) { record_stellar_search_failure(allocation,7u); continue; }
      search_stellar_cell(corners,vec2f(f32(column),f32(row))+0.5*vec2f(f32(x),f32(y)),0.5,allocation);
    }
  }
}

/** A shared physical edge midpoint makes the coarse/fine source maps meet. */
fn search_coarse_stellar_cell(cell: u32, allocation: ptr<function,StellarSeedAllocation>) {
  let columns = search_settings.columns;
  let rows = search_settings.rows;
  let column = cell%columns;
  let row = cell/columns;
  let next = (column+1u)%columns;
  let corners = array<StellarSearchSample,4>(search_samples[row*columns+column],
    search_samples[row*columns+next],search_samples[(row+1u)*columns+column],search_samples[(row+1u)*columns+next]);
  let branch = corners[0].endpoint.w;
  if (!(branch > 0.0)) { return; }
  var adjacent = array<u32,4>(0u,mixed_stellar_slots[row*columns+(column+columns-1u)%columns],
    mixed_stellar_slots[row*columns+next],0u);
  if (row > 0u) { adjacent[0] = mixed_stellar_slots[(row-1u)*columns+column]; }
  if (row+1u < rows) { adjacent[3] = mixed_stellar_slots[(row+1u)*columns+column]; }
  if (all(vec4u(adjacent[0],adjacent[1],adjacent[2],adjacent[3]) == vec4u(0u))) {
    search_stellar_cell(corners,vec2f(f32(column),f32(row)),1.0,allocation);
    return;
  }
  var sky: array<vec4f,9>;
  var screens: array<vec4f,9>;
  var branches: array<f32,9>;
  let corner_indices = array<u32,4>(0u,2u,6u,8u);
  for (var index = 0u; index < 4u; index++) {
    let at = corner_indices[index];
    sky[at] = vec4f(celestial_direction(corners[index].endpoint),corners[index].endpoint.z);
    screens[at] = corners[index].screen;
    branches[at] = branch;
  }
  let edge_indices = array<u32,4>(1u,3u,5u,7u);
  let edge_corners = array<vec2u,4>(vec2u(0u,2u),vec2u(0u,6u),vec2u(2u,8u),vec2u(6u,8u));
  let neighboring_samples = array<u32,4>(4u,3u,1u,0u);
  for (var index = 0u; index < 4u; index++) {
    let at = edge_indices[index];
    if (adjacent[index] != 0u) {
      let point = mixed_stellar_samples[5u*(adjacent[index]-1u)+neighboring_samples[index]];
      sky[at] = vec4f(0.0);
      if (point.endpoint.w > 0.0) { sky[at] = vec4f(celestial_direction(point.endpoint),point.endpoint.z); }
      screens[at] = point.screen;
      branches[at] = point.endpoint.w;
    } else {
      let indices = edge_corners[index];
      let average = 0.5*(sky[indices.x]+sky[indices.y]);
      let magnitude = length(average.xyz);
      if (!(magnitude > 0.0)) { record_stellar_search_failure(allocation,4u); return; }
      sky[at] = average;
      screens[at] = 0.5*(screens[indices.x]+screens[indices.y]);
      branches[at] = branch;
    }
  }
  let center = 0.25*(sky[0]+sky[2]+sky[6]+sky[8]);
  let magnitude = length(center.xyz);
  if (!(magnitude > 0.0)) { record_stellar_search_failure(allocation,4u); return; }
  sky[4] = center;
  screens[4] = 0.25*(screens[0]+screens[2]+screens[6]+screens[8]);
  branches[4] = branch;
  for (var y = 0u; y < 2u; y++) {
    for (var x = 0u; x < 2u; x++) {
      let at = 3u*y+x;
      if (any(vec4f(branches[at],branches[at+1u],branches[at+3u],branches[at+4u]) != vec4f(branch))) {
        record_stellar_search_failure(allocation,7u);
        continue;
      }
      query_stellar_patch(StellarPatch(sky[at],sky[at+1u],sky[at+3u],sky[at+4u]),
        vec2f(f32(column),f32(row))+0.5*vec2f(f32(x),f32(y)),0.5,branch,
        array<vec4f,4>(screens[at],screens[at+1u],screens[at+3u],screens[at+4u]),allocation);
    }
  }
}


fn search_allocated_stellar_cell(cell: u32, allocation: ptr<function,StellarSeedAllocation>) {
  let slot = mixed_stellar_slots[cell];
  if (slot == 0u) { search_coarse_stellar_cell(cell,allocation); }
  else { search_mixed_stellar_cell(cell,slot-1u,allocation); }
}

/** Count before allocation so queue overflow cannot choose images by scheduling order. */
@compute @workgroup_size(64)
fn count_stellar_seeds(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= search_settings.columns*search_settings.rows) { return; }
  var allocation = StellarSeedAllocation(0u,0u,false);
  search_allocated_stellar_cell(id.x,&allocation);
  stellar_seed_offsets[id.x] = vec2u(allocation.count,0u);
}

/** Integer work counts use a carry word; optical arithmetic remains f32. */
fn add_stellar_counts(a: vec2u, b: vec2u) -> vec2u {
  let lower = a.x+b.x;
  return vec2u(lower,a.y+b.y+u32(lower<a.x));
}

/** Exact count difference for a >= b. */
fn subtract_stellar_counts(a: vec2u, b: vec2u) -> vec2u {
  return vec2u(a.x-b.x,a.y-b.y-u32(a.x<b.x));
}

var<workgroup> stellar_prefix: array<vec2u,256>;

/** Inclusive workgroup scan, converted to an exclusive offset for every coarse cell. */
@compute @workgroup_size(256)
fn scan_stellar_seed_blocks(@builtin(global_invocation_id) id: vec3u,
    @builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
  let cells = search_settings.columns*search_settings.rows;
  var count = vec2u(0u);
  if (id.x < cells) { count = stellar_seed_offsets[id.x]; }
  stellar_prefix[lane] = count;
  workgroupBarrier();
  for (var distance = 1u; distance < 256u; distance *= 2u) {
    var preceding = vec2u(0u);
    if (lane >= distance) { preceding = stellar_prefix[lane-distance]; }
    workgroupBarrier();
    stellar_prefix[lane] = add_stellar_counts(stellar_prefix[lane],preceding);
    workgroupBarrier();
  }
  if (id.x < cells) {
    stellar_seed_offsets[id.x] = subtract_stellar_counts(stellar_prefix[lane],count);
  }
  if (lane == 255u) { stellar_seed_offsets[cells+group.x] = stellar_prefix[lane]; }
}

/** Only block sums are serial: 128 counts for the default 32768-cell grid. */
@compute @workgroup_size(1)
fn allocate_stellar_seed_blocks() {
  let cells = search_settings.columns*search_settings.rows;
  let blocks = (cells+255u)/256u;
  var total = vec2u(0u);
  for (var block = 0u; block < blocks; block++) {
    let count = stellar_seed_offsets[cells+block];
    stellar_seed_offsets[cells+block] = total;
    total = add_stellar_counts(total,count);
  }
  // Public diagnostic counters saturate; allocations retain their exact carry
  // words so billions of discarded candidates cannot wrap into retained slots.
  let reported = select(total.x,0xffffffffu,total.y != 0u);
  let retained = select(min(total.x,arrayLength(&image_seeds)),arrayLength(&image_seeds),total.y != 0u);
  let overflow = subtract_stellar_counts(total,vec2u(retained,0u));
  atomicStore(&search_statistics[0],reported);
  atomicStore(&search_statistics[5],select(overflow.x,0xffffffffu,overflow.y != 0u));
}

/** Replay the same local inversions into deterministic cell/source/root ranges. */
@compute @workgroup_size(64)
fn write_stellar_seeds(@builtin(global_invocation_id) id: vec3u) {
  let cells = search_settings.columns*search_settings.rows;
  if (id.x >= cells) { return; }
  let base = add_stellar_counts(stellar_seed_offsets[id.x],stellar_seed_offsets[cells+id.x/256u]);
  if (base.y != 0u || base.x >= arrayLength(&image_seeds)) { return; }
  var allocation = StellarSeedAllocation(0u,base.x,true);
  search_allocated_stellar_cell(id.x,&allocation);
}
