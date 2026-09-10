/** Physical image position, unweighted spectral flux, retained identity, and convergence diagnostics. */
struct StellarRefinedImage {
  screen: vec4f,
  radiance: vec4f,
  parameters: vec4f,
  convergence: vec4f,
}
@group(0) @binding(9) var<storage, read_write> refined_images: array<StellarRefinedImage>;
@group(0) @binding(10) var<storage, read_write> stellar_dispatch: array<u32>;

/** Only unfinished roots survive to the next trial; identity stays in the canonical image arrays. */
struct StellarTrials { count: atomic<u32>, current: u32, indices: array<u32> }
@group(0) @binding(17) var<storage, read_write> stellar_trials: StellarTrials;

@compute @workgroup_size(1)
fn prepare_stellar_refinement() {
  var count = atomicLoad(&stellar_trials.count);
  if (stellar_iteration.x == 0u) { count = min(atomicLoad(&search_statistics[0]),arrayLength(&refined_images)); }
  stellar_dispatch[0] = count;
  stellar_dispatch[1] = 1u;
  stellar_dispatch[2] = 1u;
  stellar_dispatch[4] = (count + 63u) / 64u;
  stellar_dispatch[5] = 1u;
  stellar_dispatch[6] = 1u;
  stellar_trials.current = count;
  atomicStore(&stellar_trials.count,0u);
}

fn stellar_trial_index(slot: u32) -> u32 {
  if (stellar_iteration.x == 0u) { return slot; }
  return stellar_trials.indices[(stellar_iteration.x & 1u) * arrayLength(&refined_images) + slot];
}

/** Prepare one trial ray; its value and Jacobian are evaluated together by the next pass. */
@compute @workgroup_size(64)
fn prepare_stellar_rays(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= stellar_trials.current) { return; }
  let index = stellar_trial_index(id.x);
  let seed = image_seeds[index];
  if (stellar_iteration.x == 0u) {
    refined_images[index] = StellarRefinedImage(vec4f(seed.screen.xy,0.0,0.0),vec4f(0.0),seed.parameters,vec4f(0.0));
    stellar_work[index].state = vec4u(0u);
    stellar_work[index].endpoint = vec4f(0.0);
    stellar_work[index].previous = vec4f(0.0);
    let circle = critical_circle64(seed.parameters.x);
    stellar_work[index].parameter_words = store_stellar_parameters(array<Soft64,3>(circle[0],circle[1],soft64_number(exp(seed.parameters.y))));
  } else if (stellar_work[index].state.x != 0u) { return; }
  let previous = stellar_work[index].previous;
  let accepted = stellar_work[index].state.w != 0u;
  let parameters = load_stellar_parameters(stellar_work[index].parameter_words);
  var frame = search_frame;
  frame.viewport = vec4f(search_frame.viewport.xy,0.0,0.0);
  let minimum_offset = soft64_number(exp(search_settings.minimum_log_offset));
  let maximum_offset = soft64_number(exp(search_settings.maximum_log_offset));
  for (var trial = stellar_work[index].state.z; trial < stellar_iteration.w; trial++) {
    if (!accepted && trial != 0u) { break; }
    var physical = parameters;
    var reported = seed.parameters.xy;
    if (accepted) {
      let step = previous.zw * exp2(-f32(trial));
      physical = critical_update64(parameters,step);
      reported = previous.xy + step;
      reported.x -= 6.283185307179586 * floor(reported.x / 6.283185307179586);
    }
    if (!soft64_valid(physical[2]) || soft64_less(physical[2],minimum_offset) || soft64_less(maximum_offset,physical[2])) { continue; }
    let orbit = critical_orbit_circle64(frame,search_observer,array<Soft64,2>(physical[0],physical[1]));
    let critical = critical_orbit_direction64(frame,search_observer,orbit);
    if (soft64_zero(critical[3])) { continue; }
    let pixel = critical_pixel64(frame,array<Soft64,3>(critical[0],critical[1],critical[2]),physical[2]);
    if (!pixel.valid) { continue; }
    // Preparation owns radial destination, ordered disk opacity and the retained
    // branch. No separate scalar ray supplies a differently rounded residual.
    let prepared = prepare_differential_sky_precise(frame,pixel.coordinate,seed.parameters.w,search_observer);
    if (!prepared.path.valid) { continue; }
    let tangent = critical_orbit_direction_derivative64(frame,search_observer,orbit);
    if (!d64_valid(tangent[0]) || !d64_valid(tangent[1]) || !d64_valid(tangent[2])) { continue; }
    stellar_work[index].prepared = store_stellar_beam(prepared);
    stellar_work[index].geometry = critical_parameter_geometry(prepared,critical_screen_derivative64(frame,tangent,physical[2]));
    stellar_work[index].trial_words = store_stellar_parameters(physical);
    stellar_work[index].trial_words[1] = vec4u(physical[2],orbit[0]);
    stellar_work[index].state.z = trial;
    refined_images[index].parameters = vec4f(reported,seed.parameters.zw);
    refined_images[index].screen = vec4f(pixel.base,pixel.jitter);
    return;
  }
  stellar_work[index].state.x = 2u;
  atomicAdd(&search_statistics[6],1u);
}

var<workgroup> stellar_prepared: PreparedSkyBeam;
var<workgroup> stellar_frame: Frame;
var<workgroup> stellar_state: u32;
var<workgroup> stellar_parameters_work: vec2f;


/** One transport supplies a coherent sky value, source residual, and solid-angle Jacobian. */
@compute @workgroup_size(32)
fn refine_stellar_images(@builtin(workgroup_id) id: vec3u, @builtin(local_invocation_index) lane: u32) {
  let index = stellar_trial_index(id.x);
  if (lane == 0u) {
    stellar_state = stellar_work[index].state.x;
    if (stellar_state == 0u) {
      stellar_prepared = load_stellar_beam(stellar_work[index].prepared);
      stellar_frame = search_frame;
      stellar_frame.viewport = vec4f(search_frame.viewport.xy,0.0,0.0);
      stellar_parameters_work = refined_images[index].parameters.xy;
    }
  }
  workgroupBarrier();
  if (workgroupUniformLoad(&stellar_state) != 0u) { return; }
  let frame = workgroupUniformLoad(&stellar_frame);
  let prepared = workgroupUniformLoad(&stellar_prepared);
  let sky = evaluate_prepared_sky(frame,prepared,lane);
  if (lane == 0u) {
    let node = star_nodes[u32(refined_images[index].parameters.z)];
    // The prepared basis is orthonormal: its solid-angle area is invariant.
    // Restoring nearly parallel f32 screen gradients first loses the small area.
    let beam = beam_from_differential(sky);
    let endpoint = vec4f(sky.mu.x,sky.phi.x,sky.energy,refined_images[index].parameters.w);
    let direction = celestial_direction(endpoint);
    let valid = beam.dx.w > 0.0 && beam.dy.w > 0.0 && celestial_has_area(direction,beam.dx.xyz,beam.dy.xyz);
    stellar_state = select(2u,0u,valid);
    let error = length(direction-node.lower.xyz);
    refined_images[index].convergence = vec4f(error,f32(stellar_iteration.x+1u),select(20.0,0.0,valid),sky.energy);
    var axis = 0u;
    if (abs(node.lower.y)>abs(node.lower.x)) { axis=1u; }
    if (abs(node.lower.z)>abs(node.lower[axis])) { axis=2u; }
    let first = (axis+1u)%3u;
    let second = (axis+2u)%3u;
    let chart = vec2f(node.lower[first],node.lower[second])/node.lower[axis];
    let residual = vec2f(direction[first],direction[second])-chart*direction[axis];
    let screen_u = vec2f(beam.dx[first],beam.dx[second])-chart*beam.dx[axis];
    let screen_v = vec2f(beam.dy[first],beam.dy[second])-chart*beam.dy[axis];
    let screen_extent = max(abs(screen_u),abs(screen_v));
    let screen_scale = max(screen_extent.x,screen_extent.y);
    var position_converged = false;
    if (valid && screen_scale > 0.0) {
      let u = screen_u/screen_scale;
      let v = screen_v/screen_scale;
      let determinant = abs(stellar_cross(u,v));
      let correction = vec2f(stellar_cross(residual,v),stellar_cross(u,residual));
      // First-order detector displacement in the orthonormal physical pixel basis.
      // An angular residual alone does not bound position near a magnified image.
      position_converged = determinant > 0.0 && length(correction) <= (0.0005*screen_scale)*determinant;
    }
    if (valid && error <= 5e-6 && position_converged) {
      let extent = max(abs(beam.dx.xyz),abs(beam.dy.xyz));
      let scale = max(max(extent.x,extent.y),extent.z);
      let area = abs(dot(direction,cross(beam.dx.xyz/scale,beam.dy.xyz/scale)));
      let energy = sky.energy;
      if (energy > 0.0 && energy >= node.upper.y/1000000.0) {
        let spectrum = blackbody_radiance(node.upper.y/energy);
        if (spectrum.a > 0.0) {
          let flux = ((node.upper.x*spectrum.rgb/scale)/scale)/area;
          refined_images[index].radiance = vec4f(flux,1.0);
          refined_images[index].convergence.z = (area*scale)*scale;
          stellar_state = 1u;
        } else { stellar_state = 2u; }
      } else { stellar_state = 2u; }
    } else if (!valid || (stellar_work[index].state.w != 0u && error >= stellar_work[index].endpoint.w)) {
      // Backtracking compares values from the same differentiated evaluator.
      // Rejected trials never replace the accepted point or its Newton direction.
      let retry = stellar_work[index].state.w != 0u &&
        stellar_work[index].state.z + 1u < stellar_iteration.w;
      stellar_state = select(2u,0u,retry);
      stellar_work[index].state.z++;
    } else {
      let geometry = stellar_work[index].geometry;
      let along = geometry.xy;
      let across = geometry.zw;
      let du = beam.dx.xyz*along.x + beam.dy.xyz*along.y;
      let dv = beam.dx.xyz*across.x + beam.dy.xyz*across.y;
      let u = vec2f(du[first],du[second])-chart*du[axis];
      let v = vec2f(dv[first],dv[second])-chart*dv[axis];
      let extent = max(abs(u),abs(v));
      let scale = max(extent.x,extent.y);
      if (!(scale>0.0)) { stellar_state=2u; }
      else {
        let determinant = stellar_cross(u/scale,v/scale);
        if (!(abs(determinant)>1e-20)) { stellar_state=2u; }
        else {
          var step = -vec2f(stellar_cross(residual/scale,v/scale),stellar_cross(u/scale,residual/scale))/determinant;
          step /= max(1.0,max(abs(step.x)/0.1,abs(step.y)/0.5));
          stellar_work[index].state.y++;
          if (stellar_work[index].state.y < stellar_iteration.z) {
            stellar_work[index].previous = vec4f(stellar_parameters_work,step);
            stellar_work[index].endpoint = vec4f(endpoint.xyz,error);
            stellar_work[index].parameter_words = stellar_work[index].trial_words;
            stellar_work[index].state.z = 0u;
            stellar_work[index].state.w = 1u;
          } else { stellar_state = 2u; }
        }
      }
    }
    if (stellar_iteration.x + 1u == stellar_iteration.y && stellar_state == 0u) { stellar_state = 2u; }
    stellar_work[index].state.x = stellar_state;
    if (stellar_state == 2u) { atomicAdd(&search_statistics[6],1u); }
    if (stellar_state == 0u) {
      let slot = atomicAdd(&stellar_trials.count,1u);
      stellar_trials.indices[((stellar_iteration.x + 1u) & 1u) * arrayLength(&refined_images) + slot] = index;
    }
  }
}
