struct KerrSurface {
  state: KerrOrbitState,
  fraction: f32,
  valid: bool,
}

/** Locate a boundary on the accepted dense segment; never reintegrate a clipped radial turn. */
fn kerr_surface(path: KerrOrbit, end: KerrOrbitState, h: f32, equator: bool, boundary_coordinate: f32) -> KerrSurface {
  let start_rate = kerr_derivative(path, path.state);
  let end_rate = kerr_derivative(path, end);
  if (!start_rate.valid || !end_rate.valid) { return KerrSurface(path.state, 0, false); }
  var left = 0.0;
  var right = 1.0;
  var endpoint = end;
  var before = path.state.radial.x - boundary_coordinate;
  if (equator) { before = path.state.direction.z; }
  for (var iteration = 0u; iteration < 20u; iteration++) {
    let middle = (left + right) / 2;
    if (middle == left || middle == right) { break; }
    let trial = kerr_interpolate(path.state, end, start_rate.state, end_rate.state, h, middle);
    var value = trial.radial.x - boundary_coordinate;
    if (equator) { value = trial.direction.z; }
    if (before * value > 0) { left = middle; }
    else { right = middle; endpoint = trial; }
  }
  return KerrSurface(endpoint, right, true);
}

/** kind: 0 unresolved, 1 asymptotic source, 2 disk surface, 3 singularity, 4 source-free characteristic. */
struct RayResult {
  path: KerrOrbit,
  block: vec3i,
  kind: u32,
  end_sign: f32,
  crossings: u32,
  transfer: Transfer,
}

struct HeatSurface {
  state: HeatState,
  fraction: f32,
  valid: bool,
}

/** Locate the first bracketed material or disk surface within an accepted Hamiltonian step. */
fn heat_surface(space: vec2f, chart: f32, state: HeatState, end: HeatState, h: f32, equator: bool, coordinate: f32) -> HeatSurface {
  var left = 0.0;
  var right = 1.0;
  let start_rate = heat_derivative(space, optical_frame.plasma.xy, optical_frame.heating, chart, state);
  let end_rate = heat_derivative(space, optical_frame.plasma.xy, optical_frame.heating, chart, end);
  if (!start_rate.valid || !end_rate.valid) { return HeatSurface(state, 0, false); }
  var endpoint = end;
  var before = state.position.x - coordinate;
  if (equator) { before = state.position.w; }
  for (var i = 0u; i < 24u; i++) {
    let middle = (left + right) / 2;
    if (middle == left || middle == right) { break; }
    let trial = heat_interpolate(state, end, start_rate.state, end_rate.state, h, middle);
    var value = trial.position.x - coordinate;
    if (equator) { value = trial.position.w; }
    if (before * value > 0) { left = middle; }
    else { right = middle; endpoint = trial; }
  }
  return HeatSurface(endpoint, right, true);
}

/** kind: 0 unresolved work, 1 return to the separable exterior, 2 emitting disk. */
struct HeatSegment {
  path: KerrOrbit,
  kind: u32,
  work: u32,
  crossings: u32,
  transfer: Transfer,
}

/** Coupled canonical flow only inside the compact heated annulus; its boundary remains transparent. */
fn trace_heat_segment(initial: KerrOrbit, budget: u32, recording: bool, block: vec3i, incoming: Transfer, screen: PolarizationScreen) -> HeatSegment {
  var transport = incoming;
  var radius = initial.state.radial.x;
  if (initial.inverse) { radius = 1 / radius; }
  let geometry = medium_geometry(initial.space, radius, initial.state.direction, initial.chart);
  var state = HeatState(vec4f(radius, normalize(initial.state.direction)), kerr_lower(geometry.frame, kerr_tangent(initial)), initial.state.radial.z);
  var h = -0.01 / max(1, length(initial.state.angular));
  var crossings = 0u;
  for (var attempt = 0u; attempt < budget; attempt++) {
    let rate = heat_derivative(initial.space, optical_frame.plasma.xy, optical_frame.heating, initial.chart, state);
    if (!rate.valid) { return HeatSegment(initial, 0, attempt + 1, crossings, transport); }
    // Resolve the known spatial and temporal pattern scales before estimating local integration error.
    let phase_rate = 6.283185307179586 * abs(rate.state.position.x) / (optical_frame.heating.y - optical_frame.heating.x)
      + 6 * length(rate.state.position.yzw) + abs(optical_frame.heating.w * rate.state.time);
    if (phase_rate > 0) { h = -min(abs(h), 1 / phase_rate); }
    let epoch = state.time + (1 - initial.chart) * kerr_chart_primitives(initial.space, state.position.x).y;
    var epoch_speed = abs(rate.state.time);
    if (initial.chart < 0) {
      let r = state.position.x;
      let delta = r * r - 2 * r + dot(initial.space, initial.space);
      if (!(delta > 0)) { return HeatSegment(initial, 0, attempt + 1, crossings, transport); }
      epoch_speed += 2 * (r * r + initial.space.x * initial.space.x) / delta * abs(rate.state.position.x);
    }
    h = matter_limit(state.position.x, state.position.yzw, abs(rate.state.position.x), rate.state.position.yzw, epoch, epoch_speed, h);
    let trial = heat_step(initial.space, optical_frame.plasma.xy, optical_frame.heating, initial.chart, state, h);
    if (trial.valid && trial.error <= optical_frame.work.x) {
      var boundary = HeatSurface(trial.state, 1, true);
      var kind = 0u;
      if ((trial.state.position.x <= optical_frame.heating.x && trial.state.position.x < state.position.x)
        || (trial.state.position.x >= optical_frame.heating.y && trial.state.position.x > state.position.x)) { kind = 1u; }
      var equator = HeatSurface(state, 0, false);
      let crossing = state.position.w != 0 && state.position.w * trial.state.position.w <= 0;
      if (crossing && optical_frame.material.x == 0) {
        equator = heat_surface(initial.space, initial.chart, state, trial.state, h, true, 0);
        if (!equator.valid) { return HeatSegment(initial, 0, attempt + 1, crossings, transport); }
      } else if (state.position.w == 0 && state.momentum.w == 0) {
        for (var i = 0u; i < 2u; i++) {
          let edge = select(optical_frame.space.z, optical_frame.space.w, i == 1u);
          if (state.position.x != edge && (state.position.x - edge) * (trial.state.position.x - edge) <= 0) {
            let candidate = heat_surface(initial.space, initial.chart, state, trial.state, h, false, edge);
            if (!candidate.valid) { return HeatSegment(initial, 0, attempt + 1, crossings, transport); }
            if (!equator.valid || candidate.fraction < equator.fraction) { equator = candidate; }
          }
        }
      }
      if (optical_frame.material.x == 0 && equator.valid && equator.fraction <= boundary.fraction && equator.state.position.x >= optical_frame.space.z && equator.state.position.x <= optical_frame.space.w) {
        boundary = equator;
        kind = 2u;
      }
      if (crossing && state.position.w * boundary.state.position.w <= 0) { crossings++; }
      {
        let midpoint = heat_step(initial.space, optical_frame.plasma.xy, optical_frame.heating, initial.chart, state, h * boundary.fraction / 2);
        if (!midpoint.valid) { return HeatSegment(initial, 0, attempt + 1, crossings, transport); }
        let sample = heat_orbit(initial.space, optical_frame.plasma.xy, optical_frame.heating, initial.chart, midpoint.state);
        let r = midpoint.state.position.x;
        let sigma = r * r + initial.space.x * initial.space.x * midpoint.state.position.w * midpoint.state.position.w;
        transport = transfer_cell(transport, sample, heat_tangent(initial.space, initial.chart, midpoint.state), abs(h) * boundary.fraction * sigma, block, screen);
        if (!transport.valid) { return HeatSegment(initial, 0, attempt + 1, crossings, transport); }
        if (transport.transmission < 0.0001) { kind = 2u; }
      }
      state = boundary.state;
      if (kind != 0u || recording) {
        var path = heat_orbit(initial.space, optical_frame.plasma.xy, optical_frame.heating, initial.chart, state);
        if (kind == 1u) { path.plasma = optical_frame.plasma.xy; }
        if (recording) { record_ray(path, block); }
        if (kind != 0u) { return HeatSegment(path, kind, attempt + 1, crossings, transport); }
      }
    }
    if (!trial.valid) { h *= 0.25; }
    else if (trial.error == 0) { h *= 2; }
    else { h *= clamp(0.9 * pow(optical_frame.work.x / trial.error, 1.0 / 3.0), 0.1, 2.0); }
    if (abs(h) < 1e-38) { break; }
  }
  return HeatSegment(initial, 0, budget, crossings, transport);
}

fn trace_ray(initial: KerrOrbit, recording: bool, screen: PolarizationScreen) -> RayResult {
  var path = initial;
  var block = optical_frame.block.xyz;
  var crossings = 0u;
  var transport = empty_transfer();
  var h = -0.01 / max(1, length(path.state.angular));
  var medium_entry = false;
  if (recording) { record_ray(path, block); }
  var source_radius = optical_frame.space.z;
  if (optical_frame.jet_spectrum.w > 0) { source_radius = min(source_radius, optical_frame.jet.x); }
  if (optical_frame.heating.z > 0) { source_radius = min(source_radius, optical_frame.heating.x); }
  let generator = source_free_horizon_generator(path, source_radius);
  let source_free = generator || source_free_radial_band(path, source_radius);
  if (generator || (source_free && !recording)) { return RayResult(path, block, 4, 0, 0, transport); }
  for (var attempt = 0u; attempt < u32(optical_frame.work.y); attempt++) {
    path = kerr_condition(path);
    // Schwarzschild R=E²r⁴+K r(2-r)>0 inside the horizon: an inward past ray cannot turn.
    // Once below all sources, causal termination avoids numerically reflecting at the singularity.
    if (all(path.space == vec2f(0)) && path.constants.x != 0) {
      var radius = path.state.radial.x;
      var inward = path.state.radial.y > 0;
      if (path.inverse) { radius = 1 / radius; inward = path.state.radial.y < 0; }
      if (radius > 0 && radius < min(2.0, source_radius) && inward) {
        return RayResult(path, block, 3, 1, crossings, transport);
      }
    }
    if (optical_frame.heating.z > 0) {
      var radius = path.state.radial.x;
      var velocity = path.state.radial.y;
      if (path.inverse) { radius = 1 / radius; velocity = -velocity * radius * radius; }
      let enters = (radius > optical_frame.heating.x && radius < optical_frame.heating.y)
        || (radius == optical_frame.heating.x && velocity < 0)
        || (radius == optical_frame.heating.y && velocity > 0);
      if (enters || medium_entry) {
        medium_entry = false;
        let heated = trace_heat_segment(path, u32(optical_frame.work.y) - attempt, recording, block, transport, screen);
        crossings += heated.crossings;
        path = heated.path;
        transport = heated.transfer;
        if (heated.kind == 0u) { return RayResult(path, block, 0, 0, crossings, transport); }
        if (heated.kind == 2u) { return RayResult(path, block, 2, 0, crossings, transport); }
        attempt += heated.work - 1u;
        h = -0.01 / max(1, length(path.state.angular));
        continue;
      }
    }
    var trial = kerr_step(path, h);
    if (trial.valid && trial.error <= optical_frame.work.x && path.state.radial.y != 0 && sign(path.state.radial.y) != sign(trial.state.radial.y)) {
      let endpoint = trial.state;
      let start_rate = kerr_derivative(path, path.state).state;
      let end_rate = kerr_derivative(path, endpoint).state;
      var left = 0.0;
      var right = 1.0;
      for (var iteration = 0u; iteration < 20u; iteration++) {
        let middle = (left + right) / 2;
        if (middle == left || middle == right) { break; }
        let candidate = kerr_interpolate(path.state, endpoint, start_rate, end_rate, h, middle);
        if (sign(path.state.radial.y) == sign(candidate.radial.y)) { left = middle; }
        else { right = middle; trial.state = candidate; }
      }
      h *= right;
    }
    if (trial.valid && trial.error <= optical_frame.work.x) {
      var boundary = KerrSurface(trial.state, 1, true);
      var kind = 0u;
      let radial_crossing = path.state.radial.x * trial.state.radial.x <= 0;
      let coplanar = path.state.direction.z == 0 && all(path.state.angular.xy == vec2f(0));
      if (radial_crossing && (path.inverse || path.space.x == 0 || coplanar)) {
        boundary = kerr_surface(path, trial.state, h, false, 0);
        if (!boundary.valid) { return RayResult(path, block, 0, 0, crossings, transport); }
        kind = select(3u, 1u, path.inverse);
      }
      if (optical_frame.heating.z > 0) {
        for (var i = 0u; i < 2u; i++) {
          var edge = select(optical_frame.heating.x, optical_frame.heating.y, i == 1u);
          if (path.inverse) { edge = 1 / edge; }
          if (path.state.radial.x != edge && (path.state.radial.x - edge) * (trial.state.radial.x - edge) <= 0) {
            let candidate = kerr_surface(path, trial.state, h, false, edge);
            if (!candidate.valid) { return RayResult(path, block, 0, 0, crossings, transport); }
            if (candidate.fraction < boundary.fraction) { boundary = candidate; kind = 5u; }
          }
        }
      }
      var equator = KerrSurface(path.state, 0, false);
      if (optical_frame.material.x == 0 && path.state.direction.z != 0 && path.state.direction.z * trial.state.direction.z <= 0) {
        equator = kerr_surface(path, trial.state, h, true, 0);
        if (!equator.valid) { return RayResult(path, block, 0, 0, crossings, transport); }
      } else if (coplanar) {
        for (var i = 0u; i < 2u; i++) {
          var boundary_coordinate = select(optical_frame.space.z, optical_frame.space.w, i == 1u);
          if (path.inverse) { boundary_coordinate = 1 / boundary_coordinate; }
          if (path.state.radial.x != boundary_coordinate && (path.state.radial.x - boundary_coordinate) * (trial.state.radial.x - boundary_coordinate) <= 0) {
            let candidate = kerr_surface(path, trial.state, h, false, boundary_coordinate);
            if (!candidate.valid) { return RayResult(path, block, 0, 0, crossings, transport); }
            if (!equator.valid || candidate.fraction < equator.fraction) { equator = candidate; }
          }
        }
      }
      if (optical_frame.material.x == 0 && equator.valid && equator.fraction <= boundary.fraction) {
        var radius = equator.state.radial.x;
        if (path.inverse && radius != 0) { radius = 1 / radius; }
        if (radius >= optical_frame.space.z && radius <= optical_frame.space.w) {
          boundary = equator;
          kind = 2u;
        }
      }
      {
        transport = transfer_segment(transport, path, boundary.state, h * boundary.fraction, block, screen);
        if (!transport.valid) { return RayResult(path, block, 0, 0, crossings, transport); }
        if (transport.transmission < 0.0001) { kind = 2u; }
      }
      if (path.state.direction.z != 0 && path.state.direction.z * boundary.state.direction.z <= 0) { crossings++; }
      if (kind == 5u) {
        medium_entry = true;
        block = kerr_continue_block(path, boundary.state, h * boundary.fraction, block);
        path.state = boundary.state;
        if (recording) { record_ray(path, block); }
        continue;
      }
      if (kind != 0u) {
        if (!boundary.valid) { break; }
        block = kerr_continue_block(path, boundary.state, h * boundary.fraction, block);
        let end_sign = sign(path.state.radial.x);
        path.state = boundary.state;
        if (kind == 1u || kind == 3u) { path.state.radial.x = 0; }
        if (recording) { record_ray(path, block); }
        return RayResult(path, block, kind, end_sign, crossings, transport);
      }
      let bifurcation_turn = path.bifurcation.z != 0 && path.state.radial.y != 0 && sign(path.state.radial.y) != sign(trial.state.radial.y);
      block = kerr_continue_block(path, trial.state, h, block);
      path.state = trial.state;
      if (bifurcation_turn) { path = kerr_bifurcation_flip(path); }
      if (recording) { record_ray(path, block); }
    }
    if (!trial.valid) { h *= 0.25; }
    else if (trial.error == 0) { h *= 2; }
    else { h *= clamp(0.9 * pow(optical_frame.work.x / trial.error, 1.0 / 3.0), 0.1, 2.0); }
    if (abs(h) < 1e-38) { break; }
  }
  return RayResult(path, block, select(0u, 4u, source_free), 0, crossings, transport);
}
