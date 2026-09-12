/**
 * Metric data and Cartesian spatial gradients in the exterior heated annulus.
 * Columns of direction_gradient are ∇n_x, ∇n_y, ∇n_z; transpose it to push a tangent.
 */
struct MediumGeometry {
  frame: KerrGeometry,
  radial_gradient: vec3f,
  direction_gradient: mat3x3f,
  sigma_gradient: vec3f,
}

fn medium_geometry(space: vec2f, radius: f32, direction: vec3f, chart: f32) -> MediumGeometry {
  let n = normalize(direction);
  let a = space.x;
  let sigma = radius * radius + a * a * n.z * n.z;
  let position = radius * n + chart * a * cross(vec3f(0, 0, 1), n);
  let cylindrical = length(n.xy);
  var polar = vec3f(n.z, 0, 0);
  var azimuthal = vec3f(0, 1, 0);
  if (cylindrical > 0) {
    polar = vec3f(n.z * n.xy / cylindrical, -cylindrical);
    azimuthal = vec3f(-n.y, n.x, 0) / cylindrical;
  }
  let g = KerrGeometry(position, n, polar, azimuthal, radius, a, sigma,
    (2 * radius - space.y * space.y) / sigma, chart);
  let radial_gradient = vec3f(radius / sigma * position.xy, (radius * radius + a * a) / sigma * n.z);
  let denominator = radius * radius + a * a;
  let nx = vec3f(radius, chart * a, 0) / denominator - (radius * n.x + chart * a * n.y) / denominator * radial_gradient;
  let ny = vec3f(-chart * a, radius, 0) / denominator - (-chart * a * n.x + radius * n.y) / denominator * radial_gradient;
  let nz = vec3f(-n.z * position.xy / sigma, radius * (1 - n.z * n.z) / sigma);
  return MediumGeometry(g, radial_gradient, mat3x3f(nx, ny, nz), 2 * radius * radial_gradient + 2 * a * a * n.z * nz);
}

/**
 * Detector-normalized χ = ν_p²/ν_o² and its covector gradient in (T,X,Y,Z).
 */
struct MediumPotential {
  cutoff: f32,
  gradient: vec4f,
}

/**
 * Evaluate prescribed constant-pressure density reduction and its full spacetime gradient.
 * The input time is selected null time w; the result differentiates with respect to KS T.
 * The sixfold pattern uses canonical ingoing coordinates in either computational chart.
 */
fn medium_potential(
  space: vec2f,
  geometry: MediumGeometry,
  profile: vec2f,
  heating: vec4f,
  time: f32
) -> MediumPotential {
  let g = geometry.frame;
  let r = g.radius;
  let r2 = r * r;
  let denominator = r2 + profile.y;
  let fraction = r2 / denominator;
  let fr = profile.x * fraction;
  let fr_derivative = 2 * (profile.x / r) * fraction * (profile.y / denominator);
  let baseline = fr / g.sigma;
  let baseline_gradient = fr_derivative / g.sigma * geometry.radial_gradient - baseline / g.sigma * geometry.sigma_gradient;
  if (heating.z == 0 || r <= heating.x || r >= heating.y) {
    return MediumPotential(baseline, vec4f(0, baseline_gradient));
  }
  let primitive = kerr_chart_primitives(space, r);
  let angle = (1 - g.chart) * primitive.x;
  let n = kerr_rotate(g.direction, angle);
  let ingoing_time = time + (1 - g.chart) * primitive.y;
  let width = heating.y - heating.x;
  let x = (r - heating.x) / width;
  let product = x * (1 - x);
  let window = 64 * product * product * product;
  let window_derivative = 192 * product * product * (1 - 2 * x) / width;
  let square = vec2f(n.x * n.x - n.y * n.y, 2 * n.x * n.y);
  let fourth = vec2f(square.x * square.x - square.y * square.y, 2 * square.x * square.y);
  let harmonic = vec2f(dot(fourth, vec2f(n.x, -n.y)), dot(fourth, n.yx));
  let sixth = vec2f(dot(fourth, vec2f(square.x, -square.y)), dot(fourth, square.yx));
  let phase = heating.w * ingoing_time - 6.283185307179586 * x;
  let c = cos(phase);
  let s = sin(phase);
  let wave = 0.5 + 0.5 * dot(sixth, vec2f(c, s));
  let phase_derivative = 0.5 * dot(sixth, vec2f(-s, c));
  let ratio = 1 + heating.z * window * wave;
  let time_derivative = heating.z * window * phase_derivative * heating.w;
  let direction_derivative = vec3f(3 * heating.z * window * vec2f(harmonic.x * c + harmonic.y * s, -harmonic.y * c + harmonic.x * s), 0);
  let local_derivative = kerr_rotate(direction_derivative, -angle);
  let delta = r2 - 2 * r + dot(space, space);
  let radial_derivative = heating.z * (window_derivative * wave - window * phase_derivative * 6.283185307179586 / width)
    + (1 - g.chart) * space.x / delta * dot(direction_derivative, cross(vec3f(0, 0, 1), n))
    + time_derivative * (g.chart + (1 - g.chart) * (r2 + space.x * space.x) / delta);
  let ratio_gradient = geometry.direction_gradient * local_derivative + geometry.radial_gradient * radial_derivative;
  return MediumPotential(baseline / ratio, vec4f(-baseline * time_derivative / (ratio * ratio),
    baseline_gradient / ratio - baseline / (ratio * ratio) * ratio_gradient));
}

/**
 * Detector-normalized canonical state in the heated annulus.
 * Position packs signed r and angular direction n; momentum is the KS covector p_μ.
 * Time stores the selected null coordinate w, not Cartesian T.
 */
struct HeatState {
  position: vec4f,
  momentum: vec4f,
  time: f32,
}
struct HeatRate {
  state: HeatState,
  valid: bool,
}

/** Conservative exponent bound leaves room for vector sums before any unsafe product is evaluated. */
fn heat_product_safe(a: f32, b: f32) -> bool {
  let ae = extractBits(bitcast<u32>(a), 23u, 8u);
  let be = extractBits(bitcast<u32>(b), 23u, 8u);
  return ae < 255u && be < 255u && ae + be <= 376u;
}
fn heat_maximum(value: vec4f) -> f32 {
  return max(max(abs(value.x), abs(value.y)), max(abs(value.z), abs(value.w)));
}

/**
 * Hamiltonian flow for H = (g^μν p_μ p_ν + χ)/2, with χ = ν_p²/ν_o².
 * Uses dλ = Σ dγ for detector-normalized affine λ; p_T and p_φ may evolve.
 * A false validity flag means unsupported arithmetic/domain, never physical termination.
 */
fn heat_derivative(space: vec2f, profile: vec2f, heating: vec4f, chart: f32, state: HeatState) -> HeatRate {
  let invalid = HeatRate(HeatState(), false);
  let radius = state.position.x;
  let norm = dot(state.position.yzw, state.position.yzw);
  if (!(radius > 2 && norm > 0 && norm < 4)) { return invalid; }
  let geometry = medium_geometry(space, radius, state.position.yzw, chart);
  let g = geometry.frame;
  let p = state.momentum;
  if (!heat_product_safe(heat_maximum(p), max(1, abs(g.factor)))) { return invalid; }
  let principal = vec4f(-1, chart * g.direction);
  let contraction = dot(principal, p);
  if (!heat_product_safe(contraction, contraction)) { return invalid; }
  let tangent = vec4f(-p.x, p.yzw) - g.factor * contraction * principal;
  if (!heat_product_safe(heat_maximum(tangent), g.sigma)) { return invalid; }
  let material = medium_potential(space, geometry, profile, heating, state.time);
  let factor_gradient_scaled = 2 * geometry.radial_gradient - g.factor * geometry.sigma_gradient;
  let principal_gradient_scaled = chart * g.sigma * (geometry.direction_gradient * p.yzw);
  if (!heat_product_safe(contraction * contraction, heat_maximum(vec4f(factor_gradient_scaled, 0)))
    || !heat_product_safe(g.factor * contraction, heat_maximum(vec4f(principal_gradient_scaled, 0)))
    || !heat_product_safe(g.sigma, heat_maximum(material.gradient))) { return invalid; }
  let momentum_rate = vec4f(-0.5 * g.sigma * material.gradient.x,
    0.5 * contraction * contraction * factor_gradient_scaled + g.factor * contraction * principal_gradient_scaled - 0.5 * g.sigma * material.gradient.yzw);
  let radial_rate = g.sigma * dot(geometry.radial_gradient, tangent.yzw);
  let direction_rate = g.sigma * (transpose(geometry.direction_gradient) * tangent.yzw);
  return HeatRate(HeatState(vec4f(radial_rate, direction_rate), momentum_rate, g.sigma * tangent.x + chart * radial_rate), true);
}

fn heat_offset(state: HeatState, rate: HeatState, step: f32) -> HeatState {
  return HeatState(state.position + step * rate.position, state.momentum + step * rate.momentum, state.time + step * rate.time);
}
struct HeatStep {
  state: HeatState,
  error: f32,
  valid: bool,
}

/** Embedded Bogacki–Shampine step for the time-dependent medium, with canonical energy free to evolve. */
fn heat_step(space: vec2f, profile: vec2f, heating: vec4f, chart: f32, state: HeatState, h: f32) -> HeatStep {
  let invalid = HeatStep(state, 0, false);
  let k1 = heat_derivative(space, profile, heating, chart, state);
  if (!k1.valid) { return invalid; }
  let k2 = heat_derivative(space, profile, heating, chart, heat_offset(state, k1.state, h / 2));
  if (!k2.valid) { return invalid; }
  let k3 = heat_derivative(space, profile, heating, chart, heat_offset(state, k2.state, 3 * h / 4));
  if (!k3.valid) { return invalid; }
  var end = heat_offset(heat_offset(heat_offset(state, k1.state, 2 * h / 9), k2.state, h / 3), k3.state, 4 * h / 9);
  let k4 = heat_derivative(space, profile, heating, chart, end);
  if (!k4.valid) { return invalid; }
  let difference = heat_offset(heat_offset(heat_offset(heat_offset(HeatState(), k1.state, -5 * h / 72), k2.state, h / 12), k3.state, h / 9), k4.state, -h / 8);
  let position_error = abs(difference.position) / vec4f(1 + abs(end.position.x), 1, 1, 1);
  let momentum_error = abs(difference.momentum) / (1 + abs(end.momentum));
  let error = max(max(heat_maximum(position_error), heat_maximum(momentum_error)), max(abs(difference.time) / (1 + abs(end.time)), abs(difference.time) * heating.w));
  end.position = vec4f(end.position.x, normalize(end.position.yzw));
  return HeatStep(end, error, true);
}

/**
 * Hermite interpolation at fraction t in [0,1] of an accepted canonical step.
 * Renormalizes n afterward; this is an event-location approximation, not an exact
 * Hamiltonian trajectory or a guarantee of substep invariant preservation.
 */
fn heat_interpolate(
  start: HeatState,
  end: HeatState,
  first: HeatState,
  last: HeatState,
  h: f32,
  t: f32
) -> HeatState {
  let t2 = t * t;
  let t3 = t2 * t;
  let weights = vec4f(2 * t3 - 3 * t2 + 1, -2 * t3 + 3 * t2, h * (t3 - 2 * t2 + t), h * (t3 - t2));
  let position = weights.x * start.position + weights.y * end.position + weights.z * first.position + weights.w * last.position;
  return HeatState(vec4f(position.x, normalize(position.yzw)),
    weights.x * start.momentum + weights.y * end.momentum + weights.z * first.momentum + weights.w * last.momentum,
    dot(weights, vec4f(start.time, end.time, first.time, last.time)));
}

/** Physical canonical tangent, without reconstructing separated constants inside the heated field. */
fn heat_tangent(space: vec2f, chart: f32, state: HeatState) -> vec4f {
  let g = medium_geometry(space, state.position.x, state.position.yzw, chart).frame;
  return kerr_raise(g, state.momentum);
}

/**
 * Reconstruct pointwise separated data from a canonical state.
 * Used for local material evaluation and for continuation after leaving heating;
 * inside the nonseparable field these data are not global constants of motion.
 */
fn heat_orbit(space: vec2f, profile: vec2f, heating: vec4f, chart: f32, state: HeatState) -> KerrOrbit {
  let geometry = medium_geometry(space, state.position.x, state.position.yzw, chart);
  let g = geometry.frame;
  let tangent = kerr_raise(g, state.momentum);
  var path = kerr_orbit(g, space.y, tangent, 0);
  let material = medium_potential(space, geometry, profile, heating, state.time);
  path.plasma = vec2f(material.cutoff * g.sigma * (1 + profile.y / (g.radius * g.radius)), profile.y);
  path.state.radial.z = state.time;
  return path;
}
