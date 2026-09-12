/**
 * Reduced state: (active radial coordinate, future Mino rate, null time/log amplitude, τ).
 * Direction n and canonical vector J avoid polar-coordinate denominators.
 * The fourth radial lane accumulates proper time only for massive motion.
 */
struct KerrOrbitState {
  radial: vec4f,
  direction: vec3f,
  angular: vec3f,
}

/**
 * Separated flow with independent radial and spacetime chart choices.
 * Constants pack (E,L,C,mass²); space packs (a,q); plasma packs (A_p,r₀²).
 * Bifurcation packs (selected root, other root, κ_h, Ω_h); κ_h = 0 disables the patch.
 */
struct KerrOrbit {
  state: KerrOrbitState,
  constants: vec4f,
  space: vec2f,
  chart: f32,
  inverse: bool,
  plasma: vec2f,
  bifurcation: vec4f,
}

struct KerrDerivative {
  state: KerrOrbitState,
  valid: bool,
}

fn kerr_orbit(g: KerrGeometry, charge: f32, p: vec4f, mass_squared: f32) -> KerrOrbit {
  let motion = kerr_motion(g, p, mass_squared);
  var radial = vec4f(g.radius, motion.radial, 0, 0);
  let inverse = abs(g.radius) > 1;
  if (inverse) { radial = vec4f(1 / g.radius, -motion.radial / (g.radius * g.radius), 0, 0); }
  return KerrOrbit(KerrOrbitState(radial, g.direction, motion.angular), motion.constants,
    vec2f(g.spin, charge), g.chart, inverse, vec2f(0, 1), vec4f(0));
}

fn kerr_derivative(path: KerrOrbit, state: KerrOrbitState) -> KerrDerivative {
  let e = path.constants.x;
  let l = path.constants.y;
  let c = path.constants.z;
  let m = path.constants.w;
  let a = path.space.x;
  let q = path.space.y;
  let x = state.radial.x;
  let v = state.radial.y;
  let k = (l - a * e) * (l - a * e) + c;
  let b = a * (a * e - l);
  let squared = a * a + q * q;
  let quadratic = 2 * e * b - k - m * squared;
  let central = b * b - squared * k;
  var acceleration: f32;
  var drag: f32;
  var null_time: f32;
  var proper = 0.0;
  let invalid = KerrDerivative(KerrOrbitState(), false);
  if (path.bifurcation.z != 0) {
    let chart_data = path.bifurcation;
    let profile_denominator = x * x + path.plasma.y;
    let fr = path.plasma.x * x * x / profile_denominator;
    let profile_derivative = 2 * path.plasma.x * x * path.plasma.y / (profile_denominator * profile_denominator);
    let radial_mass = k + m * x * x + fr;
    if (!(radial_mass > 0) || x == chart_data.y) { return invalid; }
    let mass_derivative = 2 * m * x + profile_derivative;
    let delta = (x - chart_data.x) * (x - chart_data.y);
    let acceleration = -(x - 1) * radial_mass - delta * mass_derivative / 2;
    let rotation = -chart_data.w * path.chart * (x + chart_data.x) * v / (x - chart_data.y);
    let amplitude_rate = -v * ((1 - chart_data.z * (x + chart_data.x)) / (x - chart_data.y) + mass_derivative / (2 * radial_mass));
    let n = state.direction;
    let axial = vec3f(0, 0, 1);
    return KerrDerivative(KerrOrbitState(
      vec4f(v, acceleration, amplitude_rate, select(0.0, x * x + a * a * n.z * n.z, m != 0)),
      cross(state.angular, n) + rotation * cross(axial, n),
      -a * a * m * n.z * cross(n, axial) + rotation * cross(axial, state.angular)), true);
  }
  if (path.inverse) {
    let denominator = e + b * x * x + path.chart * v;
    if (denominator == 0 || (m != 0 && x == 0)) { return invalid; }
    acceleration = m + quadratic * x + 3 * k * x * x + 2 * central * x * x * x;
    let profile_denominator = 1 + path.plasma.y * x * x;
    let fr = path.plasma.x / profile_denominator;
    let delta = 1 - 2 * x + squared * x * x;
    acceleration -= x * (1 - 3 * x + 2 * squared * x * x) * fr
      - path.plasma.x * path.plasma.y * x * x * x * delta / (profile_denominator * profile_denominator);
    drag = ((k + fr) * x * x + m) / denominator;
    var mass_term = 0.0;
    if (m != 0) { mass_term = m / (x * x); proper = 1 / (x * x) + a * a * state.direction.z * state.direction.z; }
    null_time = (1 + a * a * x * x) * (k + fr + mass_term) / denominator;
  } else {
    let p = e * x * x + b;
    let denominator = p - path.chart * v;
    let delta = x * x - 2 * x + squared;
    if (denominator == 0 && delta == 0) { return invalid; }
    acceleration = 2 * (e * e - m) * x * x * x + 3 * m * x * x + quadratic * x + k;
    let profile_denominator = x * x + path.plasma.y;
    let fr = path.plasma.x * x * x / profile_denominator;
    acceleration -= (x - 1) * fr + delta * path.plasma.x * x * path.plasma.y / (profile_denominator * profile_denominator);
    if (denominator == 0) { drag = (p + path.chart * v) / delta; }
    else { drag = (k + fr + m * x * x) / denominator; }
    null_time = (x * x + a * a) * drag;
    if (m != 0) { proper = x * x + a * a * state.direction.z * state.direction.z; }
  }
  null_time += a * (l - a * e * (1 - state.direction.z * state.direction.z));
  let rotation = a * (drag - e);
  let z = vec3f(0, 0, 1);
  let radial = vec4f(v, acceleration, null_time, proper);
  let direction = cross(state.angular, state.direction) + rotation * cross(z, state.direction);
  let angular = a * a * (e * e - m) * state.direction.z * cross(state.direction, z)
    + rotation * cross(z, state.angular);
  return KerrDerivative(KerrOrbitState(radial, direction, angular),
    all(abs(radial) < vec4f(3.4e38)) && all(abs(direction) < vec3f(3.4e38)) && all(abs(angular) < vec3f(3.4e38)));
}

fn kerr_offset(y: KerrOrbitState, d: KerrOrbitState, h: f32) -> KerrOrbitState {
  return KerrOrbitState(y.radial + h * d.radial, y.direction + h * d.direction, y.angular + h * d.angular);
}

/**
 * Hermite interpolation over an accepted step using its endpoint derivatives.
 * Requires fraction t in [0,1]; n is normalized after interpolation. Reuses the
 * same geometric segment for events and material instead of solving a new ray.
 */
fn kerr_interpolate(
  start: KerrOrbitState,
  end: KerrOrbitState,
  start_rate: KerrOrbitState,
  end_rate: KerrOrbitState,
  h: f32,
  t: f32
) -> KerrOrbitState {
  let h00 = (1 + 2 * t) * (1 - t) * (1 - t);
  let h10 = t * (1 - t) * (1 - t);
  let h01 = t * t * (3 - 2 * t);
  let h11 = t * t * (t - 1);
  return KerrOrbitState(
    h00 * start.radial + h01 * end.radial + h * (h10 * start_rate.radial + h11 * end_rate.radial),
    normalize(h00 * start.direction + h01 * end.direction + h * (h10 * start_rate.direction + h11 * end_rate.direction)),
    h00 * start.angular + h01 * end.angular + h * (h10 * start_rate.angular + h11 * end_rate.angular));
}

fn kerr_rotate(v: vec3f, angle: f32) -> vec3f {
  let c = cos(angle);
  let s = sin(angle);
  return vec3f(c * v.x - s * v.y, s * v.x + c * v.y, v.z);
}

/**
 * Return (a I, r*, valid), whose radial derivatives are a/Δ and (r²+a²)/Δ.
 * The final lane is zero at a horizon pole; transition only in a regular overlap.
 */
fn kerr_chart_primitives(space: vec2f, radius: f32) -> vec3f {
  let a = space.x;
  let d = 1 - a * a - space.y * space.y;
  let shifted = radius - 1;
  let delta = shifted * shifted - d;
  if (delta == 0) { return vec3f(0); }
  var integral: f32;
  if (d > 0) {
    let root = sqrt(d);
    integral = (log(abs(shifted - root)) - log(abs(shifted + root))) / (2 * root);
  } else if (d == 0) { integral = -1 / shifted; }
  else { let root = sqrt(-d); integral = (atan2(shifted, root) - 1.570796326794897) / root; }
  return vec3f(a * integral, radius + log(abs(delta)) + (2 - space.y * space.y) * integral, 1);
}

/**
 * Recover ordinary null-chart state from a bifurcation patch away from its sphere.
 * Requires nonzero radial velocity; exact bifurcation placement has no finite w here.
 */
fn kerr_regular(input: KerrOrbit) -> KerrOrbit {
  var path = input;
  let chart_data = path.bifurcation;
  if (chart_data.z == 0 || path.state.radial.y == 0) { return path; }
  let time = (path.state.radial.z + log(abs(path.state.radial.y))) / (path.chart * chart_data.z);
  path.state.direction = kerr_rotate(path.state.direction, chart_data.w * time);
  path.state.angular = kerr_rotate(path.state.angular, chart_data.w * time);
  path.state.radial.z = time;
  path.bifurcation = vec4f(0);
  return path;
}

/** Finite chart transition at a zero-energy radial turn on the bifurcation sphere. */
fn kerr_bifurcation_flip(input: KerrOrbit) -> KerrOrbit {
  var path = input;
  let chart_data = path.bifurcation;
  let r = path.state.radial.x;
  let a = path.space.x;
  let difference = path.constants.y - a * path.constants.x;
  let fr = path.plasma.x * r * r / (r * r + path.plasma.y);
  let mass = difference * difference + path.constants.z + path.constants.w * r * r + fr;
  let logarithm = log(abs(r - chart_data.y));
  let horizon_scale = chart_data.x * chart_data.x + a * a;
  let other_scale = chart_data.y * chart_data.y + a * a;
  path.state.radial.z = -path.state.radial.z + 2 * chart_data.z * r - (1 + other_scale / horizon_scale) * logarithm - log(mass);
  let angle = 2 * path.chart * chart_data.w * (r + 2 * logarithm);
  path.state.direction = kerr_rotate(path.state.direction, angle);
  path.state.angular = kerr_rotate(path.state.angular, angle);
  path.chart = -path.chart;
  return path;
}

/** Switch coordinate overlaps before their poles. Infinity is not a chart overlap. */
fn kerr_condition(input: KerrOrbit) -> KerrOrbit {
  var path = input;
  var x = path.state.radial.x;
  let squared = dot(path.space, path.space);
  if (path.constants.x == 0 && path.space.x * path.constants.y == 0 && squared < 1 && x != 0) {
    var r = x;
    if (path.inverse) { r = 1 / x; }
    let outer = 1 + sqrt(1 - squared);
    let inner = squared / outer;
    let is_outer = r >= 1 || inner == 0;
    let radius = select(inner, outer, is_outer);
    let other = select(outer, inner, is_outer);
    if (path.bifurcation.z != 0 && path.bifurcation.x == radius) { return path; }
    path = kerr_regular(path);
    var velocity = path.state.radial.y;
    if (path.inverse) { velocity = -velocity * r * r; }
    if (velocity != 0) {
      let scale = radius * radius + path.space.x * path.space.x;
      let kappa = (radius - 1) / scale;
      let omega = path.space.x / scale;
      let time = path.state.radial.z;
      path.state.direction = kerr_rotate(path.state.direction, -omega * time);
      path.state.angular = kerr_rotate(path.state.angular, -omega * time);
      path.state.radial = vec4f(r, velocity, path.chart * kappa * time - log(abs(velocity)), path.state.radial.w);
      path.bifurcation = vec4f(radius, other, kappa, omega);
      path.inverse = false;
      return path;
    }
  }
  if (abs(x) > 1) {
    path.state.radial.x = 1 / x;
    path.state.radial.y = -path.state.radial.y / (x * x);
    path.inverse = !path.inverse;
    x = path.state.radial.x;
  }
  let a = path.space.x;
  let b = a * (a * path.constants.x - path.constants.y);
  var p = path.constants.x * x * x + b;
  var v = path.state.radial.y;
  if (path.inverse) { p = path.constants.x + b * x * x; v = -v; }
  if (path.chart * p * v <= 0) { return path; }
  if (path.inverse && x == 0) { return path; }
  var radius = x;
  if (path.inverse) { radius = 1 / x; }
  let primitive = kerr_chart_primitives(path.space, radius);
  if (primitive.z == 0) { return path; }
  let angle = -2 * path.chart * primitive.x;
  path.state.direction = kerr_rotate(path.state.direction, angle);
  path.state.angular = kerr_rotate(path.state.angular, angle);
  path.state.radial.z -= 2 * path.chart * primitive.y;
  path.chart = -path.chart;
  return path;
}

/**
 * Recover the future affine tangent in Cartesian KS components at a regular finite point.
 * The caller must exclude reciprocal infinity, exact bifurcation points, and invalid derivatives.
 */
fn kerr_tangent(input: KerrOrbit) -> vec4f {
  let path = kerr_regular(input);
  var r = path.state.radial.x;
  var v = path.state.radial.y;
  if (path.inverse) { r = 1 / r; v = -v * r * r; }
  let derivative = kerr_derivative(path, path.state).state;
  let sigma = r * r + path.space.x * path.space.x * path.state.direction.z * path.state.direction.z;
  let spatial = v * path.state.direction + r * derivative.direction +
    path.chart * path.space.x * cross(vec3f(0, 0, 1), derivative.direction);
  return vec4f(derivative.radial.z - path.chart * v, spatial) / sigma;
}

/**
 * Update one valid adjoining block: lanes are (kind, universe, stationary side).
 * Kinds 0–5 are exterior, black hole, interior, white hole, naked, disconnected.
 * radial_sign is future-oriented even when the numerical step traces backward.
 */
fn kerr_cross_block(block: vec3i, outer: bool, radial_sign: f32, side: i32, extremal: bool) -> vec3i {
  if (extremal) {
    if (block.x == 0) { return vec3i(2, block.y - i32(radial_sign > 0), side); }
    return vec3i(0, block.y + i32(radial_sign > 0), side);
  }
  if (outer) {
    if (block.x == 0) { return vec3i(select(3, 1, radial_sign < 0), block.y, 0); }
    return vec3i(0, block.y, side);
  }
  if (block.x == 1) { return vec3i(2, block.y, side); }
  if (block.x == 3) { return vec3i(2, block.y - 1, side); }
  return vec3i(select(3, 1, radial_sign < 0), block.y + i32(radial_sign > 0), 0);
}

fn kerr_continue_block(path: KerrOrbit, end: KerrOrbitState, h: f32, initial: vec3i) -> vec3i {
  if (path.bifurcation.z != 0) {
    if (path.state.radial.y == 0 || sign(path.state.radial.y) == sign(end.radial.y)) { return initial; }
    let outer = path.bifurcation.x > path.bifurcation.y;
    let increment = select(select(-1, 1, initial.x == 1), 0, outer);
    return vec3i(select(1, 3, initial.x == 1), initial.y + increment, 0);
  }
  let squared = dot(path.space, path.space);
  if (squared > 1 || initial.x == 5) { return initial; }
  let outer = 1 + sqrt(1 - squared);
  let inner = squared / outer;
  let radial_sign = sign((end.radial.x - path.state.radial.x) * h * select(1.0, -1.0, path.inverse));
  let inward = (end.radial.x - path.state.radial.x) * select(1.0, -1.0, path.inverse) < 0;
  var block = initial;
  for (var i = 0u; i < 2u; i++) {
    let is_outer = (i == 0u) == inward;
    if (!is_outer && (inner == 0 || squared == 1)) { continue; }
    var radius = inner;
    if (is_outer) { radius = outer; }
    var coordinate = radius;
    if (path.inverse) { coordinate = 1 / radius; }
    if (path.state.radial.x == coordinate) {
      if (end.radial.x == coordinate) { continue; }
      let outside = (end.radial.x - coordinate) * select(1.0, -1.0, path.inverse) > 0;
      let stationary = select(!outside, outside, is_outer);
      let current_stationary = select(block.x == 2, block.x == 0, is_outer);
      if (stationary == current_stationary) { continue; }
    } else if (sign(path.state.radial.x - coordinate) == sign(end.radial.x - coordinate)) { continue; }
    let p = path.constants.x * (radius * radius + path.space.x * path.space.x) - path.space.x * path.constants.y;
    block = kerr_cross_block(block, is_outer, radial_sign, select(-1, 1, p >= 0), squared == 1);
  }
  return block;
}

struct KerrStep {
  state: KerrOrbitState,
  error: f32,
  valid: bool,
}

/** Radial first integral and its half derivative in the active polynomial coordinate. */
fn kerr_radial_shell(path: KerrOrbit, x: f32) -> vec2f {
  let e = path.constants.x;
  let a = path.space.x;
  let squared = dot(path.space, path.space);
  let b = a * (a * e - path.constants.y);
  let k = (path.constants.y - a * e) * (path.constants.y - a * e) + path.constants.z;
  let m = path.constants.w;
  var p = e * x * x + b;
  var delta = x * x - 2 * x + squared;
  var mass = k + m * x * x;
  var p_gradient = 2 * e * x;
  var delta_gradient = 2 * (x - 1);
  var mass_gradient = 2 * m * x;
  if (path.inverse) {
    p = e + b * x * x;
    delta = 1 - 2 * x + squared * x * x;
    mass = k * x * x + m;
    p_gradient = 2 * b * x;
    delta_gradient = 2 * (squared * x - 1);
    mass_gradient = 2 * k * x;
    if (path.plasma.x > 0) {
      let denominator = 1 + path.plasma.y * x * x;
      mass += path.plasma.x * x * x / denominator;
      mass_gradient += 2 * path.plasma.x * x / (denominator * denominator);
    }
  } else {
    if (path.plasma.x > 0) {
      let denominator = x * x + path.plasma.y;
      mass += path.plasma.x * x * x / denominator;
      mass_gradient += 2 * path.plasma.x * path.plasma.y * x / (denominator * denominator);
    }
  }
  if (squared <= 1) {
    // Use the same representable horizons as atlas transitions; subtraction of r²−2r+a²+q²
    // can otherwise change the sign of Delta immediately beside a horizon.
    let outer = 1 + sqrt(1 - squared);
    let inner = squared / outer;
    if (path.inverse) { delta = (1 - outer * x) * (1 - inner * x); }
    else { delta = (x - outer) * (x - inner); }
  }
  return vec2f(p * p - delta * mass, p * p_gradient - (delta_gradient * mass + delta * mass_gradient) / 2);
}

struct RadialProjection {
  state: KerrOrbitState,
  valid: bool,
}

/**
 * Project the radial endpoint onto its conserved first integral. Newton corrections follow the
 * scaled normal in (coordinate, velocity), so velocity carries the correction on radial legs
 * while coordinate carries it at a simple turn. An uncorrectable gradient rejects the trial;
 * exhausting subsequent work remains unresolved.
 */
fn kerr_project_radial(path: KerrOrbit, input: KerrOrbitState) -> RadialProjection {
  if (path.bifurcation.z != 0) { return RadialProjection(input, true); }
  var state = input;
  let scale = 1 + abs(input.radial.xy);
  for (var iteration = 0u; iteration < 3u; iteration++) {
    let shell = kerr_radial_shell(path, state.radial.x);
    let residual = shell.x - state.radial.y * state.radial.y;
    if (residual == 0) { break; }
    let gradient = 2 * vec2f(shell.y, -state.radial.y) * scale;
    if (!all(abs(gradient) < vec2f(3.4e38)) || !(abs(residual) < 3.4e38)) { return RadialProjection(input, false); }
    let magnitude = max(abs(gradient.x), abs(gradient.y));
    if (!(magnitude > 0)) { return RadialProjection(input, false); }
    let normalized = gradient / magnitude;
    let change = (residual / magnitude) / dot(normalized, normalized) * normalized * scale;
    let next = state.radial.xy - change;
    if (!all(abs(next) < vec2f(3.4e38))) { return RadialProjection(input, false); }
    if (all(next == state.radial.xy)) { break; }
    state.radial = vec4f(next, state.radial.zw);
  }
  return RadialProjection(state, true);
}

/** Bogacki–Shampine 3(2); four evaluations and an explicit local error estimate. */
fn kerr_step(path: KerrOrbit, h: f32) -> KerrStep {
  let invalid = KerrStep(path.state, 0, false);
  let k1 = kerr_derivative(path, path.state);
  if (!k1.valid) { return invalid; }
  let k2 = kerr_derivative(path, kerr_offset(path.state, k1.state, h / 2));
  if (!k2.valid) { return invalid; }
  let k3 = kerr_derivative(path, kerr_offset(path.state, k2.state, 3 * h / 4));
  if (!k3.valid) { return invalid; }
  let unprojected = kerr_offset(kerr_offset(kerr_offset(path.state, k1.state, 2 * h / 9), k2.state, h / 3), k3.state, 4 * h / 9);
  let projection = kerr_project_radial(path, unprojected);
  if (!projection.valid) { return invalid; }
  let state = projection.state;
  let k4 = kerr_derivative(path, state);
  if (!k4.valid) { return invalid; }
  let difference = kerr_offset(kerr_offset(kerr_offset(kerr_offset(
    KerrOrbitState(), k1.state, -5 * h / 72),
    k2.state, h / 12), k3.state, h / 9), k4.state, -h / 8);
  let r = max(abs(difference.radial), abs(state.radial - unprojected.radial)) / (1 + abs(state.radial));
  let n = abs(difference.direction);
  let j = abs(difference.angular) / (1 + abs(state.angular));
  let error = max(max(max(r.x, r.y), max(r.z, r.w)), max(max(n.x, max(n.y, n.z)), max(j.x, max(j.y, j.z))));
  return KerrStep(state, error, true);
}

/** Multiplicative controller for a 3(2) trial; invalid trials never evaluate an error ratio. */
fn kerr_step_scale(valid: bool, error: f32, tolerance: f32) -> f32 {
  if (!valid) { return 0.25; }
  if (error == 0) { return 2; }
  return clamp(0.9 * pow(tolerance / error, 1.0 / 3.0), 0.1, 2.0);
}
