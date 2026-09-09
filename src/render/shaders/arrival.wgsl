struct Arrival {
  lower: f32,
  upper: f32,
  // 0: different real component or infinite phase; 1: bracketed arrival; 2: unresolved arithmetic.
  status: u32,
}

/**
 * Refine a local sign-changing bracket. A coordinate residual alone is unreliable
 * when f32 trajectory evaluation cannot represent the boundary value exactly.
 */
fn refine_arrival(path: QuarticPath, seed: f32, endpoint: f32, velocity: f32) -> Arrival {
  if (abs(velocity) <= 1e-20 || seed <= 0.0) { return Arrival(0.0, 0.0, 2u); }
  // A bracket samples one fixed elliptic curve repeatedly. AGM ratios depend
  // only on its modulus, so prepare them once for the entire refinement.
  let plan = prepare_jacobi(path.elliptic.m);
  let increasing = velocity > 0.0;
  let sample = evaluate_quartic_prepared(path, seed, plan);
  if (!sample.valid) { return Arrival(0.0, 0.0, 2u); }
  var radius = max(4.0 * abs((sample.value - endpoint) / velocity), 8.0 * 1.1920928955e-7 * seed);
  var lower = 0.0;
  var upper = 0.0;
  var bracketed = false;
  for (var attempt = 0u; attempt < 6u; attempt++) {
    lower = max(0.0, seed - radius);
    upper = seed + radius;
    let before = evaluate_quartic_prepared(path, lower, plan);
    let after = evaluate_quartic_prepared(path, upper, plan);
    let starts_before = select(before.value >= endpoint, before.value <= endpoint, increasing);
    let ends_after = select(after.value <= endpoint, after.value >= endpoint, increasing);
    if (before.valid && after.valid && starts_before && ends_after) {
      bracketed = true;
      break;
    }
    radius *= 2.0;
  }
  if (!bracketed) { return Arrival(lower, upper, 2u); }
  for (var iteration = 0u; iteration < 12u; iteration++) {
    let middle = (lower + upper) / 2.0;
    if (upper - lower <= 4.0 * 1.1920928955e-7 * max(1.0, abs(middle))) { break; }
    let value = evaluate_quartic_prepared(path, middle, plan);
    if (!value.valid) { return Arrival(lower, upper, 2u); }
    if (select(value.value > endpoint, value.value < endpoint, increasing)) { lower = middle; }
    else { upper = middle; }
  }
  return Arrival(lower, upper, 1u);
}

/**
 * Abel addition identity, with the endpoint velocity selecting the phase branch.
 * See docs/numerics.md. This does not normalize the photon by Killing energy.
 */
fn quartic_arrival(path: QuarticPath, endpoint: f32, velocity: f32) -> Arrival {
  let difference = endpoint - path.x;
  if (difference == 0.0) { return Arrival(0.0, 0.0, 2u); }
  let d2 = difference * difference;
  let combined = path.f + path.velocity * velocity + path.first * difference / 2.0;
  let wp = combined / (2.0 * d2) + path.second / 24.0;
  let first = path.first + path.second * difference + path.third * d2 / 2.0
    + path.fourth * d2 * difference / 6.0;
  let derivative = ((path.velocity * first + path.first * velocity) * difference - 4.0 * combined * velocity)
    / (4.0 * d2 * difference);
  let p = path.elliptic;
  if (p.scale == 0.0) {
    if (wp > 0.0 && derivative < 0.0) { return refine_arrival(path, inverseSqrt(wp), endpoint, velocity); }
    return Arrival(0.0, 0.0, 0u);
  }
  let minimum = select(p.root, p.root + p.scale, p.three_real);
  // A local expansion supplies a seed when roundoff obscures the half-period minimum.
  // The seed is never accepted without a residual check against the original path.
  let uncertainty = 32.0 * 1.1920928955e-7 * (abs(combined / (2.0 * d2)) + abs(path.second / 24.0) + abs(minimum));
  if (abs(wp - minimum) <= uncertainty && p.m < 1.0) {
    let curvature = 6.0 * minimum * minimum - p.g2 / 2.0;
    let complete = carlson_rf(vec3f(0.0, 1.0 - p.m, 1.0));
    if (curvature <= 0.0 || !complete.valid) { return Arrival(0.0, 0.0, 2u); }
    return refine_arrival(path, complete.value / sqrt(p.scale) + derivative / curvature, endpoint, velocity);
  }
  var sine2 = 0.0;
  if (p.three_real) {
    let denominator = wp - p.root;
    if (denominator < p.scale) { return Arrival(0.0, 0.0, 0u); }
    sine2 = p.scale / denominator;
  } else {
    let w = wp - p.root;
    if (w < 0.0) { return Arrival(0.0, 0.0, 0u); }
    let scale = max(w, p.scale);
    let x = w / scale;
    let y = p.scale / scale;
    sine2 = 2.0 * y / (x + y + sqrt((x - y) * (x - y) + 4.0 * x * y * (1.0 - p.m)));
  }
  if (!(sine2 > 0.0 && sine2 <= 1.0)) { return Arrival(0.0, 0.0, 2u); }
  let complement = 1.0 - p.m * sine2;
  if (complement <= 0.0) { return Arrival(0.0, 0.0, 0u); }
  let incomplete = carlson_rf(vec3f(1.0 - sine2, complement, 1.0));
  if (!incomplete.valid) { return Arrival(0.0, 0.0, 2u); }
  let time = sqrt(sine2) * incomplete.value / sqrt(p.scale);
  if (derivative <= 0.0) { return refine_arrival(path, time, endpoint, velocity); }
  if (p.m == 1.0) { return Arrival(0.0, 0.0, 0u); }
  let complete = carlson_rf(vec3f(0.0, 1.0 - p.m, 1.0));
  if (!complete.valid) { return Arrival(0.0, 0.0, 2u); }
  return refine_arrival(path, 2.0 * complete.value / sqrt(p.scale) - time, endpoint, velocity);
}

/**
 * Check extrema of c.x + c.y*u² + c.z*u³ + c4*u⁴ inside a radial interval.
 * 0 is clear, 1 contains a forbidden barrier, and 2 is unresolved at a repeated root.
 */
fn radial_interval(c: vec3f, c4: f32, lower: f32, upper: f32) -> u32 {
  let a = 4.0 * c4;
  let b = 3.0 * c.z;
  let constant = 2.0 * c.y;
  var roots: array<f32, 2>;
  var count = 0u;
  if (a == 0.0) {
    if (b != 0.0) {
      roots[0] = -constant / b;
      count = 1u;
    }
  } else {
    let discriminant = b * b - 4.0 * a * constant;
    if (discriminant >= 0.0) {
      let root = sqrt(discriminant);
      let term = -0.5 * (b + select(-root, root, b >= 0.0));
      if (term != 0.0) {
        roots[0] = term / a;
        roots[1] = constant / term;
        count = 2u;
      }
    }
  }
  var blocked = 0u;
  for (var index = 0u; index < count; index++) {
    let u = roots[index];
    if (u > lower && u < upper) {
      let value = c.x + u * u * (c.y + u * (c.z + c4 * u));
      let magnitude = abs(c.x) + u * u * (abs(c.y) + u * (abs(c.z) + abs(c4) * u));
      // Roundoff near a double root must not decide capture versus escape.
      let uncertainty = 16.0 * 1.1920928955e-7 * magnitude;
      if (abs(value) <= uncertainty) { return 2u; }
      if (value < 0.0) { blocked = 1u; }
    }
  }
  return blocked;
}

/**
 * Null exterior intervals connect to the sky or horizon, or remain unresolved.
 * Return 1 for sky, 2 for horizon, 0 for an ambiguous/trapped interval.
 */
fn radial_destination(c: vec3f, c4: f32, path: QuarticPath, horizon: f32) -> u32 {
  if (path.velocity == 0.0) {
    let u = path.x;
    let derivative_scale = 2.0 * abs(c.y) * u + 3.0 * abs(c.z) * u * u + 4.0 * abs(c4) * u * u * u;
    if (abs(path.first) <= 16.0 * 1.1920928955e-7 * derivative_scale) { return 0u; }
  }
  let inward = path.velocity > 0.0 || (path.velocity == 0.0 && path.first > 0.0);
  let forward = radial_interval(c, c4, select(0.0, path.x, inward), select(path.x, horizon, inward));
  if (forward == 0u) { return select(1u, 2u, inward); }
  if (forward == 2u) { return 0u; }
  let backward = radial_interval(c, c4, select(path.x, 0.0, inward), select(horizon, path.x, inward));
  if (backward != 0u) { return 0u; }
  return select(2u, 1u, inward);
}
