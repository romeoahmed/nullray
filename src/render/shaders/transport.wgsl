struct TransportJacobi {
  radial: JacobiPlan,
  polar: JacobiPlan,
}

/**
 * Prepared immutable inputs shared by every quadrature node on one path.
 */
struct TransportKernel {
  radial: QuarticPath,
  polar: QuarticPath,
  space: vec2f,
  photon: vec2f,
  include_time: bool,
  events: EquatorEvents,
  complement: f32,
  plan: TransportJacobi,
}

struct TransportRate {
  value: vec2f,
  valid: bool,
}

struct TransportResult {
  value: vec2f,
  error: vec2f,
  valid: bool,
}

/**
 * Unwrapped primitive for the singular polar part. The remaining integrand
 * is L (1-dn)/(1-mu²), regularized with 1-dn = m sn²/(1+dn).
 */
fn polar_primitive(events: EquatorEvents, complement: f32, u: f32, zero_momentum: bool) -> f32 {
  let period = 2.0 * events.spacing * events.frequency;
  if (zero_momentum) {
    return 3.1415926536 * (floor(u / (period / 2.0)) + 0.5);
  }
  let cycles = floor((u + period / 2.0) / period);
  let j = jacobi(u - cycles * period, events.m);
  return atan2(j.x, sqrt(complement) * j.y) + cycles * 6.283185307;
}

/**
 * Positive backward Mino parameter; future-directed E and L are retained.
 */
fn transport_rate(kernel: TransportKernel, time: f32) -> TransportRate {
  let a = kernel.space.x;
  let q = kernel.space.y;
  let e = kernel.photon.x;
  let l = kernel.photon.y;
  let events = kernel.events;
  let ru = evaluate_quartic_prepared(kernel.radial, time, kernel.plan.radial);
  if (!ru.valid) { return TransportRate(vec2f(0.0), false); }
  let u = ru.value;
  let u2 = u * u;
  var sin2 = 0.0;
  var polar_rate = 0.0;
  if (events.status == 1u) {
    let j = evaluate_jacobi(kernel.plan.polar, events.phase + events.frequency * time);
    sin2 = kernel.complement + events.amplitude2 * j.x * j.x;
    if (l != 0.0) {
      if (sin2 <= 0.0) { return TransportRate(vec2f(0.0), false); }
      polar_rate = l * events.m * j.x * j.x / ((1.0 + j.z) * sin2);
    }
  } else {
    let mu = evaluate_quartic_prepared(kernel.polar, time, kernel.plan.polar);
    if (!mu.valid) { return TransportRate(vec2f(0.0), false); }
    sin2 = 1.0 - mu.value * mu.value;
    if (sin2 <= 0.0) { return TransportRate(vec2f(0.0), false); }
    polar_rate = l / sin2;
  }
  let delta = 1.0 - 2.0 * u + (a * a + q * q) * u2;
  if (delta <= 0.0) { return TransportRate(vec2f(0.0), false); }
  let phi = -(a * (2.0 * e * u - (q * q * e + a * l) * u2) / delta + polar_rate);
  var t = 0.0;
  if (kernel.include_time) {
    if (u <= 0.0) { return TransportRate(vec2f(0.0), false); }
    t = -((1.0 + a * a * u2) * (e + (a * a * e - a * l) * u2) / (u2 * delta)
      + a * l - a * a * e * sin2);
  }
  return TransportRate(vec2f(phi, t), true);
}

/**
 * Cosine-mapped quadrature resolves the sharp endpoint rates of long scattering paths.
 */
fn warped_transport_rate(kernel: TransportKernel, fraction: f32, end: f32) -> TransportRate {
  let angle = 1.5707963268 * fraction;
  let sine = sin(angle);
  let sample = transport_rate(kernel, end * sine * sine);
  return TransportRate(sample.value * (3.1415926536 * end * sine * cos(angle)), sample.valid);
}

/**
 * Four interleaved accumulators shorten rounding chains. Offset zero samples
 * the open grid; offset 1/2 samples only new midpoints in a doubled grid.
 */
fn transport_grid(kernel: TransportKernel, count: u32, offset: f32, end: f32) -> TransportRate {
  var partial: array<vec2f, 4>;
  let first = select(0u, 1u, offset == 0.0);
  for (var i = first; i < count; i++) {
    let sample = warped_transport_rate(kernel, (f32(i) + offset) / f32(count), end);
    if (!sample.valid) { return TransportRate(vec2f(0.0), false); }
    partial[i % 4u] += sample.value;
  }
  return TransportRate((partial[0] + partial[1]) + (partial[2] + partial[3]), true);
}

/**
 * Composite Simpson refinement along the analytic path. Retained trapezoidal
 * sums reuse old evaluations. Failure to meet the estimate stays unresolved.
 * Positive-C polar motion subtracts an analytic axis primitive.
 */
fn integrate_transport(radial: QuarticPath, polar: QuarticPath, a: f32, q: f32,
  e: f32, l: f32, c: f32, end: f32, include_time: bool, events: EquatorEvents) -> TransportResult {
  if (end == 0.0) { return TransportResult(vec2f(0.0), vec2f(0.0), true); }
  if (abs(polar.x) > 1.0) { return TransportResult(vec2f(0.0), vec2f(0.0), false); }
  if (events.status == 2u) { return TransportResult(vec2f(0.0), vec2f(0.0), false); }
  var complement = 0.0;
  var base = 0.0;
  if (events.status == 1u) {
    let z = events.amplitude2;
    complement = l * l * z / (c + a * a * e * e * z);
    let coefficient = select(1.0, -1.0, l < 0.0) * sqrt(c / z + a * a * e * e) / events.frequency;
    base = -coefficient * (polar_primitive(events, complement, events.phase + events.frequency * end, l == 0.0) - events.azimuth_phase);
  }
  // In spherical symmetry m=0 and frame dragging vanishes. The polar
  // primitive is the complete azimuth integral, including axis crossings.
  if (a == 0.0 && events.status == 1u && !include_time) {
    return TransportResult(vec2f(base, 0.0), vec2f(0.0), true);
  }
  // Only one polar representation is active on this path.
  let polar_m = select(polar.elliptic.m, events.m, events.status == 1u);
  let plan = TransportJacobi(prepare_jacobi(radial.elliptic.m), prepare_jacobi(polar_m));
  let kernel = TransportKernel(radial, polar, vec2f(a, q), vec2f(e, l), include_time, events, complement, plan);
  let begin_rate = transport_rate(kernel, 0.0);
  let end_rate = transport_rate(kernel, end);
  if (!begin_rate.valid || !end_rate.valid) { return TransportResult(vec2f(0.0), vec2f(0.0), false); }
  let phase = end * sqrt(l * l + 2.0 * abs(c) + 2.0 * a * a * e * e);
  var count = 8u;
  for (var i = 0u; i < 6u && f32(count) < 8.0 * phase; i++) { count *= 2u; }
  // Reserve two doubled-grid estimates within the selected work budget.
  let max_count = select(1024u, 512u, include_time);
  if (count > max_count / 4u) { return TransportResult(vec2f(0.0), vec2f(0.0), false); }
  // The change-of-variable Jacobian vanishes at both finite endpoints.
  let sum = transport_grid(kernel, count, 0.0, end);
  if (!sum.valid) { return TransportResult(vec2f(0.0), vec2f(0.0), false); }
  var trapezoid = sum.value / f32(count);
  var previous = vec2f(0.0);
  for (var level = 0u; level < 7u && count < max_count; level++) {
    let midpoints = transport_grid(kernel, count, 0.5, end);
    if (!midpoints.valid) { return TransportResult(vec2f(0.0), vec2f(0.0), false); }
    let next = trapezoid / 2.0 + midpoints.value / (2.0 * f32(count));
    let estimate = (4.0 * next - trapezoid) / 3.0;
    let error = abs(estimate - previous) / 15.0;
    // Azimuth is absolute angular error; travel time uses relative scaling.
    let tolerance = vec2f(2.0e-5, 2.0e-5 * (1.0 + abs(estimate.y)));
    if (level > 0u && all(error <= tolerance)) {
      return TransportResult(vec2f(base, 0.0) + estimate + (estimate - previous) / 15.0, error, true);
    }
    previous = estimate;
    trapezoid = next;
    count *= 2u;
  }
  return TransportResult(vec2f(0.0), vec2f(0.0), false);
}
