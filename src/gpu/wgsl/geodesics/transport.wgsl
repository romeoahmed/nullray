/** Eight-point Gauss–Legendre nodes and weights on [0,1]; exact through polynomial degree 15. */
const TRANSPORT_GAUSS = array<vec2f,8>(
  vec2f(0.0198550717512319, 0.0506142681451884),
  vec2f(0.1016667612931866, 0.1111905172266872),
  vec2f(0.2372337950418355, 0.1568533229389437),
  vec2f(0.4082826787521751, 0.1813418916891810),
  vec2f(0.5917173212478248, 0.1813418916891810),
  vec2f(0.7627662049581645, 0.1568533229389437),
  vec2f(0.8983332387068135, 0.1111905172266872),
  vec2f(0.9801449282487682, 0.0506142681451884),
);

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
  polar_coefficient: f32,
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
 * Unwrapped primitive for the singular polar part. Its unit coefficient
 * leaves the smooth residual Lm/[zw(w+dn)], where w²=1+m(1-z)/z.
 */
fn polar_primitive(events: EquatorEvents, complement: f32, u: f32, zero_momentum: bool) -> f32 {
  let period = 2.0 * events.spacing * events.frequency;
  if (zero_momentum) {
    return 3.1415926536 * (floor(u / (period / 2.0)) + 0.5);
  }
  let cycles = floor((u + period / 2.0) / period);
  return polar_angle(events,complement,u) + cycles * 6.283185307;
}

/** Nonzero-momentum primitive modulo full turns, sufficient for sky directions. */
fn polar_angle(events: EquatorEvents, complement: f32, u: f32) -> f32 {
  let period = 2.0 * events.spacing * events.frequency;
  let cycles = floor((u + period / 2.0) / period);
  let j = jacobi(u - cycles * period,events.m);
  return atan2(j.x,sqrt(complement)*j.y);
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
    let w = kernel.polar_coefficient;
    let denominator = events.amplitude2 * w * (w + j.z);
    if (denominator <= 0.0) { return TransportRate(vec2f(0.0), false); }
    polar_rate = l * events.m / denominator;
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
 * Composite eight-point Gauss–Legendre rule of the cosine-mapped rate.
 * Four interleaved sums shorten rounding chains; endpoints are checked separately.
 */
fn transport_grid(kernel: TransportKernel, count: u32, end: f32) -> TransportRate {
  var partial: array<vec2f, 4>;
  let width = 8.0 / f32(count);
  for (var i = 0u; i < count; i++) {
    let quadrature = TRANSPORT_GAUSS[i & 7u];
    let sample = warped_transport_rate(kernel, (f32(i / 8u) + quadrature.x) * width, end);
    if (!sample.valid) { return TransportRate(vec2f(0.0), false); }
    partial[i % 4u] += sample.value * (quadrature.y * width);
  }
  return TransportRate((partial[0] + partial[1]) + (partial[2] + partial[3]), true);
}

/**
 * Successive composite Gauss refinements along the analytic path. Two consecutive
 * differences must meet the angular/time targets; the fine estimate is accepted
 * without a Simpson-specific extrapolation. Exhaustion remains unresolved.
 */
fn integrate_transport_budget(radial: QuarticPath, polar: QuarticPath, a: f32, q: f32,
  e: f32, l: f32, c: f32, end: f32, include_time: bool, events: EquatorEvents, max_count: u32) -> TransportResult {
  if (end == 0.0) { return TransportResult(vec2f(0.0), vec2f(0.0), true); }
  if (abs(polar.x) > 1.0) { return TransportResult(vec2f(0.0), vec2f(0.0), false); }
  if (events.status == 2u) { return TransportResult(vec2f(0.0), vec2f(0.0), false); }
  var complement = 0.0;
  var polar_coefficient = 1.0;
  var base = 0.0;
  if (events.status == 1u) {
    let z = events.amplitude2;
    complement = l * l * z / (c + a * a * e * e * z);
    polar_coefficient = sqrt(1.0 + events.m * complement / z);
    let phase = events.phase + events.frequency * end;
    var primitive = 0.0;
    if (!include_time && l != 0.0) {
      primitive = polar_angle(events,complement,phase);
    } else { primitive = polar_primitive(events,complement,phase,l == 0.0); }
    base = -select(1.0, -1.0, l < 0.0) * (primitive - events.azimuth_phase);
  }
  // In spherical symmetry m=0 and frame dragging vanishes. The polar
  // primitive is the complete azimuth integral, including axis crossings.
  if (a == 0.0 && events.status == 1u && !include_time) {
    return TransportResult(vec2f(base, 0.0), vec2f(0.0), true);
  }
  // Only one polar representation is active on this path.
  let polar_m = select(polar.elliptic.m, events.m, events.status == 1u);
  let polar_complement = select(polar.elliptic.complement,1.0-events.m,events.status == 1u);
  let plan = TransportJacobi(prepare_jacobi_pair(radial.elliptic.m,radial.elliptic.complement), prepare_jacobi_pair(polar_m,polar_complement));
  let kernel = TransportKernel(radial, polar, vec2f(a, q), vec2f(e, l), include_time, events, complement, polar_coefficient, plan);
  let begin_rate = transport_rate(kernel, 0.0);
  let end_rate = transport_rate(kernel, end);
  if (!begin_rate.valid || !end_rate.valid) { return TransportResult(vec2f(0.0), vec2f(0.0), false); }
  let phase = end * sqrt(l * l + 2.0 * abs(c) + 2.0 * a * a * e * e);
  var count = 8u;
  // At least two nodes per estimated phase radian: a Gauss panel spans at
  // most four radians before warping, instead of hiding a full oscillation.
  for (var i = 0u; i < 6u && f32(count) < 2.0 * phase; i++) { count *= 2u; }
  // Reserve two doubled-grid estimates within the selected work budget.
  if (count > max_count / 4u) { return TransportResult(vec2f(0.0), vec2f(0.0), false); }
  let sum = transport_grid(kernel, count, end);
  if (!sum.valid) { return TransportResult(vec2f(0.0), vec2f(0.0), false); }
  var previous = sum.value;
  var agreed = false;
  for (var level = 0u; level < 7u && count < max_count; level++) {
    count *= 2u;
    let integral = transport_grid(kernel, count, end);
    if (!integral.valid) { return TransportResult(vec2f(0.0), vec2f(0.0), false); }
    let estimate = integral.value;
    let error = abs(estimate - previous);
    // Azimuth is absolute angular error; travel time uses relative scaling.
    let tolerance = vec2f(2.0e-5, 2.0e-5 * (1.0 + abs(estimate.y)));
    let converged = all(error <= tolerance);
    if (converged && agreed) {
      return TransportResult(vec2f(base, 0.0) + estimate, error, true);
    }
    agreed = converged;
    previous = estimate;
  }
  return TransportResult(vec2f(0.0), vec2f(0.0), false);
}

/** Ordinary paths retain their established budget; precision preparation owns any additional work. */
fn integrate_transport(radial: QuarticPath, polar: QuarticPath, a: f32, q: f32,
  e: f32, l: f32, c: f32, end: f32, include_time: bool, events: EquatorEvents) -> TransportResult {
  return integrate_transport_budget(radial,polar,a,q,e,l,c,end,include_time,events,select(1024u,512u,include_time));
}
