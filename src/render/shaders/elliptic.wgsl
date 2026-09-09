/**
 * Real Jacobi/Weierstrass kernel, parameter convention m = k².
 * The CPU implementation and source derivation are documented in docs/numerics.md.
 */
struct JacobiPlan {
  m: f32,
  mean: f32,
  multiplier: f32,
  period: f32,
  ratios: array<f32, 8>,
  count: u32,
}

struct Elliptic {
  g2: f32,
  g3: f32,
  root: f32,
  scale: f32,
  m: f32,
  three_real: bool,
}

struct QuarticPath {
  x: f32,
  velocity: f32,
  f: f32,
  first: f32,
  second: f32,
  third: f32,
  fourth: f32,
  elliptic: Elliptic,
}

struct Evaluation {
  value: f32,
  valid: bool,
}

/**
 * Carlson RF duplication, scaled real domain. See docs/numerics.md.
 */
fn carlson_rf(arguments: vec3f) -> Evaluation {
  let scale = max(arguments.x, max(arguments.y, arguments.z));
  if (min(arguments.x, min(arguments.y, arguments.z)) < 0.0 || scale <= 0.0
    || (arguments.x == 0.0 && arguments.y == 0.0)
    || (arguments.x == 0.0 && arguments.z == 0.0)
    || (arguments.y == 0.0 && arguments.z == 0.0)) {
    return Evaluation(0.0, false);
  }
  var v = arguments / scale;
  for (var i = 0u; i < 24u; i++) {
    let mean = (v.x + v.y + v.z) / 3.0;
    let d = (vec3f(mean) - v) / mean;
    if (all(abs(d) < vec3f(0.0025))) {
      let e2 = d.x * d.y - d.z * d.z;
      let e3 = d.x * d.y * d.z;
      return Evaluation((1.0 + (e2 / 24.0 - 0.1 - 3.0 * e3 / 44.0) * e2 + e3 / 14.0)
        / sqrt(mean) / sqrt(scale), true);
    }
    let roots = sqrt(v);
    let lambda = dot(roots, roots.yzx);
    v = (v + vec3f(lambda)) / 4.0;
  }
  return Evaluation(0.0, false);
}


fn real_cbrt(x: f32) -> f32 {
  return sign(x) * pow(abs(x), 1.0 / 3.0);
}

fn prepare_elliptic(g2: f32, g3: f32) -> Elliptic {
  let discriminant = g2 * g2 * g2 - 27.0 * g3 * g3;
  if (g2 > 0.0 && discriminant >= 0.0) {
    let amplitude = sqrt(g2 / 3.0);
    // Discriminant classification establishes the exact acos argument domain.
    let cosine = clamp(5.196152423 * g3 / (g2 * sqrt(g2)), -1.0, 1.0);
    let angle = acos(cosine) / 3.0;
    let e1 = amplitude * cos(angle);
    let e3 = amplitude * cos(angle + 2.094395102);
    let e2 = -e1 - e3;
    return Elliptic(g2, g3, e3, e1 - e3, clamp((e2 - e3) / (e1 - e3), 0.0, 1.0), true);
  }
  let root_disc = sqrt(max(0.0, g3 * g3 / 64.0 - g2 * g2 * g2 / 1728.0));
  let dominant = real_cbrt(g3 / 8.0 + select(root_disc, -root_disc, g3 < 0.0));
  var e = 0.0;
  if (dominant != 0.0) { e = dominant + g2 / (12.0 * dominant); }
  let scale = sqrt(max(0.0, 3.0 * e * e - g2 / 4.0));
  var m = 0.0;
  if (scale != 0.0) { m = clamp(0.5 - 3.0 * e / (4.0 * scale), 0.0, 1.0); }
  return Elliptic(g2, g3, e, scale, m, false);
}

fn prepare_quartic(c: vec4f, c4: f32, x: f32, velocity: f32) -> QuarticPath {
  let f = (((c4 * x + c.w) * x + c.z) * x + c.y) * x + c.x;
  let g2 = c4 * c.x - c.w * c.y / 4.0 + c.z * c.z / 12.0;
  let g3 = c4 * c.z * c.x / 6.0 + c.w * c.z * c.y / 48.0
    - c.z * c.z * c.z / 216.0 - c4 * c.y * c.y / 16.0 - c.w * c.w * c.x / 16.0;
  return QuarticPath(x, velocity, f,
    ((4.0 * c4 * x + 3.0 * c.w) * x + 2.0 * c.z) * x + c.y,
    (12.0 * c4 * x + 6.0 * c.w) * x + 2.0 * c.z,
    24.0 * c4 * x + 6.0 * c.w, 24.0 * c4, prepare_elliptic(g2, g3));
}

/**
 * Parameter-only AGM preparation, reused for every time on this elliptic path.
 */
fn prepare_jacobi(m: f32) -> JacobiPlan {
  var ratios: array<f32, 8>;
  if (m == 0.0 || m == 1.0) { return JacobiPlan(m, 1.0, 1.0, 6.283185307, ratios, 0u); }
  var a = 1.0;
  var b = sqrt(1.0 - m);
  var multiplier = 1.0;
  var count = 0u;
  for (var i = 0u; i < 8u; i++) {
    let next_a = (a + b) / 2.0;
    let ratio = (a - b) / (a + b);
    if (abs(ratio) < 1.0e-7) { break; }
    ratios[i] = ratio;
    count++;
    b = sqrt(a * b);
    a = next_a;
    multiplier *= 2.0;
  }
  return JacobiPlan(m, a, multiplier, 6.283185307 / a, ratios, count);
}

/**
 * Returns (sn, cn, dn), retaining the exact elementary parameter limits.
 */
fn evaluate_jacobi(plan: JacobiPlan, u: f32) -> vec3f {
  let m = plan.m;
  if (m == 0.0) { return vec3f(sin(u), cos(u), 1.0); }
  if (m == 1.0) {
    let e = exp(-abs(u));
    let sech = 2.0 * e / (1.0 + e * e);
    // Native tanh retains small arguments; beyond 20 its f32 value is ±1.
    // Guard first: WGSL defines tanh through sinh/cosh, which can overflow.
    var sn = sign(u);
    if (abs(u) <= 20.0) { sn = tanh(u); }
    return vec3f(sn, sech, sech);
  }
  var phase = plan.multiplier * plan.mean * (u - trunc(u / plan.period) * plan.period);
  for (var i = plan.count; i > 0u; i--) {
    phase = (phase + asin(plan.ratios[i - 1u] * sin(phase))) / 2.0;
  }
  let sn = sin(phase);
  let cn = cos(phase);
  // dn² = (1-m) + m cn² avoids cancellation near m=1 and sn=±1.
  return vec3f(sn, cn, sqrt((1.0 - m) + m * cn * cn));
}

fn jacobi(u: f32, m: f32) -> vec3f {
  return evaluate_jacobi(prepare_jacobi(m), u);
}

/**
 * Cancel the Weierstrass pole before forming its quartic quotient. The
 * trajectory is finite when sn (or sn*dn) vanishes; no pole epsilon is needed.
 */
fn evaluate_quartic_kernel(path: QuarticPath, t: f32, plan: JacobiPlan, prepared: bool) -> Evaluation {
  if (t == 0.0) { return Evaluation(path.x, true); }
  let p = path.elliptic;
  let t2 = t * t;
  let t4 = t2 * t2;
  let g2t4 = p.g2 * t4;
  let g3t6 = p.g3 * t4 * t2;
  if (abs(g2t4) + abs(g3t6) >= 1e-3) {
    let root_scale = sqrt(p.scale);
    var j: vec3f;
    if (prepared) { j = evaluate_jacobi(plan, root_scale * t); }
    else { j = jacobi(root_scale * t, p.m); }
    var z = j.x;
    var pole_numerator = p.scale;
    var slope = j.y * j.z;
    if (!p.three_real) {
      z = j.x * j.z;
      let cn2 = j.y * j.y;
      pole_numerator = p.scale * cn2;
      slope = j.y * ((1.0 - p.m) + p.m * cn2 * cn2);
    }
    let z2 = z * z;
    let z4 = z2 * z2;
    let b = (p.root - path.second / 24.0) * z2 + pole_numerator;
    let correction = path.f * path.fourth * z4 / 48.0;
    let numerator = 2.0 * path.velocity * root_scale * p.scale * slope * z
      + path.first * b * z2 / 2.0 + path.f * path.third * z4 / 24.0;
    return quartic_quotient(path.x, numerator, b, correction);
  }
  // The Laurent series after multiplication by t² is regular at t=0.
  let w = 1.0 + g2t4 / 20.0 + g3t6 / 28.0;
  let derivative = -2.0 + g2t4 / 10.0 + g3t6 / 7.0;
  let b = w - path.second * t2 / 24.0;
  let correction = path.f * path.fourth * t4 / 48.0;
  let numerator = -path.velocity * derivative * t + path.first * b * t2 / 2.0
    + path.f * path.third * t4 / 24.0;
  return quartic_quotient(path.x, numerator, b, correction);
}

fn quartic_quotient(initial: f32, numerator: f32, b: f32, correction: f32) -> Evaluation {
  let square = 2.0 * b * b;
  let denominator = square - correction;
  if (abs(denominator) <= 1e-6 * (square + abs(correction))) { return Evaluation(0.0, false); }
  return Evaluation(initial + numerator / denominator, true);
}

fn evaluate_quartic(path: QuarticPath, t: f32) -> Evaluation {
  return evaluate_quartic_kernel(path, t, JacobiPlan(), false);
}

fn evaluate_quartic_prepared(path: QuarticPath, t: f32, plan: JacobiPlan) -> Evaluation {
  return evaluate_quartic_kernel(path, t, plan, true);
}
