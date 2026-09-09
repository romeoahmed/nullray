/**
 * Real elliptic functions differentiated in physical screen coordinates.
 * Arithmetic guards are defined in derivative.wgsl.
 */

struct DifferentialJacobi { sn: vec3f, cn: vec3f, dn: vec3f }
struct DifferentialJacobiPlan {
  m: vec3f,
  a: vec3f,
  period: vec3f,
  multiplier: f32,
  ratios: array<vec3f, 8>,
  count: u32,
}

/**
 * The modulus and its screen gradient stay fixed throughout one trajectory.
 */
fn d_prepare_jacobi(status: ptr<function, bool>, m: vec3f) -> DifferentialJacobiPlan {
  var plan: DifferentialJacobiPlan;
  plan.m = m;
  // Limiting and invalid moduli are handled on evaluation, preserving lazy limits.
  if (m.x <= 0.0 || m.x >= 1.0) { return plan; }
  var a = d_constant(status, 1.0);
  var b = d_sqrt(status, d_sub(status, d_constant(status, 1.0), m));
  var multiplier = 1.0;
  var ratios: array<vec3f, 8>;
  var count = 0u;
  for (var i = 0u; i < 8u; i++) {
    let sum = d_add(status, a, b);
    let ratio = d_div(status, d_sub(status, a, b), sum);
    if (abs(ratio.x) < 1e-7) { break; }
    ratios[i] = ratio;
    count++;
    b = d_sqrt(status, d_mul(status, a, b));
    a = d_scale(status, sum, 0.5);
    multiplier *= 2.0;
  }
  plan.a = a;
  plan.period = d_div(status, d_constant(status, 6.283185307), a);
  plan.multiplier = multiplier;
  plan.ratios = ratios;
  plan.count = count;
  return plan;
}

fn d_evaluate_jacobi(status: ptr<function, bool>, plan: DifferentialJacobiPlan, u: vec3f) -> DifferentialJacobi {
  let m = plan.m;
  if (m.x < 0.0 || m.x > 1.0) {
    return DifferentialJacobi(d_fail(status), vec3f(0.0), vec3f(0.0));
  }
  if (m.x == 0.0) {
    if (any(m.yz != vec2f(0.0))) { return DifferentialJacobi(d_fail(status), vec3f(0.0), vec3f(0.0)); }
    return DifferentialJacobi(d_sin(status, u), d_cos(status, u), d_constant(status, 1.0));
  }
  if (m.x == 1.0) {
    if (any(m.yz != vec2f(0.0))) { return DifferentialJacobi(d_fail(status), vec3f(0.0), vec3f(0.0)); }
    let j = jacobi(u.x, 1.0);
    return DifferentialJacobi(d_compose(status, u, j.x, j.y * j.y),
      d_compose(status, u, j.y, -j.x * j.y), d_compose(status, u, j.z, -j.x * j.z));
  }
  let period = plan.period;
  if (!(period.x > 0.0)) { return DifferentialJacobi(d_fail(status), vec3f(0.0), vec3f(0.0)); }
  let cycles = trunc(u.x / period.x);
  var phase = d_scale(status, d_mul(status, plan.a, d_sub(status, u, d_scale(status, period, cycles))), plan.multiplier);
  for (var i = plan.count; i > 0u; i--) {
    phase = d_scale(status, d_add(status, phase, d_asin(status, d_mul(status, plan.ratios[i - 1u], d_sin(status, phase)))), 0.5);
  }
  let sn = d_sin(status, phase);
  let cn = d_cos(status, phase);
  let dn = d_sqrt(status, d_add(status, d_sub(status, d_constant(status, 1.0), m), d_mul(status, m, d_mul(status, cn, cn))));
  return DifferentialJacobi(sn, cn, dn);
}
fn d_jacobi(status: ptr<function, bool>, u: vec3f, m: vec3f) -> DifferentialJacobi {
  return d_evaluate_jacobi(status, d_prepare_jacobi(status, m), u);
}

struct DifferentialElliptic {
  g2: vec3f, g3: vec3f, root: vec3f, scale: vec3f, m: vec3f, three_real: bool,
  jacobi: DifferentialJacobiPlan,
}
struct DifferentialQuartic {
  x: vec3f, velocity: vec3f, f: vec3f, first: vec3f, second: vec3f,
  third: vec3f, fourth: vec3f, elliptic: DifferentialElliptic,
}
fn d_prepare_elliptic(status: ptr<function, bool>, g2: vec3f, g3: vec3f) -> DifferentialElliptic {
  let g2cubed = d_mul(status, d_mul(status, g2, g2), g2);
  let discriminant = d_sub(status, g2cubed, d_scale(status, d_mul(status, g3, g3), 27.0));
  if (g2.x > 0.0 && discriminant.x >= 0.0) {
    let amplitude = d_sqrt(status, d_scale(status, g2, 1.0 / 3.0));
    let cosine = d_div(status, d_scale(status, g3, 5.196152423), d_mul(status, g2, d_sqrt(status, g2)));
    let angle = d_scale(status, d_acos(status, cosine), 1.0 / 3.0);
    let e1 = d_mul(status, amplitude, d_cos(status, angle));
    let e3 = d_mul(status, amplitude, d_cos(status, d_add(status, angle, d_constant(status, 2.094395102))));
    let e2 = d_scale(status, d_add(status, e1, e3), -1.0);
    let scale = d_sub(status, e1, e3);
    let m = d_div(status, d_sub(status, e2, e3), scale);
    return DifferentialElliptic(g2, g3, e3, scale, m, true, d_prepare_jacobi(status, m));
  }
  let root_disc = d_sqrt(status, d_sub(status, d_scale(status, d_mul(status, g3, g3), 1.0 / 64.0), d_scale(status, g2cubed, 1.0 / 1728.0)));
  let dominant = d_cbrt(status, d_add(status, d_scale(status, g3, 0.125), d_scale(status, root_disc, select(1.0, -1.0, g3.x < 0.0))));
  var e = d_constant(status, 0.0);
  if (dominant.x != 0.0) { e = d_add(status, dominant, d_div(status, g2, d_scale(status, dominant, 12.0))); }
  let scale = d_sqrt(status, d_sub(status, d_scale(status, d_mul(status, e, e), 3.0), d_scale(status, g2, 0.25)));
  var m = d_constant(status, 0.0);
  if (scale.x != 0.0) { m = d_sub(status, d_constant(status, 0.5), d_div(status, d_scale(status, e, 0.75), scale)); }
  return DifferentialElliptic(g2, g3, e, scale, m, false, d_prepare_jacobi(status, m));
}

fn d_prepare_quartic(status: ptr<function, bool>, c: array<vec3f, 5>, x: vec3f, velocity: vec3f) -> DifferentialQuartic {
  let f = d_add(status, d_mul(status, d_add(status, d_mul(status, d_add(status, d_mul(status, d_add(status, d_mul(status, c[4], x), c[3]), x), c[2]), x), c[1]), x), c[0]);
  let g2 = d_add(status, d_sub(status, d_mul(status, c[4], c[0]), d_scale(status, d_mul(status, c[3], c[1]), 0.25)), d_scale(status, d_mul(status, c[2], c[2]), 1.0 / 12.0));
  let g3 = d_sub(status, d_sub(status, d_add(status, d_scale(status, d_mul(status, d_mul(status, c[4], c[2]), c[0]), 1.0 / 6.0),
    d_scale(status, d_mul(status, d_mul(status, c[3], c[2]), c[1]), 1.0 / 48.0)),
    d_scale(status, d_mul(status, d_mul(status, c[2], c[2]), c[2]), 1.0 / 216.0)),
    d_scale(status, d_add(status, d_mul(status, c[4], d_mul(status, c[1], c[1])), d_mul(status, d_mul(status, c[3], c[3]), c[0])), 1.0 / 16.0));
  let first = d_add(status, d_mul(status, d_add(status, d_mul(status, d_add(status, d_scale(status, d_mul(status, c[4], x), 4.0), d_scale(status, c[3], 3.0)), x), d_scale(status, c[2], 2.0)), x), c[1]);
  let second = d_add(status, d_mul(status, d_add(status, d_scale(status, d_mul(status, c[4], x), 12.0), d_scale(status, c[3], 6.0)), x), d_scale(status, c[2], 2.0));
  let third = d_add(status, d_scale(status, d_mul(status, c[4], x), 24.0), d_scale(status, c[3], 6.0));
  return DifferentialQuartic(x, velocity, f, first, second, third, d_scale(status, c[4], 24.0), d_prepare_elliptic(status, g2, g3));
}

fn d_primal_quartic(path: DifferentialQuartic) -> QuarticPath {
  let e = path.elliptic;
  return QuarticPath(path.x.x, path.velocity.x, path.f.x, path.first.x, path.second.x, path.third.x, path.fourth.x,
    Elliptic(e.g2.x, e.g3.x, e.root.x, e.scale.x, e.m.x, e.three_real));
}

fn d_evaluate_quartic(status: ptr<function, bool>, path: DifferentialQuartic, t: vec3f) -> vec3f {
  // At t=0 the value is known, but its time gradient must still be propagated.
  if (t.x == 0.0) { return d_add(status, path.x, d_mul(status, path.velocity, t)); }
  let p = path.elliptic;
  let t2 = d_mul(status, t, t);
  let t4 = d_mul(status, t2, t2);
  let g2t4 = d_mul(status, p.g2, t4);
  let g3t6 = d_mul(status, p.g3, d_mul(status, t4, t2));
  let w = d_add(status, d_constant(status, 1.0), d_add(status, d_scale(status, g2t4, 0.05), d_scale(status, g3t6, 1.0 / 28.0)));
  let derivative = d_add(status, d_constant(status, -2.0), d_add(status, d_scale(status, g2t4, 0.1), d_scale(status, g3t6, 1.0 / 7.0)));
  if (abs(g2t4.x) + abs(g3t6.x) >= 1e-3) {
    let root_scale = d_sqrt(status, p.scale);
    let j = d_evaluate_jacobi(status, p.jacobi, d_mul(status, root_scale, t));
    var z = j.sn;
    var wp_numerator = p.scale;
    var slope_factor = d_mul(status, j.cn, j.dn);
    if (!p.three_real) {
      z = d_mul(status, j.sn, j.dn);
      let cn2 = d_mul(status, j.cn, j.cn);
      wp_numerator = d_mul(status, p.scale, cn2);
      slope_factor = d_mul(status, j.cn, d_add(status, d_sub(status, d_constant(status, 1.0), p.m), d_mul(status, p.m, d_mul(status, cn2, cn2))));
    }
    // Multiply the Biermann numerator and denominator by z^4 before evaluation.
    // The Weierstrass pole at z=0 is removable in the quartic trajectory.
    let z2 = d_mul(status, z, z);
    let z4 = d_mul(status, z2, z2);
    let b = d_add(status, d_mul(status, d_sub(status, p.root, d_scale(status, path.second, 1.0 / 24.0)), z2), wp_numerator);
    let term = d_scale(status, d_mul(status, d_mul(status, path.f, path.fourth), z4), 1.0 / 48.0);
    let square = d_mul(status, b, b);
    let denominator = d_sub(status, d_scale(status, square, 2.0), term);
    if (!(*status)) { return vec3f(0.0); }
    if (abs(denominator.x) <= 1e-6 * (2.0 * square.x + abs(term.x))) { return d_fail(status); }
    let leading = d_scale(status, d_mul(status, d_mul(status, d_mul(status, path.velocity, root_scale), p.scale), d_mul(status, slope_factor, z)), 2.0);
    let numerator = d_add(status, d_add(status, leading, d_scale(status, d_mul(status, d_mul(status, path.first, b), z2), 0.5)),
      d_scale(status, d_mul(status, d_mul(status, path.f, path.third), z4), 1.0 / 24.0));
    return d_add(status, path.x, d_div(status, numerator, denominator));
  }
  let b = d_sub(status, w, d_scale(status, d_mul(status, path.second, t2), 1.0 / 24.0));
  let term = d_scale(status, d_mul(status, d_mul(status, path.f, path.fourth), t4), 1.0 / 48.0);
  let square = d_mul(status, b, b);
  let denominator = d_sub(status, d_scale(status, square, 2.0), term);
  if (!(*status)) { return vec3f(0.0); }
  if (abs(denominator.x) <= 1e-6 * (2.0 * square.x + abs(term.x))) { return d_fail(status); }
  let numerator = d_add(status, d_sub(status, d_scale(status, d_mul(status, d_mul(status, path.first, b), t2), 0.5), d_mul(status, d_mul(status, path.velocity, derivative), t)),
    d_scale(status, d_mul(status, d_mul(status, path.f, path.third), t4), 1.0 / 24.0));
  return d_add(status, path.x, d_div(status, numerator, denominator));
}
/**
 * Carlson RF with parameter derivatives. Zero arguments must have zero gradients;
 * endpoint amplitude derivatives of F are added analytically below.
 */
fn d_carlson_rf(status: ptr<function, bool>, x: vec3f, y: vec3f, z: vec3f) -> vec3f {
  let scale = max(x.x, max(y.x, z.x));
  if (min(x.x, min(y.x, z.x)) < 0.0 || scale <= 0.0
    || (x.x == 0.0 && y.x == 0.0) || (x.x == 0.0 && z.x == 0.0)
    || (y.x == 0.0 && z.x == 0.0)) { return d_fail(status); }
  var vx = d_div(status, x, d_constant(status, scale));
  var vy = d_div(status, y, d_constant(status, scale));
  var vz = d_div(status, z, d_constant(status, scale));
  for (var i = 0u; i < 24u; i++) {
    let mean = d_scale(status, d_add(status, d_add(status, vx, vy), vz), 1.0 / 3.0);
    let dx = d_sub(status, d_constant(status, 1.0), d_div(status, vx, mean));
    let dy = d_sub(status, d_constant(status, 1.0), d_div(status, vy, mean));
    let dz = d_sub(status, d_constant(status, 1.0), d_div(status, vz, mean));
    if (max(abs(dx.x), max(abs(dy.x), abs(dz.x))) < 0.0025) {
      let e2 = d_sub(status, d_mul(status, dx, dy), d_mul(status, dz, dz));
      let e3 = d_mul(status, d_mul(status, dx, dy), dz);
      let polynomial = d_add(status, d_add(status, d_constant(status, 1.0), d_mul(status, d_sub(status, d_sub(status, d_scale(status, e2, 1.0 / 24.0),
        d_constant(status, 0.1)), d_scale(status, e3, 3.0 / 44.0)), e2)), d_scale(status, e3, 1.0 / 14.0));
      return d_div(status, d_div(status, polynomial, d_sqrt(status, mean)), d_constant(status, sqrt(scale)));
    }
    let rx = d_sqrt(status, vx); let ry = d_sqrt(status, vy); let rz = d_sqrt(status, vz);
    let lambda = d_add(status, d_add(status, d_mul(status, rx, ry), d_mul(status, rx, rz)), d_mul(status, ry, rz));
    vx = d_scale(status, d_add(status, vx, lambda), 0.25);
    vy = d_scale(status, d_add(status, vy, lambda), 0.25);
    vz = d_scale(status, d_add(status, vz, lambda), 0.25);
  }
  return d_fail(status);
}

/**
 * F(amplitude|m), with the amplitude derivative evaluated before squaring cos.
 * This preserves dF/damplitude at a polar turn and at quarter periods.
 */
fn d_incomplete_f(status: ptr<function, bool>, amplitude: vec3f, m: vec3f) -> vec3f {
  if (m.x < 0.0 || m.x >= 1.0) { return d_fail(status); }
  let cycles = floor((amplitude.x + 1.5707963268) / 3.1415926536);
  let angle = amplitude.x - cycles * 3.1415926536;
  let sine = sin(angle); let cosine = cos(angle);
  let y = d_sub(status, d_constant(status, 1.0), d_scale(status, m, sine * sine));
  let local = d_scale(status, d_carlson_rf(status, d_constant(status, cosine * cosine), y, d_constant(status, 1.0)), sine);
  let complete = d_carlson_rf(status, d_constant(status, 0.0), d_sub(status, d_constant(status, 1.0), m), d_constant(status, 1.0));
  let value = d_add(status, local, d_scale(status, complete, 2.0 * cycles));
  return d_add(status, value, d_div(status, vec3f(0.0, amplitude.yz), d_sqrt(status, y)));
}
