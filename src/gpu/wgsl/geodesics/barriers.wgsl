/** Outward f32 enclosures for source-exclusion proofs; reversed endpoints mean inconclusive. */
fn radial_enclosure(bounds: vec2f) -> vec2f {
  if (!(bounds.x <= bounds.y) || !all(abs(bounds) < vec2f(3.4e38))) { return vec2f(1, -1); }
  var lower = -1.17549435e-38;
  var upper = 1.17549435e-38;
  // Include possible flush-to-zero results. Normal values widen by one representable neighbour.
  if (abs(bounds.x) >= 1.17549435e-38) {
    lower = bitcast<f32>(bitcast<u32>(bounds.x) + select(1u, 0xffffffffu, bounds.x > 0));
  }
  if (abs(bounds.y) >= 1.17549435e-38) {
    upper = bitcast<f32>(bitcast<u32>(bounds.y) + select(0xffffffffu, 1u, bounds.y > 0));
  }
  if (abs(lower) < 1.17549435e-38) { lower = -1.17549435e-38; }
  if (abs(upper) < 1.17549435e-38) { upper = 1.17549435e-38; }
  return vec2f(lower, upper);
}

fn radial_add(a: vec2f, b: vec2f) -> vec2f {
  if (!(a.x <= a.y && b.x <= b.y) || any(abs(a) > vec2f(1.7e38)) || any(abs(b) > vec2f(1.7e38))) {
    return vec2f(1, -1);
  }
  return radial_enclosure(a + b);
}

fn radial_multiply(a: vec2f, b: vec2f) -> vec2f {
  if (!(a.x <= a.y && b.x <= b.y)) { return vec2f(1, -1); }
  let maximum_a = max(abs(a.x), abs(a.y));
  let maximum_b = max(abs(b.x), abs(b.y));
  // Exponent bounds avoid overflowing even the guard itself. Reject uncertain large products.
  let exponents = ((bitcast<u32>(maximum_a) >> 23u) & 255u) + ((bitcast<u32>(maximum_b) >> 23u) & 255u);
  if (exponents > 379u) { return vec2f(1, -1); }
  let products = vec4f(a.x * b.x, a.x * b.y, a.y * b.x, a.y * b.y);
  return radial_enclosure(vec2f(min(min(products.x, products.y), min(products.z, products.w)),
    max(max(products.x, products.y), max(products.z, products.w))));
}

/** Enclose a stored scalar including implementations that flush subnormal inputs. */
fn radial_value(value: f32) -> vec2f {
  if (!(abs(value) < 3.4e38)) { return vec2f(1, -1); }
  if (value == 0) { return vec2f(-1.17549435e-38, 1.17549435e-38); }
  if (abs(value) < 1.17549435e-38) { return vec2f(-1.17549435e-38, 1.17549435e-38); }
  return vec2f(value);
}

/** Positive reciprocal within WGSL's specified divisor range, widened for its 2.5 ULP error. */
fn radial_reciprocal(value: vec2f) -> vec2f {
  if (!(value.x <= value.y && value.x >= 1.17549435e-38 && value.y <= 8.507059e37)) {
    return vec2f(1, -1);
  }
  // Four neighbour expansions also cover a binade boundary in the permitted division error.
  var result = vec2f(1 / value.y, 1 / value.x);
  for (var i = 0u; i < 4u; i++) { result = radial_enclosure(result); }
  return result;
}

/** Finite approximate candidate location; intervals are used to guard arithmetic, not certify extrema. */
fn radial_midpoint(value: vec2f) -> f32 {
  return 0.5 * value.x + 0.5 * value.y;
}

/** Prove R(radius)<0 on the stored constants; uncertain signs keep normal ray continuation. */
fn radial_forbidden(path: KerrOrbit, radius: f32) -> bool {
  let a = radial_value(path.space.x);
  let q = radial_value(path.space.y);
  let e = radial_value(path.constants.x);
  let l = radial_value(path.constants.y);
  let r = radial_value(radius);
  let r2 = radial_multiply(r, r);
  let a2 = radial_multiply(a, a);
  let p = radial_add(radial_multiply(e, radial_add(r2, a2)), -radial_multiply(a, l).yx);
  let difference = radial_add(l, -radial_multiply(a, e).yx);
  var k = radial_add(radial_multiply(difference, difference), radial_value(path.constants.z));
  if (path.plasma.x > 0) {
    let denominator = radial_add(r2, radial_value(path.plasma.y));
    if (!(denominator.x > 1.17549435e-38)) { return false; }
    let reciprocal = radial_reciprocal(denominator);
    k = radial_add(k, radial_multiply(radial_multiply(radial_value(path.plasma.x), r2), reciprocal));
  }
  let delta = radial_add(radial_add(r2, -radial_multiply(vec2f(2), r).yx), radial_add(a2, radial_multiply(q, q)));
  let potential = radial_add(radial_multiply(p, p), -radial_multiply(delta, k).yx);
  return potential.x <= potential.y && potential.y < 0;
}

/** Forbidden radii or the nonrotating singular boundary enclose the observer away from sources. */
fn source_free_radial_band(path: KerrOrbit, source_radius: f32) -> bool {
  if (path.constants.w != 0) { return false; }
  var radius = path.state.radial.x;
  if (path.inverse) {
    let bounds = radial_reciprocal(radial_value(radius));
    if (!(bounds.x > 0 && bounds.x <= bounds.y)) { return false; }
    radius = bounds.y;
  }
  if (!(radius > 0 && radius < source_radius)) { return false; }
  // For zero spin the complete r=0 boundary is singular and already carries zero radiance.
  if (path.space.x != 0 && !radial_forbidden(path, 0)) { return false; }
  if (radial_forbidden(path, source_radius)) { return true; }
  let e = radial_value(path.constants.x);
  let a = radial_value(path.space.x);
  let difference = radial_add(radial_value(path.constants.y), -radial_multiply(a, e).yx);
  let k = radial_add(radial_multiply(difference, difference), radial_value(path.constants.z));
  let quadratic = radial_add(-radial_multiply(radial_multiply(vec2f(2), e), radial_multiply(a, difference)).yx, -k.yx);
  let e2 = radial_multiply(e, e);
  if (!(quadratic.x <= quadratic.y && quadratic.y < 0 && e2.x > 0)) { return false; }
  let scale_interval = radial_multiply(-quadratic.yx, radial_reciprocal(radial_multiply(vec2f(6), e2)));
  if (!(scale_interval.x > 0 && scale_interval.x <= scale_interval.y)) { return false; }
  let scale_squared = radial_midpoint(scale_interval);
  let scale = sqrt(scale_squared);
  if (!(scale < source_radius && scale < 1.7e38)) { return false; }
  let denominator = radial_multiply(radial_multiply(vec2f(4), e2), radial_multiply(radial_value(scale), radial_value(scale_squared)));
  let cosine_interval = radial_multiply(-k.yx, radial_reciprocal(denominator));
  if (!(cosine_interval.x <= cosine_interval.y)) { return false; }
  let cosine = radial_midpoint(cosine_interval);
  if (!(cosine >= -1 && cosine <= 1)) { return false; }
  // The vacuum quartic's positive minimum is only a candidate: its independent interval sign
  // proves the barrier even with approximate acos/sqrt or when plasma moves the minimum.
  let barrier = 2 * scale * cos(acos(cosine) / 3);
  return barrier > radius && barrier < source_radius && radial_forbidden(path, barrier);
}

/** A detector-normalized vacuum photon with vanishing Carter data can remain on a horizon.
 * These are the exactly representable roots in binary f32; nearby radii keep ordinary tracing.
 */
fn source_free_horizon_generator(path: KerrOrbit, source_radius: f32) -> bool {
  if (any(path.constants != vec4f(0)) || path.state.radial.y != 0 || any(path.state.angular != vec3f(0)) || path.plasma.x != 0) {
    return false;
  }
  let a = path.space.x;
  let q = path.space.y;
  let schwarzschild = a == 0 && q == 0 && path.state.radial.x == select(2.0, 0.5, path.inverse);
  let extremal = ((abs(a) == 1 && q == 0) || (a == 0 && abs(q) == 1)) && path.state.radial.x == 1;
  return (schwarzschild && source_radius > 2) || (extremal && source_radius > 1);
}
