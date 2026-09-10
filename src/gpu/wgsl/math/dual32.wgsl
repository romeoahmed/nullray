/**
 * Forward derivatives with respect to the two physical screen coordinates.
 * x stores the value; yz store its gradient. No error-free floating-point transforms are assumed.
 * Failure belongs to the caller. Arithmetic mutates only this explicit flag.
 * Separate flags can evaluate independent rays within the same invocation.
 */

fn d_fail(status: ptr<function, bool>) -> vec3f {
  (*status) = false;
  return vec3f(0.0);
}

/**
 * Keep intermediates below 1e30 and guard each product before evaluation.
 * Addition of two such terms stays finite; this budget is arithmetic, not a
 * claim of physical accuracy at arbitrarily large derivative magnitudes.
 */
fn d_checked(status: ptr<function, bool>, value: vec3f) -> vec3f {
  if (any(abs(value) > vec3f(1e30))) { return d_fail(status); }
  return value;
}
fn d_product_safe(a: f32, b: f32) -> bool {
  if (abs(b) <= 1.0) { return true; }
  return abs(a) <= 1e30 / abs(b);
}
fn d_magnitude(value: vec3f) -> f32 { return max(abs(value.x), max(abs(value.y), abs(value.z))); }
fn d_constant(status: ptr<function, bool>, value: f32) -> vec3f { return d_checked(status, vec3f(value, 0.0, 0.0)); }
fn d_add(status: ptr<function, bool>, a: vec3f, b: vec3f) -> vec3f { return d_checked(status, a + b); }
fn d_sub(status: ptr<function, bool>, a: vec3f, b: vec3f) -> vec3f { return d_checked(status, a - b); }
fn d_scale(status: ptr<function, bool>, a: vec3f, b: f32) -> vec3f {
  if (abs(b) > 1e30 || !d_product_safe(d_magnitude(a), b)) { return d_fail(status); }
  return d_checked(status, a * b);
}
fn d_mul(status: ptr<function, bool>, a: vec3f, b: vec3f) -> vec3f {
  if (!d_product_safe(d_magnitude(a), d_magnitude(b))) { return d_fail(status); }
  return d_checked(status, vec3f(a.x * b.x, a.yz * b.x + a.x * b.yz));
}
fn d_div(status: ptr<function, bool>, a: vec3f, b: vec3f) -> vec3f {
  if (abs(b.x) < 1.1754943508e-38) { return d_fail(status); }
  if (abs(b.x) < 1.0 && abs(a.x) > 1e30 * abs(b.x)) { return d_fail(status); }
  let value = a.x / b.x;
  if (!d_product_safe(value, max(abs(b.y), abs(b.z)))) { return d_fail(status); }
  let numerator = a.yz - value * b.yz;
  if (abs(b.x) < 1.0 && any(abs(numerator) > vec2f(1e30 * abs(b.x)))) { return d_fail(status); }
  return d_checked(status, vec3f(value, numerator / b.x));
}
fn d_compose(status: ptr<function, bool>, input: vec3f, value: f32, slope: f32) -> vec3f {
  let gradient = d_scale(status, vec3f(0.0, input.yz), slope);
  return d_checked(status, vec3f(value, gradient.yz));
}
fn d_sqrt(status: ptr<function, bool>, input: vec3f) -> vec3f {
  if (input.x < 0.0) { return d_fail(status); }
  if (input.x == 0.0) {
    if (any(input.yz != vec2f(0.0))) { return d_fail(status); }
    return vec3f(0.0);
  }
  if (input.x < 1.1754943508e-38) { return d_fail(status); }
  let value = sqrt(input.x);
  // sqrt of a positive normal f32 cannot make this reciprocal overflow.
  return d_compose(status, input, value, 0.5 / value);
}
fn d_sin(status: ptr<function, bool>, input: vec3f) -> vec3f { return d_compose(status, input, sin(input.x), cos(input.x)); }
fn d_cos(status: ptr<function, bool>, input: vec3f) -> vec3f { return d_compose(status, input, cos(input.x), -sin(input.x)); }
fn d_asin(status: ptr<function, bool>, input: vec3f) -> vec3f {
  if (abs(input.x) > 1.0) { return d_fail(status); }
  if (abs(input.x) == 1.0) {
    if (any(input.yz != vec2f(0.0))) { return d_fail(status); }
    return d_constant(status, asin(input.x));
  }
  return d_compose(status, input, asin(input.x), inverseSqrt((1.0 - input.x) * (1.0 + input.x)));
}
/**
 * The derivative is singular at |x|=1 unless the incoming gradient vanishes.
 */
fn d_acos(status: ptr<function, bool>, input: vec3f) -> vec3f {
  if (abs(input.x) > 1.0) { return d_fail(status); }
  if (abs(input.x) == 1.0) {
    if (any(input.yz != vec2f(0.0))) { return d_fail(status); }
    return d_constant(status, acos(input.x));
  }
  return d_compose(status, input, acos(input.x), -inverseSqrt((1.0 - input.x) * (1.0 + input.x)));
}
fn d_cbrt(status: ptr<function, bool>, input: vec3f) -> vec3f {
  if (input.x == 0.0) {
    if (any(input.yz != vec2f(0.0))) { return d_fail(status); }
    return vec3f(0.0);
  }
  if (abs(input.x) < 1.1754943508e-38) { return d_fail(status); }
  let value = real_cbrt(input.x);
  return d_compose(status, input, value, 1.0 / (3.0 * value * value));
}

/**
 * atan2 is homogeneous: a common, locally constant scale cancels from its
 * derivative. Normalizing the primal pair keeps the squared norm in [1,2].
 */
fn d_atan2(status: ptr<function, bool>, y: vec3f, x: vec3f) -> vec3f {
  let scale = max(abs(x.x), abs(y.x));
  if (scale == 0.0) { return d_fail(status); }
  let denominator = d_constant(status, scale);
  let nx = d_div(status, x, denominator);
  let ny = d_div(status, y, denominator);
  let direction = vec2f(nx.x, ny.x);
  let norm = dot(direction, direction);
  let numerator = d_sub(status, d_scale(status, ny, nx.x), d_scale(status, nx, ny.x));
  let gradient = d_div(status, numerator, d_constant(status, norm));
  return d_checked(status, vec3f(atan2(y.x, x.x), gradient.yz));
}
