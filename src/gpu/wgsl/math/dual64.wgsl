/** Binary64 value and two screen-coordinate derivatives; each lane is an integer IEEE word pair. */
struct Dual64 {
  value: Soft64,
  dx: Soft64,
  dy: Soft64,
}

fn d64_constant(value: Soft64) -> Dual64 {
  return Dual64(value,vec2u(0u),vec2u(0u));
}
fn d64_from_f32_bits(value: u32) -> Dual64 {
  return d64_constant(soft64_from_f32_bits(value));
}
fn d64_valid(value: Dual64) -> bool {
  return soft64_valid(value.value) && soft64_valid(value.dx) && soft64_valid(value.dy);
}
fn d64_failure() -> Dual64 {
  return Dual64(SOFT64_INVALID,SOFT64_INVALID,SOFT64_INVALID);
}
fn d64_add(a: Dual64, b: Dual64) -> Dual64 {
  return Dual64(soft64_add(a.value,b.value),soft64_add(a.dx,b.dx),soft64_add(a.dy,b.dy));
}
fn d64_subtract(a: Dual64, b: Dual64) -> Dual64 {
  return Dual64(soft64_subtract(a.value,b.value),soft64_subtract(a.dx,b.dx),soft64_subtract(a.dy,b.dy));
}
fn d64_negate(a: Dual64) -> Dual64 {
  return Dual64(soft64_negate(a.value),soft64_negate(a.dx),soft64_negate(a.dy));
}
fn d64_multiply(a: Dual64, b: Dual64) -> Dual64 {
  return Dual64(soft64_multiply(a.value,b.value),
    soft64_add(soft64_multiply(a.dx,b.value),soft64_multiply(a.value,b.dx)),
    soft64_add(soft64_multiply(a.dy,b.value),soft64_multiply(a.value,b.dy)));
}
fn d64_divide(a: Dual64, b: Dual64) -> Dual64 {
  let quotient = soft64_divide(a.value,b.value);
  return Dual64(quotient,
    soft64_divide(soft64_subtract(a.dx,soft64_multiply(quotient,b.dx)),b.value),
    soft64_divide(soft64_subtract(a.dy,soft64_multiply(quotient,b.dy)),b.value));
}

/** Positive-argument square root; the zero-argument derivative requires a separate analytic limit. */
fn d64_sqrt(a: Dual64) -> Dual64 {
  if (!d64_valid(a) || soft64_zero(a.value)) { return d64_failure(); }
  let root = soft64_sqrt(a.value);
  let denominator = soft64_scale(root,1);
  return Dual64(root,soft64_divide(a.dx,denominator),soft64_divide(a.dy,denominator));
}
fn d64_scale(a: Dual64, power: i32) -> Dual64 {
  return Dual64(soft64_scale(a.value,power),soft64_scale(a.dx,power),soft64_scale(a.dy,power));
}
