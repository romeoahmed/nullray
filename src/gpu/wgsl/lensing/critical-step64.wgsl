/** Rounded binary64 reciprocal factorials for bounded parameter updates. */
const CRITICAL_FACTORIALS = array<Soft64,21>(
  vec2u(0x00000000u,0x3ff00000u),
  vec2u(0x00000000u,0x3ff00000u),
  vec2u(0x00000000u,0x3fe00000u),
  vec2u(0x55555555u,0x3fc55555u),
  vec2u(0x55555555u,0x3fa55555u),
  vec2u(0x11111111u,0x3f811111u),
  vec2u(0x16c16c17u,0x3f56c16cu),
  vec2u(0x1a01a01au,0x3f2a01a0u),
  vec2u(0x1a01a01au,0x3efa01a0u),
  vec2u(0xa556c734u,0x3ec71de3u),
  vec2u(0xb7789f5cu,0x3e927e4fu),
  vec2u(0x67f544e4u,0x3e5ae645u),
  vec2u(0xeff8d898u,0x3e21eed8u),
  vec2u(0x13a86d09u,0x3de61246u),
  vec2u(0xa8c07c9du,0x3da93974u),
  vec2u(0xe733b81fu,0x3d6ae7f3u),
  vec2u(0xe733b81fu,0x3d2ae7f3u),
  vec2u(0x7030ad4au,0x3ce952c7u),
  vec2u(0x63b97d97u,0x3ca68278u),
  vec2u(0x46814157u,0x3c62f49bu),
  vec2u(0xa4020225u,0x3c1e542bu));

/**
 * Rotate the Carter circle and scale its radial offset without f32 accumulation.
 * For |angle|<=1/4 and |log scale|<=1, sine/cosine degrees 13/12 and
 * exponential degree 20 have truncation errors below 3e-19. Newton's
 * existing step bounds are smaller; unsupported inputs stay unresolved.
 */
fn critical_update64(parameters: array<Soft64,3>, step: vec2f) -> array<Soft64,3> {
  let failure = array<Soft64,3>(SOFT64_INVALID,SOFT64_INVALID,SOFT64_INVALID);
  let bits = bitcast<vec2u>(step);
  if (any((bits & vec2u(0x7f800000u)) == vec2u(0x7f800000u)) || any(abs(step)>vec2f(0.25,1.0))) { return failure; }
  for (var axis = 0u; axis < 3u; axis++) { if (!soft64_valid(parameters[axis])) { return failure; } }
  if (!soft64_less(vec2u(0u),parameters[2])) { return failure; }
  if (all(step == vec2f(0.0))) { return parameters; }
  let angle = soft64_number(step.x);
  let square = soft64_multiply(angle,angle);
  var cosine_tail = vec2u(0u);
  var sine_tail = vec2u(0u);
  for (var order = 6u; order > 0u; order--) {
    cosine_tail = soft64_subtract(CRITICAL_FACTORIALS[2u*order],soft64_multiply(square,cosine_tail));
    sine_tail = soft64_subtract(CRITICAL_FACTORIALS[2u*order+1u],soft64_multiply(square,sine_tail));
  }
  let one = soft64_number(1.0);
  let cosine = soft64_subtract(one,soft64_multiply(square,cosine_tail));
  let sine = soft64_multiply(angle,soft64_subtract(one,soft64_multiply(square,sine_tail)));
  var first = soft64_subtract(soft64_multiply(parameters[0],cosine),soft64_multiply(parameters[1],sine));
  var second = soft64_add(soft64_multiply(parameters[0],sine),soft64_multiply(parameters[1],cosine));
  let norm = soft64_sqrt(soft64_add(soft64_multiply(first,first),soft64_multiply(second,second)));
  if (!soft64_valid(norm) || soft64_less(soft64_number(1e-12),soft64_abs(soft64_subtract(norm,one)))) { return failure; }
  first = soft64_divide(first,norm);
  second = soft64_divide(second,norm);
  let exponent = soft64_number(step.y);
  var scale = CRITICAL_FACTORIALS[20];
  for (var order = 20u; order > 0u; order--) {
    scale = soft64_add(CRITICAL_FACTORIALS[order-1u],soft64_multiply(exponent,scale));
  }
  let offset = soft64_multiply(parameters[2],scale);
  if (!soft64_valid(first) || !soft64_valid(second) || !soft64_valid(offset) || !soft64_less(vec2u(0u),offset)) { return failure; }
  return array<Soft64,3>(first,second,offset);
}
