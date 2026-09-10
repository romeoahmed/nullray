/**
 * Software binary64 core for critical-ray path preparation.
 * Lanes are the low/high IEEE-754 words. Integer arithmetic owns the result
 * and rounding; a native square-root estimate is corrected with exact integers.
 * Accepted values are normal finite numbers and zero. Nonzero subnormal inputs,
 * underflow, overflow, and nonfinite inputs produce the integer invalid sentinel;
 * callers must check soft64_valid before converting an optical result.
 */
alias Soft64 = vec2u;

const SOFT64_INVALID = vec2u(0u, 0x7ff80000u);

fn soft64_valid(x: Soft64) -> bool {
  let exponent = (x.y >> 20u) & 0x7ffu;
  return exponent < 2047u && (exponent != 0u || ((x.y & 0xfffffu) | x.x) == 0u);
}

fn soft64_zero(x: Soft64) -> bool {
  return ((x.y & 0x7fffffffu) | x.x) == 0u;
}

fn soft64_negate(x: Soft64) -> Soft64 {
  return vec2u(x.x, x.y ^ 0x80000000u);
}

fn soft64_words_add(a: vec2u, b: vec2u) -> vec2u {
  let low = a.x + b.x;
  return vec2u(low, a.y + b.y + u32(low < a.x));
}

fn soft64_words_subtract(a: vec2u, b: vec2u) -> vec2u {
  return vec2u(a.x - b.x, a.y - b.y - u32(a.x < b.x));
}

/** Right shift with all discarded bits accumulated into the least significant bit. */
fn soft64_shift_jam(x: vec2u, shift: u32) -> vec2u {
  if (shift == 0u) { return x; }
  if (shift < 32u) {
    let lost = x.x << (32u - shift);
    return vec2u((x.x >> shift) | (x.y << (32u - shift)) | u32(lost != 0u), x.y >> shift);
  }
  if (shift == 32u) { return vec2u(x.y | u32(x.x != 0u), 0u); }
  if (shift < 64u) {
    let lost = x.x | (x.y << (64u - shift));
    return vec2u((x.y >> (shift - 32u)) | u32(lost != 0u), 0u);
  }
  return vec2u(u32((x.x | x.y) != 0u), 0u);
}

/** Normalize a nonnegative significand with three rounding bits, then round to nearest even. */
fn soft64_pack(sign: u32, biased_exponent: i32, input: vec2u) -> Soft64 {
  if ((input.x | input.y) == 0u) { return vec2u(0u); }
  var significand = input;
  var exponent = biased_exponent;
  if (significand.y >= 0x1000000u) {
    significand = soft64_shift_jam(significand, 1u);
    exponent++;
  } else {
    let leading = select(32u + countLeadingZeros(significand.x), countLeadingZeros(significand.y), significand.y != 0u);
    let shift = leading - 8u;
    if (shift >= 32u) {
      significand = vec2u(0u, significand.x << (shift - 32u));
    } else if (shift > 0u) {
      significand = vec2u(significand.x << shift, (significand.y << shift) | (significand.x >> (32u - shift)));
    }
    exponent -= i32(shift);
  }
  if (exponent <= 0) {
    significand = soft64_shift_jam(significand, u32(1 - exponent));
    exponent = 1;
  }
  let remainder = significand.x & 7u;
  var rounded = vec2u((significand.x >> 3u) | (significand.y << 29u), significand.y >> 3u);
  if (remainder > 4u || (remainder == 4u && (rounded.x & 1u) != 0u)) {
    rounded = soft64_words_add(rounded, vec2u(1u, 0u));
  }
  if (rounded.y >= 0x200000u) {
    rounded = vec2u((rounded.x >> 1u) | (rounded.y << 31u), rounded.y >> 1u);
    exponent++;
  }
  if (rounded.y < 0x100000u || exponent >= 2047) { return SOFT64_INVALID; }
  return vec2u(rounded.x, sign | (u32(exponent) << 20u) | (rounded.y & 0xfffffu));
}

fn soft64_add(left: Soft64, right: Soft64) -> Soft64 {
  if (!soft64_valid(left) || !soft64_valid(right)) { return SOFT64_INVALID; }
  if (soft64_zero(left)) { return right; }
  if (soft64_zero(right)) { return left; }
  var a = left;
  var b = right;
  let a_high = a.y & 0x7fffffffu;
  let b_high = b.y & 0x7fffffffu;
  if (a_high < b_high || (a_high == b_high && a.x < b.x)) {
    a = right;
    b = left;
  }
  let exponent_a = i32((a.y >> 20u) & 0x7ffu);
  let exponent_b = i32((b.y >> 20u) & 0x7ffu);
  let mantissa_a = vec2u(a.x << 3u, (((a.y & 0xfffffu) | 0x100000u) << 3u) | (a.x >> 29u));
  let mantissa_b = vec2u(b.x << 3u, (((b.y & 0xfffffu) | 0x100000u) << 3u) | (b.x >> 29u));
  let aligned_b = soft64_shift_jam(mantissa_b, u32(exponent_a - exponent_b));
  var sum = vec2u(0u);
  if (((a.y ^ b.y) & 0x80000000u) == 0u) {
    sum = soft64_words_add(mantissa_a, aligned_b);
  } else {
    sum = soft64_words_subtract(mantissa_a, aligned_b);
  }
  return soft64_pack(a.y & 0x80000000u, exponent_a, sum);
}

fn soft64_subtract(a: Soft64, b: Soft64) -> Soft64 {
  return soft64_add(a, soft64_negate(b));
}

/** Exact 53-by-53-bit product in base 2^16; each multiply-add fits one u32. */
fn soft64_multiply(a: Soft64, b: Soft64) -> Soft64 {
  if (!soft64_valid(a) || !soft64_valid(b)) { return SOFT64_INVALID; }
  if (soft64_zero(a) || soft64_zero(b)) { return vec2u(0u); }
  let sign = (a.y ^ b.y) & 0x80000000u;
  var exponent = i32((a.y >> 20u) & 0x7ffu) + i32((b.y >> 20u) & 0x7ffu) - 1023;
  let ma = vec4u(a.x & 65535u, a.x >> 16u, a.y & 65535u, ((a.y >> 16u) & 15u) | 16u);
  let mb = vec4u(b.x & 65535u, b.x >> 16u, b.y & 65535u, ((b.y >> 16u) & 15u) | 16u);
  var limbs: array<u32, 8>;
  for (var i = 0u; i < 4u; i++) {
    var carry = 0u;
    for (var j = 0u; j < 4u; j++) {
      let product = ma[i] * mb[j] + limbs[i + j] + carry;
      limbs[i + j] = product & 65535u;
      carry = product >> 16u;
    }
    limbs[i + 4u] = carry;
  }
  let words = vec4u(limbs[0] | (limbs[1] << 16u), limbs[2] | (limbs[3] << 16u),
    limbs[4] | (limbs[5] << 16u), limbs[6] | (limbs[7] << 16u));
  var shift = 17u;
  if (words.w >= 512u) {
    shift = 18u;
    exponent++;
  }
  let lost = words.x | (words.y & ((1u << shift) - 1u));
  let extended = vec2u((words.y >> shift) | (words.z << (32u - shift)) | u32(lost != 0u),
    (words.z >> shift) | (words.w << (32u - shift)));
  return soft64_pack(sign, exponent, extended);
}

fn soft64_words_less(a: vec2u, b: vec2u) -> bool {
  return a.y < b.y || (a.y == b.y && a.x < b.x);
}

/** Exact product of a 53-bit significand and an eleven-bit quotient digit. */
fn soft64_digit_product(value: vec2u, digit: u32) -> vec2u {
  let lower = (value.x & 65535u) * digit;
  let upper = (value.x >> 16u) * digit + (lower >> 16u);
  return vec2u((lower & 65535u) | (upper << 16u), value.y * digit + (upper >> 16u));
}

/**
 * Radix-2048 division retains 53 significand bits, three rounding bits, and
 * the exact sticky remainder. Five digits replace 55 single-bit steps.
 * A normalized divisor has a high word >= 2^20; the high-word estimate of an
 * eleven-bit digit is exact or one too large. Correct against the full product.
 * The shifted remainder and digit product both fit in two unsigned words.
 */
fn soft64_divide(a: Soft64, b: Soft64) -> Soft64 {
  if (!soft64_valid(a) || !soft64_valid(b) || soft64_zero(b)) { return SOFT64_INVALID; }
  if (soft64_zero(a)) { return vec2u(0u); }
  let sign = (a.y ^ b.y) & 0x80000000u;
  var exponent = i32((a.y >> 20u) & 0x7ffu) - i32((b.y >> 20u) & 0x7ffu) + 1023;
  var remainder = vec2u(a.x, (a.y & 0xfffffu) | 0x100000u);
  let divisor = vec2u(b.x, (b.y & 0xfffffu) | 0x100000u);
  if (soft64_words_less(remainder, divisor)) {
    remainder = vec2u(remainder.x << 1u, (remainder.y << 1u) | (remainder.x >> 31u));
    exponent--;
  }
  remainder = soft64_words_subtract(remainder, divisor);
  var quotient = vec2u(1u, 0u);
  for (var group = 0u; group < 5u; group++) {
    remainder = vec2u(remainder.x << 11u, (remainder.y << 11u) | (remainder.x >> 21u));
    var digit = min(remainder.y / divisor.y, 2047u);
    var product = soft64_digit_product(divisor, digit);
    if (soft64_words_less(remainder, product)) {
      digit--;
      product = soft64_words_subtract(product, divisor);
    }
    remainder = soft64_words_subtract(remainder, product);
    quotient = vec2u((quotient.x << 11u) | digit, (quotient.y << 11u) | (quotient.x >> 21u));
  }
  quotient.x |= u32((remainder.x | remainder.y) != 0u);
  return soft64_pack(sign, exponent, quotient);
}

/** Two adjacent bits of an unsigned significand, with zero extension on either side. */
fn soft64_bit_pair(mantissa: vec2u, position: i32) -> u32 {
  if (position < -1 || position > 52) { return 0u; }
  if (position == -1) { return (mantissa.x & 1u) << 1u; }
  if (position < 31) { return (mantissa.x >> u32(position)) & 3u; }
  if (position == 31) { return ((mantissa.x >> 31u) | (mantissa.y << 1u)) & 3u; }
  return (mantissa.y >> u32(position - 32)) & 3u;
}

/** Exact square of a nonnegative 25-bit integer. */
fn soft64_prefix_square(value: u32) -> vec2u {
  let low = value & 65535u;
  let high = value >> 16u;
  let first = low * low;
  let middle = 2u * low * high + (first >> 16u);
  return vec2u((first & 65535u) | (middle << 16u), high * high + (middle >> 16u));
}

/**
 * Integer square root with a native estimate of the first 24 root bits.
 * Exact squared-integer comparisons correct that estimate before it enters the
 * remainder. The remaining 32 digit pairs and final sticky bit are exact;
 * native sqrt rounding therefore cannot affect the binary64 result.
 */
fn soft64_sqrt(a: Soft64) -> Soft64 {
  if (!soft64_valid(a)) { return SOFT64_INVALID; }
  if (soft64_zero(a)) { return vec2u(0u); }
  if ((a.y & 0x80000000u) != 0u) { return SOFT64_INVALID; }
  let exponent = i32((a.y >> 20u) & 0x7ffu) - 1023;
  let odd = exponent & 1;
  let mantissa = vec2u(a.x, (a.y & 0xfffffu) | 0x100000u);
  let shift = u32(6 - odd);
  let prefix = vec2u((mantissa.x >> shift) | (mantissa.y << (32u - shift)), mantissa.y >> shift);
  var estimate = u32(sqrt(f32(prefix.y) * 4294967296.0 + f32(prefix.x)));
  var square = soft64_prefix_square(estimate);
  loop {
    if (!soft64_words_less(prefix, square)) { break; }
    estimate--;
    square = soft64_prefix_square(estimate);
  }
  loop {
    let next = soft64_prefix_square(estimate + 1u);
    if (soft64_words_less(prefix, next)) { break; }
    estimate++;
    square = next;
  }
  var remainder = soft64_words_subtract(prefix, square);
  var root = vec2u(estimate, 0u);
  for (var bit = 31; bit >= 0; bit--) {
    let digit = soft64_bit_pair(mantissa, 2 * bit - 58 - odd);
    remainder = vec2u((remainder.x << 2u) | digit, (remainder.y << 2u) | (remainder.x >> 30u));
    let trial = vec2u((root.x << 2u) | 1u, (root.y << 2u) | (root.x >> 30u));
    let accepted = !soft64_words_less(remainder, trial);
    if (accepted) { remainder = soft64_words_subtract(remainder, trial); }
    root = vec2u((root.x << 1u) | u32(accepted), (root.y << 1u) | (root.x >> 31u));
  }
  root.x |= u32((remainder.x | remainder.y) != 0u);
  return soft64_pack(0u, (exponent - odd) / 2 + 1023, root);
}

/** Decode a finite binary32 bit pattern without evaluating floating-point arithmetic. */
fn soft64_from_f32_bits(bits: u32) -> Soft64 {
  let sign = bits & 0x80000000u;
  var exponent = i32((bits >> 23u) & 255u);
  var fraction = bits & 0x7fffffu;
  if (exponent == 255) { return SOFT64_INVALID; }
  if (exponent == 0) {
    if (fraction == 0u) { return vec2u(0u, sign); }
    let shift = countLeadingZeros(fraction) - 8u;
    fraction = (fraction << shift) & 0x7fffffu;
    exponent = 1 - i32(shift);
  }
  return vec2u(fraction << 29u, sign | (u32(exponent + 896) << 20u) | (fraction >> 3u));
}

/** Round to binary32 bits, preserving subnormal outputs; range errors return an integer NaN sentinel. */
fn soft64_to_f32_bits(a: Soft64) -> u32 {
  if (!soft64_valid(a)) { return 0x7fc00000u; }
  let sign = a.y & 0x80000000u;
  if (soft64_zero(a)) { return sign; }
  var exponent = i32((a.y >> 20u) & 0x7ffu) - 896;
  var extended = soft64_shift_jam(vec2u(a.x, (a.y & 0xfffffu) | 0x100000u), 26u);
  if (exponent <= 0) {
    extended = soft64_shift_jam(extended, u32(1 - exponent));
    exponent = 1;
  }
  let remainder = extended.x & 7u;
  var rounded = extended.x >> 3u;
  if (remainder > 4u || (remainder == 4u && (rounded & 1u) != 0u)) { rounded++; }
  if (rounded >= 0x1000000u) {
    rounded >>= 1u;
    exponent++;
  }
  if (rounded < 0x800000u) { exponent = 0; }
  if (exponent >= 255) { return 0x7fc00000u; }
  return sign | (u32(exponent) << 23u) | (rounded & 0x7fffffu);
}

fn soft64_abs(a: Soft64) -> Soft64 {
  return vec2u(a.x, a.y & 0x7fffffffu);
}

/** Ordered finite comparison; callers reject invalid operands first. */
fn soft64_less(a: Soft64, b: Soft64) -> bool {
  if (soft64_zero(a) && soft64_zero(b)) { return false; }
  let negative_a = (a.y & 0x80000000u) != 0u;
  let negative_b = (b.y & 0x80000000u) != 0u;
  if (negative_a != negative_b) { return negative_a; }
  if (negative_a) { return soft64_words_less(b, a); }
  return soft64_words_less(a, b);
}

/** Exact power-of-two scaling inside the normal finite arithmetic domain. */
fn soft64_scale(a: Soft64, power: i32) -> Soft64 {
  if (!soft64_valid(a)) { return SOFT64_INVALID; }
  if (soft64_zero(a)) { return a; }
  let exponent = i32((a.y >> 20u) & 2047u) + power;
  if (exponent <= 0 || exponent >= 2047) { return SOFT64_INVALID; }
  return vec2u(a.x, (a.y & 0x800fffffu) | (u32(exponent) << 20u));
}

/** Exact promotion of a normal finite native value; bit conversion handles the representation. */
fn soft64_number(a: f32) -> Soft64 {
  return soft64_from_f32_bits(bitcast<u32>(a));
}
