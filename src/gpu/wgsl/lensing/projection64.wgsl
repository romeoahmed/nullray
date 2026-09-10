/** Integer pixel base plus fractional jitter; keep absolute screen rounding out of the ray launch. */
struct CriticalPixel {
  base: vec2f,
  jitter: vec2f,
  coordinate: array<Soft64,2>,
  valid: bool,
}

fn critical_pixel_failure() -> CriticalPixel {
  return CriticalPixel(vec2f(0.0), vec2f(0.0), array<Soft64,2>(vec2u(0u),vec2u(0u)), false);
}

/** Cross products retain the exact binary32 camera components through binary64 arithmetic. */
fn critical_reciprocal_axis(first: vec3f, second: vec3f) -> array<Soft64,3> {
  var result: array<Soft64,3>;
  for (var axis = 0u; axis < 3u; axis++) {
    let next = (axis + 1u) % 3u;
    let last = (axis + 2u) % 3u;
    result[axis] = soft64_subtract(
      soft64_multiply(soft64_number(first[next]),soft64_number(second[last])),
      soft64_multiply(soft64_number(first[last]),soft64_number(second[next])));
  }
  return result;
}

/**
 * Perspective projection of a critical local direction displaced toward escape.
 * The input direction has host binary64 precision. Adding to its radial component
 * follows the curved critical strip; normalization cancels in the perspective ratios.
 */
fn critical_pixel(frame: Frame, direction: array<Soft64,3>, radial_offset: f32) -> CriticalPixel {
  return critical_pixel64(frame,direction,soft64_number(radial_offset));
}

fn critical_pixel64(frame: Frame, direction: array<Soft64,3>, radial_offset: Soft64) -> CriticalPixel {
  var local = direction;
  local[0] = soft64_add(local[0], radial_offset);
  // Quantization does not preserve exact orthonormality. Reciprocal basis
  // coefficients invert the actual launch; the common determinant cancels
  // from perspective ratios, but its sign still determines forward depth.
  let forward = critical_reciprocal_axis(frame.camera_right.xyz,frame.camera_up.xyz);
  let right = critical_reciprocal_axis(frame.camera_up.xyz,frame.camera_forward.xyz);
  let up = critical_reciprocal_axis(frame.camera_forward.xyz,frame.camera_right.xyz);
  var determinant = vec2u(0u);
  var depth = vec2u(0u);
  var horizontal = vec2u(0u);
  var vertical = vec2u(0u);
  for (var axis = 0u; axis < 3u; axis++) {
    depth = soft64_add(depth, soft64_multiply(local[axis], forward[axis]));
    determinant = soft64_add(determinant,soft64_multiply(soft64_number(frame.camera_forward[axis]),forward[axis]));
    horizontal = soft64_add(horizontal, soft64_multiply(local[axis], right[axis]));
    vertical = soft64_subtract(vertical, soft64_multiply(local[axis], up[axis]));
  }
  if (!soft64_valid(determinant) || soft64_zero(determinant)) { return critical_pixel_failure(); }
  if (soft64_less(determinant,vec2u(0u))) {
    depth = soft64_negate(depth);
    horizontal = soft64_negate(horizontal);
    vertical = soft64_negate(vertical);
  }
  if (!soft64_valid(depth) || !soft64_less(vec2u(0u), depth)) {
    return critical_pixel_failure();
  }
  let scale = soft64_divide(soft64_number(frame.viewport.y), soft64_number(frame.observer.z));
  let centered = array<Soft64,2>(soft64_multiply(soft64_divide(horizontal, depth), scale),
    soft64_multiply(soft64_divide(vertical, depth), scale));
  var base = vec2f(0.0);
  var jitter = vec2f(0.0);
  var coordinates: array<Soft64,2>;
  for (var axis = 0u; axis < 2u; axis++) {
    let coordinate = soft64_add(centered[axis],
      soft64_scale(soft64_subtract(soft64_number(frame.viewport[axis]), soft64_number(1.0)), -1));
    let bits = soft64_to_f32_bits(coordinate);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
      return critical_pixel_failure();
    }
    // The viewport itself is device-bounded. More distant projections cannot
    // retain a distinct integer pixel and are not usable search samples.
    let rounded = bitcast<f32>(bits);
    if (abs(rounded) >= 8388608.0) { return critical_pixel_failure(); }
    base[axis] = floor(rounded);
    let remainder = soft64_subtract(coordinate, soft64_number(base[axis]));
    let fraction_bits = soft64_to_f32_bits(remainder);
    if ((fraction_bits & 0x7f800000u) == 0x7f800000u) {
      return critical_pixel_failure();
    }
    jitter[axis] = bitcast<f32>(fraction_bits);
    coordinates[axis] = coordinate;
  }
  return CriticalPixel(base, jitter, coordinates, true);
}
