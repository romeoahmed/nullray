/**
 * Normalize a quantized screen ray and construct its ZAMO photon and screen derivatives.
 * Shared observer metric values arrive as host binary64 words; only
 * direction-dependent operations use the integer arithmetic per ray.
 */
struct Dual64Launch {
  source: array<Dual64, 3>,
  energy: Dual64,
  angular_momentum: Dual64,
  carter: Dual64,
  polar_velocity: Dual64,
  /** Backward du/dtau = (future dr/dgamma) / r². */
  inverse_radial_velocity: Dual64,
}

fn d64_screen_launch(frame: array<u32, 24>, pixel: vec2u, observer: Observer64) -> Dual64Launch {
  return d64_screen_launch_precise(frame,array<Soft64,2>(soft64_from_f32_bits(pixel.x),soft64_from_f32_bits(pixel.y)),observer);
}

/** Differentiate a continuous binary64 image position with respect to physical screen pixels. */
fn d64_screen_launch_precise(frame: array<u32, 24>, pixel: array<Soft64,2>, observer: Observer64) -> Dual64Launch {
  let half = d64_constant(vec2u(0u, 0x3fe00000u));
  let width = d64_from_f32_bits(frame[0]);
  let height = d64_from_f32_bits(frame[1]);
  let zoom = d64_from_f32_bits(frame[6]);
  let x = d64_subtract(d64_add(d64_add(Dual64(pixel[0],soft64_number(1.0),vec2u(0u)), half),
    d64_from_f32_bits(frame[2])), d64_multiply(width, half));
  let y = d64_subtract(d64_add(d64_add(Dual64(pixel[1],vec2u(0u),soft64_number(1.0)), half),
    d64_from_f32_bits(frame[3])), d64_multiply(height, half));
  let sx = d64_multiply(d64_divide(x, height), zoom);
  let sy = d64_multiply(d64_divide(y, height), zoom);
  var direction: array<Dual64, 3>;
  var norm_squared = d64_constant(vec2u(0u));
  for (var axis = 0u; axis < 3u; axis++) {
    direction[axis] = d64_add(d64_subtract(d64_from_f32_bits(frame[12u + axis]),
      d64_multiply(sy, d64_from_f32_bits(frame[16u + axis]))),
      d64_multiply(sx, d64_from_f32_bits(frame[20u + axis])));
    norm_squared = d64_add(norm_squared, d64_multiply(direction[axis], direction[axis]));
  }
  let norm = d64_sqrt(norm_squared);
  for (var axis = 0u; axis < 3u; axis++) {
    direction[axis] = d64_divide(direction[axis], norm);
  }
  let rr = d64_constant(observer.radius2);
  let aa = d64_constant(observer.spin2);
  let cc = d64_constant(observer.cosine2);
  let sigma = d64_constant(observer.sigma);
  let big_a = d64_constant(observer.big_a);
  let lapse = d64_constant(observer.lapse);
  let dragging = d64_constant(observer.dragging);
  let sine = d64_constant(observer.sine);
  let l = d64_multiply(d64_multiply(d64_negate(direction[2]), d64_constant(observer.azimuth_scale)), sine);
  let e = d64_add(lapse, d64_multiply(dragging, l));
  var p = d64_multiply(d64_negate(direction[1]), d64_constant(observer.polar_scale));
  if ((frame[5] & 0x7fffffffu) == 0u || frame[5] == 0x40490fdbu) {
    var meridian = d64_sqrt(d64_add(d64_multiply(direction[1],direction[1]),d64_multiply(direction[2],direction[2])));
    if ((frame[5] & 0x7fffffffu) == 0u) { meridian = d64_negate(meridian); }
    p = d64_multiply(meridian,d64_constant(observer.polar_scale));
  }
  let c = d64_add(d64_subtract(d64_multiply(d64_multiply(direction[1], direction[1]), sigma),
    d64_multiply(d64_multiply(aa, d64_multiply(e, e)), cc)),
    d64_multiply(d64_divide(d64_multiply(d64_multiply(direction[2], direction[2]), big_a), sigma), cc));
  let velocity = d64_divide(d64_multiply(d64_negate(direction[0]), d64_constant(observer.radial_scale)), rr);
  return Dual64Launch(direction, e, l, c, p, velocity);
}
