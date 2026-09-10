/**
 * Normalize a quantized screen ray and construct its ZAMO photon.
 * Shared observer metric values arrive as host binary64 words; only
 * direction-dependent operations use the integer arithmetic per ray.
 */
struct Soft64Launch {
  source: array<Soft64, 3>,
  energy: Soft64,
  angular_momentum: Soft64,
  carter: Soft64,
  polar_velocity: Soft64,
  /** Backward du/dtau = (future dr/dgamma) / r². */
  inverse_radial_velocity: Soft64,
}

fn soft64_screen_launch(frame: array<u32, 24>, pixel: vec2u, observer: Observer64) -> Soft64Launch {
  return soft64_screen_launch_precise(frame,array<Soft64,2>(soft64_from_f32_bits(pixel.x),soft64_from_f32_bits(pixel.y)),observer);
}

/** Continuous image positions keep both IEEE words through normalization and launch. */
fn soft64_screen_launch_precise(frame: array<u32, 24>, pixel: array<Soft64,2>, observer: Observer64) -> Soft64Launch {
  let half = vec2u(0u, 0x3fe00000u);
  let width = soft64_from_f32_bits(frame[0]);
  let height = soft64_from_f32_bits(frame[1]);
  let zoom = soft64_from_f32_bits(frame[6]);
  let x = soft64_subtract(soft64_add(soft64_add(pixel[0], half),
    soft64_from_f32_bits(frame[2])), soft64_multiply(width, half));
  let y = soft64_subtract(soft64_add(soft64_add(pixel[1], half),
    soft64_from_f32_bits(frame[3])), soft64_multiply(height, half));
  let sx = soft64_multiply(soft64_divide(x, height), zoom);
  let sy = soft64_multiply(soft64_divide(y, height), zoom);
  var direction: array<Soft64, 3>;
  var norm_squared = vec2u(0u);
  for (var axis = 0u; axis < 3u; axis++) {
    direction[axis] = soft64_add(soft64_subtract(soft64_from_f32_bits(frame[12u + axis]),
      soft64_multiply(sy, soft64_from_f32_bits(frame[16u + axis]))),
      soft64_multiply(sx, soft64_from_f32_bits(frame[20u + axis])));
    norm_squared = soft64_add(norm_squared, soft64_multiply(direction[axis], direction[axis]));
  }
  let norm = soft64_sqrt(norm_squared);
  for (var axis = 0u; axis < 3u; axis++) {
    direction[axis] = soft64_divide(direction[axis], norm);
  }
  let rr = observer.radius2;
  let aa = observer.spin2;
  let cc = observer.cosine2;
  let sigma = observer.sigma;
  let big_a = observer.big_a;
  let lapse = observer.lapse;
  let dragging = observer.dragging;
  let sine = observer.sine;
  let l = soft64_multiply(soft64_multiply(soft64_negate(direction[2]), observer.azimuth_scale), sine);
  let e = soft64_add(lapse, soft64_multiply(dragging, l));
  var p = soft64_multiply(soft64_negate(direction[1]), observer.polar_scale);
  if ((frame[5] & 0x7fffffffu) == 0u || frame[5] == 0x40490fdbu) {
    var meridian = soft64_sqrt(soft64_add(soft64_multiply(direction[1],direction[1]),soft64_multiply(direction[2],direction[2])));
    if ((frame[5] & 0x7fffffffu) == 0u) { meridian = soft64_negate(meridian); }
    p = soft64_multiply(meridian,observer.polar_scale);
  }
  let c = soft64_add(soft64_subtract(soft64_multiply(soft64_multiply(direction[1], direction[1]), sigma),
    soft64_multiply(soft64_multiply(aa, soft64_multiply(e, e)), cc)),
    soft64_multiply(soft64_divide(soft64_multiply(soft64_multiply(direction[2], direction[2]), big_a), sigma), cc));
  let velocity = soft64_divide(soft64_multiply(soft64_negate(direction[0]), observer.radial_scale), rr);
  return Soft64Launch(direction, e, l, c, p, velocity);
}
