/** Scalar spherical-orbit condition in the regular Carter-circle parameter. */
fn critical_equation64(centered: Soft64, gap_squared: Soft64, ac: Soft64, polar_spin2: Soft64) -> Soft64 {
  let r = soft64_add(soft64_number(1.0),centered);
  let delta = soft64_subtract(soft64_multiply(centered,centered),gap_squared);
  let first = soft64_multiply(soft64_add(soft64_multiply(r,r),polar_spin2),centered);
  let second = soft64_scale(soft64_multiply(r,delta),1);
  let third = soft64_scale(soft64_multiply(soft64_multiply(ac,r),soft64_sqrt(delta)),1);
  return soft64_subtract(soft64_subtract(first,second),third);
}

/** Normalize f32 trigonometric values in binary64; their rounding moves only along the circle. */
fn critical_circle64(phase: f32) -> array<Soft64,2> {
  let raw_cosine = soft64_number(cos(phase));
  let raw_sine = soft64_number(sin(phase));
  let phase_norm = soft64_sqrt(soft64_add(soft64_multiply(raw_cosine,raw_cosine),soft64_multiply(raw_sine,raw_sine)));
  return array<Soft64,2>(soft64_divide(raw_cosine,phase_norm),soft64_divide(raw_sine,phase_norm));

}

fn critical_orbit64(frame: Frame, observer: Observer64, phase: f32) -> array<Soft64,3> {
  return critical_orbit_circle64(frame,observer,critical_circle64(phase));
}

/** Prepare (radius−1)/cosine/sine; zero in the first pair marks unresolved work. */
fn critical_orbit_circle64(frame: Frame, observer: Observer64, circle: array<Soft64,2>) -> array<Soft64,3> {
  let failure = array<Soft64,3>(vec2u(0u),vec2u(0u),vec2u(0u));
  let one = soft64_number(1.0);
  let a = soft64_number(frame.space.x);
  let q = soft64_number(frame.space.y);
  let aa = soft64_multiply(a,a);
  let qq = soft64_multiply(q,q);
  let gap_squared = soft64_subtract(soft64_subtract(one,aa),qq);
  let gap = soft64_sqrt(gap_squared);
  let cosine = circle[0];
  let sine = circle[1];
  let ac = soft64_multiply(soft64_multiply(a,observer.sine),cosine);
  let polar_spin2 = soft64_multiply(aa,observer.cosine2);
  var lower = gap;
  if (soft64_less(soft64_multiply(lower,lower),gap_squared)) {
    // A rounded horizon can lie inside the exact quadratic's exterior domain.
    let low_word = lower.x+1u;
    lower = vec2u(low_word,lower.y+select(0u,1u,low_word == 0u));
  }
  var upper = soft64_number(3.0);
  var lower_weight = critical_equation64(lower,gap_squared,ac,polar_spin2);
  var upper_weight = critical_equation64(upper,gap_squared,ac,polar_spin2);
  if (!soft64_valid(lower_weight) || !soft64_valid(upper_weight) ||
      !soft64_less(vec2u(0u),lower_weight) || !soft64_less(upper_weight,vec2u(0u))) { return failure; }
  var previous_side = 0u;
  var converged = false;
  for (var iteration = 0u; iteration < 128u; iteration++) {
    let width = soft64_subtract(upper,lower);
    let midpoint = soft64_add(lower,soft64_scale(width,-1));
    if (all(midpoint == lower) || all(midpoint == upper)) { converged = true; break; }
    // Illinois weights affect the interpolation proposal, never the bracket signs.
    let denominator = soft64_subtract(lower_weight,upper_weight);
    if (!soft64_valid(denominator) || !soft64_less(vec2u(0u),denominator)) { return failure; }
    var middle = soft64_add(lower,soft64_multiply(width,soft64_divide(lower_weight,denominator)));
    if (!soft64_valid(middle)) { return failure; }
    // The centered radius is positive. Move an endpoint-rounded proposal one binary64 value
    // inward; the midpoint check above already excluded adjacent endpoints.
    if (!soft64_less(lower,middle)) {
      let low_word = lower.x+1u;
      middle = vec2u(low_word,lower.y+select(0u,1u,low_word == 0u));
    }
    if (!soft64_less(middle,upper)) {
      middle = vec2u(upper.x-1u,upper.y-select(0u,1u,upper.x == 0u));
    }
    let value = critical_equation64(middle,gap_squared,ac,polar_spin2);
    if (!soft64_valid(value)) { return failure; }
    if (soft64_less(vec2u(0u),value)) {
      lower = middle;
      lower_weight = value;
      if (previous_side == 1u) { upper_weight = soft64_scale(upper_weight,-1); }
      previous_side = 1u;
    } else {
      upper = middle;
      upper_weight = value;
      if (previous_side == 2u) { lower_weight = soft64_scale(lower_weight,-1); }
      previous_side = 2u;
    }
  }
  if (!converged) { return failure; }
  let centered = soft64_add(lower,soft64_scale(soft64_subtract(upper,lower),-1));
  return array<Soft64,3>(centered,cosine,sine);
}

/** Convert a spherical orbit to the local critical direction; the fourth pair marks success. */
fn critical_direction64(frame: Frame, observer: Observer64, phase: f32) -> array<Soft64,4> {
  return critical_orbit_direction64(frame,observer,critical_orbit64(frame,observer,phase));
}

/** Reuse a prepared orbit when only the scalar critical direction is needed. */
fn critical_orbit_direction64(frame: Frame, observer: Observer64, orbit: array<Soft64,3>) -> array<Soft64,4> {
  let failure = array<Soft64,4>(vec2u(0u),vec2u(0u),vec2u(0u),vec2u(0u));
  if (soft64_zero(orbit[0])) { return failure; }
  let rm1 = orbit[0];
  let r = soft64_add(soft64_number(1.0),rm1);
  let cosine = orbit[1];
  let sine = orbit[2];
  let one = soft64_number(1.0);
  let a = soft64_number(frame.space.x);
  let q = soft64_number(frame.space.y);
  let gap_squared = soft64_subtract(soft64_subtract(one,soft64_multiply(a,a)),soft64_multiply(q,q));
  let delta = soft64_subtract(soft64_multiply(rm1,rm1),gap_squared);
  let root_k = soft64_divide(soft64_scale(soft64_multiply(r,soft64_sqrt(delta)),1),rm1);
  let azimuth = soft64_add(soft64_multiply(a,observer.sine),soft64_multiply(root_k,cosine));
  let lambda = soft64_multiply(observer.sine,azimuth);
  let energy = soft64_divide(soft64_subtract(one,soft64_multiply(observer.dragging,lambda)),observer.lapse);
  let observer_r = soft64_number(frame.observer.x);
  let sum = soft64_add(observer_r,r);
  let factor = soft64_subtract(soft64_multiply(sum,sum),
    soft64_divide(soft64_scale(soft64_multiply(r,delta),2),soft64_multiply(rm1,rm1)));
  if (!soft64_less(vec2u(0u),energy) || !soft64_less(vec2u(0u),factor)) { return failure; }
  var direction = array<Soft64,3>(
    soft64_divide(soft64_multiply(soft64_subtract(rm1,soft64_subtract(observer_r,one)),soft64_sqrt(factor)),soft64_multiply(observer.radial_scale,energy)),
    soft64_negate(soft64_divide(soft64_multiply(root_k,sine),soft64_multiply(observer.polar_scale,energy))),
    soft64_negate(soft64_divide(azimuth,soft64_multiply(observer.azimuth_scale,energy))));
  var square = vec2u(0u);
  for (var axis = 0u; axis < 3u; axis++) { square = soft64_add(square,soft64_multiply(direction[axis],direction[axis])); }
  let norm = soft64_sqrt(square);
  if (!soft64_valid(norm) || soft64_less(soft64_number(1e-12),soft64_abs(soft64_subtract(norm,one)))) { return failure; }
  for (var axis = 0u; axis < 3u; axis++) { direction[axis] = soft64_divide(direction[axis],norm); }
  return array<Soft64,4>(direction[0],direction[1],direction[2],one);
}
