/** Implicit spherical-orbit derivative in the normalized Carter-circle angle. */
fn critical_direction_derivative64(frame: Frame, observer: Observer64, phase: f32) -> array<Dual64,3> {
  return critical_direction_circle_derivative64(frame,observer,critical_circle64(phase));
}

fn critical_direction_circle_derivative64(frame: Frame, observer: Observer64, circle: array<Soft64,2>) -> array<Dual64,3> {
  return critical_orbit_direction_derivative64(frame,observer,critical_orbit_circle64(frame,observer,circle));
}

/** Differentiate an accepted orbit without repeating its scalar radius solve. */
fn critical_orbit_direction_derivative64(frame: Frame, observer: Observer64, orbit: array<Soft64,3>) -> array<Dual64,3> {
  let failure = array<Dual64,3>(d64_failure(),d64_failure(),d64_failure());
  if (soft64_zero(orbit[0])) { return failure; }
  let zero = vec2u(0u);
  let unit = soft64_number(1.0);
  let one = d64_constant(unit);
  let a = d64_constant(soft64_number(frame.space.x));
  let q = d64_constant(soft64_number(frame.space.y));
  let aa = d64_multiply(a,a);
  let gap_squared = d64_subtract(d64_subtract(one,aa),d64_multiply(q,q));
  let observer_sine = d64_constant(observer.sine);
  let polar_spin2 = d64_multiply(aa,d64_constant(observer.cosine2));
  // Differentiate F(r,psi)=0 in independent r and psi directions, then
  // use dr/dpsi=-F_psi/F_r without differentiating the root-finding iterations.
  let independent_centered = Dual64(orbit[0],unit,zero);
  let independent_r = d64_add(one,independent_centered);
  let independent_cosine = Dual64(orbit[1],zero,soft64_negate(orbit[2]));
  let independent_delta = d64_subtract(d64_multiply(independent_centered,independent_centered),gap_squared);
  let equation = d64_subtract(d64_subtract(
    d64_multiply(d64_add(d64_multiply(independent_r,independent_r),polar_spin2),independent_centered),
    d64_scale(d64_multiply(independent_r,independent_delta),1)),
    d64_scale(d64_multiply(d64_multiply(d64_multiply(a,observer_sine),independent_cosine),
      d64_multiply(independent_r,d64_sqrt(independent_delta))),1));
  if (!d64_valid(equation) || soft64_zero(equation.dx)) { return failure; }
  let rm1 = Dual64(orbit[0],soft64_negate(soft64_divide(equation.dy,equation.dx)),zero);
  let r = d64_add(one,rm1);
  let cosine = Dual64(orbit[1],soft64_negate(orbit[2]),zero);
  let sine = Dual64(orbit[2],orbit[1],zero);
  let delta = d64_subtract(d64_multiply(rm1,rm1),gap_squared);
  let root_k = d64_divide(d64_scale(d64_multiply(r,d64_sqrt(delta)),1),rm1);
  let azimuth = d64_add(d64_multiply(a,observer_sine),d64_multiply(root_k,cosine));
  let lambda = d64_multiply(observer_sine,azimuth);
  let energy = d64_divide(d64_subtract(one,d64_multiply(d64_constant(observer.dragging),lambda)),d64_constant(observer.lapse));
  let observer_r = d64_constant(soft64_number(frame.observer.x));
  let sum = d64_add(observer_r,r);
  let factor = d64_subtract(d64_multiply(sum,sum),
    d64_divide(d64_scale(d64_multiply(r,delta),2),d64_multiply(rm1,rm1)));
  if (!soft64_less(zero,energy.value) || !soft64_less(zero,factor.value)) { return failure; }
  var direction = array<Dual64,3>(
    d64_divide(d64_multiply(d64_subtract(rm1,d64_subtract(observer_r,one)),d64_sqrt(factor)),d64_multiply(d64_constant(observer.radial_scale),energy)),
    d64_negate(d64_divide(d64_multiply(root_k,sine),d64_multiply(d64_constant(observer.polar_scale),energy))),
    d64_negate(d64_divide(azimuth,d64_multiply(d64_constant(observer.azimuth_scale),energy))));
  var square = d64_constant(zero);
  for (var axis = 0u; axis < 3u; axis++) { square = d64_add(square,d64_multiply(direction[axis],direction[axis])); }
  let norm = d64_sqrt(square);
  if (!d64_valid(norm) || soft64_less(soft64_number(1e-12),soft64_abs(soft64_subtract(norm.value,unit)))) { return failure; }
  for (var axis = 0u; axis < 3u; axis++) { direction[axis] = d64_divide(direction[axis],norm); }
  return direction;
}

/** Screen derivatives in critical angle and logarithmic radial displacement. */
fn critical_screen_derivative64(frame: Frame, direction: array<Dual64,3>, offset: Soft64) -> array<Dual64,2> {
  var local = direction;
  local[0] = d64_add(local[0],Dual64(offset,vec2u(0u),offset));
  let forward = critical_reciprocal_axis(frame.camera_right.xyz,frame.camera_up.xyz);
  let right = critical_reciprocal_axis(frame.camera_up.xyz,frame.camera_forward.xyz);
  let up = critical_reciprocal_axis(frame.camera_forward.xyz,frame.camera_right.xyz);
  var depth = d64_constant(vec2u(0u));
  var horizontal = d64_constant(vec2u(0u));
  var vertical = d64_constant(vec2u(0u));
  for (var axis = 0u; axis < 3u; axis++) {
    depth = d64_add(depth,d64_multiply(local[axis],d64_constant(forward[axis])));
    horizontal = d64_add(horizontal,d64_multiply(local[axis],d64_constant(right[axis])));
    vertical = d64_subtract(vertical,d64_multiply(local[axis],d64_constant(up[axis])));
  }
  let scale = d64_constant(soft64_divide(soft64_number(frame.viewport.y),soft64_number(frame.observer.z)));
  return array<Dual64,2>(d64_multiply(d64_divide(horizontal,depth),scale),d64_multiply(d64_divide(vertical,depth),scale));
}

/** Rotate parameter tangents before f32 rounding; do not subtract large restored screen gradients. */
fn critical_parameter_geometry(prepared: PreparedSkyBeam, screen: array<Dual64,2>) -> vec4f {
  var x = screen[0];
  var y = screen[1];
  if (prepared.refined) {
    let first = d64_add(d64_multiply(d64_constant(prepared.basis.xy),x),d64_multiply(d64_constant(prepared.basis.zw),y));
    let second = d64_subtract(d64_multiply(d64_constant(prepared.basis.xy),y),d64_multiply(d64_constant(prepared.basis.zw),x));
    x = first;
    y = second;
  }
  if (!d64_valid(x) || !d64_valid(y)) { return vec4f(0.0); }
  let bits = vec4u(soft64_to_f32_bits(x.dx),soft64_to_f32_bits(y.dx),soft64_to_f32_bits(x.dy),soft64_to_f32_bits(y.dy));
  if (any((bits & vec4u(0x7f800000u)) == vec4u(0x7f800000u))) { return vec4f(0.0); }
  return bitcast<vec4f>(bits);
}
