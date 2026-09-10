/**
 * Ordered source events using integer binary64 preparation and the f32 evaluator.
 * Keep this precision path separate from trace_ray: sharing its event function
 * changed native endpoint accuracy on critical-event-order.json. See numerics.md.
 */
fn screen_ray64(frame: Frame, pixel: vec2f, observer_frame: Observer64) -> RayEndpoint {
  return screen_ray_precise(frame,array<Soft64,2>(soft64_number(pixel.x),soft64_number(pixel.y)),observer_frame);
}

/** Same ordered solver with a continuous binary64 screen position. */
fn screen_ray_precise(frame: Frame, pixel: array<Soft64,2>, observer_frame: Observer64) -> RayEndpoint {
  let lanes = array<vec4f,6>(frame.viewport,frame.observer,frame.space,frame.camera_forward,frame.camera_up,frame.camera_right);
  var bits: array<u32,24>;
  for (var lane = 0u; lane < 6u; lane++) {
    for (var component = 0u; component < 4u; component++) { bits[lane*4u+component] = bitcast<u32>(lanes[lane][component]); }
  }
  let photon = soft64_screen_launch_precise(bits,pixel,observer_frame);
  let a64 = soft64_from_f32_bits(bits[8]);
  let q64 = soft64_from_f32_bits(bits[9]);
  let aa = soft64_multiply(a64,a64);
  let qq = soft64_multiply(q64,q64);
  let ae = soft64_multiply(a64,photon.energy);
  let a2e2 = soft64_multiply(ae,ae);
  let difference = soft64_subtract(photon.angular_momentum,ae);
  let k = soft64_add(soft64_multiply(difference,difference),photon.carter);
  let quadratic64 = soft64_subtract(soft64_subtract(a2e2,soft64_multiply(photon.angular_momentum,photon.angular_momentum)),photon.carter);
  let coefficients64 = array<Soft64,5>(soft64_multiply(photon.energy,photon.energy),vec2u(0u),quadratic64,
    soft64_scale(k,1),soft64_negate(soft64_add(soft64_multiply(aa,photon.carter),soft64_multiply(qq,k))));
  let radial64 = soft64_prepare_quartic(coefficients64,
    soft64_divide(soft64_number(1.0),soft64_from_f32_bits(bits[4])),photon.inverse_radial_velocity);
  if (!radial64.valid) { return RayEndpoint(unresolved(17u),0.0); }
  let constants64 = array<Soft64,9>(photon.energy,photon.angular_momentum,photon.carter,photon.polar_velocity,
    coefficients64[0],coefficients64[2],coefficients64[3],coefficients64[4],photon.source[1]);
  var constants: array<f32,9>;
  for (var index = 0u; index < 9u; index++) {
    let value = soft64_to_f32_bits(constants64[index]);
    if ((value & 0x7f800000u) == 0x7f800000u) { return RayEndpoint(unresolved(17u),0.0); }
    constants[index] = bitcast<f32>(value);
  }
  let a = frame.space.x;
  let q = frame.space.y;
  let theta = frame.observer.y;
  let e = constants[0];
  let l = constants[1];
  let c = constants[2];
  let p_theta = constants[3];
  let radial = radial64.path;
  var launch_phi = frame.observer.w;
  if (theta == 1.5707963268 && soft64_zero(photon.source[1])) { return RayEndpoint(unresolved(1u),0.0); }
  if (theta == 0.0 || theta == 3.1415926536) {
    let z_bits = soft64_to_f32_bits(photon.source[2]);
    if ((z_bits & 0x7f800000u) == 0x7f800000u) { return RayEndpoint(unresolved(17u),0.0); }
    let yz = vec2f(constants[8],bitcast<f32>(z_bits));
    if (length(yz) > 0.0) { launch_phi += atan2(yz.y,select(-yz.x,yz.x,theta == 0.0)); }
  }
  let equator = equator_events(a, e, l, c, theta, p_theta);
  if (equator.status == 2u) { return RayEndpoint(unresolved(2u), 0.0); }
  // The analytic branch reads only the initial latitude from this general-path record.
  var polar = QuarticPath();
  polar.x = bitcast<f32>(soft64_to_f32_bits(observer_frame.cosine));
  if (equator.status != 1u) {
    let polar64 = soft64_prepare_quartic(array<Soft64,5>(photon.carter,vec2u(0u),quadratic64,vec2u(0u),soft64_negate(a2e2)),
      observer_frame.cosine,soft64_multiply(observer_frame.sine,photon.polar_velocity));
    if (!polar64.valid) { return RayEndpoint(unresolved(17u),0.0); }
    polar = polar64.path;
  }
  let horizon_inverse = 1.0 / (1.0 + sqrt(1.0 - a * a - q * q));
  let horizon64 = soft64_divide(soft64_number(1.0),soft64_add(soft64_number(1.0),soft64_sqrt(soft64_subtract(soft64_subtract(soft64_number(1.0),aa),qq))));
  let destination = soft64_radial_destination(coefficients64,soft64_divide(soft64_number(1.0),soft64_from_f32_bits(bits[4])),photon.inverse_radial_velocity,horizon64);
  if (destination == 0u) { return RayEndpoint(unresolved(14u), 0.0); }
  if (destination == 1u && e <= 0.0) { return RayEndpoint(unresolved(9u), 0.0); }
  let boundary_value = select(horizon_inverse, 0.0, destination == 1u);
  let boundary_velocity = select(abs(e + (a * a * e - a * l) * horizon_inverse * horizon_inverse), -abs(e), destination == 1u);
  let arrival = quartic_arrival(radial, boundary_value, boundary_velocity);
  if (arrival.status != 1u) { return RayEndpoint(vec4f(15.0, arrival.lower, arrival.upper, -2.0), 0.0); }
  let boundary_time = (arrival.lower + arrival.upper) / 2.0;
  // All candidate disk crossings share this radial modulus and its AGM descent.
  let radial_plan = prepare_jacobi_pair(radial.elliptic.m,radial.elliptic.complement);
  var next_crossing = equator.first;
  var disk_order = 0u;
  for (var event = 0u; event < 64u && equator.status == 1u && next_crossing <= arrival.upper; event++) {
    let time = next_crossing;
    if (time >= arrival.lower) { return RayEndpoint(unresolved(16u), 0.0); }
    let order = disk_order;
    disk_order++;
    next_crossing += equator.spacing;
    let hit = evaluate_quartic_prepared(radial, time, radial_plan);
    if (!hit.valid) { return RayEndpoint(unresolved(6u), 0.0); }
    if (hit.value >= 1.0 / frame.space.w && hit.value <= 1.0 / frame.space.z) {
      let radius = 1.0 / hit.value;
      let frequency = disk_frequency(a, q, e, l, radius);
      let mapped = integrate_transport_budget(radial, polar, a, q, e, l, c, time, true, equator,1024u);
      if (frequency.x <= 0.0) { return RayEndpoint(unresolved(7u), 0.0); }
      if (!mapped.valid) { return RayEndpoint(unresolved(8u), 0.0); }
      return RayEndpoint(vec4f(radius, launch_phi + mapped.value.x, frequency.x, -3.0 - f32(order)), mapped.value.y);
    }
  }
  if (equator.status == 1u && next_crossing <= arrival.upper) { return RayEndpoint(unresolved(12u), 0.0); }
  if (destination == 1u) {
    if ((arrival.upper - arrival.lower) * sqrt(abs(c) + l * l + a * a * e * e) > 4e-5) {
      return RayEndpoint(vec4f(15.0, arrival.lower, arrival.upper, -2.0), 0.0);
    }
    let mapped = integrate_transport_budget(radial, polar, a, q, e, l, c, boundary_time, false, equator,1024u);
    // Positive-C motion shares the elliptic cosine used for equator events
    // and beam derivatives; avoid a second rounded quartic representation.
    var sky_mu = Evaluation(0.0,false);
    if (equator.status == 1u) {
      let j = jacobi(equator.phase + equator.frequency * boundary_time,equator.m);
      sky_mu = Evaluation(sqrt(equator.amplitude2)*j.y,true);
    } else { sky_mu = evaluate_quartic(polar,boundary_time); }
    if (!mapped.valid) { return RayEndpoint(unresolved(10u), 0.0); }
    if (!sky_mu.valid || abs(sky_mu.value) > 1.0) { return RayEndpoint(unresolved(11u), 0.0); }
    var branch = 1.0;
    if (equator.status == 1u && boundary_time >= equator.first) {
      branch += 1.0 + floor((boundary_time - equator.first) / equator.spacing);
    }
    // Positive alpha identifies the sky equator-crossing branch.
    return RayEndpoint(vec4f(sky_mu.value, launch_phi + mapped.value.x, e, branch), 0.0);
  }
  return RayEndpoint(vec4f(0.0), 0.0);
}

/**
 * Select extra precision near a repeated radial root. This dimensionless 4%
 * discriminant band is a work-selection policy, not an endpoint
 * error bound. Coefficient scaling avoids overflow in the invariant products.
 */
fn needs_radial_precision(frame: Frame, source: vec3f) -> bool {
  let a = frame.space.x;
  let q = frame.space.y;
  let r = frame.observer.x;
  let mu = cos(frame.observer.y);
  let sine = sin(frame.observer.y);
  let sigma = r*r+a*a*mu*mu;
  let delta = r*r-2.0*r+a*a+q*q;
  let big_a = (r*r+a*a)*(r*r+a*a)-a*a*delta*sine*sine;
  let l = -source.z*sqrt(big_a/sigma)*sine;
  let e = sqrt(sigma*delta/big_a)+a*(2.0*r-q*q)/big_a*l;
  let c = source.y*source.y*sigma-a*a*e*e*mu*mu+source.z*source.z*big_a/sigma*mu*mu;
  let k = (l-a*e)*(l-a*e)+c;
  let coefficients = vec4f(e*e,a*a*e*e-l*l-c,2.0*k,-a*a*c-q*q*k);
  let scale = max(max(abs(coefficients.x),abs(coefficients.y)),max(abs(coefficients.z),abs(coefficients.w)));
  if (scale <= 0.0) { return true; }
  let v = coefficients/scale;
  let g2 = v.w*v.x+v.y*v.y/12.0;
  let g3 = v.w*v.y*v.x/6.0-v.y*v.y*v.y/216.0-v.z*v.z*v.x/16.0;
  let first = g2*g2*g2;
  let second = 27.0*g3*g3;
  // The negative-g3 repeated-root limit has m approaching one. The positive
  // limit has m approaching zero and does not have the same critical phase.
  return g3 < 0.0 && abs(first-second) <= 0.04*(abs(first)+second);
}

fn screen_ray_refined(frame: Frame, pixel: vec2f, observer_frame: Observer64) -> RayEndpoint {
  let screen = (pixel+0.5+frame.viewport.zw-frame.viewport.xy/2.0)/frame.viewport.y;
  let direction = normalize(frame.camera_forward.xyz-screen.y*frame.observer.z*frame.camera_up.xyz
    +screen.x*frame.observer.z*frame.camera_right.xyz);
  if (needs_radial_precision(frame,direction)) { return screen_ray64(frame,pixel,observer_frame); }
  return screen_ray(frame,pixel);
}
