struct EquatorEvents {
  first: f32,
  spacing: f32,
  // 0: no transverse crossing; 1: periodic crossings; 2: unresolved.
  status: u32,
  phase: f32,
  frequency: f32,
  m: f32,
  amplitude2: f32,
  azimuth_phase: f32,
}

/**
 * Keep the unresolved state distinct from a valid orbit with no crossings.
 */
fn unresolved_equator() -> EquatorEvents {
  var result: EquatorEvents;
  result.status = 2u;
  return result;
}

/**
 * Positive-C motion is an elliptic cosine. Recover its signed phase from both
 * position and momentum, then enumerate zeros without stepping the trajectory.
 */
fn equator_events(a: f32, e: f32, l: f32, c: f32, theta: f32, p_theta: f32) -> EquatorEvents {
  if (c <= 0.0) { return EquatorEvents(); }
  let a2e2 = a * a * e * e;
  let b = l * l + c - a2e2;
  var frequency2 = c + a2e2;
  if (l != 0.0) { frequency2 = sqrt(b * b + 4.0 * a2e2 * c); }
  if (frequency2 <= 0.0) { return unresolved_equator(); }
  var amplitude2 = 0.0;
  if (l == 0.0) {
    amplitude2 = 1.0;
  } else if (b >= 0.0) {
    amplitude2 = 2.0 * c / (b + frequency2);
  } else {
    if (a2e2 <= 0.0) { return unresolved_equator(); }
    amplitude2 = (frequency2 - b) / (2.0 * a2e2);
  }
  if (amplitude2 <= 0.0) { return unresolved_equator(); }
  let m = a2e2 * amplitude2 / frequency2;
  let cosine = cos(theta) / sqrt(amplitude2);
  // Sixteen f32 ulps permit accumulated initialization roundoff at a turn.
  if (m < 0.0 || m >= 1.0 || abs(cosine) > 1.000001907) {
    return unresolved_equator();
  }
  // F(phi|0)=phi and K(0)=pi/2. Spherical spacetimes use elementary
  // phase recovery, with no duplicated Carlson iteration or inverse trig.
  var complete = Evaluation(1.5707963268, true);
  if (m != 0.0) { complete = carlson_rf(vec3f(0.0, 1.0 - m, 1.0)); }
  if (!complete.valid) { return unresolved_equator(); }
  let cn = clamp(cosine, -1.0, 1.0);
  let sin_theta = select(sin(theta), 0.0, theta == 0.0 || theta == 3.1415926536);
  let sn = abs(sin_theta * p_theta) / sqrt(amplitude2 * frequency2 * (1.0 - m + m * cn * cn));
  let angle = atan2(sn, cn);
  var phase = angle;
  if (m != 0.0) {
    let acute = min(angle, 3.141592654 - angle);
    let s = sin(acute);
    let co = cos(acute);
    let incomplete = carlson_rf(vec3f(co * co, 1.0 - m * s * s, 1.0));
    if (!incomplete.valid) { return unresolved_equator(); }
    phase = s * incomplete.value;
    if (angle > 1.570796327) { phase = 2.0 * complete.value - phase; }
  }
  if (p_theta > 0.0) { phase = -phase; }
  var azimuth_phase = 0.0;
  if (l == 0.0) {
    azimuth_phase = 3.1415926536 * (floor(phase / (2.0 * complete.value)) + 0.5);
    if (theta == 0.0) {
      azimuth_phase = 1.5707963268;
    } else if (theta == 3.1415926536) {
      azimuth_phase = select(4.7123889804, -1.5707963268, phase < 0.0);
    }
  } else {
    let complement = l * l * amplitude2 / (c + a2e2 * amplitude2);
    if (complement == 0.0 && sn == 0.0) {
      return unresolved_equator();
    }
    azimuth_phase = atan2(select(sn, -sn, p_theta > 0.0), sqrt(complement) * cn);
  }
  let frequency = sqrt(frequency2);
  let spacing = 2.0 * complete.value / frequency;
  var first = (complete.value - phase) / frequency;
  first -= floor(first / spacing) * spacing;
  if (theta == 1.570796327 || first == 0.0) { first = spacing; }
  return EquatorEvents(first, spacing, 1u, phase, frequency, m, amplitude2, azimuth_phase);
}
