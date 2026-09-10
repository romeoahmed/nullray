/**
 * Neutral Kerr–Newman circular emitter, orientation s=+1:
 * (observed/emitted frequency, coordinate angular velocity).
 * With u=1/r and v=u sqrt(u(1-q²u)), Ω=v/(1+av). Cancelling
 * the common (1+av) in u^t(E-ΩL) avoids rebuilding three metric components.
 */
fn disk_frequency(a: f32, q: f32, e: f32, l: f32, r: f32) -> vec2f {
  if (r <= q * q) { return vec2f(-1.0, 0.0); }
  let u = 1.0 / r;
  let charge = q * q * u;
  if (charge >= 1.0) { return vec2f(-1.0, 0.0); }
  let v = u * sqrt(u * (1.0 - charge));
  let denominator = 1.0 + a * v;
  let norm = 1.0 - (3.0 - 2.0 * charge) * u + 2.0 * a * v;
  let energy = e + (a * e - l) * v;
  if (denominator <= 0.0 || norm <= 0.0 || energy <= 0.0) { return vec2f(-1.0, 0.0); }
  return vec2f(sqrt(norm) / energy, v / denominator);
}
