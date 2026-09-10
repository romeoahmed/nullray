/**
 * One differentiated physical ray supplies both screen derivatives.
 * Keep failures explicit; no neighboring branch is substituted for the center.
 */
fn beam_from_differential(sky: DifferentialSky) -> SkyBeam {
  var derivative_valid = true;
  let status = &derivative_valid;
  if (!sky.valid) { return SkyBeam(vec4f(0.0), vec4f(0.0)); }
  let radius = d_sqrt(status, d_mul(status, d_sub(status, d_constant(status, 1.0), sky.mu), d_add(status, d_constant(status, 1.0), sky.mu)));
  let x = d_mul(status, radius, d_cos(status, sky.phi));
  let y = d_mul(status, radius, d_sin(status, sky.phi));
  if (!(*status)) { return SkyBeam(vec4f(0.0), vec4f(0.0)); }
  return SkyBeam(vec4f(x.y, y.y, sky.mu.y, 1.0), vec4f(x.z, y.z, sky.mu.z, 1.0));
}

/** Cartesian derivatives are rotated before rounding back to physical screen coordinates. */
fn beam_from_prepared_sky(sky: DifferentialSky, prepared: PreparedSkyBeam) -> SkyBeam {
  let beam = beam_from_differential(sky);
  if (!prepared.refined || !sky.valid) { return beam; }
  let x = d64_restore_gradient(prepared.basis,beam.dx.x,beam.dy.x);
  let y = d64_restore_gradient(prepared.basis,beam.dx.y,beam.dy.y);
  let z = d64_restore_gradient(prepared.basis,beam.dx.z,beam.dy.z);
  return SkyBeam(vec4f(x.x,y.x,z.x,beam.dx.w),vec4f(x.y,y.y,z.y,beam.dy.w));
}

/** Shared production quadrature policy for ordinary and selected critical beams. */
fn evaluate_prepared_sky(frame: Frame, prepared: PreparedSkyBeam, lane: u32) -> DifferentialSky {
  return cooperative_sky_budget(frame,prepared.path,lane,select(2048u,16384u,prepared.refined));
}

