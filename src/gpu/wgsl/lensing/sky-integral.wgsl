/** One workgroup transports one ray; elected subgroup lanes own reduction slots. */
var<workgroup> repair_sums: array<vec4f, 32>;
var<workgroup> repair_flags: array<bool, 32>;
struct DifferentialIntegral { value: vec3f, area: f32, valid: bool }
var<workgroup> repair_integral: DifferentialIntegral;


/**
 * Accumulate strided nodes privately and reduce native vec4 values once. Each
 * elected lane writes its own local-index slot; no subgroup size, index, or
 * mapping to consecutive invocations is assumed. All other slots stay zero.
 */
fn cooperative_sky_sum(frame: Frame, path: DifferentialSkyPath, count: u32, lane: u32) -> DifferentialIntegral {
  var sum = vec3f(0.0);
  var area = 0.0;
  var valid = true;
  let width = 8.0 / f32(count);
  for (var node = lane; node < count; node += 32u) {
    let quadrature = TRANSPORT_GAUSS[node & 7u];
    let fraction = (f32(node / 8u) + quadrature.x) * width;
    let rate = d_scale(&valid,d_warped_sky_rate(&valid, path.radial, path.polar, frame.space.x, frame.space.y,
      path.energy, path.momentum, path.end, fraction, path.motion),quadrature.y * width);
    sum = d_add(&valid, sum, rate);
    area += d_sky_area_rate(&valid, path.mu, rate);
  }
  let reduced = subgroupAdd(vec4f(sum, area));
  let complete = subgroupAll(valid);
  repair_sums[lane] = vec4f(0.0);
  repair_flags[lane] = true;
  if (subgroupElect()) {
    repair_sums[lane] = reduced;
    repair_flags[lane] = complete;
  }
  workgroupBarrier();
  if (lane == 0u) {
    sum = vec3f(0.0);
    area = 0.0;
    for (var i = 0u; i < 32u; i++) {
      sum = d_add(&valid, sum, repair_sums[i].xyz);
      area += repair_sums[i].w;
      valid = valid && repair_flags[i];
    }
    repair_integral = DifferentialIntegral(sum, area, valid);
  }
  return workgroupUniformLoad(&repair_integral);
}

/**
 * Composite Gauss quadrature of the regularized rate. Phase chooses the first
 * panel count; two successive refinements must agree in both components and
 * solid-angle area. The estimate is numerical evidence, not a rigorous bound.
 * All 32 invocations participate with a workgroup-uniform frame and path.
 */
fn cooperative_sky_budget(frame: Frame, path: DifferentialSkyPath, lane: u32, maximum_count: u32) -> DifferentialSky {
  if (!path.valid) { return differential_sky_failure(); }
  if (frame.space.x == 0.0 && path.motion.enabled) {
    return DifferentialSky(path.mu, path.launch, path.energy.x, true);
  }
  let a = frame.space.x;
  let phase = path.end.x * sqrt(path.momentum.x * path.momentum.x + 2.0 * abs(path.carter.x)
    + 2.0 * a * a * path.energy.x * path.energy.x);
  var count = 32u;
  for (var i = 0u; i < 6u && f32(count) < 2.0 * phase; i++) { count *= 2u; }
  if (count > 256u) { return differential_sky_failure(); }
  let initial = cooperative_sky_sum(frame, path, count, lane);
  if (!initial.valid) { return differential_sky_failure(); }
  var valid = true;
  var previous = initial.value;
  var previous_area = initial.area;
  var agreed = false;
  for (var level = 0u; level < 9u && count < maximum_count; level++) {
    count *= 2u;
    let integral = cooperative_sky_sum(frame, path, count, lane);
    if (!integral.valid) { return differential_sky_failure(); }
    let estimate = integral.value;
    let area = path.mu.y * (estimate.z + path.launch.z) - path.mu.z * (estimate.y + path.launch.y);
    let converged = abs(area) > 1e-20
      && abs(integral.area - previous_area) <= 1e-4 * abs(area)
      && all(abs(estimate - previous) <= 1e-5 * (vec3f(1.0) + abs(estimate)));
    if (converged && agreed) {
      let phi = d_add(&valid, estimate, path.launch);
      return DifferentialSky(path.mu, phi, path.energy.x, valid);
    }
    agreed = converged;
    previous = estimate;
    previous_area = integral.area;
  }
  return differential_sky_failure();
}


/** Ordinary path budget retained for direct kernel consumers. */
fn cooperative_sky(frame: Frame, path: DifferentialSkyPath, lane: u32) -> DifferentialSky {
  return cooperative_sky_budget(frame,path,lane,2048u);
}
