/** 32-byte stackless-tree record: bounds/escape index or leaf direction/flux/temperature. */
struct StellarNode {
  lower: vec4f,
  upper: vec4f,
}
@group(0) @binding(18) var<storage, read> stars: array<StellarNode>;

/**
 * Sum catalogue light within a locally linear Gaussian source footprint.
 * X/y are one-pixel direction increments in the source tangent basis; energy is
 * the positive emitted/observed frequency ratio. Alpha reports spectral support.
 * The finite 3.5-sigma cutoff omits Gaussian tails and cannot certify nonlinear image completeness.
 */
fn stellar_radiance(n: vec3f, east: vec3f, north: vec3f, x: vec2f, y: vec2f, energy: f32) -> vec4f {
  // Prescribed standard deviations: 0.65 detector pixels and 1e−6 source radians
  // (about 0.206 arcsec). These regularize the kernel, not measured stellar sizes.
  let psf_variance = 0.4225;
  let source_variance = 1e-12;
  let cxx = psf_variance * (x.x * x.x + y.x * y.x) + source_variance;
  let cyy = psf_variance * (x.y * x.y + y.y * y.y) + source_variance;
  // Evaluate det(J J^T + sI) as positive terms to retain rank-one critical footprints in f32.
  let area = x.x * y.y - x.y * y.x;
  let determinant = psf_variance * psf_variance * area * area
    + source_variance * psf_variance * (dot(x, x) + dot(y, y)) + source_variance * source_variance;
  let extent = 3.5 * (abs(east) * sqrt(cxx) + abs(north) * sqrt(cyy));
  let curvature = 0.5 * dot(extent, extent);
  let lower = n - extent - vec3f(curvature);
  let upper = n + extent + vec3f(curvature);
  var total = vec3f(0);
  var node = 0u;
  let count = arrayLength(&stars);
  while (node < count) {
    let entry = stars[node];
    if (entry.lower.w >= 0) {
      if (any(entry.upper.xyz < lower) || any(entry.lower.xyz > upper)) { node = u32(entry.lower.w); }
      else { node++; }
      continue;
    }
    node++;
    let source = entry.lower.xyz;
    if (entry.upper.x == 0 || dot(source, n) <= 0 || any(source < lower) || any(source > upper)) { continue; }
    let offset = vec2f(dot(source - n, east), dot(source - n, north));
    // adj(sI + p JJᵀ) gives a sum of squares, avoiding cancellation near a rank-one image.
    let perpendicular = vec2f(dot(offset, vec2f(x.y, -x.x)), dot(offset, vec2f(y.y, -y.x)));
    let squared = (psf_variance * dot(perpendicular, perpendicular)
      + source_variance * dot(offset, offset)) / determinant;
    if (squared > 12.25) { continue; }
    let spectrum = blackbody_radiance(entry.upper.y / energy);
    if (spectrum.a == 0) { return vec4f(0); }
    let weight = entry.upper.x * exp(-0.5 * squared) / (6.283185307179586 * sqrt(determinant));
    total += spectrum.rgb * weight;
  }
  return vec4f(total, 1);
}
