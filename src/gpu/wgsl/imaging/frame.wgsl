/**
 * 288-byte uniform ABI written by src/gpu/optics/frame.ts.
 * All physical scalars are quantized before upload; block identity remains integer data.
 */
struct OpticalFrame {
  // Spin a, charge q, disk inner/outer radii in M.
  space: vec4f,
  // Observer r in M, polar angle/rotating longitude in radians, chart sign ±1.
  point: vec4f,
  // Orthonormal tetrad columns (u, e_r, e_θ, e_φ) in Cartesian KS components.
  observer: mat4x4f,
  right: vec4f,
  up: vec4f,
  forward: vec4f,
  // Null launch time including observer offset, disk temperature K, structure strength, sky scale.
  appearance: vec4f,
  // Block kind, exterior-copy index, stationary-side sign, padding.
  block: vec4i,
  // Pixel jitter x/y, tan(vertical FOV/2), diagnostic selector 0–3.
  sampling: vec4f,
  // Local tolerance, attempt budget, inspection x/y offsets in image-height units.
  work: vec4f,
  // Launch/outer radii in M, cosine of outer half-width angle, local core speed/c.
  jet: vec4f,
  // Frequency-coordinate scale, two padding lanes, enabled flag.
  jet_spectrum: vec4f,
  // Detector-normalized radial amplitude, r₀², hν_o/k_B in K, enabled flag.
  plasma: vec4f,
  // Annulus inner/outer radii in M, fractional heating contrast, sixfold phase rate.
  heating: vec4f,
  // Peak H/r, normalized coordinate-column optical depth, other-domain light scale, column normalizer.
  material: vec4f,
}
@group(0) @binding(0) var<uniform> optical_frame: OpticalFrame;
@group(0) @binding(1) var radiance_image: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var sky_spectra: texture_cube<f32>;
@group(0) @binding(4) var sky_sampler: sampler;
@group(0) @binding(5) var<storage, read> disk_profile: array<f32>;
@group(0) @binding(6) var domain_image: texture_storage_2d<rgba32sint, write>;
@group(0) @binding(7) var<storage, read> atmosphere_table: array<vec4f>;
@group(0) @binding(8) var q_image: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var u_image: texture_storage_2d<rgba16float, write>;

/**
 * Inspection ABI: coordinate = (active radius, n_z, null time/log amplitude, inverse flag).
 * Domain = (block kind, universe, side, chart tag); doubled chart signs identify bifurcation patches.
 */
struct RayPathPoint {
  coordinate: vec4f,
  domain: vec4i,
}
/**
 * Header (stored point count, result kind, signed endpoint side, equatorial crossing count), then points.
 * A full point buffer truncates inspection storage, not the optical work budget.
 */
struct RayPathStorage {
  summary: vec4i,
  points: array<RayPathPoint>,
}
@group(0) @binding(10) var<storage, read_write> inspected_ray: RayPathStorage;

@group(0) @binding(11) var arrival_image: texture_storage_2d<rgba32float, write>;
@group(0) @binding(12) var transmission_image: texture_storage_2d<rgba16float, write>;
@group(0) @binding(19) var<storage, read> jet_table: array<vec4f>;
