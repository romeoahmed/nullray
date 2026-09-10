/** Extended observer, camera, source prescription and numerical work policy. */
struct OpticalFrame {
  space: vec4f,
  point: vec4f,
  observer: mat4x4f,
  right: vec4f,
  up: vec4f,
  forward: vec4f,
  appearance: vec4f,
  block: vec4i,
  sampling: vec4f,
  work: vec4f,
  jet: vec4f,
  jet_spectrum: vec4f,
  plasma: vec4f,
  heating: vec4f,
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

struct RayPathPoint {
  coordinate: vec4f,
  domain: vec4i,
}
struct RayPathStorage {
  summary: vec4i,
  points: array<RayPathPoint>,
}
@group(0) @binding(10) var<storage, read_write> inspected_ray: RayPathStorage;


@group(0) @binding(11) var arrival_image: texture_storage_2d<rgba32float, write>;
@group(0) @binding(12) var transmission_image: texture_storage_2d<rgba16float, write>;
@group(0) @binding(19) var<storage, read> jet_table: array<vec4f>;
