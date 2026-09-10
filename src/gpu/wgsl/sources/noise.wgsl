@group(0) @binding(0) var lattice: texture_storage_3d<rgba8unorm, write>;

/** Deterministic integer avalanche at each lattice site; no frame, clock or shared random state. */
@compute @workgroup_size(4, 4, 4)
fn generate_noise(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(lattice);
  if (any(id >= size)) { return; }
  let index = (id.z * size.y + id.y) * size.x + id.x;
  var value = (index ^ 0x6a09e667u) * 0x9e3779b1u;
  value = (value ^ (value >> 16u)) * 0x85ebca6bu;
  value = (value ^ (value >> 13u)) * 0xc2b2ae35u;
  let byte = (value ^ (value >> 16u)) >> 24u;
  textureStore(lattice, vec3i(id), vec4f(f32(byte) / 255, 0, 0, 1));
}
