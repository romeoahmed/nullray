@group(0) @binding(20) var structure_field: texture_3d<f32>;
@group(0) @binding(21) var structure_sampler: sampler;

/** C2 interpolation in a periodic scalar lattice; filtering reads eight corners in hardware. */
fn field_noise(position: vec3f) -> f32 {
  let cell = floor(position);
  let f = fract(position);
  let u = f * f * f * (f * (6 * f - 15) + 10);
  let coordinate = (cell + u + 0.5) / vec3f(textureDimensions(structure_field));
  return 2 * textureSampleLevel(structure_field, structure_sampler, coordinate, 0).r - 1;
}

