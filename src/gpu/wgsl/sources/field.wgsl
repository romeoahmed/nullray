@group(0) @binding(20) var structure_field: texture_3d<f32>;
@group(0) @binding(21) var structure_sampler: sampler;

/** Quintic fade on [0,1], with zero first and second derivatives at both lattice faces. */
fn lattice_weight(f: vec3f) -> vec3f {
  return f * f * f * (f * (6 * f - 15) + 10);
}

/** Quintic coordinate warp gives a C2 interpolant in exact arithmetic; hardware filtering adds rounding. */
fn field_noise(position: vec3f) -> f32 {
  let cell = floor(position);
  let f = fract(position);
  let u = lattice_weight(f);
  let coordinate = (cell + u + 0.5) / vec3f(textureDimensions(structure_field));
  return 2 * textureSampleLevel(structure_field, structure_sampler, coordinate, 0).r - 1;
}

