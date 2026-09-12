@group(0) @binding(0) var sky_work: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(1) var sky_output: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var sky_previous: texture_2d_array<f32>;

/** WebGPU cube-face directions; the source frame is J2000 equatorial. */
fn cube_direction(face: u32, uv: vec2f) -> vec3f {
  var direction: vec3f;
  switch face {
    case 0u: { direction = vec3f(1, -uv.y, -uv.x); }
    case 1u: { direction = vec3f(-1, -uv.y, uv.x); }
    case 2u: { direction = vec3f(uv.x, 1, uv.y); }
    case 3u: { direction = vec3f(uv.x, -1, -uv.y); }
    case 4u: { direction = vec3f(uv.x, -uv.y, 1); }
    default: { direction = vec3f(-uv.x, -uv.y, -1); }
  }
  return normalize(direction);
}

/**
 * Solid angle of two spherical triangles. Each unnormalized determinant is exactly side²;
 * this avoids subtracting four nearly equal atan values for a small distant texel.
 */
fn cube_texel_area(pixel: vec2u, width: u32) -> f32 {
  let side = 2 / f32(width);
  let lo = vec2f(pixel) * side - 1;
  let hi = lo + side;
  let a = vec3f(lo, 1);
  let b = vec3f(hi.x, lo.y, 1);
  let c = vec3f(hi, 1);
  let d = vec3f(lo.x, hi.y, 1);
  let lengths = vec4f(length(a), length(b), length(c), length(d));
  let abc = lengths.x * lengths.y * lengths.z + dot(a, b) * lengths.z
    + dot(b, c) * lengths.x + dot(c, a) * lengths.y;
  let acd = lengths.x * lengths.z * lengths.w + dot(a, c) * lengths.w
    + dot(c, d) * lengths.x + dot(d, a) * lengths.z;
  return 2 * (atan2(side * side, abc) + atan2(side * side, acd));
}

/**
 * Interpolate eight stored lattice values explicitly in f32 for sky generation.
 * The periodic lattice dimensions must be powers of two for masked addressing.
 * Exponential dust amplifies hardware filtering differences. The per-ray
 * material path retains hardware interpolation; this one-time pass uses explicit loads.
 */
fn sky_noise(position: vec3f) -> f32 {
  let cell = vec3i(floor(position));
  let f = fract(position);
  let weight = lattice_weight(f);
  let mask = vec3i(textureDimensions(structure_field)) - 1;
  let a = textureLoad(structure_field, cell & mask, 0).r;
  let b = textureLoad(structure_field, (cell + vec3i(1, 0, 0)) & mask, 0).r;
  let c = textureLoad(structure_field, (cell + vec3i(0, 1, 0)) & mask, 0).r;
  let d = textureLoad(structure_field, (cell + vec3i(1, 1, 0)) & mask, 0).r;
  let e = textureLoad(structure_field, (cell + vec3i(0, 0, 1)) & mask, 0).r;
  let fz = textureLoad(structure_field, (cell + vec3i(1, 0, 1)) & mask, 0).r;
  let g = textureLoad(structure_field, (cell + vec3i(0, 1, 1)) & mask, 0).r;
  let h = textureLoad(structure_field, (cell + vec3i(1, 1, 1)) & mask, 0).r;
  return 2 * mix(mix(mix(a, b, weight.x), mix(c, d, weight.x), weight.y),
    mix(mix(e, fz, weight.x), mix(g, h, weight.x), weight.y), weight.z) - 1;
}

/** Spectral populations, rather than RGB colors, remain linear through spatial filtering. */
@compute @workgroup_size(8, 8)
fn generate_sky(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(sky_work);
  if (any(id.xy >= size) || id.z >= 6u) { return; }
  let n = cube_direction(id.z, 2 * (vec2f(id.xy) + 0.5) / vec2f(size) - 1);
  // J2000 north Galactic pole, in the equatorial source frame.
  let latitude = dot(n, vec3f(-0.8676661490, -0.1980763734, 0.4559837762));
  let warp = sky_noise(5 * n);
  var clouds = 0.0;
  var frequency = 12.0;
  var weight = 0.6;
  for (var octave = 0u; octave < 5u; octave++) {
    clouds += weight * sky_noise(frequency * n + vec3f(warp, 0, 0));
    frequency *= 2;
    weight *= 0.6;
  }
  let band = exp(-0.5 * (latitude / 0.13) * (latitude / 0.13));
  let lane = (latitude + 0.025 * warp) / 0.035;
  let dust = exp(-3 * exp(2 * clouds) * exp(-0.5 * lane * lane));
  let population = band * exp(2.4 * clouds);
  let coefficients = 1024 * (population * vec3f(0.0012 * sqrt(dust), 0.0018 * dust, 0.00012 * dust * dust)
    + vec3f(0, 0.000008, 0));
  textureStore(sky_work, vec2i(id.xy), i32(id.z), vec4f(coefficients, cube_texel_area(id.xy, size.x)));
  textureStore(sky_output, vec2i(id.xy), i32(id.z), vec4f(coefficients, 0));
}

/** Parallel solid-angle reduction. Full-precision work mips avoid cumulative half-float rounding. */
@compute @workgroup_size(8, 8)
fn reduce_sky(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(sky_work);
  if (any(id.xy >= size) || id.z >= 6u) { return; }
  var weighted = vec3f(0);
  var area = 0.0;
  for (var y = 0; y < 2; y++) {
    for (var x = 0; x < 2; x++) {
      let value = textureLoad(sky_previous, 2 * vec2i(id.xy) + vec2i(x, y), i32(id.z), 0);
      weighted += value.rgb * value.a;
      area += value.a;
    }
  }
  let coefficients = weighted / area;
  textureStore(sky_work, vec2i(id.xy), i32(id.z), vec4f(coefficients, area));
  textureStore(sky_output, vec2i(id.xy), i32(id.z), vec4f(coefficients, 0));
}
