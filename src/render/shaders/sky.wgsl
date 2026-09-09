@group(0) @binding(0) var optical_map: texture_2d<f32>;
@group(0) @binding(1) var sky_image: texture_storage_2d<rgba16float, write>;
/**
 * Binding 2 is the shared blackbody table.
 */
@group(0) @binding(3) var celestial_spectra: texture_cube<f32>;
@group(0) @binding(4) var celestial_sampler: sampler;
/** Three host vec4 lanes: spacetime/epoch, source appearance, and resolve options. */
struct RadiationFrame {
  space_time: vec4f,
  appearance: vec4f,
  options: vec4f,
}
@group(0) @binding(5) var<uniform> radiation_frame: RadiationFrame;
@group(0) @binding(6) var<storage, read> disk_profile: array<f32>;
@group(0) @binding(7) var emission_time: texture_2d<f32>;
@group(0) @binding(8) var<storage, read> star_nodes: array<StarNode>;
@group(0) @binding(9) var beam_index: texture_2d<u32>;
@group(0) @binding(10) var<storage, read> repaired_beams: array<SkyBeam>;

/** Interpolate the normalized disk temperature on the prepared logarithmic radial grid. */
fn radial_temperature(r: f32) -> f32 {
  let size = arrayLength(&disk_profile);
  let coordinate = clamp(log(r / radiation_frame.space_time.z) / log(radiation_frame.appearance.w / radiation_frame.space_time.z), 0.0, 1.0) * f32(size - 1u);
  let lower = min(u32(coordinate), size - 2u);
  return mix(disk_profile[lower], disk_profile[lower + 1u], coordinate - f32(lower));
}

@group(0) @binding(11) var edge_index: texture_2d<u32>;
@group(0) @binding(12) var edge_samples: texture_2d<f32>;
@group(0) @binding(13) var edge_times: texture_2d<f32>;
@group(0) @binding(14) var edge_radiance_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(15) var edge_radiance: texture_2d<f32>;
@group(0) @binding(16) var<storage, read_write> edge_statistics: array<atomic<u32>>;
@group(0) @binding(17) var edge_beam_index: texture_2d<u32>;
@group(0) @binding(18) var<storage, read> edge_beams: array<SkyBeam>;

/**
 * Integrate radiance for each endpoint separately; never interpolate event coordinates across branches.
 */
fn shade_endpoint(sample: vec4f, relative_time: f32, dx: vec3f, dy: vec3f) -> vec4f {
  var color = vec3f(0.0);
  var resolved = 1.0;
  if (sample.w == -2.0) { resolved = 0.0; }
  if (sample.w > 0.0) {
    if (radiation_frame.appearance.z == 0.0) { return vec4f(0.0, 0.0, 0.0, 1.0); }
    let direction = celestial_direction(sample);
    let stars = point_stars(direction, dx, dy, sample.z);
    resolved = stars.a;
    // Undo the diffuse half-float encoding (skyCoefficientScale in physics/sky.ts).
    let coefficients = textureSampleGrad(celestial_spectra, celestial_sampler, direction, dx, dy).rgb / 1024.0;
    // Observer energy is one: g = 1/E. Shift each source spectrum without a second g³ factor.
    color = stars.rgb;
    let temperatures = vec3f(4500.0, 6500.0, 12000.0);
    for (var population = 0u; population < 3u; population++) {
      if (coefficients[population] <= 0.0) { continue; }
      // Guard the table range before division, including arbitrarily small positive E.
      if (!(sample.z >= temperatures[population] / 1000000.0)) {
        resolved = 0.0;
        break;
      }
      let spectrum = blackbody_radiance(temperatures[population] / sample.z);
      resolved *= spectrum.a;
      color += coefficients[population] * spectrum.rgb;
    }
    color *= radiation_frame.appearance.z;
  } else if (sample.w <= -3.0) {
    let a = radiation_frame.space_time.x;
    let q = radiation_frame.space_time.y;
    let r = sample.x;
    let root = sqrt(r - q * q);
    let omega = root / (r * r + a * root);
    let time = radiation_frame.space_time.w + relative_time;
    let temperature = radiation_frame.appearance.x * radial_temperature(r)
      * pow(disk_modulation_epoch(r, radiation_frame.space_time.z, sample.y, time, radiation_frame.options.w, omega, radiation_frame.appearance.y), 0.25);
    if (!(sample.z > 0.0) || !(temperature >= 0.0)) {
      resolved = 0.0;
    } else if (temperature > 0.0 && sample.z > 1000000.0 / temperature) {
      resolved = 0.0;
    } else {
      let spectrum = blackbody_radiance(sample.z * temperature);
      color = spectrum.rgb;
      resolved = spectrum.a;
    }
  }
  if (resolved == 0.0 || any(abs(color) > vec3f(65504.0))) {
    return vec4f(0.0);
  }
  return vec4f(color, resolved);
}

@compute @workgroup_size(8, 8)
fn resolve_sky(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= textureDimensions(optical_map))) { return; }
  let pixel = vec2i(id.xy);
  let sample = textureLoad(optical_map, pixel, 0);
  if (radiation_frame.options.x > 0.0) {
    textureStore(sky_image, pixel, diagnostic_endpoint(sample, radiation_frame.options.x));
    return;
  }
  if (radiation_frame.options.y > 0.0) {
    let edge = textureLoad(edge_index, pixel, 0).x;
    if (edge == 0xffffffffu) {
      textureStore(sky_image, pixel, vec4f(0.0));
      return;
    }
    if (edge > 0u) {
      let columns = textureDimensions(edge_radiance).x / 4u;
      let origin = vec2i(vec2u((edge - 1u) % columns, (edge - 1u) / columns) * 4u);
      var sum = vec4f(0.0);
      for (var y = 0; y < 4; y++) {
        for (var x = 0; x < 4; x++) {
          sum += textureLoad(edge_radiance, origin + vec2i(x, y), 0);
        }
      }
      // RGB contains resolved light only; alpha retains the resolved sample fraction.
      textureStore(sky_image, pixel, sum / 16.0);
      return;
    }
  }
  var dx = vec3f(0.0);
  var dy = vec3f(0.0);
  if (sample.w > 0.0) {
    let stencil = celestial_beam(optical_map, pixel, sample);
    dx = stencil.dx.xyz;
    dy = stencil.dy.xyz;
    let repair = textureLoad(beam_index, pixel, 0).x;
    if (repair > 0u) {
      let beam = repaired_beams[repair - 1u];
      dx = beam.dx.xyz;
      dy = beam.dy.xyz;
    }
  }
  textureStore(sky_image, pixel, shade_endpoint(sample, textureLoad(emission_time, pixel, 0).x, dx, dy));
}

/**
 * Each spectral query owns one invocation; the image resolve only averages stored radiance.
 */
@compute @workgroup_size(64)
fn shade_edges(@builtin(global_invocation_id) id: vec3u) {
  let index = id.x / 16u;
  if (index >= min(atomicLoad(&edge_statistics[0]), u32(radiation_frame.options.z))) { return; }
  let columns = textureDimensions(edge_samples).x / 4u;
  let origin = vec2i(vec2u(index % columns, index / columns) * 4u);
  let position = origin + vec2i(i32(id.x % 4u), i32((id.x % 16u) / 4u));
  let endpoint = textureLoad(edge_samples, position, 0);
  var dx = vec3f(0.0);
  var dy = vec3f(0.0);
  if (endpoint.w > 0.0) {
    let stencil = celestial_patch_beam(edge_samples, position, origin, endpoint);
    dx = stencil.dx.xyz;
    dy = stencil.dy.xyz;
    let repair = textureLoad(edge_beam_index, position, 0).x;
    if (repair > 0u) {
      let beam = edge_beams[repair - 1u];
      dx = beam.dx.xyz;
      dy = beam.dy.xyz;
    }
  }
  let color = shade_endpoint(endpoint, textureLoad(edge_times, position, 0).x, dx, dy);
  if (color.a == 0.0) {
    if (endpoint.w == -2.0) { atomicAdd(&edge_statistics[1], 1u); }
    else if (endpoint.w > 0.0 && !celestial_has_area(celestial_direction(endpoint), dx, dy)) {
      atomicAdd(&edge_statistics[2], 1u);
    } else { atomicAdd(&edge_statistics[3], 1u); }
  }
  textureStore(edge_radiance_output, position, color);
}
