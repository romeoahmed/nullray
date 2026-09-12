/** One local camera basis for imaging and inspection; offsets use image-height units. */
fn detector_direction(coordinate: vec2f) -> vec3f {
  return normalize(optical_frame.forward.xyz + 2 * optical_frame.sampling.z *
    (coordinate.x * optical_frame.right.xyz - coordinate.y * optical_frame.up.xyz));
}

struct DetectorRay {
  orbit: KerrOrbit,
  momentum: vec4f,
}

/** Initialize a propagating photon after the caller establishes cutoff < 1. */
fn detector_ray(g: KerrGeometry, local: vec3f, cutoff: f32) -> DetectorRay {
  let p = optical_frame.observer * vec4f(1, -sqrt(1 - cutoff) * local);
  var initial = kerr_orbit(g, optical_frame.space.y, p, 0);
  initial.plasma = optical_frame.plasma.xy;
  if (optical_frame.heating.z > 0 && g.radius > optical_frame.heating.x && g.radius < optical_frame.heating.y) {
    initial.plasma.x = cutoff * g.sigma * (1 + optical_frame.plasma.y / (g.radius * g.radius));
  }
  initial.state.radial.z = optical_frame.appearance.x;
  return DetectorRay(initial, p);
}

@compute @workgroup_size(8, 8)
fn render_image(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(radiance_image);
  if (any(id.xy >= size)) { return; }
  let pixel = vec2i(id.xy);
  textureStore(arrival_image, pixel, vec4f(0));
  textureStore(transmission_image, pixel, vec4f(0));
  let coordinate = (vec2f(id.xy) + 0.5 + optical_frame.sampling.xy - vec2f(size) / 2) / f32(size.y);
  let local = detector_direction(coordinate);
  let g = kerr_geometry(optical_frame.space.xy, optical_frame.point.xyz, optical_frame.point.w);
  let detector_cutoff = plasma_cutoff(g, optical_frame.appearance.x);
  let index_squared = 1 - detector_cutoff;
  if (!(index_squared > 0)) {
    textureStore(radiance_image, pixel, vec4f(0));
    textureStore(q_image, pixel, vec4f(0));
    textureStore(u_image, pixel, vec4f(0));
    textureStore(domain_image, pixel, vec4i(0));
    return;
  }
  let launch = detector_ray(g, local, detector_cutoff);
  let initial = launch.orbit;
  let screen_up = normalize(optical_frame.up.xyz - local * dot(optical_frame.up.xyz, local));
  let screen_right = cross(local, screen_up);
  let up = optical_frame.observer * vec4f(0, screen_up);
  let right = optical_frame.observer * vec4f(0, screen_right);
  let screen = PolarizationScreen(walker_penrose(g, launch.momentum, up), walker_penrose(g, launch.momentum, right), vec2f(-right.w, up.w));
  let ray = trace_ray(initial, false, screen);
  var color = vec4f(0);
  var stokes_q = vec3f(0);
  var stokes_u = vec3f(0);
  var shift = 0.0;
  if (ray.kind == 3u || ray.kind == 4u || (ray.kind == 2u && optical_frame.material.x > 0)) { color.a = 1; }
  if (ray.kind == 1u) {
    let energy = f32(ray.block.z) * ray.path.constants.x;
    if (energy > 0) {
      let direction = ray.end_sign * normalize(ray.path.state.direction);
      textureStore(arrival_image, pixel, vec4f(direction, energy));
      textureStore(transmission_image, pixel, vec4f(ray.transfer.transmission * source_light(ray.block), f32(ray.crossings), 0, 0));
      color.a = 1;
      // Vacuum background filtering uses the completed neighboring rays in the celestial pass.
      if (optical_frame.plasma.w != 0) {
        let coefficients = textureSampleLevel(sky_spectra, sky_sampler, direction, 0).rgb / 1024;
        let temperatures = vec3f(4500, 6500, 12000);
        for (var population = 0u; population < 3u; population++) {
          let spectrum = source_spectrum(temperatures[population] / energy);
          color = vec4f(color.rgb + optical_frame.appearance.w * source_light(ray.block) * coefficients[population] * spectrum.rgb, color.a * spectrum.a);
        }
      }
      shift = 1 / energy;
    }
  }
  if (ray.kind == 2u && optical_frame.material.x == 0) {
    var r = ray.path.state.radial.x;
    if (ray.path.inverse) { r = 1 / r; }
    let a = ray.path.space.x;
    let q = ray.path.space.y;
    if (r > q * q) {
      let root = sqrt(r - q * q);
      let omega = root / (r * r + a * root);
      let phi = atan2(ray.path.state.direction.y, ray.path.state.direction.x);
      let emitter_g = kerr_geometry(ray.path.space, vec3f(r, 1.570796326794897, phi), ray.path.chart);
      let side = select(ray.block.z, 1, ray.block.x == 4);
      let emitter = kerr_circular_velocity(emitter_g, omega, side);
      if (emitter.x != 0) {
        let ut = emitter.x;
        let frequency = ut * (ray.path.constants.x - omega * ray.path.constants.y);
        if (frequency > 0 && frequency * frequency > plasma_cutoff(emitter_g, ray.path.state.radial.z)) {
          let photon = kerr_tangent(ray.path);
          var atmosphere = vec2f(1, 0);
          if (optical_frame.plasma.w == 0) { atmosphere = disk_atmosphere(abs(photon.w) / frequency); }
          var emission_phi = phi;
          var emission_time = ray.path.state.radial.z;
          if (ray.path.chart < 0) {
            let primitive = kerr_chart_primitives(ray.path.space, r);
            emission_phi += 2 * primitive.x;
            emission_time += 2 * primitive.y;
          }
          var cloud = vec3f(0);
          if (optical_frame.appearance.z > 0) {
            cloud = disk_cloud(ray.path.space, optical_frame.space.z, r, emission_phi, 0, emission_time, omega, vec4f(0));
          }
          let temperature = optical_frame.appearance.y * disk_temperature(r)
            * exp(0.9 * optical_frame.appearance.z * cloud.z) * pow(source_light(ray.block), 0.25);
          if (optical_frame.plasma.w != 0 || frequency >= temperature / 1000000) {
            let spectrum = source_spectrum(temperature / frequency);
            color = vec4f(spectrum.rgb * atmosphere.x, spectrum.a);
            shift = 1 / frequency;
            if (atmosphere.y > 0 && color.a > 0) {
              var direction: vec2f;
              if (initial.state.direction.z == 0 && all(initial.state.angular.xy == vec2f(0))) {
                // Reflection symmetry makes the equatorial normal parallel transported.
                // Its orthogonal screen direction supplies the degenerate principal-ray limit.
                direction = screen.equatorial;
              } else {
                let f = electric_vector(emitter_g, photon, emitter, vec4f(0, 0, 0, 1));
                let source = walker_penrose(emitter_g, photon, f);
                direction = vec2f(dot(source, screen.up), dot(source, screen.right));
              }
              let norm = dot(direction, direction);
              if (norm > 0) {
                let linear = vec2f(direction.x * direction.x - direction.y * direction.y, 2 * direction.x * direction.y) / norm;
                stokes_q = spectrum.rgb * atmosphere.y * linear.x;
                stokes_u = spectrum.rgb * atmosphere.y * linear.y;
              } else { color = vec4f(0); }
            }
          }
        }
      }
    }
  }
  if (color.a > 0) {
    color = vec4f(ray.transfer.intensity + ray.transfer.transmission * color.rgb, color.a);
    stokes_q = ray.transfer.q + ray.transfer.transmission * stokes_q;
    stokes_u = ray.transfer.u + ray.transfer.transmission * stokes_u;
    if (ray.transfer.frequency > 0) { shift = 1 / ray.transfer.frequency; }
  }
  if (optical_frame.plasma.w != 0) { color = vec4f(color.rgb * index_squared, color.a); }
  if (optical_frame.sampling.w == 1 && color.a > 0) {
    let value = log2(max(shift, 1e-20));
    color = vec4f(max(value, 0.0), 0.15, max(-value, 0.0), 1);
  } else if (optical_frame.sampling.w == 2 && color.a > 0) {
    color = vec4f(0.5 + 0.5 * cos(vec3f(0, 2.1, 4.2) + f32(ray.crossings)), 1);
  }
  if (optical_frame.sampling.w == 3 && ray.kind != 0u) {
    if (ray.kind == 3u) { color = vec4f(0.08, 0.08, 0.08, 1); }
    else if (ray.kind == 4u) { color = vec4f(0.02, 0.04, 0.08, 1); }
    else if (ray.end_sign < 0) { color = vec4f(0.5, 0.1, 0.9, 1); }
    else {
      let phase = f32(ray.block.y) * 2.39996323;
      let domain = 0.5 + 0.45 * cos(phase + vec3f(0, 2.1, 4.2));
      color = vec4f(domain * select(0.35, 1.0, ray.kind == 2u), 1);
    }
  }
  let peak = max(max(abs(color.r), abs(color.g)), abs(color.b));
  // Share the f16 storage scale across I/Q/U; saturation remains lossy and
  // does not mark the sample unresolved.
  if (peak > 65504) {
    let storage_scale = 65504 / peak;
    color = vec4f(color.rgb * storage_scale, color.a);
    stokes_q *= storage_scale;
    stokes_u *= storage_scale;
  }
  if (color.a == 0) { stokes_q = vec3f(0); stokes_u = vec3f(0); }
  textureStore(radiance_image, pixel, color);
  textureStore(q_image, pixel, vec4f(stokes_q, color.a));
  textureStore(u_image, pixel, vec4f(stokes_u, color.a));
  textureStore(domain_image, pixel, vec4i(i32(ray.kind), ray.block.y, ray.block.z, i32(ray.end_sign)));
}

/** Inspect an unjittered detector direction using the image integrator and source ordering. */
@compute @workgroup_size(1)
fn inspect_ray() {
  inspected_ray.summary = vec4i(0);
  let local = detector_direction(optical_frame.work.zw);
  let g = kerr_geometry(optical_frame.space.xy, optical_frame.point.xyz, optical_frame.point.w);
  let detector_cutoff = plasma_cutoff(g, optical_frame.appearance.x);
  let index_squared = 1 - detector_cutoff;
  if (!(index_squared > 0)) { return; }
  let launch = detector_ray(g, local, detector_cutoff);
  let initial = launch.orbit;
  let ray = trace_ray(initial, true, PolarizationScreen());
  inspected_ray.summary.y = i32(ray.kind);
  inspected_ray.summary.z = i32(ray.end_sign);
  inspected_ray.summary.w = i32(ray.crossings);
}
