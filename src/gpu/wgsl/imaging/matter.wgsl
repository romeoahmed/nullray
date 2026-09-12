/**
 * Observer-to-source accumulation in a common detector Stokes basis.
 * I/Q/U are spectral RGB coefficients; transmission is scalar foreground attenuation.
 * Frequency records a sampled emitting-frame energy for the diagnostic, not an averaged redshift.
 */
struct Transfer {
  intensity: vec3f,
  q: vec3f,
  u: vec3f,
  transmission: f32,
  frequency: f32,
  valid: bool,
}

fn empty_transfer() -> Transfer {
  return Transfer(vec3f(0), vec3f(0), vec3f(0), 1, 0, true);
}

/** Illumination is boundary data, independent of the maximal extension of the metric. */
fn source_light(block: vec3i) -> f32 {
  if ((block.y == 0 && block.z == 1) || block.x == 4 || block.x == 5) { return 1; }
  return optical_frame.material.z;
}

/** Compact radial envelope, with unit maximum at one third of the emitting annulus. */
fn disk_envelope(r: f32) -> f32 {
  let x = (r - optical_frame.space.z) / (optical_frame.space.w - optical_frame.space.z);
  if (x <= 0 || x >= 1) { return 0; }
  return sqrt(6.75 * x) * (1 - x);
}

/** Prescribed width in M: inner |z|^0.6 collimation approaches conical growth outside. */
fn jet_width(height: f32) -> f32 {
  let cosine = optical_frame.jet.z;
  let relative = height / optical_frame.jet.y;
  return sqrt(1 - cosine * cosine) / cosine * optical_frame.jet.y
    * pow(relative, 0.6) * pow((1 + 16 * relative * relative) / 17, 0.2);
}

/** Logarithmic width slope d(log w)/d(log |z|), used by the same material streamline. */
fn jet_expansion(height: f32) -> f32 {
  let transition = 16 * height * height / (optical_frame.jet.y * optical_frame.jet.y);
  return 0.6 + 0.4 * transition / (1 + transition);
}

/**
 * Finite prescribed support for the corrugated Gaussian disk and displaced jet.
 * The disk cutoff truncates tails; this is not the full support of an infinite Gaussian.
 */
fn material_support(space: vec2f, r: f32, n: vec3f) -> vec2<bool> {
  let disk = optical_frame.material.x > 0 && r > optical_frame.space.z && r < optical_frame.space.w
    && abs(n.z) < (4.5 + 2.4 * optical_frame.appearance.z) * optical_frame.material.x * disk_envelope(r);
  var jet = false;
  if (optical_frame.jet_spectrum.w > 0 && r > optical_frame.jet.x && r < optical_frame.jet.y) {
    let height = r * abs(n.z);
    if (height > optical_frame.jet.x) {
      let width = 1.5 * jet_width(height);
      jet = (r * r + space.x * space.x) * dot(n.xy, n.xy) < width * width;
    }
  }
  return vec2<bool>(disk, jet);
}

/** Finite-cutoff synchrotron spectrum; asymptotes are nu^(1/3) and nu^(-1). */
fn jet_spectrum(shift: f32) -> vec3f {
  let count = arrayLength(&jet_table);
  let logarithm = log(shift * optical_frame.jet_spectrum.x);
  if (logarithm <= -10) { return jet_table[0].rgb * exp((logarithm + 10) / 3); }
  if (logarithm >= 10) { return jet_table[count - 1u].rgb * exp(10 - logarithm); }
  let coordinate = (logarithm + 10) / 20 * f32(count - 1u);
  let index = min(u32(coordinate), count - 2u);
  return mix(jet_table[index].rgb, jet_table[index + 1u].rgb, coordinate - f32(index));
}

/** Electric-screen Walker–Penrose data, shared by every emitting volume along one vacuum ray. */
struct PolarizationScreen {
  up: vec2f,
  right: vec2f,
  equatorial: vec2f,
}

/**
 * Form a unit electric-screen tangent using the four-dimensional Hodge product.
 * Emitter and photon are tangents; normal is a covector. Returns zero for a
 * degenerate norm. Reversing the electric-vector sign leaves linear Stokes data unchanged.
 */
fn electric_vector(g: KerrGeometry, photon: vec4f, emitter: vec4f, normal: vec4f) -> vec4f {
  let u = kerr_lower(g, emitter);
  let p = kerr_lower(g, photon);
  let n = normal;
  let f = vec4f(-dot(u.yzw, cross(p.yzw, n.yzw)),
    u.x * cross(p.yzw, n.yzw) - p.x * cross(u.yzw, n.yzw) + n.x * cross(u.yzw, p.yzw));
  let norm = dot(kerr_lower(g, f), f);
  if (!(norm > 0)) { return vec4f(0); }
  return f * inverseSqrt(norm);
}

/**
 * Return detector (Q,U) per unit polarized intensity from vacuum WP contractions.
 * An equatorial fallback handles selected degeneracies; zero output elsewhere
 * is a model limitation, not a proof that degenerate emission is unpolarized.
 */
fn linear_polarization(
  g: KerrGeometry,
  photon: vec4f,
  emitter: vec4f,
  normal: vec4f,
  screen: PolarizationScreen
) -> vec2f {
  let electric = electric_vector(g, photon, emitter, normal);
  let wp = walker_penrose(g, photon, electric);
  var direction = vec2f(dot(wp, screen.up), dot(wp, screen.right));
  if (dot(direction, direction) == 0) { direction = screen.equatorial; }
  let norm = dot(direction, direction);
  if (!(norm > 0)) { return vec2f(0); }
  return vec2f(direction.x * direction.x - direction.y * direction.y, 2 * direction.x * direction.y) / norm;
}

/**
 * Accumulate one positive detector-normalized affine cell length dl in M.
 * Sources are sampled locally; constant-coefficient attenuation is analytic, with
 * a small-depth series to avoid cancellation. Inhomogeneous-cell accuracy remains approximate.
 */
fn transfer_cell(
  input: Transfer,
  path: KerrOrbit,
  photon: vec4f,
  dl: f32,
  block: vec3i,
  screen: PolarizationScreen
) -> Transfer {
  var result = input;
  if (dl <= 0) { return result; }
  var r = path.state.radial.x;
  if (path.inverse) { if (r <= 0) { return result; } r = 1 / r; }
  if (r <= 0) { return result; }
  let n = normalize(path.state.direction);
  let support = material_support(path.space, r, n);
  let disk = support.x;
  let jet = support.y;
  if (!disk && !jet) { return result; }
  let geometry = medium_geometry(path.space, r, n, path.chart);
  let g = geometry.frame;
  var emission = vec3f(0);
  var q = vec3f(0);
  var u = vec3f(0);
  var absorption = 0.0;
  var source_frequency = 0.0;
  if (disk && r > path.space.y * path.space.y) {
    let root = sqrt(r - path.space.y * path.space.y);
    let omega = root / (r * r + path.space.x * root);
    let side = select(block.z, 1, block.x == 4);
    let emitter = kerr_circular_velocity(g, omega, side);
    if (emitter.x == 0) { result.valid = false; return result; }
    let frequency = -dot(kerr_lower(g, photon), emitter);
    if (!(frequency > 0)) { result.valid = false; return result; }
    let envelope = disk_envelope(r);
    let thickness = optical_frame.material.x * envelope;
    let nominal_height = n.z / thickness;
    let primitive = kerr_chart_primitives(path.space, r);
    let phase = atan2(n.y, n.x) + (1 - path.chart) * primitive.x;
    let epoch = path.state.radial.z + (1 - path.chart) * primitive.y;
    var cloud = vec3f(0);
    if (optical_frame.appearance.z > 0) {
      let radial_span = dl * dot(geometry.radial_gradient, photon.yzw);
      let direction_span = dl * (transpose(geometry.direction_gradient) * photon.yzw);
      let cylindrical = dot(n.xy, n.xy);
      var longitude_span = 0.0;
      if (cylindrical > 0) {
        longitude_span = (n.x * direction_span.y - n.y * direction_span.x) / cylindrical;
      }
      var epoch_span = dl * photon.x + path.chart * radial_span;
      if (path.chart < 0) {
        let delta = r * r - 2 * r + dot(path.space, path.space);
        if (!(delta > 0)) { result.valid = false; return result; }
        longitude_span += 2 * path.space.x / delta * radial_span;
        epoch_span += 2 * (r * r + path.space.x * path.space.x) / delta * radial_span;
      }
      let x = (r - optical_frame.space.z) / (optical_frame.space.w - optical_frame.space.z);
      let envelope_gradient = (0.5 / x - 1 / (1 - x)) / (optical_frame.space.w - optical_frame.space.z);
      let height_span = direction_span.z / thickness - nominal_height * envelope_gradient * radial_span;
      cloud = disk_cloud(path.space, optical_frame.space.z, r, phase, nominal_height, epoch, omega,
        vec4f(radial_span, longitude_span, height_span, epoch_span));
    }
    let log_density = 6 * optical_frame.appearance.z * cloud.x;
    let density = exp(log_density);
    let corrugation = 1 + 0.4 * optical_frame.appearance.z * cloud.y;
    let height = (nominal_height - 0.6 * optical_frame.appearance.z * cloud.y) / corrugation;
    let column = envelope * optical_frame.space.z / r;
    let edges = column * column * optical_frame.material.w;
    let alpha = optical_frame.material.y * exp(-0.5 * height * height) * density * edges
      / (2.506628274631 * thickness * corrugation * r);
    let index_squared = 1 - plasma_cutoff(g, path.state.radial.z) / (frequency * frequency);
    if (!(index_squared > 0)) { result.valid = false; return result; }
    absorption = alpha * frequency * sqrt(index_squared);
    // Large clouds carry opacity; finer sheared structure prescribes the local heating contrast.
    let temperature = optical_frame.appearance.y * disk_temperature(r)
      * exp(0.9 * optical_frame.appearance.z * cloud.z) * pow(source_light(block), 0.25);
    let spectrum = source_spectrum(temperature / frequency);
    if (spectrum.a == 0) { result.valid = false; return result; }
    var angular = vec2f(1, 0);
    if (optical_frame.plasma.w == 0) {
      let normal = vec4f(0, geometry.direction_gradient[2]);
      let normal_norm = dot(normal, kerr_raise(g, normal));
      if (normal_norm > 0) {
        let unit = normal * inverseSqrt(normal_norm);
        let mu = abs(dot(unit, photon)) / frequency;
        angular = disk_atmosphere(mu);
        let polarization = linear_polarization(g, photon, emitter, unit, screen);
        q = absorption * spectrum.rgb * angular.y * polarization.x;
        u = absorption * spectrum.rgb * angular.y * polarization.y;
      }
    }
    emission = absorption * spectrum.rgb * angular.x;
    source_frequency = frequency;
  }
  if (jet) {
    let a = path.space.x;
    let delta = r * r - 2 * r + dot(path.space, path.space);
    if (!(delta > 0)) { result.valid = false; return result; }
    let base = optical_frame.jet.x;
    let height = abs(g.position.z);
    let width = jet_width(height);
    let transverse = length(g.position.xy) / width;
    let omega = a * g.factor / (r * r + a * a * (1 + g.factor * (1 - n.z * n.z)));
    let side = select(block.z, 1, block.x == 4);
    let rest = kerr_circular_velocity(g, omega, side);
    if (rest.x == 0) { result.valid = false; return result; }
    let beta = optical_frame.jet.w * (0.65 + 0.35 * exp(-4 * transverse * transverse));
    let trial_direction = vec4f(0, jet_expansion(height) * g.position.xy / height, sign(g.position.z));
    let projected = trial_direction + rest * dot(kerr_lower(g, rest), trial_direction);
    let direction_norm = dot(kerr_lower(g, projected), projected);
    if (!(direction_norm > 0)) { result.valid = false; return result; }
    let emitter = (rest + beta * projected * inverseSqrt(direction_norm)) * inverseSqrt(1 - beta * beta);
    let frequency = -dot(kerr_lower(g, photon), emitter);
    if (!(frequency > 0)) { result.valid = false; return result; }
    let primitive = kerr_chart_primitives(path.space, r);
    let epoch = path.state.radial.z + (1 - path.chart) * primitive.y;
    let longitude = atan2(n.y, n.x) + (1 - path.chart) * primitive.x;
    let launch_time = epoch - (height - base) / beta;
    let flow_phase = launch_time / (4 * base);
    let winding = flow_phase + 1.5 * log(height / base);
    let bend = 0.13 * (1 - base / height) * vec2f(cos(winding), sin(winding));
    let shell_radius = length(g.position.xy / width - bend);
    let core = 0.16 * exp(-10 * shell_radius * shell_radius);
    let sheath = exp(-18 * (shell_radius - 0.68) * (shell_radius - 0.68));
    let knots = exp(2.5 * field_noise(vec3f(3 * cos(longitude - winding), 3 * sin(longitude - winding), 1.5 * flow_phase))
      + 0.75 * field_noise(vec3f(10 * cos(longitude), 10 * sin(longitude), 5 * flow_phase)));
    let taper = smoothstep(base, 1.8 * base, height) * (1 - smoothstep(0.55 * optical_frame.jet.y, optical_frame.jet.y, r));
    let boundary = 1 - smoothstep(1.05, 1.35, shell_radius);
    let dilution = jet_width(base) / width;
    let density = dilution * dilution * (core + sheath) * knots * taper * boundary;
    if (density > 0) {
      let index_squared = 1 - plasma_cutoff(g, path.state.radial.z) / (frequency * frequency);
      if (!(index_squared > 0)) { result.valid = false; return result; }
      emission += source_light(block) * jet_spectrum(frequency / sqrt(density)) * density * sqrt(density) / (frequency * frequency * sqrt(index_squared));
    }
    if (source_frequency == 0) { source_frequency = frequency; }
  }
  let depth = absorption * dl;
  let attenuation = exp(-depth);
  var integral = dl;
  if (depth > 0.001) { integral = (1 - attenuation) / absorption; }
  else { integral *= 1 - depth / 2 + depth * depth / 6; }
  let weight = result.transmission * integral;
  result.intensity += weight * emission;
  result.q += weight * q;
  result.u += weight * u;
  result.transmission *= attenuation;
  if (source_frequency > 0) { result.frequency = source_frequency; }
  return result;
}

/**
 * Limit a backward Mino interval by prescribed material and pattern scales.
 * Rates/epoch must refer to the current physical radius and canonical source time.
 * This sampling estimate is separate from geodesic error control and is not a pixel-beam bound.
 */
fn matter_limit(
  r: f32,
  direction: vec3f,
  radial_speed: f32,
  direction_rate: vec3f,
  epoch: f32,
  time_speed: f32,
  h: f32
) -> f32 {
  if (r <= 0) { return h; }
  var outer = optical_frame.space.w;
  var inner = optical_frame.space.z;
  if (optical_frame.jet_spectrum.w > 0) { outer = max(outer, optical_frame.jet.y); }
  if (optical_frame.jet_spectrum.w > 0) { inner = min(inner, optical_frame.jet.x); }
  if (r < inner) {
    if (radial_speed > 0) { return -min(abs(h), max(0.02 * inner, inner - r) / radial_speed); }
    return h;
  }
  if (r > outer) {
    if (radial_speed > 0) { return -min(abs(h), max(0.02 * outer, r - outer) / radial_speed); }
    return h;
  }
  var step_length = abs(h);
  let angular_speed = length(direction_rate);
  if (optical_frame.material.x > 0 && r >= optical_frame.space.z && r <= optical_frame.space.w) {
    let height = optical_frame.material.x;
    let extent = (4.5 + 2.4 * optical_frame.appearance.z) * height;
    let distance = max(abs(direction.z) - extent, 0.3 * height * max(0.15, disk_envelope(r)));
    if (abs(direction_rate.z) > 0) { step_length = min(step_length, distance / abs(direction_rate.z)); }
    if (abs(direction.z) < extent + height) {
      // Radial and azimuthal structure have their own scales; azimuth is not constrained by H/r.
      if (radial_speed > 0) { step_length = min(step_length, 0.025 * r / radial_speed); }
      if (angular_speed > 0) { step_length = min(step_length, 0.04 / angular_speed); }
      if (optical_frame.appearance.z > 0) {
        let space = optical_frame.space.xy;
        let inner = optical_frame.space.z;
        let root = sqrt(inner - space.y * space.y);
        let lifetime = 12.566370614359172 * (inner * inner + space.x * root) / root;
        let radial_root = sqrt(r - space.y * space.y);
        let denominator = r * r + space.x * radial_root;
        let omega = radial_root / denominator;
        let omega_derivative = abs((r * r / (2 * radial_root) - 2 * r * radial_root) / (denominator * denominator));
        let age = fract(epoch / lifetime);
        let blend = smoothstep(0.0, 1.0, age);
        let weighted_age = lifetime * mix(age, 1 - age, blend);
        let phase_speed = angular_speed + (3 * sqrt(inner / r) / r + weighted_age * omega_derivative) * radial_speed + omega * time_speed;
        let lattice_speed = 7 * radial_speed / r + (4 + 7 * log(r / inner)) * phase_speed;
        // Schedule two midpoint cells against the estimated sheared base-lattice rate.
        // Fine-octave filtering supplements this spacing; it does not certify convergence.
        if (lattice_speed > 0) { step_length = min(step_length, 0.3 / lattice_speed); }
      }
    }
  }
  if (optical_frame.jet_spectrum.w > 0 && r >= optical_frame.jet.x) {
    let height = r * abs(direction.z);
    if (height > optical_frame.jet.x) {
      let width = jet_width(height);
      let transverse = r * length(direction.xy);
      let distance = max(transverse - 1.5 * width, 0.16 * width);
      let speed = radial_speed + r * angular_speed;
      if (speed > 0) { step_length = min(step_length, distance / speed); }
    } else {
      let speed = radial_speed * abs(direction.z) + r * abs(direction_rate.z);
      if (speed > 0) { step_length = min(step_length, max(optical_frame.jet.x - height, 0.1 * optical_frame.jet.x) / speed); }
    }
  }
  return -step_length;
}

/**
 * March a turn-clipped geometric segment with two ordered midpoint cells per interval.
 * Preserve foreground light when leaving material. Stagnation or the 512-interval
 * budget sets valid = false; low transmission permits the declared opacity cutoff.
 */
fn transfer_segment(
  input: Transfer,
  path: KerrOrbit,
  end: KerrOrbitState,
  h: f32,
  block: vec3i,
  screen: PolarizationScreen
) -> Transfer {
  if ((optical_frame.material.x == 0 && optical_frame.jet_spectrum.w == 0) || h == 0) { return input; }
  var inner = optical_frame.space.z;
  var outer = optical_frame.space.w;
  if (optical_frame.material.x == 0) { inner = optical_frame.jet.x; outer = optical_frame.jet.y; }
  if (optical_frame.jet_spectrum.w > 0) { inner = min(inner, optical_frame.jet.x); outer = max(outer, optical_frame.jet.y); }
  let lo = min(path.state.radial.x, end.radial.x);
  let hi = max(path.state.radial.x, end.radial.x);
  let material_lo = select(inner, 1 / outer, path.inverse);
  let material_hi = select(outer, 1 / inner, path.inverse);
  // Geometry clips at radial turns, so a disjoint radial interval contains no material.
  if (hi < material_lo || lo > material_hi) { return input; }
  var result = input;
  let start_rate = kerr_derivative(path, path.state).state;
  let end_rate = kerr_derivative(path, end).state;
  // Bernstein coefficients bound derivatives of the unnormalized cubic.
  // Renormalizing direction afterward makes its use here a local sampling estimate.
  let radial_bound = max(max(abs(start_rate.radial.x), abs(end_rate.radial.x)),
    abs(3 * (end.radial.x - path.state.radial.x) / h - start_rate.radial.x - end_rate.radial.x));
  let direction_bound = max(max(abs(start_rate.direction), abs(end_rate.direction)),
    abs(3 * (end.direction - path.state.direction) / h - start_rate.direction - end_rate.direction));
  let time_bound = max(max(abs(start_rate.radial.z), abs(end_rate.radial.z)),
    abs(3 * (end.radial.z - path.state.radial.z) / h - start_rate.radial.z - end_rate.radial.z));
  var fraction = 0.0;
  var state = path.state;
  for (var cell = 0u; cell < 512u; cell++) {
    // A segment may enter material and then reach infinity. Stop sampling its empty tail.
    if (max(state.radial.x, end.radial.x) < material_lo
      || min(state.radial.x, end.radial.x) > material_hi) { return result; }
    var radius = state.radial.x;
    var speed = radial_bound;
    if (path.inverse) {
      if (radius == 0) { return result; }
      radius = 1 / radius;
      speed *= radius * radius;
    }
    var epoch_speed = time_bound;
    var epoch = state.radial.z;
    if (path.chart < 0 && radius > inner && radius < outer) {
      let delta = radius * radius - 2 * radius + dot(path.space, path.space);
      if (delta > 0) {
        epoch_speed += 2 * (radius * radius + path.space.x * path.space.x) / delta * speed;
        epoch += 2 * kerr_chart_primitives(path.space, radius).y;
      }
    }
    let step = matter_limit(radius, state.direction, speed, direction_bound, epoch, epoch_speed, h * (1 - fraction));
    let next = min(1.0, fraction + abs(step / h));
    if (!(next > fraction)) { result.valid = false; return result; }
    for (var sample_index = 0u; sample_index < 2u; sample_index++) {
      var sample = path;
      let location = mix(fraction, next, (f32(sample_index) + 0.5) / 2);
      sample.state = kerr_interpolate(path.state, end, start_rate, end_rate, h, location);
      var r = sample.state.radial.x;
      if (path.inverse) { if (r == 0) { continue; } r = 1 / r; }
      if (!any(material_support(path.space, r, sample.state.direction))) { continue; }
      let sigma = r * r + path.space.x * path.space.x * sample.state.direction.z * sample.state.direction.z;
      result = transfer_cell(result, sample, kerr_tangent(sample), abs(h) * (next - fraction) * sigma / 2, block, screen);
      if (!result.valid || result.transmission < 0.0001) { return result; }
    }
    if (next == 1) { return result; }
    fraction = next;
    state = kerr_interpolate(path.state, end, start_rate, end_rate, h, fraction);
  }
  result.valid = false;
  return result;
}
