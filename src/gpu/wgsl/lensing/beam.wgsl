struct SkyBeam {
  dx: vec4f,
  dy: vec4f,
}

/**
 * Test |n·(dx×dy)| > 1e-20 without forming the potentially overflowing area.
 * The same physical threshold governs primary repair, subpixel repair, and diagnostics.
 */
fn celestial_has_area(direction: vec3f, dx: vec3f, dy: vec3f) -> bool {
  let magnitude = max(abs(dx), abs(dy));
  let scale = max(magnitude.x, max(magnitude.y, magnitude.z));
  if (scale == 0.0) { return false; }
  let area = abs(dot(direction, cross(dx / scale, dy / scale)));
  return sqrt(area) > 1e-10 / scale;
}

fn celestial_direction(sample: vec4f) -> vec3f {
  let radial = sqrt(max(0.0, 1.0 - sample.x * sample.x));
  return vec3f(radial * cos(sample.y), radial * sin(sample.y), sample.x);
}

/**
 * Difference two endpoint directions without subtracting nearly equal Cartesian trigonometric values.
 */
fn celestial_difference(after: vec4f, before: vec4f) -> vec3f {
  let radius_after = sqrt((1.0 - after.x) * (1.0 + after.x));
  let radius_before = sqrt((1.0 - before.x) * (1.0 + before.x));
  let sum = radius_after + radius_before;
  let delta_mu = after.x - before.x;
  if (sum == 0.0) { return vec3f(0.0, 0.0, delta_mu); }
  let delta_radius = -(after.x + before.x) * delta_mu / sum;
  let middle = (after.y + before.y) / 2.0;
  let half_difference = (after.y - before.y) / 2.0;
  let radial = delta_radius * cos(half_difference);
  let angular = sum * sin(half_difference);
  return vec3f(radial * cos(middle) - angular * sin(middle),
    radial * sin(middle) + angular * cos(middle), delta_mu);
}

/**
 * One least-squares fit supplies both tangent axes from same-branch samples.
 * Integer screen offsets give an exact normal matrix; rank-one support is unresolved.
 */
fn celestial_stencil_beam(map: texture_2d<f32>, pixel: vec2i, lower: vec2i, upper: vec2i,
  center: vec4f, scale: f32) -> SkyBeam {
  var normal = vec3f(0.0);
  var bx = vec3f(0.0);
  var by = vec3f(0.0);
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let offset = vec2i(x, y);
      let neighbor = pixel + offset;
      if (all(offset == vec2i(0)) || any(neighbor < lower) || any(neighbor >= upper)) { continue; }
      let sample = textureLoad(map, neighbor, 0);
      if (sample.w != center.w) { continue; }
      let delta = celestial_difference(sample, center);
      let screen = vec2f(offset);
      normal += vec3f(screen.x * screen.x, screen.x * screen.y, screen.y * screen.y);
      bx += screen.x * delta;
      by += screen.y * delta;
    }
  }
  let determinant = normal.x * normal.z - normal.y * normal.y;
  if (determinant <= 0.0) { return SkyBeam(vec4f(0.0), vec4f(0.0)); }
  let weight = scale / determinant;
  return SkyBeam(vec4f(weight * (normal.z * bx - normal.y * by), 0.0),
    vec4f(weight * (normal.x * by - normal.y * bx), 0.0));
}

/**
 * W marks stencil availability, not physical coverage. Never difference across event branches.
 */
fn celestial_axis(map: texture_2d<f32>, pixel: vec2i, axis: vec2i, center: vec4f) -> vec4f {
  let limit = vec2i(textureDimensions(map)) - 1;
  let before = textureLoad(map, clamp(pixel - axis, vec2i(0), limit), 0);
  let after = textureLoad(map, clamp(pixel + axis, vec2i(0), limit), 0);
  let left = all(pixel - axis >= vec2i(0)) && before.w == center.w;
  let right = all(pixel + axis <= limit) && after.w == center.w;
  if (left && right) { return vec4f(0.5 * celestial_difference(after, before), 1.0); }
  if (right) { return vec4f(celestial_difference(after, center), 1.0); }
  if (left) { return vec4f(celestial_difference(center, before), 1.0); }
  return vec4f(0.0);
}

fn celestial_beam(map: texture_2d<f32>, pixel: vec2i, center: vec4f) -> SkyBeam {
  let dx = celestial_axis(map, pixel, vec2i(1, 0), center);
  let dy = celestial_axis(map, pixel, vec2i(0, 1), center);
  var fitted = SkyBeam(vec4f(0.0), vec4f(0.0));
  if (dx.w == 0.0 || dy.w == 0.0) {
    fitted = celestial_stencil_beam(map, pixel, vec2i(0), vec2i(textureDimensions(map)), center, 1.0);
  }
  return SkyBeam(vec4f(select(fitted.dx.xyz, dx.xyz, dx.w > 0.0), 0.0),
    vec4f(select(fitted.dy.xyz, dy.xyz, dy.w > 0.0), 0.0));
}

/**
 * Patch spacing is one quarter of an original pixel, so slopes need a factor of four.
 */
fn celestial_patch_beam(map: texture_2d<f32>, pixel: vec2i, origin: vec2i, center: vec4f) -> SkyBeam {
  return celestial_stencil_beam(map, pixel, origin, origin + vec2i(4), center, 4.0);
}
