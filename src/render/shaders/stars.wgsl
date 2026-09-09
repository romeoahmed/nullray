/** 32-byte stackless tree node; CPU packing and escape indices live in physics/stars.ts. */
struct StarNode {
  lower: vec4f,
  upper: vec4f,
}

/**
 * Unit-integral screen-space tent applied to a locally affine source mapping.
 * Source flux / source solid angle already includes lensing amplification.
 * The result's alpha is zero when the local beam cannot be inverted safely.
 */
fn point_stars(direction: vec3f, gradient_x: vec3f, gradient_y: vec3f, energy: f32) -> vec4f {
  if (!(energy > 0.0)) { return vec4f(0.0); }
  let dx = gradient_x - direction * dot(direction, gradient_x);
  let dy = gradient_y - direction * dot(direction, gradient_y);
  let extent = max(abs(dx), abs(dy));
  let scale = max(max(extent.x, extent.y), extent.z);
  if (!(scale > 0.0)) { return vec4f(0.0); }
  // Normalize before cross products: differentiated beams can have large slopes
  // even when their inverse is small. J = scale² * area is never formed directly.
  let sx = dx / scale;
  let sy = dy / scale;
  let area = dot(direction, cross(sx, sy));
  if (!(sqrt(abs(area)) * scale > 1e-10)) { return vec4f(0.0); }
  let denominator = area * scale;
  let inverse_x = cross(sy, direction);
  let inverse_y = cross(direction, sx);
  // Maximum tangent displacement occurs at a corner of the tent support.
  // Its norm bounds the normalized source chord; the forward hemisphere also
  // bounds squared chord distance by two. Guard the square before evaluating it.
  let plus = sx + sy;
  let minus = sx - sy;
  let radius = scale * sqrt(max(dot(plus, plus), dot(minus, minus)));
  var radius_squared = 2.0;
  if (radius < 1.4142135623730951) { radius_squared = radius * radius; }
  var radiance = vec3f(0.0);
  var index = 0u;
  loop {
    if (index >= arrayLength(&star_nodes)) { break; }
    let node = star_nodes[index];
    if (node.lower.w >= 0.0) {
      let nearest = clamp(direction, node.lower.xyz, node.upper.xyz);
      let offset = nearest - direction;
      if (dot(offset, offset) > radius_squared) {
        index = u32(node.lower.w);
        continue;
      }
    } else {
      let alignment = dot(node.lower.xyz, direction);
      if (alignment > 0.0) {
        let tangent = node.lower.xyz - alignment * direction;
        let numerator = vec2f(dot(tangent, inverse_x), dot(tangent, inverse_y));
        // Reject outside support before dividing by a possibly tiny alignment.
        if (all(abs(numerator) < vec2f(abs(denominator) * alignment))) {
          let weight = 1.0 - abs((numerator / denominator) / alignment);
          if (node.upper.x > 0.0) {
            // Guard the lookup domain before division; exhausted spectral range is unresolved.
            if (energy < node.upper.y / 1000000.0) { return vec4f(0.0); }
            let spectrum = blackbody_radiance(node.upper.y / energy);
            if (spectrum.a == 0.0) { return vec4f(0.0); }
            radiance += node.upper.x * spectrum.rgb * weight.x * weight.y;
          }
        }
      }
    }
    index += 1u;
  }
  return vec4f((radiance / scale) / abs(denominator), 1.0);
}
