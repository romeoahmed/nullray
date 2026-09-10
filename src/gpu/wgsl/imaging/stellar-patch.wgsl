/** A normalized bilinear sky map over one physical-pixel square. */
struct StellarPatch {
  lower_left: vec4f,
  lower_right: vec4f,
  upper_left: vec4f,
  upper_right: vec4f,
}

struct StellarRoots {
  positions: array<vec2f, 2>,
  count: u32,
  valid: bool,
}

fn stellar_cross(a: vec2f, b: vec2f) -> f32 {
  return a.x * b.y - a.y * b.x;
}

/** Exact product of two binary32 significands, with a separate signed exponent. */
struct StellarProduct {
  words: vec2u,
  exponent: i32,
  negative: bool,
}

fn stellar_product(a: f32, b: f32) -> StellarProduct {
  let bits_a = bitcast<u32>(a);
  let bits_b = bitcast<u32>(b);
  let exponent_a = (bits_a >> 23u) & 255u;
  let exponent_b = (bits_b >> 23u) & 255u;
  let ma = (bits_a & 0x7fffffu) | select(0u, 0x800000u, exponent_a > 0u);
  let mb = (bits_b & 0x7fffffu) | select(0u, 0x800000u, exponent_b > 0u);
  let p0 = (ma & 65535u) * (mb & 65535u);
  let p1 = (ma >> 16u) * (mb & 65535u);
  let p2 = (ma & 65535u) * (mb >> 16u);
  let middle = (p0 >> 16u) + (p1 & 65535u) + (p2 & 65535u);
  var low = (p0 & 65535u) | (middle << 16u);
  var high = (ma >> 16u) * (mb >> 16u) + (p1 >> 16u) + (p2 >> 16u) + (middle >> 16u);
  if ((low | high) == 0u) { return StellarProduct(vec2u(0u), 0, false); }
  var width = 32u - countLeadingZeros(low);
  if (high != 0u) { width = 64u - countLeadingZeros(high); }
  let shift = 48u - width;
  if (shift >= 32u) { high = low << (shift - 32u); low = 0u; }
  else if (shift > 0u) { high = (high << shift) | (low >> (32u - shift)); low <<= shift; }
  let exponent = i32(max(exponent_a, 1u)) + i32(max(exponent_b, 1u)) - 300 + i32(width);
  return StellarProduct(vec2u(low, high), exponent, ((bits_a ^ bits_b) & 0x80000000u) != 0u);
}

/**
 * Canonical shared-edge orientation. The fast determinant handles separated
 * products; 48-bit integer products decide cancellation and underflow exactly
 * for the already rounded binary32 chart coordinates.
 */
fn stellar_orientation(a: vec2f, b: vec2f) -> i32 {
  let first = a.x * b.y;
  let second = a.y * b.x;
  let difference = first - second;
  // Below 2^-106 even the floating error bound can underflow; use words there.
  if (min(abs(first), abs(second)) >= 1.232595164407831e-32 &&
      abs(difference) > 9.5367431640625e-7 * (abs(first) + abs(second))) {
    return select(-1, 1, difference > 0.0);
  }
  let p = stellar_product(a.x, b.y);
  let q = stellar_product(a.y, b.x);
  let p_zero = all(p.words == vec2u(0u));
  let q_zero = all(q.words == vec2u(0u));
  if (p_zero && q_zero) { return 0; }
  if (p_zero) { return select(-1, 1, q.negative); }
  if (q_zero) { return select(1, -1, p.negative); }
  if (p.negative != q.negative) { return select(1, -1, p.negative); }
  var comparison = 0;
  if (p.exponent != q.exponent) { comparison = select(-1, 1, p.exponent > q.exponent); }
  else if (p.words.y != q.words.y) { comparison = select(-1, 1, p.words.y > q.words.y); }
  else if (p.words.x != q.words.x) { comparison = select(-1, 1, p.words.x > q.words.x); }
  return select(comparison, -comparison, p.negative);
}

struct StellarParameters {
  values: array<f32, 2>,
  count: u32,
  valid: bool,
}

/**
 * Select roots in [0,1) using polynomial signs at the two canonical cell edges,
 * not a rounded root coordinate. A boundary root is factored exactly before
 * inversion. This prevents adjoining cells from both owning a rounded 1-ulp.
 */
fn stellar_parameters(coefficients: vec3f, edge_signs: vec2i) -> StellarParameters {
  var result: StellarParameters;
  result.valid = true;
  let scale = max(max(abs(coefficients.x), abs(coefficients.y)), abs(coefficients.z));
  if (!(scale > 0.0)) { result.valid = false; return result; }
  var polynomial = coefficients / scale;
  if (edge_signs.x == 0) { polynomial.z = 0.0; }
  if (edge_signs.y == 0) { polynomial.y = -polynomial.x - polynomial.z; }
  if (polynomial.x == 0.0) {
    if (polynomial.y == 0.0) {
      if (polynomial.z == 0.0) { result.valid = false; }
      return result;
    }
    let orientation = select(-1, 1, polynomial.y > 0.0);
    if (edge_signs.x * orientation > 0 || edge_signs.y * orientation <= 0) { return result; }
    if (abs(polynomial.z) > 2.0 * abs(polynomial.y)) { result.valid = false; return result; }
    let value = -polynomial.z / polynomial.y;
    if (value < 0.0 || value > 1.0) { result.valid = false; return result; }
    result.values[0] = value;
    result.count = 1u;
    return result;
  }
  let square = polynomial.y * polynomial.y;
  let product = 4.0 * polynomial.x * polynomial.z;
  let discriminant = square - product;
  let uncertainty = 3.814697265625e-6 * (square + abs(product));
  if (discriminant < -uncertainty) { return result; }
  if (discriminant <= uncertainty) { result.valid = false; return result; }
  let root = sqrt(discriminant);
  let q = -0.5 * (polynomial.y + select(root, -root, polynomial.y < 0.0));
  let candidates = array<vec2f, 2>(vec2f(q, polynomial.x), vec2f(polynomial.z, q));
  let orientation = select(-1, 1, polynomial.x > 0.0);
  let edges = edge_signs * orientation;
  let vertex_after_left = polynomial.y * f32(orientation) < 0.0;
  let vertex_before_right = (2.0 * polynomial.x + polynomial.y) * f32(orientation) > 0.0;
  let first_is_lower = (polynomial.y >= 0.0) == (polynomial.x > 0.0);
  for (var index = 0u; index < 2u; index++) {
    let lower = (index == 0u) == first_is_lower;
    var after_left = vertex_after_left;
    if (edges.x < 0) { after_left = !lower; }
    else if (edges.x == 0) { after_left = vertex_after_left || !lower; }
    var before_right = vertex_before_right;
    if (edges.y < 0) { before_right = lower; }
    else if (edges.y == 0) { before_right = vertex_before_right && lower; }
    if (!after_left || !before_right) { continue; }
    let quotient = candidates[index];
    if (quotient.y == 0.0 || abs(quotient.x) > 2.0 * abs(quotient.y)) { result.valid = false; return result; }
    let value = quotient.x / quotient.y;
    if (value < 0.0 || value > 1.0) { result.valid = false; return result; }
    result.values[result.count] = value;
    result.count += 1u;
  }
  return result;
}

/**
 * Solve both projected bilinear equations, retaining both roots of a folded
 * cell. The largest source component defines a division-safe tangent chart.
 * Both elimination polynomials use canonical shared-edge ownership; a final
 * two-equation residual pairs their scalar roots without cross-cell snapping.
 */
fn stellar_patch_roots(cell: StellarPatch, direction: vec3f) -> StellarRoots {
  var result: StellarRoots;
  result.valid = true;
  var axis = 0u;
  if (abs(direction.y) > abs(direction.x)) { axis = 1u; }
  if (abs(direction.z) > abs(direction[axis])) { axis = 2u; }
  if (direction[axis] == 0.0) { result.valid = false; return result; }
  let first = (axis + 1u) % 3u;
  let second = (axis + 2u) % 3u;
  let chart = vec2f(direction[first], direction[second]) / direction[axis];
  let p0 = vec2f(cell.lower_left[first], cell.lower_left[second]) - chart * cell.lower_left[axis];
  let p1 = vec2f(cell.lower_right[first], cell.lower_right[second]) - chart * cell.lower_right[axis];
  let p2 = vec2f(cell.upper_left[first], cell.upper_left[second]) - chart * cell.upper_left[axis];
  let p3 = vec2f(cell.upper_right[first], cell.upper_right[second]) - chart * cell.upper_right[axis];
  let lower = min(min(p0, p1), min(p2, p3));
  let upper = max(max(p0, p1), max(p2, p3));
  // A bilinear interpolation is a convex combination of its four corners.
  // Widen this rejection bound for the tangent-chart subtraction only.
  let projection_roundoff = 9.5367431640625e-7;
  if (any(lower > vec2f(projection_roundoff)) || any(upper < vec2f(-projection_roundoff))) { return result; }
  let extent = max(max(abs(p0), abs(p1)), max(abs(p2), abs(p3)));
  let scale = max(extent.x, extent.y);
  if (!(scale > 0.0)) { result.valid = false; return result; }
  let a = p0 / scale;
  let b = (p1 - p0) / scale;
  let c = (p2 - p0) / scale;
  let d = ((p3 - p2) - (p1 - p0)) / scale;
  let x = stellar_parameters(vec3f(stellar_cross(b, d), stellar_cross(a, d) + stellar_cross(b, c), stellar_cross(a, c)),
    vec2i(stellar_orientation(p0, p2), stellar_orientation(p1, p3)));
  if (x.valid && x.count == 0u) { return result; }
  let y = stellar_parameters(vec3f(stellar_cross(c, d), stellar_cross(a, d) + stellar_cross(c, b), stellar_cross(a, b)),
    vec2i(stellar_orientation(p0, p1), stellar_orientation(p2, p3)));
  if (y.valid && y.count == 0u) { return result; }
  if (!x.valid || !y.valid) { result.valid = false; return result; }
  var used_x = 0u;
  var used_y = 0u;
  for (var ix = 0u; ix < x.count; ix++) {
    for (var iy = 0u; iy < y.count; iy++) {
      let u = x.values[ix];
      let v = y.values[iy];
      let residual = a + b * u + c * v + d * u * v;
      let bound = 7.62939453125e-6 * (abs(a) + abs(b * u) + abs(c * v) + abs(d * u * v));
      if (any(abs(residual) > bound)) { continue; }
      // At a fixed u the equations are linear in v, and conversely. Multiple
      // matches along either axis indicate unresolved pairing or a continuum.
      if ((used_x & (1u << ix)) != 0u || (used_y & (1u << iy)) != 0u) { result.valid = false; return result; }
      used_x |= 1u << ix;
      used_y |= 1u << iy;
      result.positions[result.count] = vec2f(u, v);
      result.count += 1u;
    }
  }
  return result;
}

/** Sum the two possible images of one source, each at its own Jacobian and energy. */
fn stellar_patch_flux(cell: StellarPatch, origin: vec2f, node: StarNode, footprint: StellarFootprint) -> vec4f {
  var color = vec3f(0.0);
  let roots = stellar_patch_roots(cell, node.lower.xyz);
  if (!roots.valid) { return vec4f(0.0); }
  for (var root = 0u; root < roots.count; root++) {
    let position = roots.positions[root];
    if (stellar_filtered_image(origin + position, footprint)) { continue; }
    let weight = max(vec2f(0.0), 1.0 - abs(origin + position));
    if (any(weight == vec2f(0.0))) { continue; }
    let bottom = mix(cell.lower_left, cell.lower_right, position.x);
    let top = mix(cell.upper_left, cell.upper_right, position.x);
    let value = mix(bottom, top, position.y);
    let energy = value.w;
    if (!(energy > 0.0) || energy < node.upper.y / 1000000.0) { return vec4f(0.0); }
    let spectrum = blackbody_radiance(node.upper.y / energy);
    if (spectrum.a == 0.0) { return vec4f(0.0); }
    let dx = mix(cell.lower_right.xyz - cell.lower_left.xyz,
      cell.upper_right.xyz - cell.upper_left.xyz, position.y);
    let dy = top.xyz - bottom.xyz;
    let extent = max(abs(dx), abs(dy));
    let scale = max(max(extent.x, extent.y), extent.z);
    let value_scale = max(max(abs(value.x), abs(value.y)), abs(value.z));
    if (!(scale > 0.0) || !(value_scale > 0.0)) { return vec4f(0.0); }
    let scaled_value = value.xyz / value_scale;
    let norm = length(scaled_value) * value_scale;
    let direction = normalize(scaled_value);
    if (!(dot(direction, node.lower.xyz) > 0.0)) { continue; }
    let area_root = sqrt(abs(dot(direction, cross(dx / scale, dy / scale))));
    // J = area * (scale / |value|)^2. Its reciprocal square root is
    // bounded by the existing J > 1e-20 floor before amplification.
    if (!(area_root * scale > 1e-10 * norm)) { return vec4f(0.0); }
    let inverse_root = (norm / scale) / area_root;
    color += node.upper.x * spectrum.rgb * weight.x * weight.y * inverse_root * inverse_root;
  }
  return vec4f(color, 1.0);
}

/**
 * Integrate all point images of one cell with a unit-integral detector tent.
 * Corner xyz is a unit source direction; w is unnormalized Killing energy.
 * Origin is the cell's lower-left corner relative to the detector pixel.
 */
fn point_stars_patch(cell: StellarPatch, origin: vec2f, footprint: StellarFootprint) -> vec4f {
  let sum = cell.lower_left.xyz + cell.lower_right.xyz + cell.upper_left.xyz + cell.upper_right.xyz;
  let magnitude = max(max(abs(sum.x), abs(sum.y)), abs(sum.z));
  if (!(magnitude > 0.0)) { return vec4f(0.0); }
  let axis = normalize(sum / magnitude);
  let alignment = min(min(dot(axis, cell.lower_left.xyz), dot(axis, cell.lower_right.xyz)),
    min(dot(axis, cell.upper_left.xyz), dot(axis, cell.upper_right.xyz)));
  // A common open hemisphere excludes a zero homogeneous vector everywhere.
  if (!(alignment > 0.0)) { return vec4f(0.0); }
  // A normalized convex combination stays in the corner spherical cap.
  // Direct chords avoid cancellation in 2*(1-cos(angle)) for narrow cells.
  // Eight f32 epsilons widen only the chord work bound, never the image.
  let d0 = cell.lower_left.xyz - axis;
  let d1 = cell.lower_right.xyz - axis;
  let d2 = cell.upper_left.xyz - axis;
  let d3 = cell.upper_right.xyz - axis;
  let radius = sqrt(max(max(dot(d0, d0), dot(d1, d1)), max(dot(d2, d2), dot(d3, d3)))) + 9.5367431640625e-7;
  let radius_squared = radius * radius;
  var color = vec3f(0.0);
  var index = 0u;
  loop {
    if (index >= arrayLength(&star_nodes)) { break; }
    let node = star_nodes[index];
    if (node.lower.w >= 0.0) {
      let nearest = clamp(axis, node.lower.xyz, node.upper.xyz);
      let distance = nearest - axis;
      if (dot(distance, distance) > radius_squared) { index = u32(node.lower.w); continue; }
    } else if (node.upper.x > 0.0 && dot(axis, node.lower.xyz) > 0.0) {
      let distance = node.lower.xyz - axis;
      if (dot(distance, distance) > radius_squared) { index += 1u; continue; }
      let contribution = stellar_patch_flux(cell, origin, node, footprint);
      if (contribution.a == 0.0) { return vec4f(0.0); }
      color += contribution.rgb;
    }
    index += 1u;
  }
  return vec4f(color, 1.0);
}
