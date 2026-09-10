/** Same six vec4 lanes as the optical frame, without importing the ray solver. */
struct StellarView {
  viewport: vec4f,
  observer: vec4f,
  spacetime: vec4f,
  camera_forward: vec4f,
  camera_up: vec4f,
  camera_right: vec4f,
}
@group(0) @binding(19) var<uniform> stellar_view: StellarView;
/** Sorted tangent angle, radial/tangent ratio, reserved, valid. */
@group(0) @binding(20) var<storage, read> stellar_partition: array<vec4f>;

/** Physical screen origin and number of physical pixels per local filter coordinate. */
struct StellarFootprint {
  origin: vec2f,
  scale: f32,
}

/**
 * A shared computational partition, evaluated at each image rather than its
 * receiving pixel. The interpolant defines ownership; it is not a replacement
 * for the critical curve or a claim that the finite search is complete.
 */
fn stellar_search_owns(position: vec2f) -> bool {
  let count = arrayLength(&stellar_partition);
  if (count < 3u || stellar_partition[0].w == 0.0) { return false; }
  let screen = (position + 0.5 - 0.5 * stellar_view.viewport.xy) *
    (stellar_view.observer.z / stellar_view.viewport.y);
  let local = stellar_view.camera_forward.xyz + screen.x * stellar_view.camera_right.xyz -
    screen.y * stellar_view.camera_up.xyz;
  let tangent = length(local.yz);
  if (!(tangent > 0.0)) { return false; }
  var angle = atan2(local.z, local.y);
  if (angle < stellar_partition[0].x) { angle += 6.283185307179586; }
  var lower = 0u;
  var upper = count;
  loop {
    if (upper - lower <= 1u) { break; }
    let middle = lower + (upper - lower) / 2u;
    if (stellar_partition[middle].x <= angle) { lower = middle; }
    else { upper = middle; }
  }
  let first = stellar_partition[lower];
  var second = stellar_partition[upper % count];
  if (upper == count) { second.x += 6.283185307179586; }
  if (first.w == 0.0 || second.w == 0.0 || !(second.x > first.x)) { return false; }
  let ratio = mix(first.y, second.y, (angle - first.x) / (second.x - first.x));
  let ratio_scale = max(1.0, abs(ratio));
  let critical = normalize(vec2f(ratio / ratio_scale, 1.0 / ratio_scale));
  // Cross-multiply the signed offset test. Neither a nearly radial local ray
  // nor a large critical ratio requires an overflowing quotient or square.
  let displacement = local.x * critical.y - tangent * critical.x;
  // Both sides include a margin around the interpolated vacuum separatrix.
  // The searched escape strip extends to +0.25; image transport still decides visibility.
  return displacement >= -0.01 * tangent && displacement < 0.05 * tangent;
}

fn stellar_filtered_image(position: vec2f, footprint: StellarFootprint) -> bool {
  if (!(footprint.scale > 0.0)) { return false; }
  return stellar_search_owns(footprint.origin + position * footprint.scale);
}
