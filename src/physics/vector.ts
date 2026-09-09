/** Cartesian components in a documented orthonormal frame. */
export type Vec3 = readonly [number, number, number];

/** Euclidean scalar product in a shared orthonormal frame. */
export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** Right-handed vector product in a shared orthonormal frame. */
export function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/**
 * Return a unit vector without changing the input.
 * @throws RangeError when the Euclidean length is zero or nonfinite.
 */
export function normalize(v: Vec3): Vec3 {
  const length = Math.hypot(...v);
  if (!(length > 0) || !Number.isFinite(length)) {
    throw new RangeError("A direction must have finite, nonzero length.");
  }
  return [v[0] / length, v[1] / length, v[2] / length];
}
