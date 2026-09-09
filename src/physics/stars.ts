import { blackbodyXYZ } from "./radiation.ts";
import type { Star } from "./sky.ts";
import { normalize } from "./vector.ts";
import type { Vec3 } from "./vector.ts";

interface SpectralPoint {
  readonly direction: Vec3;
  readonly coefficient: number;
  readonly temperature: number;
}

const axes = [0, 1, 2] as const;

/** Bounds of a nonempty set of already quantized directions. */
function bounds(points: readonly SpectralPoint[]): readonly [Vec3, Vec3] {
  const lower: [number, number, number] = [Infinity, Infinity, Infinity];
  const upper: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (const { direction } of points) {
    for (const axis of axes) {
      lower[axis] = Math.min(lower[axis], direction[axis]);
      upper[axis] = Math.max(upper[axis], direction[axis]);
    }
  }
  return [lower, upper];
}

/** Write one preorder subtree into caller-owned storage; return its exclusive end node. */
function writeSubtree(
  nodes: Float32Array,
  points: readonly SpectralPoint[],
  index: number,
): number {
  const [source] = points;
  if (!source) {
    throw new Error("A stellar subtree must contain a source.");
  }
  if (points.length === 1) {
    nodes.set(
      [...source.direction, -(index + 1), source.coefficient, source.temperature, 0, 0],
      index * 8,
    );
    return index + 1;
  }
  const [lower, upper] = bounds(points);
  const axis = axes.reduce((widest, candidate) =>
    upper[candidate] - lower[candidate] > upper[widest] - lower[widest] ? candidate : widest,
  );
  const ordered = points.toSorted((left, right) => left.direction[axis] - right.direction[axis]);
  const middle = Math.floor(ordered.length / 2);
  const right = writeSubtree(nodes, ordered.slice(0, middle), index + 1);
  const end = writeSubtree(nodes, ordered.slice(middle), right);
  nodes.set([...lower, end, ...upper, 0], index * 8);
  return end;
}

/**
 * Preorder bounding-volume tree for point sources on the unit sphere.
 * Each 32-byte internal node stores lower/escape and upper bounds. A negative
 * escape index tags a leaf: the same lanes store direction, coefficient, and temperature.
 * Escape indices allow stackless GPU traversal without a per-pixel candidate cap.
 *
 * @param stars - At most 100,000 finite sources; directions need not be unit length.
 * @returns Fresh f32 storage owned by the caller, including a dark leaf for an empty sky.
 * @throws RangeError if a direction, temperature, or spectral coefficient cannot be represented safely.
 */
export function createStarTree(stars: readonly Star[]): Float32Array<ArrayBuffer> {
  if (stars.length > 100_000) {
    throw new RangeError("The point-source catalogue supports at most 100,000 stars.");
  }
  const reference = blackbodyXYZ(6500)[1];
  // Catalogue B−V values repeat. Reuse exact spectral integrals locally, with no
  // temperature quantization, interpolated approximation, or persistent cache.
  const luminance = new Map<number, number>([[6500, reference]]);

  const sources: readonly SpectralPoint[] = stars.map((star) => {
    if (
      !Number.isFinite(star.flux) ||
      star.flux < 0 ||
      !Number.isFinite(star.temperature) ||
      star.temperature < 1000 ||
      star.temperature > 50000
    ) {
      throw new RangeError(
        "A star requires finite nonnegative flux and a color temperature in [1000, 50000] K.",
      );
    }
    const [x, y, z] = normalize(star.direction);
    const coefficient =
      star.flux /
      (luminance.getOrInsertComputed(
        star.temperature,
        (temperature) => blackbodyXYZ(temperature)[1],
      ) /
        reference);
    if (!Number.isFinite(Math.fround(coefficient))) {
      throw new RangeError("Stellar flux exceeds f32 storage.");
    }
    return {
      direction: [Math.fround(x), Math.fround(y), Math.fround(z)],
      coefficient,
      temperature: star.temperature,
    };
  });
  // Together with the shader's determinant guard, this bounds spectral arithmetic in f32.
  const total = sources.reduce((sum, source) => sum + source.coefficient, 0);
  if (total > 1e6) {
    throw new RangeError("The catalogue's total spectral coefficient exceeds 1e6.");
  }
  // A zero-flux leaf keeps the storage binding valid for an empty catalogue.
  if (sources.length === 0) {
    return new Float32Array([0, 0, 1, -1, 0, 6500, 0, 0]);
  }
  const nodes = new Float32Array((2 * sources.length - 1) * 8);
  writeSubtree(nodes, sources, 0);
  return nodes;
}
