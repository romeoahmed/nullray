import { blackbodyXYZ } from "./radiation.ts";
import type { Star } from "./sky.ts";
import { normalize } from "./vector.ts";
import type { Vec3 } from "./vector.ts";

interface SpectralPoint {
  readonly direction: Vec3;
  readonly coefficient: number;
  readonly temperature: number;
  readonly code: number;
}

/** Spread ten spatial-order bits into every third integer bit. Coordinates themselves stay f32. */
function spread(value: number): number {
  let bits = Math.min(1023, Math.floor((value + 1) * 512));
  bits = (bits | (bits << 16)) & 0x030000ff;
  bits = (bits | (bits << 8)) & 0x0300f00f;
  bits = (bits | (bits << 4)) & 0x030c30c3;
  return (bits | (bits << 2)) & 0x09249249;
}

/** Write one radix subtree from a sorted range; bounds are merged bottom-up in caller-owned storage. */
function writeSubtree(
  nodes: Float32Array,
  points: readonly SpectralPoint[],
  start: number,
  end: number,
  index: number,
): number {
  const first = points[start],
    last = points[end - 1];
  if (!first || !last) {
    throw new Error("A stellar subtree must contain a source.");
  }
  if (end - start === 1) {
    nodes.set([...first.direction, -1, first.coefficient, first.temperature, 0, 0], index * 8);
    return index + 1;
  }
  // A common Morton prefix defines one spatial cell. Identical codes split evenly, retaining every source.
  let middle = (start + end) >>> 1;
  if (first.code !== last.code) {
    const prefix = Math.clz32(first.code ^ last.code);
    let left = start + 1,
      right = end - 1;
    while (left < right) {
      const trial = (left + right) >>> 1;
      const candidate = points[trial];
      if (!candidate) {
        throw new Error("Missing stellar ordering key.");
      }
      if (Math.clz32(first.code ^ candidate.code) > prefix) {
        left = trial + 1;
      } else {
        right = trial;
      }
    }
    middle = left;
  }
  const left = index + 1;
  const right = writeSubtree(nodes, points, start, middle, left);
  const finish = writeSubtree(nodes, points, middle, end, right);
  for (let axis = 0; axis < 3; axis++) {
    const lowLeft = nodes[left * 8 + axis] ?? NaN;
    const lowRight = nodes[right * 8 + axis] ?? NaN;
    const highLeft = nodes[left * 8 + 3] === -1 ? lowLeft : (nodes[left * 8 + 4 + axis] ?? NaN);
    const highRight = nodes[right * 8 + 3] === -1 ? lowRight : (nodes[right * 8 + 4 + axis] ?? NaN);
    nodes[index * 8 + axis] = Math.min(lowLeft, lowRight);
    nodes[index * 8 + 4 + axis] = Math.max(highLeft, highRight);
  }
  nodes[index * 8 + 3] = finish;
  return finish;
}

/**
 * Preorder bounding-volume tree for point sources on the unit sphere.
 * Each 32-byte internal node stores lower/escape and upper bounds. A negative
 * escape index tags a leaf: the same lanes store direction, coefficient, and temperature.
 * Escape indices allow stackless GPU traversal without a per-pixel candidate cap.
 *
 * @param stars - At most 200,000 finite sources; directions need not be unit length.
 * @returns Fresh f32 storage owned by the caller, including a dark leaf for an empty sky.
 * @throws RangeError if a direction, temperature, or spectral coefficient cannot be represented safely.
 */
export function createStarTree(stars: readonly Star[]): Float32Array<ArrayBuffer> {
  if (stars.length > 200_000) {
    throw new RangeError("The point-source catalogue supports at most 200,000 stars.");
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
      code: spread(x) | (spread(y) << 1) | (spread(z) << 2),
    };
  });
  // Together with the finite source variance, this bounds spectral arithmetic in f32.
  const total = sources.reduce((sum, source) => sum + source.coefficient, 0);
  if (total > 1e6) {
    throw new RangeError("The catalogue's total spectral coefficient exceeds 1e6.");
  }
  // A zero-flux leaf keeps the storage binding valid for an empty catalogue.
  if (sources.length === 0) {
    return new Float32Array([0, 0, 1, -1, 0, 6500, 0, 0]);
  }
  const nodes = new Float32Array((2 * sources.length - 1) * 8);
  const ordered = sources.toSorted((a, b) => a.code - b.code);
  writeSubtree(nodes, ordered, 0, ordered.length, 0);
  return nodes;
}
