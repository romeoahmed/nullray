/** Live frames have no temporal history; settled views and photographs freeze their epoch. */
export type SamplingMode = "live" | "settle" | "photograph";

/** Sample budgets per unchanged epoch/raster; spatial resolution is a separate policy. */
export const sampleCounts = { live: 1, settle: 16, photograph: 64 } as const;

/**
 * Select deterministic pixel-box quadrature for the current image sequence.
 *
 * @remarks
 * Live uses the center, settling uses 4×4 stratum centers, and photography
 * uses the first 64 base-2/base-3 Halton points. No random state or reprojection
 * is involved; finite quadrature does not guarantee every subpixel image is sampled.
 *
 * @param index - Zero-based integer below the mode's sample count.
 * @returns Horizontal/vertical offsets in pixel units relative to the pixel center.
 * @throws RangeError - If the sample index is outside the sequence.
 */
export function sampleOffset(mode: SamplingMode, index: number): readonly [number, number] {
  if (!Number.isInteger(index) || index < 0 || index >= sampleCounts[mode]) {
    throw new RangeError("The sample index must belong to the current image sequence.");
  }
  if (mode === "live") {
    return [0, 0];
  }
  if (mode === "settle") {
    // Interleave quadrants so early partial means cover the pixel before filling each cell.
    const x = ((index & 1) << 1) | ((index >> 2) & 1);
    const y = (((index >> 1) & 1) << 1) | ((index >> 3) & 1);
    return [(x + 0.5) / 4 - 0.5, (y + 0.5) / 4 - 0.5];
  }
  return pixelJitter(index);
}

/** Base-2/base-3 radical inverses shifted by −1/2; a finite sequence need not have zero mean. */
function pixelJitter(index: number): readonly [number, number] {
  const radicalInverse = (base: number) => {
    let integer = index + 1;
    let fraction = 1 / base;
    let value = 0;
    while (integer > 0) {
      value += (integer % base) * fraction;
      integer = Math.floor(integer / base);
      fraction /= base;
    }
    return value - 0.5;
  };
  return [radicalInverse(2), radicalInverse(3)];
}
