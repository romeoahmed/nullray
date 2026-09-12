/**
 * Summarize finite pixel values and missing weights after benchmark timing.
 *
 * @returns Pixel count, missing-weight sum, and absolute RGB sum. The latter is
 * a sanity statistic, not physical integrated flux or a reference-image error.
 * @throws Error - If a pixel is nonfinite or its coverage is outside [0,1].
 */
export function imageCoverage(pixels: Float32Array) {
  let unresolvedWeight = 0;
  let radiance = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    const [r = NaN, g = NaN, b = NaN, weight = NaN] = pixels.subarray(i, i + 4);
    if (![r, g, b, weight].every(Number.isFinite) || weight < 0 || weight > 1) {
      throw new Error("The rendered benchmark image contains invalid data.");
    }
    unresolvedWeight += 1 - weight;
    radiance += Math.abs(r) + Math.abs(g) + Math.abs(b);
  }
  return { pixels: pixels.length / 4, unresolvedWeight, radiance };
}
