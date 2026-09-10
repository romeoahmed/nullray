import { milne } from "../data/milne.ts";

/**
 * Uniform μ samples of the gray Milne atmosphere: normalized I, polarized intensity, two pads.
 * Linear interpolation acts on Stokes intensities. The hemispheric flux is πBν(T), preserving
 * the effective temperature of the zero-torque disk. Returned storage belongs to the caller.
 */
export function createAtmosphereTable(size = 256): Float32Array<ArrayBuffer> {
  if (!Number.isSafeInteger(size) || size < 2) {
    throw new RangeError("An atmosphere table needs at least two angular samples.");
  }
  let integral = 0;
  for (let i = 1; i < milne.length; i++) {
    const before = milne[i - 1];
    const after = milne[i];
    if (!before || !after) {
      throw new Error("Missing atmosphere data.");
    }
    const [mu, intensity] = before;
    const width = after[0] - mu;
    const change = after[1] - intensity;
    integral +=
      width * (mu * intensity + (mu * change + width * intensity) / 2 + (width * change) / 3);
  }
  const table = new Float32Array(size * 4);
  let segment = 1;
  for (let i = 0; i < size; i++) {
    const mu = i / (size - 1);
    while (segment < milne.length - 1 && mu > (milne[segment]?.[0] ?? 1)) {
      segment++;
    }
    const before = milne[segment - 1];
    const after = milne[segment];
    if (!before || !after) {
      throw new Error("Missing atmosphere data.");
    }
    const fraction = (mu - before[0]) / (after[0] - before[0]);
    table[4 * i] = (before[1] + fraction * (after[1] - before[1])) / (2 * integral);
    const polarized = before[1] * before[2];
    table[4 * i + 1] =
      (polarized + fraction * (after[1] * after[2] - polarized)) / (200 * integral);
  }
  return table;
}
