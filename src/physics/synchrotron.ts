import { cie1931 } from "../data/cie1931.ts";
import { blackbodyXYZ, xyzToLinearRGB } from "./radiation.ts";
import type { Jet } from "./jet.ts";

/**
 * Build a local lookup of the normalized cumulative single-electron kernel.
 *
 * @remarks
 * Integrates F(x) in log-frequency cells, using the low-frequency power-law
 * limit below the grid and a saturated cumulative integral above it.
 * The returned closure owns the table and expects positive x.
 */
function synchrotronCumulative(): (x: number) => number {
  const lower = -20,
    upper = Math.log(100),
    count = 512;
  const step = (upper - lower) / (count - 1);
  const integral = new Float64Array(count);
  let previous = 0;
  let sum = 0;
  for (let i = 0; i < count; i++) {
    const x = Math.exp(lower + i * step);
    // F(x)=x ∫ exp(-x cosh t) cosh(5t/3)/cosh(t) dt, after exchanging the Bessel integrals.
    const limit = Math.acosh(80 / x + 1);
    const spacing = limit / 128;
    let kernel = 0;
    for (let j = 0; j <= 128; j++) {
      const t = j * spacing;
      const cosh = Math.cosh(t);
      kernel +=
        ((j === 0 || j === 128 ? 1 : j % 2 === 0 ? 2 : 4) *
          Math.exp(-x * cosh) *
          Math.cosh((5 * t) / 3)) /
        cosh;
    }
    const value = (x * x * kernel * spacing) / 3;
    if (i > 0) {
      sum += ((value + previous) * step) / 2;
    } else {
      sum = value * 0.75;
    }
    integral[i] = sum;
    previous = value;
  }
  const normalization = integral[count - 1] ?? 1;
  return (x: number) => {
    const coordinate = (Math.log(x) - lower) / step;
    if (coordinate <= 0) {
      return ((integral[0] ?? 0) / normalization) * Math.exp((4 * coordinate * step) / 3);
    }
    if (coordinate >= count - 1) {
      return 1;
    }
    const i = Math.floor(coordinate);
    const a = integral[i] ?? 0,
      b = integral[i + 1] ?? 0;
    return (a + (b - a) * (coordinate - i)) / normalization;
  };
}

/**
 * Tabulate finite-cutoff synchrotron emission for isotropic p = 3 electrons.
 *
 * @remarks
 * Uses a tangled field and converts one M of path length using the black-hole
 * mass. Rows span the dimensionless frequency interval exp(-10) to exp(10).
 * The GPU supplies the remaining density, frequency, and refractive factors.
 *
 * @param jet - Validated source prescription, including positive field and electron cutoff.
 * @param frequencyGHz - Optional positive observing frequency in GHz; selects a
 * monochrome spectrum normalized to a 6500 K blackbody instead of CIE color.
 * @returns Caller-owned f32 RGB/padding rows and their frequency-coordinate scale.
 * @throws RangeError - If a stored spectrum is nonfinite.
 */
export function createJetTable(
  jet: Jet,
  frequencyGHz?: number,
): { readonly data: Float32Array<ArrayBuffer>; readonly scale: number } {
  const cumulative = synchrotronCumulative();
  const normalization = blackbodyXYZ(6500)[1];
  const electronSquared = 2.307077552e-19;
  const cyclotron = 2.799248987e6 * jet.field;
  const light = 2.99792458e10;
  const radius = 1.476625038e5 * jet.massSolar;
  const coefficient =
    ((((4 * Math.PI) / 9) * electronSquared * cyclotron ** 2) / light) *
    jet.density *
    jet.gammaMin ** 2 *
    radius;
  const cutoff = (frequency: number) => {
    let average = 0;
    // Uniform cos(theta) quadrature averages an isotropic field without a polar singularity.
    for (let i = 0; i < 12; i++) {
      const cosine = (i + 0.5) / 12;
      const sineSquared = 1 - cosine * cosine;
      average +=
        sineSquared *
        cumulative(frequency / (1.5 * cyclotron * Math.sqrt(sineSquared) * jet.gammaMin ** 2));
    }
    return average / 8;
  };
  const scale = ((frequencyGHz ?? 500000) * 1e9) / (1.5 * cyclotron * jet.gammaMin ** 2);
  const count = 256;
  const table = new Float32Array(count * 4);
  for (let row = 0; row < count; row++) {
    const shift = Math.exp(-10 + (20 * row) / (count - 1)) / scale;
    if (frequencyGHz !== undefined) {
      const frequency = frequencyGHz * 1e9;
      const thermal =
        (2 * 6.62607015e-27 * frequency ** 3) /
        light ** 2 /
        Math.expm1((6.62607015e-27 * frequency) / (1.380649e-16 * 6500));
      const value = ((coefficient / (frequency * shift)) * cutoff(frequency * shift)) / thermal;
      table.set([value, value, value, 0], row * 4);
    } else {
      let x = 0,
        y = 0,
        z = 0;
      for (const [i, match] of cie1931.entries()) {
        const wavelength = 360 + i;
        const weight =
          (((i === 0 || i === cie1931.length - 1 ? 0.5 : 1) * coefficient * 1e-3) /
            (normalization * wavelength * shift)) *
          cutoff((2.99792458e17 / wavelength) * shift);
        x += weight * match[0];
        y += weight * match[1];
        z += weight * match[2];
      }
      table.set([...xyzToLinearRGB([x, y, z]), 0], row * 4);
    }
  }
  if (!table.every(Number.isFinite)) {
    throw new RangeError("The jet spectrum exceeds GPU storage.");
  }
  return { data: table, scale };
}
