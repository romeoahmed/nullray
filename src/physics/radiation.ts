import { cie1931 } from "../data/cie1931.ts";
import { circularOrbit } from "./spacetime.ts";
import type { Spacetime } from "./spacetime.ts";
import type { MotionConstants } from "./motion.ts";
import type { Vec3 } from "./vector.ts";

/**
 * Compute diagonal linear-P3 gains that neutralize a reference blackbody.
 *
 * @remarks
 * Preserves that reference's Y luminance, not every input color's luminance.
 * This camera calibration does not change source temperature or physical redshift.
 *
 * @param temperature - Reference temperature in kelvin, from 2500 through 12000.
 * @throws RangeError - If the reference temperature is nonfinite or outside that range.
 */
export function blackbodyWhiteBalance(temperature: number): Vec3 {
  if (!Number.isFinite(temperature) || temperature < 2500 || temperature > 12000) {
    throw new RangeError("White balance must lie between 2500 K and 12000 K.");
  }
  const xyz = blackbodyXYZ(temperature);
  const [r, g, b] = xyzToLinearRGB(xyz);
  return [
    xyz[1] / (0.82246197 * r + 0.17753803 * g),
    xyz[1] / (0.0331942 * r + 0.9668058 * g),
    xyz[1] / (0.01708263 * r + 0.07239744 * g + 0.91051993 * b),
  ];
}

/**
 * Evaluate Planck spectral radiance per wavelength, in W sr⁻¹ m⁻³.
 *
 * @remarks
 * This is Bλ, not Bν. Extreme otherwise-admissible inputs can still overflow
 * intermediate arithmetic; callers choose the supported spectral range.
 *
 * @param wavelength - Positive finite wavelength in metres.
 * @param temperature - Nonnegative finite temperature in kelvin; zero emits no radiation.
 * @throws RangeError - If the input-domain checks fail.
 */
export function planck(wavelength: number, temperature: number): number {
  if (!(wavelength > 0) || !(temperature >= 0) || !Number.isFinite(wavelength + temperature)) {
    throw new RangeError(
      "Planck radiation requires positive finite wavelength and nonnegative finite temperature.",
    );
  }
  if (temperature === 0) {
    return 0;
  }
  const h = 6.62607015e-34;
  const c = 299792458;
  const k = 1.380649e-23;
  return (2 * h * c * c) / (wavelength ** 5 * Math.expm1((h * c) / (wavelength * k * temperature)));
}

/**
 * Integrate a blackbody against the 360–830 nm CIE samples by trapezoidal quadrature.
 *
 * @param temperature - Nonnegative temperature in kelvin, subject to {@link planck} validation.
 * @returns Unnormalized XYZ radiance, including the nanometre-to-metre integration factor.
 */
export function blackbodyXYZ(temperature: number): Vec3 {
  let x = 0;
  let y = 0;
  let z = 0;
  for (const [i, match] of cie1931.entries()) {
    const endpointWeight = i === 0 || i === cie1931.length - 1 ? 0.5 : 1;
    const power = planck((360 + i) * 1e-9, temperature) * 1e-9 * endpointWeight;
    x += power * match[0];
    y += power * match[1];
    z += power * match[2];
  }
  return [x, y, z];
}

/** Linear sRGB primaries/D65 matrix; the emitted spectrum is not white-balanced. */
export function xyzToLinearRGB([x, y, z]: Vec3): Vec3 {
  return [
    3.2409699419 * x - 1.5373831776 * y - 0.4986107603 * z,
    -0.9692436363 * x + 1.8759675015 * y + 0.0415550574 * z,
    0.0556300797 * x - 0.2039769589 * y + 1.0569715142 * z,
  ];
}

/**
 * Compute observed/emitted frequency for a positive-BL-time circular source.
 *
 * @param photon - Killing constants of a photon normalized to unit detector frequency.
 * @param radius - Positive emitting radius in M.
 * @returns The positive ratio, or `undefined` for an unsupported emitter or emitted energy.
 * Zero and negative Killing energy remain admissible; source-block orientation is not handled here.
 */
export function diskFrequencyRatio(
  space: Spacetime,
  photon: Pick<MotionConstants, "energy" | "angularMomentum">,
  radius: number,
  direction: 1 | -1,
): number | undefined {
  const emitter = circularOrbit(space, radius, direction);
  if (!emitter) {
    return undefined;
  }
  const energy = emitter.ut * (photon.energy - emitter.omega * photon.angularMomentum);
  return energy > 0 && Number.isFinite(energy) ? 1 / energy : undefined;
}

/** Caller-owned RGBA storage sampled uniformly in log temperature. */
export interface BlackbodyTable {
  readonly data: Float32Array;
  readonly minimum: number;
  readonly maximum: number;
  readonly size: number;
}

/**
 * Tabulate scene-linear sRGB radiance with the common normalization `Y(6500 K) = 1`.
 *
 * @remarks
 * The 1024 rows span 100–1,000,000 K uniformly in log temperature, matching
 * WGSL lookup bounds. Shift temperature once; do not renormalize rows or add
 * a second frequency-power multiplier after lookup.
 *
 * @returns Caller-owned f32 RGBA rows; alpha is one and signed RGB is preserved.
 */
export function createBlackbodyTable(): BlackbodyTable {
  const size = 1024;
  const minimum = 100;
  const maximum = 1000000;
  const normalization = blackbodyXYZ(6500)[1];
  const data = new Float32Array(size * 4);
  for (let i = 0; i < size; i++) {
    const temperature = minimum * (maximum / minimum) ** (i / (size - 1));
    const rgb = xyzToLinearRGB(blackbodyXYZ(temperature));
    data.set([rgb[0] / normalization, rgb[1] / normalization, rgb[2] / normalization, 1], i * 4);
  }
  return { data, minimum, maximum, size };
}
