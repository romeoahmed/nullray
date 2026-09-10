import { cie1931 } from "../data/cie1931.ts";
import { circularOrbit } from "./spacetime.ts";
import type { Spacetime } from "./spacetime.ts";
import type { MotionConstants } from "./motion.ts";
import type { Vec3 } from "./vector.ts";

/** Diagonal linear-P3 camera calibration: a blackbody at this temperature becomes neutral, with Y unchanged. */
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
 * Planck spectral radiance per unit wavelength, in W sr⁻¹ m⁻³.
 * @param wavelength - Positive finite wavelength in metres.
 * @param temperature - Nonnegative finite temperature in kelvin; zero emits no radiation.
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

/** Integrate CIE XYZ over 360–830 nm using the source's 1 nm sampling. */
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

/** Positive local emission energy is required even for E <= 0 photons. */
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
 * Log-temperature lookup in scene-linear RGB, normalized to Y(6500 K) = 1.
 * The texture retains intensity variation with temperature; no per-row RGB
 * normalization or extra frequency-power multiplier belongs in the shader.
 * The fixed [100, 1e6] K range is shared with the WGSL lookup contract.
 * @returns Fresh binary32 RGB/alpha rows; negative working RGB components are preserved.
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
