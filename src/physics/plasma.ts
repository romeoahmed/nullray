import type { KerrGeometry } from "./geometry.ts";

/**
 * Cold, nonmagnetized test-plasma inputs for a single observing frequency.
 */
export interface Plasma {
  /** Density scale n₀ in cm⁻³; the equatorial density at r₀ is n₀/2. */
  readonly density: number;
  /** Radial scale r₀ in M. */
  readonly radius: number;
  /** Detector frequency in GHz, not angular frequency. */
  readonly frequencyGHz: number;
  /** Peak fractional temperature increase above 100,000 K; zero selects the stationary radial profile. */
  readonly heating: number;
}

/** Separable radial term f_r = amplitude r²/(r² + scaleSquared), normalized by detector frequency². */
export interface PlasmaProfile {
  readonly amplitude: number;
  readonly scaleSquared: number;
}

/** Electron plasma frequency squared per cm⁻³, in Hz²; SI constants from CODATA 2022. */
export const plasmaFrequencySquared =
  (1e6 * 1.602176634e-19 ** 2) / (4 * Math.PI ** 2 * 8.8541878188e-12 * 9.1093837139e-31);

/**
 * Quantize the separable radial coefficients for shared host/GPU propagation.
 *
 * @remarks
 * Requires validated plasma inputs. The amplitude is detector-frequency
 * normalized; quantization alone does not check propagation at the observer.
 */
export function plasmaProfile(plasma: Plasma): PlasmaProfile {
  const scaleSquared = Math.fround(plasma.radius ** 2);
  return {
    amplitude: Math.fround(
      (plasmaFrequencySquared * plasma.density * scaleSquared) / (plasma.frequencyGHz * 1e9) ** 2,
    ),
    scaleSquared,
  };
}

/** Local plasma frequency² / detector frequency² at a regular event. */
export function plasmaCutoff(profile: PlasmaProfile, geometry: KerrGeometry): number {
  const r2 = geometry.point.radius ** 2;
  return (profile.amplitude * r2) / ((r2 + profile.scaleSquared) * geometry.sigma);
}

/** Compact heated annulus and rotating six-fold pattern; frequency is the phase rate in rad/M. */
export interface HeatingProfile {
  readonly inner: number;
  readonly outer: number;
  readonly contrast: number;
  readonly frequency: number;
}

/**
 * Prepare the compact heated annulus, or return `null` when heating is zero.
 *
 * @remarks
 * Requires validated inputs. The sixfold phase rate is `1.5/r₀^(3/2)`;
 * its angular pattern speed is one sixth of that rate. Scene construction
 * checks the annulus location and its conservative timelike bound.
 */
export function heatingProfile(plasma: Plasma): HeatingProfile | null {
  if (plasma.heating === 0) {
    return null;
  }
  return {
    inner: Math.fround(plasma.radius),
    outer: Math.fround(3 * plasma.radius),
    contrast: Math.fround(plasma.heating),
    frequency: Math.fround(1.5 / plasma.radius ** 1.5),
  };
}
