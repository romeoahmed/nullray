import type { KerrGeometry } from "./geometry.ts";

/** Nonmagnetized test plasma; density in cm⁻³, scale radius in M, detector frequency in GHz. */
export interface Plasma {
  readonly density: number;
  readonly radius: number;
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

/** Prepare the two f32 radial coefficients used by both host and GPU propagation. */
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

/** The prescribed heating pattern rotates at one quarter of the Kepler frequency scale. */
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
