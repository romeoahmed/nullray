import { circularOrbit } from "./spacetime.ts";
import type { Spacetime } from "./spacetime.ts";

/** Logarithmic radial samples of the zero-torque Page–Thorne flux, for unit accretion rate. */
export interface DiskProfile {
  readonly inner: number;
  readonly outer: number;
  /** Multiplier giving the tapered inverse-square vertical column unit peak. */
  readonly columnNormalization: number;
  readonly flux: Float64Array<ArrayBuffer>;
  /** Temperature divided by the peak temperature, ready for GPU interpolation. */
  readonly temperature: Float32Array<ArrayBuffer>;
}

/**
 * Normalize the tapered inverse-square vertical column to unit peak.
 * Radii are finite and positive, with outer > inner; this is a source prescription.
 */
export function diskColumnNormalization(inner: number, outer: number): number {
  if (!(inner > 0 && outer > inner) || !Number.isFinite(outer)) {
    throw new RangeError("A disk column requires ordered positive finite radii.");
  }
  const ratio = (outer - inner) / inner;
  // x(1-x)^2 / (1 + ratio*x)^2 has a single interior maximum.
  const x = 2 / (3 + ratio + Math.hypot(3 + ratio, 2 * Math.sqrt(ratio)));
  return (1 + ratio * x) ** 2 / (6.75 * x * (1 - x) ** 2);
}

/** Analytic charged-orbit derivatives entering the Page–Thorne radial conservation law. */
function orbitGradient(space: Spacetime, r: number) {
  const orbit = circularOrbit(space, r, 1);
  if (!orbit) {
    throw new RangeError("Disk flux requires timelike circular emitters.");
  }
  const { spin: a, charge: q } = space;
  const { omega, ut } = orbit;
  const root = Math.sqrt(r - q * q);
  const denominator = r * r + a * root;
  const omegaDerivative = ((r * r) / (2 * root) - 2 * r * root) / denominator ** 2;
  const gravity = 2 / r - (q * q) / (r * r);
  const gravityDerivative = -2 / (r * r) + (2 * q * q) / (r * r * r);
  const tphi = -a * gravity;
  const phiPhi = r * r + a * a + a * a * gravity;
  const momentum = tphi + omega * phiPhi;
  const momentumDerivative =
    -a * gravityDerivative + omegaDerivative * phiPhi + omega * (2 * r + a * a * gravityDerivative);
  // Circular radial force balance gives (u^t)' = (u^t)^3 Ω' (g_tφ + Ω g_φφ).
  const angularMomentumDerivative =
    momentumDerivative * ut + momentum * momentum * ut ** 3 * omegaDerivative;
  return {
    integrand: angularMomentumDerivative / ut,
    prefactor: (-omegaDerivative * ut * ut) / (4 * Math.PI * r),
  };
}

/**
 * Integrate the stationary, axisymmetric thin-disk conservation law with zero inner torque.
 * @param space - Spacetime with a supported neutral circular emitter of orientation +1.
 * @param inner - Inner edge on the stable exterior orbit branch, in gravitational radii.
 * @param outer - Finite outer edge strictly beyond the inner edge, in gravitational radii.
 * @param size - Number of logarithmically spaced samples, including both edges.
 * @returns Fresh caller-owned flux and peak-normalized temperature arrays.
 * @throws RangeError when sampling, orbital stability, or flux normalization is invalid.
 */
export function createDiskProfile(
  space: Spacetime,
  inner: number,
  outer: number,
  size = 512,
): DiskProfile {
  if (
    !(inner > 0 && outer > inner) ||
    !Number.isFinite(outer) ||
    !Number.isSafeInteger(size) ||
    size < 2
  ) {
    throw new RangeError("A disk profile requires ordered finite radii and at least two samples.");
  }
  const flux = new Float64Array(size);
  const temperature = new Float32Array(size);
  const logRange = Math.log(outer / inner);
  let integral = 0;
  let previousRadius = inner;
  let previous = orbitGradient(space, inner);
  let maximum = 0;
  for (let index = 1; index < size; index++) {
    const r = inner * Math.exp((logRange * index) / (size - 1));
    const current = orbitGradient(space, r);
    const middle = orbitGradient(space, (r + previousRadius) / 2);
    integral +=
      ((r - previousRadius) * (previous.integrand + 4 * middle.integrand + current.integrand)) / 6;
    const value = current.prefactor * integral;
    if (!(value >= 0) || !Number.isFinite(value)) {
      throw new RangeError("Disk flux left the stable circular-emitter domain.");
    }
    flux[index] = value;
    maximum = Math.max(maximum, value);
    previous = current;
    previousRadius = r;
  }
  if (!(maximum > 0)) {
    throw new RangeError("Disk flux has no positive normalization.");
  }
  for (let index = 0; index < size; index++) {
    temperature[index] = ((flux[index] ?? 0) / maximum) ** 0.25;
  }
  return {
    inner,
    outer,
    flux,
    temperature,
    columnNormalization: diskColumnNormalization(inner, outer),
  };
}
