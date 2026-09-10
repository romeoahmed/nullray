import { combine, contract, inner, lower, raise } from "../../src/physics/geometry.ts";
import type { FourVector, KerrGeometry } from "../../src/physics/geometry.ts";
import type { Spacetime } from "../../src/physics/spacetime.ts";

/** Conserved real and imaginary contractions with the principal tensor and its Hodge dual. */
export interface WalkerPenrose {
  readonly real: number;
  readonly imaginary: number;
}

/** An orthonormal polarization screen in the observer gauge; undefined for a principal null ray. */
export interface PolarizationScreen {
  readonly first: FourVector;
  readonly second: FourVector;
}

/**
 * Apply h and *h to a null tangent. In Cartesian Kerr–Schild coordinates,
 * h = dT ∧ (X dX + Y dY + Z dZ) + a dX ∧ dY, including nonzero charge.
 * The spacetime orientation is ε(T,X,Y,Z)=+1 and the metric signature is −+++.
 */
export function principalContractions(
  space: Spacetime,
  geometry: KerrGeometry,
  photon: FourVector,
): readonly [FourVector, FourVector] {
  const [x, y, z] = geometry.position;
  const a = space.spin;
  const [t, px, py, pz] = photon;
  const electric: FourVector = [x * px + y * py + z * pz, -x * t + a * py, -y * t - a * px, -z * t];
  // h maps the KS null direction to a multiple of itself: the rank-one metric
  // correction cancels when raising both indices. Its dual has the flat-space form.
  const magnetic: FourVector = [
    a * pz,
    -z * py + y * pz,
    z * px - x * pz,
    -a * t - y * px + x * py,
  ];
  return [electric, magnetic];
}

/** Walker–Penrose data for a parallel-transported unit polarization, invariant under f → f + αp. */
export function walkerPenrose(
  space: Spacetime,
  geometry: KerrGeometry,
  photon: FourVector,
  polarization: FourVector,
): WalkerPenrose {
  const [h, dual] = principalContractions(space, geometry, photon);
  return { real: contract(h, polarization), imaginary: contract(dual, polarization) };
}

/** Evaluate the conserved polarization axes on a finite observer's measured screen. */
export function polarizationScreen(
  space: Spacetime,
  geometry: KerrGeometry,
  photon: FourVector,
  observer: FourVector,
): PolarizationScreen | undefined {
  const frequency = -contract(lower(geometry, photon), observer);
  if (!(frequency > 0)) {
    return undefined;
  }
  const screen = (covector: FourVector): FourVector => {
    const vector = raise(geometry, covector);
    return combine(vector, 1, photon, inner(geometry, observer, vector) / frequency);
  };
  const [h, dual] = principalContractions(space, geometry, photon);
  const first = screen(h);
  const second = screen(dual);
  const squared = inner(geometry, first, first);
  if (!(squared > 0) || !Number.isFinite(squared)) {
    return undefined;
  }
  const scale = 1 / Math.sqrt(squared);
  return { first: combine(first, scale, first, 0), second: combine(second, scale, second, 0) };
}
