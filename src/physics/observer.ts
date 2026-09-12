import { combine, contract, inner, lower } from "./geometry.ts";
import type { FourVector, KerrGeometry } from "./geometry.ts";
import type { Vec3 } from "./vector.ts";

/**
 * Local orthonormal tetrad expressed as Cartesian Kerr–Schild tangents.
 *
 * @remarks
 * `velocity` is unit timelike. The three spatial axes label the local boost
 * and camera components; after a boost they need not follow coordinate lines.
 */
export interface ObserverFrame {
  readonly velocity: FourVector;
  readonly radial: FourVector;
  readonly polar: FourVector;
  readonly azimuthal: FourVector;
}

/** A horizon-regular reference frame; its accelerated motion is not free fall. */
export function principalFrame(geometry: KerrGeometry): ObserverFrame {
  const { principal: k, factor: f } = geometry;
  const sign = geometry.point.chart === "ingoing" ? 1 : -1;
  return {
    velocity: [1 + f / 2, (-f * k[1]) / 2, (-f * k[2]) / 2, (-f * k[3]) / 2],
    radial: [
      (sign * f) / 2,
      sign * (1 - f / 2) * k[1],
      sign * (1 - f / 2) * k[2],
      sign * (1 - f / 2) * k[3],
    ],
    polar: geometry.polar,
    azimuthal: geometry.azimuthal,
  };
}

/**
 * Lorentz-boost an orthonormal tetrad by measured local velocity.
 *
 * @param velocity - Components along the supplied spatial axes, in units of c.
 * @returns A new frame, or `undefined` for nonfinite speed or speed at least c.
 */
export function boostFrame(frame: ObserverFrame, velocity: Vec3): ObserverFrame | undefined {
  const speed2 = velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2;
  if (!Number.isFinite(speed2) || speed2 >= 1) {
    return undefined;
  }
  const gamma = 1 / Math.sqrt(1 - speed2);
  const spatial = combine(
    combine(frame.radial, velocity[0], frame.polar, velocity[1]),
    1,
    frame.azimuthal,
    velocity[2],
  );
  // γ²/(γ+1) is (γ−1)/β² without cancellation or a zero-speed division.
  const boost = combine(frame.velocity, gamma, spatial, (gamma * gamma) / (gamma + 1));
  return {
    velocity: combine(frame.velocity, gamma, spatial, gamma),
    radial: combine(frame.radial, 1, boost, velocity[0]),
    polar: combine(frame.polar, 1, boost, velocity[1]),
    azimuthal: combine(frame.azimuthal, 1, boost, velocity[2]),
  };
}

/**
 * Construct a rest frame from KS coordinate velocity `d(X, Y, Z)/dT`.
 *
 * @returns A frame, or `undefined` when the trial tangent is nonfinite,
 * non-timelike, or not future-directed relative to the regular reference frame.
 * A Euclidean coordinate-speed bound is not the local light-cone condition.
 */
export function coordinateObserver(
  geometry: KerrGeometry,
  velocity: Vec3,
): ObserverFrame | undefined {
  if (!velocity.every(Number.isFinite)) {
    return undefined;
  }
  const tangent: FourVector = [1, ...velocity];
  const norm = inner(geometry, tangent, tangent);
  const reference = principalFrame(geometry);
  const frequency = -inner(geometry, reference.velocity, tangent);
  if (!(norm < 0) || !(frequency > 0) || !Number.isFinite(norm + frequency)) {
    return undefined;
  }
  return boostFrame(reference, [
    inner(geometry, reference.radial, tangent) / frequency,
    inner(geometry, reference.polar, tangent) / frequency,
    inner(geometry, reference.azimuthal, tangent) / frequency,
  ]);
}

/** Static Killing observer; absent inside an ergoregion or on its stationary limit. */
export function staticObserver(geometry: KerrGeometry): ObserverFrame | undefined {
  return coordinateObserver(geometry, [0, 0, 0]);
}

/** Zero-angular-momentum circular observer wherever its stationary worldline is timelike. */
export function zamoObserver(geometry: KerrGeometry, spin: number): ObserverFrame | undefined {
  const { radius } = geometry.point;
  const sine2 = geometry.principal[1] ** 2 + geometry.principal[2] ** 2;
  const denominator = radius * radius + spin * spin * (1 + geometry.factor * sine2);
  if (!(denominator > 0)) {
    return undefined;
  }
  const omega = (spin * geometry.factor) / denominator;
  return coordinateObserver(geometry, [
    -omega * geometry.position[1],
    omega * geometry.position[0],
    0,
  ]);
}

/**
 * Reconstruct a rest frame from a timelike tangent at its current event.
 *
 * @remarks
 * The reference time orientation follows the supplied tangent. Spatial axes
 * come from a local boost, not parallel transport of the launch tetrad.
 *
 * @returns A frame, or `undefined` when a finite timelike boost cannot be formed.
 */
export function comovingFrame(
  geometry: KerrGeometry,
  velocity: FourVector,
): ObserverFrame | undefined {
  let reference = principalFrame(geometry);
  let frequency = -inner(geometry, reference.velocity, velocity);
  if (frequency < 0) {
    reference = {
      ...reference,
      velocity: combine(reference.velocity, -1, reference.velocity, 0),
      radial: combine(reference.radial, -1, reference.radial, 0),
    };
    frequency = -frequency;
  }
  if (!(frequency > 0) || !(inner(geometry, velocity, velocity) < 0)) {
    return undefined;
  }
  return boostFrame(reference, [
    inner(geometry, reference.radial, velocity) / frequency,
    inner(geometry, reference.polar, velocity) / frequency,
    inner(geometry, reference.azimuthal, velocity) / frequency,
  ]);
}

/**
 * Launch the future-directed arriving vacuum photon with unit detector frequency.
 *
 * @param source - Unit direction toward the apparent source in the frame's spatial axes.
 * @returns `u - sourceᵢ eᵢ`; trace it with negative affine/Mino steps.
 * The caller supplies an orthonormal frame and normalized direction.
 */
export function observerPhoton(frame: ObserverFrame, source: Vec3): FourVector {
  return combine(
    combine(combine(frame.velocity, 1, frame.radial, -source[0]), 1, frame.polar, -source[1]),
    1,
    frame.azimuthal,
    -source[2],
  );
}

/**
 * Evaluate the signed local energy contraction `-p · u`.
 *
 * @remarks
 * It is positive for a physical future photon and future unit observer.
 * This helper does not validate, take an absolute value, or renormalize its inputs.
 */
export function measuredFrequency(
  geometry: KerrGeometry,
  photon: FourVector,
  observer: FourVector,
): number {
  return -contract(lower(geometry, photon), observer);
}
