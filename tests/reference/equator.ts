import { carlsonRF } from "./elliptic.ts";
import type { Photon } from "../../src/physics/photon.ts";
import type { Spacetime } from "../../src/physics/spacetime.ts";

export type EquatorCrossings =
  | {
      readonly kind: "crossings";
      readonly first: number;
      readonly spacing: number;
      readonly phase: number;
      readonly frequency: number;
      readonly m: number;
      readonly amplitude2: number;
      /** Initial unwrapped polar azimuth primitive, without a lossy phase round trip. */
      readonly azimuthPhase: number;
    }
  | { readonly kind: "none" }
  | { readonly kind: "unresolved" };

/**
 * Exact periodic transverse equator events in positive backward Mino time.
 * For C > 0, mu = amplitude cn(phase + frequency tau, m). Invert the initial
 * phase with Carlson RF; adjacent cn zeros are separated by 2 K(m).
 * C <= 0 cannot cross transversely. Coplanar overlap is a visibility decision.
 */
export function equatorCrossings(
  space: Spacetime,
  photon: Photon,
  theta: number,
): EquatorCrossings {
  const c = photon.carter;
  if (c <= 0) {
    return { kind: "none" };
  }
  const a2e2 = (space.spin * photon.energy) ** 2;
  const b = photon.angularMomentum ** 2 + c - a2e2;
  const frequency2 =
    photon.angularMomentum === 0 ? c + a2e2 : Math.hypot(b, 2 * Math.sqrt(a2e2 * c));
  const amplitude2 =
    photon.angularMomentum === 0
      ? 1
      : b >= 0
        ? (2 * c) / (b + frequency2)
        : (frequency2 - b) / (2 * a2e2);
  const m = (a2e2 * amplitude2) / frequency2;
  const cosine = Math.cos(theta) / Math.sqrt(amplitude2);
  // The amplitude is an analytic turning point. Allow only binary64 roundoff
  // in the initial ratio, and report a failure if m loses its distance from 1.
  if (!(m >= 0 && m < 1) || Math.abs(cosine) > 1 + 16 * Number.EPSILON) {
    return { kind: "unresolved" };
  }
  const k = carlsonRF(0, 1 - m, 1);
  const cn = Math.max(-1, Math.min(1, cosine));
  const sinTheta = theta === 0 || theta === Math.PI ? 0 : Math.sin(theta);
  // Recover sn from the initial derivative: acos(cn) loses the launch phase near a pole.
  const sn =
    Math.abs(sinTheta * photon.polarVelocity) /
    Math.sqrt(amplitude2 * frequency2 * (1 - m + m * cn * cn));
  const angle = Math.atan2(sn, cn);
  const acute = Math.min(angle, Math.PI - angle);
  const sin = Math.sin(acute);
  const principal = sin * carlsonRF(Math.cos(acute) ** 2, 1 - m * sin * sin, 1);
  let phase = angle <= Math.PI / 2 ? principal : 2 * k - principal;
  if (photon.polarVelocity > 0) {
    phase = -phase;
  }
  let azimuthPhase: number;
  if (photon.angularMomentum === 0) {
    azimuthPhase = Math.PI * (Math.floor(phase / (2 * k)) + 0.5);
    if (theta === 0) {
      azimuthPhase = Math.PI / 2;
    } else if (theta === Math.PI) {
      azimuthPhase = phase < 0 ? -Math.PI / 2 : (3 * Math.PI) / 2;
    }
  } else {
    const complement = (photon.angularMomentum ** 2 * amplitude2) / (c + a2e2 * amplitude2);
    azimuthPhase = Math.atan2(photon.polarVelocity > 0 ? -sn : sn, Math.sqrt(complement) * cn);
  }
  const spacing = (2 * k) / Math.sqrt(frequency2);
  let first = (k - phase) / Math.sqrt(frequency2);
  first -= Math.floor(first / spacing) * spacing;
  // Exclude the observer event even when it lies exactly on the equator.
  if (theta === Math.PI / 2 || first === 0) {
    first = spacing;
  }
  return {
    kind: "crossings",
    first,
    spacing,
    phase,
    frequency: Math.sqrt(frequency2),
    m,
    amplitude2,
    azimuthPhase,
  };
}
