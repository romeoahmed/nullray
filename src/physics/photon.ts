import { metric } from "./spacetime.ts";
import type { Spacetime } from "./spacetime.ts";
import type { Vec3 } from "./vector.ts";

/** Conserved future-directed photon data; never divided by Killing energy. */
export interface Photon {
  /** Killing energy E = -p_t; zero and negative values remain meaningful. */
  readonly energy: number;
  /** Axial angular momentum L_z = p_phi. */
  readonly angularMomentum: number;
  /** Unnormalized Carter constant C, which may be negative. */
  readonly carter: number;
  /** Future-directed dr/dgamma in the unscaled Mino parameter. */
  readonly radialVelocity: number;
  /** Future-directed dtheta/dgamma = p_theta. */
  readonly polarVelocity: number;
}

/**
 * Initialize a unit-frequency ZAMO photon; screen direction points toward its source.
 * At a pole, polar momentum follows the departing meridian. Its initial azimuth
 * is supplied by the caller's tangent-plane direction, not this conserved data.
 *
 * @param space - Subextremal spacetime in units G = c = M = 1.
 * @param r - Exterior Boyer–Lindquist radius in gravitational radii.
 * @param theta - Colatitude in radians, including either axis.
 * @param source - Unit vector in the observer's orthonormal (r, theta, phi) frame.
 * @returns Future-directed constants and Mino velocities for backward evaluation.
 */
export function photonFromLocal(space: Spacetime, r: number, theta: number, source: Vec3): Photon {
  const g = metric(space, r, theta);
  const axis = theta === 0 || theta === Math.PI;
  const angularMomentum = axis ? 0 : -source[2] * g.azimuthScale * Math.sin(theta);
  const energy = g.lapse + g.dragging * angularMomentum;
  const pTheta = axis
    ? (theta === 0 ? -1 : 1) * Math.hypot(source[1], source[2]) * Math.sqrt(g.sigma)
    : -source[1] * Math.sqrt(g.sigma);
  const cos2 = Math.cos(theta) ** 2;
  return {
    energy,
    angularMomentum,
    carter:
      source[1] ** 2 * g.sigma -
      space.spin ** 2 * energy * energy * cos2 +
      source[2] ** 2 * g.azimuthScale ** 2 * cos2,
    radialVelocity: -source[0] * Math.sqrt(g.delta * g.sigma),
    polarVelocity: pTheta,
  };
}

/** Squared future-directed radial Mino velocity R(r); a negative value forbids that radius. */
export function radialPotential(space: Spacetime, photon: Photon, r: number): number {
  const { spin: a, charge: q } = space;
  const { energy: e, angularMomentum: l, carter: c } = photon;
  return (
    (e * (r * r + a * a) - a * l) ** 2 - (r * r - 2 * r + a * a + q * q) * ((l - a * e) ** 2 + c)
  );
}

/**
 * Squared polar Mino velocity Θ(θ), in Boyer–Lindquist colatitude radians.
 * The zero-angular-momentum axis limit is finite; nonzero L excludes the axis.
 */
export function polarPotential(space: Spacetime, photon: Photon, theta: number): number {
  const cosine = Math.cos(theta);
  const regular = photon.carter + (space.spin * photon.energy * cosine) ** 2;
  if (photon.angularMomentum === 0) {
    return regular;
  }
  if (theta === 0 || theta === Math.PI) {
    return -Infinity;
  }
  return regular - ((photon.angularMomentum * cosine) / Math.sin(theta)) ** 2;
}
