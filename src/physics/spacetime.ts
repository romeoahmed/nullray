/** Dimensionless Kerr–Newman parameters, in units G = c = M = 1. */
export interface Spacetime {
  readonly spin: number;
  readonly charge: number;
}

/** Outer horizon used by the subextremal circular-orbit branch. */
export function outerHorizon({ spin: a, charge: q }: Spacetime): number {
  return 1 + Math.sqrt(1 - a * a - q * q);
}

/** Specific neutral-particle constants for a circular equatorial emitter. */
export interface CircularOrbit {
  readonly omega: number;
  readonly ut: number;
  readonly energy: number;
  readonly angularMomentum: number;
  /** Second radial-potential derivative at fixed constants; negative is stable. */
  readonly radialStability: number;
}

/** Neutral equatorial circular emitter, including horizonless geometry; absent when not timelike. */
export function circularOrbit(
  space: Spacetime,
  r: number,
  direction: 1 | -1,
): CircularOrbit | undefined {
  const { spin: a, charge: q } = space;
  const delta = r * r - 2 * r + a * a + q * q;
  if (!(r > q * q) || !(delta > 0) || !Number.isFinite(r + a + q)) {
    return undefined;
  }
  const root = Math.sqrt(r - q * q);
  const omega = (direction * root) / (r * r + direction * a * root);
  const factor = (2 * r - q * q) / (r * r);
  const tt = factor - 1;
  const tphi = -a * factor;
  const phiPhi = r * r + a * a * (1 + factor);
  const norm = -(tt + 2 * omega * tphi + omega * omega * phiPhi);
  if (!(norm > 0)) {
    return undefined;
  }
  const ut = 1 / Math.sqrt(norm);
  const energy = -(tt + omega * tphi) * ut;
  const angularMomentum = (tphi + omega * phiPhi) * ut;
  const p = energy * (r * r + a * a) - a * angularMomentum;
  const b = r * r + (angularMomentum - a * energy) ** 2;
  const radialStability =
    4 * energy * p + 8 * energy * energy * r * r - 2 * b - 4 * r * (2 * r - 2) - 2 * delta;
  return { omega, ut, energy, angularMomentum, radialStability };
}

/** Follow the stable exterior branch inward, then refine its marginal orbit. */
export function isco(space: Spacetime, direction: 1 | -1): number {
  const horizon = outerHorizon(space);
  let stable = 16;
  let unstable = horizon;
  // Geometric spacing resolves the narrow near-horizon prograde branch.
  for (let i = 1; i <= 256; i++) {
    const r = horizon + (16 - horizon) * Math.exp(-i / 16);
    const orbit = circularOrbit(space, r, direction);
    if (!orbit || orbit.radialStability >= 0) {
      unstable = r;
      break;
    }
    stable = r;
  }
  for (let i = 0; i < 64; i++) {
    const r = (stable + unstable) / 2;
    const orbit = circularOrbit(space, r, direction);
    if (orbit && orbit.radialStability < 0) {
      stable = r;
    } else {
      unstable = r;
    }
  }
  return (stable + unstable) / 2;
}
