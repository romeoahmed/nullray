/** Dimensionless Kerr–Newman parameters, in units G = c = M = 1. */
export interface Spacetime {
  readonly spin: number;
  readonly charge: number;
}

/** Covariant Boyer-Lindquist metric coefficients outside the horizon. */
export interface Metric {
  readonly tt: number;
  readonly tphi: number;
  readonly rr: number;
  readonly thetaTheta: number;
  readonly phiPhi: number;
  readonly sigma: number;
  readonly delta: number;
  readonly lapse: number;
  readonly dragging: number;
  /** Proper azimuthal scale divided by sin(theta), regular at either axis. */
  readonly azimuthScale: number;
}

/** Outer Boyer–Lindquist horizon radius; the caller supplies subextremal parameters. */
export function outerHorizon({ spin: a, charge: q }: Spacetime): number {
  return 1 + Math.sqrt(1 - a * a - q * q);
}

/** Evaluate the exterior metric. The caller owns coordinate/domain validation. */
export function metric({ spin: a, charge: q }: Spacetime, r: number, theta: number): Metric {
  const sin2 = Math.sin(theta) ** 2;
  const sigma = r * r + a * a * Math.cos(theta) ** 2;
  const delta = r * r - 2 * r + a * a + q * q;
  const bigA = (r * r + a * a) ** 2 - a * a * delta * sin2;
  return {
    tt: -(delta - a * a * sin2) / sigma,
    tphi: (-a * (2 * r - q * q) * sin2) / sigma,
    rr: sigma / delta,
    thetaTheta: sigma,
    phiPhi: (bigA * sin2) / sigma,
    sigma,
    delta,
    lapse: Math.sqrt((sigma * delta) / bigA),
    dragging: (a * (2 * r - q * q)) / bigA,
    azimuthScale: Math.sqrt(bigA / sigma),
  };
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

/** Neutral equatorial emitter outside the horizon; undefined when no timelike orbit exists. */
export function circularOrbit(
  space: Spacetime,
  r: number,
  direction: 1 | -1,
): CircularOrbit | undefined {
  const { spin: a, charge: q } = space;
  if (r <= outerHorizon(space)) {
    return undefined;
  }
  const root = Math.sqrt(r - q * q);
  const omega = (direction * root) / (r * r + direction * a * root);
  const g = metric(space, r, Math.PI / 2);
  const norm = -(g.tt + 2 * omega * g.tphi + omega * omega * g.phiPhi);
  if (!(norm > 0)) {
    return undefined;
  }
  const ut = 1 / Math.sqrt(norm);
  const energy = -(g.tt + omega * g.tphi) * ut;
  const angularMomentum = (g.tphi + omega * g.phiPhi) * ut;
  const p = energy * (r * r + a * a) - a * angularMomentum;
  const b = r * r + (angularMomentum - a * energy) ** 2;
  const radialStability =
    4 * energy * p + 8 * energy * energy * r * r - 2 * b - 4 * r * (2 * r - 2) - 2 * g.delta;
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
