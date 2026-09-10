import { metric, outerHorizon } from "./spacetime.ts";
import type { Spacetime } from "./spacetime.ts";
import type { Vec3 } from "./vector.ts";

/** A point on the observer's vacuum capture/escape separatrix, before disk occlusion. */
export type CriticalDirection =
  | {
      readonly kind: "point";
      /** Unit source direction in the same local (r, theta, phi) frame as photonFromLocal. */
      readonly direction: Vec3;
      readonly sphericalRadius: number;
      /** Width of the final binary64 (r−1) bracket, not a certified error bound. */
      readonly radiusBracketWidth: number;
    }
  | { readonly kind: "invalid" | "unresolved" };

/**
 * Locate one critical direction for a stationary exterior ZAMO.
 *
 * The phase parameter resolves the polar potential as a circle in Carter's
 * separated momentum variables. It is periodic but is not screen position angle.
 * Solving R = R' = 0 in this parameter avoids division by spin or sin(theta),
 * including the nonrotating and axis limits. See docs/numerics.md.
 * This is a seed for image searches, not a finite-time endpoint or a disk hit.
 */
export function criticalDirection(
  space: Spacetime,
  observerRadius: number,
  inclination: number,
  phase: number,
): CriticalDirection {
  const { spin: a, charge: q } = space;
  if (
    ![a, q, observerRadius, inclination, phase].every(Number.isFinite) ||
    a * a + q * q >= 1 ||
    observerRadius <= outerHorizon(space) ||
    inclination < 0 ||
    inclination > Math.PI
  ) {
    return { kind: "invalid" };
  }
  const gapSquared = 1 - a * a - q * q;
  const horizonGap = Math.sqrt(gapSquared);
  const sine = inclination === 0 || inclination === Math.PI ? 0 : Math.sin(inclination);
  const cosine = Math.cos(inclination);
  const phaseCosine = Math.cos(phase);
  const equation = (centered: number) => {
    const r = 1 + centered;
    const delta = centered * centered - gapSquared;
    return (
      (r * r + a * a * cosine * cosine) * centered -
      2 * r * delta -
      2 * a * r * Math.sqrt(delta) * sine * phaseCosine
    );
  };
  // Solve in r−1 to preserve near-horizon differences; the outer bound r<4 becomes 3.
  let lower = horizonGap;
  if (lower * lower < gapSquared) {
    // Use the next exterior binary64 value if horizon rounding falls inside.
    const next = new Float64Array([lower]);
    const words = new BigUint64Array(next.buffer);
    words[0] = (words[0] ?? 0n) + 1n;
    lower = next[0] ?? NaN;
  }
  let upper = 3;
  if (!(equation(lower) > 0 && equation(upper) < 0)) {
    return { kind: "unresolved" };
  }
  let converged = false;
  for (let iteration = 0; iteration < 128; iteration++) {
    const middle = lower + (upper - lower) / 2;
    if (middle === lower || middle === upper) {
      converged = true;
      break;
    }
    const value = equation(middle);
    if (!Number.isFinite(value)) {
      return { kind: "unresolved" };
    }
    if (value > 0) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  if (!converged) {
    return { kind: "unresolved" };
  }
  const centered = lower + (upper - lower) / 2;
  const radius = 1 + centered;
  const delta = centered * centered - gapSquared;
  const rootK = (2 * radius * Math.sqrt(delta)) / centered;
  const azimuthMomentum = a * sine + rootK * phaseCosine;
  const lambda = sine * azimuthMomentum;
  const g = metric(space, observerRadius, inclination);
  const localEnergy = (1 - g.dragging * lambda) / g.lapse;
  // Factor out the known double root instead of subtracting the radial potential
  // at the observer. This also selects outward approach when the observer is inside.
  const radialFactor =
    (observerRadius + radius) ** 2 - (4 * radius * delta) / (centered * centered);
  if (!(localEnergy > 0 && g.delta > 0 && radialFactor > 0)) {
    return { kind: "unresolved" };
  }
  const direction: Vec3 = [
    ((centered - (observerRadius - 1)) * Math.sqrt(radialFactor)) /
      (Math.sqrt(g.delta * g.sigma) * localEnergy),
    (-rootK * Math.sin(phase)) / (Math.sqrt(g.sigma) * localEnergy),
    -azimuthMomentum / (g.azimuthScale * localEnergy),
  ];
  const norm = Math.hypot(...direction);
  // A failed null normalization is evidence of conditioning, not permission to
  // project arbitrary arithmetic onto the unit sphere and report success.
  if (!Number.isFinite(norm) || Math.abs(norm - 1) > 1e-12) {
    return { kind: "unresolved" };
  }
  return {
    kind: "point",
    direction: [direction[0] / norm, direction[1] / norm, direction[2] / norm],
    sphericalRadius: radius,
    radiusBracketWidth: upper - lower,
  };
}
