import { evaluateGeodesic, prepareGeodesic } from "./geodesic.ts";
import type { SeparatedPath } from "./geodesic.ts";
import { evaluateQuartic } from "./quartic.ts";
import type { QuarticPath } from "./quartic.ts";
import { outerHorizon } from "../../src/physics/spacetime.ts";
import type { Spacetime } from "../../src/physics/spacetime.ts";
import type { Photon } from "../../src/physics/photon.ts";
import { equatorCrossings } from "./equator.ts";

export interface DiskAnnulus {
  readonly inner: number;
  readonly outer: number;
}

export type Visibility =
  | {
      readonly kind: "disk";
      readonly time: number;
      readonly radius: number;
      readonly order: number;
    }
  | { readonly kind: "sky"; readonly time: number }
  | { readonly kind: "captured"; readonly time: number }
  | {
      readonly kind: "unresolved";
      readonly reason: "arithmetic-domain" | "budget-exhausted" | "surface-degenerate";
    };

/** Refine a sign-changing analytic coordinate bracket without integrating an ODE. */
function crossing(
  path: QuarticPath,
  target: number,
  begin: number,
  end: number,
): number | undefined {
  const initial = evaluateQuartic(path, begin);
  if (initial === undefined) {
    return undefined;
  }
  let lower = begin;
  let upper = end;
  for (let iteration = 0; iteration < 40; iteration++) {
    const middle = (lower + upper) / 2;
    const value = evaluateQuartic(path, middle);
    if (value === undefined) {
      return undefined;
    }
    if ((value - target) * (initial - target) > 0) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return (lower + upper) / 2;
}

/**
 * Enumerate candidate surface/boundary events along a prepared analytic path.
 * Equator events use their exact polar phase spacing. Radial boundary brackets
 * still use bounded subdivisions pending critical-boundary coverage validation.
 */
export function traceVisibility(
  space: Spacetime,
  photon: Photon,
  radius: number,
  theta: number,
  disk?: DiskAnnulus,
  subdivisions = 64,
): { readonly path: SeparatedPath; readonly outcome: Visibility } {
  const path = prepareGeodesic(space, photon, radius, theta);
  const horizonInverse = 1 / outerHorizon(space);
  const frequency = Math.sqrt(
    Math.abs(photon.carter) + photon.angularMomentum ** 2 + (space.spin * photon.energy) ** 2 + 1,
  );
  const step = 1 / (subdivisions * frequency);
  const failure = (reason: "arithmetic-domain" | "budget-exhausted" | "surface-degenerate") => ({
    path,
    outcome: { kind: "unresolved", reason } as const,
  });
  // Equatorial symmetry is exact physically even when cos(pi/2) rounds away from zero.
  const coplanar = theta === Math.PI / 2 && photon.polarVelocity === 0;
  const equator = equatorCrossings(space, photon, theta);
  if (disk && equator.kind === "unresolved") {
    return failure("arithmetic-domain");
  }
  let nextCrossing = equator.kind === "crossings" ? equator.first : Infinity;
  let diskOrder = 0;
  for (let index = 1; index <= subdivisions * 64; index++) {
    const end = step * index;
    const begin = end - step;
    const next = evaluateGeodesic(path, end);
    if (!next || Math.abs(next.cosineTheta) > 1 + 1e-9) {
      return failure("arithmetic-domain");
    }
    let outcome: Exclude<Visibility, { readonly kind: "unresolved" }> | undefined;
    if (next.inverseRadius <= 0) {
      const time = crossing(path.radial, 0, begin, end);
      if (time === undefined || photon.energy <= 0) {
        return failure("arithmetic-domain");
      }
      outcome = { kind: "sky", time };
    } else if (next.inverseRadius >= horizonInverse) {
      const time = crossing(path.radial, horizonInverse, begin, end);
      if (time === undefined) {
        return failure("arithmetic-domain");
      }
      outcome = { kind: "captured", time };
    }
    if (disk) {
      if (
        coplanar &&
        next.inverseRadius >= 1 / disk.outer &&
        next.inverseRadius <= 1 / disk.inner
      ) {
        return failure("surface-degenerate");
      }
      for (
        let event = 0;
        event < 64 && nextCrossing <= end && equator.kind === "crossings";
        event++
      ) {
        const time = nextCrossing;
        const order = diskOrder++;
        nextCrossing += equator.spacing;
        const inverse = evaluateQuartic(path.radial, time);
        if (inverse === undefined) {
          return failure("arithmetic-domain");
        }
        const hitRadius = 1 / inverse;
        if (
          hitRadius >= disk.inner &&
          hitRadius <= disk.outer &&
          (!outcome || time < outcome.time)
        ) {
          outcome = { kind: "disk", time, radius: hitRadius, order };
        }
      }
      if (nextCrossing <= end) {
        return failure("budget-exhausted");
      }
    }
    if (outcome) {
      return { path, outcome };
    }
  }
  return failure("budget-exhausted");
}
