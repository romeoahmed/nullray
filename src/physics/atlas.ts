import type { KerrChart } from "./geometry.ts";
import { horizons } from "./geometry.ts";
import type { Spacetime } from "./spacetime.ts";

/** A block in the unquotiented analytic extension; negative r belongs to an interior block. */
export type SpacetimeBlock =
  | { readonly kind: "exterior" | "interior"; readonly universe: number; readonly side: 1 | -1 }
  | { readonly kind: "black-hole" | "white-hole"; readonly universe: number }
  | { readonly kind: "naked" }
  | { readonly kind: "disconnected" };

/** The original right-hand asymptotically flat exterior. */
export const originalExterior: SpacetimeBlock = { kind: "exterior", universe: 0, side: 1 };

/** Choose the reference block for a newly placed observer; continuation retains its own history. */
export function initialBlock(space: Spacetime, radius: number, chart: KerrChart): SpacetimeBlock {
  // With zero spin, r=0 is singular at every latitude; the negative-mass end is disconnected.
  if (space.spin === 0 && radius < 0) {
    return { kind: "disconnected" };
  }
  const structure = horizons(space);
  switch (structure.kind) {
    case "none":
      return { kind: "naked" };
    case "single":
      return radius > structure.outer
        ? originalExterior
        : { kind: chart === "ingoing" ? "black-hole" : "white-hole", universe: 0 };
    case "extremal":
      return radius > structure.radius
        ? originalExterior
        : { kind: "interior", universe: 0, side: 1 };
    case "pair":
      if (radius > structure.outer) {
        return originalExterior;
      }
      return radius < structure.inner
        ? { kind: "interior", universe: 0, side: 1 }
        : { kind: chart === "ingoing" ? "black-hole" : "white-hole", universe: 0 };
    default:
      throw new RangeError("Unknown horizon topology.");
  }
}

/**
 * Cross one nondegenerate horizon, preserving its causal component.
 * radialSign is the future radial tangent, even when evaluating a ray backward.
 * side is the sign of P on the stationary side of the crossed horizon.
 */
export function crossHorizon(
  block: SpacetimeBlock,
  horizon: "outer" | "inner",
  radialSign: 1 | -1,
  side: 1 | -1,
): SpacetimeBlock {
  if (horizon === "outer") {
    switch (block.kind) {
      case "exterior":
        return { kind: radialSign < 0 ? "black-hole" : "white-hole", universe: block.universe };
      case "black-hole":
      case "white-hole":
        return { kind: "exterior", universe: block.universe, side };
      case "interior":
      case "naked":
      case "disconnected":
      default:
        throw new RangeError("An outer horizon must adjoin an exterior block.");
    }
  }
  switch (block.kind) {
    case "black-hole":
      return { kind: "interior", universe: block.universe, side };
    case "white-hole":
      return { kind: "interior", universe: block.universe - 1, side };
    case "interior":
      return {
        kind: radialSign < 0 ? "black-hole" : "white-hole",
        universe: block.universe + (radialSign > 0 ? 1 : 0),
      };
    case "exterior":
    case "naked":
    case "disconnected":
    default:
      throw new RangeError("An inner horizon must adjoin an interior block.");
  }
}

/** Degenerate-horizon continuation has no intervening black/white-hole block. */
export function crossExtremalHorizon(
  block: SpacetimeBlock,
  radialSign: 1 | -1,
  side: 1 | -1,
): SpacetimeBlock {
  switch (block.kind) {
    case "exterior":
      return { kind: "interior", universe: block.universe - (radialSign > 0 ? 1 : 0), side };
    case "interior":
      return { kind: "exterior", universe: block.universe + (radialSign > 0 ? 1 : 0), side };
    case "black-hole":
    case "white-hole":
    case "naked":
    case "disconnected":
    default:
      throw new RangeError("A degenerate horizon joins an exterior and an interior block.");
  }
}

/**
 * Antiderivatives of 1/Δ and (r²+a²)/Δ, normalized by explicit logarithms in units M.
 * Undefined at a horizon: transition charts in an overlap, not on the pole.
 */
export function chartPrimitives(
  { spin: a, charge: q }: Spacetime,
  radius: number,
): readonly [longitude: number, tortoise: number] | undefined {
  const discriminant = 1 - a * a - q * q;
  const x = radius - 1;
  const delta = x * x - discriminant;
  if (delta === 0 || !Number.isFinite(delta)) {
    return undefined;
  }
  let integral: number;
  if (discriminant > 0) {
    const root = Math.sqrt(discriminant);
    integral = (Math.log(Math.abs(x - root)) - Math.log(Math.abs(x + root))) / (2 * root);
  } else if (discriminant === 0) {
    integral = -1 / x;
  } else {
    const root = Math.sqrt(-discriminant);
    integral = (Math.atan2(x, root) - Math.PI / 2) / root;
  }
  const tortoise = radius + Math.log(Math.abs(delta)) + (2 - q * q) * integral;
  return Number.isFinite(integral + tortoise) ? [a * integral, tortoise] : undefined;
}

/** Coordinate sign; it is independent of the time orientation of a block. */
export const chartSign = (chart: KerrChart): 1 | -1 => (chart === "ingoing" ? 1 : -1);
