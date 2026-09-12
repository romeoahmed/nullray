import type { SpacetimeBlock } from "./atlas.ts";

/**
 * Accepted inspection state, not an exact horizon intersection.
 *
 * @remarks
 * Radius is signed in M and inclination is in radians. Time is selected
 * null-chart time, or logarithmic amplitude when the chart tag identifies a bifurcation patch.
 */
export interface RayPoint {
  readonly radius: number;
  readonly inclination: number;
  readonly time: number;
  readonly block: SpacetimeBlock["kind"];
  readonly universe: number;
  readonly side: number;
  readonly chart: "ingoing" | "outgoing" | "bifurcation-ingoing" | "bifurcation-outgoing";
}

/**
 * Selected GPU ray with accepted-step samples and its final geometric classification.
 *
 * @remarks
 * `disk` also covers termination by volume opacity. A finite source-free
 * prefix need not reach a geometric endpoint; the inspector is not a pixel beam.
 */
export interface RayPath {
  readonly kind: "unresolved" | "infinity" | "disk" | "singularity" | "source-free";
  readonly points: readonly RayPoint[];
  readonly equatorialCrossings: number;
}
