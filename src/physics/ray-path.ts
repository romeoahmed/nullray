import type { SpacetimeBlock } from "./atlas.ts";

/** An accepted GPU ray state in signed radius and polar angle; time is null-chart time or the logarithmic amplitude in a bifurcation patch. */
export interface RayPoint {
  readonly radius: number;
  readonly inclination: number;
  readonly time: number;
  readonly block: SpacetimeBlock["kind"];
  readonly universe: number;
  readonly side: number;
  readonly chart: "ingoing" | "outgoing" | "bifurcation-ingoing" | "bifurcation-outgoing";
}

/** One selected detector ray, sampled at accepted steps; horizon intersections remain bracketed. */
export interface RayPath {
  readonly kind: "unresolved" | "infinity" | "disk" | "singularity" | "source-free";
  readonly points: readonly RayPoint[];
  readonly equatorialCrossings: number;
}
