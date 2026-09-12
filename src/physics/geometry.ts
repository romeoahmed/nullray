import type { Spacetime } from "./spacetime.ts";
import type { Vec3 } from "./vector.ts";

/**
 * Four components ordered as Cartesian Kerr–Schild `(T, X, Y, Z)`.
 *
 * @remarks
 * The operation determines whether these are tangent or covector components;
 * the tuple type does not encode index position. Coordinate lengths use `M = 1`.
 */
export type FourVector = readonly [number, number, number, number];

/** Ingoing and outgoing charts cover different horizon components of the extension. */
export type KerrChart = "ingoing" | "outgoing";

/**
 * Oblate coordinates at one event in the selected regular chart.
 *
 * @remarks
 * `radius` is signed and measured in M. `inclination` lies in [0, π];
 * `azimuth` is the null chart's rotating longitude, not BL longitude.
 * Both angles are in radians.
 */
export interface KerrPoint {
  readonly radius: number;
  readonly inclination: number;
  readonly azimuth: number;
  readonly chart: KerrChart;
}

/** Real regular horizons. Schwarzschild's zero root is a singularity, not an inner horizon. */
export type Horizons =
  | { readonly kind: "none" }
  | { readonly kind: "single"; readonly outer: number }
  | { readonly kind: "extremal"; readonly radius: number }
  | { readonly kind: "pair"; readonly outer: number; readonly inner: number };

/**
 * Classify the roots of Δ without clamping a horizonless geometry.
 *
 * @returns Regular horizon radii in M, distinguishing the singular Schwarzschild zero root.
 * @throws RangeError - If the squared spin/charge magnitude is nonfinite.
 */
export function horizons({ spin: a, charge: q }: Spacetime): Horizons {
  const squared = a * a + q * q;
  if (!Number.isFinite(squared)) {
    throw new RangeError("Spacetime parameters must have finite squared magnitude.");
  }
  if (squared > 1) {
    return { kind: "none" };
  }
  if (squared === 1) {
    return { kind: "extremal", radius: 1 };
  }
  if (squared === 0) {
    return { kind: "single", outer: 2 };
  }
  const outer = 1 + Math.sqrt(1 - squared);
  return { kind: "pair", outer, inner: squared / outer };
}

/**
 * Find the algebraic stationary-limit roots at a polar angle in radians.
 *
 * @returns Inner/outer radii in M, or `undefined` for nonfinite data or no real roots.
 * The caller must exclude singular points; these roots do not classify event horizons.
 */
export function stationaryLimits(
  { spin: a, charge: q }: Spacetime,
  inclination: number,
): readonly [inner: number, outer: number] | undefined {
  const squared = (a * Math.cos(inclination)) ** 2 + q * q;
  if (!Number.isFinite(squared) || squared > 1) {
    return undefined;
  }
  const outer = 1 + Math.sqrt(1 - squared);
  return [squared / outer, outer];
}

/**
 * Kerr–Schild metric data for `g = η + factor k ⊗ k`, with signature −+++.
 *
 * @remarks
 * `principal` stores the null covector k; `polar` and `azimuthal` are spatial
 * tangents. `position` is Cartesian and measured in M. `sigma` and `delta`
 * are the oblate metric functions in M². The input point is borrowed.
 */
export interface KerrGeometry {
  readonly point: KerrPoint;
  readonly position: Vec3;
  readonly principal: FourVector;
  readonly polar: FourVector;
  readonly azimuthal: FourVector;
  readonly factor: number;
  readonly sigma: number;
  readonly delta: number;
}

/**
 * Construct metric data at a finite event, including regular axes and signed radius.
 *
 * @returns Geometry borrowing `point`, or `undefined` for invalid coordinates,
 * Σ = 0, or nonrepresentable intermediate metric data.
 */
export function kerrGeometry(space: Spacetime, point: KerrPoint): KerrGeometry | undefined {
  const { spin: a, charge: q } = space;
  const { radius: r, inclination: theta, azimuth: phi } = point;
  if (
    ![a, q, r, theta, phi].every(Number.isFinite) ||
    theta < 0 ||
    theta > Math.PI ||
    (point.chart !== "ingoing" && point.chart !== "outgoing")
  ) {
    return undefined;
  }
  const sine = theta === 0 || theta === Math.PI ? 0 : Math.sin(theta);
  const cosine = theta === Math.PI / 2 ? 0 : Math.cos(theta);
  const cp = Math.cos(phi);
  const sp = Math.sin(phi);
  const sign = point.chart === "ingoing" ? 1 : -1;
  const sigma = r * r + (a * cosine) ** 2;
  const delta = r * r - 2 * r + a * a + q * q;
  if (!(sigma > 0) || !Number.isFinite(sigma + delta)) {
    return undefined;
  }
  const factor = (2 * r - q * q) / sigma;
  if (!Number.isFinite(factor)) {
    return undefined;
  }
  return {
    point,
    position: [(r * cp - sign * a * sp) * sine, (r * sp + sign * a * cp) * sine, r * cosine],
    principal: [1, sign * sine * cp, sign * sine * sp, sign * cosine],
    polar: [0, cosine * cp, cosine * sp, -sine],
    azimuthal: [0, -sp, cp, 0],
    factor,
    sigma,
    delta,
  };
}

/** Pair a covector with a tangent, without a metric operation. */
export function contract(covector: FourVector, tangent: FourVector): number {
  return (
    covector[0] * tangent[0] +
    covector[1] * tangent[1] +
    covector[2] * tangent[2] +
    covector[3] * tangent[3]
  );
}

/** Metric-lower a Cartesian tangent. The returned covector belongs to the same chart. */
export function lower(geometry: KerrGeometry, tangent: FourVector): FourVector {
  const k = geometry.principal;
  const weight = geometry.factor * contract(k, tangent);
  return [
    -tangent[0] + weight * k[0],
    tangent[1] + weight * k[1],
    tangent[2] + weight * k[2],
    tangent[3] + weight * k[3],
  ];
}

/** Raise a covector with the algebraic Kerr–Schild inverse in the same chart. */
export function raise(geometry: KerrGeometry, covector: FourVector): FourVector {
  const k = geometry.principal;
  const weight =
    geometry.factor *
    (-k[0] * covector[0] + k[1] * covector[1] + k[2] * covector[2] + k[3] * covector[3]);
  return [
    -covector[0] + weight * k[0],
    covector[1] - weight * k[1],
    covector[2] - weight * k[2],
    covector[3] - weight * k[3],
  ];
}

/** Scalar product of two tangents; valid across horizons and on the regular zero-radius disk. */
export function inner(geometry: KerrGeometry, left: FourVector, right: FourVector): number {
  return contract(lower(geometry, left), right);
}

/** Linear combination used for tetrads and physical four-momenta. */
export function combine(left: FourVector, a: number, right: FourVector, b: number): FourVector {
  return [
    a * left[0] + b * right[0],
    a * left[1] + b * right[1],
    a * left[2] + b * right[2],
    a * left[3] + b * right[3],
  ];
}

/**
 * Push an oblate `(T, r, θ, ψ)` tangent into Cartesian Kerr–Schild components.
 *
 * @remarks
 * The time component is KS time T, not null time w. `space` must match
 * `geometry`; both representations use the same path parameter.
 */
export function toCartesian(
  space: Spacetime,
  geometry: KerrGeometry,
  tangent: FourVector,
): FourVector {
  const { radius: r, inclination: theta } = geometry.point;
  const sign = geometry.point.chart === "ingoing" ? 1 : -1;
  const sine = theta === 0 || theta === Math.PI ? 0 : Math.sin(theta);
  const cosine = theta === Math.PI / 2 ? 0 : Math.cos(theta);
  const radial = tangent[1] - sign * space.spin * sine * sine * tangent[3];
  const polar = r * tangent[2] - sign * space.spin * sine * cosine * tangent[3];
  const azimuthal = sign * space.spin * cosine * tangent[2] + r * sine * tangent[3];
  const n = geometry.principal;
  const t = geometry.polar;
  const p = geometry.azimuthal;
  return [
    tangent[0],
    sign * radial * n[1] + polar * t[1] + azimuthal * p[1],
    sign * radial * n[2] + polar * t[2] + azimuthal * p[2],
    sign * radial * n[3] + polar * t[3] + azimuthal * p[3],
  ];
}

/**
 * Pull a Cartesian tangent into oblate `(T, r, θ, ψ)` components.
 *
 * @returns A fresh tuple, or `undefined` on either axis where longitude is undefined.
 * `space` must match `geometry`; this helper does not validate arbitrary tangents.
 */
export function fromCartesian(
  space: Spacetime,
  geometry: KerrGeometry,
  tangent: FourVector,
): FourVector | undefined {
  const { radius: r, inclination: theta } = geometry.point;
  if (theta === 0 || theta === Math.PI) {
    return undefined;
  }
  const sign = geometry.point.chart === "ingoing" ? 1 : -1;
  const sine = Math.sin(theta);
  const cosine = theta === Math.PI / 2 ? 0 : Math.cos(theta);
  const radial = sign * (contract(geometry.principal, tangent) - tangent[0]);
  const polar = contract(geometry.polar, tangent);
  const azimuthal = contract(geometry.azimuthal, tangent);
  const thetaVelocity = (r * polar + sign * space.spin * cosine * azimuthal) / geometry.sigma;
  const scaledPhiVelocity = (-sign * space.spin * cosine * polar + r * azimuthal) / geometry.sigma;
  return [
    tangent[0],
    radial + sign * space.spin * sine * scaledPhiVelocity,
    thetaVelocity,
    scaledPhiVelocity / sine,
  ];
}
