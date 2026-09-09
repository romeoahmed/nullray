import { evaluateGeodesic } from "./geodesic.ts";
import type { SeparatedPath } from "./geodesic.ts";
import type { Photon } from "../../src/physics/photon.ts";
import type { Spacetime } from "../../src/physics/spacetime.ts";
import { equatorCrossings } from "./equator.ts";
import { jacobi } from "./elliptic.ts";

type Pair = readonly [azimuth: number, coordinateTime: number];

export type Transport =
  | {
      readonly kind: "resolved";
      readonly azimuth: number;
      readonly coordinateTime: number;
      readonly errorEstimate: Pair;
      readonly evaluations: number;
    }
  | { readonly kind: "unresolved"; readonly reason: "arithmetic-domain" | "budget-exhausted" };

const simpson = (left: Pair, middle: Pair, right: Pair, width: number): Pair => [
  (width * (left[0] + 4 * middle[0] + right[0])) / 6,
  (width * (left[1] + 4 * middle[1] + right[1])) / 6,
];
const failure = (reason: "arithmetic-domain" | "budget-exhausted"): Transport => ({
  kind: "unresolved",
  reason,
});

/**
 * Integrate separated azimuth/time rates along an analytic exterior segment.
 * Adaptive Simpson quadrature is a candidate transport implementation, not a
 * second geodesic integrator. The returned time is relative to the observer.
 * Disable time at the sky endpoint, where absolute coordinate time diverges.
 * Positive-C polar motion removes the axis singularity analytically. Other
 * Carter branches currently retain direct, coordinate-regular quadrature.
 */
export function integrateTransport(
  space: Spacetime,
  photon: Photon,
  path: SeparatedPath,
  end: number,
  includeTime: boolean,
  tolerance = 1e-9,
  budget = 8192,
): Transport {
  const { spin: a, charge: q } = space;
  const { energy: e, angularMomentum: l } = photon;
  const polarPhase = equatorCrossings(space, photon, path.inclination);
  if (polarPhase.kind === "unresolved") {
    return failure("arithmetic-domain");
  }
  let baseAzimuth = 0;
  let complement = 0;
  if (polarPhase.kind === "crossings") {
    const { amplitude2: z, frequency, phase, spacing, m } = polarPhase;
    // (1-z)(C+A z)=L² z avoids cancellation in 1-z close to an axis.
    complement = (l * l * z) / (photon.carter + (a * e) ** 2 * z);
    const primitive = (u: number) => {
      const period = 2 * spacing * frequency;
      if (l === 0) {
        // Right-continuous chart at an axis: the initial value uses the departing meridian.
        return Math.PI * (Math.floor(u / (period / 2)) + 0.5);
      }
      const cycles = Math.floor((u + period / 2) / period);
      const j = jacobi(u - cycles * period, m);
      return Math.atan2(j.sn, Math.sqrt(complement) * j.cn) + cycles * 2 * Math.PI;
    };
    const coefficient =
      ((l < 0 ? -1 : 1) * Math.sqrt(photon.carter / z + (a * e) ** 2)) / frequency;
    baseAzimuth = -coefficient * (primitive(phase + frequency * end) - polarPhase.azimuthPhase);
  }
  let evaluations = 0;
  const rate = (time: number): Pair | undefined => {
    evaluations++;
    const point = evaluateGeodesic(path, time);
    if (!point) {
      return undefined;
    }
    const u = point.inverseRadius;
    const u2 = u * u;
    let sin2 = 1 - point.cosineTheta ** 2;
    let polarRate = 0;
    if (polarPhase.kind === "crossings") {
      const j = jacobi(polarPhase.phase + polarPhase.frequency * time, polarPhase.m);
      sin2 = complement + polarPhase.amplitude2 * j.sn * j.sn;
      if (l !== 0) {
        polarRate = (l * polarPhase.m * j.sn * j.sn) / ((1 + j.dn) * sin2);
      }
    } else {
      if (!(sin2 > 0)) {
        return undefined;
      }
      polarRate = l / sin2;
    }
    const delta = 1 - 2 * u + (a * a + q * q) * u2;
    if (!(delta > 0) || (includeTime && !(u > 0))) {
      return undefined;
    }
    // Combine a P/Delta - a E before evaluation: the asymptotic radial term
    // tends smoothly to zero without subtracting two nearly equal constants.
    const phi = -((a * (2 * e * u - (q * q * e + a * l) * u2)) / delta + polarRate);
    const t = includeTime
      ? -(
          ((1 + a * a * u2) * (e + (a * a * e - a * l) * u2)) / (u2 * delta) +
          a * l -
          a * a * e * sin2
        )
      : 0;
    return Number.isFinite(phi + t) ? [phi, t] : undefined;
  };
  if (!(end >= 0) || !Number.isFinite(end) || !(tolerance > 0) || !Number.isFinite(tolerance)) {
    return failure("arithmetic-domain");
  }
  if (end === 0) {
    return {
      kind: "resolved",
      azimuth: 0,
      coordinateTime: 0,
      errorEstimate: [0, 0],
      evaluations: 0,
    };
  }
  // Seed intervals below the polar phase scale so a full oscillation cannot
  // disappear between the initial Simpson samples. Local refinement still
  // controls narrow radial/polar features; its error is an estimate, not a bound.
  const frequency = Math.sqrt(l * l + 2 * Math.abs(photon.carter) + 2 * (a * e) ** 2);
  const panels = Math.max(1, Math.ceil(end * frequency * 8));
  if (!Number.isSafeInteger(budget) || panels * 3 > budget) {
    return failure("budget-exhausted");
  }
  const stack: Array<{
    begin: number;
    end: number;
    left: Pair;
    middle: Pair;
    right: Pair;
    estimate: Pair;
    tolerance: number;
  }> = [];
  for (let i = panels - 1; i >= 0; i--) {
    const begin = (end * i) / panels;
    const finish = (end * (i + 1)) / panels;
    const left = rate(begin);
    const middle = rate((begin + finish) / 2);
    const right = rate(finish);
    if (!left || !middle || !right) {
      return failure("arithmetic-domain");
    }
    stack.push({
      begin,
      end: finish,
      left,
      middle,
      right,
      estimate: simpson(left, middle, right, finish - begin),
      tolerance: tolerance / panels,
    });
  }
  let azimuth = baseAzimuth;
  let coordinateTime = 0;
  const errorEstimate: [number, number] = [0, 0];
  while (stack.length) {
    if (evaluations + 2 > budget) {
      return failure("budget-exhausted");
    }
    const segment = stack.pop();
    if (!segment) {
      throw new Error("Missing quadrature segment.");
    }
    const center = (segment.begin + segment.end) / 2;
    if (center === segment.begin || center === segment.end) {
      return failure("budget-exhausted");
    }
    const lm = rate((segment.begin + center) / 2);
    const rm = rate((center + segment.end) / 2);
    if (!lm || !rm) {
      return failure("arithmetic-domain");
    }
    const lower = simpson(segment.left, lm, segment.middle, center - segment.begin);
    const upper = simpson(segment.middle, rm, segment.right, segment.end - center);
    const phiError = (lower[0] + upper[0] - segment.estimate[0]) / 15;
    const timeError = (lower[1] + upper[1] - segment.estimate[1]) / 15;
    if (Math.abs(phiError) <= segment.tolerance && Math.abs(timeError) <= segment.tolerance) {
      azimuth += lower[0] + upper[0] + phiError;
      coordinateTime += lower[1] + upper[1] + timeError;
      errorEstimate[0] += Math.abs(phiError);
      errorEstimate[1] += Math.abs(timeError);
    } else {
      stack.push(
        {
          begin: center,
          end: segment.end,
          left: segment.middle,
          middle: rm,
          right: segment.right,
          estimate: upper,
          tolerance: segment.tolerance / 2,
        },
        {
          begin: segment.begin,
          end: center,
          left: segment.left,
          middle: lm,
          right: segment.middle,
          estimate: lower,
          tolerance: segment.tolerance / 2,
        },
      );
    }
  }
  return { kind: "resolved", azimuth, coordinateTime, errorEstimate, evaluations };
}
