import { prepareQuartic, evaluateQuartic } from "./quartic.ts";
import type { QuarticPath } from "./quartic.ts";
import type { Photon } from "../../src/physics/photon.ts";
import type { Spacetime } from "../../src/physics/spacetime.ts";

export interface SeparatedPath {
  readonly inclination: number;
  readonly radial: QuarticPath;
  readonly polar: QuarticPath;
}

/**
 * Prepare inverse radius and cos(theta) as functions of backward Mino time.
 * Inverse radius keeps the sky endpoint finite and avoids large observer-radius
 * powers. Both polynomials retain zero and negative Killing-energy branches.
 */
export function prepareGeodesic(
  space: Spacetime,
  photon: Photon,
  radius: number,
  theta: number,
): SeparatedPath {
  const { energy: e, angularMomentum: l, carter: c } = photon;
  const a2e2 = (space.spin * e) ** 2;
  const k = (l - space.spin * e) ** 2 + c;
  const quadratic = a2e2 - l * l - c;
  return {
    inclination: theta,
    radial: prepareQuartic(
      [e * e, 0, quadratic, 2 * k, -(space.spin ** 2) * c - space.charge ** 2 * k],
      1 / radius,
      photon.radialVelocity / (radius * radius),
    ),
    polar: prepareQuartic(
      [c, 0, quadratic, 0, -a2e2],
      Math.cos(theta),
      Math.sin(theta) * photon.polarVelocity,
    ),
  };
}

/** Evaluate geometry only; visibility and exterior-segment bounds are separate. */
export function evaluateGeodesic(
  path: SeparatedPath,
  backwardTime: number,
): { readonly inverseRadius: number; readonly cosineTheta: number } | undefined {
  const inverseRadius = evaluateQuartic(path.radial, backwardTime);
  const cosineTheta = evaluateQuartic(path.polar, backwardTime);
  if (inverseRadius === undefined || cosineTheta === undefined) {
    return undefined;
  }
  return { inverseRadius, cosineTheta };
}
