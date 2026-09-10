import { contract, lower } from "./geometry.ts";
import type { FourVector, KerrGeometry } from "./geometry.ts";
import type { Spacetime } from "./spacetime.ts";
import type { Vec3 } from "./vector.ts";

/** Neutral geodesic invariants; massSquared is zero for light and one for unit four-velocity. */
export interface MotionConstants {
  readonly energy: number;
  readonly angularMomentum: number;
  readonly carter: number;
  readonly massSquared: number;
}

/** Separated future Mino derivatives, without normalizing by Killing energy. */
export interface MotionInitial {
  readonly constants: MotionConstants;
  readonly radialVelocity: number;
  readonly polarVelocity: number;
  readonly angular: Vec3;
}

/** Extract Carter data from a Cartesian tangent, including the axis limit. */
export function motionFromTangent(
  space: Spacetime,
  geometry: KerrGeometry,
  tangent: FourVector,
  massSquared: 0 | 1,
): MotionInitial {
  const momentum = lower(geometry, tangent);
  const { radius: r, inclination: theta } = geometry.point;
  const sine = theta === 0 || theta === Math.PI ? 0 : Math.sin(theta);
  const cosine = theta === Math.PI / 2 ? 0 : Math.cos(theta);
  const sign = geometry.point.chart === "ingoing" ? 1 : -1;
  const energy = -momentum[0];
  const nMomentum = sign * (contract(geometry.principal, momentum) - momentum[0]);
  const thetaMomentum = contract(geometry.polar, momentum);
  const phiMomentum = contract(geometry.azimuthal, momentum);
  const pTheta = r * thetaMomentum + sign * space.spin * cosine * phiMomentum;
  const scaledL = r * phiMomentum - sign * space.spin * (sine * nMomentum + cosine * thetaMomentum);
  const nVelocity = sign * (contract(geometry.principal, tangent) - tangent[0]);
  const thetaVelocity = contract(geometry.polar, tangent);
  const phiVelocity = contract(geometry.azimuthal, tangent);
  return {
    constants: {
      energy,
      angularMomentum: sine * scaledL,
      carter:
        pTheta * pTheta +
        cosine * cosine * (space.spin ** 2 * (massSquared - energy * energy) + scaledL * scaledL),
      massSquared,
    },
    radialVelocity:
      geometry.sigma * nVelocity +
      sign * space.spin * sine * (-sign * space.spin * cosine * thetaVelocity + r * phiVelocity),
    polarVelocity: pTheta,
    angular: [
      pTheta * geometry.azimuthal[1] - scaledL * geometry.polar[1],
      pTheta * geometry.azimuthal[2] - scaledL * geometry.polar[2],
      pTheta * geometry.azimuthal[3] - scaledL * geometry.polar[3],
    ],
  };
}

/** Squared radial Mino velocity, valid at signed radius and in every regular block. */
export function motionRadialPotential(
  space: Spacetime,
  constants: MotionConstants,
  radius: number,
): number {
  const { spin: a, charge: q } = space;
  const { energy: e, angularMomentum: l, carter: c, massSquared: m } = constants;
  const p = e * (radius * radius + a * a) - a * l;
  return (
    p * p -
    (radius * radius - 2 * radius + a * a + q * q) * ((l - a * e) ** 2 + c + m * radius * radius)
  );
}
