import type { Jet } from "../physics/jet.ts";
import { finite, record } from "./decode.ts";

export const initialJet: Jet = {
  inner: 3,
  outer: 100,
  openingAngle: Math.PI / 18,
  speed: 0.9,
  density: 1e6,
  field: 1000,
  massSolar: 1e6,
  gammaMin: 600,
};

/** Validate a complete material prescription; null disables the outflow. */
export function decodeJet(input: unknown): Jet | null | undefined {
  if (input === null) {
    return null;
  }
  if (!record(input)) {
    return undefined;
  }
  const { inner, outer, openingAngle, speed, density, field, massSolar, gammaMin } = input;
  if (
    !finite(inner) ||
    !finite(outer) ||
    !finite(openingAngle) ||
    !finite(speed) ||
    !finite(density) ||
    !finite(field) ||
    !finite(massSolar) ||
    !finite(gammaMin)
  ) {
    return undefined;
  }
  if (
    ![inner, outer, openingAngle, speed, density, field, massSolar, gammaMin].every((value) =>
      Number.isFinite(Math.fround(value)),
    ) ||
    !(
      Math.fround(inner) > 0 &&
      Math.fround(outer) > Math.fround(inner) &&
      openingAngle > 0 &&
      openingAngle < Math.PI / 2 &&
      Math.fround(Math.cos(openingAngle)) < 1 &&
      Math.fround(speed) > 0 &&
      Math.fround(speed) < 1 &&
      density > 0 &&
      field > 0 &&
      massSolar > 0 &&
      gammaMin >= 1 &&
      gammaMin <= 1e6
    )
  ) {
    return undefined;
  }
  return {
    inner: Math.fround(inner),
    outer: Math.fround(outer),
    openingAngle,
    speed: Math.fround(speed),
    density,
    field,
    massSolar,
    gammaMin,
  };
}
