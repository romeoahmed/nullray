import { heatingProfile, plasmaProfile } from "../physics/plasma.ts";
import type { Plasma } from "../physics/plasma.ts";
import { between, finite, record } from "./decode.ts";

export const initialPlasma: Plasma = { density: 1e12, radius: 10, frequencyGHz: 10, heating: 0 };

/** Validate physical inputs and their GPU coefficients; null selects vacuum optical imaging. */
export function decodePlasma(input: unknown): Plasma | null | undefined {
  if (input === null) {
    return null;
  }
  if (!record(input)) {
    return undefined;
  }
  const { density, radius, frequencyGHz, heating } = input;
  if (
    !finite(density) ||
    !finite(radius) ||
    !finite(frequencyGHz) ||
    !between(heating, 0, 9) ||
    density <= 0 ||
    radius <= 0 ||
    frequencyGHz <= 0
  ) {
    return undefined;
  }
  const plasma = { density, radius, frequencyGHz, heating: Math.fround(heating) };
  const { amplitude, scaleSquared } = plasmaProfile(plasma);
  if (
    !Number.isFinite(amplitude) ||
    !Number.isFinite(scaleSquared) ||
    scaleSquared <= 0 ||
    !Number.isFinite(Math.fround(frequencyGHz * 0.04799243073366221))
  ) {
    return undefined;
  }
  const heat = heatingProfile(plasma);
  if (
    heat &&
    ![heat.inner, heat.outer, heat.frequency, Math.fround(heat.outer ** 2 + scaleSquared)].every(
      Number.isFinite,
    )
  ) {
    return undefined;
  }
  return plasma;
}
