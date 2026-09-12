import { between, record } from "./decode.ts";
import type { Result } from "./decode.ts";

const validated = Symbol("SourceAppearance");

/** Source emission parameters, independent of optical geometry and display exposure. */
export interface SourceAppearance {
  readonly [validated]: true;
  /** Peak effective temperature of the unmodulated radial disk profile, in kelvin. */
  readonly diskTemperature: number;
  /** Strength in [0, 1] of prescribed density, corrugation, and thermal variation. */
  readonly diskStructure: number;
  /** Peak unperturbed coordinate scale height H/r; zero selects a separate opaque surface model. */
  readonly diskThickness: number;
  /** Peak of the normalized radial coordinate-column optical depth before density modulation. */
  readonly diskOpticalDepth: number;
  /** Luminosity scale for other stationary source domains; leaves their disk extinction intact. */
  readonly otherUniverses: number;
  /** Linear multiplier shared by point stars and diffuse sky radiation. */
  readonly skyBrightness: number;
}

/**
 * Validate source controls and quantize accepted values to f32 once.
 *
 * @returns A complete appearance record or a range/type error. A positive
 * volume thickness must remain positive after quantization; zero explicitly selects the surface model.
 */
export function createAppearance(input: unknown): Result<SourceAppearance> {
  if (!record(input)) {
    return { ok: false, error: "Invalid source appearance." };
  }
  const {
    diskTemperature,
    diskStructure,
    diskThickness,
    diskOpticalDepth,
    otherUniverses,
    skyBrightness,
  } = input;
  if (
    !between(diskTemperature, 1000, 30000) ||
    !between(diskStructure, 0, 1) ||
    !between(diskThickness, 0, Math.fround(0.2)) ||
    (diskThickness > 0 && Math.fround(diskThickness) === 0) ||
    !between(diskOpticalDepth, 0, 100) ||
    !(Math.fround(diskOpticalDepth) > 0) ||
    !between(otherUniverses, 0, 1) ||
    !between(skyBrightness, 0, 64)
  ) {
    return { ok: false, error: "Source appearance is outside the supported control range." };
  }
  return {
    ok: true,
    value: {
      [validated]: true,
      diskTemperature: Math.fround(diskTemperature),
      diskStructure: Math.fround(diskStructure) + 0,
      diskThickness: Math.fround(diskThickness) + 0,
      diskOpticalDepth: Math.fround(diskOpticalDepth),
      otherUniverses: Math.fround(otherUniverses) + 0,
      skyBrightness: Math.fround(skyBrightness) + 0,
    },
  };
}

const initial = createAppearance({
  diskTemperature: 3000,
  diskStructure: 1,
  diskThickness: 0.018,
  diskOpticalDepth: 1.2,
  otherUniverses: 0,
  skyBrightness: 0.003,
});
if (!initial.ok) {
  throw new Error(initial.error);
}
export const initialAppearance = initial.value;
