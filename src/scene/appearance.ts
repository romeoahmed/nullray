import { between, record } from "./decode.ts";
import type { Result } from "./decode.ts";

const validated = Symbol("SourceAppearance");

/** Source emission parameters, independent of optical geometry and display exposure. */
export interface SourceAppearance {
  readonly [validated]: true;
  /** Peak effective temperature of the unmodulated radial disk profile, in kelvin. */
  readonly diskTemperature: number;
  /** Fraction of the prescribed spatial and temporal emissivity variation. */
  readonly diskStructure: number;
  /** Peak unperturbed vertical scale height H/r; zero selects the equatorial surface limit. */
  readonly diskThickness: number;
  /** Peak vertical gray optical depth before radial taper and turbulent modulation. */
  readonly diskOpticalDepth: number;
  /** Relative source luminosity in exterior universes other than the original illuminated domain. */
  readonly otherUniverses: number;
  /** Linear multiplier shared by point stars and diffuse sky radiation. */
  readonly skyBrightness: number;
}

/** Validate and quantize source inputs once, before spectral preparation or persistence. */
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
  diskTemperature: 2800,
  diskStructure: 1,
  diskThickness: 0.022,
  diskOpticalDepth: 1.8,
  otherUniverses: 0,
  skyBrightness: 0.035,
});
if (!initial.ok) {
  throw new Error(initial.error);
}
export const initialAppearance = initial.value;
