const validated = Symbol("SourceAppearance");

/** Source emission parameters, independent of optical geometry and display exposure. */
export interface SourceAppearance {
  readonly [validated]: true;
  /** Peak effective temperature of the unmodulated radial disk profile, in kelvin. */
  readonly diskTemperature: number;
  /** Fraction of the prescribed spatial and temporal emissivity variation. */
  readonly diskStructure: number;
  /** Linear multiplier shared by point stars and diffuse sky radiation. */
  readonly skyBrightness: number;
}

/** Validate UI and persisted source parameters before they reach spectral evaluation. */
export function createAppearance(
  input: unknown,
):
  | { readonly ok: true; readonly value: SourceAppearance }
  | { readonly ok: false; readonly error: string } {
  if (typeof input !== "object" || input === null || Array.isArray(input)) {
    return { ok: false, error: "Invalid source appearance." };
  }
  if (!("diskTemperature" in input) || !("diskStructure" in input) || !("skyBrightness" in input)) {
    return { ok: false, error: "Missing source appearance parameters." };
  }
  const { diskTemperature, diskStructure, skyBrightness } = input;
  if (
    typeof diskTemperature !== "number" ||
    !Number.isFinite(diskTemperature) ||
    typeof diskStructure !== "number" ||
    !Number.isFinite(diskStructure) ||
    typeof skyBrightness !== "number" ||
    !Number.isFinite(skyBrightness) ||
    diskTemperature < 1000 ||
    diskTemperature > 30000 ||
    diskStructure < 0 ||
    diskStructure > 1 ||
    skyBrightness < 0 ||
    skyBrightness > 8
  ) {
    return { ok: false, error: "Source appearance is outside the supported control range." };
  }
  return {
    ok: true,
    value: {
      [validated]: true,
      diskTemperature: Math.fround(diskTemperature),
      diskStructure: Math.fround(diskStructure) + 0,
      skyBrightness: Math.fround(skyBrightness) + 0,
    },
  };
}

const initial = createAppearance({ diskTemperature: 7000, diskStructure: 0.65, skyBrightness: 1 });
if (!initial.ok) {
  throw new Error(initial.error);
}
export const initialAppearance = initial.value;
