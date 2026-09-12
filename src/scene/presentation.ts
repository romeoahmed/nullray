import { between, finite } from "./decode.ts";

/** Accepted enum choices shared by persisted views and native controls. */
export const viewChoices = {
  navigation: ["orbit", "free"],
  display: ["auto", "hdr", "sdr"],
  diagnostic: ["image", "frequency", "order", "domain", "polarization", "angle"],
} as const;

/** Detector/display values; source emission and optical preparation remain independent. */
export interface Presentation {
  /** Log₂ display exposure multiplier in stops, from −6 through +6. */
  readonly exposureEV: number;
  /** Photographic neutral blackbody in kelvin; independent of source temperature. */
  readonly whiteBalance: number;
  /** Fraction of direct radiance redistributed through glare, in [0, 1]. */
  readonly bloom: number;
  /** Linear analyzer angle in [0, π) radians; null bypasses the analyzer. */
  readonly analyzer: number | null;
  readonly diagnostic: (typeof viewChoices.diagnostic)[number];
}

/** Closed display ranges; action names map directly to their persisted field. */
export const displayControls = {
  exposure: { field: "exposureEV", minimum: -6, maximum: 6 },
  "white-balance": { field: "whiteBalance", minimum: 2500, maximum: 12000 },
  bloom: { field: "bloom", minimum: 0, maximum: 1 },
} as const satisfies Record<
  string,
  {
    readonly field: keyof Presentation;
    readonly minimum: number;
    readonly maximum: number;
  }
>;

/** Validate a display value at its shared closed range, without coercion or quantization. */
export function displayValue(
  control: keyof typeof displayControls,
  value: unknown,
): value is number {
  const { minimum, maximum } = displayControls[control];
  return between(value, minimum, maximum);
}

/** A bypassed analyzer or a finite angle in the half-open interval [0, π). */
export const analyzerValue = (value: unknown): value is number | null =>
  value === null || (finite(value) && value >= 0 && value < Math.PI);
