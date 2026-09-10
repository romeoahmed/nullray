/** Atomic validation outcome; failures never carry a partially prepared value. */
export type Result<T> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: string };

/** JSON-like object boundary. Arrays and null are not domain records. */
export const record = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** Finite binary64 input, before any deliberate GPU quantization. */
export const finite = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

/** Closed finite control range. Use explicit strict comparisons for physical open boundaries. */
export const between = (value: unknown, minimum: number, maximum: number): value is number =>
  finite(value) && value >= minimum && value <= maximum;
