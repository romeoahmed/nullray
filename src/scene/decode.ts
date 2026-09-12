import type { Vec3 } from "../physics/vector.ts";

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

/** Map an accepted value without invoking the transform or replacing the error on failure. */
export function mapResult<T, U>(result: Result<T>, transform: (value: T) => U): Result<U> {
  return result.ok ? { ok: true, value: transform(result.value) } : result;
}

/** Narrow one external choice without coercion or a type assertion. */
export function oneOf<const T extends string>(value: unknown, choices: readonly T[]): value is T {
  return choices.some((choice) => choice === value);
}

/** Finite vector with exactly three present components; sparse arrays are rejected. */
export const vector3 = (value: unknown): value is Vec3 =>
  Array.isArray(value) &&
  value.length === 3 &&
  finite(value[0]) &&
  finite(value[1]) &&
  finite(value[2]);
