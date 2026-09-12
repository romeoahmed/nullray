import { record } from "./decode.ts";
import { decodeView } from "./view.ts";

/** A named, validated view fragment; persistence belongs to the browser adapter. */
export interface Bookmark {
  readonly name: string;
  readonly fragment: string;
}

/**
 * Decode a saved-view collection atomically.
 *
 * @param serialized - Stored JSON; null alone means no library has been stored.
 * @returns Fresh validated entries with unique names; corrupt storage is not an empty library.
 * @throws SyntaxError - If the stored text is not JSON.
 * @throws Error - If collection limits, names, or view fragments fail validation.
 */
export function decodeBookmarks(serialized: string | null): readonly Bookmark[] {
  if (serialized === null) {
    return [];
  }
  const values: unknown = JSON.parse(serialized);
  if (!Array.isArray(values) || values.length > 100) {
    throw new Error("Invalid saved-view collection.");
  }
  const entries: Bookmark[] = [];
  const names = new Set<string>();
  const items: readonly unknown[] = values;
  for (const value of items) {
    if (
      !record(value) ||
      typeof value.name !== "string" ||
      !value.name.trim() ||
      value.name.length > 80 ||
      typeof value.fragment !== "string" ||
      decodeView(value.fragment).kind !== "view" ||
      names.has(value.name)
    ) {
      throw new Error("Invalid saved view.");
    }
    names.add(value.name);
    entries.push({ name: value.name, fragment: value.fragment });
  }
  return entries;
}
