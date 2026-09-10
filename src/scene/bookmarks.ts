import { decodeView } from "./view.ts";

/** A named, validated view fragment; persistence belongs to the browser adapter. */
export interface Bookmark {
  readonly name: string;
  readonly fragment: string;
}

/**
 * Decode the complete collection atomically; corrupt storage must never become an empty library.
 * @param serialized - Stored JSON, or null when the browser has no collection yet.
 * @returns Fresh entries with unique names and validated versioned view fragments.
 * @throws Error when JSON, collection limits, names, or any view fail validation.
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
      typeof value !== "object" ||
      value === null ||
      !("name" in value) ||
      !("fragment" in value) ||
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
