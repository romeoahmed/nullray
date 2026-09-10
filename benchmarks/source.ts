/** Hash a canonical path/content sequence without relying on directory enumeration order. */
function fingerprint(modules: Record<string, string>): Promise<ArrayBuffer> {
  const source = Object.entries(modules)
    .toSorted(([a], [b]) => a.localeCompare(b))
    .map(([path, contents]) => `${path}\n${contents}`)
    .join("\n");
  return crypto.subtle.digest("SHA-256", new TextEncoder().encode(source));
}

/** Fingerprint every production TypeScript/WGSL module, including data and CPU preparation. */
export function sourceFingerprint(): Promise<ArrayBuffer> {
  return fingerprint(
    import.meta.glob<string>("../src/**/*.{ts,wgsl}", {
      query: "?raw",
      import: "default",
      eager: true,
    }),
  );
}

/** Identify the measuring code and browser project configuration separately from production. */
export function benchmarkFingerprint(): Promise<ArrayBuffer> {
  return fingerprint(
    import.meta.glob<string>(["./*.ts", "../vitest.config.ts"], {
      query: "?raw",
      import: "default",
      eager: true,
    }),
  );
}
