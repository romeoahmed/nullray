import faintStarsURL from "../src/data/faint-stars.bin?url";

/** Hash a canonical path/content sequence in native UTF-16 key order, independent of enumeration order and locale. */
function fingerprint(modules: Record<string, string>): Promise<ArrayBuffer> {
  const source = Object.keys(modules)
    .toSorted()
    .map((path) => JSON.stringify([path, modules[path]]))
    .join("\n");
  return crypto.subtle.digest("SHA-256", new TextEncoder().encode(source));
}

/**
 * Hash production TS/WGSL text and the bundled binary catalogue for run provenance.
 *
 * @remarks
 * Includes comments, so a documentation-only edit changes this identity without
 * changing optical behavior. UI markup/CSS are outside this optical source hash.
 *
 * @returns SHA-256 bytes; catalogue fetch or digest failures reject.
 */
export async function sourceFingerprint(): Promise<ArrayBuffer> {
  const response = await fetch(faintStarsURL);
  if (!response.ok) {
    throw new Error("The benchmark requires the complete stellar catalogue.");
  }
  const binaryHash = await crypto.subtle.digest("SHA-256", await response.arrayBuffer());
  return fingerprint({
    ...import.meta.glob<string>("../src/**/*.{ts,wgsl}", {
      query: "?raw",
      import: "default",
      eager: true,
    }),
    "../src/data/faint-stars.bin SHA-256": new Uint8Array(binaryHash).toHex(),
  });
}

/**
 * Hash benchmark code, browser project configuration, and dependency manifests.
 *
 * @returns SHA-256 bytes identifying the measurement harness separately from production.
 */
export function benchmarkFingerprint(): Promise<ArrayBuffer> {
  return fingerprint(
    import.meta.glob<string>(
      [
        "./*.ts",
        "../tests/support/*.ts",
        "../vitest.config.ts",
        "../package.json",
        "../pnpm-lock.yaml",
      ],
      {
        query: "?raw",
        import: "default",
        eager: true,
      },
    ),
  );
}
