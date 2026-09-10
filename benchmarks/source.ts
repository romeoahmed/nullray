import faintStarsURL from "../src/data/faint-stars.bin?url";

/** Hash a canonical path/content sequence without relying on directory enumeration order. */
function fingerprint(modules: Record<string, string>): Promise<ArrayBuffer> {
  const source = Object.entries(modules)
    .toSorted(([a], [b]) => a.localeCompare(b))
    .map(([path, contents]) => JSON.stringify([path, contents]))
    .join("\n");
  return crypto.subtle.digest("SHA-256", new TextEncoder().encode(source));
}

/** Fingerprint production code and catalogue content, including the fetched binary source. */
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

/** Identify the measuring code and browser project configuration separately from production. */
export function benchmarkFingerprint(): Promise<ArrayBuffer> {
  return fingerprint(
    import.meta.glob<string>(
      ["./*.ts", "../vitest.config.ts", "../package.json", "../pnpm-lock.yaml"],
      {
        query: "?raw",
        import: "default",
        eager: true,
      },
    ),
  );
}
