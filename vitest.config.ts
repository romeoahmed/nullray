import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";
import type { BrowserConfigOptions } from "vitest/node";

/** Fresh project options: Vitest assigns instance names while expanding browser projects. */
const browser = (): BrowserConfigOptions => ({
  enabled: true,
  headless: true,
  provider: playwright({ launchOptions: { channel: "chromium" } }),
  instances: [{ browser: "chromium" }],
});

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: "cpu",
          include: ["tests/cpu/**/*.test.ts"],
          experimental: { viteModuleRunner: false },
          benchmark: { include: ["benchmarks/cpu.bench.ts"] },
        },
      },
      {
        optimizeDeps: { include: ["fast-check"] },
        test: {
          name: "gpu",
          include: ["tests/gpu/**/*.test.ts"],
          benchmark: { include: ["benchmarks/optics.bench.ts", "benchmarks/startup.bench.ts"] },
          fileParallelism: false,
          browser: browser(),
        },
      },
      {
        test: {
          name: "browser",
          include: ["tests/browser/**/*.test.ts"],
          fileParallelism: false,
          browser: browser(),
        },
      },
    ],
  },
});
