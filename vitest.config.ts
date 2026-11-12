import { playwright } from "@vitest/browser-playwright";
import { defineConfig, defineProject } from "vitest/config";

/** Browser projects share one physical GPU: run them in order, one file at a time. */
const browserProject = (name: string, order: number, benchmark: string[] = []) =>
  defineProject({
    test: {
      name,
      include: [`tests/${name}/**/*.test.ts`],
      sequence: { groupOrder: order },
      fileParallelism: false,
      benchmark: { include: benchmark, retainSamples: true },
      browser: {
        enabled: true,
        headless: true,
        provider: playwright({ launchOptions: { channel: "chromium" } }),
        instances: [{ browser: "chromium" }],
      },
    },
  });

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: "cpu",
          include: ["tests/cpu/**/*.test.ts"],
          // Node runs erasable TypeScript directly.
          experimental: { viteModuleRunner: false },
          benchmark: { include: ["benchmarks/cpu*.bench.ts"] },
        },
      },
      browserProject("gpu", 1, ["benchmarks/!(cpu*).bench.ts"]),
      browserProject("browser", 2),
      browserProject("visual", 3),
    ],
  },
});
