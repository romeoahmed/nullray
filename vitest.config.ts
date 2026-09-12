import { playwright } from "@vitest/browser-playwright";
import { defineConfig, defineProject } from "vitest/config";

export default defineConfig({
  test: {
    // Inline projects inherit this opt-out; only CPU and GPU own benchmarks.
    benchmark: { include: [] },
    projects: [
      {
        test: {
          name: "cpu",
          include: ["tests/cpu/**/*.test.ts"],
          // Node runs erasable TypeScript directly, without module-runner overhead in benchmarks.
          experimental: { viteModuleRunner: false },
          benchmark: { include: ["benchmarks/cpu*.bench.ts"] },
        },
      },
      ...["gpu", "browser", "visual"].map((name, index) =>
        defineProject({
          test: {
            name,
            include: [`tests/${name}/**/*.test.ts`],
            // These projects share the physical GPU, even when selected together.
            sequence: { groupOrder: index + 1 },
            fileParallelism: false,
            ...(name === "gpu" && {
              benchmark: {
                include: ["benchmarks/!(cpu*).bench.ts"],
                retainSamples: true,
              },
            }),
            // Construct fresh instances: Vitest assigns names during project expansion.
            browser: {
              enabled: true,
              headless: true,
              provider: playwright({ launchOptions: { channel: "chromium" } }),
              instances: [{ browser: "chromium" }],
            },
          },
        }),
      ),
    ],
  },
});
