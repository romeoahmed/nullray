import { expect, test } from "vitest";
import { server } from "vitest/browser";
import { createRenderer } from "../src/gpu/renderer.ts";
import { sampleCounts } from "../src/gpu/imaging/sampling.ts";
import type { SamplingMode } from "../src/gpu/imaging/sampling.ts";
import { presets } from "../src/scene/presets.ts";
import { checkDevice } from "../tests/support/gpu.ts";
import { gpuContext } from "./context.ts";

const workloads = presets
  .filter((preset) => ["Classic disk", "Blue accretion flow"].includes(preset.name))
  .flatMap((preset) =>
    (["live", "settle", "photograph", "display"] as const).map((mode) => ({
      name: `${preset.name}-${mode}`,
      view: preset.view,
      mode,
    })),
  );

test.for(workloads)("$name presentation", async ({ name, view, mode }, { bench }) => {
  using owned = new DisposableStack();
  const canvas = new OffscreenCanvas(320, 180);
  const renderer = owned.adopt(await createRenderer(canvas), (value) => value.dispose());
  renderer.resize(320, 180);
  const sampling: SamplingMode = mode === "display" ? "settle" : mode;
  const settings = {
    appearance: view.appearance,
    exposureEV: view.exposureEV,
    whiteBalance: view.whiteBalance,
    bloom: view.bloom,
    hdr: false,
    view: view.diagnostic,
    resolutionScale: 0.5,
    sampling,
    time: 0,
  } as const;
  // Configure the real context before timing so metadata never comes from a second adapter request.
  renderer.render(view.scene, settings);
  await renderer.finished();
  const device = canvas.getContext("webgpu")?.getConfiguration()?.device;
  if (!device) {
    throw new Error("The benchmark renderer did not configure its canvas.");
  }
  const check = checkDevice(device);
  if (mode === "display") {
    for (let i = 1; i < sampleCounts.settle; i++) {
      renderer.render(view.scene, settings);
      // Match the production one-frame submission boundary.
      // oxlint-disable-next-line no-await-in-loop
      await renderer.finished();
    }
  }
  let epoch = 0;
  let completedSamples = 0;
  await bench(
    `${name} / completed sequence wall-clock ms`,
    { writeResult: `test-results/bench/${name}.json` },
    async () => {
      epoch++;
      const count = mode === "display" ? 1 : sampleCounts[sampling];
      for (let i = 0; i < count; i++) {
        completedSamples = renderer.render(view.scene, {
          ...settings,
          // Display edits retain the exact optical history and emission epoch.
          time: mode === "display" ? 0 : epoch,
          exposureEV: mode === "display" ? view.exposureEV - (epoch % 2) * 0.1 : view.exposureEV,
        });
        // oxlint-disable-next-line no-await-in-loop
        await renderer.finished();
      }
    },
  ).run({ iterations: 3, time: 0, warmupIterations: 1, warmupTime: 0 });
  expect(completedSamples).toBe(sampleCounts[sampling]);
  await check();
  const coverage = sampling === "live" ? null : await renderer.coverage();
  if (coverage) {
    expect(coverage.samplesPerPixel).toBe(sampleCounts[sampling]);
    expect(coverage.pixels).toBe(canvas.width * canvas.height);
  }
  await server.commands.writeFile(
    `test-results/bench/${name}-context.json`,
    JSON.stringify(
      {
        ...(await gpuContext(device)),
        measurements: { iterations: 3, warmupIterations: 1, time: 0, warmupTime: 0 },
        name,
        view,
        output: [canvas.width, canvas.height],
        samples: mode === "display" ? 0 : sampleCounts[sampling],
        sampling,
        epochs: mode === "display" ? [0] : Array.from({ length: epoch }, (_, i) => i + 1),
        coverage,
        scope:
          "Reused renderer: complete submitted optical/accumulation/polarimeter/bloom/presentation sequence, awaiting each frame. Excludes initialization, worker/message/rAF pacing, readback and PNG encoding. Display workload reuses a settled image. Different sample counts and raster sizes are not equal-quality speed comparisons.",
      },
      null,
      2,
    ),
  );
});
