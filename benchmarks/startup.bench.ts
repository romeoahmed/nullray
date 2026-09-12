import { requiredFeatures, requiredLimits } from "../src/gpu/device.ts";
import { test } from "../tests/support/gpu.ts";
import { server } from "vitest/browser";
import { createOptics } from "../src/gpu/optics/engine.ts";
import { gpuContext } from "./context.ts";

test("optical initialization on an existing device", async ({ device, bench, annotate }) => {
  await bench(
    "optical initialization / wall-clock ms",
    { writeResult: "test-results/bench/gpu-startup.json" },
    async () => {
      const optics = await createOptics(device);
      try {
        await device.queue.onSubmittedWorkDone();
      } finally {
        optics.dispose();
      }
    },
  ).run({ iterations: 5, time: 0, warmupIterations: 1, warmupTime: 0 });
  const body = JSON.stringify(
    {
      ...(await gpuContext(device)),
      sampling: { iterations: 5, warmupIterations: 1, time: 0, warmupTime: 0 },
      requiredFeatures: [...requiredFeatures],
      requiredLimits,
      scope:
        "Existing device; module diagnostics, pipeline creation, catalogue decoding and tree construction, spectral tables, GPU noise/cube generation, uploads, completion and disposal. No optical frame or canvas presentation.",
      cache:
        "One warmup followed by five samples on the same device; browser/driver shader caches are not cleared. This is not a cold-launch measurement.",
    },
    null,
    2,
  );
  await server.commands.writeFile("test-results/bench/gpu-startup-context.json", body);
  await annotate("Optical initialization context", {
    body,
    bodyEncoding: "utf-8",
    contentType: "application/json",
  });
});
