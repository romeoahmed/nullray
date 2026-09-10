import { requestDevice, requiredFeatures, requiredLimits } from "../src/gpu/device.ts";
import { expect, test } from "vitest";
import { server } from "vitest/browser";
import { createOptics } from "../src/gpu/optics/engine.ts";
import { sourceFingerprint, benchmarkFingerprint } from "./source.ts";

test("optical initialization on an existing device", async ({ bench, annotate }) => {
  const device = await requestDevice();
  using owned = new DisposableStack();
  owned.defer(() => device.destroy());
  device.pushErrorScope("validation");
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
  expect(await device.popErrorScope()).toBeNull();
  const digest = await sourceFingerprint();
  const body = JSON.stringify(
    {
      schema: 1,
      requiredFeatures: [...requiredFeatures],
      requiredLimits,
      enabledFeatures: [...device.features],
      browser: navigator.userAgent,
      adapter: {
        vendor: device.adapterInfo.vendor,
        architecture: device.adapterInfo.architecture,
        device: device.adapterInfo.device,
        description: device.adapterInfo.description,
      },
      sourceSHA256: new Uint8Array(digest).toHex(),
      harnessSHA256: new Uint8Array(await benchmarkFingerprint()).toHex(),
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
