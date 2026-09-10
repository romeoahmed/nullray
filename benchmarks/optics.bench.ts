import { test } from "vitest";
import { server } from "vitest/browser";
import { createOptics } from "../src/gpu/optics/engine.ts";
import type { OpticalImage } from "../src/gpu/optics/engine.ts";
import { requestDevice } from "../src/gpu/device.ts";
import { createScene, initialScene } from "../src/scene/scene.ts";
import type { SceneInput } from "../src/scene/scene.ts";
import { initialAppearance } from "../src/scene/appearance.ts";
import type { SourceAppearance } from "../src/scene/appearance.ts";
import { presets } from "../src/scene/presets.ts";
import { sourceFingerprint, benchmarkFingerprint } from "./source.ts";
import { captureRadiance, imageCoverage } from "./capture.ts";
import { initialJet } from "../src/scene/jet.ts";
import { initialPlasma } from "../src/scene/plasma.ts";

const blueFlow = presets.find((preset) => preset.name === "Blue accretion flow");
if (!blueFlow) {
  throw new Error("Missing blue-flow benchmark view.");
}
const workloads: readonly {
  readonly name: string;
  readonly input: SceneInput;
  readonly appearance?: SourceAppearance;
}[] = [
  { name: "disk", input: initialScene },
  { name: "jet", input: { ...initialScene, jet: initialJet } },
  { name: "blue-flow", input: blueFlow.view.scene, appearance: blueFlow.view.appearance },
  { name: "plasma", input: { ...initialScene, plasma: initialPlasma } },
  {
    name: "interior",
    input: {
      ...initialScene,
      observer: { ...initialScene.observer, radius: 1, motion: { kind: "regular" as const } },
    },
  },
  {
    name: "naked",
    input: { ...initialScene, space: { spin: 1.2, charge: 0.3 }, disk: { inner: 16, outer: 64 } },
  },
];

test.for(workloads)(
  "$name optical frame",
  async ({ name, input, appearance = initialAppearance }, { bench, annotate }) => {
    const scene = createScene(input);
    if (!scene.ok) {
      throw new Error(scene.error);
    }
    const device = await requestDevice();
    using owned = new DisposableStack();
    owned.defer(() => device.destroy());
    const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
    const width = 320,
      height = 180;
    let image: OpticalImage | undefined;
    await bench(
      `${name} / submitted frame wall-clock ms`,
      { writeResult: `test-results/bench/${name}.json` },
      async () => {
        const encoder = device.createCommandEncoder();
        image = optics.encode(encoder, scene.value, width, height, {
          appearance,
          time: 0,
          view: "image",
          jitter: [0, 0],
        });
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
      },
    ).run({ iterations: 8, time: 0, warmupIterations: 1, warmupTime: 0 });
    if (!image) {
      throw new Error("No benchmark image was rendered.");
    }
    const pixels = await captureRadiance(device, image.radiance);
    const body = JSON.stringify(
      {
        schema: 1,
        name,
        width,
        height,
        samplesPerPixel: 1,
        appearance,
        scene: {
          space: scene.value.space,
          observer: scene.value.observer,
          camera: scene.value.camera,
          disk: scene.value.disk,
          jet: scene.value.jet,
          plasma: scene.value.plasma,
        },
        browser: navigator.userAgent,
        adapter: {
          vendor: device.adapterInfo.vendor,
          architecture: device.adapterInfo.architecture,
          device: device.adapterInfo.device,
        },
        sourceSHA256: new Uint8Array(await sourceFingerprint()).toHex(),
        harnessSHA256: new Uint8Array(await benchmarkFingerprint()).toHex(),
        coverage: imageCoverage(pixels),
        scope:
          "One native optical sample on reused resources; initialization, readback, bloom and presentation are outside timing. Results are comparative measurements, not performance gates.",
      },
      null,
      2,
    );
    await server.commands.writeFile(`test-results/bench/${name}-context.json`, body);
    await annotate("Frame context", {
      body,
      bodyEncoding: "utf-8",
      contentType: "application/json",
    });
  },
);
