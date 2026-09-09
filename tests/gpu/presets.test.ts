import { expect, test } from "vitest";
import { createOptics } from "../../src/render/optics.ts";
import { requestDevice } from "../../src/render/device.ts";
import { createScene, initialScene } from "../../src/model/scene.ts";
import { initialAppearance } from "../../src/model/appearance.ts";
import { presets } from "../../src/model/presets.ts";

test("presets and lower-resolution disk sampling fit the optical work budget", async ({
  annotate,
}) => {
  const device = await requestDevice();
  using owned = new DisposableStack();
  owned.defer(() => device.destroy());
  device.pushErrorScope("validation");
  const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
  const staging = owned.adopt(
    device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
    (value) => value.destroy(),
  );
  const records: {
    name: string;
    boundary: number[];
    derivatives: number[];
    capacity: number;
    derivativeCapacity: number;
  }[] = [];
  const close = createScene({
    ...initialScene,
    observer: { ...initialScene.observer, radius: 18, fieldOfView: 2 * Math.atan(0.6) },
  });
  if (!close.ok) {
    throw new Error(close.error);
  }
  const workloads = presets.flatMap(({ name, view }) => [
    { name: `${name} / native`, view, width: 2560, height: 1440 },
    { name: `${name} / 360p`, view, width: 640, height: 360 },
  ]);
  const closeWorkload = {
    name: "Close charged disk / 360p",
    view: { scene: close.value, appearance: initialAppearance, time: 0 },
    width: 640,
    height: 360,
  };
  for (const { name, view, width, height } of [...workloads, closeWorkload]) {
    const encoder = device.createCommandEncoder();
    const image = optics.encode(encoder, view.scene, width, height, {
      appearance: view.appearance,
      time: view.time,
      antialias: true,
    });
    encoder.copyBufferToBuffer(image.edgeStatistics, 0, staging, 0, 16);
    encoder.copyBufferToBuffer(image.edgeBeamStatistics, 0, staging, 16, 16);
    device.queue.submit([encoder.finish()]);
    // Submissions reuse the optical target and staging buffer.
    // oxlint-disable-next-line no-await-in-loop
    await staging.mapAsync(GPUMapMode.READ);
    const counts = new Uint32Array(staging.getMappedRange()).slice();
    staging.unmap();
    records.push({
      name,
      boundary: Array.from(counts.subarray(0, 4)),
      derivatives: Array.from(counts.subarray(4, 7)),
      capacity: image.edgeCapacity,
      derivativeCapacity: image.edgeBeamCapacity,
    });
  }
  await annotate("Preset work budgets", {
    body: JSON.stringify(records),
    bodyEncoding: "utf-8",
    contentType: "application/json",
  });
  expect(await device.popErrorScope()).toBeNull();
  for (const record of records) {
    expect.soft(record.boundary[0], record.name).toBeLessThanOrEqual(record.capacity);
    expect.soft(record.derivatives[0], record.name).toBeLessThanOrEqual(record.derivativeCapacity);
  }
}, 30000);
