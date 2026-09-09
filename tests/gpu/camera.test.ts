import { expect, test } from "vitest";
import { createOptics } from "../../src/render/optics.ts";
import { requestDevice } from "../../src/render/device.ts";
import { createScene, initialScene } from "../../src/model/scene.ts";
import type { Scene } from "../../src/model/scene.ts";
import { normalize } from "../../src/physics/vector.ts";
import type { Vec3 } from "../../src/physics/vector.ts";
import { traceDirections } from "./visibility-probe.ts";

test("rotated raster rays retain event order and outward-looking endpoint geometry", async () => {
  const device = await requestDevice();
  using owned = new DisposableStack();
  owned.defer(() => device.destroy());
  device.pushErrorScope("validation");
  const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
  const staging = owned.adopt(
    device.createBuffer({
      size: 32 * 32 * 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    }),
    (value) => value.destroy(),
  );
  async function raster(scene: Scene) {
    const encoder = device.createCommandEncoder();
    const image = optics.encode(encoder, scene, 32, 32);
    encoder.copyTextureToBuffer(
      { texture: image.endpoints },
      { buffer: staging, bytesPerRow: 32 * 16 },
      [32, 32],
    );
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();
    return result;
  }
  const rotated = createScene({ ...initialScene, camera: { forward: [-1, 0, 0], up: [0, 1, 0] } });
  const outward = createScene({ ...initialScene, camera: { forward: [1, 0, 0], up: [0, -1, 0] } });
  if (!rotated.ok || !outward.ok) {
    throw new Error("Invalid test cameras.");
  }
  const original = await raster(initialScene);
  const rolled = await raster(rotated.value);
  for (let pixel = 0; pixel < 1024; pixel++) {
    expect(rolled.subarray(pixel * 4, pixel * 4 + 4)).toEqual(
      original.subarray((1023 - pixel) * 4, (1024 - pixel) * 4),
    );
  }
  const result = await raster(outward.value);
  const rays: Vec3[] = [];
  const scale = Math.fround(2 * Math.tan(initialScene.observer.fieldOfView / 2));
  for (let y = 0; y < 32; y++) {
    for (let x = 0; x < 32; x++) {
      rays.push(
        normalize([
          1,
          Math.fround(((y - 15.5) / 32) * scale),
          -Math.fround(((x - 15.5) / 32) * scale),
        ]),
      );
    }
  }
  const expected = await traceDirections(rays, initialScene.observer.inclination, initialScene);
  let escaped = 0;
  for (let pixel = 0; pixel < 1024; pixel++) {
    const offset = pixel * 4;
    expect(result[offset + 3]).toBe(expected[offset + 3]);
    if ((result[offset + 3] ?? 0) > 0) {
      escaped++;
      expect(result[offset]).toBeCloseTo(expected[offset] ?? NaN, 4);
      expect(result[offset + 1]).toBeCloseTo(expected[offset + 1] ?? NaN, 4);
      expect(result[offset + 2]).toBeCloseTo(expected[offset + 2] ?? NaN, 6);
    }
  }
  expect(escaped).toBeGreaterThan(900);
  expect(await device.popErrorScope()).toBeNull();
});
