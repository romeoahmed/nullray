import { expect, test } from "vitest";
import { createOptics } from "../../src/render/optics.ts";
import { requestDevice } from "../../src/render/device.ts";
import { createAppearance, initialAppearance } from "../../src/model/appearance.ts";
import { initialScene } from "../../src/model/scene.ts";

test("source animation changes disk light while retaining geometry and sky", async () => {
  const device = await requestDevice();
  using owned = new DisposableStack();
  owned.defer(() => device.destroy());
  device.pushErrorScope("validation");
  const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
  const width = 32;
  const height = 24;
  const count = width * height;
  const colorBytes = count * 8;
  const staging = owned.adopt(
    device.createBuffer({
      size: colorBytes * 3 + 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
    (value) => value.destroy(),
  );
  async function frame(time: number, trace: boolean, appearance = initialAppearance) {
    const encoder = device.createCommandEncoder();
    const image = optics.encode(encoder, initialScene, width, height, { time, trace, appearance });
    encoder.copyTextureToBuffer(
      { texture: image.radiance },
      { buffer: staging, bytesPerRow: width * 8 },
      [width, height],
    );
    encoder.copyBufferToBuffer(image.beamStatistics, 0, staging, colorBytes * 3, 16);
    encoder.copyTextureToBuffer(
      { texture: image.endpoints },
      { buffer: staging, offset: colorBytes, bytesPerRow: width * 16 },
      [width, height],
    );
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const mapped = staging.getMappedRange();
    const color = new Float16Array(mapped, 0, count * 4).slice();
    const endpoints = new Float32Array(mapped, colorBytes, count * 4).slice();
    const beams = new Uint32Array(mapped, colorBytes * 3, 3).slice();
    staging.unmap();
    return { color, endpoints, beams };
  }
  const first = await frame(0, true);
  const later = await frame(100, false);
  const repeated = await frame(100, false);
  expect(later.endpoints).toEqual(first.endpoints);
  expect(repeated.color).toEqual(later.color);
  expect(first.beams[0]).toBeGreaterThan(0);
  expect(later.beams).toEqual(first.beams);
  expect(repeated.beams).toEqual(first.beams);
  let disk = 0;
  let changed = 0;
  let sky = 0;
  for (let i = 0; i < count; i++) {
    const tag = first.endpoints[i * 4 + 3] ?? Number.NaN;
    const before = first.color[i * 4] ?? Number.NaN;
    const after = later.color[i * 4] ?? Number.NaN;
    expect(Number.isFinite(before + after)).toBe(true);
    const resolved = first.color[i * 4 + 3];
    expect([0, 1]).toContain(resolved);
    if (resolved === 0) {
      expect(tag).toBeGreaterThan(0);
      expect(before).toBe(0);
      expect(first.color[i * 4 + 1]).toBe(0);
    }
    if (tag <= -3) {
      disk++;
      if (Math.abs(before - after) > 0.001) {
        changed++;
      }
    } else if (tag > 0) {
      sky++;
      expect(after).toBe(before);
    }
  }
  expect(disk).toBeGreaterThan(20);
  expect(changed).toBeGreaterThan(disk / 2);
  expect(sky).toBeGreaterThan(0);
  const longTime = await frame(240_000_000, false);
  const longTimeStep = await frame(240_000_000.125, false);
  expect(longTimeStep.endpoints).toEqual(first.endpoints);
  expect(longTimeStep.beams).toEqual(first.beams);
  let longTimeChanges = 0;
  for (let index = 0; index < count; index++) {
    const tag = first.endpoints[index * 4 + 3] ?? NaN;
    if (tag <= -3 && longTime.color[index * 4] !== longTimeStep.color[index * 4]) {
      longTimeChanges++;
    } else if (tag > 0) {
      expect(longTimeStep.color.subarray(index * 4, index * 4 + 4)).toEqual(
        longTime.color.subarray(index * 4, index * 4 + 4),
      );
    }
  }
  expect(longTimeChanges).toBeGreaterThan(0);
  const bright = createAppearance({ ...initialAppearance, skyBrightness: 2 });
  const flat = createAppearance({ ...initialAppearance, diskStructure: 0 });
  const hot = createAppearance({ ...initialAppearance, diskStructure: 0, diskTemperature: 10000 });
  const dark = createAppearance({ ...initialAppearance, skyBrightness: 0 });
  if (!bright.ok || !flat.ok || !dark.ok || !hot.ok) {
    throw new Error("Invalid source fixture.");
  }
  const brighterSky = await frame(0, false, bright.value);
  const steady = await frame(0, false, flat.value);
  const steadyLater = await frame(100, false, flat.value);
  const darkSky = await frame(0, false, dark.value);
  const hotter = await frame(100, false, hot.value);
  let heated = 0;
  expect(brighterSky.endpoints).toEqual(first.endpoints);
  expect(brighterSky.beams).toEqual(first.beams);
  expect(steadyLater.color).toEqual(steady.color);
  for (let i = 0; i < count; i++) {
    const tag = first.endpoints[i * 4 + 3] ?? Number.NaN;
    if (tag > 0) {
      expect(darkSky.color[i * 4 + 3]).toBe(1);
      expect(hotter.color.subarray(i * 4, i * 4 + 4)).toEqual(
        steadyLater.color.subarray(i * 4, i * 4 + 4),
      );
      for (let channel = 0; channel < 3; channel++) {
        expect(darkSky.color[i * 4 + channel]).toBe(0);
        if (first.color[i * 4 + 3] === 1) {
          const expected = 2 * (first.color[i * 4 + channel] ?? Number.NaN);
          expect(brighterSky.color[i * 4 + channel]).toBeCloseTo(expected, 4);
        }
      }
    } else if (tag <= -3) {
      if ((hotter.color[i * 4] ?? 0) > (steadyLater.color[i * 4] ?? 0)) {
        heated++;
      }
      for (let channel = 0; channel < 4; channel++) {
        expect(brighterSky.color[i * 4 + channel]).toBe(first.color[i * 4 + channel]);
      }
    }
  }

  expect(heated).toBeGreaterThan(20);
  expect(await device.popErrorScope()).toBeNull();
});
