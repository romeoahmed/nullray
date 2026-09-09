import { expect, test } from "vitest";
import { createBloom } from "../../src/render/bloom.ts";
import { requestDevice } from "../../src/render/device.ts";

import { referenceBloom } from "../reference/bloom.ts";

test("bloom preserves constant light and interior point flux across pixel phases and resize", async () => {
  const device = await requestDevice();
  using owned = new DisposableStack();
  owned.defer(() => device.destroy());
  device.pushErrorScope("validation");
  const bloom = owned.adopt(await createBloom(device), (value) => value.dispose());
  async function filter(width: number, height: number, pixels: Float16Array) {
    using frame = new DisposableStack();
    const source = frame.adopt(
      device.createTexture({
        size: [width, height],
        format: "rgba16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      }),
      (value) => value.destroy(),
    );
    device.queue.writeTexture({ texture: source }, pixels, { bytesPerRow: width * 8 }, [
      width,
      height,
    ]);
    const encoder = device.createCommandEncoder();
    const result = bloom.encode(encoder, source);
    const bytesPerRow = Math.ceil((result.width * 8) / 256) * 256;
    const staging = frame.adopt(
      device.createBuffer({
        size: bytesPerRow * result.height,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
      (value) => value.destroy(),
    );
    encoder.copyTextureToBuffer({ texture: result }, { buffer: staging, bytesPerRow }, [
      result.width,
      result.height,
    ]);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const data = new Float16Array(staging.getMappedRange());
    const output = new Float16Array(result.width * result.height * 4);
    for (let y = 0; y < result.height; y++) {
      output.set(
        data.subarray((y * bytesPerRow) / 2, (y * bytesPerRow) / 2 + result.width * 4),
        y * result.width * 4,
      );
    }
    staging.unmap();
    return { width: result.width, height: result.height, data: output };
  }
  for (const [width, height] of [
    [31, 17],
    [1, 1],
    [64, 32],
  ] as const) {
    const pixels = new Float16Array(width * height * 4);
    for (let i = 0; i < pixels.length; i += 4) {
      // Missing coverage is excluded from the filter; only its resolved RGB is scattered.
      pixels.set([0.25, 0.5, 1, 0.5], i);
    }
    // oxlint-disable-next-line no-await-in-loop
    const result = await filter(width, height, pixels);
    for (let i = 0; i < result.data.length; i += 4) {
      expect(Array.from(result.data.subarray(i, i + 4))).toEqual([0.25, 0.5, 1, 1]);
    }
  }
  for (const phase of [0, 1]) {
    const pixels = new Float16Array(512 * 512 * 4);
    pixels.set([1024, 512, 256, 1], (256 * 512 + 256 + phase) * 4);
    // oxlint-disable-next-line no-await-in-loop
    const result = await filter(512, 512, pixels);
    let flux = 0;
    let centroid = 0;
    let halo = 0;
    for (let y = 0; y < result.height; y++) {
      for (let x = 0; x < result.width; x++) {
        const value = result.data[(y * result.width + x) * 4] ?? NaN;
        expect(Number.isFinite(value) && value >= 0).toBe(true);
        flux += value * 4;
        centroid += value * 4 * (2 * x + 1);
        if (Math.abs(2 * x + 1 - 256.5 - phase) > 8) {
          halo += value;
        }
      }
    }
    expect(Math.abs(flux / 1024 - 1)).toBeLessThan(0.005);
    expect(Math.abs(centroid / flux - (256.5 + phase))).toBeLessThan(0.05);
    expect(halo).toBeGreaterThan(0);
  }
  // Nonconstant inputs exercise fractional texel phases, clamped edges, and thin targets.
  for (const [width, height] of [
    [31, 17],
    [17, 1],
    [1, 19],
  ] as const) {
    const pixels = Float16Array.from({ length: width * height * 4 }, (_, index) =>
      index % 4 === 3 ? 1 : ((index * 37) % 101) / 8,
    );
    const expected = referenceBloom(width, height, pixels);
    // oxlint-disable-next-line no-await-in-loop
    const actual = await filter(width, height, pixels);
    for (const [index, value] of actual.data.entries()) {
      // Binary16 targets and hardware texture interpolation introduce bounded roundoff.
      expect(Math.abs(value - (expected.data[index] ?? NaN))).toBeLessThan(0.025);
    }
  }
  expect(await device.popErrorScope()).toBeNull();
});
