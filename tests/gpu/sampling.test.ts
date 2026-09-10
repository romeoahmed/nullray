import { test, readPixels } from "./compute.ts";
import { describe, expect } from "vitest";
import { createAccumulation } from "../../src/gpu/imaging/accumulation.ts";
import { createCoverageReader } from "../../src/gpu/imaging/coverage.ts";

describe("Photographic history", () => {
  test("photographic history averages HDR radiance, preserves unresolved coverage, and resets", async ({
    device,
  }) => {
    using owned = new DisposableStack();

    device.pushErrorScope("validation");
    const accumulation = owned.adopt(await createAccumulation(device), (value) => value.dispose());
    const texture = (format: GPUTextureFormat) =>
      owned.adopt(
        device.createTexture({
          size: [2, 1],
          format,
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        }),
        (value) => value.destroy(),
      );
    const radiance = texture("rgba16float");
    async function frame(index: number, color: number, resolvedWeight = 1) {
      device.queue.writeTexture(
        { texture: radiance },
        new Float16Array([color, 2 * color, 3 * color, 1, color, color, color, resolvedWeight]),
        { bytesPerRow: 16 },
        [2, 1],
      );
      const encoder = device.createCommandEncoder();
      const output = accumulation.encode(encoder, radiance, index);
      device.queue.submit([encoder.finish()]);
      return readPixels(device, output);
    }
    await frame(0, 8, 0);
    const average = await frame(1, 16);
    expect(Array.from(average.slice(0, 4))).toEqual([12, 24, 36, 1]);
    expect(Array.from(average.slice(4))).toEqual([8, 8, 8, 0.5]);
    expect(await accumulation.coverage()).toEqual({
      pixels: 2,
      samplesPerPixel: 2,
      unresolvedPixels: 1,
      unresolvedSamples: 1,
    });
    const reset = await frame(0, 2);
    expect(Array.from(reset)).toEqual([2, 4, 6, 1, 2, 2, 2, 1]);
    expect((await accumulation.coverage()).unresolvedSamples).toBe(0);
    await frame(0, 8, 15 / 16);
    const partial = await frame(1, 16);
    expect(Array.from(partial.slice(4))).toEqual([12, 12, 12, 31 / 32]);
    expect(await accumulation.coverage()).toEqual({
      pixels: 2,
      samplesPerPixel: 2,
      unresolvedPixels: 1,
      unresolvedSamples: 1 / 16,
    });
    for (const index of [-1, 0.5, 3, 64, NaN]) {
      expect(() => accumulation.encode(device.createCommandEncoder(), radiance, index)).toThrow(
        RangeError,
      );
    }
    for (let index = 0; index < 64; index++) {
      // Sequential uploads reuse the sample textures and accumulated history.
      // oxlint-disable-next-line no-await-in-loop
      await frame(index, 2, index % 3 === 0 || index % 7 === 0 ? 0 : 1);
    }
    expect(await accumulation.coverage()).toEqual({
      pixels: 2,
      samplesPerPixel: 64,
      unresolvedPixels: 1,
      unresolvedSamples: 28,
    });
    expect(await device.popErrorScope()).toBeNull();
  });
});

describe("Unresolved coverage", () => {
  test("coverage reduction counts partial workgroups and all 64 sample weights", async ({
    device,
  }) => {
    using owned = new DisposableStack();

    const readCoverage = await createCoverageReader(device);
    const width = 17;
    const height = 19;
    const data = new Float32Array(width * height * 4);
    let unresolvedPixels = 0;
    let unresolvedSamples = 0;
    for (let pixel = 0; pixel < width * height; pixel++) {
      const failed = Math.min(64, (pixel % 65) + (pixel % 16) / 16);
      data[pixel * 4 + 3] = failed / 64;
      unresolvedPixels += Number(failed > 0);
      unresolvedSamples += failed;
    }
    const history = owned.adopt(
      device.createTexture({
        size: [width, height],
        format: "rgba32float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      }),
      (value) => value.destroy(),
    );
    device.queue.writeTexture({ texture: history }, data, { bytesPerRow: width * 16 }, [
      width,
      height,
    ]);
    expect(await readCoverage(history, 64)).toEqual({
      pixels: width * height,
      samplesPerPixel: 64,
      unresolvedPixels,
      unresolvedSamples,
    });
  });
});
