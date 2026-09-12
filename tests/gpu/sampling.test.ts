import { test, readPixels } from "./compute.ts";
import { describe, expect } from "vitest";
import * as fc from "fast-check";
import { createAccumulation } from "../../src/gpu/imaging/accumulation.ts";
import { createCoverageReader } from "../../src/gpu/imaging/coverage.ts";

describe("Photographic history", () => {
  test("photographic history averages HDR radiance, preserves unresolved coverage, and resets", async ({
    device,
  }) => {
    using owned = new DisposableStack();

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

test("generated HDR histories match direct sums without renormalizing missing samples", async ({
  device,
}) => {
  using owned = new DisposableStack();
  const history = owned.adopt(await createAccumulation(device), (value) => value.dispose());
  const input = owned.adopt(
    device.createTexture({
      size: [1, 1],
      format: "rgba16float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    }),
    (value) => value.destroy(),
  );
  const sample = fc.tuple(fc.integer({ min: -256, max: 1024 }), fc.integer({ min: 0, max: 16 }));
  await fc.assert(
    fc.asyncProperty(fc.array(sample, { minLength: 1, maxLength: 64 }), async (samples) => {
      let sum = 0;
      let missing = 0;
      let output: GPUTexture | undefined;
      for (const [index, [intensity, coverage]] of samples.entries()) {
        // Dyadic inputs are exact in f16; compare the incremental GPU mean with a direct binary64 sum.
        device.queue.writeTexture(
          { texture: input },
          new Float16Array([intensity, intensity / 2, -intensity / 4, coverage / 16]),
          { bytesPerRow: 8 },
          [1, 1],
        );
        const encoder = device.createCommandEncoder();
        output = history.encode(encoder, input, index);
        device.queue.submit([encoder.finish()]);
        sum += coverage === 0 ? 0 : intensity;
        missing += 1 - coverage / 16;
      }
      if (!output) {
        throw new Error("Missing generated history.");
      }
      const mean = sum / samples.length;
      const actual = await readPixels(device, output);
      for (const [channel, expected] of [
        mean,
        mean / 2,
        -mean / 4,
        1 - missing / samples.length,
      ].entries()) {
        // f32 mean arithmetic and one f16 output rounding; absolute term covers cancellation near zero.
        expect(Math.abs((actual[channel] ?? NaN) - expected)).toBeLessThanOrEqual(
          0.001 * Math.abs(expected) + 1e-4,
        );
      }
      const coverage = await history.coverage();
      expect(coverage.pixels).toBe(1);
      expect(coverage.samplesPerPixel).toBe(samples.length);
      expect(coverage.unresolvedPixels).toBe(Number(missing > 0));
      expect(coverage.unresolvedSamples).toBeCloseTo(missing, 4);
    }),
    { numRuns: 16, examples: [[Array.from({ length: 64 }, (_, i) => [i * 16, i % 17])]] },
  );
});
