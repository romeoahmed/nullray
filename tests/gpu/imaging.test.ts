import { describe, expect } from "vitest";
import presentation from "../../src/gpu/wgsl/passes/present.wgsl?raw";
import { compileShader } from "../../src/gpu/device.ts";
import { createBloom } from "../../src/gpu/bloom.ts";
import { referenceBloom } from "../reference/bloom.ts";
import diagnostic from "../../src/gpu/wgsl/imaging/diagnostic.wgsl?raw";
import { test, computeReadback, readPixels } from "./compute.ts";

describe("Display conversion", () => {
  test.for([1, 4])(
    "presentation preserves the intended linear signal at peak %i",
    async (peak, { device }) => {
      using resources = new DisposableStack();

      const input = resources.adopt(
        device.createTexture({
          size: [6, 1],
          format: "rgba16float",
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        }),
        (texture) => texture.destroy(),
      );
      const output = resources.adopt(
        device.createTexture({
          size: [6, 1],
          format: "rgba16float",
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        }),
        (texture) => texture.destroy(),
      );
      const uniform = resources.adopt(
        device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
        (buffer) => buffer.destroy(),
      );
      const pixels = new Float16Array([
        0, 0, 0, 1, 16, 16, 16, 1, 4, 0, 0, 1, -0.5, 4, 0, 1, 0, 0, 0, 0, 1, 2, 3, 0.5,
      ]);
      device.queue.writeTexture({ texture: input }, pixels, { bytesPerRow: 48 }, [6, 1]);
      // Minus two exposure stops turns input 16 into linear intensity 4.
      device.queue.writeBuffer(uniform, 0, new Float32Array([0.25, peak, 0, 0]));
      const module = await compileShader(device, presentation, "presentation-test");
      const pipeline = await device.createRenderPipelineAsync({
        layout: "auto",
        vertex: { module, entryPoint: "vertex" },
        fragment: { module, entryPoint: "fragment", targets: [{ format: "rgba16float" }] },
      });
      const bindings = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: input.createView() },
          { binding: 1, resource: device.createSampler() },
          { binding: 2, resource: { buffer: uniform } },
          { binding: 3, resource: input.createView() },
        ],
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: output.createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: [0, 0, 0, 1],
          },
        ],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindings);
      pass.draw(3);
      pass.end();
      device.queue.submit([encoder.finish()]);
      const encoded = await readPixels(device, output);
      const linear = Array.from(encoded, (value) =>
        value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4,
      );
      expect(linear.slice(0, 3)).toEqual([0, 0, 0]);
      for (const channel of linear.slice(4, 7)) {
        expect(channel).toBeCloseTo((4 * peak) / (peak + 4), 2);
      }
      if (peak === 4) {
        expect(encoded[4]).toBeGreaterThan(1);
      } else {
        expect(encoded[4]).toBeLessThan(1);
      }
      // sRGB red must become a mixture of P3 primaries, not be relabeled as P3 red.
      const red = linear[8];
      const green = linear[9];
      const blue = linear[10];
      if (red === undefined || green === undefined || blue === undefined) {
        throw new Error("Missing primary-color sample.");
      }
      expect(green / red).toBeCloseTo(0.0403596, 4);
      expect(blue / red).toBeCloseTo(0.0207701, 4);
      // Signed working RGB is transformed before clipping in the destination gamut.
      expect(linear[12]).toBeGreaterThan(0);
      expect(linear[13]).toBeGreaterThan(linear[12] ?? 0);
      // Failure color is added after radiance processing, weighted by missing coverage.
      for (const [pixel, rgb] of [
        [4, [0.65, 0.015, 0.24]],
        [5, [1.325, 2.0075, 3.12]],
      ] as const) {
        const p3 = [
          0.82246197 * rgb[0] + 0.17753803 * rgb[1],
          0.0331942 * rgb[0] + 0.9668058 * rgb[1],
          0.01708263 * rgb[0] + 0.07239744 * rgb[1] + 0.91051993 * rgb[2],
        ].map((value) => value * 0.25);
        const maximum = Math.max(...p3);
        for (const [channel, value] of p3.entries()) {
          expect(linear[pixel * 4 + channel]).toBeCloseTo((value * peak) / (peak + maximum), 3);
        }
        expect(encoded[pixel * 4 + 3]).toBe(1);
      }
      expect(Array.from(encoded).every(Number.isFinite)).toBe(true);
      expect([encoded[3], encoded[7], encoded[11], encoded[15]]).toEqual([1, 1, 1, 1]);
    },
  );
});

describe("Scattered highlights", () => {
  test("bloom preserves constant light and interior point flux across pixel phases and resize", async ({
    device,
  }) => {
    using owned = new DisposableStack();

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
      device.queue.submit([encoder.finish()]);
      return { width: result.width, height: result.height, data: await readPixels(device, result) };
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
});

describe("Optical inspection", () => {
  test("inspection preserves failed coverage, image order, and signed logarithmic frequency", async () => {
    const endpoints = new Float32Array([
      14, 0, 0, -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 1, 0, 0, 0.5, 1, 0, 0, 1e-30, 1, 6, 0, 2, -3,
      6, 0, 0.5, -4, 6, 0, 1, -5,
    ]);
    const result = await computeReadback(
      `${diagnostic}
    @group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
    @group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
    @compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id: vec3u) {
      outputs[2u * id.x] = diagnostic_endpoint(inputs[id.x], 1.0);
      outputs[2u * id.x + 1u] = diagnostic_endpoint(inputs[id.x], 2.0);
    }`,
      endpoints,
      endpoints.length * 2,
      endpoints.length / 4,
    );
    const color = (index: number, mode: number) =>
      Array.from(result.subarray(index * 8 + mode * 4, index * 8 + mode * 4 + 4));
    const red = Array.from(new Float32Array([0.8, 0.04, 0.015, 1]));
    const blue = Array.from(new Float32Array([0.025, 0.25, 0.8, 1]));
    expect(result.every(Number.isFinite)).toBe(true);
    expect(color(0, 0)).toEqual([0, 0, 0, 0]);
    expect(color(0, 1)).toEqual([0, 0, 0, 0]);
    expect(color(1, 0)).toEqual([0, 0, 0, 1]);
    expect(color(2, 0)).toEqual(Array.from(new Float32Array([0.3, 0.3, 0.3, 1])));
    // The sky stores E=1/g; disk endpoints already store g.
    expect(color(3, 0)).toEqual(red);
    expect(color(4, 0)).toEqual(blue);
    expect(color(5, 0)).toEqual(blue);
    expect(color(6, 0)).toEqual(blue);
    expect(color(7, 0)).toEqual(red);
    expect(color(6, 1)).toEqual(Array.from(new Float32Array([0.1, 0.1, 0.1, 1])));
    expect(color(7, 1)).toEqual(Array.from(new Float32Array([0.02, 0.35, 0.6, 1])));
    expect(color(8, 1)).toEqual(Array.from(new Float32Array([0.8, 0.3, 0.02, 1])));
  });
});
