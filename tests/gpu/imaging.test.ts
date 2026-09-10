import { describe, expect } from "vitest";
import presentation from "../../src/gpu/wgsl/passes/present.wgsl?raw";
import stars from "../../src/gpu/wgsl/passes/stars.wgsl?raw";
import stellar from "../../src/gpu/wgsl/sources/stars.wgsl?raw";
import frameSource from "../../src/gpu/wgsl/imaging/frame.wgsl?raw";
import radiation from "../../src/gpu/wgsl/imaging/radiation.wgsl?raw";
import { compileShader } from "../../src/gpu/device.ts";
import { createBloom } from "../../src/gpu/imaging/bloom.ts";
import { referenceBloom } from "../reference/bloom.ts";
import { test, readPixels, computeReadback } from "./compute.ts";

test("stellar footprints preserve smooth lensing and exclude neighboring image branches", async () => {
  const cases = [
    [-0.02, 0.02, 0, 0, 0, 0.02],
    [-0.02, 0.12, 0, 1, 0, 0.02],
    [-0.02, 0.5, 0, 0, 0, 0.02],
    [-0.02, 0.02, 1, 1, 0, 0],
    [-0.02, 0.1, 0, 0, 1, 0.02],
    [-0.01, 0.08, 0, 0, 0, 0.01],
  ] as const;
  const endpoints = new Float32Array(cases.length * 12);
  const transmissions = new Float32Array(cases.length * 12);
  const domains = new Int32Array(cases.length * 12);
  for (const [row, [left, right, leftOrder, rightOrder, rightDomain]] of cases.entries()) {
    for (const [column, x] of [left, 0, right].entries()) {
      const offset = row * 12 + column * 4;
      endpoints.set([x, 0, Math.sqrt(1 - x * x), 1], offset);
      transmissions.set(
        [1, column === 0 ? leftOrder : column === 2 ? rightOrder : 0, 0, 0],
        offset,
      );
    }
    domains[row * 12 + 8] = rightDomain;
  }
  const result = await computeReadback(
    `${frameSource}\n${radiation}\n${stellar}\n${stars}
    @group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
    @compute @workgroup_size(1) fn check(@builtin(global_invocation_id) id: vec3u) {
      let center = vec4f(0, 0, 1, 1);
      output[id.x] = vec4f(arrival_difference(vec2i(1, i32(id.x)), vec2i(1, 0), center, vec4i(0), optical_frame.sampling.x), 1);
    }`,
    new Float32Array(72),
    cases.length * 4,
    cases.length,
    [],
    "check",
    (device) => {
      const textures = (
        [
          ["rgba32float", endpoints],
          ["rgba32float", transmissions],
          ["rgba32sint", domains],
        ] as const
      ).map(([format, data]) => {
        const texture = device.createTexture({
          size: [3, cases.length],
          format,
          usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
        });
        device.queue.writeTexture({ texture }, data, { bytesPerRow: 48 }, [3, cases.length]);
        return texture;
      });
      return {
        entries: textures.map((texture, index) => ({
          binding: 15 + index,
          resource: texture.createView(),
        })),
        dispose: () => textures.forEach((texture) => texture.destroy()),
      };
    },
  );
  expect(result.every(Number.isFinite)).toBe(true);
  for (const [row, values] of cases.entries()) {
    expect(result[row * 4]).toBeCloseTo(values[5], 6);
  }
});

describe("Display conversion", () => {
  test("photographic highlights retain color direction and approach a neutral finite peak", async () => {
    const colors = [
      [0, 0, 0],
      [0.18, 0.18, 0.18],
      [0.7, 0.3, 0.1],
      [2, 0.5, 0.1],
      [100, 25, 5],
      [1000, 1000, 1000],
    ];
    const cases = [1, 4].flatMap((peak) =>
      colors.map((color) => [...color.map((c) => c * peak), peak]),
    );
    const result = await computeReadback(
      `${presentation}
      @group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
      @group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
      @compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id: vec3u) {
        let p = inputs[id.x];
        outputs[id.x] = vec4f(photographic_shoulder(p.rgb, p.w), 1);
      }`,
      new Float32Array(cases.flat()),
      cases.length * 4,
      cases.length,
      [],
      "probe",
    );
    for (const [index, input] of cases.entries()) {
      const peak = input[3] ?? NaN;
      const [r = NaN, g = NaN, b = NaN] = result.subarray(index * 4, index * 4 + 3);
      expect(
        [r, g, b].every((value) => Number.isFinite(value) && value >= 0 && value <= peak),
      ).toBe(true);
      if (index % colors.length < 3) {
        for (const [channel, value] of [r, g, b].entries()) {
          expect(value).toBeCloseTo(input[channel] ?? NaN, 6);
        }
      }
      if (index % colors.length === 3 || index % colors.length === 4) {
        expect((r - g) / (g - b)).toBeCloseTo(3.75, 4);
      }
      if (index % colors.length === 4) {
        expect((r - b) / r).toBeLessThan(0.12);
      }
      if (index % colors.length === 5) {
        expect(r).toBeCloseTo(peak, 2);
        expect(r).toBe(g);
        expect(g).toBe(b);
      }
    }
  });

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
          size: 32,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
        (buffer) => buffer.destroy(),
      );
      const pixels = new Float16Array([
        0, 0, 0, 1, 16, 16, 16, 1, 4, 0, 0, 1, -0.5, 4, 0, 1, 0, 0, 0, 0, 1, 2, 3, 0.5,
      ]);
      device.queue.writeTexture({ texture: input }, pixels, { bytesPerRow: 48 }, [6, 1]);
      // Minus two exposure stops turns input 16 into linear intensity 4.
      device.queue.writeBuffer(uniform, 0, new Float32Array([0.25, peak, 0, 0, 1, 1, 1, 0]));
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
      const expectedRed = (0.82246197 * peak) / (peak + 0.82246197);
      // Half-float encoding followed by gamma decoding amplifies the storage quantization error.
      expect(Math.abs(red / expectedRed - 1)).toBeLessThan(0.002);
      expect(green).toBeCloseTo((0.0331942 * peak) / (peak + 0.0331942), 4);
      expect(blue).toBeCloseTo((0.01708263 * peak) / (peak + 0.01708263), 4);
      // Signed working RGB is transformed before clipping in the destination gamut.
      expect(linear[12]).toBeGreaterThan(0);
      expect(linear[13]).toBeGreaterThan(linear[12] ?? 0);
      // A photograph contains only resolved radiance; coverage does not inject diagnostic light.
      for (const [pixel, rgb] of [
        [4, [0, 0, 0]],
        [5, [1, 2, 3]],
      ] as const) {
        const p3 = [
          0.82246197 * rgb[0] + 0.17753803 * rgb[1],
          0.0331942 * rgb[0] + 0.9668058 * rgb[1],
          0.01708263 * rgb[0] + 0.07239744 * rgb[1] + 0.91051993 * rgb[2],
        ].map((value) => value * 0.25);
        for (const [channel, value] of p3.entries()) {
          const expected = (value * peak) / (peak + value);
          expect(Math.abs((linear[pixel * 4 + channel] ?? NaN) - expected)).toBeLessThan(
            0.002 * expected + 1e-6,
          );
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
