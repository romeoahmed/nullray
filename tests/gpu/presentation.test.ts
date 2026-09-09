import { expect, test } from "vitest";
import presentation from "../../src/render/shaders/present.wgsl?raw";
import { compileShader, requestDevice } from "../../src/render/device.ts";

test.each([1, 4])("presentation preserves the intended linear signal at peak %i", async (peak) => {
  const device = await requestDevice();
  using resources = new DisposableStack();
  resources.defer(() => device.destroy());
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
  const readback = resources.adopt(
    device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
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
  encoder.copyTextureToBuffer({ texture: output }, { buffer: readback, bytesPerRow: 256 }, [6, 1]);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const encoded = new Float16Array(readback.getMappedRange()).slice(0, 24);
  readback.unmap();
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
});
