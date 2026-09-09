import { expect, test } from "vitest";
import { opticalSources } from "../../src/render/optics.ts";
import { createStarTree } from "../../src/physics/stars.ts";
import type { Star } from "../../src/physics/sky.ts";
import { compileShader, requestDevice } from "../../src/render/device.ts";
import { createSkyTexture } from "../../src/render/sky.ts";
import { blackbodyXYZ, createBlackbodyTable, xyzToLinearRGB } from "../../src/physics/radiation.ts";
import {
  createSkyMap,
  cubeDirection,
  cubeTexelSolidAngle,
  skyTemperatures,
  skyCoefficientScale,
} from "../../src/physics/sky.ts";
import type { SkyLevel } from "../../src/physics/sky.ts";
import { normalize } from "../../src/physics/vector.ts";

/** Identity sky projection over six separate cube-face tiles, with a common endpoint energy. */
async function resolveCube(
  levels: readonly SkyLevel[],
  size: number,
  energy: number,
  stars: readonly Star[] = [],
) {
  const device = await requestDevice();
  using owned = new DisposableStack();
  owned.defer(() => device.destroy());
  const width = size * 6;
  const texture = (descriptor: GPUTextureDescriptor) =>
    owned.adopt(device.createTexture(descriptor), (value) => value.destroy());
  const buffer = (byteLength: number, usage: GPUBufferUsageFlags) =>
    owned.adopt(device.createBuffer({ size: byteLength, usage }), (value) => value.destroy());
  const sky = owned.adopt(createSkyTexture(device, levels), (value) => value.destroy());
  const map = texture({
    size: [width, size],
    format: "rgba32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const output = texture({
    size: [width, size],
    format: "rgba16float",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const emissionTime = texture({
    size: [width, size],
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING,
  });
  const beamIndex = texture({
    size: [width, size],
    format: "r32uint",
    usage: GPUTextureUsage.TEXTURE_BINDING,
  });
  const repairedBeams = buffer(32, GPUBufferUsage.STORAGE);
  const data = new Float32Array(width * size * 4);
  for (let face = 0; face < 6; face++) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const direction = cubeDirection(
          face,
          (2 * (x + 0.5)) / size - 1,
          (2 * (y + 0.5)) / size - 1,
        );
        data.set(
          [direction[2], Math.atan2(direction[1], direction[0]), energy, face + 1],
          (y * width + face * size + x) * 4,
        );
      }
    }
  }
  device.queue.writeTexture({ texture: map }, data, { bytesPerRow: width * 16 }, [width, size]);
  const table = createBlackbodyTable();
  const blackbody = buffer(table.data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(blackbody, 0, table.data);
  const module = await compileShader(device, opticalSources.sky, "spectral-sky-test");
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "resolve_sky" },
  });
  const radiationUniform = buffer(48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const diskProfile = buffer(8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(diskProfile, 0, new Float32Array([1, 1]));
  device.queue.writeBuffer(
    radiationUniform,
    0,
    new Float32Array([0.7, 0.2, 3.5, 0, 7000, 0.65, 1, 0]),
  );
  const starData = createStarTree(stars);
  const starTree = buffer(starData.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(starTree, 0, starData);
  const bindings = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: map.createView() },
      { binding: 1, resource: output.createView() },
      { binding: 2, resource: { buffer: blackbody } },
      { binding: 3, resource: sky.createView({ dimension: "cube" }) },
      { binding: 5, resource: { buffer: radiationUniform } },
      { binding: 6, resource: { buffer: diskProfile } },
      { binding: 7, resource: emissionTime.createView() },
      { binding: 8, resource: { buffer: starTree } },
      { binding: 9, resource: beamIndex.createView() },
      { binding: 10, resource: { buffer: repairedBeams } },
      { binding: 11, resource: beamIndex.createView() },
      { binding: 15, resource: map.createView() },
      {
        binding: 4,
        resource: device.createSampler({
          minFilter: "linear",
          magFilter: "linear",
          mipmapFilter: "linear",
          maxAnisotropy: 4,
        }),
      },
    ],
  });
  const bytesPerRow = Math.ceil((width * 8) / 256) * 256;
  const readback = buffer(bytesPerRow * size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindings);
  pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(size / 8));
  pass.end();
  encoder.copyTextureToBuffer({ texture: output }, { buffer: readback, bytesPerRow }, [
    width,
    size,
  ]);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const result = new Float16Array(readback.getMappedRange()).slice();
  readback.unmap();
  return { data: result, rowElements: bytesPerRow / 2 };
}

test("constant spectral sky stays uniform through cube filtering and frequency transfer", async () => {
  const coefficients = [1 / 64, 1 / 32, 1 / 16];
  const levels = [8, 4, 2, 1].map((size) => {
    const data = new Float16Array(size * size * 6 * 4);
    for (let pixel = 0; pixel < size * size * 6; pixel++) {
      data.set(
        coefficients.map((value) => value * skyCoefficientScale),
        pixel * 4,
      );
    }
    return { size, data };
  });
  const energy = Math.fround(0.8);
  const result = await resolveCube(levels, 16, energy);
  const normalization = blackbodyXYZ(6500)[1];
  const target = [0, 0, 0];
  for (const [population, temperature] of skyTemperatures.entries()) {
    const rgb = xyzToLinearRGB(blackbodyXYZ(temperature / energy));
    for (let channel = 0; channel < 3; channel++) {
      target[channel] =
        (target[channel] ?? 0) +
        ((rgb[channel] ?? Number.NaN) * (coefficients[population] ?? Number.NaN)) / normalization;
    }
  }
  for (let y = 0; y < 16; y++) {
    for (let x = 0; x < 16 * 6; x++) {
      for (const [channel, expected] of target.entries()) {
        const value = result.data[y * result.rowElements + x * 4 + channel] ?? Number.NaN;
        expect(Math.abs(value / expected - 1)).toBeLessThan(0.005);
      }
    }
  }
});

test.each([0.8, 1, 1.2])(
  "point stars preserve spectral flux with observer energy %s",
  async (energy) => {
    const stars = [
      { direction: normalize([1, 0.2, 0.3]), temperature: 3270, flux: 0.002 },
      { direction: normalize([0, 0, -1]), temperature: 15500, flux: 0.004 },
    ];
    const dark = createSkyMap(8).map(({ size, data }) => ({
      size,
      data: new Float16Array(data.length),
    }));
    const result = await resolveCube(dark, 64, energy, stars);
    let flux = 0;
    for (let y = 0; y < 64; y++) {
      for (let x = 0; x < 64 * 6; x++) {
        const offset = y * result.rowElements + x * 4;
        const luminance =
          0.2126390059 * (result.data[offset] ?? Number.NaN) +
          0.7151686788 * (result.data[offset + 1] ?? Number.NaN) +
          0.0721923154 * (result.data[offset + 2] ?? Number.NaN);
        flux += luminance * cubeTexelSolidAngle(64, x % 64, y);
      }
    }
    const expected = stars.reduce(
      (sum, star) =>
        sum +
        (star.flux * blackbodyXYZ(star.temperature / energy)[1]) /
          blackbodyXYZ(star.temperature)[1],
      0,
    );
    expect(Math.abs(flux / expected - 1)).toBeLessThan(0.03);
  },
);

test("diffuse spectral range failures propagate through the production sky pass", async () => {
  const size = 8;
  const data = new Float16Array(size * size * 6 * 4);
  for (let pixel = 0; pixel < size * size * 6; pixel++) {
    data[pixel * 4 + 2] = skyCoefficientScale * 1e-5;
  }
  const result = await resolveCube([{ size, data }], size, 1e-6);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size * 6; x++) {
      expect(result.data[y * result.rowElements + x * 4 + 3]).toBe(0);
    }
  }
  expect(result.data.every(Number.isFinite)).toBe(true);
});
