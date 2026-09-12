import { expect } from "vitest";
import { server } from "vitest/browser";
import { createStructureField } from "../../src/gpu/sources/structure.ts";
import { createSkyTexture } from "../../src/gpu/sources/sky.ts";
import { createNoiseVolume } from "../reference/structure.ts";
import { createSkyMap, cubeTexelSolidAngle } from "../reference/sky.ts";
import { readPixels, test, computeReadback } from "./compute.ts";
import { createStarTree } from "../../src/physics/stars.ts";
import type { Star } from "../../src/physics/sky.ts";
import { blackbodyXYZ, createBlackbodyTable, xyzToLinearRGB } from "../../src/physics/radiation.ts";
import radiation from "../../src/gpu/wgsl/imaging/radiation.wgsl?raw";
import stellar from "../../src/gpu/wgsl/sources/stars.wgsl?raw";

test("stellar tree queries agree with direct flux sums through repeated positions and nearly rank-one footprints", async () => {
  const catalogue: Star[] = Array.from({ length: 97 }, (_, index) => {
    const x = ((index % 13) - 6) * 0.002;
    const y = (Math.floor(index / 13) - 3) * 0.003;
    return {
      direction: [x, y, Math.sqrt(1 - x * x - y * y)],
      temperature: 6500,
      flux: (index + 1) * 1e-12,
    };
  });
  catalogue.push(
    ...Array.from({ length: 9 }, () => ({
      direction: [0, 0, 1] as const,
      temperature: 6500,
      flux: 1e-10,
    })),
  );
  catalogue.push({ direction: [0, 0, -1], temperature: 6500, flux: 1e-5 });
  const footprints = new Float32Array([
    0.01, 0, 0, 0.02, 0.03, 0.03, 0.02, 0.020001, 0.001, 0.0002, -0.0003, 0.002,
  ]);
  const normalization = blackbodyXYZ(6500)[1];
  const spectrum = xyzToLinearRGB(blackbodyXYZ(6500)).map((value) => value / normalization);
  const tree = createStarTree(catalogue);
  const result = await computeReadback(
    `${radiation}\n${stellar}
    @group(0) @binding(0) var<storage, read> input: array<vec4f>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
    @compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id: vec3u) {
      let p = input[id.x];
      output[id.x] = stellar_radiance(vec3f(0, 0, 1), vec3f(1, 0, 0), vec3f(0, 1, 0), p.xy, p.zw, 1);
    }`,
    footprints,
    footprints.length,
    footprints.length / 4,
    [createBlackbodyTable().data],
    "probe",
    (device) => {
      const buffer = device.createBuffer({
        size: tree.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
      });
      device.queue.writeBuffer(buffer, 0, tree);
      return { entries: [{ binding: 18, resource: { buffer } }], dispose: () => buffer.destroy() };
    },
  );
  for (let index = 0; index < footprints.length; index += 4) {
    const [xx = NaN, xy = NaN, yx = NaN, yy = NaN] = footprints.slice(index, index + 4);
    const cxx = 0.4225 * (xx * xx + yx * yx) + 1e-12;
    const cyy = 0.4225 * (xy * xy + yy * yy) + 1e-12;
    const cxy = 0.4225 * (xx * xy + yx * yy);
    // Compare the stable f32 form with a direct binary64 inverse on this near-singular fixture.
    const determinant = cxx * cyy - cxy * cxy;
    let flux = 0;
    for (const star of catalogue) {
      const norm = Math.hypot(...star.direction);
      const [x, y, z] = star.direction.map((value) => Math.fround(value / norm));
      if (x === undefined || y === undefined || z === undefined || z <= 0) {
        continue;
      }
      const squared = (cyy * x * x - 2 * cxy * x * y + cxx * y * y) / determinant;
      if (squared <= 12.25) {
        flux +=
          (Math.fround(star.flux) * Math.exp(-0.5 * squared)) /
          (2 * Math.PI * Math.sqrt(determinant));
      }
    }
    expect(result[index + 3]).toBe(1);
    for (let channel = 0; channel < 3; channel++) {
      const expected = flux * (spectrum[channel] ?? NaN);
      expect(Math.abs((result[index + channel] ?? NaN) / expected - 1)).toBeLessThan(0.006);
    }
  }
});

test("parallel material lattice retains every periodic scalar sample", async ({ device }) => {
  using owned = new DisposableStack();
  const field = owned.adopt(await createStructureField(device), (value) => value.dispose());
  const reference = createNoiseVolume();
  const stride = reference.size * 4;
  const staging = owned.adopt(
    device.createBuffer({
      size: stride * reference.size ** 2,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
    (value) => value.destroy(),
  );
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture: field.texture },
    {
      buffer: staging,
      bytesPerRow: stride,
      rowsPerImage: reference.size,
    },
    [reference.size, reference.size, reference.size],
  );
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(staging.getMappedRange());
  const values = Uint8Array.from(reference.data, (_, index) => bytes[index * 4] ?? 0);
  expect(values).toEqual(reference.data);
  staging.unmap();
});

test.for([1, 64])(
  "GPU spectral cube at face size %s agrees with an independent source and conserves mip flux",
  async (size, { device }) => {
    using owned = new DisposableStack();
    const field = owned.adopt(await createStructureField(device), (value) => value.dispose());
    const texture = owned.adopt(await createSkyTexture(device, field, size), (value) =>
      value.destroy(),
    );
    const reference = createSkyMap(size);
    let initialFlux: readonly number[] | undefined;
    let maximumErrorRatio = 0;
    const report = [];
    for (const [mip, level] of reference.entries()) {
      const flux = [0, 0, 0];
      let squaredError = 0,
        squaredReference = 0,
        maximumAbsolute = 0;
      let worst = { face: 0, pixel: 0, channel: 0, actual: 0, expected: 0, ratio: 0 };
      // Read all faces together; each copy uses an independent staging buffer.
      // oxlint-disable-next-line no-await-in-loop
      const faces = await Promise.all(
        Array.from({ length: 6 }, (_, face) => readPixels(device, texture, mip, face)),
      );
      for (const [face, pixels] of faces.entries()) {
        expect(pixels.every(Number.isFinite)).toBe(true);
        for (let pixel = 0; pixel < level.size ** 2; pixel++) {
          const area = cubeTexelSolidAngle(
            level.size,
            pixel % level.size,
            Math.floor(pixel / level.size),
          );
          for (let channel = 0; channel < 3; channel++) {
            const actual = pixels[pixel * 4 + channel] ?? NaN;
            const expected = level.data[(face * level.size ** 2 + pixel) * 4 + channel] ?? NaN;
            // f32 source arithmetic and f16 storage perturb the binary64 procedural reference.
            const difference = Math.abs(actual - expected);
            const ratio = difference / (0.01 * Math.abs(expected) + 2e-5);
            squaredError += difference ** 2 * area;
            squaredReference += expected ** 2 * area;
            maximumAbsolute = Math.max(maximumAbsolute, difference);
            maximumErrorRatio = Math.max(maximumErrorRatio, ratio);
            if (ratio > worst.ratio) {
              worst = { face, pixel, channel, actual, expected, ratio };
            }
            flux[channel] = (flux[channel] ?? 0) + actual * area;
          }
        }
      }
      report.push({
        mip,
        rmsRelative: Math.sqrt(squaredError / squaredReference),
        maximumAbsolute,
        worst,
        flux,
      });
      initialFlux ??= flux;
      for (const [channel, value] of flux.entries()) {
        // Allow the compared mip integrals' separate f16 storage roundings in this tolerance.
        expect(Math.abs(value / (initialFlux[channel] ?? NaN) - 1)).toBeLessThan(0.001);
      }
    }
    await server.commands.writeFile(
      `test-results/sky-${size}.json`,
      JSON.stringify(report, null, 2),
    );
    expect(maximumErrorRatio).toBeLessThan(1);
  },
);
