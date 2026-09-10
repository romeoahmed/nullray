import { describe, expect, test } from "vitest";
import { opticalSources } from "../../src/gpu/optics.ts";
import { createStarTree } from "../../src/physics/stars.ts";
import type { Star, SkyLevel } from "../../src/physics/sky.ts";
import { compileShader, requestDevice } from "../../src/gpu/device.ts";
import { createSkyTexture } from "../../src/gpu/sky.ts";
import { blackbodyXYZ, createBlackbodyTable, xyzToLinearRGB } from "../../src/physics/radiation.ts";
import {
  createSkyMap,
  cubeDirection,
  cubeTexelSolidAngle,
  skyTemperatures,
  skyCoefficientScale,
} from "../../src/physics/sky.ts";
import { cross, dot, normalize } from "../../src/physics/vector.ts";
import type { Vec3 } from "../../src/physics/vector.ts";
import { server } from "vitest/browser";
import radiationSource from "../../src/gpu/wgsl/imaging/radiation.wgsl?raw";
import starsSource from "../../src/gpu/wgsl/imaging/source-tree.wgsl?raw";
import { computeReadback } from "./compute.ts";

/** Identity sky projection over six separate cube-face tiles, with a common endpoint energy. */
async function resolveCube(
  levels: readonly SkyLevel[],
  size: number,
  energy: number,
  stars: readonly Star[] = [],
  directionAt?: (face: number, x: number, y: number) => Vec3,
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
        const direction =
          directionAt?.(face, x, y) ??
          cubeDirection(face, (2 * (x + 0.5)) / size - 1, (2 * (y + 0.5)) / size - 1);
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
      { binding: 19, resource: { buffer: buffer(96, GPUBufferUsage.UNIFORM) } },
      { binding: 20, resource: { buffer: buffer(16, GPUBufferUsage.STORAGE) } },
      { binding: 21, resource: { buffer: buffer(64, GPUBufferUsage.STORAGE) } },
      { binding: 23, resource: { buffer: buffer(width * size * 4, GPUBufferUsage.STORAGE) } },
      { binding: 24, resource: { buffer: buffer(16, GPUBufferUsage.STORAGE) } },
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

describe("Diffuse sky", () => {
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

  test("production stellar resolve retains two folded images within one grid cell", async () => {
    const scale = 0.2;
    const corners = [
      [0, 0, 1],
      normalize([scale, 0, 1]),
      normalize([scale, 0, 1]),
      normalize([2 * scale, scale, 1]),
    ] as const;
    const u = 0.2,
      v = 0.8;
    const value = (axis: 0 | 1 | 2) =>
      (1 - u) * (1 - v) * corners[0][axis] +
      u * (1 - v) * corners[1][axis] +
      (1 - u) * v * corners[2][axis] +
      u * v * corners[3][axis];
    const homogeneous: Vec3 = [value(0), value(1), value(2)];
    const direction = normalize(homogeneous);
    const derivative = (x: number, y: number): Vec3 => {
      const component = (axis: 0 | 1 | 2) =>
        x *
          ((1 - v) * (corners[1][axis] - corners[0][axis]) +
            v * (corners[3][axis] - corners[2][axis])) +
        y *
          ((1 - u) * (corners[2][axis] - corners[0][axis]) +
            u * (corners[3][axis] - corners[1][axis]));
      return [component(0), component(1), component(2)];
    };
    const jacobian =
      Math.abs(dot(direction, cross(derivative(1, 0), derivative(0, 1)))) /
      dot(homogeneous, homogeneous);
    const dark = [{ size: 1, data: new Float16Array(24) }];
    const result = await resolveCube(
      dark,
      8,
      1,
      [{ direction, temperature: 6500, flux: 1e-5 }],
      (_face, x, y) => normalize([scale * (x + y - 6), scale * (x - 3) * (y - 3), 1]),
    );
    let flux = 0;
    for (let y = 2; y <= 5; y++) {
      for (let x = 2; x <= 5; x++) {
        const offset = y * result.rowElements + (3 * 8 + x) * 4;
        expect(result.data[offset + 3]).toBe(1);
        flux +=
          0.2126390059 * (result.data[offset] ?? NaN) +
          0.7151686788 * (result.data[offset + 1] ?? NaN) +
          0.0721923154 * (result.data[offset + 2] ?? NaN);
      }
    }
    expect(Math.abs(flux / (2e-5 / jacobian) - 1)).toBeLessThan(0.002);
  });

  test.for([0.8, 1, 1.2])(
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
});

describe("Point sources", () => {
  // These local-map checks integrate the entire map; production ownership is checked through resolve_sky.
  const unpartitioned = `struct StellarFootprint { origin: vec2f, scale: f32, }
fn stellar_filtered_image(position: vec2f, footprint: StellarFootprint) -> bool { return false; }
`;

  const source = `${unpartitioned}\n${radiationSource}
${starsSource}
@group(0) @binding(0) var<storage, read> beams: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> colors: array<vec4f>;
@group(0) @binding(3) var<storage, read> star_nodes: array<StarNode>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&colors)) { return; }
  colors[id.x] = point_stars(beams[3u * id.x].xyz, beams[3u * id.x + 1u].xyz, beams[3u * id.x + 2u].xyz, beams[3u * id.x].w, StellarFootprint(vec2f(0.0), 0.0));
}`;

  test("point-source flux is stable through subpixel motion, shear, parity, and cube-edge directions", async () => {
    const directions: readonly Vec3[] = [[0, 0, 1], normalize([1, 1, 1]), normalize([-1, 0.2, 1])];
    const beams: number[] = [];
    const areas: number[] = [];
    const groups: number[] = [];
    // Each isolated patch observes one source; all three sources share a tree.
    const catalogue = directions.map((direction) => ({
      direction,
      temperature: 6500,
      flux: 1e-5,
    }));
    for (const direction of directions) {
      const horizontal = normalize(cross(direction, [0, 1, 0]));
      const vertical = cross(direction, horizontal);
      for (const scale of [0.001, 0.01]) {
        for (const stretch of [-3, 0.3, 1]) {
          for (const phase of [-0.49, -0.2, 0, 0.23, 0.49]) {
            groups.push(areas.length);
            const position = (x: number, y: number): Vec3 => {
              const component = (axis: 0 | 1 | 2) =>
                direction[axis] +
                scale * ((x + 0.35 * y) * horizontal[axis] + stretch * y * vertical[axis]);
              return normalize([component(0), component(1), component(2)]);
            };
            for (let y = -3; y <= 3; y++) {
              for (let x = -3; x <= 3; x++) {
                const center = position(x + phase, y + phase);
                const left = position(x + phase - 1, y + phase);
                const right = position(x + phase + 1, y + phase);
                const before = position(x + phase, y + phase - 1);
                const after = position(x + phase, y + phase + 1);
                const dx: Vec3 = [
                  (right[0] - left[0]) / 2,
                  (right[1] - left[1]) / 2,
                  (right[2] - left[2]) / 2,
                ];
                const dy: Vec3 = [
                  (after[0] - before[0]) / 2,
                  (after[1] - before[1]) / 2,
                  (after[2] - before[2]) / 2,
                ];
                beams.push(...center, 1, ...dx, 0, ...dy, 0);
                areas.push(Math.abs(dot(center, cross(dx, dy))));
              }
            }
          }
        }
      }
    }
    const result = await computeReadback(
      source,
      new Float32Array(beams),
      areas.length * 4,
      Math.ceil(areas.length / 64),
      [createBlackbodyTable().data, createStarTree(catalogue)],
    );
    for (const start of groups) {
      let flux = 0;
      for (let index = start; index < start + 49; index++) {
        expect(result[index * 4 + 3]).toBe(1);
        flux +=
          (0.2126390059 * (result[index * 4] ?? Number.NaN) +
            0.7151686788 * (result[index * 4 + 1] ?? Number.NaN) +
            0.0721923154 * (result[index * 4 + 2] ?? Number.NaN)) *
          (areas[index] ?? Number.NaN);
      }
      expect(Math.abs(flux / 1e-5 - 1)).toBeLessThan(0.002);
    }
  });

  test("stellar spectral range failures remain unresolved before unsafe frequency division", async () => {
    const energies = [1, 1e-6, 0, -1];
    const result = await computeReadback(
      source,
      new Float32Array(
        energies.flatMap((energy) => [0, 0, 1, energy, 0.01, 0, 0, 0, 0, 0.01, 0, 0]),
      ),
      energies.length * 4,
      1,
      [
        createBlackbodyTable().data,
        createStarTree([{ direction: [0, 0, 1], temperature: 6500, flux: 1e-5 }]),
      ],
    );
    expect(result.every(Number.isFinite)).toBe(true);
    expect(Array.from(result).filter((_, index) => index % 4 === 3)).toEqual([1, 0, 0, 0]);
  });

  test("stellar inversion retains finite coverage across large isotropic beam scales", async () => {
    const scales = [1e-11, 1e-9, 1, 1e15, 1e25].map(Math.fround);
    const result = await computeReadback(
      source,
      new Float32Array(scales.flatMap((scale) => [0, 0, 1, 1, scale, 0, 0, 0, 0, scale, 0, 0])),
      scales.length * 4,
      1,
      [
        createBlackbodyTable().data,
        createStarTree([{ direction: [0, 0, 1], temperature: 6500, flux: 1e-5 }]),
      ],
    );
    expect(result.every(Number.isFinite)).toBe(true);
    expect(result[3]).toBe(0);
    for (const [index, scale] of scales.entries()) {
      if (index === 0) {
        continue;
      }
      expect(result[index * 4 + 3]).toBe(1);
      const luminance =
        0.2126390059 * (result[index * 4] ?? NaN) +
        0.7151686788 * (result[index * 4 + 1] ?? NaN) +
        0.0721923154 * (result[index * 4 + 2] ?? NaN);
      if (scale > 1e20) {
        expect(luminance).toBe(0);
      } else {
        expect(Math.abs((luminance * scale * scale) / 1e-5 - 1)).toBeLessThan(0.002);
      }
    }
  });

  test.for([0.2, 0.75, 1.5])(
    "folded stellar map with image radius %s",
    async (radius, { annotate }) => {
      const scale = 0.001;
      const flux = 1e-5;
      const sourceDirection = normalize([scale * radius * radius, 0, 1]);
      const tree = createStarTree([{ direction: sourceDirection, temperature: 6500, flux }]);
      // Exact roots of the uploaded source's gnomonic coordinate under u = scale * x².
      const sourceU = Math.fround(sourceDirection[0]) / Math.fround(sourceDirection[2]);
      const root = Math.sqrt(sourceU / scale);
      const expected = (flux * (1 + sourceU * sourceU) ** 1.5) / (scale * scale * root);
      const beams: number[] = [];
      const groups = [];
      for (const subdivisions of [1, 2, 4, 8, 16, 32]) {
        for (const phase of [0.125, 0.375]) {
          const start = beams.length / 12;
          const step = 1 / subdivisions;
          for (let iy = -3 * subdivisions; iy < 3 * subdivisions; iy++) {
            for (let ix = -3 * subdivisions; ix < 3 * subdivisions; ix++) {
              const x = (ix + phase) * step;
              const y = (iy + phase) * step;
              const vector: Vec3 = [scale * x * x, scale * y, 1];
              const length = Math.hypot(...vector);
              const direction = normalize(vector);
              const derivative = (raw: Vec3): Vec3 => {
                const projection = dot(direction, raw);
                return [
                  ((raw[0] - direction[0] * projection) * step) / length,
                  ((raw[1] - direction[1] * projection) * step) / length,
                  ((raw[2] - direction[2] * projection) * step) / length,
                ];
              };
              beams.push(
                ...direction,
                1,
                ...derivative([2 * scale * x, 0, 0]),
                0,
                ...derivative([0, scale, 0]),
                0,
              );
            }
          }
          groups.push({ subdivisions, phase, start, end: beams.length / 12 });
        }
      }
      const result = await computeReadback(
        source,
        new Float32Array(beams),
        beams.length / 3,
        Math.ceil(beams.length / 12 / 64),
        [createBlackbodyTable().data, tree],
      );
      const records = groups.map(({ subdivisions, phase, start, end }) => {
        let measured = 0;
        for (let index = start; index < end; index++) {
          expect(result[index * 4 + 3]).toBe(1);
          measured +=
            (0.2126390059 * (result[index * 4] ?? NaN) +
              0.7151686788 * (result[index * 4 + 1] ?? NaN) +
              0.0721923154 * (result[index * 4 + 2] ?? NaN)) /
            subdivisions ** 2;
        }
        return { subdivisions, phase, measured, relativeError: Math.abs(measured / expected - 1) };
      });
      const body = JSON.stringify({ radius, sourceU, expected, records }, null, 2);
      await server.commands.writeFile(`test-results/stellar-fold-${radius}.json`, body);
      await annotate("Analytic two-image fold integration", {
        body,
        bodyEncoding: "utf-8",
        contentType: "application/json",
      });
      for (const record of records.filter((value) => value.subdivisions === 32)) {
        expect(record.relativeError).toBeLessThan(0.02);
      }
    },
  );

  test("wide normalized affine beams use the solid-angle Jacobian at the stellar image", async () => {
    const flux = 1e-5;
    const tree = createStarTree([{ direction: [0.8, 0.6, 1], temperature: 6500, flux }]);
    const u = (tree[0] ?? NaN) / (tree[2] ?? NaN);
    const v = (tree[1] ?? NaN) / (tree[2] ?? NaN);
    const scales = [
      [1.2, 0.9],
      [-1.2, 0.9],
      [2, 1.5],
      [10, -20],
    ].map((pair) => pair.map(Math.fround));
    const result = await computeReadback(
      source,
      new Float32Array(
        scales.flatMap(([x, y]) => [0, 0, 1, 1, x ?? NaN, 0, 0, 0, 0, y ?? NaN, 0, 0]),
      ),
      scales.length * 4,
      1,
      [createBlackbodyTable().data, tree],
    );
    for (const [index, [x = NaN, y = NaN]] of scales.entries()) {
      // n(X,Y) = normalize([x*X, y*Y, 1]); its exact solid-angle density
      // at the source is |x*y| / (1+u²+v²)^(3/2).
      const area = Math.abs(x * y) / (1 + u * u + v * v) ** 1.5;
      const weight = (1 - Math.abs(u / x)) * (1 - Math.abs(v / y));
      const expected = (flux * weight) / area;
      const measured =
        0.2126390059 * (result[index * 4] ?? NaN) +
        0.7151686788 * (result[index * 4 + 1] ?? NaN) +
        0.0721923154 * (result[index * 4 + 2] ?? NaN);
      expect(result[index * 4 + 3]).toBe(1);
      expect(Math.abs(measured / expected - 1)).toBeLessThan(0.002);
    }
  });

  test.for([
    { cosine: 1e-8, scale: 1e9, resolved: true },
    { cosine: 1e-20, scale: 1e21, resolved: true },
    { cosine: 1e-24, scale: 1e25, resolved: false },
  ])(
    "stellar image normalization near the tangent horizon: $cosine",
    async ({ cosine, scale, resolved }) => {
      const flux = 1e-5;
      const tree = createStarTree([{ direction: [1, 0, cosine], temperature: 6500, flux }]);
      const roundedScale = Math.fround(scale);
      const result = await computeReadback(
        source,
        new Float32Array([0, 0, 1, 1, roundedScale, 0, 0, 0, 0, roundedScale, 0, 0]),
        4,
        1,
        [createBlackbodyTable().data, tree],
      );
      expect(result.every(Number.isFinite)).toBe(true);
      expect(result[3]).toBe(resolved ? 1 : 0);
      if (resolved) {
        const u = (tree[0] ?? NaN) / (tree[2] ?? NaN);
        const jacobian = roundedScale ** 2 / (1 + u * u) ** 1.5;
        const expected = (flux * (1 - u / roundedScale)) / jacobian;
        const measured =
          0.2126390059 * (result[0] ?? NaN) +
          0.7151686788 * (result[1] ?? NaN) +
          0.0721923154 * (result[2] ?? NaN);
        expect(Math.abs(measured / expected - 1)).toBeLessThan(0.002);
      }
    },
  );
});
