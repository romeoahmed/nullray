import { expect, test } from "vitest";
import boundaryFixture from "../fixtures/boundary-beams.json" with { type: "json" };
import { createBeamQueue } from "../../src/render/beams.ts";
import { opticalSources } from "../../src/render/optics.ts";
import { compileShader, requestDevice } from "../../src/render/device.ts";
import { photonFromLocal } from "../../src/physics/photon.ts";
import { normalize } from "../../src/physics/vector.ts";
import { traceVisibility } from "../reference/visibility.ts";
import { evaluateGeodesic } from "../reference/geodesic.ts";
import { integrateTransport } from "../reference/transport.ts";

const zoom = Math.fround(2 / Math.sqrt(3));
const inclination = Math.fround(1.2);
const space = { spin: Math.fround(0.7), charge: Math.fround(0.2) };
const pixels = [
  [0, 0],
  [8, 0],
  [0, 8],
  [8, 8],
  [8, 4],
];

const scatteredCamera = {
  width: 9,
  height: 9,
  radius: 30,
  zoom,
  inner: 10000,
  outer: 20000,
  positions: pixels,
  referenceStep: 1 / 64,
  relativeTolerance: 0.01,
};

function skyDirection(x: number, y: number, camera = scatteredCamera) {
  const source = normalize([
    -1,
    ((y + 0.5 - camera.height / 2) / camera.height) * Math.fround(camera.zoom),
    ((x + 0.5 - camera.width / 2) / camera.height) * Math.fround(camera.zoom),
  ]);
  const photon = photonFromLocal(space, camera.radius, inclination, source);
  const { path, outcome } = traceVisibility(space, photon, camera.radius, inclination, {
    inner: camera.inner,
    outer: camera.outer,
  });
  if (outcome.kind !== "sky") {
    throw new Error("The beam fixture must escape.");
  }
  const mapped = integrateTransport(space, photon, path, outcome.time, false, 1e-11);
  const mu = evaluateGeodesic(path, outcome.time)?.cosineTheta;
  if (mapped.kind !== "resolved" || mu === undefined) {
    throw new Error("Unresolved reference beam.");
  }
  const radial = Math.sqrt(1 - mu * mu);
  return [radial * Math.cos(mapped.azimuth), radial * Math.sin(mapped.azimuth), mu];
}

test("differentiated beam agrees with smaller binary64 sky differences and rejects a wrong branch", async () => {
  const device = await requestDevice();
  using owned = new DisposableStack();
  owned.defer(() => device.destroy());
  const buffer = (size: number, usage: GPUBufferUsageFlags) =>
    owned.adopt(device.createBuffer({ size, usage }), (value) => value.destroy());
  const uniform = buffer(96, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(uniform, 48, new Float32Array([-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0]));
  device.queue.writeBuffer(
    uniform,
    0,
    new Float32Array([
      9,
      9,
      0,
      0,
      30,
      inclination,
      zoom,
      0,
      space.spin,
      space.charge,
      10000,
      20000,
    ]),
  );
  const output = buffer(pixels.length * 48, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const staging = buffer(output.size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const module = await compileShader(
    device,
    `${opticalSources.beam}
    @group(0) @binding(7) var<storage, read_write> result: array<vec4f>;
    @compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id: vec3u) {
      let positions = array<vec2f, 5>(vec2f(0.0), vec2f(8.0, 0.0), vec2f(0.0, 8.0), vec2f(8.0), vec2f(8.0, 4.0));
      let pixel = positions[id.x];
      let center = screen_ray(frame, pixel).value;
      let beam = differential_beam(pixel, center.w);
      result[id.x * 3u] = beam.dx;
      result[id.x * 3u + 1u] = beam.dy;
      result[id.x * 3u + 2u] = differential_beam(pixel, center.w + 100.0).dx;
    }`,
    "refined-beam-test",
  );
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "probe" },
  });
  const bindings = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 7, resource: { buffer: output } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindings);
  pass.dispatchWorkgroups(pixels.length);
  pass.end();
  encoder.copyBufferToBuffer(output, 0, staging, 0, output.size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange()).slice();
  staging.unmap();
  for (const [index, pixel] of pixels.entries()) {
    const [x = NaN, y = NaN] = pixel;
    for (let axis = 0; axis < 2; axis++) {
      const before = skyDirection(x - (axis === 0 ? 1 / 64 : 0), y - (axis === 1 ? 1 / 64 : 0));
      const after = skyDirection(x + (axis === 0 ? 1 / 64 : 0), y + (axis === 1 ? 1 / 64 : 0));
      const expected = after.map((value, component) => (value - (before[component] ?? NaN)) * 32);
      const scale = Math.hypot(...expected);
      expect(result[index * 12 + axis * 4 + 3]).toBe(1);
      const error = expected.map(
        (value, component) => (result[index * 12 + axis * 4 + component] ?? NaN) - value,
      );
      expect(Math.hypot(...error) / scale).toBeLessThan(0.01);
    }
    expect(Array.from(result.subarray(index * 12 + 8, index * 12 + 12))).toEqual([0, 0, 0, 0]);
  }
});

test.for([
  { name: "scattered rays", camera: scatteredCamera, cellX: 1, cellY: 2 },
  { name: "thin branch regressions", camera: boundaryFixture, cellX: 1, cellY: 2 },
  {
    name: "critical vertical boundary sample",
    camera: {
      ...boundaryFixture,
      width: 1280,
      height: 720,
      positions: [[524, 360]],
      referenceStep: 1 / 1024,
    },
    cellX: 2,
    cellY: 2,
  },
  {
    name: "rounded horizontal boundary regression",
    camera: {
      ...boundaryFixture,
      width: 1280,
      height: 720,
      positions: [[686, 191]],
      referenceStep: 1 / 1024,
    },
    cellX: 2,
    cellY: 1,
  },
])(
  "$name: boundary atlas repairs retain fractional camera coordinates and binary64 derivatives",
  async ({ camera, cellX, cellY }) => {
    const positions = camera.positions;
    const device = await requestDevice();
    using owned = new DisposableStack();
    owned.defer(() => device.destroy());
    device.pushErrorScope("validation");
    const buffer = (size: number, usage: GPUBufferUsageFlags) =>
      owned.adopt(device.createBuffer({ size, usage }), (value) => value.destroy());
    const uniform = buffer(96, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(uniform, 48, new Float32Array([-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0]));
    device.queue.writeBuffer(
      uniform,
      0,
      new Float32Array([
        camera.width,
        camera.height,
        0,
        0,
        camera.radius,
        inclination,
        camera.zoom,
        0,
        space.spin,
        space.charge,
        camera.inner,
        camera.outer,
      ]),
    );
    const origins = buffer(positions.length * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(origins, 0, new Uint32Array(positions.flat()));
    const count = buffer(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(count, 0, new Uint32Array([positions.length, 0, 0, 0]));
    const map = owned.adopt(
      device.createTexture({
        size: [4, positions.length * 4],
        format: "rgba32float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      }),
      (value) => value.destroy(),
    );
    const output = buffer(positions.length * 32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const readback = buffer(output.size + 16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const module = await compileShader(
      device,
      `${opticalSources.edgeBeam}
    @group(0) @binding(9) var atlas: texture_storage_2d<rgba32float, write>;
    @group(0) @binding(10) var repaired_index: texture_2d<u32>;
    @group(0) @binding(12) var<storage, read_write> result: array<vec4f>;
    @compute @workgroup_size(1) fn seed(@builtin(global_invocation_id) id: vec3u) {
      let pixel = vec2f(boundary_pixels[id.x]) + vec2f(${(cellX + 0.5) / 4 - 0.5}, ${(cellY + 0.5) / 4 - 0.5});
      textureStore(atlas, vec2i(${cellX}, i32(id.x * 4u + ${cellY}u)), screen_ray(frame, pixel).value);
    }
    @compute @workgroup_size(1) fn read_beams(@builtin(global_invocation_id) id: vec3u) {
      let index = textureLoad(repaired_index, vec2i(${cellX}, i32(id.x * 4u + ${cellY}u)), 0).x;
      if (index == 0u) { return; }
      let beam = repaired_beams[index - 1u];
      result[id.x * 2u] = beam.dx;
      result[id.x * 2u + 1u] = beam.dy;
    }`,
      "boundary-atlas-reference",
    );
    const pipeline = (entryPoint: string) =>
      device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint } });
    const find = await pipeline("find_edge_beams");
    const repair = await pipeline("repair_edge_beams");
    const queue = createBeamQueue(
      device,
      owned,
      uniform,
      map,
      { find, repair },
      {
        find: [
          { binding: 7, resource: { buffer: origins } },
          { binding: 8, resource: { buffer: count } },
        ],
        repair: [{ binding: 7, resource: { buffer: origins } }],
      },
    );
    const seed = await pipeline("seed");
    const read = await pipeline("read_beams");
    const seedBindings = device.createBindGroup({
      layout: seed.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 7, resource: { buffer: origins } },
        { binding: 9, resource: map.createView() },
      ],
    });
    const readBindings = device.createBindGroup({
      layout: read.getBindGroupLayout(0),
      entries: [
        { binding: 3, resource: { buffer: queue.beams } },
        { binding: 10, resource: queue.index.createView() },
        { binding: 12, resource: { buffer: output } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const seedPass = encoder.beginComputePass();
    seedPass.setPipeline(seed);
    seedPass.setBindGroup(0, seedBindings);
    seedPass.dispatchWorkgroups(positions.length);
    seedPass.end();
    // A second geometry pass must overwrite partial or complete results left by the first.
    for (let revision = 0; revision < 2; revision++) {
      encoder.clearBuffer(queue.statistics);
      queue.encode(encoder);
    }
    const readPass = encoder.beginComputePass();
    readPass.setPipeline(read);
    readPass.setBindGroup(0, readBindings);
    readPass.dispatchWorkgroups(positions.length);
    readPass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, output.size);
    encoder.copyBufferToBuffer(queue.statistics, 0, readback, output.size, 16);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const mapped = readback.getMappedRange();
    const result = new Float32Array(mapped, 0, output.size / 4).slice();
    const statistics = new Uint32Array(mapped, output.size, 3).slice();
    if (cellX === 2) {
      expect(Array.from(statistics)).toEqual([1, 1, 1]);
    }
    readback.unmap();
    expect(await device.popErrorScope()).toBeNull();
    for (const [index, pixel] of positions.entries()) {
      const x = (pixel[0] ?? NaN) + (cellX + 0.5) / 4 - 0.5;
      const y = (pixel[1] ?? NaN) + (cellY + 0.5) / 4 - 0.5;
      for (let axis = 0; axis < 2; axis++) {
        const before = skyDirection(
          x - (axis === 0 ? camera.referenceStep : 0),
          y - (axis === 1 ? camera.referenceStep : 0),
          camera,
        );
        const after = skyDirection(
          x + (axis === 0 ? camera.referenceStep : 0),
          y + (axis === 1 ? camera.referenceStep : 0),
          camera,
        );
        const expected = after.map(
          (value, component) => (value - (before[component] ?? NaN)) / (2 * camera.referenceStep),
        );
        const error = expected.map(
          (value, component) => (result[index * 8 + axis * 4 + component] ?? NaN) - value,
        );
        expect(result[index * 8 + axis * 4 + 3]).toBe(1);
        expect(Math.hypot(...error) / Math.hypot(...expected)).toBeLessThan(
          camera.relativeTolerance,
        );
      }
    }
  },
);
