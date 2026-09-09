import { expect, test } from "vitest";
import type { TestContext } from "vitest";
import { server } from "vitest/browser";
import { createOptics, opticalSources } from "../../src/render/optics.ts";
import { compileShader, requestDevice } from "../../src/render/device.ts";
import { createAppearance, initialAppearance } from "../../src/model/appearance.ts";
import { initialScene } from "../../src/model/scene.ts";

interface DiskAuditSource {
  readonly name: string;
  readonly temperature: number;
  readonly contrast: number;
  readonly time: number;
  readonly offset: number;
}

async function auditDisk(source: DiskAuditSource, annotate: TestContext["annotate"]) {
  const appearanceResult = createAppearance({
    ...initialAppearance,
    diskTemperature: source.temperature,
    diskStructure: source.contrast,
  });
  if (!appearanceResult.ok) {
    throw new Error(appearanceResult.error);
  }
  const appearance = appearanceResult.value;
  const device = await requestDevice();
  using owned = new DisposableStack();
  owned.defer(() => device.destroy());
  device.pushErrorScope("validation");
  const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
  const width = 128;
  const height = 96;
  const count = width * height;
  const colorBytes = count * 8;
  const staging = owned.adopt(
    device.createBuffer({
      size: colorBytes * 3 + 32,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
    (value) => value.destroy(),
  );
  async function frame(
    antialias: boolean,
    jitter: readonly [number, number] = [0, 0],
    trace = true,
    time = source.time,
  ) {
    const encoder = device.createCommandEncoder();
    const image = optics.encode(encoder, initialScene, width, height, {
      time,
      trace,
      antialias,
      jitter: [jitter[0] + source.offset, jitter[1] + source.offset / 2],
      appearance,
    });
    encoder.copyTextureToBuffer(
      { texture: image.radiance },
      { buffer: staging, bytesPerRow: width * 8 },
      [width, height],
    );
    encoder.copyTextureToBuffer(
      { texture: image.endpoints },
      { buffer: staging, offset: colorBytes, bytesPerRow: width * 16 },
      [width, height],
    );
    encoder.copyBufferToBuffer(image.edgeStatistics, 0, staging, colorBytes * 3, 16);
    encoder.copyBufferToBuffer(image.edgeBeamStatistics, 0, staging, colorBytes * 3 + 16, 16);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const mapped = staging.getMappedRange();
    const color = new Float16Array(mapped, 0, count * 4).slice();
    const endpoints = new Float32Array(mapped, colorBytes, count * 4).slice();
    const candidates = new Uint32Array(mapped, colorBytes * 3, 1)[0] ?? 0;
    const beamStatistics = new Uint32Array(mapped, colorBytes * 3 + 16, 3).slice();
    staging.unmap();
    return { color, endpoints, candidates, beamStatistics };
  }
  const center = await frame(false);
  const enabled = await frame(true, [0, 0], false);
  const adaptive = await frame(true);
  expect(enabled.color).toEqual(adaptive.color);
  const repeated = await frame(true);
  expect(repeated.candidates).toBe(adaptive.candidates);
  expect(repeated.color).toEqual(adaptive.color);
  const animated = await frame(true, [0, 0], false, source.time + 40);
  const animatedAgain = await frame(true, [0, 0], false, source.time + 40);
  expect(animated.candidates).toBe(adaptive.candidates);
  expect(animated.beamStatistics).toEqual(adaptive.beamStatistics);
  expect(animatedAgain.beamStatistics).toEqual(adaptive.beamStatistics);
  expect(animated.endpoints).toEqual(adaptive.endpoints);
  if (source.contrast > 0) {
    expect(animated.color).not.toEqual(adaptive.color);
  } else {
    expect(animated.color).toEqual(adaptive.color);
  }
  expect(animatedAgain.color).toEqual(animated.color);
  const reference = new Float64Array(count * 3);
  const eligible = new Uint8Array(count).fill(1);
  const mixed = new Uint8Array(count);
  for (let sample = 0; sample < 16; sample++) {
    // Each submission must finish before the next frame overwrites its uniforms and textures.
    // oxlint-disable-next-line no-await-in-loop
    const dense = await frame(false, [
      ((sample % 4) + 0.5) / 4 - 0.5,
      (Math.floor(sample / 4) + 0.5) / 4 - 0.5,
    ]);
    for (let pixel = 0; pixel < count; pixel++) {
      const tag = dense.endpoints[pixel * 4 + 3] ?? Number.NaN;
      if (!(tag === 0 || tag <= -3) || dense.color[pixel * 4 + 3] !== 1) {
        eligible[pixel] = 0;
      }
      if (tag !== center.endpoints[pixel * 4 + 3]) {
        mixed[pixel] = 1;
      }
      for (let channel = 0; channel < 3; channel++) {
        const index = pixel * 3 + channel;
        reference[index] = (reference[index] ?? 0) + (dense.color[pixel * 4 + channel] ?? 0) / 16;
      }
    }
  }
  const denseReference = new Float64Array(count * 3);
  const interior = new Uint8Array(count);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const pixel = y * width + x;
      const tag = center.endpoints[pixel * 4 + 3] ?? NaN;
      if (tag > -3 || !eligible[pixel] || mixed[pixel]) {
        continue;
      }
      let matching = true;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          matching &&= center.endpoints[((y + dy) * width + x + dx) * 4 + 3] === tag;
        }
      }
      interior[pixel] = matching ? 1 : 0;
    }
  }
  for (let sample = 0; sample < 64; sample++) {
    // Serialized submissions protect the shared uniforms and readback storage.
    // oxlint-disable-next-line no-await-in-loop
    const dense = await frame(false, [
      ((sample % 8) + 0.5) / 8 - 0.5,
      (Math.floor(sample / 8) + 0.5) / 8 - 0.5,
    ]);
    for (let pixel = 0; pixel < count; pixel++) {
      if (
        dense.endpoints[pixel * 4 + 3] !== center.endpoints[pixel * 4 + 3] ||
        dense.color[pixel * 4 + 3] !== 1
      ) {
        interior[pixel] = 0;
      }
      for (let channel = 0; channel < 3; channel++) {
        const index = pixel * 3 + channel;
        denseReference[index] =
          (denseReference[index] ?? 0) + (dense.color[pixel * 4 + channel] ?? 0) / 64;
      }
    }
  }
  const interiorErrors = [];
  for (let pixel = 0; pixel < count; pixel++) {
    if (!interior[pixel]) {
      continue;
    }
    let flux = 0;
    let error = 0;
    let convergence = 0;
    const residual: number[] = [];
    for (let channel = 0; channel < 3; channel++) {
      const expected = denseReference[pixel * 3 + channel] ?? NaN;
      flux += Math.abs(expected);
      const difference = (adaptive.color[pixel * 4 + channel] ?? NaN) - expected;
      residual.push(difference);
      error += Math.abs(difference);
      convergence += Math.abs((reference[pixel * 3 + channel] ?? NaN) - expected);
    }
    interiorErrors.push({
      x: pixel % width,
      y: Math.floor(pixel / width),
      branch: center.endpoints[pixel * 4 + 3],
      flux,
      error,
      convergence,
      residual,
    });
  }
  expect(interiorErrors.length).toBeGreaterThan(100);
  expect(interiorErrors.every((record) => Number.isFinite(record.error + record.convergence))).toBe(
    true,
  );
  const totalFlux = interiorErrors.reduce((sum, record) => sum + record.flux, 0);
  const totalError = interiorErrors.reduce((sum, record) => sum + record.error, 0);
  // Allow one subnormal half-float quantum per RGB channel for very dark sources.
  const absoluteFloor = 3 * 2 ** -24 * interiorErrors.length;
  const tolerance =
    source.name === "default" ? 0.001 * totalFlux : 0.001 * totalFlux + absoluteFloor;
  const body = JSON.stringify(
    {
      scene: initialScene,
      width,
      height,
      time: source.time,
      jitter: [source.offset, source.offset / 2],
      appearance,
      totalError,
      totalFlux,
      tolerance,
      coarseSamples: 16,
      fineSamples: 64,
      candidates: adaptive.candidates,
      interiorErrors,
    },
    null,
    2,
  );
  await server.commands.writeFile(
    source.name === "default"
      ? "test-results/disk-interior-sampling.json"
      : `test-results/disk-interior-${source.name}.json`,
    body,
  );
  await annotate("Same-branch disk sampling audit", {
    body,
    bodyEncoding: "utf-8",
    contentType: "application/json",
  });
  expect(totalError).toBeLessThan(tolerance);
  let tested = 0;
  let centerError = 0;
  let adaptiveError = 0;
  for (let pixel = 0; pixel < count; pixel++) {
    if (!eligible[pixel] || !mixed[pixel]) {
      continue;
    }
    tested++;
    expect(adaptive.color[pixel * 4 + 3]).toBe(1);
    for (let channel = 0; channel < 3; channel++) {
      const expected = reference[pixel * 3 + channel] ?? Number.NaN;
      const actual = adaptive.color[pixel * 4 + channel] ?? Number.NaN;
      centerError += Math.abs((center.color[pixel * 4 + channel] ?? Number.NaN) - expected);
      adaptiveError += Math.abs(actual - expected);
      expect(Math.abs(actual - expected)).toBeLessThan(0.001 * expected + 0.0001);
    }
  }
  expect(await device.popErrorScope()).toBeNull();
  expect(adaptive.candidates).toBeGreaterThan(10);
  expect(tested).toBeGreaterThan(10);
  if (source.name === "default") {
    expect(centerError).toBeGreaterThan(0.01);
    expect(adaptiveError).toBeLessThan(centerError * 0.02);
  }
  return { interiorErrors, totalError, totalFlux, candidates: adaptive.candidates };
}

test.for([
  { name: "default", temperature: 7000, contrast: 0.65, time: 137, offset: 0 },
  { name: "cold", temperature: 1000, contrast: 1, time: 24, offset: 0 },
  { name: "hot", temperature: 30000, contrast: 1, time: 100001, offset: 0 },
  { name: "steady", temperature: 7000, contrast: 0, time: 240000000, offset: 0 },
])("$name disk quadrature agrees with dense pixel integration", async (source, { annotate }) => {
  await auditDisk(source, annotate);
});

test("disk integration remains stable across subpixel frustum motion", async ({ annotate }) => {
  let previous: Awaited<ReturnType<typeof auditDisk>> | undefined;
  const frames = [];
  const transitions = [];
  for (const offset of [0, 0.125, 0.25, 0.375, 0.5]) {
    // Freeze the source while moving both adaptive and dense pixel footprints together.
    // oxlint-disable-next-line no-await-in-loop
    const current = await auditDisk(
      {
        name: `shift-${offset}`,
        temperature: 7000,
        contrast: 0.65,
        time: 137,
        offset,
      },
      annotate,
    );
    frames.push({
      offset,
      totalError: current.totalError,
      totalFlux: current.totalFlux,
      candidates: current.candidates,
      pixels: current.interiorErrors.length,
    });
    if (previous) {
      const earlier = new Map(
        previous.interiorErrors.map((record) => [record.y * 128 + record.x, record]),
      );
      let error = 0;
      let flux = 0;
      let pixels = 0;
      for (const record of current.interiorErrors) {
        const before = earlier.get(record.y * 128 + record.x);
        if (!before || before.branch !== record.branch) {
          continue;
        }
        pixels++;
        flux += (before.flux + record.flux) / 2;
        for (let channel = 0; channel < 3; channel++) {
          // Difference of errors subtracts the actual reference image change caused by motion.
          error += Math.abs((record.residual[channel] ?? NaN) - (before.residual[channel] ?? NaN));
        }
      }
      transitions.push({ from: offset - 0.125, to: offset, pixels, error, flux });
      expect(pixels).toBeGreaterThan(9000);
      expect(error).toBeLessThan(0.001 * flux);
    }
    previous = current;
  }
  const body = JSON.stringify(
    {
      width: 128,
      height: 96,
      time: 137,
      displacement: "[offset, offset / 2] pixels",
      relativeLimit: 0.001,
      frames,
      transitions,
    },
    null,
    2,
  );
  await server.commands.writeFile("test-results/disk-motion.json", body);
  await annotate("Frozen-source subpixel motion", {
    body,
    bodyEncoding: "utf-8",
    contentType: "application/json",
  });
});

test("exhausted boundary queues explicitly retain unresolved pixels", async () => {
  const device = await requestDevice();
  using owned = new DisposableStack();
  owned.defer(() => device.destroy());
  device.pushErrorScope("validation");
  const texture = (format: GPUTextureFormat, usage: GPUTextureUsageFlags) =>
    owned.adopt(device.createTexture({ size: [4, 4], format, usage }), (value) => value.destroy());
  const buffer = (size: number, usage: GPUBufferUsageFlags) =>
    owned.adopt(device.createBuffer({ size, usage }), (value) => value.destroy());
  const uniform = buffer(96, GPUBufferUsage.UNIFORM);
  const times = texture("r32float", GPUTextureUsage.TEXTURE_BINDING);
  const map = texture("rgba32float", GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST);
  const indices = texture("r32uint", GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC);
  const candidates = buffer(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const queue = buffer(8, GPUBufferUsage.STORAGE);
  const readback = buffer(1040, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const data = new Float32Array(64);
  for (let pixel = 0; pixel < 16; pixel++) {
    data[pixel * 4 + 3] = pixel % 2 === 0 ? 0 : -3;
  }
  device.queue.writeTexture({ texture: map }, data, { bytesPerRow: 64 }, [4, 4]);
  const module = await compileShader(device, opticalSources.edge, "boundary queue exhaustion");
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "find_edges" },
  });
  const bindings = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 7, resource: times.createView() },
      { binding: 1, resource: map.createView() },
      { binding: 2, resource: indices.createView() },
      { binding: 3, resource: { buffer: candidates } },
      { binding: 4, resource: { buffer: queue } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindings);
  pass.dispatchWorkgroups(1, 1);
  pass.end();
  encoder.copyTextureToBuffer({ texture: indices }, { buffer: readback, bytesPerRow: 256 }, [4, 4]);
  encoder.copyBufferToBuffer(candidates, 0, readback, 1024, 16);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const result = new Uint32Array(readback.getMappedRange());
  expect(result[256]).toBe(16);
  let sampled = 0;
  let unresolved = 0;
  for (let y = 0; y < 4; y++) {
    for (let x = 0; x < 4; x++) {
      const index = result[y * 64 + x];
      if (index === 1) {
        sampled++;
      } else if (index === 0xffffffff) {
        unresolved++;
      }
    }
  }
  expect(sampled).toBe(1);
  expect(unresolved).toBe(15);
  readback.unmap();
  expect(await device.popErrorScope()).toBeNull();
});
