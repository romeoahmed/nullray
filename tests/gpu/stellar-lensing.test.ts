import { stellarImageFrame } from "../support/frames.ts";
import criticalImages from "../fixtures/critical-stellar-images.json" with { type: "json" };
import { describe, expect } from "vitest";
import { server } from "vitest/browser";
import { createStellarSearch, stellarSearchSource } from "../../src/gpu/stellar-search.ts";
import { requestDevice, compileShader } from "../../src/gpu/device.ts";
import { createStarTree } from "../../src/physics/stars.ts";
import { criticalDirection } from "../../src/physics/critical.ts";
import { blackbodyXYZ } from "../../src/physics/radiation.ts";
import schwarzschildImages from "../fixtures/schwarzschild-stellar-images.json" with { type: "json" };
import { initialCamera, turnCamera } from "../../src/scene/camera.ts";
import type { Vec3 } from "../../src/physics/vector.ts";
import { projectStellarDirection, reprojectStellarImages } from "../reference/projection.ts";
import { referenceStellarImage } from "../reference/stellar-image.ts";
import { createStars } from "../../src/physics/sky.ts";
import { createOptics, opticalSources } from "../../src/gpu/optics.ts";
import { createScene } from "../../src/scene/scene.ts";
import { createAppearance } from "../../src/scene/appearance.ts";
import observer64 from "../../src/gpu/wgsl/geodesics/observer64.wgsl?raw";
import { referenceRay, referenceSky } from "../reference/sky.ts";
import soft64 from "../../src/gpu/wgsl/math/binary64.wgsl?raw";
import launch64 from "../../src/gpu/wgsl/geodesics/launch64.wgsl?raw";
import prepare64 from "../../src/gpu/wgsl/math/quartic64.wgsl?raw";
import trace64 from "../../src/gpu/wgsl/geodesics/trace64.wgsl?raw";
import { test, computeReadback, observerData } from "./compute.ts";

const quantized = (v: Vec3): Vec3 => [Math.fround(v[0]), Math.fround(v[1]), Math.fround(v[2])];

function luminosity(image: Float16Array | undefined, offset: number): number {
  return (
    0.2126390059 * (image?.[offset] ?? NaN) +
    0.7151686788 * (image?.[offset + 1] ?? NaN) +
    0.0721923154 * (image?.[offset + 2] ?? NaN)
  );
}

describe("Image search", () => {
  test("candidate totals beyond u32 preserve ordered allocation and expose overflow", async ({
    device,
  }) => {
    using owned = new DisposableStack();
    device.pushErrorScope("validation");
    const cells = 768,
      blocks = 3;
    const buffer = (size: number, usage: GPUBufferUsageFlags) =>
      owned.adopt(device.createBuffer({ size, usage }), (value) => value.destroy());
    const settings = buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(settings, 0, new Uint32Array([32, 24, 0, 0]));
    const counts = new Uint32Array((cells + blocks) * 2);
    for (let i = 0; i < cells; i++) {
      counts[i * 2] = i % 61 === 0 ? 600000000 : i % 7;
    }
    const offsets = buffer(
      counts.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    );
    device.queue.writeBuffer(offsets, 0, counts);
    const seeds = buffer(16 * 32, GPUBufferUsage.STORAGE);
    const statistics = buffer(32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const staging = buffer(
      counts.byteLength + 32,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    );
    const module = await compileShader(device, stellarSearchSource, "candidate count overflow");
    const encoder = device.createCommandEncoder();
    for (const [entryPoint, entries, groups] of [
      [
        "scan_stellar_seed_blocks",
        [
          { binding: 8, resource: { buffer: settings } },
          { binding: 16, resource: { buffer: offsets } },
        ],
        blocks,
      ],
      [
        "allocate_stellar_seed_blocks",
        [
          { binding: 6, resource: { buffer: seeds } },
          { binding: 7, resource: { buffer: statistics } },
          { binding: 8, resource: { buffer: settings } },
          { binding: 16, resource: { buffer: offsets } },
        ],
        1,
      ],
    ] as const) {
      // The second pass consumes the first pass's complete block sums.
      // oxlint-disable-next-line no-await-in-loop
      const pipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: { module, entryPoint },
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(
        0,
        device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [...entries] }),
      );
      pass.dispatchWorkgroups(groups);
      pass.end();
    }
    encoder.copyBufferToBuffer(offsets, 0, staging, 0, counts.byteLength);
    encoder.copyBufferToBuffer(statistics, 0, staging, counts.byteLength, 32);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const actual = new Uint32Array(staging.getMappedRange()).slice();
    staging.unmap();
    const value = (index: number) =>
      BigInt(actual[index * 2] ?? 0) + (BigInt(actual[index * 2 + 1] ?? 0) << 32n);
    let expected = 0n;
    for (let i = 0; i < cells; i++) {
      expect(value(i) + value(cells + Math.floor(i / 256)), `cell ${i}`).toBe(expected);
      expected += BigInt(counts[i * 2] ?? 0);
    }
    expect(actual[counts.length]).toBe(0xffffffff);
    expect(actual[counts.length + 5]).toBe(0xffffffff);
    expect(await device.popErrorScope()).toBeNull();
  });

  const fixture = schwarzschildImages;
  test.for([
    { name: "axis", camera: initialCamera },
    { name: "yaw-pitch", camera: turnCamera(initialCamera, 0.08, 0.04) },
    { name: "roll", camera: { forward: [-1, 0, 0], up: [0, -Math.cos(0.37), Math.sin(0.37)] } },
  ])(
    "critical-strip search finds each independent inner-gap image: $name",
    { timeout: 30000 },
    async ({ name, camera }, { device }) => {
      using owned = new DisposableStack();

      device.pushErrorScope("validation");
      const search = owned.adopt(
        await createStellarSearch(
          device,
          createStarTree([
            {
              direction: [
                fixture.source[0] ?? NaN,
                fixture.source[1] ?? NaN,
                fixture.source[2] ?? NaN,
              ],
              temperature: 6500,
              flux: 1e-5,
            },
          ]),
        ),
        (value) => value.dispose(),
      );
      const staging = owned.adopt(
        device.createBuffer({
          size: 32 + search.seeds.size + search.refined.size,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        (value) => value.destroy(),
      );
      const frame = stellarImageFrame(camera);
      const encoder = device.createCommandEncoder();
      search.encode(encoder, frame);
      encoder.copyBufferToBuffer(search.statistics, 0, staging, 0, 32);
      encoder.copyBufferToBuffer(search.seeds, 0, staging, 32, search.seeds.size);
      encoder.copyBufferToBuffer(
        search.refined,
        0,
        staging,
        32 + search.seeds.size,
        search.refined.size,
      );
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const mapped = staging.getMappedRange();
      const statistics = Array.from(new Uint32Array(mapped, 0, 8));
      const count = statistics[0] ?? NaN;
      const values = new Float32Array(
        mapped,
        32,
        Math.min(count, search.seeds.size / 32) * 8,
      ).slice();
      const images = new Float32Array(
        mapped,
        32 + search.seeds.size,
        Math.min(count, search.refined.size / 64) * 16,
      ).slice();
      staging.unmap();
      const candidates = Array.from({ length: values.length / 8 }, (_, index) => {
        const phase = values[index * 8] ?? NaN;
        const logarithm = values[index * 8 + 1] ?? NaN;
        const point = criticalDirection({ spin: 0, charge: 0 }, fixture.frame.radius, 0, phase);
        if (point.kind !== "point") {
          throw new Error("Candidate critical reconstruction failed");
        }
        const local: Vec3 = [
          point.direction[0] + Math.exp(logarithm),
          point.direction[1],
          point.direction[2],
        ];
        const { pixel } = projectStellarDirection(frame, local);
        return {
          phase,
          logarithm,
          branch: values[index * 8 + 3],
          seedPixel: pixel,
          pixel: [
            (images[index * 16] ?? NaN) + (images[index * 16 + 2] ?? NaN),
            (images[index * 16 + 1] ?? NaN) + (images[index * 16 + 3] ?? NaN),
          ],
          raw: Array.from(values.subarray(index * 8, index * 8 + 8)),
          refined: Array.from(images.subarray(index * 16, index * 16 + 16)),
        };
      });
      const records = reprojectStellarImages(frame, stellarImageFrame(), schwarzschildImages.cases)
        .filter((sample) => !sample.occulted)
        .map((sample) => {
          const matching = candidates.filter(
            (candidate) => candidate.branch === 2 * sample.order + 2,
          );
          const distance = (candidate: (typeof candidates)[number]) =>
            Math.hypot(
              (candidate.pixel[0] ?? NaN) - (sample.pixel[0] ?? NaN),
              (candidate.pixel[1] ?? NaN) - (sample.pixel[1] ?? NaN),
            );
          const closest = matching.toSorted(
            (first, second) => distance(first) - distance(second),
          )[0];
          const image = closest?.refined;
          const measured =
            0.2126390059 * (image?.[4] ?? NaN) +
            0.7151686788 * (image?.[5] ?? NaN) +
            0.0721923154 * (image?.[6] ?? NaN);
          const expected =
            (1e-5 * blackbodyXYZ(6500 / fixture.energy)[1]) /
            blackbodyXYZ(6500)[1] /
            sample.jacobian;
          return {
            order: sample.order,
            side: sample.side,
            distance: closest ? distance(closest) : Infinity,
            resolved: image?.[7],
            measured,
            expected,
            relativeFluxError: Math.abs(measured / expected - 1),
          };
        });
      await server.commands.writeFile(
        `test-results/stellar-search-${name}.json`,
        JSON.stringify(
          {
            browser: navigator.userAgent,
            shaderHash: new Uint8Array(
              await crypto.subtle.digest("SHA-256", new TextEncoder().encode(stellarSearchSource)),
            ).toHex(),
            frame: Array.from(frame),
            statistics,
            candidates,
            records,
          },
          null,
          2,
        ),
      );
      expect(await device.popErrorScope()).toBeNull();
      expect(statistics[5]).toBe(0);
      expect(count).toBe(4);
      for (const record of records) {
        expect.soft(record.distance, JSON.stringify(record)).toBeLessThan(0.001);
        expect.soft(record.resolved, JSON.stringify(record)).toBe(1);
        expect.soft(record.relativeFluxError, JSON.stringify(record)).toBeLessThan(0.002);
      }
    },
  );
});

describe("Deep critical images", () => {
  test.for(criticalImages.images)(
    "critical search recovers the retained Kerr-Newman stellar image: $name",
    async (fixture, { device }) => {
      const name = fixture.name;
      const frame = new Float32Array(criticalImages.frame);
      const source: Vec3 = [
        fixture.source[0] ?? NaN,
        fixture.source[1] ?? NaN,
        fixture.source[2] ?? NaN,
      ];
      const reference = referenceStellarImage(
        frame,
        source,
        [fixture.initialParameters[0] ?? NaN, fixture.initialParameters[1] ?? NaN],
        (phase, logarithm) => {
          // The capture curve is only a coordinate chart, checked independently in
          // critical-curve.test.ts; propagation and image-area references are binary64.
          const critical = criticalDirection(
            { spin: frame[8] ?? NaN, charge: frame[9] ?? NaN },
            frame[4] ?? NaN,
            frame[5] ?? NaN,
            phase,
          );
          if (critical.kind !== "point") {
            throw new Error("Invalid reference parameter chart");
          }
          return projectStellarDirection(frame, [
            critical.direction[0] + Math.exp(logarithm),
            critical.direction[1],
            critical.direction[2],
          ]).pixel;
        },
      );
      expect(reference.center.chord).toBeLessThan(1e-7);
      expect(reference.areaConvergence).toBeLessThan(1e-4);

      using owned = new DisposableStack();

      device.pushErrorScope("validation");
      // Preserve the quantized catalogue's physical source and spectral coefficient;
      // no candidate position, search cell, branch, or reference ray enters the GPU search.
      const search = owned.adopt(
        await createStellarSearch(
          device,
          new Float32Array([...source, -1, fixture.spectralCoefficient, fixture.temperature, 0, 0]),
        ),
        (value) => value.dispose(),
      );
      const staging = owned.adopt(
        device.createBuffer({
          size: 32 + search.refined.size,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        (value) => value.destroy(),
      );
      const encoder = device.createCommandEncoder();
      search.encode(encoder, frame);
      encoder.copyBufferToBuffer(search.statistics, 0, staging, 0, 32);
      encoder.copyBufferToBuffer(search.refined, 0, staging, 32, search.refined.size);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const mapped = staging.getMappedRange();
      const statistics = Array.from(new Uint32Array(mapped, 0, 8));
      const count = Math.min(statistics[0] ?? 0, search.refined.size / 64);
      const values = new Float32Array(mapped, 32, count * 16);
      const candidates = Array.from({ length: count }, (_, index) =>
        Array.from(values.subarray(index * 16, index * 16 + 16)),
      );
      staging.unmap();
      const error = await device.popErrorScope();
      expect(error?.message).toBeUndefined();
      const records = candidates
        .filter((candidate) => candidate[11] === fixture.provenance.branch)
        .map((candidate) => {
          const pixel = [
            (candidate[0] ?? NaN) + (candidate[2] ?? NaN),
            (candidate[1] ?? NaN) + (candidate[3] ?? NaN),
          ];
          const measured =
            0.2126390059 * (candidate[4] ?? NaN) +
            0.7151686788 * (candidate[5] ?? NaN) +
            0.0721923154 * (candidate[6] ?? NaN);
          const expected =
            (fixture.spectralCoefficient *
              blackbodyXYZ(fixture.temperature / reference.center.energy)[1]) /
            blackbodyXYZ(6500)[1] /
            reference.jacobian;
          return {
            candidate,
            pixel,
            resolved: candidate[7],
            distance: Math.hypot(
              (pixel[0] ?? NaN) - (reference.center.pixel[0] ?? NaN),
              (pixel[1] ?? NaN) - (reference.center.pixel[1] ?? NaN),
            ),
            measured,
            expected,
            relativeFluxError: Math.abs(measured / expected - 1),
          };
        });
      await server.commands.writeFile(
        `test-results/${name}-stellar-image.json`,
        JSON.stringify(
          {
            browser: navigator.userAgent,
            frame: Array.from(frame),
            source,
            reference,
            statistics,
            candidates,
            records,
            shaderHash: new Uint8Array(
              await crypto.subtle.digest("SHA-256", new TextEncoder().encode(stellarSearchSource)),
            ).toHex(),
          },
          null,
          2,
        ),
      );
      expect(statistics[5]).toBe(0);
      const matching = records.filter((record) => record.distance < 0.001);
      expect(matching).toHaveLength(1);
      for (const record of matching) {
        expect.soft(record.resolved).toBe(1);
        expect.soft(record.distance).toBeLessThan(0.001);
        expect.soft(record.relativeFluxError).toBeLessThan(0.002);
      }
    },
  );
});

describe("Repeated evaluation", () => {
  const fixture = criticalImages;
  /** Repeated devices exercise complete, overflowed, and partial-block candidate allocation. */
  test.for([
    { name: "complete", columns: 256, rows: 128, capacity: 32768 },
    { name: "overflow", columns: 256, rows: 128, capacity: 64 },
    { name: "partial-block", columns: 255, rows: 127, capacity: 64 },
  ])(
    "$name: identical stellar searches retain the same physical candidates",
    { timeout: 30000 },
    async ({ name, columns, rows, capacity }) => {
      const source = createStarTree(createStars());
      const frame = new Float32Array(fixture.frame);
      const measurements: { statistics: number[]; sha256: string }[] = [];
      // Searches must finish before the next device starts; concurrent GPU load is a different workload.
      // oxlint-disable no-await-in-loop
      for (let repeat = 0; repeat < 8; repeat++) {
        const device = await requestDevice();
        using owned = new DisposableStack();
        owned.defer(() => device.destroy());
        device.pushErrorScope("validation");
        const search = owned.adopt(
          await createStellarSearch(device, source, {
            columns,
            rows,
            capacity,
            minimumOffset: 1e-8,
            maximumOffset: 0.25,
          }),
          (value) => value.dispose(),
        );
        const staging = owned.adopt(
          device.createBuffer({
            size: 32 + search.seeds.size + search.refined.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          }),
          (value) => value.destroy(),
        );
        const encoder = device.createCommandEncoder();
        search.encode(encoder, frame);
        encoder.copyBufferToBuffer(search.statistics, 0, staging, 0, 32);
        encoder.copyBufferToBuffer(search.seeds, 0, staging, 32, search.seeds.size);
        encoder.copyBufferToBuffer(
          search.refined,
          0,
          staging,
          32 + search.seeds.size,
          search.refined.size,
        );
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const mapped = staging.getMappedRange();
        const statistics = Array.from(new Uint32Array(mapped, 0, 8));
        const total = statistics[0] ?? 0;
        const count = Math.min(total, capacity);
        expect(total).toBeGreaterThan(0);
        expect(statistics[5]).toBe(Math.max(0, total - capacity));
        if (name !== "complete") {
          expect(total).toBeGreaterThan(capacity);
        }
        const seeds = new Float32Array(mapped, 32, count * 8);
        const images = new Float32Array(mapped, 32 + search.seeds.size, count * 16);
        // Candidate order is now part of deterministic allocation, including overflow.
        const records = Array.from({ length: count }, (_, index) =>
          JSON.stringify([
            ...seeds.subarray(index * 8, index * 8 + 8),
            ...images.subarray(index * 16, index * 16 + 16),
          ]),
        );
        staging.unmap();
        const sha256 = new Uint8Array(
          await crypto.subtle.digest("SHA-256", new TextEncoder().encode(records.join("\n"))),
        ).toHex();
        measurements.push({ statistics, sha256 });
        expect((await device.popErrorScope())?.message).toBeUndefined();
      }
      // oxlint-enable no-await-in-loop
      await server.commands.writeFile(
        `test-results/stellar-repeatability${name === "complete" ? "" : `-${name}`}.json`,
        JSON.stringify(
          {
            browser: navigator.userAgent,
            shaderSHA256: new Uint8Array(
              await crypto.subtle.digest("SHA-256", new TextEncoder().encode(stellarSearchSource)),
            ).toHex(),
            measurements,
          },
          null,
          2,
        ),
      );
      for (const measurement of measurements) {
        expect(measurement).toEqual(measurements[0]);
      }
    },
  );
});

describe("Detector image flux", () => {
  const fixture = schwarzschildImages;

  /** Independent image locations, opacity, and Jacobians; no measured shader value sets the target. */
  test.for([
    {
      name: "yaw-pitch",
      antialias: true,
      jitter: [0, 0] as const,
      camera: turnCamera(initialCamera, 0.08, 0.04),
    },
    {
      name: "roll",
      antialias: false,
      jitter: [0.23, -0.17] as const,
      camera: { forward: [-1, 0, 0] as const, up: [0, -Math.cos(0.37), Math.sin(0.37)] as const },
    },
    { name: "boundary", antialias: true, jitter: [0, 0] as const },
    { name: "primary", antialias: false, jitter: [0, 0] as const },
    { name: "jitter-positive", antialias: true, jitter: [0.37, -0.23] as const },
    { name: "jitter-negative", antialias: false, jitter: [-0.49, 0.49] as const },
  ])(
    "production stellar filtering retains inner-gap image flux: $name",
    { timeout: 30000 },
    async ({ name, antialias, jitter, camera = initialCamera }, { device }) => {
      using owned = new DisposableStack();

      device.pushErrorScope("validation");
      const flux = 1e-5;
      const optics = owned.adopt(
        await createOptics(device, {
          celestial: {
            levels: [{ size: 1, data: new Float16Array(24) }],
            stars: createStarTree([
              {
                direction: [
                  fixture.source[0] ?? NaN,
                  fixture.source[1] ?? NaN,
                  fixture.source[2] ?? NaN,
                ],
                temperature: 6500,
                flux,
              },
            ]),
          },
        }),
        (value) => value.dispose(),
      );
      const { width, height, radius, zoom } = fixture.frame;
      const scene = createScene({
        space: { spin: 0, charge: 0 },
        camera,
        observer: {
          radius,
          inclination: 0,
          azimuth: 0,
          fieldOfView: 2 * Math.atan(zoom / 2),
        },
      });
      if (!scene.ok) {
        throw new Error(scene.error);
      }
      const forward = quantized(scene.value.camera.forward);
      const right = quantized(scene.value.camera.right);
      const up = quantized(scene.value.camera.up);
      const referenceImages = reprojectStellarImages(
        stellarImageFrame(scene.value.camera),
        stellarImageFrame(),
        schwarzschildImages.cases,
      );
      const staging = owned.adopt(
        device.createBuffer({
          size: width * height * 8 + 32,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        }),
        (value) => value.destroy(),
      );
      const images: Float16Array[] = [];
      const work: number[][] = [];
      for (const skyBrightness of [1, 0]) {
        const appearance = createAppearance({
          diskTemperature: 1000,
          diskStructure: 0,
          skyBrightness,
        });
        if (!appearance.ok) {
          throw new Error(appearance.error);
        }
        const encoder = device.createCommandEncoder();
        const image = optics.encode(encoder, scene.value, width, height, {
          appearance: appearance.value,
          reuse: skyBrightness === 1 ? "sources" : "geometry",
          antialias,
          jitter,
        });
        encoder.copyTextureToBuffer(
          { texture: image.radiance },
          { buffer: staging, bytesPerRow: width * 8 },
          [width, height],
        );
        encoder.copyBufferToBuffer(image.stellarStatistics, 0, staging, width * height * 8, 32);
        device.queue.submit([encoder.finish()]);
        // The second render subtracts disk emission without making the disk transparent.
        // oxlint-disable-next-line no-await-in-loop
        await staging.mapAsync(GPUMapMode.READ);
        const mapped = staging.getMappedRange();
        images.push(new Float16Array(mapped, 0, width * height * 4).slice());
        work.push(Array.from(new Uint32Array(mapped, width * height * 8, 8)));
        staging.unmap();
      }
      const spectralFlux = (flux * blackbodyXYZ(6500 / fixture.energy)[1]) / blackbodyXYZ(6500)[1];
      const records = [-1, 1].map((side) => {
        const cases = referenceImages.filter((sample) => !sample.occulted && sample.side === side);
        const center = (cases[0]?.pixel[0] ?? NaN) - jitter[0];
        const centerY = (cases[0]?.pixel[1] ?? NaN) - jitter[1];
        const pixels = [];
        let measured = 0,
          expected = 0;
        for (let y = Math.floor(centerY) - 1; y <= Math.floor(centerY) + 2; y++) {
          for (let x = Math.floor(center) - 1; x <= Math.floor(center) + 2; x++) {
            const offset = (y * width + x) * 4;
            const reference = cases.reduce(
              (sum, sample) =>
                sum +
                (spectralFlux / sample.jacobian) *
                  Math.max(0, 1 - Math.abs(x + jitter[0] - (sample.pixel[0] ?? NaN))) *
                  Math.max(0, 1 - Math.abs(y + jitter[1] - (sample.pixel[1] ?? NaN))),
              0,
            );
            const actual = luminosity(images[0], offset) - luminosity(images[1], offset);
            measured += actual;
            expected += reference;
            pixels.push({
              x,
              y,
              actual,
              expected: reference,
              resolved: images[0]?.[offset + 3],
              baselineResolved: images[1]?.[offset + 3],
            });
          }
        }
        return {
          side,
          measured,
          expected,
          relativeError: Math.abs(measured / expected - 1),
          pixels,
        };
      });
      const report = {
        browser: navigator.userAgent,
        adapter: {
          vendor: device.adapterInfo.vendor,
          architecture: device.adapterInfo.architecture,
          device: device.adapterInfo.device,
          description: device.adapterInfo.description,
        },
        frame: fixture.frame,
        camera: { forward, right, up },
        referenceImages,
        source: fixture.source,
        sampling: { antialias, jitter, boundarySamples: antialias ? 16 : 0 },
        work,
        sourceSHA256: new Uint8Array(
          await crypto.subtle.digest(
            "SHA-256",
            new TextEncoder().encode(JSON.stringify({ opticalSources, stellarSearchSource })),
          ),
        ).toHex(),
        fixture: "schwarzschild-stellar-images.json",
        flux,
        spectralFlux,
        records,
      };
      await server.commands.writeFile(
        `test-results/stellar-images-acceptance-${name}.json`,
        JSON.stringify(report, null, 2),
      );
      expect(await device.popErrorScope()).toBeNull();
      expect(work[1]).toEqual(work[0]);
      expect(work[0]?.[5]).toBe(0);
      expect(work[0]?.[6]).toBe(0);
      for (const record of records) {
        expect.soft(record.relativeError, JSON.stringify(record)).toBeLessThan(0.002);
        for (const pixel of record.pixels) {
          expect.soft(pixel.resolved).toBe(1);
          expect.soft(pixel.baselineResolved).toBe(1);
        }
      }
    },
  );
});

describe("Independent image rays", () => {
  const fixture = schwarzschildImages;
  test("quantized stellar image rays retain disk opacity and higher-order sky branches", async () => {
    // Integer base plus an uploaded fractional jitter avoids rounding the entire
    // absolute pixel to f32. CPU references use these same quantized inputs.
    const pixels = fixture.cases.map((sample) => sample.pixel.map(Math.floor));
    const frames = fixture.cases.map((sample, index) => {
      const frame = stellarImageFrame();
      frame[2] = (sample.pixel[0] ?? NaN) - (pixels[index]?.[0] ?? NaN);
      frame[3] = (sample.pixel[1] ?? NaN) - (pixels[index]?.[1] ?? NaN);
      return frame;
    });
    const observers = new Float64Array(frames.flatMap((frame) => Array.from(observerData(frame))));
    const output = await computeReadback(
      `${opticalSources.ray}\n${soft64}\n${observer64}\n${launch64}\n${prepare64}\n${trace64}
@group(0) @binding(0) var<storage, read> pixels: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
@group(0) @binding(2) var<storage, read> frames: array<Frame>;
@group(0) @binding(3) var<storage, read> observers: array<Observer64>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  output[id.x] = screen_ray_refined(frames[id.x],pixels[id.x],observers[id.x]).value;
}`,
      new Float32Array(pixels.flat()),
      fixture.cases.length * 4,
      fixture.cases.length,
      [
        new Float32Array(frames.flatMap((frame) => Array.from(frame))),
        new Float32Array(observers.buffer),
      ],
    );
    const records = fixture.cases.map((sample, index) => {
      const frame = frames[index];
      if (!frame) {
        throw new Error("Missing stellar image frame");
      }
      const x = pixels[index]?.[0] ?? NaN,
        y = pixels[index]?.[1] ?? NaN;
      const outcome = referenceRay(x, y, frame).outcome;
      const endpoint = Array.from(output.subarray(index * 4, index * 4 + 4));
      const mu = endpoint[0] ?? NaN,
        phi = endpoint[1] ?? NaN;
      if (outcome.kind === "disk") {
        return {
          order: sample.order,
          side: sample.side,
          endpoint,
          expectedTag: -3,
          error: Math.abs(1 / mu - 1 / outcome.radius),
          tolerance: 2e-6 + 2e-5 / outcome.radius,
        };
      }
      if (outcome.kind !== "sky" || sample.occulted) {
        throw new Error("Stellar image reference destination changed");
      }
      const sky = referenceSky(x, y, frame);
      const transverse = Math.sqrt(1 - mu * mu);
      return {
        order: sample.order,
        side: sample.side,
        endpoint,
        expectedTag: 2 * sample.order + 2,
        error: Math.hypot(
          transverse * Math.cos(phi) - sky.direction[0],
          transverse * Math.sin(phi) - sky.direction[1],
          mu - sky.direction[2],
        ),
        tolerance: 5e-5,
      };
    });
    await server.commands.writeFile(
      "test-results/stellar-image-rays.json",
      JSON.stringify(
        {
          browser: navigator.userAgent,
          shaderHash: new Uint8Array(
            await crypto.subtle.digest(
              "SHA-256",
              new TextEncoder().encode(
                [opticalSources.ray, soft64, observer64, launch64, prepare64, trace64].join("\n"),
              ),
            ),
          ).toHex(),
          pixels,
          frames: frames.map((frame) => Array.from(frame)),
          records,
        },
        null,
        2,
      ),
    );
    for (const record of records) {
      const label = JSON.stringify(record);
      expect(record.endpoint.every(Number.isFinite), label).toBe(true);
      expect(record.endpoint[3], label).toBe(record.expectedTag);
      expect(record.error, label).toBeLessThan(record.tolerance);
    }
  });
});
