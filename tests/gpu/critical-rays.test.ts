import { tracePixels } from "./visibility-probe.ts";
import { referenceSkyGradient } from "../reference/sky-derivative.ts";
import { createTestFrame } from "../support/frames.ts";
import { describe, expect } from "vitest";
import { server } from "vitest/browser";
import originalFixture from "../fixtures/camera-roundoff.json" with { type: "json" };
import { opticalSources, createOptics } from "../../src/gpu/optics.ts";
import { compileShader } from "../../src/gpu/device.ts";
import { integrateTransport } from "../reference/transport.ts";
import { observerFrameBytes } from "../../src/gpu/observer.ts";
import { test, computeReadback, observerData } from "./compute.ts";
import { referenceRay, referenceSky, referenceEndpoint } from "../reference/sky.ts";
import boundaryCoverage from "../fixtures/boundary-coverage.json" with { type: "json" };
import criticalPrimaryRaysFixture from "../fixtures/critical-primary-rays.json" with { type: "json" };
import { createScene } from "../../src/scene/scene.ts";

describe("Retained camera subpixels", () => {
  test.for([
    {
      name: "camera roundoff",
      width: 1280,
      height: 720,
      centers: [...originalFixture.coverageChanges, { x: 685, y: 191 }],
      step: 1 / 1024,
    },
    {
      name: "small viewport",
      width: 128,
      height: 72,
      centers: [
        { x: 60, y: 21 },
        { x: 61, y: 20 },
      ],
      step: 1 / 10240,
    },
    {
      name: "long phase",
      width: 128,
      height: 72,
      centers: [
        { x: 64, y: 19 },
        { x: 66, y: 19 },
        { x: 56, y: 24 },
      ],
      step: 1 / 81920,
    },
  ])(
    "$name resolves production sky derivatives against converged references",
    async ({ name, width, height, centers, step }) => {
      const frame = createTestFrame(originalFixture.frame);
      frame[0] = width;
      frame[1] = height;
      const positions = new Map<string, readonly [number, number]>();
      for (const { x, y } of centers) {
        for (let row = -4; row <= 4; row++) {
          for (let column = -4; column <= 4; column++) {
            const pixel = [x + 0.125 + column / 4, y + 0.125 + row / 4] as const;
            positions.set(pixel.join(","), pixel);
          }
        }
        for (let i = 0; i < 16; i++) {
          const pixel = [
            x + ((i % 4) + 0.5) / 4 - 0.5,
            y + (Math.floor(i / 4) + 0.5) / 4 - 0.5,
          ] as const;
          positions.set(pixel.join(","), pixel);
        }
      }
      const pixels = [...positions.values()].filter(
        ([x, y]) => referenceRay(x, y, frame).outcome.kind === "sky",
      );
      expect(pixels.length).toBeGreaterThan(0);
      const values = await computeReadback(
        `${opticalSources.beam}
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
@group(0) @binding(2) var<storage, read> pixels: array<vec2f>;
@group(0) @binding(3) var<storage, read> observer: Observer64;
var<workgroup> prepared: PreparedSkyBeam;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) id: vec3u, @builtin(local_invocation_index) lane: u32) {
  if (lane == 0u) {
    let endpoint = screen_ray_refined(frame, pixels[id.x], observer).value;
    outputs[3u * id.x + 2u] = endpoint;
    prepared = prepare_sky_beam(frame, pixels[id.x], endpoint.w, observer);
  }
  let path = workgroupUniformLoad(&prepared);
  let beam = beam_from_prepared_sky(evaluate_prepared_sky(frame, path, lane), path);
  if (lane == 0u) { outputs[3u * id.x] = beam.dx; outputs[3u * id.x + 1u] = beam.dy; }
}`,
        frame,
        pixels.length * 12,
        pixels.length,
        [new Float32Array(pixels.flat()), new Float32Array(observerData(frame).buffer)],
        "main",
      );
      const records = pixels.map(([x, y], index) => {
        const errors = [0, 1].map((axis) => {
          const fine = referenceSkyGradient(x, y, axis, step, frame);
          const coarse = referenceSkyGradient(x, y, axis, 2 * step, frame);
          const actual = values.slice(index * 12 + axis * 4, index * 12 + axis * 4 + 3);
          const scale = Math.hypot(...fine);
          return {
            referenceChange:
              Math.hypot(...fine.map((value, i) => value - (coarse[i] ?? NaN))) / scale,
            error: Math.hypot(...actual.map((value, i) => value - (fine[i] ?? NaN))) / scale,
            resolved: values[index * 12 + axis * 4 + 3],
          };
        });
        return { pixel: [x, y], tag: values[index * 12 + 11], errors };
      });
      await server.commands.writeFile(
        `test-results/camera-${name.replaceAll(" ", "-")}.json`,
        JSON.stringify(records, null, 2),
      );
      for (const record of records) {
        const label = JSON.stringify(record);
        expect(record.tag, label).toBeGreaterThan(0);
        for (const axis of record.errors) {
          expect(axis.resolved, label).toBe(1);
          expect(axis.referenceChange, label).toBeLessThan(1e-4);
          expect(axis.error, label).toBeLessThan(0.002);
        }
      }
    },
  );

  test("long scattering disk hits retain radius, azimuth and emission delay", async () => {
    const frame = createTestFrame(originalFixture.frame);
    frame[0] = 128;
    frame[1] = 72;
    const pixels = [
      [64.375, 19.125],
      [65.625, 18.875],
    ] as const;
    const values = await tracePixels(frame, pixels);
    for (const [index, [x, y]] of pixels.entries()) {
      const expected = referenceEndpoint(x, y, frame);
      if (expected.kind !== "disk") {
        throw new Error("The retained ray must hit the disk.");
      }
      const at = (offset: number) => values[index * 8 + offset] ?? NaN;
      const label = `${x},${y}: ${JSON.stringify(Array.from(values.slice(index * 8, index * 8 + 8)))}`;
      expect(at(3), label).toBe(expected.tag);
      expect(Math.abs(1 / at(0) - 1 / expected.radius), label).toBeLessThan(
        2e-6 + 2e-5 / expected.radius,
      );
      expect(Math.abs(at(1) - expected.azimuth), label).toBeLessThan(5e-5);
      expect(Math.abs(at(4) - expected.delay) / (1 + Math.abs(expected.delay)), label).toBeLessThan(
        5e-5,
      );
    }
  });
});

describe("Boundary precision", () => {
  const fixture = boundaryCoverage;
  test("retained boundary samples resolve their independently classified destinations", async ({
    device,
  }) => {
    using owned = new DisposableStack();

    device.pushErrorScope("validation");
    const buffer = (size: number, usage: GPUBufferUsageFlags) =>
      owned.adopt(device.createBuffer({ size, usage }), (value) => value.destroy());
    const texture = (format: GPUTextureFormat) =>
      owned.adopt(
        device.createTexture({
          size: [4, fixture.pixels.length * 4],
          format,
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
        }),
        (value) => value.destroy(),
      );
    const frame = createTestFrame(fixture.frame);
    frame[0] = fixture.width;
    frame[1] = fixture.height;
    const uniform = buffer(96, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(uniform, 0, frame);
    const observers = buffer(observerFrameBytes, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(observers, 0, observerData(frame));
    const pixels = buffer(
      fixture.pixels.length * 8,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );
    device.queue.writeBuffer(pixels, 0, new Uint32Array(fixture.pixels.flat()));
    const count = buffer(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(count, 0, new Uint32Array([fixture.pixels.length, 0, 0, 0]));
    const endpoints = texture("rgba32float");
    const times = texture("r32float");
    const staging = buffer(
      fixture.pixels.length * 8 * 256,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    );
    const module = await compileShader(device, opticalSources.edge, "retained-boundary-production");
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "sample_edges" },
    });
    const bindings = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 3, resource: { buffer: count } },
        { binding: 4, resource: { buffer: pixels } },
        { binding: 5, resource: endpoints.createView() },
        { binding: 6, resource: times.createView() },
        { binding: 8, resource: { buffer: observers } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.dispatchWorkgroups(Math.ceil((fixture.pixels.length * 16) / 64));
    pass.end();
    encoder.copyTextureToBuffer({ texture: endpoints }, { buffer: staging, bytesPerRow: 256 }, [
      4,
      fixture.pixels.length * 4,
    ]);
    encoder.copyTextureToBuffer(
      { texture: times },
      { buffer: staging, offset: fixture.pixels.length * 4 * 256, bytesPerRow: 256 },
      [4, fixture.pixels.length * 4],
    );
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();
    const failures = [];
    const accuracyFailures = [];
    const stages: Record<string, number> = {};
    for (const [index, [x = NaN, y = NaN]] of fixture.pixels.entries()) {
      for (let sample = 0; sample < 16; sample++) {
        const sx = x + ((sample % 4) + 0.5) / 4 - 0.5;
        const sy = y + (Math.floor(sample / 4) + 0.5) / 4 - 0.5;
        const offset = (index * 4 + Math.floor(sample / 4)) * 64 + (sample % 4) * 4;
        const endpoint = Array.from(values.subarray(offset, offset + 4));
        const ray = referenceRay(sx, sy, frame);
        const reference = ray.outcome;
        const phase =
          reference.kind === "unresolved"
            ? NaN
            : reference.time *
              Math.sqrt(
                ray.photon.angularMomentum ** 2 +
                  2 * Math.abs(ray.photon.carter) +
                  2 * (frame[8] ?? NaN) ** 2 * ray.photon.energy ** 2,
              );
        const status = endpoint[3] ?? NaN;
        const actual =
          status === -2
            ? "unresolved"
            : status <= -3
              ? "disk"
              : status > 0
                ? "sky"
                : status === 0
                  ? "captured"
                  : "invalid";
        if (reference.kind === "sky" && actual === "sky") {
          const expected = referenceSky(sx, sy, frame).direction;
          const mu = endpoint[0] ?? NaN,
            phi = endpoint[1] ?? NaN;
          const radius = Math.sqrt(1 - mu * mu);
          const error = Math.hypot(
            radius * Math.cos(phi) - expected[0],
            radius * Math.sin(phi) - expected[1],
            mu - expected[2],
          );
          if (!(error <= 5e-5)) {
            accuracyFailures.push({ screen: [sx, sy], skyVectorError: error });
          }
        }
        if (reference.kind === "disk" && actual === "disk") {
          const transport = integrateTransport(
            ray.space,
            ray.photon,
            ray.path,
            reference.time,
            true,
            1e-8,
          );
          if (transport.kind !== "resolved") {
            throw new Error(
              `Unresolved disk transport reference at ${sx},${sy}: ${JSON.stringify(transport)}`,
            );
          }
          const coarse = integrateTransport(
            ray.space,
            ray.photon,
            ray.path,
            reference.time,
            true,
            2e-8,
          );
          if (coarse.kind !== "resolved") {
            throw new Error("Unresolved coarse disk transport reference");
          }
          expect
            .soft(Math.abs(transport.azimuth - coarse.azimuth), "reference azimuth convergence")
            .toBeLessThan(1e-7);
          expect
            .soft(
              Math.abs(transport.coordinateTime - coarse.coordinateTime) /
                (1 + Math.abs(transport.coordinateTime)),
              "reference time convergence",
            )
            .toBeLessThan(1e-7);
          const radiusError = Math.abs(1 / (endpoint[0] ?? NaN) - 1 / reference.radius);
          const angle = (endpoint[1] ?? NaN) - ray.launchAzimuth - transport.azimuth;
          const azimuthError = Math.abs(Math.atan2(Math.sin(angle), Math.cos(angle)));
          const gpuTime =
            values[
              fixture.pixels.length * 256 + (index * 4 + Math.floor(sample / 4)) * 64 + (sample % 4)
            ] ?? NaN;
          const timeError =
            Math.abs(gpuTime - transport.coordinateTime) / (1 + Math.abs(transport.coordinateTime));
          if (
            radiusError > 2e-6 + 2e-5 / reference.radius ||
            azimuthError > 5e-5 ||
            timeError > 5e-5
          ) {
            accuracyFailures.push({
              screen: [sx, sy],
              radiusError,
              azimuthError,
              timeError,
              reference,
              endpoint,
            });
          }
        }
        if (actual === "unresolved") {
          const stage = String(endpoint[0]);
          stages[stage] = (stages[stage] ?? 0) + 1;
        }
        if (actual !== reference.kind || reference.kind === "unresolved") {
          failures.push({ pixel: [x, y], sample, screen: [sx, sy], endpoint, reference, phase });
        }
      }
    }
    await server.commands.writeFile(
      "test-results/boundary-precision-acceptance.json",
      JSON.stringify({ stages, failures, accuracyFailures }, null, 2),
    );
    expect(await device.popErrorScope()).toBeNull();
    expect.soft(accuracyFailures.length, JSON.stringify(accuracyFailures.slice(0, 4))).toBe(0);
    expect(failures.length, JSON.stringify(failures.slice(0, 4))).toBe(0);
  });
});

describe("Critical endpoints", () => {
  const fixture = criticalPrimaryRaysFixture;
  test("critical primary endpoints meet optical accuracy targets", async ({ device, annotate }) => {
    using owned = new DisposableStack();

    device.pushErrorScope("validation");
    const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
    const staging = owned.adopt(
      device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      (value) => value.destroy(),
    );
    const scene = createScene({
      space: { spin: 0.7, charge: 0.2 },
      observer: { radius: 18, inclination: 1.2, azimuth: 0, fieldOfView: 2 * Math.atan(0.6) },
    });
    if (!scene.ok) {
      throw new Error(scene.error);
    }
    const records = [];
    for (const sample of fixture.cases) {
      // Trace the full production frame: instrumented shader copies can change f32 rounding.
      const encoder = device.createCommandEncoder();
      const image = optics.encode(encoder, scene.value, sample.width, sample.height, {
        antialias: true,
      });
      encoder.copyTextureToBuffer(
        { texture: image.endpoints, origin: [sample.x, sample.y] },
        { buffer: staging, bytesPerRow: 256 },
        [1, 1],
      );
      device.queue.submit([encoder.finish()]);
      // The production target and staging buffer are reused after each completed readback.
      // oxlint-disable-next-line no-await-in-loop
      await staging.mapAsync(GPUMapMode.READ);
      const endpoint = new Float32Array(staging.getMappedRange(), 0, 4).slice();
      staging.unmap();
      const coordinate = endpoint[0] ?? NaN;
      const azimuth = endpoint[1] ?? NaN;
      const label = `${sample.width}×${sample.height} (${sample.x}, ${sample.y})`;
      const frame = createTestFrame(fixture.frame);
      frame[0] = sample.width;
      frame[1] = sample.height;
      const expected = referenceEndpoint(sample.x, sample.y, frame);
      if (expected.kind === "sky") {
        const transverse = Math.sqrt(1 - coordinate * coordinate);
        const error = Math.hypot(
          transverse * Math.cos(azimuth) - (expected.direction[0] ?? NaN),
          transverse * Math.sin(azimuth) - (expected.direction[1] ?? NaN),
          coordinate - (expected.direction[2] ?? NaN),
        );
        records.push({
          kind: "sky" as const,
          label,
          endpoint: Array.from(endpoint),
          skyVectorError: error,
        });
      } else if (expected.kind === "disk") {
        const inverseRadius = 1 / coordinate;
        const referenceInverseRadius = 1 / expected.radius;
        const radiusError = Math.abs(inverseRadius - referenceInverseRadius);
        const radiusTolerance = 2e-6 + 2e-5 * Math.abs(referenceInverseRadius);
        const difference = azimuth - expected.azimuth;
        const azimuthError = Math.abs(Math.atan2(Math.sin(difference), Math.cos(difference)));
        records.push({
          kind: "disk" as const,
          label,
          endpoint: Array.from(endpoint),
          inverseRadiusError: radiusError,
          inverseRadiusTolerance: radiusTolerance,
          azimuthError,
        });
      }
    }
    const validationError = await device.popErrorScope();
    const shaderHash = new Uint8Array(
      await crypto.subtle.digest(
        "SHA-256",
        new TextEncoder().encode(JSON.stringify(opticalSources)),
      ),
    ).toHex();
    const report = JSON.stringify(
      {
        browser: navigator.userAgent,
        adapter: {
          vendor: device.adapterInfo.vendor,
          architecture: device.adapterInfo.architecture,
          device: device.adapterInfo.device,
          description: device.adapterInfo.description,
        },
        shaderHash,
        records,
      },
      null,
      2,
    );
    await server.commands.writeFile("test-results/critical-primary-acceptance.json", report);
    await annotate("Critical primary accuracy", {
      body: report,
      bodyEncoding: "utf-8",
      contentType: "application/json",
    });
    expect(validationError).toBeNull();
    for (const record of records) {
      const { label, endpoint } = record;
      const tag = endpoint[3] ?? NaN;
      expect.soft(endpoint.every(Number.isFinite), label).toBe(true);
      if (record.kind === "sky") {
        expect.soft(tag, `${label}: sky destination`).toBeGreaterThan(0);
        expect.soft(record.skyVectorError, `${label}: sky direction vector`).toBeLessThan(5e-5);
      } else {
        expect.soft(tag, `${label}: disk destination`).toBeLessThanOrEqual(-3);
        expect.soft(endpoint[0], `${label}: disk radius`).toBeGreaterThan(0);
        expect
          .soft(record.inverseRadiusError, `${label}: inverse radius`)
          .toBeLessThan(record.inverseRadiusTolerance);
        expect.soft(record.azimuthError, `${label}: disk azimuth`).toBeLessThan(5e-5);
      }
    }
  }, 30000);
});
