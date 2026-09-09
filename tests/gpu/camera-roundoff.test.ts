import { expect, test } from "vitest";
import { server } from "vitest/browser";
import originalFixture from "../fixtures/camera-roundoff.json" with { type: "json" };
import { opticalSources } from "../../src/render/optics.ts";
import { compileShader, requestDevice } from "../../src/render/device.ts";
import { photonFromLocal } from "../../src/physics/photon.ts";
import { normalize } from "../../src/physics/vector.ts";
import { traceVisibility } from "../reference/visibility.ts";
import { evaluateGeodesic } from "../reference/geodesic.ts";
import { integrateTransport } from "../reference/transport.ts";

function skyReference(
  fixture: Pick<typeof originalFixture, "scene" | "width" | "height">,
  x: number,
  y: number,
) {
  const { space, observer, diskInner, diskOuter } = fixture.scene;
  const inclination = Math.fround(observer.inclination);
  const scale = Math.fround(2 * Math.tan(observer.fieldOfView / 2));
  const source = normalize([
    -1,
    ((y + 0.5 - fixture.height / 2) / fixture.height) * scale,
    ((x + 0.5 - fixture.width / 2) / fixture.height) * scale,
  ]);
  const photon = photonFromLocal(space, observer.radius, inclination, source);
  const { path, outcome } = traceVisibility(space, photon, observer.radius, inclination, {
    inner: diskInner,
    outer: diskOuter,
  });
  if (outcome.kind !== "sky") {
    return undefined;
  }
  const transport = integrateTransport(space, photon, path, outcome.time, false, 1e-11);
  const mu = evaluateGeodesic(path, outcome.time)?.cosineTheta;
  if (transport.kind !== "resolved" || mu === undefined) {
    return undefined;
  }
  const radius = Math.sqrt(1 - mu * mu);
  return [radius * Math.cos(transport.azimuth), radius * Math.sin(transport.azimuth), mu];
}

const endpointDirection = (sample: Float32Array) => {
  const mu = sample[0] ?? NaN;
  const phi = sample[1] ?? NaN;
  const radius = Math.sqrt((1 - mu) * (1 + mu));
  return [radius * Math.cos(phi), radius * Math.sin(phi), mu];
};

const cases = [
  { name: "camera roundoff", fixture: originalFixture, expectedPositions: 68, referenceScale: 1 },
  {
    name: "small viewport",
    fixture: {
      width: 128,
      height: 72,
      scene: originalFixture.scene,
      coverageChanges: [
        { x: 60, y: 21 },
        { x: 61, y: 20 },
      ],
    },
    expectedPositions: 20,
    referenceScale: 1,
  },
  {
    name: "long phase",
    fixture: {
      width: 128,
      height: 72,
      scene: originalFixture.scene,
      coverageChanges: [
        { x: 64, y: 19 },
        { x: 66, y: 19 },
        { x: 56, y: 24 },
      ],
    },
    expectedPositions: 17,
    referenceScale: 1 / 8,
  },
];

test.for(cases)(
  "$name retains the affected subpixel derivative evidence",
  async ({ fixture, expectedPositions, name, referenceScale }, { annotate }) => {
    const reference = (x: number, y: number) => skyReference(fixture, x, y);
    const device = await requestDevice();
    using owned = new DisposableStack();
    owned.defer(() => device.destroy());
    device.pushErrorScope("validation");
    const buffer = (size: number, usage: GPUBufferUsageFlags) =>
      owned.adopt(device.createBuffer({ size, usage }), (value) => value.destroy());
    const centers = fixture.coverageChanges.flatMap(({ x, y }) =>
      Array.from({ length: 16 }, (_, i) => [
        x + ((i % 4) + 0.5) / 4 - 0.5,
        y + (Math.floor(i / 4) + 0.5) / 4 - 0.5,
        0,
        0,
      ]),
    );
    // Nearby positions test derivative accuracy beyond the minimized failures.
    for (const { x, y } of [
      ...fixture.coverageChanges,
      ...(name === "camera roundoff" ? [{ x: 685, y: 191 }] : []),
    ]) {
      for (let row = -4; row <= 4; row++) {
        for (let column = -4; column <= 4; column++) {
          const center = [x + 0.125 + column / 4, y + 0.125 + row / 4, 0, 0];
          if (!centers.some((value) => value[0] === center[0] && value[1] === center[1])) {
            centers.push(center);
          }
        }
      }
    }
    const positions = buffer(centers.length * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(positions, 0, new Float32Array(centers.flat()));
    const { space, observer, diskInner, diskOuter } = fixture.scene;
    const uniform = buffer(96, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(
      uniform,
      0,
      new Float32Array([
        fixture.width,
        fixture.height,
        0,
        0,
        observer.radius,
        observer.inclination,
        2 * Math.tan(observer.fieldOfView / 2),
        observer.azimuth,
        space.spin,
        space.charge,
        diskInner,
        diskOuter,
        -1,
        0,
        0,
        0,
        0,
        -1,
        0,
        0,
        0,
        0,
        1,
        0,
      ]),
    );
    const samples = buffer(
      centers.length * 33 * 2 * 16,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );
    const gradients = buffer(
      centers.length * 2 * 16,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );
    const readback = buffer(
      samples.size + gradients.size,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    );
    const module = await compileShader(
      device,
      `${opticalSources.edgeBeam}
    @group(0) @binding(9) var<storage, read> positions: array<vec4f>;
    @group(0) @binding(10) var<storage, read_write> samples: array<vec4f>;
    @group(0) @binding(11) var<storage, read_write> gradients: array<vec4f>;
    @compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= arrayLength(&positions) * 33u) { return; }
      let index = id.x / 33u;
      let sample = id.x % 33u;
      var pixel = positions[index].xy;
      if (sample > 0u) {
        let offset = sample - 1u;
        let step = exp2(-f32((offset % 16u) / 2u)) * select(-1.0, 1.0, offset % 2u == 1u);
        pixel += select(vec2f(step, 0.0), vec2f(0.0, step), offset >= 16u);
      }
      samples[2u * id.x] = screen_ray(frame, pixel).value;
      let screen = (pixel + 0.5 - frame.viewport.xy / 2.0) / frame.viewport.y;
      samples[2u * id.x + 1u] = trace_ray(frame, normalize(vec3f(-1.0, screen.y * frame.observer.z, screen.x * frame.observer.z))).value;
    }
    @compute @workgroup_size(32) fn summarize(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= arrayLength(&positions)) { return; }
      let endpoint = samples[id.x * 66u];
      if (endpoint.w <= 0.0) { return; }
      let beam = differential_beam(positions[id.x].xy, endpoint.w);
      gradients[id.x * 2u] = beam.dx;
      gradients[id.x * 2u + 1u] = beam.dy;
    }`,
      "camera-roundoff-probe",
    );
    const pipeline = (entryPoint: string) =>
      device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint } });
    const probe = await pipeline("probe");
    const summarize = await pipeline("summarize");
    const encoder = device.createCommandEncoder();
    const run = (selected: GPUComputePipeline, entries: GPUBindGroupEntry[], count: number) => {
      const pass = encoder.beginComputePass();
      pass.setPipeline(selected);
      pass.setBindGroup(
        0,
        device.createBindGroup({ layout: selected.getBindGroupLayout(0), entries }),
      );
      pass.dispatchWorkgroups(count);
      pass.end();
    };
    run(
      probe,
      [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 9, resource: { buffer: positions } },
        { binding: 10, resource: { buffer: samples } },
      ],
      Math.ceil((centers.length * 33) / 64),
    );
    run(
      summarize,
      [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 9, resource: { buffer: positions } },
        { binding: 10, resource: { buffer: samples } },
        { binding: 11, resource: { buffer: gradients } },
      ],
      Math.ceil(centers.length / 32),
    );
    encoder.copyBufferToBuffer(samples, 0, readback, 0, samples.size);
    encoder.copyBufferToBuffer(gradients, 0, readback, samples.size, gradients.size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const mapped = readback.getMappedRange();
    const endpoints = new Float32Array(mapped, 0, samples.size / 4).slice();
    const values = new Float32Array(mapped, samples.size, gradients.size / 4).slice();
    readback.unmap();
    const records = centers.flatMap(([x = NaN, y = NaN], index) => {
      if ((endpoints[index * 264 + 3] ?? 0) <= 0) {
        return [];
      }
      const axes = [0, 1].map((axis) => {
        const estimates = [1 / 512, 1 / 1024].map((referenceStep) => {
          const step = (referenceStep * referenceScale * fixture.height) / 720;
          const before = reference(x - (axis === 0 ? step : 0), y - (axis === 1 ? step : 0));
          const after = reference(x + (axis === 0 ? step : 0), y + (axis === 1 ? step : 0));
          return before && after
            ? after.map((value, i) => (value - (before[i] ?? NaN)) / (2 * step))
            : undefined;
        });
        const fine = estimates[1];
        const coarseReference = estimates[0];
        const scale = fine ? Math.hypot(...fine) : NaN;
        const error = (value: readonly number[]) =>
          fine
            ? Math.hypot(...value.map((component, i) => component - (fine[i] ?? NaN))) / scale
            : null;
        const current = Array.from(values.subarray(index * 8 + axis * 4, index * 8 + axis * 4 + 4));
        const center = endpoints.subarray(index * 264, index * 264 + 4);
        const scales = Array.from({ length: 8 }, (_, level) => {
          const offset = (index * 33 + 1 + axis * 16 + level * 2) * 8;
          const before = endpoints.subarray(offset, offset + 4);
          const after = endpoints.subarray(offset + 8, offset + 12);
          const left = before[3] === center[3];
          const right = after[3] === center[3];
          const step = 1 / 2 ** level;
          const a = endpointDirection(right ? after : center);
          const b = endpointDirection(left ? before : center);
          const gradient = a.map(
            (component, i) => (component - (b[i] ?? NaN)) / (step * (left && right ? 2 : 1)),
          );
          const beforeReference = reference(
            x - (left && axis === 0 ? step : 0),
            y - (left && axis === 1 ? step : 0),
          );
          const afterReference = reference(
            x + (right && axis === 0 ? step : 0),
            y + (right && axis === 1 ? step : 0),
          );
          const sameStep =
            (left || right) && beforeReference && afterReference
              ? afterReference.map(
                  (component, i) =>
                    (component - (beforeReference[i] ?? NaN)) / (step * (left && right ? 2 : 1)),
                )
              : undefined;
          return {
            step,
            referenceAtScale: sameStep ?? null,
            truncationError: sameStep ? error(sameStep) : null,
            endpointError:
              sameStep && scale > 0
                ? Math.hypot(...gradient.map((component, i) => component - (sameStep[i] ?? NaN))) /
                  scale
                : null,
            stencil: left && right ? "central" : left ? "left" : right ? "right" : "none",
            gradient,
            error: left || right ? error(gradient) : null,
          };
        });
        return {
          axis,
          scales,
          current,
          reference: fine ?? null,
          referenceChange: coarseReference ? error(coarseReference) : null,
          currentError: current[3] === 1 ? error(current.slice(0, 3)) : null,
        };
      });
      return [{ x, y, branch: endpoints[index * 264 + 3], axes }];
    });
    if (name === "long phase") {
      const endpointsAt = (x: number, y: number) => {
        const index = centers.findIndex((center) => center[0] === x && center[1] === y);
        expect(index).toBeGreaterThanOrEqual(0);
        return endpoints.subarray(index * 264, index * 264 + 4);
      };
      // Expanded disk transport budgets failed independent radius/time checks.
      // Their original failure status remains explicit until that precision issue is resolved.
      expect(Array.from(endpointsAt(64.375, 19.125))).toEqual([8, 0, 0, -2]);
      expect(Array.from(endpointsAt(65.625, 18.875))).toEqual([8, 0, 0, -2]);
      const current = endpointsAt(56.375, 23.875);
      const expected = reference(56.375, 23.875);
      if (!expected) {
        throw new Error("Unresolved long-phase sky reference");
      }
      const vectorError = Math.hypot(
        ...endpointDirection(current).map((value, axis) => value - (expected[axis] ?? NaN)),
      );
      expect(current[3]).toBeGreaterThan(0);
      expect(vectorError).toBeLessThan(5e-5);
      await server.commands.writeFile(
        "test-results/long-phase-sky.json",
        JSON.stringify(
          {
            pixel: [56.375, 23.875],
            endpoint: Array.from(current),
            referenceDirection: expected,
            vectorError,
          },
          null,
          2,
        ),
      );
    }
    const body = JSON.stringify(records, null, 2);
    await server.commands.writeFile(
      `test-results/${name === "camera roundoff" ? "camera-roundoff" : name === "small viewport" ? "small-viewport-derivatives" : "long-phase-derivatives"}.json`,
      body,
    );
    await annotate("Subpixel beam references", {
      body,
      bodyEncoding: "utf-8",
      contentType: "application/json",
    });
    for (const record of records) {
      for (const axis of record.axes) {
        expect(axis.referenceChange ?? NaN).toBeLessThan(1e-4);
        expect(axis.current[3]).toBe(1);
        expect(axis.currentError ?? NaN).toBeLessThan(0.002);
      }
    }
    expect(records).toHaveLength(expectedPositions);
    expect(endpoints.every(Number.isFinite)).toBe(true);
    expect(values.every(Number.isFinite)).toBe(true);
    expect(await device.popErrorScope()).toBeNull();
  },
);
