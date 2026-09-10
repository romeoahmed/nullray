import { createTestFrame } from "../support/frames.ts";
import { describe, expect } from "vitest";
import { createOptics, opticalSources } from "../../src/gpu/optics.ts";
import { createScene, initialScene } from "../../src/scene/scene.ts";
import type { Scene } from "../../src/scene/scene.ts";
import { normalize } from "../../src/physics/vector.ts";
import type { Vec3 } from "../../src/physics/vector.ts";
import { traceDirections } from "./visibility-probe.ts";
import observer64 from "../../src/gpu/wgsl/geodesics/observer64.wgsl?raw";
import arithmetic from "../../src/gpu/wgsl/math/binary64.wgsl?raw";
import derivatives from "../../src/gpu/wgsl/math/dual64.wgsl?raw";
import launch from "../../src/gpu/wgsl/geodesics/launch-dual64.wgsl?raw";
import fixture from "../fixtures/critical-primary-rays.json" with { type: "json" };
import { referenceRay, referenceSky } from "../reference/sky.ts";
import { test, computeReadback, observerData } from "./compute.ts";
import { server } from "vitest/browser";
import { integrateTransport } from "../reference/transport.ts";
import { equatorCrossings } from "../reference/equator.ts";
import launch64 from "../../src/gpu/wgsl/geodesics/launch64.wgsl?raw";
import prepare64 from "../../src/gpu/wgsl/math/quartic64.wgsl?raw";
import trace64 from "../../src/gpu/wgsl/geodesics/trace64.wgsl?raw";

function constants(x: number, y: number, frame: Float32Array) {
  const { photon } = referenceRay(x, y, frame);
  return [
    photon.energy,
    photon.angularMomentum,
    photon.carter,
    photon.polarVelocity,
    photon.radialVelocity / (frame[4] ?? NaN) ** 2,
  ];
}

describe("Camera basis", () => {
  test("rotated raster rays retain event order and outward-looking endpoint geometry", async ({
    device,
  }) => {
    using owned = new DisposableStack();

    device.pushErrorScope("validation");
    const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
    const staging = owned.adopt(
      device.createBuffer({
        size: 32 * 32 * 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
      (value) => value.destroy(),
    );
    async function raster(scene: Scene) {
      const encoder = device.createCommandEncoder();
      const image = optics.encode(encoder, scene, 32, 32);
      encoder.copyTextureToBuffer(
        { texture: image.endpoints },
        { buffer: staging, bytesPerRow: 32 * 16 },
        [32, 32],
      );
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(staging.getMappedRange()).slice();
      staging.unmap();
      return result;
    }
    const rotated = createScene({
      ...initialScene,
      camera: { forward: [-1, 0, 0], up: [0, 1, 0] },
    });
    const outward = createScene({
      ...initialScene,
      camera: { forward: [1, 0, 0], up: [0, -1, 0] },
    });
    if (!rotated.ok || !outward.ok) {
      throw new Error("Invalid test cameras.");
    }
    const original = await raster(initialScene);
    const rolled = await raster(rotated.value);
    for (let pixel = 0; pixel < 1024; pixel++) {
      expect(rolled.subarray(pixel * 4, pixel * 4 + 4)).toEqual(
        original.subarray((1023 - pixel) * 4, (1024 - pixel) * 4),
      );
    }
    const result = await raster(outward.value);
    const rays: Vec3[] = [];
    const scale = Math.fround(2 * Math.tan(initialScene.observer.fieldOfView / 2));
    for (let y = 0; y < 32; y++) {
      for (let x = 0; x < 32; x++) {
        rays.push(
          normalize([
            1,
            Math.fround(((y - 15.5) / 32) * scale),
            -Math.fround(((x - 15.5) / 32) * scale),
          ]),
        );
      }
    }
    const expected = await traceDirections(rays, initialScene.observer.inclination, initialScene);
    let escaped = 0;
    for (let pixel = 0; pixel < 1024; pixel++) {
      const offset = pixel * 4;
      expect(result[offset + 3]).toBe(expected[offset + 3]);
      if ((result[offset + 3] ?? 0) > 0) {
        escaped++;
        expect(result[offset]).toBeCloseTo(expected[offset] ?? NaN, 4);
        expect(result[offset + 1]).toBeCloseTo(expected[offset + 1] ?? NaN, 4);
        expect(result[offset + 2]).toBeCloseTo(expected[offset + 2] ?? NaN, 6);
      }
    }
    expect(escaped).toBeGreaterThan(900);
    expect(await device.popErrorScope()).toBeNull();
  });
});

describe("Launch derivatives", () => {
  test("binary64 differentiated launch matches converged photon-constant references", async () => {
    const frames = fixture.cases.map(({ width, height }) => {
      const frame = createTestFrame();
      frame[0] = width;
      frame[1] = height;
      return frame;
    });
    const observers = new Float64Array(frames.flatMap((frame) => Array.from(observerData(frame))));
    const output = await computeReadback(
      `${arithmetic}\n${observer64}\n${derivatives}\n${launch}
@group(0) @binding(0) var<storage, read> pixels: array<vec2u>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2u>;
@group(0) @binding(2) var<storage, read> frames: array<array<u32,24>>;
@group(0) @binding(3) var<storage, read> observers: array<Observer64>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u) {
  let photon=d64_screen_launch(frames[id.x],pixels[id.x],observers[id.x]);
  let values=array<Dual64,5>(photon.energy,photon.angular_momentum,photon.carter,photon.polar_velocity,photon.inverse_radial_velocity);
  for (var i=0u;i<5u;i++) {
    output[id.x*15u+i*3u]=values[i].value;
    output[id.x*15u+i*3u+1u]=values[i].dx;
    output[id.x*15u+i*3u+2u]=values[i].dy;
  }
}`,
      new Float32Array(fixture.cases.flatMap(({ x, y }) => [x, y])),
      fixture.cases.length * 30,
      fixture.cases.length,
      [
        new Float32Array(frames.flatMap((frame) => Array.from(frame))),
        new Float32Array(observers.buffer),
      ],
    );
    const actual = new Float64Array(output.buffer);
    for (const [index, { x, y }] of fixture.cases.entries()) {
      const frame = frames[index];
      if (!frame) {
        throw new Error("Missing launch frame");
      }
      const expected = constants(x, y, frame);
      for (const [component, value] of expected.entries()) {
        expect(Math.abs((actual[index * 15 + component * 3] ?? NaN) - value)).toBeLessThan(
          2e-14 * Math.max(1, Math.abs(value)),
        );
      }
      for (const axis of [0, 1] as const) {
        const estimates = [0.005, 0.01].map((step) => {
          const before = constants(x - (axis === 0 ? step : 0), y - (axis === 1 ? step : 0), frame);
          const after = constants(x + (axis === 0 ? step : 0), y + (axis === 1 ? step : 0), frame);
          return after.map((value, component) => (value - (before[component] ?? NaN)) / (2 * step));
        });
        for (let component = 0; component < 5; component++) {
          const fine = estimates[0]?.[component] ?? NaN,
            coarse = estimates[1]?.[component] ?? NaN;
          const reference = (4 * fine - coarse) / 3;
          expect(Math.abs(fine - coarse), "reference convergence").toBeLessThan(1e-9);
          expect(
            Math.abs((actual[index * 15 + component * 3 + axis + 1] ?? NaN) - reference),
            `ray ${index}, constant ${component}, axis ${axis}`,
          ).toBeLessThan(1e-10);
        }
      }
    }
  });
});

describe("Precision escalation", () => {
  const soft64 = arithmetic;
  test("integer preparation and ordered f32 tracing meet the retained critical endpoint targets", async () => {
    const frames = fixture.cases.map(({ width, height }) => {
      const frame = createTestFrame();
      frame[0] = width;
      frame[1] = height;
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
  let endpoint = screen_ray64(frames[id.x],pixels[id.x],observers[id.x]);
  output[id.x*2u] = endpoint.value;
  output[id.x*2u+1u] = vec4f(endpoint.relative_time,0.0,0.0,0.0);
}`,
      new Float32Array(fixture.cases.flatMap(({ x, y }) => [x, y])),
      fixture.cases.length * 8,
      fixture.cases.length,
      [
        new Float32Array(frames.flatMap((frame) => Array.from(frame))),
        new Float32Array(observers.buffer),
      ],
    );
    const records = fixture.cases.map((sample, index) => {
      const frame = frames[index];
      if (!frame) {
        throw new Error("Missing critical frame");
      }
      const { space, photon, path, outcome } = referenceRay(sample.x, sample.y, frame);
      if (
        outcome.kind !== sample.outcome.kind ||
        (outcome.kind !== "disk" && outcome.kind !== "sky")
      ) {
        throw new Error("The retained reference destination changed");
      }
      const events = equatorCrossings(space, photon, frame[5] ?? NaN);
      if (events.kind !== "crossings") {
        throw new Error("Expected transverse reference crossings");
      }
      const order = Math.round((outcome.time - events.first) / events.spacing);
      const expectedTag =
        outcome.kind === "disk"
          ? -3 - order
          : 2 + Math.floor((outcome.time - events.first) / events.spacing);
      const endpoint = Array.from(output.subarray(index * 8, index * 8 + 5));
      const coordinate = endpoint[0] ?? NaN,
        phi = endpoint[1] ?? NaN;
      if (outcome.kind === "sky") {
        const expected = referenceSky(sample.x, sample.y, frame).direction;
        const transverse = Math.sqrt(1 - coordinate * coordinate);
        return {
          kind: "sky" as const,
          endpoint,
          expectedTag,
          directionError: Math.hypot(
            transverse * Math.cos(phi) - (expected[0] ?? NaN),
            transverse * Math.sin(phi) - (expected[1] ?? NaN),
            coordinate - (expected[2] ?? NaN),
          ),
        };
      }
      const transport = integrateTransport(space, photon, path, outcome.time, true, 1e-10);
      if (transport.kind !== "resolved") {
        throw new Error("Unresolved reference disk transport");
      }
      return {
        kind: "disk" as const,
        endpoint,
        expectedTag,
        inverseRadiusError: Math.abs(1 / coordinate - 1 / outcome.radius),
        inverseRadiusTolerance: 2e-6 + 2e-5 / outcome.radius,
        azimuthError: Math.abs(phi - transport.azimuth),
        delayError: Math.abs((endpoint[4] ?? NaN) - transport.coordinateTime),
        delayTolerance: 2e-5 * (1 + Math.abs(transport.coordinateTime)),
      };
    });
    await server.commands.writeFile(
      "test-results/mixed-precision.json",
      JSON.stringify(records, null, 2),
    );
    for (const [index, record] of records.entries()) {
      expect(record.endpoint.every(Number.isFinite), `finite endpoint ${index}`).toBe(true);
      expect(record.endpoint[3], `destination and image order ${index}`).toBe(record.expectedTag);
      if (record.kind === "sky") {
        expect(record.directionError, `sky direction ${index}`).toBeLessThan(5e-5);
      } else {
        expect(record.inverseRadiusError, `disk inverse radius ${index}`).toBeLessThan(
          record.inverseRadiusTolerance,
        );
        expect(record.azimuthError, `disk azimuth ${index}`).toBeLessThan(5e-5);
        expect(record.delayError, `disk emission time ${index}`).toBeLessThan(
          record.delayTolerance,
        );
      }
    }
  });
});
