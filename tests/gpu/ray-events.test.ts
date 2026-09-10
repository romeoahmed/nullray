import { sceneFrame } from "../support/frames.ts";
import { referenceEndpoint } from "../reference/sky.ts";
import { describe, expect, test } from "vitest";
import elliptic from "../../src/gpu/wgsl/math/elliptic.wgsl?raw";
import equator from "../../src/gpu/wgsl/geodesics/equator.wgsl?raw";
import arrival from "../../src/gpu/wgsl/geodesics/arrival.wgsl?raw";
import { photonFromLocal } from "../../src/physics/photon.ts";
import { traceVisibility } from "../reference/visibility.ts";
import { outerHorizon } from "../../src/physics/spacetime.ts";
import { prepareQuartic, quarticArrival } from "../reference/quartic.ts";
import { normalize } from "../../src/physics/vector.ts";
import { computeReadback } from "./compute.ts";
import { equatorCrossings } from "../reference/equator.ts";
import { createScene } from "../../src/scene/scene.ts";
import type { Vec3 } from "../../src/physics/vector.ts";
import { traceDirections, tracePixels } from "./visibility-probe.ts";
import launchRoundoff from "../fixtures/launch-roundoff.json" with { type: "json" };
import { integrateTransport } from "../reference/transport.ts";
import { evaluateGeodesic, prepareGeodesic } from "../reference/geodesic.ts";
import criticalFixture from "../fixtures/critical-event-order.json" with { type: "json" };
import transportSource from "../../src/gpu/wgsl/geodesics/transport.wgsl?raw";

const skyVector = (mu: number, phi: number): Vec3 => {
  const radial = Math.sqrt(1 - mu * mu);
  return [radial * Math.cos(phi), radial * Math.sin(phi), mu];
};

describe("Boundary arrivals", () => {
  const source = `${elliptic}\n${equator}\n${arrival}
@group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coefficients = inputs[3u * id.x];
  let point = inputs[3u * id.x + 1u];
  let endpoint = inputs[3u * id.x + 2u];
  let path = prepare_quartic(coefficients, point.x, point.y, point.z);
  let hit = quartic_arrival(path, endpoint.x, endpoint.y);
  let time = (hit.lower + hit.upper) / 2.0;
  var residual = 0.0;
  if (hit.status == 1u) {
    let value = evaluate_quartic(path, time);
    if (value.valid) { residual = value.value - endpoint.x; }
    else { residual = 1.0; }
  }
  outputs[id.x] = vec4f(time, f32(hit.status), residual, f32(radial_destination(coefficients.xzw, point.x, path, endpoint.z)));
}`;

  test("GPU inverse quartic phase agrees with CPU at sky and horizon endpoints", async () => {
    const records: number[] = [];
    for (const spin of [-0.8, 0, 0.8]) {
      for (const charge of [0, 0.3]) {
        for (const radius of [4, 10, 30, 160, 200]) {
          for (const inclination of [0.1, 1.2, 2.7]) {
            for (const direction of [
              normalize([-1, 0.1, 0.2]),
              normalize([-1, 0.8, -0.3]),
              normalize([1, 0.2, 0.4]),
              normalize([0, 0.6, 0.8]),
            ]) {
              const space = { spin, charge };
              const photon = photonFromLocal(space, radius, inclination, direction);
              const { energy: e, angularMomentum: l, carter: c } = photon;
              const k = (l - spin * e) ** 2 + c;
              const horizon = 1 / outerHorizon(space);
              const reference = traceVisibility(space, photon, radius, inclination).outcome;
              expect(["sky", "captured"]).toContain(reference.kind);
              const target = reference.kind === "sky" ? 0 : horizon;
              const velocity =
                reference.kind === "sky"
                  ? -Math.abs(e)
                  : Math.abs(e + (spin * spin * e - spin * l) * horizon ** 2);
              records.push(
                e * e,
                0,
                spin * spin * e * e - l * l - c,
                2 * k,
                -spin * spin * c - charge * charge * k,
                1 / radius,
                photon.radialVelocity / radius ** 2,
                0,
                target,
                velocity,
                horizon,
                0,
              );
            }
          }
        }
      }
    }
    // Future-directed zero/negative Killing energy can turn outside the horizon,
    // but cannot reach the distant source boundary.
    for (const e of [0, -0.1]) {
      const spin = 0.9,
        charge = 0.1,
        l = -1,
        c = 0.1,
        u = 1 / 1.5;
      const k = (l - spin * e) ** 2 + c;
      const c2 = spin * spin * e * e - l * l - c;
      const c3 = 2 * k;
      const c4 = -spin * spin * c - charge * charge * k;
      const horizon = 1 / outerHorizon({ spin, charge });
      const speed = Math.sqrt(e * e + u * u * (c2 + u * (c3 + c4 * u)));
      for (const sign of [-1, 1]) {
        records.push(
          e * e,
          0,
          c2,
          c3,
          c4,
          u,
          sign * speed,
          0,
          horizon,
          Math.abs(e + (spin * spin * e - spin * l) * horizon * horizon),
          horizon,
          0,
        );
      }
    }
    const input = new Float32Array(records);
    const count = input.length / 12;
    const result = await computeReadback(source, input, count * 4, count);
    for (let index = 0; index < count; index++) {
      const row = input.subarray(index * 12, index * 12 + 12);
      const at = (offset: number) => row[offset] ?? Number.NaN;
      const path = prepareQuartic([at(0), at(1), at(2), at(3), at(4)], at(5), at(6));
      const expected = quarticArrival(path, at(8), at(9));
      expect(result[index * 4 + 3], `destination row ${index}`).toBe(at(8) === 0 ? 1 : 2);
      expect(expected, `CPU arrival row ${index}`).toBeDefined();
      expect(result[index * 4 + 1], `row ${index}: input ${JSON.stringify(Array.from(row))}`).toBe(
        1,
      );
      expect(
        Math.abs((result[index * 4] ?? Infinity) - (expected ?? Infinity)),
        `time row ${index}`,
      ).toBeLessThan(2e-5);
      expect(Math.abs(result[index * 4 + 2] ?? Infinity), `residual row ${index}`).toBeLessThan(
        2e-5,
      );
    }
  });
});

describe("Polar crossings", () => {
  test("GPU equator phases agree with CPU for identical quantized photon constants", async () => {
    const cases: number[][] = [];
    for (const spin of [0, -0.8, 0.8]) {
      for (const theta of [0.5, 1.4, 1.8, 2.5]) {
        for (const polar of [-1, -0.1, 0.1, 1]) {
          const photon = photonFromLocal(
            { spin, charge: 0.3 },
            8,
            theta,
            normalize([-0.2, polar, 0.7]),
          );
          cases.push([
            spin,
            photon.energy,
            photon.angularMomentum,
            photon.carter,
            theta,
            photon.polarVelocity,
            0,
            0,
          ]);
        }
      }
    }
    const data = new Float32Array(cases.flat());
    const result = await computeReadback(
      `${elliptic}\n${equator}
      @group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
      @group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
      @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
        let p = inputs[id.x * 2u]; let q = inputs[id.x * 2u + 1u];
        let result = equator_events(p.x, p.y, p.z, p.w, q.x, q.y);
        outputs[id.x] = vec4f(result.first, result.spacing, f32(result.status), 0.0);
      }`,
      data,
      cases.length * 4,
      cases.length,
    );
    for (let i = 0; i < cases.length; i++) {
      const [spin, energy, angularMomentum, carter, theta, polarVelocity] = data.slice(
        i * 8,
        i * 8 + 6,
      );
      if (
        spin === undefined ||
        energy === undefined ||
        angularMomentum === undefined ||
        carter === undefined ||
        theta === undefined ||
        polarVelocity === undefined
      ) {
        throw new Error("Missing quantized fixture input.");
      }
      const expected = equatorCrossings(
        { spin, charge: 0.3 },
        { energy, angularMomentum, carter, polarVelocity, radialVelocity: 0 },
        theta,
      );
      expect(expected.kind).toBe("crossings");
      if (expected.kind !== "crossings") {
        throw new Error("Expected ordinary transverse events.");
      }
      expect(result[i * 4 + 2]).toBe(1);
      expect(result[i * 4]).toBeCloseTo(expected.first, 5);
      expect(result[i * 4 + 1]).toBeCloseTo(expected.spacing, 5);
    }
  });
});

describe("Ordered visibility", () => {
  test("critical launch remains an escaping ray under the quantized source contract", async () => {
    const { space, radius, inclination } = launchRoundoff;
    const scene = createScene({
      space,
      observer: { radius, inclination, azimuth: 0, fieldOfView: 1 },
    });
    if (!scene.ok) {
      throw new Error(scene.error);
    }
    const [x, y, z] = launchRoundoff.direction;
    if (x === undefined || y === undefined || z === undefined) {
      throw new Error("Missing critical launch direction.");
    }
    const local: Vec3 = [x, y, z];
    const photon = photonFromLocal(space, radius, inclination, local);
    const reference = traceVisibility(space, photon, radius, inclination, {
      inner: scene.value.diskInner,
      outer: scene.value.diskOuter,
    });
    expect(reference.outcome.kind).toBe("sky");
    const endpoint = await traceDirections([local], inclination, scene.value);
    expect(endpoint[3]).toBeGreaterThan(0);
  });

  test.for([
    { spin: 0.7, charge: 0.2, radius: 18, inclination: 1.2, width: 1280, height: 720 },
    { spin: 0.7, charge: 0.2, radius: 18, inclination: 1.2, width: 128, height: 72 },
    { spin: -0.8, charge: 0.3, radius: 30, inclination: 2.2, width: 128, height: 72 },
    { spin: 0, charge: 0, radius: 30, inclination: 0, width: 128, height: 72 },
  ])(
    "production event order, sky direction and disk transfer agree with binary64: $spin / $charge / $width × $height",
    async (parameters) => {
      const scene = createScene({
        space: parameters,
        observer: { ...parameters, azimuth: 0, fieldOfView: 2 * Math.atan(0.6) },
      });
      if (!scene.ok) {
        throw new Error(scene.error);
      }
      const frame = sceneFrame(scene.value, parameters.width, parameters.height);
      // Cover the complete viewport with a fixed stratified grid. Minimized
      // critical pixels have their own strict production endpoint/derivative gates.
      const pixels = Array.from({ length: 192 }, (_, index): readonly [number, number] => [
        (((index % 16) + 0.5) * parameters.width) / 16 - 0.5,
        ((Math.floor(index / 16) + 0.5) * parameters.height) / 12 - 0.5,
      ]);
      const data = await tracePixels(frame, pixels);
      for (const [index, [x, y]] of pixels.entries()) {
        const expected = referenceEndpoint(x, y, frame);
        const at = (offset: number) => data[index * 8 + offset] ?? NaN;
        const label = `${x},${y}: ${expected.kind}`;
        expect(at(3), label).toBe(expected.kind === "captured" ? 0 : expected.tag);
        if (expected.kind === "sky") {
          const actual = skyVector(at(0), at(1));
          expect(
            Math.hypot(...actual.map((value, axis) => value - (expected.direction[axis] ?? NaN))),
            label,
          ).toBeLessThan(5e-5);
          expect(Math.abs(at(2) / expected.energy - 1), label).toBeLessThan(2e-5);
        } else if (expected.kind === "disk") {
          expect(Math.abs(1 / at(0) - 1 / expected.radius), label).toBeLessThan(
            2e-6 + 2e-5 / expected.radius,
          );
          const angle = at(1) - expected.azimuth;
          expect(Math.abs(Math.atan2(Math.sin(angle), Math.cos(angle))), label).toBeLessThan(5e-5);
          expect(Math.abs(at(2) / expected.frequency - 1), label).toBeLessThan(2e-5);
          expect(
            Math.abs(at(4) - expected.delay) / (1 + Math.abs(expected.delay)),
            label,
          ).toBeLessThan(5e-5);
        }
      }
    },
  );
});

describe("Capture and escape", () => {
  test("Schwarzschild capture and higher-order images approach the analytic critical impact parameter", async () => {
    const critical = 3 * Math.sqrt(3);
    const impacts = [
      critical - 0.03,
      critical - 0.01,
      critical - 0.003,
      ...Array.from({ length: 48 }, (_, index) => critical + 0.001 * 1000 ** (index / 47)),
    ];
    const directions: Vec3[] = impacts.map((impact) => {
      const sine = (impact * Math.sqrt(1 - 2 / 30)) / 30;
      return [-Math.sqrt(1 - sine * sine), sine, 0];
    });
    const result = await traceDirections(directions, 0);
    expect(
      impacts
        .map((impact, index) => ({
          impact,
          stage: result[index * 4],
          candidateTime: result[index * 4 + 1],
          residual: result[index * 4 + 2],
          tag: result[index * 4 + 3],
        }))
        .filter((sample) => sample.tag === -2),
    ).toEqual([]);
    let higherOrders = 0;
    let secondary = 0;
    for (const [index, impact] of impacts.entries()) {
      const tag = result[index * 4 + 3] ?? Number.NaN;
      if (impact < critical) {
        expect(tag).toBe(0);
      } else {
        expect(tag).not.toBe(0);
        if (tag <= -5) {
          higherOrders++;
          expect(impact - critical).toBeLessThan(0.06);
        }
        if (tag === -4) {
          secondary++;
        }
      }
    }
    expect(higherOrders).toBeGreaterThan(0);
    expect(secondary).toBeGreaterThan(0);
  });

  test("approaching and receding disk rays recover the Schwarzschild endpoint frequency law", async () => {
    const inclination = Math.fround(1.2);
    const directions = [normalize([-1, 0.1, -0.3]), normalize([-1, 0.1, 0.3])];
    const result = await traceDirections(directions, inclination);
    for (const [index, direction] of directions.entries()) {
      const radius = result[index * 4] ?? Number.NaN;
      const frequency = result[index * 4 + 2] ?? Number.NaN;
      expect(result[index * 4 + 3]).toBeLessThanOrEqual(-3);
      const energy = Math.sqrt(1 - 2 / 30);
      const momentum = -Math.fround(direction[2]) * 30 * Math.sin(inclination);
      const expected = Math.sqrt(1 - 3 / radius) / (energy - radius ** -1.5 * momentum);
      expect(Math.abs(frequency - expected)).toBeLessThan(2e-5);
    }
    expect(Math.abs((result[0] ?? Number.NaN) - (result[4] ?? Number.NaN))).toBeLessThan(1e-4);
    expect(result[2]).toBeGreaterThan(result[6] ?? Number.NaN);
  });

  test("higher-order endpoint stays within the independent critical-pixel reference", async () => {
    const { space, radius, inclination, direction } = criticalFixture;
    const scene = createScene({
      space,
      observer: { radius, inclination, azimuth: 0, fieldOfView: 1 },
    });
    if (!scene.ok) {
      throw new Error(scene.error);
    }
    const result = await traceDirections(
      [normalize([direction[0] ?? 0, direction[1] ?? 0, direction[2] ?? 0])],
      inclination,
      scene.value,
    );
    expect(result[3]).toBe(-5);
    expect(
      Math.abs(1 / (result[0] ?? Number.NaN) - 1 / criticalFixture.referenceRadius),
    ).toBeLessThan(1e-5);
    expect(Math.abs((result[1] ?? Number.NaN) - criticalFixture.referencePhi)).toBeLessThan(5e-5);
  });

  test("an exact Schwarzschild spherical launch remains unresolved rather than captured", async () => {
    const scene = createScene({
      space: { spin: 0, charge: 0 },
      observer: { radius: 3, inclination: 1.2, azimuth: 0, fieldOfView: 1 },
    });
    if (!scene.ok) {
      throw new Error(scene.error);
    }
    const result = await traceDirections([[0, 1, 0]], 1.2, scene.value);
    expect(result[3]).toBe(-2);
    expect(result[0]).toBe(14);
  });
});

describe("Azimuth and travel time", () => {
  const transport = transportSource;
  test("GPU transport matches CPU quadrature for exterior and axis-observer paths", async () => {
    const rows: number[][] = [];
    const critical = photonFromLocal(
      { spin: 0, charge: 0 },
      30,
      0,
      [-0.9858951568603516, 0.1673642098903656, 0],
    );
    for (const end of [1, 1.5, 2.18]) {
      rows.push([
        0,
        0,
        critical.energy,
        critical.angularMomentum,
        critical.carter,
        30,
        0,
        end,
        critical.radialVelocity,
        critical.polarVelocity,
        0,
        0,
      ]);
    }
    const criticalCount = rows.length;
    for (const spin of [0, -0.8, 0.8]) {
      for (const direction of [
        [-1, 0.3, 0.5],
        [0, -0.3, -0.8],
      ] as const) {
        const photon = photonFromLocal({ spin, charge: 0.4 }, 8, 1.1, normalize(direction));
        rows.push([
          spin,
          0.4,
          photon.energy,
          photon.angularMomentum,
          photon.carter,
          8,
          1.1,
          0.04,
          photon.radialVelocity,
          photon.polarVelocity,
          0,
          0,
        ]);
      }
    }
    for (const spin of [0, 0.8]) {
      for (const azimuth of [-1e-5, 0, 1e-5]) {
        const photon = photonFromLocal(
          { spin, charge: 0.4 },
          8,
          0.5,
          normalize([-0.2, -1, azimuth]),
        );
        rows.push([
          spin,
          0.4,
          photon.energy,
          photon.angularMomentum,
          photon.carter,
          8,
          0.5,
          0.1,
          photon.radialVelocity,
          photon.polarVelocity,
          0,
          0,
        ]);
      }
    }
    for (const spin of [0, 0.8]) {
      for (const theta of [0, 1e-4, Math.PI - 1e-4, Math.PI]) {
        const photon = photonFromLocal({ spin, charge: 0.4 }, 8, theta, normalize([0.2, 0.7, 0.3]));
        rows.push([
          spin,
          0.4,
          photon.energy,
          photon.angularMomentum,
          photon.carter,
          8,
          theta,
          0.04,
          photon.radialVelocity,
          photon.polarVelocity,
          0,
          0,
        ]);
      }
    }
    const inputs = new Float32Array(rows.flat());
    const result = await computeReadback(
      `${elliptic}\n${equator}\n${transport}
    @group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
    @group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
    @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
      let s = inputs[id.x * 3u]; let p = inputs[id.x * 3u + 1u]; let v = inputs[id.x * 3u + 2u];
      let a = s.x; let q = s.y; let e = s.z; let l = s.w; let c = p.x;
      let k = (l - a * e) * (l - a * e) + c;
      let b = a * a * e * e - l * l - c;
      let radial = prepare_quartic(vec4f(e * e, 0.0, b, 2.0 * k), -a * a * c - q * q * k, 1.0 / p.y, v.x / (p.y * p.y));
      let sine = select(sin(p.z), 0.0, p.z == 0.0 || p.z == 3.1415926536);
      let polar = prepare_quartic(vec4f(c, 0.0, b, 0.0), -a * a * e * e, cos(p.z), sine * v.y);
      let events = equator_events(a, e, l, c, p.z, v.y);
      let value = integrate_transport(radial, polar, a, q, e, l, c, p.w, true, events);
      outputs[id.x] = vec4f(value.value, value.error.x, select(0.0, 1.0, value.valid));
    }`,
      inputs,
      rows.length * 4,
      rows.length,
    );
    for (let i = 0; i < rows.length; i++) {
      const [
        spin,
        charge,
        energy,
        angularMomentum,
        carter,
        r,
        theta,
        time,
        radialVelocity,
        polarVelocity,
      ] = inputs.slice(i * 12, i * 12 + 10);
      if (
        spin === undefined ||
        charge === undefined ||
        energy === undefined ||
        angularMomentum === undefined ||
        carter === undefined ||
        r === undefined ||
        theta === undefined ||
        time === undefined ||
        radialVelocity === undefined ||
        polarVelocity === undefined
      ) {
        throw new Error("Missing fixture input.");
      }
      const space = { spin, charge };
      const photon = { energy, angularMomentum, carter, radialVelocity, polarVelocity };
      const expected = integrateTransport(
        space,
        photon,
        prepareGeodesic(space, photon, r, theta === Math.fround(Math.PI) ? Math.PI : theta),
        time,
        true,
        1e-11,
        65_536,
      );
      if (expected.kind !== "resolved") {
        throw new Error(`CPU reference failed at row ${i}, time ${time}: ${expected.reason}.`);
      }
      expect(result[i * 4 + 3]).toBe(1);
      expect(result[i * 4]).toBeCloseTo(expected.azimuth, 4);
      if (i < criticalCount) {
        const endpoint = evaluateGeodesic(prepareGeodesic(space, photon, r, theta), time);
        if (!endpoint || endpoint.inverseRadius <= 0) {
          throw new Error("Missing exterior reference endpoint.");
        }
        const timeError = Math.abs((result[i * 4 + 1] ?? Number.NaN) - expected.coordinateTime);
        // Critical-ray f32 conditioning is assessed in the highest emitted Fourier phase.
        // This does not claim the ordinary-ray absolute travel-time accuracy here.
        const omega = endpoint.inverseRadius ** 1.5;
        expect(23 * omega * timeError).toBeLessThan(0.005);
      } else {
        expect(result[i * 4 + 1]).toBeCloseTo(expected.coordinateTime, 3);
      }
    }
  });
});
