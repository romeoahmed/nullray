import { describe, expect, test } from "vitest";
import { server } from "vitest/browser";
import { criticalDirection } from "../../src/physics/critical.ts";
import { metric, outerHorizon } from "../../src/physics/spacetime.ts";
import { photonFromLocal, radialPotential } from "../../src/physics/photon.ts";
import type { Vec3 } from "../../src/physics/vector.ts";
import { opticalSources } from "../../src/gpu/shaders.ts";
import criticalSource from "../../src/gpu/wgsl/lensing/critical64.wgsl?raw";
import { computeReadback, observerData } from "./compute.ts";
import { normalize } from "../../src/physics/vector.ts";
import { createCamera, initialCamera, turnCamera } from "../../src/scene/camera.ts";
import { referenceRay, referenceSky } from "../reference/sky.ts";
import projection from "../../src/gpu/wgsl/lensing/projection64.wgsl?raw";
import derivative from "../../src/gpu/wgsl/lensing/critical-tangent64.wgsl?raw";
import arithmetic from "../../src/gpu/wgsl/math/binary64.wgsl?raw";
import update from "../../src/gpu/wgsl/lensing/critical-step64.wgsl?raw";

function referenceCriticalDirection(frame: Float32Array, phase: number): Vec3 {
  const point = criticalDirection(
    { spin: frame[8] ?? NaN, charge: frame[9] ?? NaN },
    frame[4] ?? NaN,
    frame[5] === Math.fround(Math.PI) ? Math.PI : (frame[5] ?? NaN),
    phase,
  );
  if (point.kind !== "point") {
    throw new Error("Independent critical direction unavailable");
  }
  return point.direction;
}

const vectorDistance = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

describe("Spherical photon directions", () => {
  const source = `${opticalSources.precision}\n${criticalSource}
@group(0) @binding(0) var<storage, read> frames: array<Frame>;
@group(0) @binding(1) var<storage, read_write> directions: array<array<Soft64,4>>;
@group(0) @binding(2) var<storage, read> observers: array<Observer64>;
@compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id: vec3u) {
  directions[id.x] = critical_direction64(frames[id.x], observers[id.x], frames[id.x].camera_right.w);
}`;

  test("GPU critical directions remain on the double-root family across spin, charge, axes and observer radius", async () => {
    const cases = [];
    for (const [spin = NaN, charge = NaN] of [
      [0, 0],
      [0.7, 0.2],
      [-0.5, 0.7],
      [0.98, 0.1],
      [0.999, 0.04],
      [0, 0.999],
    ]) {
      const space = { spin: Math.fround(spin), charge: Math.fround(charge) };
      for (const distance of [outerHorizon(space) + 0.1, 2.1, 3, 30, 200]) {
        const radius = Math.fround(distance);
        for (const theta of [0, 1e-6, 1.2, Math.PI / 2, Math.PI]) {
          const inclination = theta === Math.PI ? Math.PI : Math.fround(theta);
          for (const parameter of [0, 0.2, 1.3, 2.8, Math.PI, 3.4, 5, 6.2]) {
            const phase = Math.fround(parameter);
            const frame = new Float32Array([
              1,
              1,
              0,
              0,
              radius,
              inclination,
              1,
              0,
              space.spin,
              space.charge,
              0.001,
              0.002,
              1,
              0,
              0,
              0,
              0,
              1,
              0,
              0,
              0,
              0,
              1,
              phase,
            ]);
            cases.push({ space, radius, inclination, phase, frame, observer: observerData(frame) });
          }
        }
      }
    }
    const packed = new Float64Array(cases.length * 12);
    for (const [index, value] of cases.entries()) {
      packed.set(value.observer, index * 12);
    }
    const words = await computeReadback(
      source,
      new Float32Array(cases.flatMap((value) => Array.from(value.frame))),
      cases.length * 8,
      cases.length,
      [new Float32Array(packed.buffer)],
    );
    const results = new Float64Array(words.buffer);
    const records = cases.map(({ space, radius, inclination, phase }, index) => {
      const direction: Vec3 = [
        results[index * 4] ?? NaN,
        results[index * 4 + 1] ?? NaN,
        results[index * 4 + 2] ?? NaN,
      ];
      const valid = results[index * 4 + 3] === 1;
      const photon = photonFromLocal(space, radius, inclination, direction);
      const g = metric(space, radius, inclination);
      const sine = inclination === 0 || inclination === Math.PI ? 0 : Math.sin(inclination);
      // Recover the actual Carter-circle phase: native trigonometric rounding may
      // move along the curve, but must not introduce a displacement across it.
      const actualPhase = Math.atan2(
        (-direction[1] * Math.sqrt(g.sigma)) / photon.energy,
        (-direction[2] * g.azimuthScale) / photon.energy - space.spin * sine,
      );
      const reference = criticalDirection(space, radius, inclination, actualPhase);
      if (reference.kind !== "point") {
        throw new Error("Independent critical preparation failed.");
      }
      const r = reference.sphericalRadius;
      const k = (photon.angularMomentum - space.spin * photon.energy) ** 2 + photon.carter;
      const p = photon.energy * (r * r + space.spin ** 2) - space.spin * photon.angularMomentum;
      const radialDerivative = 4 * photon.energy * r * p - 2 * (r - 1) * k;
      const radialScale =
        p * p + Math.abs((r * r - 2 * r + space.spin ** 2 + space.charge ** 2) * k);
      return {
        space,
        radius,
        inclination,
        phase,
        valid,
        direction,
        sphericalRadius: r,
        photon,
        normalError: Math.hypot(
          ...direction.map((value, axis) => value - (reference.direction[axis] ?? NaN)),
        ),
        normError: Math.abs(Math.hypot(...direction) - 1),
        potentialResidual: Math.abs(radialPotential(space, photon, r)) / radialScale,
        derivativeResidual:
          Math.abs(radialDerivative) /
          (Math.abs(4 * photon.energy * r * p) + Math.abs(2 * (r - 1) * k)),
      };
    });
    await server.commands.writeFile(
      "test-results/critical-direction64.json",
      JSON.stringify(
        {
          browser: navigator.userAgent,
          shaderHash: new Uint8Array(
            await crypto.subtle.digest("SHA-256", new TextEncoder().encode(source)),
          ).toHex(),
          records,
        },
        null,
        2,
      ),
    );
    for (const record of records) {
      expect.soft(record.valid, JSON.stringify(record)).toBe(true);
      expect.soft(record.normalError, JSON.stringify(record)).toBeLessThan(1e-12);
      expect.soft(record.normError, JSON.stringify(record)).toBeLessThan(1e-12);
      expect.soft(record.potentialResidual, JSON.stringify(record)).toBeLessThan(1e-12);
      expect.soft(record.derivativeResidual, JSON.stringify(record)).toBeLessThan(1e-12);
    }
  });
});

describe("Search coordinates", () => {
  test("critical seeds separate vacuum capture and escape through the production GPU ray", async () => {
    const cases = [];
    for (const [spin, charge] of [
      [0, 0],
      [0.7, 0.2],
      [-0.5, 0.7],
    ]) {
      const space = { spin: Math.fround(spin ?? NaN), charge: Math.fround(charge ?? NaN) };
      for (const observer of [outerHorizon(space) + 0.5, 30]) {
        const radius = Math.fround(observer);
        for (const theta of [0, 1.2, Math.PI]) {
          const inclination = theta === Math.PI ? theta : Math.fround(theta);
          for (const phase of [0.2, 1.7, 3.4, 5]) {
            const point = criticalDirection(space, radius, inclination, phase);
            if (point.kind !== "point") {
              throw new Error("Critical seed preparation failed");
            }
            for (const offset of [-0.001, 0.001]) {
              const direction = normalize([
                point.direction[0] + offset,
                point.direction[1],
                point.direction[2],
              ]);
              const camera = createCamera({ forward: direction, up: [0, 0, 1] });
              if (!camera) {
                throw new Error("Critical probe camera failed");
              }
              // The annulus is entirely below the horizon in this vacuum kernel
              // probe. Application scenes retain the ordinary opaque disk.
              const frame = new Float32Array([
                1,
                1,
                0,
                0,
                radius,
                inclination,
                1,
                0,
                space.spin,
                space.charge,
                0.001,
                0.002,
                ...camera.forward,
                0,
                ...camera.up,
                0,
                ...camera.right,
                0,
              ]);
              const outcome = referenceRay(0, 0, frame).outcome;
              expect(outcome.kind).toBe(offset > 0 ? "sky" : "captured");
              cases.push({
                frame,
                offset,
                phase,
                criticalRadius: point.sphericalRadius,
                sky: outcome.kind === "sky" ? referenceSky(0, 0, frame).direction : null,
              });
            }
          }
        }
      }
    }
    const observers = new Float64Array(
      cases.flatMap(({ frame }) => Array.from(observerData(frame))),
    );
    const output = await computeReadback(
      `${opticalSources.precision}
@group(0) @binding(0) var<storage, read> frames: array<Frame>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
@group(0) @binding(2) var<storage, read> observers: array<Observer64>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  output[id.x] = screen_ray_refined(frames[id.x],vec2f(0.0),observers[id.x]).value;
}`,
      new Float32Array(cases.flatMap(({ frame }) => Array.from(frame))),
      cases.length * 4,
      cases.length,
      [new Float32Array(observers.buffer)],
    );
    const records = cases.map((sample, index) => {
      const endpoint = Array.from(output.subarray(index * 4, index * 4 + 4));
      const mu = endpoint[0] ?? NaN,
        phi = endpoint[1] ?? NaN;
      const transverse = Math.sqrt(1 - mu * mu);
      return {
        offset: sample.offset,
        phase: sample.phase,
        criticalRadius: sample.criticalRadius,
        sky: sample.sky,
        frame: Array.from(sample.frame),
        endpoint,
        skyError:
          sample.sky === null
            ? 0
            : Math.hypot(
                transverse * Math.cos(phi) - sample.sky[0],
                transverse * Math.sin(phi) - sample.sky[1],
                mu - sample.sky[2],
              ),
      };
    });
    await server.commands.writeFile(
      "test-results/critical-seeds.json",
      JSON.stringify(
        {
          browser: navigator.userAgent,
          shaderHash: new Uint8Array(
            await crypto.subtle.digest(
              "SHA-256",
              new TextEncoder().encode(opticalSources.precision),
            ),
          ).toHex(),
          records,
        },
        null,
        2,
      ),
    );
    for (const record of records) {
      const label = JSON.stringify(record);
      if (record.offset > 0) {
        expect.soft(record.endpoint[3], label).toBeGreaterThan(0);
        expect.soft(record.skyError, label).toBeLessThan(5e-5);
      } else {
        expect.soft(record.endpoint[3], label).toBe(0);
      }
    }
  });
});

describe("Detector projection", () => {
  const source = `${opticalSources.precision}\n${projection}
@group(0) @binding(0) var<storage, read> frames: array<Frame>;
@group(0) @binding(1) var<storage, read_write> results: array<array<Soft64,4>>;
@group(0) @binding(2) var<storage, read> observers: array<Observer64>;
@group(0) @binding(3) var<storage, read> directions: array<array<Soft64,4>>;
@compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id: vec3u) {
  let frame = frames[id.x];
  let d = directions[id.x];
  let pixel = critical_pixel(frame,array<Soft64,3>(d[0],d[1],d[2]),frame.camera_forward.w);
  if (!pixel.valid) { results[id.x] = array<Soft64,4>(vec2u(0u),vec2u(0u),vec2u(0u),vec2u(0u)); return; }
  let lanes = array<vec4f,6>(frame.viewport,frame.observer,frame.space,frame.camera_forward,frame.camera_up,frame.camera_right);
  var words: array<u32,24>;
  for (var lane=0u;lane<6u;lane++) { for(var axis=0u;axis<4u;axis++) { words[lane*4u+axis]=bitcast<u32>(lanes[lane][axis]); } }
  let photon = soft64_screen_launch_precise(words,pixel.coordinate,observers[id.x]);
  results[id.x] = array<Soft64,4>(photon.source[0],photon.source[1],photon.source[2],soft64_number(1.0));
}`;

  test("critical projection inverts the actual quantized camera launch", async () => {
    const roll = createCamera({ forward: [-1, 0, 0], up: [0, -Math.cos(0.37), Math.sin(0.37)] });
    if (!roll) {
      throw new Error("Invalid camera.");
    }
    const cases = [];
    for (const camera of [initialCamera, turnCamera(initialCamera, 0.08, 0.04), roll]) {
      for (const phase of [0.2, 1.3, 3.4]) {
        const critical = criticalDirection({ spin: 0, charge: 0 }, 30, 0, phase);
        if (critical.kind !== "point") {
          throw new Error("Invalid critical direction.");
        }
        for (const displacement of [0.001, 1e-8]) {
          const offset = Math.fround(displacement);
          const frame = new Float32Array([
            640,
            360,
            0,
            0,
            30,
            0,
            1.2,
            0,
            0,
            0,
            6,
            64,
            ...camera.forward,
            offset,
            ...camera.up,
            0,
            ...camera.right,
            0,
          ]);
          cases.push({
            frame,
            direction: critical.direction,
            expected: normalize([
              critical.direction[0] + offset,
              critical.direction[1],
              critical.direction[2],
            ]),
          });
        }
      }
    }
    const observers = new Float64Array(cases.length * 12);
    const directions = new Float64Array(cases.length * 4);
    for (const [index, value] of cases.entries()) {
      observers.set(observerData(value.frame), index * 12);
      directions.set([...value.direction, 1], index * 4);
    }
    const output = await computeReadback(
      source,
      new Float32Array(cases.flatMap((value) => Array.from(value.frame))),
      cases.length * 8,
      cases.length,
      [new Float32Array(observers.buffer), new Float32Array(directions.buffer)],
    );
    const values = new Float64Array(output.buffer);
    const records = cases.map((value, index) => {
      const actual = Array.from(values.subarray(index * 4, index * 4 + 3));
      return {
        index,
        frame: Array.from(value.frame),
        critical: value.direction,
        expected: value.expected,
        actual,
        valid: values[index * 4 + 3] === 1,
        error: Math.hypot(
          ...value.expected.map((component, axis) => component - (actual[axis] ?? NaN)),
        ),
      };
    });
    await server.commands.writeFile(
      "test-results/critical-projection.json",
      JSON.stringify(
        {
          browser: navigator.userAgent,
          shaderSHA256: new Uint8Array(
            await crypto.subtle.digest("SHA-256", new TextEncoder().encode(source)),
          ).toHex(),
          records,
        },
        null,
        2,
      ),
    );
    for (const record of records) {
      expect(record.valid).toBe(true);
      expect.soft(record.error, JSON.stringify(record)).toBeLessThan(2e-14);
    }
  });
});

describe("Critical tangent", () => {
  const critical = criticalSource;
  const source = `${opticalSources.derivative}\n${critical}\n${projection}\n${derivative}
@group(0) @binding(0) var<storage, read> frames: array<Frame>;
@group(0) @binding(1) var<storage, read_write> outputs: array<array<Dual64,3>>;
@group(0) @binding(2) var<storage, read> observers: array<Observer64>;
@compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id: vec3u) {
  outputs[id.x] = critical_direction_derivative64(frames[id.x],observers[id.x],frames[id.x].camera_right.w);
}`;

  function gradient(frame: Float32Array, phase: number, step: number): Vec3 {
    const before = referenceCriticalDirection(frame, phase - step);
    const after = referenceCriticalDirection(frame, phase + step);
    return [
      (after[0] - before[0]) / (2 * step),
      (after[1] - before[1]) / (2 * step),
      (after[2] - before[2]) / (2 * step),
    ];
  }

  test("implicit critical tangents agree with converged independent directional differences", async () => {
    const frames: Float32Array[] = [];
    for (const [spin = NaN, charge = NaN] of [
      [0, 0],
      [0.7, 0.2],
      [-0.5, 0.7],
      [0.98, 0.1],
    ]) {
      for (const radius of [2.1, 3, 30, 200]) {
        for (const theta of [0, 1.2, Math.PI]) {
          for (const phase of [0.2, 1.3, 3.4, 6.2]) {
            frames.push(
              new Float32Array([
                640,
                360,
                0,
                0,
                radius,
                theta,
                1.2,
                0,
                spin,
                charge,
                0.001,
                0.001,
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
                phase,
              ]),
            );
          }
        }
      }
    }
    const observerWords = new Float64Array(
      frames.flatMap((frame) => Array.from(observerData(frame))),
    );
    const result = await computeReadback(
      source,
      new Float32Array(frames.flatMap((frame) => Array.from(frame))),
      frames.length * 18,
      frames.length,
      [new Float32Array(observerWords.buffer)],
      "probe",
    );
    const values = new Float64Array(result.buffer, result.byteOffset, result.length / 2);
    const records = frames.map((frame, index) => {
      const phase = frame[23] ?? NaN;
      const value: Vec3 = [
        values[index * 9] ?? NaN,
        values[index * 9 + 3] ?? NaN,
        values[index * 9 + 6] ?? NaN,
      ];
      const actual: Vec3 = [
        values[index * 9 + 1] ?? NaN,
        values[index * 9 + 4] ?? NaN,
        values[index * 9 + 7] ?? NaN,
      ];
      const fine = gradient(frame, phase, 1e-5);
      const coarse = gradient(frame, phase, 2e-5);
      const norm = Math.hypot(...fine);
      return {
        frame: Array.from(frame),
        value,
        actual,
        fine,
        coarse,
        valueError: vectorDistance(value, referenceCriticalDirection(frame, phase)),
        gradientError: vectorDistance(actual, fine) / norm,
        referenceConvergence: vectorDistance(fine, coarse) / norm,
        normalComponent:
          Math.abs(value[0] * actual[0] + value[1] * actual[1] + value[2] * actual[2]) /
          Math.hypot(...actual),
      };
    });
    await server.commands.writeFile(
      "test-results/critical-parameter-derivative.json",
      JSON.stringify(
        {
          browser: navigator.userAgent,
          shaderHash: new Uint8Array(
            await crypto.subtle.digest("SHA-256", new TextEncoder().encode(source)),
          ).toHex(),
          records,
        },
        null,
        2,
      ),
    );
    for (const record of records) {
      const label = JSON.stringify(record);
      expect.soft(record.referenceConvergence, label).toBeLessThan(1e-7);
      expect.soft(record.valueError, label).toBeLessThan(2e-7);
      expect.soft(record.gradientError, label).toBeLessThan(2e-6);
      expect.soft(record.normalComponent, label).toBeLessThan(2e-14);
    }
  });
});

describe("Subpixel corrections", () => {
  const source = `${arithmetic}\n${update}
struct ParameterInput { circle: vec4u, offset: vec2u, step: vec2f }
@group(0) @binding(0) var<storage,read> inputs:array<ParameterInput>;
@group(0) @binding(1) var<storage,read_write> outputs:array<array<Soft64,3>>;
@compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id:vec3u) {
  let input=inputs[id.x];
  var parameters=array<Soft64,3>(input.circle.xy,input.circle.zw,input.offset);
  let count=select(1u,100u,id.x==arrayLength(&inputs)-1u);
  for(var iteration=0u;iteration<count;iteration++) { parameters=critical_update64(parameters,input.step); }
  outputs[id.x]=parameters;
}`;

  test("bounded binary64 critical updates preserve tiny steps and agree with native trigonometric references", async () => {
    const cases = [];
    for (const phase of [0, 0.2, 1.7, Math.PI, 6.2]) {
      for (const angle of [-0.25, -0.1, -1e-8, 0, 1e-8, 0.1, 0.25]) {
        for (const logarithm of [-1, -0.5, -1e-8, 0, 1e-8, 0.5, 1]) {
          for (const offset of [1e-8, 0.01, 0.25]) {
            cases.push({
              phase,
              offset,
              angle: Math.fround(angle),
              logarithm: Math.fround(logarithm),
              count: 1,
            });
          }
        }
      }
    }
    cases.push({
      phase: 1.7,
      offset: 1e-8,
      angle: Math.fround(1e-8),
      logarithm: Math.fround(1e-8),
      count: 100,
    });
    const bytes = new ArrayBuffer(cases.length * 32);
    const view = new DataView(bytes);
    for (const [index, sample] of cases.entries()) {
      view.setFloat64(index * 32, Math.cos(sample.phase), true);
      view.setFloat64(index * 32 + 8, Math.sin(sample.phase), true);
      view.setFloat64(index * 32 + 16, sample.offset, true);
      view.setFloat32(index * 32 + 24, sample.angle, true);
      view.setFloat32(index * 32 + 28, sample.logarithm, true);
    }
    const result = await computeReadback(
      source,
      new Float32Array(bytes),
      cases.length * 6,
      cases.length,
      [],
      "probe",
    );
    const values = new Float64Array(result.buffer, result.byteOffset, result.length / 2);
    const records = cases.map((sample, index) => {
      const phase = sample.phase + sample.count * sample.angle;
      const actual = [
        values[index * 3] ?? NaN,
        values[index * 3 + 1] ?? NaN,
        values[index * 3 + 2] ?? NaN,
      ];
      const expected = [
        Math.cos(phase),
        Math.sin(phase),
        sample.offset * Math.exp(sample.count * sample.logarithm),
      ];
      return {
        sample,
        actual,
        expected,
        circleError: Math.hypot(
          (actual[0] ?? NaN) - (expected[0] ?? NaN),
          (actual[1] ?? NaN) - (expected[1] ?? NaN),
        ),
        offsetRelativeError: Math.abs((actual[2] ?? NaN) / (expected[2] ?? NaN) - 1),
      };
    });
    await server.commands.writeFile(
      "test-results/critical-update64.json",
      JSON.stringify(
        {
          browser: navigator.userAgent,
          sourceSHA256: new Uint8Array(
            await crypto.subtle.digest("SHA-256", new TextEncoder().encode(source)),
          ).toHex(),
          records,
        },
        null,
        2,
      ),
    );
    for (const record of records) {
      expect.soft(record.circleError, JSON.stringify(record)).toBeLessThan(2e-14);
      expect.soft(record.offsetRelativeError, JSON.stringify(record)).toBeLessThan(2e-14);
    }
  });
});
