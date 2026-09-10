import { describe, expect, test } from "vitest";
import radiation from "../../src/gpu/wgsl/imaging/radiation.wgsl?raw";
import fieldSource from "../../src/gpu/wgsl/sources/field.wgsl?raw";
import structure from "../../src/gpu/wgsl/imaging/structure.wgsl?raw";
import {
  blackbodyXYZ,
  createBlackbodyTable,
  diskFrequencyRatio,
  xyzToLinearRGB,
} from "../../src/physics/radiation.ts";
import { computeReadback } from "./compute.ts";
import { circularOrbit, isco } from "../../src/physics/spacetime.ts";
import { createStructureField } from "../../src/gpu/sources/structure.ts";

describe("Spectral transport", () => {
  const source = radiation;

  test("advected material joins continuously in azimuth and at renewed emission epochs", async () => {
    const space = { spin: 0.7, charge: 0.2 };
    const inner = 3.3;
    const orbit = circularOrbit(space, inner, 1);
    if (!orbit) {
      throw new Error("Timelike material fixture required.");
    }
    const records = [6, 16, 30].flatMap((radius) =>
      [-1, 0, 1, 1000].flatMap((cycle) =>
        [0, 1].map((footprint) => [radius, 0.7, (4 * Math.PI * cycle) / orbit.omega, footprint]),
      ),
    );
    const result = await computeReadback(
      `${fieldSource}\n${structure}
      @group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
      struct Samples { value: vec3f, wrap: vec3f, past: vec3f, future: vec3f }
      @group(0) @binding(1) var<storage, read_write> outputs: array<Samples>;
      @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
        let p = inputs[id.x];
        let root = sqrt(p.x - 0.04);
        let omega = root / (p.x * p.x + 0.7 * root);
        let span = p.w * vec4f(0.1, 0.005, 0.2, 0.3);
        outputs[id.x] = Samples(
          disk_cloud(vec2f(0.7, 0.2), 3.3, p.x, p.y, 0, p.z, omega, span),
          disk_cloud(vec2f(0.7, 0.2), 3.3, p.x, p.y + 6.283185307179586, 0, p.z, omega, span),
          disk_cloud(vec2f(0.7, 0.2), 3.3, p.x, p.y, 0, p.z - 0.001, omega, span),
          disk_cloud(vec2f(0.7, 0.2), 3.3, p.x, p.y, 0, p.z + 0.001, omega, span));
      }`,
      new Float32Array(records.flat()),
      records.length * 16,
      records.length,
      [],
      "main",
      async (device) => {
        const field = await createStructureField(device);
        return {
          entries: [
            { binding: 20, resource: field.texture.createView() },
            { binding: 21, resource: field.sampler },
          ],
          dispose: () => field.dispose(),
        };
      },
    );
    expect(result.every(Number.isFinite)).toBe(true);
    for (let i = 0; i < result.length; i += 16) {
      for (const channel of [0, 1, 2]) {
        expect(
          Math.abs((result[i + channel] ?? NaN) - (result[i + 4 + channel] ?? NaN)),
        ).toBeLessThan(2e-4);
        expect(
          Math.abs((result[i + 8 + channel] ?? NaN) - (result[i + 12 + channel] ?? NaN)),
        ).toBeLessThan(0.01);
      }
    }
  });

  test("charged circular-emitter frequency agrees with metric contraction for signed Killing energy", async () => {
    const records = [-0.95, -0.5, 0, 0.5, 0.95].flatMap((spin) =>
      [-0.2, 0, 0.2].flatMap((charge) => {
        const space = { spin: Math.fround(spin), charge: Math.fround(charge) };
        return [isco(space, 1) * 1.001, 16, 64].flatMap((radius) =>
          [
            [1, -3],
            [1, 3],
            [0, -3],
            [-0.01, -3],
            [0, 3],
          ].map(([energy = 0, momentum = 0]) => [
            space.spin,
            space.charge,
            radius,
            energy,
            momentum,
            0,
            0,
            0,
          ]),
        );
      }),
    );
    const inputs = new Float32Array(records.flat());
    const result = await computeReadback(
      `${radiation}
    struct Input { space: vec4f, momentum: vec4f }
    @group(0) @binding(0) var<storage, read> inputs: array<Input>;
    @group(0) @binding(1) var<storage, read_write> outputs: array<vec2f>;
    @compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id: vec3u) {
      let p = inputs[id.x];
      outputs[id.x] = disk_frequency(p.space.x, p.space.y, p.space.w, p.momentum.x, p.space.z);
    }`,
      inputs,
      records.length * 2,
      records.length,
    );
    for (let index = 0; index < records.length; index++) {
      const [spin = NaN, charge = NaN, radius = NaN, energy = NaN, momentum = NaN] =
        inputs.subarray(index * 8, index * 8 + 5);
      const emitter = circularOrbit({ spin, charge }, radius, 1);
      expect(emitter).toBeDefined();
      if (!emitter) {
        throw new Error("Missing timelike reference emitter.");
      }
      const contraction = emitter.ut * (energy - emitter.omega * momentum);
      const frequency = result[index * 2] ?? NaN;
      const omega = result[index * 2 + 1] ?? NaN;
      if (contraction <= 0) {
        expect(frequency).toBe(-1);
        expect(omega).toBe(0);
      } else {
        expect(Math.abs(frequency * contraction - 1)).toBeLessThan(3e-6);
        expect(Math.abs(omega / emitter.omega - 1)).toBeLessThan(5e-7);
      }
    }
  });

  test("GPU thermal radiation agrees with direct CIE integration and endpoint frequency", async () => {
    const table = createBlackbodyTable();
    const input = new Float32Array([
      800, 0, 0, 0, 3500, 0, 0, 0, 6500, 0, 0, 0, 20000, 0, 0, 0, 0, 6, 0.9, -3, 0, 6, 0.9, 3,
    ]);
    const count = input.length / 4;
    const result = await computeReadback(
      `${source}
    @group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
    @group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
    @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
      let p = inputs[id.x];
      var color = vec4f(0.0);
      if (p.x > 0.0) { color = blackbody_radiance(p.x); }
      else { color = blackbody_radiance(6500.0 * disk_frequency(0.7, 0.2, p.z, p.w, p.y).x); }
      outputs[id.x] = color;
    }`,
      input,
      count * 4,
      count,
      [table.data],
    );
    const normalization = blackbodyXYZ(6500)[1];
    for (let i = 0; i < count; i++) {
      let temperature = input[i * 4];
      if (temperature === undefined) {
        throw new Error("Missing temperature.");
      }
      if (temperature === 0) {
        const radius = input[i * 4 + 1];
        const energy = input[i * 4 + 2];
        const angularMomentum = input[i * 4 + 3];
        if (radius === undefined || energy === undefined || angularMomentum === undefined) {
          throw new Error("Missing emitter input.");
        }
        const shift = diskFrequencyRatio(
          { spin: Math.fround(0.7), charge: Math.fround(0.2) },
          { energy, angularMomentum },
          radius,
          1,
        );
        if (shift === undefined) {
          throw new Error("Invalid reference emitter.");
        }
        temperature = shift * 6500;
      }
      const expected = xyzToLinearRGB(blackbodyXYZ(temperature)).map((x) => x / normalization);
      const scale = Math.max(...expected.map(Math.abs));
      for (let channel = 0; channel < 3; channel++) {
        const value = result[i * 4 + channel];
        const target = expected[channel];
        if (value === undefined || target === undefined) {
          throw new Error("Missing radiation channel.");
        }
        expect(Math.abs(value - target)).toBeLessThan(0.005 * scale + 1e-9);
      }
    }
  });

  test("thermal lookup distinguishes a negligible visible tail from unsupported temperatures", async () => {
    const temperatures = new Float32Array([0, 99, 100, 1000000, 1000001, -1]);
    const result = await computeReadback(
      `${radiation}
    @group(0) @binding(0) var<storage, read> inputs: array<f32>;
    @group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
    @compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id: vec3u) {
      outputs[id.x] = blackbody_radiance(inputs[id.x]);
    }`,
      temperatures,
      temperatures.length * 4,
      temperatures.length,
      [createBlackbodyTable().data],
    );
    expect(result.every(Number.isFinite)).toBe(true);
    expect(Array.from(result).filter((_, index) => index % 4 === 3)).toEqual([1, 1, 1, 1, 0, 0]);
    expect(Array.from(result.slice(0, 3))).toEqual([0, 0, 0]);
  });
});
