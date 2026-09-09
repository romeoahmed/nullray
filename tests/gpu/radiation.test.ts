import { expect, test } from "vitest";
import radiation from "../../src/render/shaders/radiation.wgsl?raw";
import orbit from "../../src/render/shaders/orbit.wgsl?raw";
import {
  blackbodyXYZ,
  createBlackbodyTable,
  diskFrequencyRatio,
  xyzToLinearRGB,
} from "../../src/physics/radiation.ts";
import { computeReadback } from "./compute.ts";
import { circularOrbit, isco } from "../../src/physics/spacetime.ts";

const source = `${orbit}\n${radiation}`;

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
    `${orbit}
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
    const [spin = NaN, charge = NaN, radius = NaN, energy = NaN, momentum = NaN] = inputs.subarray(
      index * 8,
      index * 8 + 5,
    );
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
        { energy, angularMomentum, carter: 0, radialVelocity: 0, polarVelocity: 0 },
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

test("finite-coherence emission preserves positive mean flux across retarded epochs", async () => {
  const times = [-25, -24, -23, 0, 23.999, 24, 24.001, 100_000];
  const angles = 256;
  const input = new Float32Array(times.length * angles * 4);
  for (const [epoch, time] of times.entries()) {
    for (let angle = 0; angle < angles; angle++) {
      input.set([6, (2 * Math.PI * angle) / angles, time, 0.065], 4 * (epoch * angles + angle));
    }
  }
  const result = await computeReadback(
    `${source}
    @group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
    @group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
    @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
      let p = inputs[id.x];
      outputs[id.x] = vec4f(disk_modulation_epoch(p.x, 3.3, p.y, p.z, 0.0, p.w, 0.65));
    }`,
    input,
    input.length,
    input.length / 4,
  );
  for (let epoch = 0; epoch < times.length; epoch++) {
    let sum = 0;
    for (let angle = 0; angle < angles; angle++) {
      const value = result[4 * (epoch * angles + angle)] ?? Number.NaN;
      expect(value).toBeGreaterThanOrEqual(0.35);
      expect(value).toBeLessThanOrEqual(1.65);
      sum += value;
    }
    expect(Math.abs(sum / angles - 1)).toBeLessThan(1e-5);
  }
  for (let angle = 0; angle < angles; angle++) {
    const before = result[4 * (4 * angles + angle)] ?? Number.NaN;
    const after = result[4 * (6 * angles + angle)] ?? Number.NaN;
    expect(Math.abs(before - after)).toBeLessThan(0.002);
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

test("localized disk modes retain mean flux across radii and join smoothly at radial cells", async () => {
  const angles = 256;
  const radii = [3.3, 3.8, 6, 16, 40, 64];
  const epochs = [-48, -0, 0, 12, 24, 63, 100000];
  const inputs: number[] = [];
  for (const r of radii) {
    const omega = Math.sqrt(r - 0.04) / (r * r + 0.7 * Math.sqrt(r - 0.04));
    for (const time of epochs) {
      for (let angle = 0; angle < angles; angle++) {
        inputs.push(r, (2 * Math.PI * angle) / angles, time, omega);
      }
    }
  }
  const result = await computeReadback(
    `${radiation}
    @group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
    @group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
    @compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id: vec3u) {
      let p = inputs[id.x];
      let value = disk_modulation_epoch(p.x, 3.3, p.y, p.z, 0.0, p.w, 0.65);
      outputs[id.x] = vec4f(value);
    }`,
    new Float32Array(inputs),
    inputs.length,
    inputs.length / 4,
  );
  for (let group = 0; group < radii.length * epochs.length; group++) {
    let total = 0;
    for (let angle = 0; angle < angles; angle++) {
      const value = result[4 * (group * angles + angle)] ?? Number.NaN;
      expect(value).toBeGreaterThanOrEqual(0.35);
      expect(value).toBeLessThanOrEqual(1.65);
      total += value;
    }
    expect(Math.abs(total / angles - 1)).toBeLessThan(1e-5);
  }
  const boundaries: number[] = [];
  for (const density of [3, 6, 12, 24]) {
    for (const cell of [0, 1, 3, 8]) {
      for (const displacement of [-1e-4, 0, 1e-4]) {
        boundaries.push(Math.exp(cell / density + displacement) * 3.3, 0.37, -2, 0);
      }
    }
  }
  const joins = await computeReadback(
    `${radiation}
    @group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
    @group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
    @compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id: vec3u) {
      let p = inputs[id.x];
      outputs[id.x] = vec4f(disk_structure(p.x, 3.3, p.y, p.z));
    }`,
    new Float32Array(boundaries),
    boundaries.length,
    boundaries.length / 4,
  );
  for (let group = 0; group < boundaries.length / 12; group++) {
    const left = joins[group * 12] ?? Number.NaN;
    const center = joins[group * 12 + 4] ?? Number.NaN;
    const right = joins[group * 12 + 8] ?? Number.NaN;
    expect(Math.abs(right - left)).toBeLessThan(0.002);
    expect(Math.abs(right - 2 * center + left)).toBeLessThan(1e-5);
  }
});
