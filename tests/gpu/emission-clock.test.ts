import { expect, test } from "vitest";
import radiation from "../../src/render/shaders/radiation.wgsl?raw";
import { computeReadback } from "./compute.ts";

function phase(cell: number, mode: number, epoch: number): number {
  const bits = new DataView(new ArrayBuffer(4));
  bits.setFloat32(0, epoch === 0 ? 0 : epoch);
  let seed = (Math.imul(cell, 747796405) + Math.imul(mode, 2891336453) + bits.getUint32(0)) >>> 0;
  seed ^= seed << 13;
  seed ^= seed >>> 17;
  seed ^= seed << 5;
  seed = Math.imul(seed, 277803737);
  return (seed >>> 8) * ((2 * Math.PI) / 16777216);
}

const smooth = (value: number) => value * value * (3 - 2 * value);

function reference(time: number, phi: number): number {
  const epoch = Math.floor(time / 24);
  const age = time - epoch * 24;
  const wave = (cohort: number, localAge: number) => {
    let sum = 0;
    for (const [mode = 0, density = 0, weight = 0] of [
      [3, 3, 0.45],
      [7, 6, 0.3],
      [13, 12, 0.15],
      [23, 24, 0.1],
    ]) {
      const coordinate = Math.log(6 / Math.fround(3.3)) * density;
      const cell = Math.floor(coordinate);
      const blend = smooth(coordinate - cell);
      const angle = mode * (phi - Math.fround(0.065) * localAge);
      sum +=
        weight *
        ((1 - blend) * Math.sin(angle + phase(cell, mode, cohort)) +
          blend * Math.sin(angle + phase(cell + 1, mode, cohort)));
    }
    return sum;
  };
  const blend = smooth(age / 24);
  return (
    1 + Math.fround(0.65) * ((1 - blend) * wave(epoch, age) + blend * wave(epoch + 1, age - 24))
  );
}

test("split emission time retains subframe changes and retarded cohort transitions at long epochs", async () => {
  const records = [0, 10, 100_000, 1_000_000, 10_000_000].flatMap((epoch) =>
    [0, 0.125, 23.999, 24, 24.125].flatMap((age) =>
      [0, -53.25].map((delay) => [epoch, age, delay, 0.4]),
    ),
  );
  const inputs = new Float32Array(records.flat());
  const output = await computeReadback(
    `${radiation}
    @group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
    @group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
    @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
      let p = inputs[id.x];
      outputs[id.x] = vec4f(
        disk_modulation_epoch(6.0, 3.3, p.w, p.y + p.z, p.x, 0.065, 0.65),
        0.0, 0.0, 0.0);
    }`,
    inputs,
    inputs.length,
    records.length,
  );
  for (let index = 0; index < records.length; index++) {
    const [epoch = NaN, age = NaN, delay = NaN, phi = NaN] = inputs.subarray(
      index * 4,
      index * 4 + 4,
    );
    const expected = reference(epoch * 24 + age + delay, phi);
    expect(Math.abs((output[index * 4] ?? NaN) - expected)).toBeLessThan(2e-5);
  }
});
