import { expect, test } from "vitest";
import regression from "../fixtures/jacobi-dn.json" with { type: "json" };
import { jacobi } from "../reference/elliptic.ts";
import { computeReadback } from "./compute.ts";
import source from "../../src/render/shaders/elliptic.wgsl?raw";
import { prepareQuartic, evaluateQuartic } from "../reference/quartic.ts";
import type { Quartic } from "../reference/quartic.ts";

test("real GPU quartic evaluation agrees with the CPU on identical f32 inputs", async () => {
  const data = new Float32Array([
    1,
    0,
    -1,
    0,
    0,
    0,
    1,
    0.7,
    1,
    0,
    0,
    0,
    1,
    0,
    1,
    0.3,
    1,
    0.4,
    -2,
    0,
    0.3,
    0,
    -1,
    0.5,
    2,
    -1,
    0.5,
    0.2,
    -0.1,
    0,
    Math.sqrt(2),
    0.5,
  ]);
  const results = await computeReadback(
    `${source}
@group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec2f>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  let coefficients = inputs[id.x * 2u];
  let state = inputs[id.x * 2u + 1u];
  let result = evaluate_quartic(prepare_quartic(coefficients, state.x, state.y, state.z), state.w);
  outputs[id.x] = vec2f(result.value, f32(result.valid));
}`,
    data,
    8,
    4,
  );
  for (let i = 0; i < 4; i++) {
    const values = data.subarray(i * 8, i * 8 + 8);
    const coefficients: Quartic = [
      values[0] ?? NaN,
      values[1] ?? NaN,
      values[2] ?? NaN,
      values[3] ?? NaN,
      values[4] ?? NaN,
    ];
    const expected = evaluateQuartic(
      prepareQuartic(coefficients, values[5] ?? NaN, values[6] ?? NaN),
      values[7] ?? NaN,
    );
    expect(results[i * 2 + 1]).toBe(1);
    expect(results[i * 2]).toBeCloseTo(expected ?? NaN, 4);
  }
});

// Preserve phase over repeated periods and near m = 1.
test("GPU Jacobi phases agree with binary64 across the real parameter interval", async () => {
  const input = new Float32Array(
    [0, 1e-5, 0.1, 0.5, 0.9, 0.999, regression.m, 1].flatMap((m) =>
      [-50, regression.u, -1, -0.01, 0, 0.3, 2, 8, 25, 50].flatMap((u) => [u, m, 0, 0]),
    ),
  );
  const count = input.length / 4;
  const result = await computeReadback(
    `${source}
    @group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
    @group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) id: vec3u) {
      let value = inputs[id.x];
      outputs[id.x] = vec4f(jacobi(value.x, value.y), 1.0);
    }
  `,
    input,
    count * 4,
    count,
  );
  for (let index = 0; index < count; index++) {
    const expected = jacobi(input[index * 4] ?? Number.NaN, input[index * 4 + 1] ?? Number.NaN);
    for (const [channel, value] of [expected.sn, expected.cn, expected.dn].entries()) {
      expect(
        Math.abs((result[index * 4 + channel] ?? Number.NaN) - value),
        `u=${input[index * 4]}, m=${input[index * 4 + 1]}, component=${channel}`,
      ).toBeLessThan(3e-5);
    }
  }
});

test("a periodic trajectory stays resolved through removable Weierstrass poles", async () => {
  const input = new Float32Array(
    [-2, -1, 0, 1, 2].flatMap((period) =>
      [-1e-5, 0, 1e-5].flatMap((offset) => [4 * Math.PI * period + offset, 0, 0, 0]),
    ),
  );
  const values = await computeReadback(
    `${source}
@group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec2f>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  let path = prepare_quartic(vec4f(1.0,0.0,-1.0,0.0),0.0,0.0,1.0);
  let value = evaluate_quartic(path, inputs[id.x].x);
  outputs[id.x] = vec2f(value.value, f32(value.valid));
}`,
    input,
    input.length / 2,
    input.length / 4,
  );
  for (let i = 0; i < input.length / 4; i++) {
    expect(values[i * 2 + 1]).toBe(1);
    expect(Math.abs((values[i * 2] ?? NaN) - Math.sin(input[i * 4] ?? NaN))).toBeLessThan(3e-6);
  }
});
