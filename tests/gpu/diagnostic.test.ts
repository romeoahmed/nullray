import { expect, test } from "vitest";
import diagnostic from "../../src/render/shaders/diagnostic.wgsl?raw";
import { computeReadback } from "./compute.ts";

test("inspection preserves failed coverage, image order, and signed logarithmic frequency", async () => {
  const endpoints = new Float32Array([
    14, 0, 0, -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 1, 0, 0, 0.5, 1, 0, 0, 1e-30, 1, 6, 0, 2, -3, 6,
    0, 0.5, -4, 6, 0, 1, -5,
  ]);
  const result = await computeReadback(
    `${diagnostic}
    @group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
    @group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
    @compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id: vec3u) {
      outputs[2u * id.x] = diagnostic_endpoint(inputs[id.x], 1.0);
      outputs[2u * id.x + 1u] = diagnostic_endpoint(inputs[id.x], 2.0);
    }`,
    endpoints,
    endpoints.length * 2,
    endpoints.length / 4,
  );
  const color = (index: number, mode: number) =>
    Array.from(result.subarray(index * 8 + mode * 4, index * 8 + mode * 4 + 4));
  const red = Array.from(new Float32Array([0.8, 0.04, 0.015, 1]));
  const blue = Array.from(new Float32Array([0.025, 0.25, 0.8, 1]));
  expect(result.every(Number.isFinite)).toBe(true);
  expect(color(0, 0)).toEqual([0, 0, 0, 0]);
  expect(color(0, 1)).toEqual([0, 0, 0, 0]);
  expect(color(1, 0)).toEqual([0, 0, 0, 1]);
  expect(color(2, 0)).toEqual(Array.from(new Float32Array([0.3, 0.3, 0.3, 1])));
  // The sky stores E=1/g; disk endpoints already store g.
  expect(color(3, 0)).toEqual(red);
  expect(color(4, 0)).toEqual(blue);
  expect(color(5, 0)).toEqual(blue);
  expect(color(6, 0)).toEqual(blue);
  expect(color(7, 0)).toEqual(red);
  expect(color(6, 1)).toEqual(Array.from(new Float32Array([0.1, 0.1, 0.1, 1])));
  expect(color(7, 1)).toEqual(Array.from(new Float32Array([0.02, 0.35, 0.6, 1])));
  expect(color(8, 1)).toEqual(Array.from(new Float32Array([0.8, 0.3, 0.02, 1])));
});
