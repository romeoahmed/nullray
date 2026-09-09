import { expect, test } from "vitest";
import beam from "../../src/render/shaders/beam.wgsl?raw";
import { computeReadback } from "./compute.ts";

const direction = (mu: number, phi: number) => {
  const radius = Math.sqrt((1 - mu) * (1 + mu));
  return [radius * Math.cos(phi), radius * Math.sin(phi), mu];
};

test("endpoint differences retain small angular signals and exact polar limits", async () => {
  const pairs = [
    [0.4, 1.2, 0.4, 1.2 + 2 ** -22],
    [0.4, 40, 0.4 + 2 ** -24, 40 + 2 ** -18],
    [1 - 2 ** -24, 0.7, 1 - 2 ** -23, 0.7 + 2 ** -22],
    [-1 + 2 ** -24, 0.7, -1 + 2 ** -23, 0.7 - 2 ** -22],
    [0, 0, 0, 1e-8],
    [0.3, 400, 0.3 + 2 ** -24, 400 + 2 ** -15],
    [1, 0, 1, 2],
    [1, 0, -1, 2],
    [1, 0, 1 - 2 ** -24, 2],
    [0.3, -3.14, 0.3, 3.14],
  ];
  const input = new Float32Array(pairs.flat());
  const result = await computeReadback(
    `${beam}
    @group(0) @binding(0) var<storage, read> data: array<vec4f>;
    @group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
    @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
      let pair = data[id.x];
      output[2u * id.x] = vec4f(celestial_difference(vec4f(pair.zw, 1.0, 1.0), vec4f(pair.xy, 1.0, 1.0)), 1.0);
      output[2u * id.x + 1u] = vec4f(celestial_direction(vec4f(pair.zw, 1.0, 1.0)) - celestial_direction(vec4f(pair.xy, 1.0, 1.0)), 1.0);
    }`,
    input,
    pairs.length * 8,
    pairs.length,
  );
  let largestDirectError = 0;
  for (let i = 0; i < pairs.length; i++) {
    const before = direction(input[i * 4] ?? NaN, input[i * 4 + 1] ?? NaN);
    const after = direction(input[i * 4 + 2] ?? NaN, input[i * 4 + 3] ?? NaN);
    const expected = after.map((x, j) => x - (before[j] ?? NaN));
    const error = expected.map((x, j) => x - (result[i * 8 + j] ?? NaN));
    expect(Math.hypot(...error)).toBeLessThanOrEqual(3e-5 * Math.hypot(...expected) + 1e-15);
    if (Math.hypot(...expected) > 0) {
      const direct = expected.map((x, j) => x - (result[i * 8 + 4 + j] ?? NaN));
      largestDirectError = Math.max(
        largestDirectError,
        Math.hypot(...direct) / Math.hypot(...expected),
      );
    }
  }
  expect(largestDirectError).toBeGreaterThan(0.01);
});
