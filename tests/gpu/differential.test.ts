import derivative from "../../src/render/shaders/derivative.wgsl?raw";
import { expect, test } from "vitest";
import differential from "../../src/render/shaders/differential.wgsl?raw";
import { opticalSources } from "../../src/render/optics.ts";
import { carlsonRF, jacobi } from "../reference/elliptic.ts";
import { prepareQuartic, evaluateQuartic } from "../reference/quartic.ts";
import { computeReadback } from "./compute.ts";

const prefix = `${opticalSources.ray}\n${derivative}\n${differential}
@group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;`;

const check = (actual: number, expected: number, relative = 5e-4) => {
  expect(Number.isFinite(actual)).toBe(true);
  expect(Math.abs(actual - expected)).toBeLessThan(2e-5 + relative * Math.abs(expected));
};

test("Jacobi forward derivatives agree with binary64 parameter and phase differences", async () => {
  const input = new Float32Array(
    [0.01, 0.3, 0.8, 0.98].flatMap((m) => [-2, 0.1, 1, 3].flatMap((u) => [u, m, 0, 0])),
  );
  const values = await computeReadback(
    `${prefix}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  var derivative_valid = true;
  let status = &derivative_valid;
  let p = inputs[id.x];
  let j = d_jacobi(status, vec3f(p.x, 1.0, 0.0), vec3f(p.y, 0.0, 1.0));
  outputs[id.x * 3u] = vec4f(j.sn, f32((*status)));
  outputs[id.x * 3u + 1u] = vec4f(j.cn, f32((*status)));
  outputs[id.x * 3u + 2u] = vec4f(j.dn, f32((*status)));
}`,
    input,
    input.length * 3,
    input.length / 4,
  );
  for (let index = 0; index < input.length / 4; index++) {
    const u = input[index * 4] ?? NaN;
    const m = input[index * 4 + 1] ?? NaN;
    const h = 1e-5;
    const center = jacobi(u, m);
    const beforeU = jacobi(u - h, m),
      afterU = jacobi(u + h, m);
    const beforeM = jacobi(u, m - h),
      afterM = jacobi(u, m + h);
    for (const [channel, name] of (["sn", "cn", "dn"] as const).entries()) {
      const offset = index * 12 + channel * 4;
      expect(values[offset + 3]).toBe(1);
      check(values[offset] ?? NaN, center[name]);
      check(values[offset + 1] ?? NaN, (afterU[name] - beforeU[name]) / (2 * h));
      check(values[offset + 2] ?? NaN, (afterM[name] - beforeM[name]) / (2 * h));
    }
  }
});

const reference = (constant: number, time: number) => {
  const x = Math.fround(0.05);
  const f = constant + x * x * (-10 + x * (15 - x));
  return evaluateQuartic(prepareQuartic([constant, 0, -10, 15, -1], x, Math.sqrt(f)), time) ?? NaN;
};

test.for([1, 100])(
  "quartic forward derivatives retain coefficient and time dependence at momentum scale %i",
  async (scale) => {
    const x = Math.fround(0.05);
    const elliptic = prepareQuartic(
      [1, 0, -10, 15, -1],
      x,
      Math.sqrt(1 + x * x * (-10 + x * (15 - x))),
    ).elliptic;
    const period = (2 * carlsonRF(0, 1 - elliptic.parameter, 1)) / Math.sqrt(elliptic.scale);
    const input = new Float32Array(
      [0, 0.05, 0.2, 0.5, 0.8, period - 1e-5, period, period + 1e-5].flatMap((time) => [
        time,
        scale,
        0,
        0,
      ]),
    );
    const values = await computeReadback(
      `${prefix}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  var derivative_valid = true;
  let status = &derivative_valid;
  let scale = inputs[id.x].y;
  let squared = scale * scale;
  let c = array<vec3f, 5>(vec3f(squared, squared, 0.0), d_constant(status, 0.0), d_constant(status, -10.0 * squared), d_constant(status, 15.0 * squared), d_constant(status, -squared));
  let x = d_constant(status, 0.05);
  let f = d_add(status, c[0], d_mul(status, d_mul(status, x, x), d_add(status, c[2], d_mul(status, x, d_add(status, c[3], d_mul(status, c[4], x))))));
  let path = d_prepare_quartic(status, c, x, d_sqrt(status, f));
  let value = d_evaluate_quartic(status, path, vec3f(inputs[id.x].x / scale, 0.0, 1.0 / scale));
  outputs[id.x] = vec4f(value, f32((*status)));
}`,
      input,
      input.length,
      input.length / 4,
    );

    for (let offset = 0; offset < input.length; offset += 4) {
      const time = input[offset] ?? NaN;
      const h = 1e-5;
      expect(values[offset + 3]).toBe(1);
      check(values[offset] ?? NaN, reference(1, time));
      check(values[offset + 1] ?? NaN, (reference(1 + h, time) - reference(1 - h, time)) / (2 * h));
      check(values[offset + 2] ?? NaN, (reference(1, time + h) - reference(1, time - h)) / (2 * h));
    }
  },
);

test("differential arithmetic rejects undefined and over-budget derivatives before evaluation", async () => {
  const result = await computeReadback(
    `${prefix}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  var derivative_valid = true;
  let status = &derivative_valid;
  var value = vec3f(0.0);
  switch u32(inputs[id.x].x) {
    case 0u: { value = d_div(status, vec3f(1.0, 1.0, 0.0), d_constant(status, 0.0)); }
    case 1u: { value = d_sqrt(status, vec3f(-1.0, 1.0, 0.0)); }
    case 2u: { value = d_acos(status, vec3f(1.1, 1.0, 0.0)); }
    case 3u: { value = d_mul(status, d_constant(status, 1e20), d_constant(status, 1e20)); }
    case 4u: { value = d_sqrt(status, vec3f(0.0, 1.0, 0.0)); }
    case 5u: { value = d_sqrt(status, d_constant(status, 0.0)); }
    case 6u: { value = d_mul(status, vec3f(1e10, 1e10, 0.0), vec3f(1e10, 1e10, 0.0)); }
    case 7u: { value = d_div(status, vec3f(1.0, 0.0, 1.0), d_constant(status, 1e-35)); }
    case 8u: { value = d_carlson_rf(status, d_constant(status, -1.0), d_constant(status, 1.0), d_constant(status, 1.0)); }
    case 9u: { value = d_carlson_rf(status, d_constant(status, 0.0), d_constant(status, 0.0), d_constant(status, 1.0)); }
    case 10u: { value = d_carlson_rf(status, vec3f(0.0, 1.0, 0.0), d_constant(status, 1.0), d_constant(status, 1.0)); }
    case 11u: { value = d_incomplete_f(status, d_constant(status, 0.1), d_constant(status, 1.0)); }
    default: { value = d_incomplete_f(status, d_constant(status, 0.1), d_constant(status, -0.1)); }
  }
  outputs[id.x] = vec4f(value, f32((*status)));
}`,
    new Float32Array(Array.from({ length: 13 }, (_, index) => [index, 0, 0, 0]).flat()),
    52,
    13,
  );
  expect(result.every(Number.isFinite)).toBe(true);
  for (let index = 0; index < 13; index++) {
    expect(result[index * 4 + 3]).toBe(index === 5 || index === 6 ? 1 : 0);
  }
  expect(Math.abs((result[24] ?? NaN) / 1e20 - 1)).toBeLessThan(1e-6);
  expect(Math.abs((result[25] ?? NaN) / 2e20 - 1)).toBeLessThan(1e-6);
});

test("azimuth-meridian derivatives agree with binary64 differences in each quadrant", async () => {
  const input = new Float32Array(
    [
      [1, 2],
      [-1, 2],
      [1, -2],
      [-1, -2],
      [0, 1],
      [1, 0],
    ].flatMap(([y, x]) => [y ?? NaN, x ?? NaN, 0, 0]),
  );
  const result = await computeReadback(
    `${prefix}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  var derivative_valid = true;
  let status = &derivative_valid;
  let value = d_atan2(status, vec3f(inputs[id.x].x, 1.0, 0.0), vec3f(inputs[id.x].y, 0.0, 1.0));
  outputs[id.x] = vec4f(value, f32((*status)));
}`,
    input,
    input.length,
    input.length / 4,
  );
  for (let offset = 0; offset < input.length; offset += 4) {
    const y = input[offset] ?? NaN,
      x = input[offset + 1] ?? NaN;
    const h = 1e-5;
    expect(result[offset + 3]).toBe(1);
    check(result[offset] ?? NaN, Math.atan2(y, x));
    check(result[offset + 1] ?? NaN, (Math.atan2(y + h, x) - Math.atan2(y - h, x)) / (2 * h));
    check(result[offset + 2] ?? NaN, (Math.atan2(y, x + h) - Math.atan2(y, x - h)) / (2 * h));
  }
});

/**
 * Direct quadrature of the defining integral and its parameter derivative.
 * No Carlson duplication or amplitude reduction is shared with the shader.
 */
const ellipticIntegralReference = (angle: number, m: number, count: number) => {
  const sums = [0, 0];
  for (let i = 0; i <= count; i++) {
    const sine2 = Math.sin((angle * i) / count) ** 2;
    const d = 1 - m * sine2;
    const weight = i === 0 || i === count ? 1 : i % 2 === 0 ? 2 : 4;
    sums[0] = (sums[0] ?? 0) + weight / Math.sqrt(d);
    sums[1] = (sums[1] ?? 0) + (weight * sine2) / (2 * d ** 1.5);
  }
  return sums.map((value) => (value * angle) / (3 * count));
};

test("incomplete elliptic derivatives retain quarter-period and modulus limits", async () => {
  const input = new Float32Array(
    [0, 0.01, 0.8, 0.98].flatMap((m) =>
      [-7, -Math.PI, -Math.PI / 2, -0.1, 0, Math.PI / 2, Math.PI, 7].flatMap((angle) => [
        angle,
        m,
        0,
        0,
      ]),
    ),
  );
  const values = await computeReadback(
    `${prefix}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  var derivative_valid = true;
  let status = &derivative_valid;
  let p = inputs[id.x];
  let result = d_incomplete_f(status, vec3f(p.x, 1.0, 0.0), vec3f(p.y, 0.0, 1.0));
  outputs[id.x] = vec4f(result, f32((*status)));
}`,
    input,
    input.length,
    input.length / 4,
  );
  for (let i = 0; i < input.length; i += 4) {
    const angle = input[i] ?? NaN,
      m = input[i + 1] ?? NaN;
    const fine = ellipticIntegralReference(angle, m, 16384),
      coarse = ellipticIntegralReference(angle, m, 8192);
    expect(values[i + 3]).toBe(1);
    for (const axis of [0, 1]) {
      expect(Math.abs((fine[axis] ?? NaN) - (coarse[axis] ?? NaN))).toBeLessThan(1e-8);
    }
    check(values[i] ?? NaN, fine[0] ?? NaN, 2e-5);
    check(values[i + 1] ?? NaN, 1 / Math.sqrt(1 - m * Math.sin(angle) ** 2), 2e-5);
    check(values[i + 2] ?? NaN, fine[1] ?? NaN, 2e-5);
  }
});

test("azimuth derivatives remain scale invariant across finite f32 magnitudes", async () => {
  const input = new Float32Array([1e-20, 1, 1e20].flatMap((scale) => [scale, 0, 0, 0]));
  const values = await computeReadback(
    `${prefix}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  var valid = true;
  let scale = inputs[id.x].x;
  let angle = d_atan2(&valid, vec3f(scale, scale, 0.0), vec3f(2.0 * scale, 0.0, scale));
  outputs[id.x] = vec4f(angle, f32(valid));
}`,
    input,
    input.length,
    input.length / 4,
  );
  for (let offset = 0; offset < values.length; offset += 4) {
    expect(values[offset + 3]).toBe(1);
    check(values[offset] ?? NaN, Math.atan2(1, 2), 2e-6);
    check(values[offset + 1] ?? NaN, 2 / 5, 2e-6);
    check(values[offset + 2] ?? NaN, -1 / 5, 2e-6);
  }
});

test("a failed derivative does not contaminate a separately owned evaluation", async () => {
  const values = await computeReadback(
    `${prefix}
@compute @workgroup_size(1) fn main() {
  var failed_evaluation = true;
  var independent_evaluation = true;
  let bad = d_sqrt(&failed_evaluation, inputs[0].xyz);
  let good = d_sqrt(&independent_evaluation, inputs[1].xyz);
  let subsequent = d_sqrt(&failed_evaluation, inputs[1].xyz);
  outputs[0] = vec4f(bad, f32(failed_evaluation));
  outputs[1] = vec4f(good, f32(independent_evaluation));
  outputs[2] = vec4f(subsequent, f32(failed_evaluation));
}`,
    new Float32Array([-1, 1, 0, 0, 4, 1, 0, 0]),
    12,
    1,
  );
  expect(Array.from(values)).toEqual([0, 0, 0, 0, 2, 0.25, 0, 1, 2, 0.25, 0, 0]);
});
