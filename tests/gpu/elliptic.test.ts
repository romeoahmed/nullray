import { describe, expect, test } from "vitest";
import regression from "../fixtures/jacobi-dn.json" with { type: "json" };
import { jacobi } from "../reference/elliptic.ts";
import { computeReadback } from "./compute.ts";
import source from "../../src/gpu/wgsl/math/elliptic.wgsl?raw";
import { prepareQuartic, evaluateQuartic } from "../reference/quartic.ts";
import type { Quartic } from "../reference/quartic.ts";
import arithmetic from "../../src/gpu/wgsl/math/binary64.wgsl?raw";
import preparation from "../../src/gpu/wgsl/math/quartic64.wgsl?raw";
import derivatives from "../../src/gpu/wgsl/math/dual64.wgsl?raw";
import differentiated from "../../src/gpu/wgsl/math/cubic-dual64.wgsl?raw";
import { opticalSources } from "../../src/gpu/shaders.ts";

describe("Elliptic trajectories", () => {
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

  test("a retained complementary parameter survives modulus rounding to one", async () => {
    const cases = [25, 30, 36].flatMap((power) => {
      const complement = 2 ** -power;
      const m = 1 - complement;
      return [5, 10, 15].map((u) => ({ m, complement, u: Math.fround(u) }));
    });
    const input = new Float32Array(cases.flatMap(({ m, complement, u }) => [m, complement, u, 0]));
    const output = await computeReadback(
      `${source}
@group(0) @binding(0) var<storage,read> input: array<vec4f>;
@group(0) @binding(1) var<storage,read_write> output: array<vec4f>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u) {
  let sample=input[id.x];
  output[id.x]=vec4f(evaluate_jacobi(prepare_jacobi_pair(sample.x,sample.y),sample.z),1.0);
}`,
      input,
      cases.length * 4,
      cases.length,
    );
    for (const [index, { m, u }] of cases.entries()) {
      const expected = jacobi(u, m);
      for (const [axis, value] of [expected.sn, expected.cn, expected.dn].entries()) {
        expect(
          Math.abs((output[index * 4 + axis] ?? NaN) - value),
          `case ${index}, component ${axis}`,
        ).toBeLessThan(3e-5);
      }
    }
  });
});

describe("Nearly repeated roots", () => {
  const elliptic = source;
  test("integer elliptic preparation matches factored real and complex cubic families", async () => {
    const cases = [{ g2: 0, g3: 0, root: 0, scale: 0, parameter: 0, threeReal: false }];
    for (const power of [-100, -10, 0, 10, 100]) {
      const unit = 2 ** power;
      for (const sign of [-1, 1]) {
        for (const gap of [0, 2 ** -24, 2 ** -16, 2 ** -8, 0.5, 1, 2, 3]) {
          const r = sign * unit;
          const d = gap * unit;
          const [lower = NaN, middle = NaN, upper = NaN] = [r + d, r - d, -2 * r].toSorted(
            (a, b) => a - b,
          );
          cases.push({
            g2: 12 * r * r + 4 * d * d,
            g3: -8 * r * r * r + 8 * r * d * d,
            root: lower,
            scale: upper - lower,
            parameter: (middle - lower) / (upper - lower),
            threeReal: true,
          });
        }
      }
      for (const relativeRoot of [-2, -0.5, 0, 0.5, 2]) {
        const r = relativeRoot * unit;
        const b = unit;
        const scale = Math.hypot(1.5 * r, b);
        cases.push({
          g2: 3 * r * r - 4 * b * b,
          g3: r * r * r + 4 * r * b * b,
          root: r,
          scale,
          parameter: 0.5 - (3 * r) / (4 * scale),
          threeReal: false,
        });
      }
    }
    const input = new Float64Array(cases.flatMap(({ g2, g3 }) => [g2, g3]));
    const output = await computeReadback(
      `${elliptic}\n${arithmetic}\n${preparation}
@group(0) @binding(0) var<storage, read> input: array<vec4u>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2u>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  let e = soft64_elliptic(input[id.x].xy,input[id.x].zw);
  output[id.x*4u] = e.root;
  output[id.x*4u+1u] = e.scale;
  output[id.x*4u+2u] = e.parameter;
  output[id.x*4u+3u] = vec2u(u32(e.valid),u32(e.three_real));
}`,
      new Float32Array(input.buffer),
      cases.length * 8,
      cases.length,
    );
    const values = new Float64Array(output.buffer);
    const words = new Uint32Array(output.buffer);
    const failures = [];
    for (const [index, reference] of cases.entries()) {
      const root = values[index * 4] ?? NaN;
      const scale = values[index * 4 + 1] ?? NaN;
      const parameter = values[index * 4 + 2] ?? NaN;
      const valid = words[index * 8 + 6] === 1;
      const threeReal = words[index * 8 + 7] === 1;
      const tolerance = 2e-12 * Math.max(Math.abs(reference.root), reference.scale);
      if (
        !valid ||
        threeReal !== reference.threeReal ||
        Math.abs(root - reference.root) > tolerance ||
        Math.abs(scale - reference.scale) > tolerance ||
        Math.abs(parameter - reference.parameter) > 2e-12
      ) {
        failures.push({ reference, root, scale, parameter, valid, threeReal });
      }
    }
    expect(failures.length, JSON.stringify(failures.slice(0, 10))).toBe(0);
  });
});

describe("Root derivatives", () => {
  const elliptic = source;
  test("binary64 elliptic derivatives match factored cubic families", async () => {
    const cases = [];
    for (const power of [-20, 0, 20]) {
      const unit = 2 ** power;
      for (const sign of [-1, 1]) {
        for (const gap of [2 ** -12, 2 ** -8, 0.5, 1, 2]) {
          const r = sign * unit;
          const d = gap * unit;
          const [lower, middle, upper] = [
            [r - d, 1, -1],
            [r + d, 1, 1],
            [-2 * r, -2, 0],
          ].toSorted((a, b) => (a[0] ?? NaN) - (b[0] ?? NaN));
          if (!lower || !middle || !upper) {
            throw new Error("Missing cubic roots");
          }
          const scale = upper.map((value, axis) => value - (lower[axis] ?? NaN));
          const span = scale[0] ?? NaN;
          const parameter = ((middle[0] ?? NaN) - (lower[0] ?? NaN)) / span;
          cases.push({
            input: [
              12 * r * r + 4 * d * d,
              24 * r,
              8 * d,
              -8 * r ** 3 + 8 * r * d * d,
              -24 * r * r + 8 * d * d,
              16 * r * d,
            ],
            expected: [
              ...lower,
              ...scale,
              parameter,
              ...[1, 2].map(
                (axis) =>
                  ((middle[axis] ?? NaN) -
                    (lower[axis] ?? NaN) -
                    parameter * (scale[axis] ?? NaN)) /
                  span,
              ),
            ],
            threeReal: true,
            unit,
          });
        }
      }
      for (const relativeRoot of [-2, -0.5, 0, 0.5, 2]) {
        const r = relativeRoot * unit;
        const b = unit;
        const scale = Math.hypot(1.5 * r, b);
        const sr = (2.25 * r) / scale;
        const sb = b / scale;
        cases.push({
          input: [
            3 * r * r - 4 * b * b,
            6 * r,
            -8 * b,
            r ** 3 + 4 * r * b * b,
            3 * r * r + 4 * b * b,
            8 * r * b,
          ],
          expected: [
            r,
            1,
            0,
            scale,
            sr,
            sb,
            0.5 - (0.75 * r) / scale,
            -0.75 / scale + (0.75 * r * sr) / scale ** 2,
            (0.75 * r * sb) / scale ** 2,
          ],
          threeReal: false,
          unit,
        });
      }
    }
    const input = new Float64Array(cases.flatMap((entry) => entry.input));
    const output = await computeReadback(
      `${elliptic}\n${arithmetic}\n${preparation}\n${derivatives}\n${differentiated}
@group(0) @binding(0) var<storage, read> input: array<Dual64>;
@group(0) @binding(1) var<storage, read_write> output: array<Soft64>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  let e = d64_elliptic(input[2u*id.x],input[2u*id.x+1u]);
  let values = array<Dual64,3>(e.root,e.scale,e.parameter);
  for (var i=0u;i<3u;i++) {
    output[id.x*10u+i*3u]=values[i].value;
    output[id.x*10u+i*3u+1u]=values[i].dx;
    output[id.x*10u+i*3u+2u]=values[i].dy;
  }
  output[id.x*10u+9u]=vec2u(u32(e.valid),u32(e.three_real));
}`,
      new Float32Array(input.buffer),
      cases.length * 20,
      cases.length,
    );
    const actual = new Float64Array(output.buffer);
    const words = new Uint32Array(output.buffer);
    for (const [index, entry] of cases.entries()) {
      expect(words[index * 20 + 18], `case ${index} valid`).toBe(1);
      expect(words[index * 20 + 19]).toBe(Number(entry.threeReal));
      for (const [component, expected] of entry.expected.entries()) {
        const naturalScale =
          component < 6
            ? component % 3 === 0
              ? entry.unit
              : 1
            : component === 6
              ? 1
              : 1 / entry.unit;
        expect(
          Math.abs((actual[index * 10 + component] ?? NaN) - expected) / naturalScale,
          `case ${index} component ${component}`,
        ).toBeLessThan(2e-8);
      }
    }
  });
});

describe("Reduced-degree limits", () => {
  test("quadratic limiting paths preserve harmonic and hyperbolic amplitude/frequency derivatives", async () => {
    const cases = [];
    for (const sign of [-1, 1]) {
      for (const frequency of [2 ** -10, 0.7, 3, 256]) {
        for (const amplitude of [0.3, 1.5]) {
          for (const phase of [0.1, 0.8, 2, 5.7]) {
            const offset = 0.2;
            const time = Math.fround(phase / frequency);
            const squared = frequency ** 2;
            const sine = sign < 0 ? Math.sin(frequency * time) : Math.sinh(frequency * time);
            const cosine = sign < 0 ? Math.cos(frequency * time) : Math.cosh(frequency * time);
            cases.push({
              input: [
                squared * (amplitude ** 2 + sign * offset ** 2),
                2 * squared * amplitude,
                2 * frequency * (amplitude ** 2 + sign * offset ** 2),
                -2 * sign * squared * offset,
                0,
                -4 * sign * frequency * offset,
                sign * squared,
                0,
                2 * sign * frequency,
                0,
                0,
                0,
                0,
                0,
                0,
                offset,
                0,
                0,
                amplitude * frequency,
                frequency,
                amplitude,
                time,
                0,
                0,
              ],
              expected: [offset + amplitude * sine, sine, amplitude * time * cosine],
            });
          }
        }
      }
    }
    const input = new Float64Array(cases.flatMap((entry) => entry.input));
    const output = await computeReadback(
      `${opticalSources.derivative}
@group(0) @binding(0) var<storage,read> inputs: array<array<Dual64,8>>;
@group(0) @binding(1) var<storage,read_write> outputs: array<vec4f>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u) {
  let sample=inputs[id.x];
  let c=array<Dual64,5>(sample[0],sample[1],sample[2],sample[3],sample[4]);
  let prepared=d64_prepare_quartic(c,sample[5],sample[6]);
  var valid=prepared.valid;
  let basis=array<Soft64,2>(soft64_number(1.0),vec2u(0u));
  let path=d64_round_quartic(&valid,basis,prepared);
  let time=d64_round(&valid,basis,sample[7]);
  let value=d_evaluate_quartic(&valid,path,time);
  outputs[id.x*2u]=vec4f(value,f32(valid));
  let scalar=soft64_prepare_quartic(array<Soft64,5>(c[0].value,c[1].value,c[2].value,c[3].value,c[4].value),sample[5].value,sample[6].value);
  let primal=evaluate_quartic(scalar.path,time.x);
  outputs[id.x*2u+1u]=vec4f(primal.value,f32(scalar.valid&&primal.valid),0.0,0.0);
}`,
      new Float32Array(input.buffer),
      cases.length * 8,
      cases.length,
    );
    for (const [index, sample] of cases.entries()) {
      const actual = output.subarray(index * 8, index * 8 + 8);
      const label = `case ${index}: ${JSON.stringify(sample)}`;
      expect(actual[3], label).toBe(1);
      expect(actual[5], label).toBe(1);
      for (const [axis, value] of sample.expected.entries()) {
        expect(Math.abs((actual[axis] ?? NaN) - value), label).toBeLessThan(
          2e-5 + 2e-5 * Math.abs(value),
        );
      }
      expect(Math.abs((actual[4] ?? NaN) - (sample.expected[0] ?? NaN)), label).toBeLessThan(
        2e-5 + 2e-5 * Math.abs(sample.expected[0] ?? NaN),
      );
    }
  });

  test("leaving the quadratic family does not inherit a constant elliptic modulus", async () => {
    const output = await computeReadback(
      `${opticalSources.derivative}
@group(0) @binding(0) var<storage,read> input: array<f32>;
@group(0) @binding(1) var<storage,read_write> output: array<f32>;
@compute @workgroup_size(1) fn main() {
  let zero=d64_constant(vec2u(0u));
  let one=d64_constant(soft64_number(1.0));
  let cubic=Dual64(vec2u(0u),soft64_number(input[0]),vec2u(0u));
  let c=array<Dual64,5>(one,zero,d64_constant(soft64_number(-12.0)),cubic,zero);
  output[0]=f32(d64_prepare_quartic(c,zero,one).valid);
}`,
      new Float32Array([1]),
      1,
      1,
    );
    expect(output[0]).toBe(0);
  });
});

describe("Connected radial intervals", () => {
  const elliptic = source;
  test("binary64 radial intervals distinguish barriers from nearly repeated allowed minima", async () => {
    const cases = [];
    for (const power of [-100, 0, 100]) {
      const scale = 2 ** power;
      for (const offset of [-(2 ** -40), 0, 2 ** -40]) {
        for (const x of [0.125, 0.75]) {
          for (const sign of [-1, 1]) {
            // f(u) = ((u² - 1/4)² + offset) * scale. Only negative offsets create a barrier.
            const coefficients = [0.0625 + offset, 0, -0.5, 0, 1].map((value) => value * scale);
            const towardMinimum = (x < 0.5 && sign > 0) || (x > 0.5 && sign < 0);
            const connected = sign > 0 ? 2 : 1;
            const reflected = sign > 0 ? 1 : 2;
            const expected = towardMinimum
              ? offset === 0
                ? 0
                : offset < 0
                  ? reflected
                  : connected
              : connected;
            cases.push({
              input: [
                ...coefficients,
                x,
                sign * Math.sqrt(((x * x - 0.25) ** 2 + offset) * scale),
                1,
              ],
              expected,
            });
          }
        }
      }
    }
    cases.push({ input: [0.0625, 0, -0.5, 0, 1, 0.5, 0, 1], expected: 0 });
    const input = new Float64Array(cases.flatMap((entry) => entry.input));
    const output = await computeReadback(
      `${elliptic}\n${arithmetic}\n${preparation}
struct Case { c: array<Soft64,5>, x:Soft64, velocity:Soft64, horizon:Soft64 }
@group(0) @binding(0) var<storage,read> input: array<Case>;
@group(0) @binding(1) var<storage,read_write> output: array<f32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u) {
  let sample=input[id.x];
  output[id.x]=f32(soft64_radial_destination(sample.c,sample.x,sample.velocity,sample.horizon));
}`,
      new Float32Array(input.buffer),
      cases.length,
      cases.length,
    );
    for (const [index, entry] of cases.entries()) {
      expect(output[index], `case ${index}`).toBe(entry.expected);
    }
  });
});
