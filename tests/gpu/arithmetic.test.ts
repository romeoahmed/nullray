import { createTestFrame } from "../support/frames.ts";
import observer64 from "../../src/gpu/wgsl/geodesics/observer64.wgsl?raw";
import { expect, test } from "vitest";
import * as fc from "fast-check";
import source from "../../src/gpu/wgsl/math/binary64.wgsl?raw";
import launch from "../../src/gpu/wgsl/geodesics/launch64.wgsl?raw";
import fixture from "../fixtures/critical-primary-rays.json" with { type: "json" };
import { referenceRay } from "../reference/sky.ts";
import { computeReadback, observerData } from "./compute.ts";

test("integer binary64 arithmetic matches native rounding and rejects unsupported results", async () => {
  const normal = fc
    .tuple(
      fc.integer({ min: 0, max: 0xffffffff }),
      fc.integer({ min: 0, max: 0xfffff }),
      fc.integer({ min: 1, max: 2046 }),
      fc.boolean(),
    )
    .map(([low, fraction, exponent, negative]) => {
      const words = new Uint32Array([
        low,
        fraction | (exponent << 20) | (negative ? 0x80000000 : 0),
      ]);
      return new Float64Array(words.buffer)[0] ?? NaN;
    });
  const random = fc.sample(fc.tuple(normal, normal), { seed: 0x641053, numRuns: 4096 });
  const values = [
    0,
    -0,
    1,
    -1,
    2,
    1 + 2 ** -52,
    1 - 2 ** -53,
    2 ** -1022,
    Number.MAX_VALUE,
    2 ** -500,
    2 ** 500,
    Math.PI,
    -Math.PI,
    1e-20,
    -1e20,
    Number.MIN_VALUE,
    Infinity,
    -Infinity,
    NaN,
  ];
  const edge = values.flatMap((a) => values.map((b) => [a, b] as const));
  const cancellation = random.flatMap(([a]) => [
    [a, -a] as const,
    [a, -a * (1 - 2 ** -52)] as const,
  ]);
  const squareBoundaries = [2 ** 23, 2 ** 23 + 1, 2 ** 24 - 1].flatMap((root) =>
    [-1, 0, 1].map((offset) => [root * root + offset, root] as const),
  );
  const pairs = [...edge, ...random, ...cancellation, ...squareBoundaries];
  const inputs = new Float64Array(pairs.flatMap(([a, b]) => [a, b]));
  const result = await computeReadback(
    `${source}
@group(0) @binding(0) var<storage, read> inputs: array<vec4u>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec2u>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  let pair = inputs[id.x];
  outputs[5u * id.x] = soft64_add(pair.xy, pair.zw);
  outputs[5u * id.x + 1u] = soft64_subtract(pair.xy, pair.zw);
  outputs[5u * id.x + 2u] = soft64_multiply(pair.xy, pair.zw);
  outputs[5u * id.x + 3u] = soft64_divide(pair.xy, pair.zw);
  outputs[5u * id.x + 4u] = soft64_sqrt(pair.xy);
}`,
    new Float32Array(inputs.buffer),
    pairs.length * 10,
    pairs.length,
  );
  const actual = new Float64Array(result.buffer);
  const words = new Uint32Array(result.buffer);
  const failures = [];
  for (let index = 0; index < pairs.length; index++) {
    const pair = pairs[index];
    if (!pair) {
      throw new Error("Missing arithmetic inputs");
    }
    const [a, b] = pair;
    const expected = [a + b, a - b, a * b, a / b, Math.sqrt(a)];
    for (let operation = 0; operation < expected.length; operation++) {
      const value = expected[operation] ?? NaN;
      const offset = index * 5 + operation;
      const unsupported =
        !Number.isFinite(a) ||
        (a !== 0 && Math.abs(a) < 2 ** -1022) ||
        (operation !== 4 && (!Number.isFinite(b) || (b !== 0 && Math.abs(b) < 2 ** -1022))) ||
        !Number.isFinite(value) ||
        (value !== 0 && Math.abs(value) < 2 ** -1022) ||
        ((operation === 2 || operation === 3) && a !== 0 && b !== 0 && value === 0);
      const matches = unsupported
        ? words[offset * 2] === 0 && words[offset * 2 + 1] === 0x7ff80000
        : actual[offset] === value;
      if (!matches) {
        failures.push({ a, b, operation, expected: value, actual: actual[offset] });
      }
    }
  }
  expect(failures.length, JSON.stringify(failures.slice(0, 10))).toBe(0);
});

test("integer binary64 launch retains critical and axis camera precision", async () => {
  const initial = fixture.cases[0];
  if (!initial) {
    throw new Error("Missing launch fixture");
  }
  const samples = [
    ...fixture.cases.map((sample) => ({ sample, inclination: 1.2 })),
    ...[0, Math.PI, 1e-6, Math.PI - 1e-6].map((inclination) => ({ sample: initial, inclination })),
  ];
  const frames = samples.map(({ sample: { width, height }, inclination }) => {
    const frame = createTestFrame();
    frame[0] = width;
    frame[1] = height;
    frame[5] = inclination;
    return frame;
  });
  const observers = new Float64Array(frames.flatMap((frame) => Array.from(observerData(frame))));
  const output = await computeReadback(
    `${source}\n${observer64}\n${launch}
@group(0) @binding(0) var<storage, read> pixels: array<vec2u>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec2u>;
@group(0) @binding(2) var<storage, read> frames: array<array<u32, 24>>;
@group(0) @binding(3) var<storage, read> observers: array<Observer64>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  let photon = soft64_screen_launch(frames[id.x], pixels[id.x], observers[id.x]);
  let base = 8u * id.x;
  outputs[base] = photon.source[0];
  outputs[base + 1u] = photon.source[1];
  outputs[base + 2u] = photon.source[2];
  outputs[base + 3u] = photon.energy;
  outputs[base + 4u] = photon.angular_momentum;
  outputs[base + 5u] = photon.carter;
  outputs[base + 6u] = photon.polar_velocity;
  outputs[base + 7u] = photon.inverse_radial_velocity;
}`,
    new Float32Array(samples.flatMap(({ sample: { x, y } }) => [x, y])),
    samples.length * 16,
    samples.length,
    [
      new Float32Array(frames.flatMap((frame) => Array.from(frame))),
      new Float32Array(observers.buffer),
    ],
  );
  const actual = new Float64Array(output.buffer);
  for (const [index, { sample }] of samples.entries()) {
    const frame = frames[index];
    if (!frame) {
      throw new Error("Missing critical frame");
    }
    const { photon } = referenceRay(sample.x, sample.y, frame);
    const expected = [
      ...sample.hamiltonian.source,
      photon.energy,
      photon.angularMomentum,
      photon.carter,
      photon.polarVelocity,
      photon.radialVelocity / (frame[4] ?? NaN) ** 2,
    ];
    for (const [component, value] of expected.entries()) {
      const error = Math.abs((actual[index * 8 + component] ?? NaN) - value);
      expect(error, `sample ${index}, component ${component}`).toBeLessThan(
        2e-14 * Math.max(1, Math.abs(value)),
      );
    }
  }
});

test("binary32 bit conversion preserves finite values, subnormals, and rounding ties", async () => {
  const raw = fc.sample(fc.integer({ min: 0, max: 0xffffffff }), { seed: 0x6432, numRuns: 4096 });
  const floatBits = [
    0,
    0x80000000,
    1,
    0x007fffff,
    0x00800000,
    0x7f7fffff,
    0x7f800000,
    0x7fc00000,
    ...raw,
  ];
  const singles = new Float32Array(new Uint32Array(floatBits).buffer);
  const doubleValues = floatBits.map((_, index) => {
    const value = singles[index] ?? NaN;
    return value * (1 + 2 ** -24);
  });
  const doubles = new Float64Array(doubleValues);
  const doubleWords = new Uint32Array(doubles.buffer);
  const input = new Uint32Array(
    floatBits.flatMap((bits, index) => [
      doubleWords[index * 2] ?? 0,
      doubleWords[index * 2 + 1] ?? 0,
      bits,
      0,
    ]),
  );
  const result = await computeReadback(
    `${source}
@group(0) @binding(0) var<storage, read> inputs: array<vec4u>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4u>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  let value = inputs[id.x];
  outputs[id.x] = vec4u(soft64_from_f32_bits(value.z), soft64_to_f32_bits(value.xy), 0u);
}`,
    new Float32Array(input.buffer),
    input.length,
    floatBits.length,
  );
  const outputWords = new Uint32Array(result.buffer);
  const outputDoubles = new Float64Array(result.buffer);
  const expectedSingle = new Float32Array(1);
  const expectedBits = new Uint32Array(expectedSingle.buffer);
  const failures = [];
  for (let index = 0; index < floatBits.length; index++) {
    const original = singles[index] ?? NaN;
    const converted = outputDoubles[index * 2];
    const correctDecode = Number.isFinite(original)
      ? Object.is(converted, original)
      : outputWords[index * 4] === 0 && outputWords[index * 4 + 1] === 0x7ff80000;
    expectedSingle[0] = doubles[index] ?? NaN;
    const rounded = expectedSingle[0] ?? NaN;
    const correctEncode =
      outputWords[index * 4 + 2] === (Number.isFinite(rounded) ? expectedBits[0] : 0x7fc00000);
    if (!correctDecode || !correctEncode) {
      failures.push({ index, original, converted, correctDecode, correctEncode });
    }
  }
  expect(failures.length, JSON.stringify(failures.slice(0, 10))).toBe(0);
});
