import { createTestFrame } from "../support/frames.ts";
import { describe, expect, test } from "vitest";
import derivative from "../../src/gpu/wgsl/math/dual32.wgsl?raw";
import differential from "../../src/gpu/wgsl/math/quartic-dual32.wgsl?raw";
import { opticalSources } from "../../src/gpu/optics.ts";
import { carlsonRF, jacobi } from "../reference/elliptic.ts";
import { prepareQuartic, evaluateQuartic } from "../reference/quartic.ts";
import { computeReadback, observerData } from "./compute.ts";
import cooperative from "../../src/gpu/wgsl/lensing/sky-integral.wgsl?raw";
import { server } from "vitest/browser";
import imageSource from "../../src/gpu/wgsl/lensing/sky-jacobian32.wgsl?raw";
import { cross, dot } from "../../src/physics/vector.ts";
import type { Vec3 } from "../../src/physics/vector.ts";
import { referenceRay, referenceSky } from "../reference/sky.ts";
import boundaryDerivatives from "../fixtures/critical-derivative-rays.json" with { type: "json" };
import fixture from "../fixtures/critical-primary-rays.json" with { type: "json" };
import { referenceSkyArea, referenceSkyGradient } from "../reference/sky-derivative.ts";

const check = (actual: number, expected: number, relative = 5e-4) => {
  expect(Number.isFinite(actual)).toBe(true);
  expect(Math.abs(actual - expected)).toBeLessThan(2e-5 + relative * Math.abs(expected));
};

const referenceTrajectory = (constant: number, time: number) => {
  const x = Math.fround(0.05);
  const f = constant + x * x * (-10 + x * (15 - x));
  return evaluateQuartic(prepareQuartic([constant, 0, -10, 15, -1], x, Math.sqrt(f)), time) ?? NaN;
};

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

function cartesianGradient(mu: number, phi: number, dmu: number, dphi: number): Vec3 {
  const radius = Math.sqrt(1 - mu * mu);
  const dr = (-mu * dmu) / radius;
  return [
    dr * Math.cos(phi) - radius * Math.sin(phi) * dphi,
    dr * Math.sin(phi) + radius * Math.cos(phi) * dphi,
    dmu,
  ];
}

const relativeVectorError = (actual: Vec3, reference: Vec3) =>
  Math.hypot(...actual.map((value, axis) => value - (reference[axis] ?? NaN))) /
  Math.hypot(...reference);

const difference = (a: Vec3, b: Vec3) =>
  Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) / Math.hypot(...b);

describe("Analytic path gradients", () => {
  const prefix = `${opticalSources.ray}\n${derivative}\n${differential}
@group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;`;

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
        check(values[offset] ?? NaN, referenceTrajectory(1, time));
        check(
          values[offset + 1] ?? NaN,
          (referenceTrajectory(1 + h, time) - referenceTrajectory(1 - h, time)) / (2 * h),
        );
        check(
          values[offset + 2] ?? NaN,
          (referenceTrajectory(1, time + h) - referenceTrajectory(1, time - h)) / (2 * h),
        );
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
});

describe("Sky solid angle", () => {
  const referenceFrame = createTestFrame();

  const ordinaryAndCritical = [
    ...(
      [
        [524.125, 360.125],
        [524.15625, 360.125],
        [524.125, 360.09375],
        [524.125, 360.15625],
        [524.15625, 360.15625],
      ] as const
    ).flatMap(([x, y]) =>
      (
        [
          [0, 0],
          [0.0073, 0.0119],
          [-0.0061, -0.0107],
        ] as const
      ).map(([dx, dy]) => [x + dx, y + dy] as const),
    ),
    [0, 0],
    [1279, 0],
    [0, 719],
    [1279, 719],
  ] as const;

  function axisFrame(theta: number) {
    const frame = referenceFrame.slice();
    frame[5] = theta;
    frame[12] = 1;
    frame[17] = 1;
    return frame;
  }

  const scatteringFrame = referenceFrame.slice();
  scatteringFrame[6] = 2.4;

  test.for([
    { name: "charged", frameData: referenceFrame, positions: ordinaryAndCritical },
    {
      name: "meridian-scattering",
      frameData: scatteringFrame,
      positions: [0, 100].flatMap((y) =>
        [-1, -0.01, 0, 0.01, 1].map((offset) => [639.5 + offset, y] as const),
      ),
    },
    ...[0.1, 1.2, 3.04].map((theta) => ({
      name: `meridian-${theta}`,
      frameData: axisFrame(theta),
      positions: [
        ...(theta === 1.2 ? [335, 350, 370, 450, 600] : [450, 600]).flatMap((y) =>
          [-1, -0.01, -1 / 16384, 0, 1 / 16384, 0.01, 1].map(
            (offset) => [639.5 + offset, y] as const,
          ),
        ),
        ...(theta === 1.2 ? [400, 700, 900].map((x) => [x, 359.5] as const) : []),
      ],
    })),
    {
      name: "north",
      frameData: axisFrame(0),
      positions: [
        [400, 200],
        [700, 500],
      ],
    },
    {
      name: "south",
      frameData: axisFrame(Math.PI),
      positions: [
        [400, 200],
        [700, 500],
      ],
    },
  ] as const)(
    "$name cooperative sky derivatives agree with refined binary64 screen differences",
    async ({ name, frameData, positions }) => {
      const input = new Float32Array(positions.flatMap(([x, y]) => [x, y, 0, 0]));
      const frame = Array.from(
        { length: 6 },
        (_, i) => `vec4f(${Array.from(frameData.subarray(i * 4, i * 4 + 4)).join(",")})`,
      ).join(",");
      const result = await computeReadback(
        `enable subgroups;\n${opticalSources.ray}\n${derivative}\n${differential}\n${imageSource}\n${cooperative}
@group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
var<workgroup> prepared: DifferentialSkyPath;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) id: vec3u, @builtin(local_invocation_index) lane: u32) {
  let frame = Frame(${frame});
  if (lane == 0u) {
    let endpoint = screen_ray(frame, inputs[id.x].xy).value;
    prepared = prepare_differential_sky(frame, inputs[id.x].xy, endpoint.w);
    outputs[id.x * 3u + 2u] = endpoint;
  }
  let path = workgroupUniformLoad(&prepared);
  let sample = cooperative_sky(frame, path, lane);
  if (lane == 0u) {
    outputs[id.x * 3u] = vec4f(sample.mu, f32(sample.valid));
    outputs[id.x * 3u + 1u] = vec4f(sample.phi, sample.energy);
  }
}`,
        input,
        positions.length * 12,
        positions.length,
      );
      const records = positions.map((_, index) => {
        const x = input[index * 4] ?? NaN,
          y = input[index * 4 + 1] ?? NaN;
        const output = Array.from(result.subarray(index * 12, index * 12 + 12));
        const value = (offset: number) => output[offset] ?? NaN;
        const outcome = referenceRay(x, y, frameData).outcome;
        if (outcome.kind !== "sky") {
          return { kind: "occluded" as const, position: [x, y], outcome, output };
        }
        const reference = referenceSky(x, y, frameData);
        const fine = [
          referenceSkyGradient(x, y, 0, 1 / 8192, frameData),
          referenceSkyGradient(x, y, 1, 1 / 8192, frameData),
        ] as const;
        const coarse = [
          referenceSkyGradient(x, y, 0, 1 / 4096, frameData),
          referenceSkyGradient(x, y, 1, 1 / 4096, frameData),
        ] as const;
        const gradients = [
          cartesianGradient(value(0), value(4), value(1), value(5)),
          cartesianGradient(value(0), value(4), value(2), value(6)),
        ] as const;
        const area = Math.abs(value(1) * value(6) - value(2) * value(5));
        const referenceArea = Math.abs(dot(reference.direction, cross(...fine)));
        return {
          kind: "sky" as const,
          position: [x, y],
          output,
          reference,
          gradients,
          fine,
          coarse,
          area,
          referenceArea,
        };
      });
      await server.commands.writeFile(
        `test-results/stellar-differential-${name}.json`,
        JSON.stringify({ frame: Array.from(frameData), records }, null, 2),
      );
      for (const record of records) {
        const label = JSON.stringify(record);
        if (record.kind === "occluded") {
          expect(["disk", "captured"], label).toContain(record.outcome.kind);
          expect(record.output[3], label).toBe(0);
          continue;
        }
        expect(record.output[3], label).toBe(1);
        for (const axis of [0, 1] as const) {
          expect(relativeVectorError(record.fine[axis], record.coarse[axis]), label).toBeLessThan(
            1e-5,
          );
          expect(
            relativeVectorError(record.gradients[axis], record.fine[axis]),
            label,
          ).toBeLessThan(0.002);
        }
        expect(Math.abs(record.area / record.referenceArea - 1), label).toBeLessThan(0.002);
      }
    },
  );
});

describe("Retained critical gradients", () => {
  test.for(["adaptive", "precise"] as const)(
    "%s sky transport retains critical direction gradients and solid angle",
    async (mode) => {
      const cases = [
        ...fixture.cases
          .filter((sample) => sample.outcome.kind === "sky")
          .map(({ width, height, x, y }) => ({ width, height, x, y, frame: fixture.frame })),
        ...boundaryDerivatives.cases.map(({ width, height, x, y }) => ({
          width,
          height,
          x,
          y,
          frame: boundaryDerivatives.frame,
        })),
      ];
      const frames = cases.map(({ width, height, frame: values }) => {
        const frame = createTestFrame(values);
        frame[0] = width;
        frame[1] = height;
        return frame;
      });
      const observers = new Float64Array(
        frames.flatMap((frame) => Array.from(observerData(frame))),
      );
      const result = await computeReadback(
        `${opticalSources.beam}
@group(0) @binding(0) var<storage, read> pixels: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
@group(0) @binding(2) var<storage, read> frames: array<Frame>;
@group(0) @binding(3) var<storage, read> observers: array<Observer64>;
var<workgroup> prepared: PreparedSkyBeam;
@compute @workgroup_size(32) fn main(@builtin(workgroup_id) id: vec3u,@builtin(local_invocation_index) lane:u32) {
  let frame=frames[id.x];
  if (lane==0u) {
    let endpoint=screen_ray_refined(frame,pixels[id.x],observers[id.x]).value;
    prepared=${mode === "adaptive" ? "prepare_sky_beam" : "prepare_differential_sky64"}(frame,pixels[id.x],endpoint.w,observers[id.x]);
    outputs[id.x*4u+2u]=endpoint;
  }
  let preparation=workgroupUniformLoad(&prepared);
  let path=preparation.path;
  let sample=${mode === "adaptive" ? "evaluate_prepared_sky(frame,preparation,lane)" : "cooperative_sky_budget(frame,path,lane,16384u)"};
  if (lane==0u) {
    let beam=beam_from_prepared_sky(sample,preparation);
    outputs[id.x*4u]=beam.dx;
    outputs[id.x*4u+1u]=beam.dy;
    outputs[id.x*4u+3u]=vec4f(path.end,f32(path.valid));
  }
}`,
        new Float32Array(cases.flatMap(({ x, y }) => [x, y])),
        cases.length * 16,
        cases.length,
        [
          new Float32Array(frames.flatMap((frame) => Array.from(frame))),
          new Float32Array(observers.buffer),
        ],
        "main",
      );
      const records = cases.map(({ x, y }, index) => {
        const frame = frames[index];
        if (!frame) {
          throw new Error("Missing critical frame");
        }
        const at = (offset: number) => result[index * 16 + offset] ?? NaN;
        const fine = [
          referenceSkyGradient(x, y, 0, 1 / 524288, frame),
          referenceSkyGradient(x, y, 1, 1 / 524288, frame),
        ] as const;
        const coarse = [
          referenceSkyGradient(x, y, 0, 1 / 262144, frame),
          referenceSkyGradient(x, y, 1, 1 / 262144, frame),
        ] as const;
        const actual = [
          [at(0), at(1), at(2)],
          [at(4), at(5), at(6)],
        ] as const;
        const area = Math.abs(dot(referenceSky(x, y, frame).direction, cross(...actual)));
        const referenceArea = referenceSkyArea(x, y, frame, fine);
        return {
          pixel: [x, y],
          referenceArea,
          output: Array.from(result.subarray(index * 16, index * 16 + 16)),
          preparationResolved: at(15) === 1,
          transportResolved: at(3) === 1,
          convergence: fine.map((value, axis) =>
            difference(value, coarse[axis] ?? [NaN, NaN, NaN]),
          ),
          gradientError:
            at(3) === 1
              ? actual.map((value, axis) => difference(value, fine[axis] ?? [NaN, NaN, NaN]))
              : null,
          areaRelativeError: at(3) === 1 ? Math.abs(area / referenceArea.area - 1) : null,
        };
      });
      await server.commands.writeFile(
        `test-results/critical-derivative-${mode}.json`,
        JSON.stringify(records, null, 2),
      );
      for (const record of records) {
        expect.soft(record.transportResolved, JSON.stringify(record.pixel)).toBe(true);
        expect
          .soft(record.referenceArea.convergence, "reference area convergence")
          .toBeLessThan(1e-5);
        for (const error of record.convergence) {
          expect.soft(error, "reference convergence").toBeLessThan(1e-5);
        }
        for (const error of record.gradientError ?? []) {
          expect.soft(error, "Cartesian sky gradient").toBeLessThan(0.002);
        }
        if (record.areaRelativeError !== null) {
          expect.soft(record.areaRelativeError, "solid-angle Jacobian").toBeLessThan(0.002);
        }
      }
    },
  );
});
