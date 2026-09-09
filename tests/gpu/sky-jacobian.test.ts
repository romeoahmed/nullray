import derivative from "../../src/render/shaders/derivative.wgsl?raw";
import { expect, test } from "vitest";
import { server } from "vitest/browser";
import differential from "../../src/render/shaders/differential.wgsl?raw";
import imageSource from "../../src/render/shaders/stellar-image.wgsl?raw";
import { opticalSources } from "../../src/render/optics.ts";
import { cross, dot } from "../../src/physics/vector.ts";
import type { Vec3 } from "../../src/physics/vector.ts";
import { computeReadback } from "./compute.ts";
import { createReferenceFrame, referenceRay, referenceSky } from "../reference/sky.ts";

const referenceFrame = createReferenceFrame();

/** Central binary64 difference in one physical pixel coordinate at fixed quantized camera. */
function referenceGradient(
  x: number,
  y: number,
  axis: 0 | 1,
  step: number,
  frame: Float32Array,
): Vec3 {
  const before = referenceSky(
    x - (axis === 0 ? step : 0),
    y - (axis === 1 ? step : 0),
    frame,
  ).direction;
  const after = referenceSky(
    x + (axis === 0 ? step : 0),
    y + (axis === 1 ? step : 0),
    frame,
  ).direction;
  return [
    (after[0] - before[0]) / (2 * step),
    (after[1] - before[1]) / (2 * step),
    (after[2] - before[2]) / (2 * step),
  ];
}

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
  "$name sky forward derivatives agree with refined binary64 screen differences",
  async ({ name, frameData, positions }) => {
    const input = new Float32Array(positions.flatMap(([x, y]) => [x, y, 0, 0]));
    const frame = Array.from(
      { length: 6 },
      (_, i) => `vec4f(${Array.from(frameData.subarray(i * 4, i * 4 + 4)).join(",")})`,
    ).join(",");
    const result = await computeReadback(
      `${opticalSources.ray}\n${derivative}\n${differential}\n${imageSource}
@group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  let frame = Frame(${frame});
  let endpoint = screen_ray(frame, inputs[id.x].xy).value;
  let sample = differential_sky(frame, inputs[id.x].xy, endpoint.w);
  outputs[id.x * 3u] = vec4f(sample.mu, f32(sample.valid));
  outputs[id.x * 3u + 1u] = vec4f(sample.phi, sample.energy);
  outputs[id.x * 3u + 2u] = endpoint;
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
        referenceGradient(x, y, 0, 1 / 8192, frameData),
        referenceGradient(x, y, 1, 1 / 8192, frameData),
      ] as const;
      const coarse = [
        referenceGradient(x, y, 0, 1 / 4096, frameData),
        referenceGradient(x, y, 1, 1 / 4096, frameData),
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
        expect(relativeVectorError(record.gradients[axis], record.fine[axis]), label).toBeLessThan(
          0.002,
        );
      }
      expect(Math.abs(record.area / record.referenceArea - 1), label).toBeLessThan(0.002);
    }
  },
);
