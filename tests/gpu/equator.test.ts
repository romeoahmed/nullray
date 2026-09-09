import { expect, test } from "vitest";
import elliptic from "../../src/render/shaders/elliptic.wgsl?raw";
import equator from "../../src/render/shaders/equator.wgsl?raw";
import { computeReadback } from "./compute.ts";
import { photonFromLocal } from "../../src/physics/photon.ts";
import { equatorCrossings } from "../reference/equator.ts";
import { normalize } from "../../src/physics/vector.ts";

test("GPU equator phases agree with CPU for identical quantized photon constants", async () => {
  const cases: number[][] = [];
  for (const spin of [0, -0.8, 0.8]) {
    for (const theta of [0.5, 1.4, 1.8, 2.5]) {
      for (const polar of [-1, -0.1, 0.1, 1]) {
        const photon = photonFromLocal(
          { spin, charge: 0.3 },
          8,
          theta,
          normalize([-0.2, polar, 0.7]),
        );
        cases.push([
          spin,
          photon.energy,
          photon.angularMomentum,
          photon.carter,
          theta,
          photon.polarVelocity,
          0,
          0,
        ]);
      }
    }
  }
  const data = new Float32Array(cases.flat());
  const result = await computeReadback(
    `${elliptic}\n${equator}
      @group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
      @group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
      @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
        let p = inputs[id.x * 2u]; let q = inputs[id.x * 2u + 1u];
        let result = equator_events(p.x, p.y, p.z, p.w, q.x, q.y);
        outputs[id.x] = vec4f(result.first, result.spacing, f32(result.status), 0.0);
      }`,
    data,
    cases.length * 4,
    cases.length,
  );
  for (let i = 0; i < cases.length; i++) {
    const [spin, energy, angularMomentum, carter, theta, polarVelocity] = data.slice(
      i * 8,
      i * 8 + 6,
    );
    if (
      spin === undefined ||
      energy === undefined ||
      angularMomentum === undefined ||
      carter === undefined ||
      theta === undefined ||
      polarVelocity === undefined
    ) {
      throw new Error("Missing quantized fixture input.");
    }
    const expected = equatorCrossings(
      { spin, charge: 0.3 },
      { energy, angularMomentum, carter, polarVelocity, radialVelocity: 0 },
      theta,
    );
    expect(expected.kind).toBe("crossings");
    if (expected.kind !== "crossings") {
      throw new Error("Expected ordinary transverse events.");
    }
    expect(result[i * 4 + 2]).toBe(1);
    expect(result[i * 4]).toBeCloseTo(expected.first, 5);
    expect(result[i * 4 + 1]).toBeCloseTo(expected.spacing, 5);
  }
});
