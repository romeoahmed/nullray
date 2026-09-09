import { expect, test } from "vitest";
import elliptic from "../../src/render/shaders/elliptic.wgsl?raw";
import equator from "../../src/render/shaders/equator.wgsl?raw";
import arrival from "../../src/render/shaders/arrival.wgsl?raw";
import { photonFromLocal } from "../../src/physics/photon.ts";
import { traceVisibility } from "../reference/visibility.ts";
import { outerHorizon } from "../../src/physics/spacetime.ts";
import { prepareQuartic, quarticArrival } from "../reference/quartic.ts";
import { normalize } from "../../src/physics/vector.ts";
import { computeReadback } from "./compute.ts";

const source = `${elliptic}\n${equator}\n${arrival}
@group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coefficients = inputs[3u * id.x];
  let point = inputs[3u * id.x + 1u];
  let endpoint = inputs[3u * id.x + 2u];
  let path = prepare_quartic(coefficients, point.x, point.y, point.z);
  let hit = quartic_arrival(path, endpoint.x, endpoint.y);
  let time = (hit.lower + hit.upper) / 2.0;
  var residual = 0.0;
  if (hit.status == 1u) {
    let value = evaluate_quartic(path, time);
    if (value.valid) { residual = value.value - endpoint.x; }
    else { residual = 1.0; }
  }
  outputs[id.x] = vec4f(time, f32(hit.status), residual, f32(radial_destination(coefficients.xzw, point.x, path, endpoint.z)));
}`;

test("GPU inverse quartic phase agrees with CPU at sky and horizon endpoints", async () => {
  const records: number[] = [];
  for (const spin of [-0.8, 0, 0.8]) {
    for (const charge of [0, 0.3]) {
      for (const radius of [4, 10, 30]) {
        for (const inclination of [0.1, 1.2, 2.7]) {
          for (const direction of [
            normalize([-1, 0.1, 0.2]),
            normalize([-1, 0.8, -0.3]),
            normalize([1, 0.2, 0.4]),
            normalize([0, 0.6, 0.8]),
          ]) {
            const space = { spin, charge };
            const photon = photonFromLocal(space, radius, inclination, direction);
            const { energy: e, angularMomentum: l, carter: c } = photon;
            const k = (l - spin * e) ** 2 + c;
            const horizon = 1 / outerHorizon(space);
            const reference = traceVisibility(space, photon, radius, inclination).outcome;
            expect(["sky", "captured"]).toContain(reference.kind);
            const target = reference.kind === "sky" ? 0 : horizon;
            const velocity =
              reference.kind === "sky"
                ? -Math.abs(e)
                : Math.abs(e + (spin * spin * e - spin * l) * horizon ** 2);
            records.push(
              e * e,
              0,
              spin * spin * e * e - l * l - c,
              2 * k,
              -spin * spin * c - charge * charge * k,
              1 / radius,
              photon.radialVelocity / radius ** 2,
              0,
              target,
              velocity,
              horizon,
              0,
            );
          }
        }
      }
    }
  }
  // Future-directed zero/negative Killing energy can turn outside the horizon,
  // but cannot reach the distant source boundary.
  for (const e of [0, -0.1]) {
    const spin = 0.9,
      charge = 0.1,
      l = -1,
      c = 0.1,
      u = 1 / 1.5;
    const k = (l - spin * e) ** 2 + c;
    const c2 = spin * spin * e * e - l * l - c;
    const c3 = 2 * k;
    const c4 = -spin * spin * c - charge * charge * k;
    const horizon = 1 / outerHorizon({ spin, charge });
    const speed = Math.sqrt(e * e + u * u * (c2 + u * (c3 + c4 * u)));
    for (const sign of [-1, 1]) {
      records.push(
        e * e,
        0,
        c2,
        c3,
        c4,
        u,
        sign * speed,
        0,
        horizon,
        Math.abs(e + (spin * spin * e - spin * l) * horizon * horizon),
        horizon,
        0,
      );
    }
  }
  const input = new Float32Array(records);
  const count = input.length / 12;
  const result = await computeReadback(source, input, count * 4, count);
  for (let index = 0; index < count; index++) {
    const row = input.subarray(index * 12, index * 12 + 12);
    const at = (offset: number) => row[offset] ?? Number.NaN;
    const path = prepareQuartic([at(0), at(1), at(2), at(3), at(4)], at(5), at(6));
    const expected = quarticArrival(path, at(8), at(9));
    expect(result[index * 4 + 3], `destination row ${index}`).toBe(at(8) === 0 ? 1 : 2);
    expect(expected, `CPU arrival row ${index}`).toBeDefined();
    expect(result[index * 4 + 1], `row ${index}: input ${JSON.stringify(Array.from(row))}`).toBe(1);
    expect(
      Math.abs((result[index * 4] ?? Infinity) - (expected ?? Infinity)),
      `time row ${index}`,
    ).toBeLessThan(2e-5);
    expect(Math.abs(result[index * 4 + 2] ?? Infinity), `residual row ${index}`).toBeLessThan(2e-5);
  }
});
