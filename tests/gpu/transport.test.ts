import { expect, test } from "vitest";
import elliptic from "../../src/render/shaders/elliptic.wgsl?raw";
import equator from "../../src/render/shaders/equator.wgsl?raw";
import transport from "../../src/render/shaders/transport.wgsl?raw";
import { photonFromLocal } from "../../src/physics/photon.ts";
import { evaluateGeodesic, prepareGeodesic } from "../reference/geodesic.ts";
import { integrateTransport } from "../reference/transport.ts";
import { normalize } from "../../src/physics/vector.ts";
import { computeReadback } from "./compute.ts";

test("GPU transport matches CPU quadrature for exterior and axis-observer paths", async () => {
  const rows: number[][] = [];
  const critical = photonFromLocal(
    { spin: 0, charge: 0 },
    30,
    0,
    [-0.9858951568603516, 0.1673642098903656, 0],
  );
  for (const end of [1, 1.5, 2.18]) {
    rows.push([
      0,
      0,
      critical.energy,
      critical.angularMomentum,
      critical.carter,
      30,
      0,
      end,
      critical.radialVelocity,
      critical.polarVelocity,
      0,
      0,
    ]);
  }
  const criticalCount = rows.length;
  for (const spin of [0, -0.8, 0.8]) {
    for (const direction of [
      [-1, 0.3, 0.5],
      [0, -0.3, -0.8],
    ] as const) {
      const photon = photonFromLocal({ spin, charge: 0.4 }, 8, 1.1, normalize(direction));
      rows.push([
        spin,
        0.4,
        photon.energy,
        photon.angularMomentum,
        photon.carter,
        8,
        1.1,
        0.04,
        photon.radialVelocity,
        photon.polarVelocity,
        0,
        0,
      ]);
    }
  }
  for (const spin of [0, 0.8]) {
    for (const azimuth of [-1e-5, 0, 1e-5]) {
      const photon = photonFromLocal({ spin, charge: 0.4 }, 8, 0.5, normalize([-0.2, -1, azimuth]));
      rows.push([
        spin,
        0.4,
        photon.energy,
        photon.angularMomentum,
        photon.carter,
        8,
        0.5,
        0.1,
        photon.radialVelocity,
        photon.polarVelocity,
        0,
        0,
      ]);
    }
  }
  for (const spin of [0, 0.8]) {
    for (const theta of [0, 1e-4, Math.PI - 1e-4, Math.PI]) {
      const photon = photonFromLocal({ spin, charge: 0.4 }, 8, theta, normalize([0.2, 0.7, 0.3]));
      rows.push([
        spin,
        0.4,
        photon.energy,
        photon.angularMomentum,
        photon.carter,
        8,
        theta,
        0.04,
        photon.radialVelocity,
        photon.polarVelocity,
        0,
        0,
      ]);
    }
  }
  const inputs = new Float32Array(rows.flat());
  const result = await computeReadback(
    `${elliptic}\n${equator}\n${transport}
    @group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
    @group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
    @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
      let s = inputs[id.x * 3u]; let p = inputs[id.x * 3u + 1u]; let v = inputs[id.x * 3u + 2u];
      let a = s.x; let q = s.y; let e = s.z; let l = s.w; let c = p.x;
      let k = (l - a * e) * (l - a * e) + c;
      let b = a * a * e * e - l * l - c;
      let radial = prepare_quartic(vec4f(e * e, 0.0, b, 2.0 * k), -a * a * c - q * q * k, 1.0 / p.y, v.x / (p.y * p.y));
      let sine = select(sin(p.z), 0.0, p.z == 0.0 || p.z == 3.1415926536);
      let polar = prepare_quartic(vec4f(c, 0.0, b, 0.0), -a * a * e * e, cos(p.z), sine * v.y);
      let events = equator_events(a, e, l, c, p.z, v.y);
      let value = integrate_transport(radial, polar, a, q, e, l, c, p.w, true, events);
      outputs[id.x] = vec4f(value.value, value.error.x, select(0.0, 1.0, value.valid));
    }`,
    inputs,
    rows.length * 4,
    rows.length,
  );
  for (let i = 0; i < rows.length; i++) {
    const [
      spin,
      charge,
      energy,
      angularMomentum,
      carter,
      r,
      theta,
      time,
      radialVelocity,
      polarVelocity,
    ] = inputs.slice(i * 12, i * 12 + 10);
    if (
      spin === undefined ||
      charge === undefined ||
      energy === undefined ||
      angularMomentum === undefined ||
      carter === undefined ||
      r === undefined ||
      theta === undefined ||
      time === undefined ||
      radialVelocity === undefined ||
      polarVelocity === undefined
    ) {
      throw new Error("Missing fixture input.");
    }
    const space = { spin, charge };
    const photon = { energy, angularMomentum, carter, radialVelocity, polarVelocity };
    const expected = integrateTransport(
      space,
      photon,
      prepareGeodesic(space, photon, r, theta === Math.fround(Math.PI) ? Math.PI : theta),
      time,
      true,
      1e-11,
      65_536,
    );
    if (expected.kind !== "resolved") {
      throw new Error(`CPU reference failed at row ${i}, time ${time}: ${expected.reason}.`);
    }
    expect(result[i * 4 + 3]).toBe(1);
    expect(result[i * 4]).toBeCloseTo(expected.azimuth, 4);
    if (i < criticalCount) {
      const endpoint = evaluateGeodesic(prepareGeodesic(space, photon, r, theta), time);
      if (!endpoint || endpoint.inverseRadius <= 0) {
        throw new Error("Missing exterior reference endpoint.");
      }
      const timeError = Math.abs((result[i * 4 + 1] ?? Number.NaN) - expected.coordinateTime);
      // Critical-ray f32 conditioning is assessed in the highest emitted Fourier phase.
      // This does not claim the ordinary-ray absolute travel-time accuracy here.
      const omega = endpoint.inverseRadius ** 1.5;
      expect(23 * omega * timeError).toBeLessThan(0.005);
    } else {
      expect(result[i * 4 + 1]).toBeCloseTo(expected.coordinateTime, 3);
    }
  }
});
