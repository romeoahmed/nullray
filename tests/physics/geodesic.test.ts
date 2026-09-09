import { describe, expect, test } from "vitest";
import * as fc from "fast-check";
import { prepareGeodesic, evaluateGeodesic } from "../reference/geodesic.ts";
import { photonFromLocal, radialPotential, polarPotential } from "../../src/physics/photon.ts";
import { normalize } from "../../src/physics/vector.ts";
import { referenceGeodesic } from "../reference/hamiltonian.ts";
import { integrateTransport } from "../reference/transport.ts";
import criticalFixture from "../fixtures/critical-primary-rays.json" with { type: "json" };

describe("separated backward geodesics", () => {
  test("agrees with the Hamiltonian oracle over ordinary exterior rays", () => {
    // A uniform shape distribution complements hand-picked degenerate fixtures.
    const unit = fc.integer({ min: 0, max: 1000000 }).map((x) => x / 1000000);
    fc.assert(
      fc.property(
        fc.tuple(unit, unit, unit, unit, unit, unit),
        ([shape, angle, distance, polar, azimuth, inclination]) => {
          const space = {
            spin: 0.98 * shape * Math.cos(2 * Math.PI * angle),
            charge: 0.98 * shape * Math.sin(2 * Math.PI * angle),
          };
          const r = 5 + 25 * distance;
          const theta = 0.4 + polar * (Math.PI - 0.8);
          const radial = 2 * inclination - 1;
          const tangent = Math.sqrt(1 - radial * radial);
          const source = [
            radial,
            tangent * Math.cos(2 * Math.PI * azimuth),
            tangent * Math.sin(2 * Math.PI * azimuth),
          ] as const;
          const photon = photonFromLocal(space, r, theta, source);
          const time = 0.12 / r;
          const result = evaluateGeodesic(prepareGeodesic(space, photon, r, theta), time);
          const reference = referenceGeodesic(space, photon, r, theta, time, 1e-12);
          expect(reference.maxResidual).toBeLessThan(1e-9);
          expect(result?.inverseRadius).toBeCloseTo(1 / reference.state[0], 9);
          expect(result?.cosineTheta).toBeCloseTo(Math.cos(reference.state[1]), 9);
        },
      ),
      { numRuns: 150 },
    );
  });

  test("matches an independent metric Hamiltonian integration, including initial turns and negative energy", () => {
    const cases = [
      {
        space: { spin: 0, charge: 0 },
        r: 8,
        theta: 1.1,
        direction: [-1, 0.3, 0.5] as const,
        time: 0.12,
      },
      {
        space: { spin: 0.8, charge: 0.4 },
        r: 8,
        theta: 1.1,
        direction: [0, 0.6, 0.8] as const,
        time: 0.05,
      },
      {
        space: { spin: -0.8, charge: 0.4 },
        r: 6,
        theta: 1.4,
        direction: [-1, 0, 0.6] as const,
        time: 0.12,
      },
      {
        space: { spin: 0, charge: 0.95 },
        r: 5,
        theta: 0.5,
        direction: [-1, 0.4, -0.4] as const,
        time: 0.08,
      },
      {
        space: { spin: 0.95, charge: 0 },
        r: 1.7,
        theta: Math.PI / 2,
        direction: [0.1, 0.1, 1] as const,
        time: 0.02,
      },
    ];
    for (const { space, r, theta, direction, time } of cases) {
      const photon = photonFromLocal(space, r, theta, normalize(direction));
      if (r === 1.7) {
        expect(photon.energy).toBeLessThan(0);
      }
      const path = prepareGeodesic(space, photon, r, theta);
      const actual = evaluateGeodesic(path, time);
      const reference = referenceGeodesic(space, photon, r, theta, time);
      const tighter = referenceGeodesic(space, photon, r, theta, time, 1e-13);
      expect(actual).toBeDefined();
      expect(reference.maxResidual).toBeLessThan(1e-9);
      expect(tighter.maxResidual).toBeLessThan(1e-10);
      expect(1 / reference.state[0]).toBeCloseTo(1 / tighter.state[0], 9);
      expect(reference.state[1]).toBeCloseTo(tighter.state[1], 9);
      expect(actual?.inverseRadius).toBeCloseTo(1 / tighter.state[0], 9);
      expect(actual?.cosineTheta).toBeCloseTo(Math.cos(tighter.state[1]), 9);
    }
  });

  test.for(criticalFixture.cases)(
    "checks the critical primary-ray reference at $width × $height, ($x, $y)",
    ({ hamiltonian: { space, r, theta, time, source } }) => {
      const [radial = NaN, polar = NaN, azimuth = NaN] = source;
      const photon = photonFromLocal(space, r, theta, [radial, polar, azimuth]);
      const path = prepareGeodesic(space, photon, r, theta);
      const actual = evaluateGeodesic(path, time);
      const transport = integrateTransport(space, photon, path, time, false, 1e-10, 65536);
      const reference = referenceGeodesic(space, photon, r, theta, time, 1e-12);
      const tighter = referenceGeodesic(space, photon, r, theta, time, 1e-13);
      if (!actual || transport.kind !== "resolved") {
        throw new Error("Critical binary64 reference did not resolve.");
      }
      for (const sample of [reference, tighter]) {
        expect(sample.maxResidual).toBeLessThan(1e-9);
      }
      expect(Math.abs(1 / reference.state[0] - 1 / tighter.state[0])).toBeLessThan(1e-6);
      expect(Math.abs(Math.cos(reference.state[1]) - Math.cos(tighter.state[1]))).toBeLessThan(
        1e-8,
      );
      expect(Math.abs(reference.state[2] - tighter.state[2])).toBeLessThan(1e-6);
      expect(Math.abs(actual.inverseRadius - 1 / tighter.state[0])).toBeLessThan(1e-6);
      expect(Math.abs(actual.cosineTheta - Math.cos(tighter.state[1]))).toBeLessThan(1e-8);
      expect(Math.abs(transport.azimuth - tighter.state[2])).toBeLessThan(1e-6);
    },
  );

  test("Schwarzschild radial rays have linear inverse radius in Mino time", () => {
    const space = { spin: 0, charge: 0 };
    for (const direction of [-1, 1] as const) {
      const photon = photonFromLocal(space, 10, 1.2, [direction, 0, 0]);
      const path = prepareGeodesic(space, photon, 10, 1.2);
      for (const t of [0, 0.01, 0.05]) {
        const result = evaluateGeodesic(path, t);
        expect(result?.inverseRadius).toBeCloseTo(0.1 - direction * photon.energy * t, 13);
        expect(result?.cosineTheta).toBeCloseTo(Math.cos(1.2), 13);
      }
    }
  });

  test("analytic derivatives recover charged radial and polar potentials", () => {
    const space = { spin: 0.8, charge: 0.4 };
    for (const direction of [
      [-1, 0.2, 0.15],
      [1, -0.4, 0.7],
    ] as const) {
      const photon = photonFromLocal(space, 8, 1.1, normalize(direction));
      const path = prepareGeodesic(space, photon, 8, 1.1);
      for (const t of [0.001, 0.01, 0.04]) {
        const h = 1e-6;
        const center = evaluateGeodesic(path, t);
        const before = evaluateGeodesic(path, t - h);
        const after = evaluateGeodesic(path, t + h);
        expect(center).toBeDefined();
        if (!center || !before || !after) {
          throw new Error("An ordinary exterior path must evaluate.");
        }
        const r = 1 / center.inverseRadius;
        const theta = Math.acos(center.cosineTheta);
        const radialDerivative = (1 / after.inverseRadius - 1 / before.inverseRadius) / (2 * h);
        const polarDerivative =
          (Math.acos(after.cosineTheta) - Math.acos(before.cosineTheta)) / (2 * h);
        expect(radialDerivative ** 2 / radialPotential(space, photon, r)).toBeCloseTo(1, 7);
        expect(polarDerivative ** 2 / polarPotential(space, photon, theta)).toBeCloseTo(1, 7);
      }
    }
  });
});
