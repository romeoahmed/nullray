import { expect, test } from "vitest";
import { photonFromLocal } from "../../src/physics/photon.ts";
import { normalize } from "../../src/physics/vector.ts";
import { traceVisibility } from "../reference/visibility.ts";
import { evaluateGeodesic } from "../reference/geodesic.ts";
import { integrateTransport } from "../reference/transport.ts";

test("axis launch constants are finite and invariant under tangent-plane rotations", () => {
  for (const inclination of [0, Math.PI]) {
    for (const spin of [0, 0.8]) {
      const space = { spin, charge: 0.2 };
      const base = photonFromLocal(space, 10, inclination, [0.6, 0.8, 0]);
      for (const angle of [0, 0.7, 1.8, 4]) {
        const ray = photonFromLocal(space, 10, inclination, [
          0.6,
          0.8 * Math.cos(angle),
          0.8 * Math.sin(angle),
        ]);
        expect(Object.values(ray).every(Number.isFinite)).toBe(true);
        expect(ray.angularMomentum).toBe(0);
        expect(ray.energy).toBeCloseTo(base.energy, 13);
        expect(ray.carter).toBeCloseTo(base.carter, 11);
        expect(ray.polarVelocity).toBeCloseTo(base.polarVelocity, 12);
      }
    }
  }
});

test("escaped sky directions approach the axis-observer limit from regular coordinates", () => {
  for (const spin of [0, 0.8]) {
    const space = { spin, charge: 0.2 };
    const source = normalize([0.2, 0.7, 0.3]);
    for (const inclination of [0, Math.PI]) {
      const directions = [inclination, inclination === 0 ? 1e-5 : Math.PI - 1e-5].map((theta) => {
        const ray = photonFromLocal(space, 10, theta, source);
        const { path, outcome } = traceVisibility(space, ray, 10, theta);
        expect(outcome.kind).toBe("sky");
        if (outcome.kind !== "sky") {
          throw new Error("Expected an outgoing escape.");
        }
        const mapped = integrateTransport(space, ray, path, outcome.time, false);
        expect(mapped.kind).toBe("resolved");
        if (mapped.kind !== "resolved") {
          throw new Error("Expected resolved angular transport.");
        }
        const mu = evaluateGeodesic(path, outcome.time)?.cosineTheta;
        if (mu === undefined) {
          throw new Error("Missing polar endpoint.");
        }
        const longitude =
          mapped.azimuth +
          (theta === inclination ? Math.atan2(source[2], (theta === 0 ? 1 : -1) * source[1]) : 0);
        return [
          Math.sqrt(1 - mu * mu) * Math.cos(longitude),
          Math.sqrt(1 - mu * mu) * Math.sin(longitude),
          mu,
        ];
      });
      const axis = directions[0];
      const near = directions[1];
      if (!axis || !near) {
        throw new Error("Missing comparison directions.");
      }
      for (let i = 0; i < 3; i++) {
        expect(Math.abs((axis[i] ?? NaN) - (near[i] ?? NaN))).toBeLessThan(2e-5);
      }
    }
  }
});
