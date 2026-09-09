import { describe, expect, test } from "vitest";
import { equatorCrossings } from "../reference/equator.ts";
import { photonFromLocal } from "../../src/physics/photon.ts";
import { evaluateGeodesic, prepareGeodesic } from "../reference/geodesic.ts";
import { normalize } from "../../src/physics/vector.ts";
import { referenceGeodesic } from "../reference/hamiltonian.ts";

describe("polar event phases", () => {
  test("enumerates successive transverse zeros in either polar direction", () => {
    for (const spin of [0, -0.8, 0.8]) {
      const space = { spin, charge: 0.3 };
      for (const theta of [0.5, 1.4, 1.8, 2.5]) {
        for (const polar of [-1, 0, 1]) {
          const photon = photonFromLocal(space, 8, theta, normalize([-0.2, polar, 0.7]));
          const events = equatorCrossings(space, photon, theta);
          expect(events.kind).toBe("crossings");
          if (events.kind !== "crossings") {
            throw new Error("This fixture must cross the equator.");
          }
          const path = prepareGeodesic(space, photon, 8, theta);
          expect(events.first).toBeGreaterThan(0);
          expect(events.first).toBeLessThanOrEqual(events.spacing);
          for (let i = 0; i < 8; i++) {
            const time = events.first + i * events.spacing;
            expect(evaluateGeodesic(path, time)?.cosineTheta).toBeCloseTo(0, 10);
          }
        }
      }
    }
  });

  test("the first crossing agrees with an independent Hamiltonian path", () => {
    const space = { spin: 0.8, charge: 0.4 };
    const theta = 1.4;
    const photon = photonFromLocal(space, 8, theta, normalize([-1, 0.8, 0.3]));
    const events = equatorCrossings(space, photon, theta);
    if (events.kind !== "crossings") {
      throw new Error("Expected transverse crossing.");
    }
    const reference = referenceGeodesic(space, photon, 8, theta, events.first, 1e-13);
    expect(reference.state[1]).toBeCloseTo(Math.PI / 2, 10);
  });
});
