import { describe, expect, test } from "vitest";
import * as fc from "fast-check";
import { circularOrbit, isco, metric } from "../../src/physics/spacetime.ts";
import { photonFromLocal, polarPotential, radialPotential } from "../../src/physics/photon.ts";
import { normalize } from "../../src/physics/vector.ts";

describe("Kerr–Newman geometry", () => {
  test("recovers Schwarzschild circular motion and its marginal stable orbit", () => {
    const space = { spin: 0, charge: 0 };
    expect(isco(space, 1)).toBeCloseTo(6, 11);
    expect(isco(space, -1)).toBeCloseTo(6, 11);
    const orbit = circularOrbit(space, 10, 1);
    expect(orbit?.omega).toBeCloseTo(10 ** -1.5, 14);
    expect(orbit?.ut).toBeCloseTo(1 / Math.sqrt(0.7), 14);
    expect(orbit?.radialStability).toBeLessThan(0);
  });

  test("neutral orbital geometry is invariant under charge sign reversal", () => {
    fc.assert(
      fc.property(
        fc.double({ min: -0.8, max: 0.8, noNaN: true }),
        fc.double({ min: -0.5, max: 0.5, noNaN: true }),
        (spin, charge) => {
          expect(isco({ spin, charge }, 1)).toBe(isco({ spin, charge: -charge }, 1));
        },
      ),
    );
  });

  test("local initial momenta satisfy both separated potentials", () => {
    fc.assert(
      fc.property(
        fc.tuple(
          fc.double({ min: -1, max: 1, noNaN: true }),
          fc.double({ min: -1, max: 1, noNaN: true }),
          fc.double({ min: -1, max: 1, noNaN: true }),
        ),
        (components) => {
          const [x, y, z] = components;
          if (Math.hypot(x, y, z) < 1e-10) {
            return;
          }
          const space = { spin: 0.7, charge: 0.3 };
          const ray = photonFromLocal(space, 8, 1.2, normalize([x, y, z]));
          expect(radialPotential(space, ray, 8)).toBeCloseTo(ray.radialVelocity ** 2, 8);
          expect(polarPotential(space, ray, 1.2)).toBeCloseTo(ray.polarVelocity ** 2, 10);
        },
      ),
    );
  });

  test("stationary frame normalization and photon energy agree", () => {
    const space = { spin: 0.9, charge: 0.2 };
    const g = metric(space, 2, 1.4);
    expect(
      (g.tt + 2 * g.dragging * g.tphi + g.dragging ** 2 * g.phiPhi) / g.lapse ** 2,
    ).toBeCloseTo(-1, 12);
    const photon = photonFromLocal(space, 2, 1.4, [0, 0, 1]);
    expect((photon.energy - g.dragging * photon.angularMomentum) / g.lapse).toBeCloseTo(1, 12);
  });
});
