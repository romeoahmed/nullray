import { describe, expect, test } from "vitest";
import * as fc from "fast-check";
import { circularOrbit, isco, metric, outerHorizon } from "../../src/physics/spacetime.ts";
import { photonFromLocal, polarPotential, radialPotential } from "../../src/physics/photon.ts";
import { normalize } from "../../src/physics/vector.ts";
import { traceVisibility } from "../reference/visibility.ts";
import { evaluateGeodesic } from "../reference/geodesic.ts";
import { integrateTransport } from "../reference/transport.ts";
import { criticalDirection } from "../../src/physics/critical.ts";
import fixture from "../fixtures/schwarzschild-stellar-images.json" with { type: "json" };
import { referenceRay, referenceSky } from "../reference/sky.ts";
import { stellarImageFrame } from "../support/frames.ts";

describe("Local null geometry", () => {
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
});

describe("Polar observers", () => {
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
});

describe("Spherical photon orbits", () => {
  test.for([0, 0.2, 0.9, -0.999])("nonrotating critical cone retains charge %s", (charge) => {
    const space = { spin: 0, charge };
    const radius = (3 + Math.sqrt(9 - 8 * charge * charge)) / 2;
    const impact = radius ** 2 / Math.sqrt(radius * radius - 2 * radius + charge * charge);
    for (const observer of [outerHorizon(space) + 0.1, radius, 30, 2000]) {
      for (const inclination of [0, 1.2, Math.PI]) {
        for (const phase of [0, 0.7, Math.PI, 5]) {
          const point = criticalDirection(space, observer, inclination, phase);
          expect(point.kind).toBe("point");
          if (point.kind !== "point") {
            throw new Error("Unresolved critical cone");
          }
          expect(point.sphericalRadius).toBeCloseTo(radius, 13);
          const photon = photonFromLocal(space, observer, inclination, point.direction);
          expect(
            Math.sqrt(photon.carter + photon.angularMomentum ** 2) / photon.energy,
          ).toBeCloseTo(impact, 11);
          if (observer !== radius) {
            expect(Math.sign(point.direction[0])).toBe(Math.sign(radius - observer));
          } else {
            expect(Math.abs(point.direction[0])).toBeLessThan(1e-14);
          }
        }
      }
    }
  });

  test.for([-0.99, -0.3, 0.3, 0.99])("equatorial Kerr critical endpoints at spin %s", (spin) => {
    for (const phase of [0, Math.PI]) {
      const point = criticalDirection({ spin, charge: 0 }, 30, Math.PI / 2, phase);
      expect(point.kind).toBe("point");
      if (point.kind !== "point") {
        throw new Error("Unresolved equatorial critical direction");
      }
      const radius = 2 * (1 + Math.cos((2 / 3) * Math.acos(-spin * Math.cos(phase))));
      expect(point.sphericalRadius).toBeCloseTo(radius, 13);
    }
  });

  test("charged rotating critical directions satisfy the double-root and local null invariants", () => {
    fc.assert(
      fc.property(
        fc.double({ min: -0.99, max: 0.99, noNaN: true }),
        fc.double({ min: -0.99, max: 0.99, noNaN: true }),
        fc.double({ min: 0, max: Math.PI, noNaN: true }),
        fc.double({ min: 0, max: 2 * Math.PI, noNaN: true }),
        fc.double({ min: 0.1, max: 1000, noNaN: true }),
        (spin, chargeFraction, inclination, phase, height) => {
          const space = { spin, charge: chargeFraction * Math.sqrt(1 - spin * spin) };
          const observer = outerHorizon(space) + height;
          const point = criticalDirection(space, observer, inclination, phase);
          expect(point.kind).toBe("point");
          if (point.kind !== "point") {
            throw new Error("Unresolved property sample");
          }
          const photon = photonFromLocal(space, observer, inclination, point.direction);
          const r = point.sphericalRadius;
          const { energy: e, angularMomentum: l, carter: c } = photon;
          const k = (l - spin * e) ** 2 + c;
          const p = e * (r * r + spin * spin) - spin * l;
          const scale =
            Math.abs(p * p) + Math.abs(k * (r * r - 2 * r + spin * spin + space.charge ** 2));
          expect(Math.abs(radialPotential(space, photon, r)) / scale).toBeLessThan(2e-12);
          expect(Math.abs(4 * e * r * p - 2 * (r - 1) * k) / scale).toBeLessThan(2e-12);
          expect(4 * e * p + 8 * e * e * r * r - 2 * k).toBeGreaterThan(0);
          expect(photon.energy).toBeGreaterThan(0);
          expect(Math.hypot(...point.direction)).toBeCloseTo(1, 14);
        },
      ),
      { seed: 314159, numRuns: 500 },
    );
  });

  test("near-extremal critical directions retain the small radius-minus-one displacement", () => {
    const space = { spin: Math.fround(0.999), charge: Math.fround(0.04) };
    const inclination = Math.fround(Math.PI / 2);
    for (const observer of [3, 30, 200]) {
      const point = criticalDirection(space, observer, inclination, Math.fround(0.2));
      if (point.kind !== "point") {
        throw new Error("Unresolved near-extremal critical direction");
      }
      const photon = photonFromLocal(space, observer, inclination, point.direction);
      const { energy: e, angularMomentum: l, carter: c } = photon;
      const { spin: a, charge: q } = space;
      const r = point.sphericalRadius;
      const p = e * (r * r + a * a) - a * l;
      const k = (l - a * e) ** 2 + c;
      const scale = p * p + Math.abs((r * r - 2 * r + a * a + q * q) * k);
      expect(Math.abs(radialPotential(space, photon, r)) / scale).toBeLessThan(1e-12);
      const first = 4 * e * r * p;
      const second = 2 * (r - 1) * k;
      expect(Math.abs(first - second) / (Math.abs(first) + Math.abs(second))).toBeLessThan(1e-12);
    }
  });

  test("critical seed input errors are distinct from unresolved arithmetic", () => {
    expect(criticalDirection({ spin: 1, charge: 0 }, 30, 1, 0).kind).toBe("invalid");
    expect(criticalDirection({ spin: 0, charge: 0 }, 2, 1, 0).kind).toBe("invalid");
    expect(criticalDirection({ spin: 0, charge: 0 }, 30, -1, 0).kind).toBe("invalid");
    expect(criticalDirection({ spin: 0, charge: 0 }, 30, 1, NaN).kind).toBe("invalid");
    expect(criticalDirection({ spin: 0, charge: 0 }, 1e300, 1, 0).kind).toBe("unresolved");
  });
});

describe("Relativistic image orders", () => {
  test.for(fixture.cases)(
    "Schwarzschild image order $order side $side agrees with the independent planar orbit",
    (sample) => {
      const frame = stellarImageFrame();
      const x = sample.pixel[0] ?? NaN,
        y = sample.pixel[1] ?? NaN;
      const { outcome } = referenceRay(x, y, frame);
      expect(outcome.kind).toBe(sample.occulted ? "disk" : "sky");
      if (outcome.kind === "disk") {
        expect(Math.abs(outcome.radius - (sample.crossingRadii[0] ?? NaN))).toBeLessThan(1e-9);
        return;
      }
      const sky = referenceSky(x, y, frame);
      const source = normalize([
        fixture.source[0] ?? NaN,
        fixture.source[1] ?? NaN,
        fixture.source[2] ?? NaN,
      ]);
      expect(
        Math.hypot(...sky.direction.map((value, axis) => value - (source[axis] ?? NaN))),
      ).toBeLessThan(1e-8);
      expect(Math.abs(sky.energy - fixture.energy)).toBeLessThan(1e-14);
      // Rotational symmetry gives the tangential derivative exactly; refine only
      // the radial difference, independently of the implicit oracle derivative.
      const areas = [5e-7, 2.5e-7].map((step) => {
        const before = referenceSky(x - step, y, frame).direction;
        const after = referenceSky(x + step, y, frame).direction;
        const slope = Math.hypot(
          ...after.map((value, axis) => (value - (before[axis] ?? NaN)) / (2 * step)),
        );
        return (
          (slope * Math.hypot(sky.direction[0], sky.direction[1])) /
          Math.abs(x - (fixture.frame.width - 1) / 2)
        );
      });
      expect(Math.abs((areas[0] ?? NaN) / (areas[1] ?? NaN) - 1)).toBeLessThan(4e-6);
      expect(Math.abs((areas[1] ?? NaN) / sample.jacobian - 1)).toBeLessThan(2e-6);
    },
  );
});
