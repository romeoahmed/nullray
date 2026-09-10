import { describe, expect, test } from "vitest";
import * as fc from "fast-check";
import { prepareGeodesic, evaluateGeodesic } from "../reference/geodesic.ts";
import { photonFromLocal, radialPotential, polarPotential } from "../../src/physics/photon.ts";
import { normalize } from "../../src/physics/vector.ts";
import { referenceGeodesic } from "../reference/hamiltonian.ts";
import { integrateTransport } from "../reference/transport.ts";
import primaryRays from "../fixtures/critical-primary-rays.json" with { type: "json" };
import criticalEventOrderCriticalFixture from "../fixtures/critical-event-order.json" with { type: "json" };
import diskFixture from "../fixtures/outer-disk-arrival.json" with { type: "json" };
import radialFixture from "../fixtures/nearly-radial-arrival.json" with { type: "json" };
import { quarticArrival, prepareQuartic, evaluateQuartic } from "../reference/quartic.ts";
import { outerHorizon } from "../../src/physics/spacetime.ts";
import { traceVisibility } from "../reference/visibility.ts";
import { equatorCrossings } from "../reference/equator.ts";

describe("Hamiltonian agreement", () => {
  const criticalFixture = primaryRays;
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
});

describe("Boundary arrivals", () => {
  const criticalFixture = criticalEventOrderCriticalFixture;
  test("quartic arrival retains the outgoing and returning phases of harmonic motion", () => {
    const path = prepareQuartic([1, 0, -1, 0, 0], 0, 1);
    for (const time of [0.1, 1, 2, 3, 3.5, 4, 5, 6]) {
      expect(quarticArrival(path, Math.sin(time), Math.cos(time))).toBeCloseTo(time, 12);
    }
    const linear = prepareQuartic([1, 0, 0, 0, 0], 0, 1);
    expect(quarticArrival(linear, 4, 1)).toBe(4);
    expect(quarticArrival(linear, -4, 1)).toBeUndefined();
  });

  test("direct radial arrivals recover scanned Kerr–Newman boundary times", () => {
    const component = fc.integer({ min: -1000000, max: 1000000 });
    fc.assert(
      fc.property(
        fc.double({ min: -0.9, max: 0.9, noNaN: true }),
        fc.double({ min: 0, max: 0.3, noNaN: true }),
        fc.double({ min: 4, max: 30, noNaN: true }),
        fc.double({ min: 0.1, max: Math.PI - 0.1, noNaN: true }),
        fc.tuple(component, component, component),
        (spin, charge, radius, theta, components) => {
          fc.pre(components.some((value) => value !== 0));
          const direction = normalize(components);
          const space = { spin, charge };
          const photon = photonFromLocal(space, radius, theta, direction);
          const path = prepareGeodesic(space, photon, radius, theta).radial;
          const horizon = 1 / outerHorizon(space);
          const skyTime = quarticArrival(path, 0, -Math.abs(photon.energy));
          const horizonVelocity = Math.abs(
            photon.energy +
              (spin * spin * photon.energy - spin * photon.angularMomentum) * horizon * horizon,
          );
          const horizonTime = quarticArrival(path, horizon, horizonVelocity);
          const scanned = traceVisibility(space, photon, radius, theta).outcome;
          expect(scanned.kind).not.toBe("unresolved");
          const escape = (skyTime ?? Infinity) < (horizonTime ?? Infinity);
          expect(scanned.kind).toBe(escape ? "sky" : "captured");
          const direct = escape ? skyTime : horizonTime;
          if (scanned.kind === "captured" || scanned.kind === "sky") {
            expect(Math.abs((direct ?? Infinity) - scanned.time)).toBeLessThan(1e-7);
            expect(
              Math.abs(
                (evaluateQuartic(path, direct ?? Infinity) ?? Infinity) - (escape ? 0 : horizon),
              ),
            ).toBeLessThan(1e-7);
          }
        },
      ),
      { numRuns: 200 },
    );
  });

  test("nearly radial escape retains a finite arrival with subnormal invariants", () => {
    const { space, radius, inclination, components } = radialFixture;
    const source = normalize([components[0] ?? 0, components[1] ?? 0, components[2] ?? 0]);
    const photon = photonFromLocal(space, radius, inclination, source);
    const path = prepareGeodesic(space, photon, radius, inclination).radial;
    const arrival = quarticArrival(path, 0, -photon.energy);
    const scanned = traceVisibility(space, photon, radius, inclination).outcome;
    expect(scanned.kind).toBe("sky");
    expect(arrival).toBeDefined();
    if (scanned.kind === "sky") {
      expect(Math.abs((arrival ?? Infinity) - scanned.time)).toBeLessThan(1e-10);
    }
    expect(Math.abs(evaluateQuartic(path, arrival ?? Infinity) ?? Infinity)).toBeLessThan(1e-10);
  });

  test("outer-disk inverse radius agrees with the independent Hamiltonian", () => {
    const { space, radius, inclination, direction, eventTime } = diskFixture;
    const photon = photonFromLocal(space, radius, inclination, [
      direction[0] ?? 0,
      direction[1] ?? 0,
      direction[2] ?? 0,
    ]);
    const coarse = referenceGeodesic(space, photon, radius, inclination, eventTime, 1e-11);
    const fine = referenceGeodesic(space, photon, radius, inclination, eventTime, 1e-13);
    expect(Math.abs(1 / coarse.state[0] - 1 / fine.state[0])).toBeLessThan(1e-9);
    expect(Math.abs(fine.state[1] - Math.PI / 2)).toBeLessThan(1e-8);
    for (const measured of [diskFixture.directRadius, diskFixture.scannedRadius]) {
      expect(Math.abs(1 / measured - 1 / fine.state[0])).toBeLessThan(1e-6);
    }
  });

  test("the changed higher-order pixel agrees with a converged null Hamiltonian reference", () => {
    const { space, radius, inclination, direction, eventTime } = criticalFixture;
    const source = normalize([direction[0] ?? 0, direction[1] ?? 0, direction[2] ?? 0]);
    const photon = photonFromLocal(space, radius, inclination, source);
    const coarse = referenceGeodesic(space, photon, radius, inclination, eventTime, 1e-11);
    const fine = referenceGeodesic(space, photon, radius, inclination, eventTime, 1e-13);
    expect(fine.maxResidual).toBeLessThan(1e-10);
    expect(Math.abs(coarse.state[0] - fine.state[0])).toBeLessThan(1e-6);
    expect(Math.abs(fine.state[0] - criticalFixture.referenceRadius)).toBeLessThan(1e-7);
    expect(Math.abs(fine.state[1] - Math.PI / 2)).toBeLessThan(1e-8);
    expect(Math.abs(1 / criticalFixture.directRadius - 1 / fine.state[0])).toBeLessThan(1e-5);
  });
});

describe("Disk crossings", () => {
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
});

describe("Capture and escape", () => {
  describe("analytic visibility", () => {
    test("Schwarzschild capture/escape boundary matches the local critical angle", () => {
      const space = { spin: 0, charge: 0 };
      const radius = 10;
      const critical = Math.asin((3 * Math.sqrt(3) * Math.sqrt(1 - 2 / radius)) / radius);
      for (const [offset, expected] of [
        [-0.02, "captured"],
        [0.02, "sky"],
      ] as const) {
        const angle = critical + offset;
        const photon = photonFromLocal(space, radius, Math.PI / 2, [
          -Math.cos(angle),
          0,
          Math.sin(angle),
        ]);
        const { outcome } = traceVisibility(space, photon, radius, Math.PI / 2);
        expect(outcome.kind).toBe(expected);
      }
    });

    test("an outward radial photon reaches the asymptotic boundary", () => {
      const space = { spin: 0, charge: 0 };
      const photon = photonFromLocal(space, 10, 1, [1, 0, 0]);
      const { outcome } = traceVisibility(space, photon, 10, 1);
      expect(outcome.kind).toBe("sky");
      if (outcome.kind === "sky") {
        expect(outcome.time).toBeCloseTo(0.1 / photon.energy, 12);
      }
    });
  });
});

describe("Azimuth and travel time", () => {
  test("polar-axis passage has a continuous sky direction as axial momentum changes sign", () => {
    const space = { spin: 0, charge: 0 };
    for (const azimuth of [-1e-7, 0, 1e-7]) {
      const photon = photonFromLocal(space, 8, 0.5, normalize([-0.2, -1, azimuth]));
      const path = prepareGeodesic(space, photon, 8, 0.5);
      const result = integrateTransport(space, photon, path, 0.1, false);
      expect(result.kind).toBe("resolved");
      if (result.kind !== "resolved") {
        throw new Error("Axis passage must resolve in this spherical case.");
      }
      expect(Math.cos(result.azimuth)).toBeCloseTo(-1, 10);
      expect(Math.abs(Math.sin(result.azimuth))).toBeLessThan(1e-6);
    }
  });

  test("complete Schwarzschild scattering approaches the weak-field 4/b deflection", () => {
    const space = { spin: 0, charge: 0 };
    const radius = 100000;
    for (const impact of [100, 200]) {
      const angle = Math.asin((impact * Math.sqrt(1 - 2 / radius)) / radius);
      const photon = photonFromLocal(space, radius, Math.PI / 2, [
        -Math.cos(angle),
        0,
        Math.sin(angle),
      ]);
      const { path, outcome } = traceVisibility(space, photon, radius, Math.PI / 2);
      expect(outcome.kind).toBe("sky");
      if (outcome.kind !== "sky") {
        throw new Error("Weak-field scattering must escape.");
      }
      const transport = integrateTransport(space, photon, path, outcome.time, false);
      expect(transport.kind).toBe("resolved");
      if (transport.kind !== "resolved") {
        throw new Error("Expected finite sky azimuth.");
      }
      // Restore the leading finite-observer angular tail before the weak-field comparison.
      const deflection = transport.azimuth - Math.PI + Math.asin(impact / radius);
      expect(Math.abs(deflection / (4 / impact) - 1)).toBeLessThan(0.04);
    }
  });

  test("analytic-path azimuth and coordinate time agree with the independent Hamiltonian", () => {
    for (const space of [
      { spin: 0, charge: 0 },
      { spin: 0.8, charge: 0.4 },
      { spin: -0.8, charge: 0.4 },
    ]) {
      for (const direction of [
        [-1, 0.3, 0.5],
        [0, -0.3, -0.8],
      ] as const) {
        const r = 8;
        const theta = 1.1;
        const time = 0.04;
        const photon = photonFromLocal(space, r, theta, normalize(direction));
        const path = prepareGeodesic(space, photon, r, theta);
        const actual = integrateTransport(space, photon, path, time, true, 1e-10);
        const reference = referenceGeodesic(space, photon, r, theta, time, 1e-13);
        expect(actual.kind).toBe("resolved");
        if (actual.kind !== "resolved") {
          throw new Error("Expected ordinary exterior transport.");
        }
        expect(actual.azimuth).toBeCloseTo(reference.state[2], 9);
        expect(actual.coordinateTime).toBeCloseTo(reference.state[3], 9);
        expect(actual.coordinateTime).toBeLessThan(0);
      }
    }
  });

  test("sky azimuth stays finite at inverse radius zero, without integrating divergent time", () => {
    const space = { spin: 0, charge: 0 };
    const photon = photonFromLocal(space, 10, 1.1, [1, 0, 0]);
    const path = prepareGeodesic(space, photon, 10, 1.1);
    const actual = integrateTransport(space, photon, path, 0.1 / photon.energy, false);
    expect(actual.kind).toBe("resolved");
    if (actual.kind === "resolved") {
      expect(actual.azimuth).toBe(0);
      expect(actual.coordinateTime).toBe(0);
    }
    expect(integrateTransport(space, photon, path, 0.01, true, 1e-15, 3).kind).toBe("unresolved");
  });
});
