import { expect, test } from "vitest";
import criticalFixture from "../fixtures/critical-event-order.json" with { type: "json" };
import diskFixture from "../fixtures/outer-disk-arrival.json" with { type: "json" };
import radialFixture from "../fixtures/nearly-radial-arrival.json" with { type: "json" };
import { referenceGeodesic } from "../reference/hamiltonian.ts";
import * as fc from "fast-check";
import { quarticArrival, prepareQuartic, evaluateQuartic } from "../reference/quartic.ts";
import { prepareGeodesic } from "../reference/geodesic.ts";
import { photonFromLocal } from "../../src/physics/photon.ts";
import { outerHorizon } from "../../src/physics/spacetime.ts";
import { traceVisibility } from "../reference/visibility.ts";
import { normalize } from "../../src/physics/vector.ts";

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
