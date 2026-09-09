import { expect, test } from "vitest";
import criticalFixture from "../fixtures/critical-event-order.json" with { type: "json" };
import { createScene } from "../../src/model/scene.ts";
import { normalize } from "../../src/physics/vector.ts";
import type { Vec3 } from "../../src/physics/vector.ts";
import { traceDirections } from "./visibility-probe.ts";

test("Schwarzschild capture and higher-order images approach the analytic critical impact parameter", async () => {
  const critical = 3 * Math.sqrt(3);
  const impacts = [
    critical - 0.03,
    critical - 0.01,
    critical - 0.003,
    ...Array.from({ length: 48 }, (_, index) => critical + 0.001 * 1000 ** (index / 47)),
  ];
  const directions: Vec3[] = impacts.map((impact) => {
    const sine = (impact * Math.sqrt(1 - 2 / 30)) / 30;
    return [-Math.sqrt(1 - sine * sine), sine, 0];
  });
  const result = await traceDirections(directions, 0);
  expect(
    impacts
      .map((impact, index) => ({
        impact,
        stage: result[index * 4],
        candidateTime: result[index * 4 + 1],
        residual: result[index * 4 + 2],
        tag: result[index * 4 + 3],
      }))
      .filter((sample) => sample.tag === -2),
  ).toEqual([]);
  let higherOrders = 0;
  let secondary = 0;
  for (const [index, impact] of impacts.entries()) {
    const tag = result[index * 4 + 3] ?? Number.NaN;
    if (impact < critical) {
      expect(tag).toBe(0);
    } else {
      expect(tag).not.toBe(0);
      if (tag <= -5) {
        higherOrders++;
        expect(impact - critical).toBeLessThan(0.06);
      }
      if (tag === -4) {
        secondary++;
      }
    }
  }
  expect(higherOrders).toBeGreaterThan(0);
  expect(secondary).toBeGreaterThan(0);
});

test("approaching and receding disk rays recover the Schwarzschild endpoint frequency law", async () => {
  const inclination = Math.fround(1.2);
  const directions = [normalize([-1, 0.1, -0.3]), normalize([-1, 0.1, 0.3])];
  const result = await traceDirections(directions, inclination);
  for (const [index, direction] of directions.entries()) {
    const radius = result[index * 4] ?? Number.NaN;
    const frequency = result[index * 4 + 2] ?? Number.NaN;
    expect(result[index * 4 + 3]).toBeLessThanOrEqual(-3);
    const energy = Math.sqrt(1 - 2 / 30);
    const momentum = -Math.fround(direction[2]) * 30 * Math.sin(inclination);
    const expected = Math.sqrt(1 - 3 / radius) / (energy - radius ** -1.5 * momentum);
    expect(Math.abs(frequency - expected)).toBeLessThan(2e-5);
  }
  expect(Math.abs((result[0] ?? Number.NaN) - (result[4] ?? Number.NaN))).toBeLessThan(1e-4);
  expect(result[2]).toBeGreaterThan(result[6] ?? Number.NaN);
});

test("higher-order endpoint stays within the independent critical-pixel reference", async () => {
  const { space, radius, inclination, direction } = criticalFixture;
  const scene = createScene({
    space,
    observer: { radius, inclination, azimuth: 0, fieldOfView: 1 },
  });
  if (!scene.ok) {
    throw new Error(scene.error);
  }
  const result = await traceDirections(
    [normalize([direction[0] ?? 0, direction[1] ?? 0, direction[2] ?? 0])],
    inclination,
    scene.value,
  );
  expect(result[3]).toBe(-5);
  expect(
    Math.abs(1 / (result[0] ?? Number.NaN) - 1 / criticalFixture.referenceRadius),
  ).toBeLessThan(1e-5);
  expect(Math.abs((result[1] ?? Number.NaN) - criticalFixture.referencePhi)).toBeLessThan(5e-5);
});

test("an exact Schwarzschild spherical launch remains unresolved rather than captured", async () => {
  const scene = createScene({
    space: { spin: 0, charge: 0 },
    observer: { radius: 3, inclination: 1.2, azimuth: 0, fieldOfView: 1 },
  });
  if (!scene.ok) {
    throw new Error(scene.error);
  }
  const result = await traceDirections([[0, 1, 0]], 1.2, scene.value);
  expect(result[3]).toBe(-2);
  expect(result[0]).toBe(14);
});
