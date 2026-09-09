import { expect, test } from "vitest";
import * as fc from "fast-check";
import { createScene, initialScene } from "../../src/model/scene.ts";
import { outerHorizon } from "../../src/physics/spacetime.ts";

test("scene construction retains a valid coupled spacetime after GPU quantization", () => {
  fc.assert(
    fc.property(
      fc.double({ min: 0, max: 0.99, noNaN: true }),
      fc.double({ min: -Math.PI, max: Math.PI, noNaN: true }),
      fc.double({ min: 2.1, max: 200, noNaN: true }),
      (magnitude, angle, radius) => {
        const result = createScene({
          ...initialScene,
          space: { spin: magnitude * Math.cos(angle), charge: magnitude * Math.sin(angle) },
          observer: { ...initialScene.observer, radius },
        });
        expect(result.ok).toBe(true);
        if (!result.ok) {
          throw new Error(result.error);
        }
        const { space, observer, diskInner, diskOuter } = result.value;
        expect(space.spin ** 2 + space.charge ** 2).toBeLessThan(1);
        expect(observer.radius).toBeGreaterThan(outerHorizon(space));
        expect(diskInner).toBeGreaterThan(outerHorizon(space));
        expect(diskInner).toBeLessThan(diskOuter);
        expect(Math.fround(observer.radius)).toBe(observer.radius);
      },
    ),
  );
});

test("scene construction rejects invalid inputs and boundaries erased by quantization", () => {
  for (const space of [
    { spin: 0.8, charge: 0.8 },
    { spin: 1 - Number.EPSILON, charge: 0 },
    { spin: NaN, charge: 0 },
  ]) {
    expect(createScene({ ...initialScene, space }).ok).toBe(false);
  }
  for (const radius of [0, 2, 2 + 1e-9, Infinity]) {
    expect(
      createScene({
        ...initialScene,
        space: { spin: 0, charge: 0 },
        observer: { ...initialScene.observer, radius },
      }).ok,
    ).toBe(false);
  }
  for (const fieldOfView of [0, 1e-99, Math.PI, Infinity]) {
    expect(
      createScene({ ...initialScene, observer: { ...initialScene.observer, fieldOfView } }).ok,
    ).toBe(false);
  }
  for (const inclination of [0, Math.PI]) {
    expect(
      createScene({ ...initialScene, observer: { ...initialScene.observer, inclination } }).ok,
    ).toBe(true);
  }
});

const equatorial = (radius: number, inclination = Math.PI / 2) =>
  createScene({
    ...initialScene,
    observer: { ...initialScene.observer, radius, inclination },
  });

test("scene construction excludes the emitting surface after quantization without excluding its hole", () => {
  for (const radius of [
    initialScene.diskInner,
    30,
    initialScene.diskOuter,
    initialScene.diskOuter + 1e-8,
  ]) {
    expect(equatorial(radius)).toEqual({
      ok: false,
      error: "The observer cannot lie on the emitting disk surface.",
    });
  }
  expect(equatorial(30, Math.PI / 2 + 1e-8).ok).toBe(false);
  expect(equatorial((outerHorizon(initialScene.space) + initialScene.diskInner) / 2).ok).toBe(true);
  expect(equatorial(initialScene.diskOuter + 1e-4).ok).toBe(true);
  expect(equatorial(30, Math.PI / 2 - 1e-4).ok).toBe(true);
  expect(equatorial(30, Math.PI / 2 + 1e-4).ok).toBe(true);
});
