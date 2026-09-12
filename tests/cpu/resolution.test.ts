import { expect, test } from "vitest";
import * as fc from "fast-check";
import { resolutionScale, nextPixelBudget } from "../../src/runtime/worker/resolution.ts";

test("explicit resolution stays exact and only Auto playback uses the pixel budget", () => {
  fc.assert(
    fc.property(
      fc.integer({ min: 1, max: 16384 }),
      fc.integer({ min: 1, max: 16384 }),
      fc.double({ min: 0.25, max: 1, noNaN: true }),
      fc.integer({ min: 14400, max: 2073600 }),
      (width, height, ceiling, budget) => {
        expect(resolutionScale(ceiling, true, width, height, budget)).toBe(ceiling);
        expect(resolutionScale(ceiling, false, width, height, budget)).toBe(ceiling);
        expect(resolutionScale("auto", false, width, height, budget)).toBe(1);
        const scale = resolutionScale("auto", true, width, height, budget);
        expect(scale).toBeGreaterThan(0);
        expect(scale).toBeLessThanOrEqual(1);
        expect(Math.floor(width * scale) * Math.floor(height * scale)).toBeLessThanOrEqual(budget);
      },
    ),
  );
});

test("live budget converges under a pixel-proportional load without resizing in its dead band", () => {
  let budget = 640 * 360;
  for (let i = 0; i < 30; i++) {
    budget = nextPixelBudget(budget, budget, budget / 2000);
  }
  expect(budget / 2000).toBeGreaterThanOrEqual(24);
  expect(budget / 2000).toBeLessThanOrEqual(42);
  expect(nextPixelBudget(budget, budget, 33)).toBe(budget);
  expect(nextPixelBudget(budget, budget, NaN)).toBe(budget);
  expect(nextPixelBudget(budget, budget, 0)).toBe(budget);
});
