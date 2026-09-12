import { expect, test } from "vitest";
import * as fc from "fast-check";
import { sampleCounts, sampleOffset } from "../../src/gpu/imaging/sampling.ts";

test("settled pixel quadrature spreads unique samples and integrates affine radiance without bias", () => {
  const points = Array.from({ length: sampleCounts.settle }, (_, i) => sampleOffset("settle", i));
  expect(new Set(points.map(([x, y]) => `${x},${y}`)).size).toBe(points.length);
  for (const axis of [0, 1] as const) {
    const values = points.map((point) => point[axis]);
    expect(Math.min(...values)).toBeLessThan(-0.25);
    expect(Math.max(...values)).toBeGreaterThan(0.25);
  }
  fc.assert(
    fc.property(
      fc.tuple(
        fc.double({ min: -100, max: 100, noNaN: true }),
        fc.double({ min: -100, max: 100, noNaN: true }),
        fc.double({ min: 0, max: 100, noNaN: true }),
      ),
      ([a, b, c]) => {
        const mean = points.reduce((sum, [x, y]) => sum + a * x + b * y + c, 0) / points.length;
        expect(Math.abs(mean - c)).toBeLessThan(1e-12);
      },
    ),
  );
});

test("image sample sequences stay inside the detector pixel and reject exhausted work", () => {
  for (const mode of ["live", "settle", "photograph"] as const) {
    for (let i = 0; i < sampleCounts[mode]; i++) {
      expect(sampleOffset(mode, i).every((x) => Number.isFinite(x) && x > -0.5 && x < 0.5)).toBe(
        true,
      );
    }
    for (const index of [-1, 0.5, sampleCounts[mode], NaN]) {
      expect(() => sampleOffset(mode, index)).toThrow(RangeError);
    }
  }
});
