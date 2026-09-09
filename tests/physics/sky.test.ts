import { brightStars } from "../../src/data/bright-stars.ts";
import { createStarTree } from "../../src/physics/stars.ts";
import { expect, test } from "vitest";
import { createSkyMap, createStars, cubeTexelSolidAngle } from "../../src/physics/sky.ts";

test("cube texels cover exactly four pi steradians at every mip size", () => {
  for (const size of [1, 2, 8, 64]) {
    let area = 0;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        area += 6 * cubeTexelSolidAngle(size, x, y);
      }
    }
    expect(Math.abs(area - 4 * Math.PI)).toBeLessThan(1e-12);
  }
});

test("diffuse radiance retains integrated spectral flux through every mip", () => {
  const levels = createSkyMap(64);
  const integrals = levels.map((level) => {
    const integral = [0, 0, 0];
    for (let face = 0; face < 6; face++) {
      for (let row = 0; row < level.size; row++) {
        for (let col = 0; col < level.size; col++) {
          const offset = ((face * level.size + row) * level.size + col) * 4;
          for (let channel = 0; channel < 3; channel++) {
            integral[channel] =
              (integral[channel] ?? 0) +
              (level.data[offset + channel] ?? 0) * cubeTexelSolidAngle(level.size, col, row);
          }
        }
      }
    }
    return integral;
  });
  for (const integral of integrals) {
    for (let channel = 0; channel < 3; channel++) {
      expect(
        Math.abs((integral[channel] ?? 0) / (integrals[0]?.[channel] ?? Number.NaN) - 1),
      ).toBeLessThan(0.01);
    }
  }
});

test("HYG bright stars retain source positions, magnitudes, and missing-color records", () => {
  const stars = createStars();
  expect(stars).toHaveLength(8920);
  expect(new Set(brightStars.map(([id]) => id)).size).toBe(stars.length);
  expect(brightStars.filter(([, , , , color]) => color === null)).toHaveLength(40);
  const sirius = stars[brightStars.findIndex(([id]) => id === 32263)];
  const polaris = stars[brightStars.findIndex(([id]) => id === 11734)];
  if (!sirius || !polaris) {
    throw new Error("Missing reference star.");
  }
  expect((Math.atan2(sirius.direction[1], sirius.direction[0]) * 12) / Math.PI).toBeCloseTo(
    6.752481,
    5,
  );
  expect((Math.asin(sirius.direction[2]) * 180) / Math.PI).toBeCloseTo(-16.716116, 5);
  expect((Math.asin(polaris.direction[2]) * 180) / Math.PI).toBeCloseTo(89.264109, 5);
  expect(sirius.flux / polaris.flux).toBeCloseTo(10 ** (0.4 * (1.97 + 1.44)), 10);
  for (const [index, star] of stars.entries()) {
    expect(Math.hypot(...star.direction)).toBeCloseTo(1, 14);
    expect(star.flux).toBeGreaterThan(0);
    expect(star.temperature).toBeGreaterThan(1800);
    expect(star.temperature).toBeLessThan(18000);
    if (brightStars[index]?.[4] === null) {
      expect(star.temperature).toBe(6500);
    }
  }
  const tree = createStarTree(stars);
  expect(tree.byteLength).toBeLessThanOrEqual(stars.length * 64);
  expect(tree.every(Number.isFinite)).toBe(true);
});
