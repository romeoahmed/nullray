import { cubeTexelSolidAngle } from "../reference/sky.ts";
import { describe, expect, test } from "vitest";
import { readFileSync } from "node:fs";
import { createDiskProfile, diskColumnNormalization } from "../../src/physics/disk.ts";
import { isco } from "../../src/physics/spacetime.ts";
import {
  blackbodyWhiteBalance,
  blackbodyXYZ,
  createBlackbodyTable,
  diskFrequencyRatio,
  planck,
  xyzToLinearRGB,
} from "../../src/physics/radiation.ts";
import { brightStars } from "../../src/data/bright-stars.ts";
import { createStarTree } from "../../src/physics/stars.ts";
import { createJetTable } from "../../src/physics/synchrotron.ts";
import { initialJet } from "../../src/scene/jet.ts";
import { createStars, decodeFaintStars } from "../../src/physics/sky.ts";

test("finite-cutoff synchrotron recovers the analytic isotropic power-law tail", () => {
  const jet = { ...initialJet, gammaMin: 1 };
  const observing = 1e14;
  const table = createJetTable(jet, observing / 1e9);
  const shift = Math.exp(10) / table.scale;
  const light = 2.99792458e10;
  const coefficient =
    ((((4 * Math.PI) / 9) * 2.307077552e-19 * (2.799248987e6 * jet.field) ** 2) / light) *
    jet.density *
    1.476625038e5 *
    jet.massSolar;
  const thermal =
    (2 * 6.62607015e-27 * observing ** 3) /
    light ** 2 /
    Math.expm1((6.62607015e-27 * observing) / (1.380649e-16 * 6500));
  const expected = coefficient / (observing * shift) / thermal;
  expect(Math.abs((table.data.at(-4) ?? NaN) / expected - 1)).toBeLessThan(0.002);
});

/** Page–Thorne (1974), equation 15n: independent closed Kerr expression. */
function kerrFlux(a: number, inner: number, r: number): number {
  const x = Math.sqrt(r);
  const x0 = Math.sqrt(inner);
  const roots = [
    2 * Math.cos(Math.acos(a) / 3 - Math.PI / 3),
    2 * Math.cos(Math.acos(a) / 3 + Math.PI / 3),
    -2 * Math.cos(Math.acos(a) / 3),
  ];
  let integral = x - x0 - 1.5 * a * Math.log(x / x0);
  for (const root of roots) {
    const others = roots.filter((value) => value !== root);
    const product = others.reduce((value, other) => value * (root - other), 1);
    integral -= ((3 * (root - a) ** 2) / (root * product)) * Math.log((x - root) / (x0 - root));
  }
  return (3 * integral) / (8 * Math.PI * x ** 4 * (x ** 3 - 3 * x + 2 * a));
}

describe("Charged thermal disks", () => {
  test("tapered disk columns retain their peak optical depth for compact and extended annuli", () => {
    for (const outer of [6.01, 14, 36, 1000]) {
      const scale = diskColumnNormalization(6, outer);
      let peak = 0;
      let minimum = 1;
      // Uniform source-coordinate sampling supplies an independent maximum estimate.
      for (let index = 0; index <= 10000; index++) {
        const x = index / 10000;
        const r = 6 + (outer - 6) * x;
        const depth = scale * 6.75 * x * (1 - x) ** 2 * (6 / r) ** 2;
        minimum = Math.min(minimum, depth);
        peak = Math.max(peak, depth);
      }
      expect(minimum).toBeGreaterThanOrEqual(0);
      expect(peak).toBeLessThanOrEqual(1 + 1e-12);
      expect(peak).toBeGreaterThan(0.9999);
    }
  });
  test.each([-0.9, -0.4, 0.4, 0.9, 0.99])(
    "disk flux matches the independent closed Kerr profile at spin %s",
    (spin) => {
      const space = { spin, charge: 0 };
      const inner = isco(space, 1);
      const profile = createDiskProfile(space, inner, 64);
      const peak = Math.max(...profile.flux);
      expect(profile.flux[0]).toBe(0);
      for (let index = 1; index < profile.flux.length; index++) {
        const r = inner * (64 / inner) ** (index / (profile.flux.length - 1));
        expect(
          Math.abs((profile.flux[index] ?? Number.NaN) - kerrFlux(spin, inner, r)),
        ).toBeLessThan(peak * 2e-7);
      }
    },
  );

  test("charged neutral-emitter profiles converge under radial grid refinement", () => {
    for (const space of [
      { spin: 0, charge: 0.9 },
      { spin: 0.7, charge: 0.2 },
      { spin: -0.7, charge: 0.5 },
    ]) {
      const inner = isco(space, 1);
      const coarse = createDiskProfile(space, inner, 64, 257);
      const fine = createDiskProfile(space, inner, 64, 513);
      const peak = Math.max(...fine.flux);
      for (let index = 0; index < coarse.flux.length; index++) {
        expect(
          Math.abs((coarse.flux[index] ?? Number.NaN) - (fine.flux[2 * index] ?? Number.NaN)),
        ).toBeLessThan(peak * 1e-6);
      }
    }
  });
});

describe("Spectral transfer", () => {
  test("blackbody temperature shifting obeys wavelength radiance transport", () => {
    for (const g of [0.3, 1, 3]) {
      for (const wavelength of [400e-9, 550e-9, 700e-9]) {
        const shifted = planck(wavelength, 6000 * g);
        const transported = g ** 5 * planck(wavelength * g, 6000);
        expect(shifted / transported).toBeCloseTo(1, 12);
      }
    }
    expect(planck(500e-9, 0)).toBe(0);
  });

  test("CIE integration retains the Planckian chromaticity and absolute brightness", () => {
    const xyz = blackbodyXYZ(6500);
    const sum = xyz[0] + xyz[1] + xyz[2];
    expect(xyz[0] / sum).toBeCloseTo(0.3135, 3);
    expect(xyz[1] / sum).toBeCloseTo(0.3236, 3);
    expect(blackbodyXYZ(13000)[1]).toBeGreaterThan(4 * xyz[1]);
    for (const temperature of [2500, 5000, 12000]) {
      const reference = blackbodyXYZ(temperature);
      const [r, g, b] = xyzToLinearRGB(reference);
      const p3 = [
        0.82246197 * r + 0.17753803 * g,
        0.0331942 * r + 0.9668058 * g,
        0.01708263 * r + 0.07239744 * g + 0.91051993 * b,
      ];
      const gains = blackbodyWhiteBalance(temperature);
      for (const [channel, value] of p3.entries()) {
        expect((value * (gains[channel] ?? NaN)) / reference[1]).toBeCloseTo(1, 12);
      }
    }
  });

  test("log-temperature lookup agrees with direct spectral integration", () => {
    const table = createBlackbodyTable();
    const normalization = blackbodyXYZ(6500)[1];
    for (let i = 0; i < 80; i++) {
      const temperature = 800 * (1000000 / 800) ** (i / 79);
      const coordinate =
        (Math.log(temperature / table.minimum) / Math.log(table.maximum / table.minimum)) *
        (table.size - 1);
      const lower = Math.min(table.size - 2, Math.floor(coordinate));
      const weight = coordinate - lower;
      const expected = xyzToLinearRGB(blackbodyXYZ(temperature)).map((x) => x / normalization);
      const scale = Math.max(...expected.map(Math.abs));
      for (let channel = 0; channel < 3; channel++) {
        const left = table.data[lower * 4 + channel];
        const right = table.data[(lower + 1) * 4 + channel];
        const target = expected[channel];
        if (left === undefined || right === undefined || target === undefined) {
          throw new Error("Missing lookup channel.");
        }
        expect(Math.abs(left + weight * (right - left) - target)).toBeLessThan(
          0.005 * scale + 1e-9,
        );
      }
    }
  });

  test("Schwarzschild disk frequency includes orbital time dilation", () => {
    const photon = {
      energy: 1,
      angularMomentum: 0,
      carter: 0,
      radialVelocity: 1,
      polarVelocity: 0,
    };
    expect(diskFrequencyRatio({ spin: 0, charge: 0 }, photon, 6, 1)).toBeCloseTo(
      Math.sqrt(0.5),
      12,
    );
  });
});

describe("Distant sky", () => {
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
    const data = new Uint8Array(
      readFileSync(new URL("../../src/data/faint-stars.bin", import.meta.url)),
    ).buffer;
    const faint = decodeFaintStars(data);
    expect(faint).toHaveLength(99151);
    expect(faint.every((star) => star.flux > 0 && star.flux < 4e-5 * 10 ** (-0.4 * 6.5))).toBe(
      true,
    );
    expect(() => decodeFaintStars(data.slice(0, data.byteLength - 1))).toThrow();
  });
});
