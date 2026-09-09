import { expect, test } from "vitest";
import {
  blackbodyXYZ,
  createBlackbodyTable,
  diskFrequencyRatio,
  planck,
  xyzToLinearRGB,
} from "../../src/physics/radiation.ts";

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
      expect(Math.abs(left + weight * (right - left) - target)).toBeLessThan(0.005 * scale + 1e-9);
    }
  }
});

test("Schwarzschild disk frequency includes orbital time dilation", () => {
  const photon = { energy: 1, angularMomentum: 0, carter: 0, radialVelocity: 1, polarVelocity: 0 };
  expect(diskFrequencyRatio({ spin: 0, charge: 0 }, photon, 6, 1)).toBeCloseTo(Math.sqrt(0.5), 12);
});
