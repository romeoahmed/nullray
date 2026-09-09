import { describe, expect, test } from "vitest";
import { carlsonRF, jacobi } from "../reference/elliptic.ts";

describe("real elliptic functions", () => {
  test("RF has the elementary equal-argument and complete trigonometric limits", () => {
    expect(carlsonRF(1, 1, 1)).toBe(1);
    expect(carlsonRF(0, 1, 1)).toBeCloseTo(Math.PI / 2, 14);
    expect(carlsonRF(4, 4, 4)).toBe(0.5);
    expect(carlsonRF(1e200, 1e200, 1e200)).toBeCloseTo(1e-100, 110);
    expect(() => carlsonRF(0, 0, 1)).toThrow(RangeError);
    expect(() => carlsonRF(-1, 2, 3)).toThrow(RangeError);
  });

  test("Jacobi identities and period survive positive and negative parameters", () => {
    for (const m of [-10, -1, 0, 0.3, 0.9, 0.999999]) {
      const quarter = carlsonRF(0, 1 - m, 1);
      expect(jacobi(quarter, m).sn).toBeCloseTo(1, 12);
      for (const u of [-13, -1, 0, 0.2, 2, 9]) {
        const v = jacobi(u, m);
        expect(v.sn ** 2 + v.cn ** 2).toBeCloseTo(1, 13);
        expect(v.dn ** 2 + m * v.sn ** 2).toBeCloseTo(1, 12);
        expect(jacobi(u + 4 * quarter, m).sn).toBeCloseTo(v.sn, 11);
        const h = 1e-5;
        const derivative = (jacobi(u + h, m).sn - jacobi(u - h, m).sn) / (2 * h);
        expect(derivative).toBeCloseTo(v.cn * v.dn, 7);
      }
    }
  });
});
