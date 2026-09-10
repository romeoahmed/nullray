import { describe, expect, test } from "vitest";
import { carlsonRF, jacobi } from "../reference/elliptic.ts";
import { evaluateQuartic, prepareQuartic } from "../reference/quartic.ts";
import type { Quartic } from "../reference/quartic.ts";

/** Independent fixed-step RK4 on x'' = f'(x)/2, used only in these tests. */
function reference(coefficients: Quartic, initial: number, velocity: number, time: number): number {
  const [, c1, c2, c3, c4] = coefficients;
  const acceleration = (x: number) => (((4 * c4 * x + 3 * c3) * x + 2 * c2) * x) / 2 + c1 / 2;
  let x = initial;
  let v = velocity;
  const h = time / 4096;
  for (let i = 0; i < 4096; i++) {
    const a1 = acceleration(x);
    const v2 = v + (h * a1) / 2;
    const a2 = acceleration(x + (h * v) / 2);
    const v3 = v + (h * a2) / 2;
    const a3 = acceleration(x + (h * v2) / 2);
    const v4 = v + h * a3;
    const a4 = acceleration(x + h * v3);
    x += (h * (v + 2 * v2 + 2 * v3 + v4)) / 6;
    v += (h * (a1 + 2 * a2 + 2 * a3 + a4)) / 6;
  }
  return x;
}

describe("Real elliptic functions", () => {
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
});

describe("Quartic trajectories", () => {
  describe("quartic analytic trajectories", () => {
    test("recovers reduced-degree harmonic and quartic elementary solutions", () => {
      const oscillator = prepareQuartic([1, 0, -1, 0, 0], 0, 1);
      const rational = prepareQuartic([0, 0, 0, 0, 1], 1, 1);
      for (const t of [1e-8, 0.1, 0.3, 0.7]) {
        expect(evaluateQuartic(oscillator, t)).toBeCloseTo(Math.sin(t), 11);
        expect(evaluateQuartic(rational, t)).toBeCloseTo(1 / (1 - t), 11);
      }
    });

    test("matches independent integration for both directions and cubic discriminant branches", () => {
      const cases: Quartic[] = [
        [1, 0, 0, 0, 1],
        [1, 0.4, -2, 0, 0.3],
        [2, -1, 0.5, 0.2, -0.1],
      ];
      for (const coefficients of cases) {
        for (const sign of [-1, 1]) {
          const velocity = sign * Math.sqrt(coefficients[0]);
          const path = prepareQuartic(coefficients, 0, velocity);
          for (const t of [0.01, 0.3, 1]) {
            expect(evaluateQuartic(path, t)).toBeCloseTo(
              reference(coefficients, 0, velocity, t),
              9,
            );
          }
        }
      }
    });
  });
});
