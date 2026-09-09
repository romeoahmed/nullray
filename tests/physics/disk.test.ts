import { expect, test } from "vitest";
import { createDiskProfile } from "../../src/physics/disk.ts";
import { isco } from "../../src/physics/spacetime.ts";

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
      expect(Math.abs((profile.flux[index] ?? Number.NaN) - kerrFlux(spin, inner, r))).toBeLessThan(
        peak * 2e-7,
      );
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
