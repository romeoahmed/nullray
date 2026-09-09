import { describe, expect, test } from "vitest";
import { photonFromLocal } from "../../src/physics/photon.ts";
import { traceVisibility } from "../reference/visibility.ts";

describe("analytic visibility", () => {
  test("Schwarzschild capture/escape boundary matches the local critical angle", () => {
    const space = { spin: 0, charge: 0 };
    const radius = 10;
    const critical = Math.asin((3 * Math.sqrt(3) * Math.sqrt(1 - 2 / radius)) / radius);
    for (const [offset, expected] of [
      [-0.02, "captured"],
      [0.02, "sky"],
    ] as const) {
      const angle = critical + offset;
      const photon = photonFromLocal(space, radius, Math.PI / 2, [
        -Math.cos(angle),
        0,
        Math.sin(angle),
      ]);
      const { outcome } = traceVisibility(space, photon, radius, Math.PI / 2);
      expect(outcome.kind).toBe(expected);
    }
  });

  test("an outward radial photon reaches the asymptotic boundary", () => {
    const space = { spin: 0, charge: 0 };
    const photon = photonFromLocal(space, 10, 1, [1, 0, 0]);
    const { outcome } = traceVisibility(space, photon, 10, 1);
    expect(outcome.kind).toBe("sky");
    if (outcome.kind === "sky") {
      expect(outcome.time).toBeCloseTo(0.1 / photon.energy, 12);
    }
  });
});
