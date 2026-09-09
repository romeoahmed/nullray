import { expect, test } from "vitest";
import { photonFromLocal } from "../../src/physics/photon.ts";
import { prepareGeodesic } from "../reference/geodesic.ts";
import { integrateTransport } from "../reference/transport.ts";
import { normalize } from "../../src/physics/vector.ts";
import { referenceGeodesic } from "../reference/hamiltonian.ts";
import { traceVisibility } from "../reference/visibility.ts";

test("polar-axis passage has a continuous sky direction as axial momentum changes sign", () => {
  const space = { spin: 0, charge: 0 };
  for (const azimuth of [-1e-7, 0, 1e-7]) {
    const photon = photonFromLocal(space, 8, 0.5, normalize([-0.2, -1, azimuth]));
    const path = prepareGeodesic(space, photon, 8, 0.5);
    const result = integrateTransport(space, photon, path, 0.1, false);
    expect(result.kind).toBe("resolved");
    if (result.kind !== "resolved") {
      throw new Error("Axis passage must resolve in this spherical case.");
    }
    expect(Math.cos(result.azimuth)).toBeCloseTo(-1, 10);
    expect(Math.abs(Math.sin(result.azimuth))).toBeLessThan(1e-6);
  }
});

test("complete Schwarzschild scattering approaches the weak-field 4/b deflection", () => {
  const space = { spin: 0, charge: 0 };
  const radius = 100000;
  for (const impact of [100, 200]) {
    const angle = Math.asin((impact * Math.sqrt(1 - 2 / radius)) / radius);
    const photon = photonFromLocal(space, radius, Math.PI / 2, [
      -Math.cos(angle),
      0,
      Math.sin(angle),
    ]);
    const { path, outcome } = traceVisibility(space, photon, radius, Math.PI / 2);
    expect(outcome.kind).toBe("sky");
    if (outcome.kind !== "sky") {
      throw new Error("Weak-field scattering must escape.");
    }
    const transport = integrateTransport(space, photon, path, outcome.time, false);
    expect(transport.kind).toBe("resolved");
    if (transport.kind !== "resolved") {
      throw new Error("Expected finite sky azimuth.");
    }
    // Restore the leading finite-observer angular tail before the weak-field comparison.
    const deflection = transport.azimuth - Math.PI + Math.asin(impact / radius);
    expect(Math.abs(deflection / (4 / impact) - 1)).toBeLessThan(0.04);
  }
});

test("analytic-path azimuth and coordinate time agree with the independent Hamiltonian", () => {
  for (const space of [
    { spin: 0, charge: 0 },
    { spin: 0.8, charge: 0.4 },
    { spin: -0.8, charge: 0.4 },
  ]) {
    for (const direction of [
      [-1, 0.3, 0.5],
      [0, -0.3, -0.8],
    ] as const) {
      const r = 8;
      const theta = 1.1;
      const time = 0.04;
      const photon = photonFromLocal(space, r, theta, normalize(direction));
      const path = prepareGeodesic(space, photon, r, theta);
      const actual = integrateTransport(space, photon, path, time, true, 1e-10);
      const reference = referenceGeodesic(space, photon, r, theta, time, 1e-13);
      expect(actual.kind).toBe("resolved");
      if (actual.kind !== "resolved") {
        throw new Error("Expected ordinary exterior transport.");
      }
      expect(actual.azimuth).toBeCloseTo(reference.state[2], 9);
      expect(actual.coordinateTime).toBeCloseTo(reference.state[3], 9);
      expect(actual.coordinateTime).toBeLessThan(0);
    }
  }
});

test("sky azimuth stays finite at inverse radius zero, without integrating divergent time", () => {
  const space = { spin: 0, charge: 0 };
  const photon = photonFromLocal(space, 10, 1.1, [1, 0, 0]);
  const path = prepareGeodesic(space, photon, 10, 1.1);
  const actual = integrateTransport(space, photon, path, 0.1 / photon.energy, false);
  expect(actual.kind).toBe("resolved");
  if (actual.kind === "resolved") {
    expect(actual.azimuth).toBe(0);
    expect(actual.coordinateTime).toBe(0);
  }
  expect(integrateTransport(space, photon, path, 0.01, true, 1e-15, 3).kind).toBe("unresolved");
});
