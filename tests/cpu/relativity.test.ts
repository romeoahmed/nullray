import { expect, test } from "vitest";
import * as fc from "fast-check";
import {
  fromCartesian,
  combine,
  horizons,
  inner,
  kerrGeometry,
  lower,
  raise,
  stationaryLimits,
  toCartesian,
} from "../../src/physics/geometry.ts";
import type { FourVector, KerrChart } from "../../src/physics/geometry.ts";
import {
  boostFrame,
  coordinateObserver,
  measuredFrequency,
  observerPhoton,
  principalFrame,
  staticObserver,
} from "../../src/physics/observer.ts";
import { motionFromTangent, motionRadialPotential } from "../../src/physics/motion.ts";
import {
  advanceGeodesic,
  conditionOrbit,
  createGeodesic,
  orbitPoint,
  orbitTangent,
  regularOrbit,
} from "../../src/physics/geodesic.ts";
import { chartPrimitives, chartSign } from "../../src/physics/atlas.ts";
import { referenceGeodesic } from "../reference/hamiltonian.ts";
import { polarizationScreen, walkerPenrose } from "../reference/polarization.ts";
import { plasmaCutoff } from "../../src/physics/plasma.ts";

test("vacuum and refracted sphere paths agree with independent metric Hamiltonian trajectories", () => {
  for (const { space, plasma } of [
    { space: { spin: 0.8, charge: 0.3 }, plasma: null },
    { space: { spin: 1.2, charge: 0.6 }, plasma: null },
    { space: { spin: 0.8, charge: 0.3 }, plasma: { amplitude: 50, scaleSquared: 100 } },
    { space: { spin: 1.2, charge: 0.6 }, plasma: { amplitude: 50, scaleSquared: 100 } },
  ]) {
    const radius = 8;
    const inclination = 0.9;
    const geometry = kerrGeometry(space, { radius, inclination, azimuth: 0, chart: "ingoing" });
    if (!geometry) {
      throw new Error("Regular fixture required.");
    }
    const observer = principalFrame(geometry);
    const vacuum = observerPhoton(observer, [0.6, 0.48, 0.64]);
    const speed = Math.sqrt(1 - (plasma ? plasmaCutoff(plasma, geometry) : 0));
    const tangent = combine(observer.velocity, 1 - speed, vacuum, speed);
    const motion = motionFromTangent(space, geometry, tangent, 0);
    const photon = {
      ...motion.constants,
      radialVelocity: motion.radialVelocity,
      polarVelocity: motion.polarVelocity,
    };
    const initial = createGeodesic(space, geometry, tangent, 0, 0, undefined, plasma);
    const conditioned = conditionOrbit(initial);
    if (!conditioned) {
      throw new Error("Regular initial chart required.");
    }
    const restored = orbitTangent(conditioned);
    expect(restored).toBeDefined();
    const duration = 0.015;
    const reference = referenceGeodesic(
      space,
      photon,
      radius,
      inclination,
      duration,
      1e-11,
      plasma ?? undefined,
    );
    const result = advanceGeodesic(initial, -duration, { tolerance: 1e-12 });
    expect(result.kind).toBe("complete");
    const point = orbitPoint(result.path);
    if (!point) {
      throw new Error("Expected finite endpoint.");
    }
    const before = chartPrimitives(space, radius);
    const after = chartPrimitives(space, point.radius);
    if (!before || !after) {
      throw new Error("Expected chart overlaps.");
    }
    expect(point.radius).toBeCloseTo(reference.state[0], 8);
    expect(point.inclination).toBeCloseTo(reference.state[1], 8);
    expect(point.azimuth - chartSign(point.chart) * after[0] + before[0]).toBeCloseTo(
      reference.state[2],
      8,
    );
    expect(result.path.state.radial[2] - chartSign(point.chart) * after[1] + before[1]).toBeCloseTo(
      reference.state[3],
      8,
    );
  }
});

test("distinguishes horizon topology from stationary limits and curvature singularities", () => {
  expect(horizons({ spin: 0, charge: 0 })).toEqual({ kind: "single", outer: 2 });
  expect(horizons({ spin: 1, charge: 0 })).toEqual({ kind: "extremal", radius: 1 });
  expect(horizons({ spin: 1.2, charge: 0 })).toEqual({ kind: "none" });
  const space = { spin: 0.7, charge: 0.2 };
  const roots = horizons(space);
  expect(roots.kind).toBe("pair");
  if (roots.kind !== "pair") {
    throw new Error("Expected two regular horizons.");
  }
  expect(roots.inner * roots.outer).toBeCloseTo(space.spin ** 2 + space.charge ** 2, 14);
  expect(stationaryLimits({ spin: 1.2, charge: 0 }, 0)).toBeUndefined();
  expect(stationaryLimits({ spin: 1.2, charge: 0 }, Math.PI / 2)?.[1]).toBe(2);
  const point = { radius: 0, inclination: Math.PI / 2, azimuth: 0, chart: "ingoing" as const };
  expect(kerrGeometry(space, point)).toBeUndefined();
  expect(kerrGeometry(space, { ...point, inclination: 0 })).toBeDefined();
  expect(kerrGeometry({ spin: 0, charge: 0.5 }, { ...point, inclination: 0 })).toBeUndefined();
});

test("principal rays cross both horizons and the regular disk into negative radius, then end at infinity", () => {
  const schwarzschild = { spin: 0, charge: 0 };
  const horizon = kerrGeometry(schwarzschild, {
    radius: 2,
    inclination: 0,
    azimuth: 0,
    chart: "ingoing",
  });
  if (!horizon) {
    throw new Error("Regular horizon fixture required.");
  }
  const launched = advanceGeodesic(createGeodesic(schwarzschild, horizon, [1, 0, 0, -1], 0), -0.1);
  expect(launched.kind).toBe("complete");
  expect(orbitPoint(launched.path)?.radius).toBeCloseTo(2.5, 10);
  expect(launched.path.block).toEqual({ kind: "exterior", universe: 0, side: 1 });
  const space = { spin: 0.8, charge: 0.3 };
  const geometry = kerrGeometry(space, { radius: 3, inclination: 0, azimuth: 0, chart: "ingoing" });
  if (!geometry) {
    throw new Error("Regular axial fixture required.");
  }
  // Exact ingoing principal null tangent: T'=1, z'=-1 in Cartesian KS coordinates.
  const path = createGeodesic(space, geometry, [1, 0, 0, -1], 0);
  const radius = -2;
  const duration = (Math.atan(3 / space.spin) - Math.atan(radius / space.spin)) / space.spin;
  const result = advanceGeodesic(path, duration, { tolerance: 1e-12 });
  expect(result.kind).toBe("complete");
  expect(orbitPoint(result.path)?.radius).toBeCloseTo(radius, 8);
  expect(result.path.state.radial[2]).toBeCloseTo(0, 11);
  const endpoint = advanceGeodesic(path, 10, { tolerance: 1e-12 });
  expect(endpoint.kind).toBe("infinity");
  expect(endpoint.elapsed).toBeCloseTo((Math.atan(3 / space.spin) + Math.PI / 2) / space.spin, 8);
  expect(orbitPoint(endpoint.path)).toBeUndefined();
});

test("Walker–Penrose screens retain the Carter norm and the equatorial parallel-transport solution", () => {
  const space = { spin: 0.8, charge: 0.3 };
  for (const radius of [8, 1, 0.1, -2]) {
    const g = kerrGeometry(space, { radius, inclination: 0.9, azimuth: 0.3, chart: "ingoing" });
    if (!g) {
      throw new Error("Regular fixture required.");
    }
    const observer = principalFrame(g);
    const p = observerPhoton(observer, [0.6, 0.48, 0.64]);
    const screen = polarizationScreen(space, g, p, observer.velocity);
    if (!screen) {
      throw new Error("Nonprincipal ray required.");
    }
    expect(inner(g, screen.first, screen.first)).toBeCloseTo(1, 9);
    expect(inner(g, screen.second, screen.second)).toBeCloseTo(1, 9);
    expect(inner(g, screen.first, screen.second)).toBeCloseTo(0, 9);
    expect(inner(g, screen.first, p)).toBeCloseTo(0, 9);
    expect(inner(g, screen.first, observer.velocity)).toBeCloseTo(0, 9);
    const constants = motionFromTangent(space, g, p, 0).constants;
    const wp = walkerPenrose(space, g, p, screen.first);
    expect(wp.real ** 2 + wp.imaginary ** 2).toBeCloseTo(
      constants.carter + (constants.angularMomentum - space.spin * constants.energy) ** 2,
      8,
    );
  }
  const g = kerrGeometry(space, {
    radius: 8,
    inclination: Math.PI / 2,
    azimuth: 0,
    chart: "ingoing",
  });
  if (!g) {
    throw new Error("Regular fixture required.");
  }
  const p = observerPhoton(principalFrame(g), [0.6, 0, 0.8]);
  const initial = walkerPenrose(space, g, p, [0, 0, 0, 1]);
  const result = advanceGeodesic(createGeodesic(space, g, p, 0), -0.02, { tolerance: 1e-12 });
  const point = orbitPoint(result.path);
  const tangent = orbitTangent(result.path);
  const end = point && kerrGeometry(space, point);
  if (!end || !tangent) {
    throw new Error("Regular endpoint required.");
  }
  const final = walkerPenrose(space, end, tangent, [0, 0, 0, 1]);
  expect(final.real).toBeCloseTo(initial.real, 8);
  expect(final.imaginary).toBeCloseTo(initial.imaginary, 8);
});

test("generic polarization transport agrees with the independent metric connection", () => {
  const space = { spin: 0.8, charge: 0.3 };
  const g = kerrGeometry(space, { radius: 6, inclination: 0.9, azimuth: 0, chart: "ingoing" });
  if (!g) {
    throw new Error("Regular polarization fixture required.");
  }
  const frame = principalFrame(g);
  const photon = observerPhoton(frame, [0.6, 0.48, 0.64]);
  const f = combine(combine(frame.radial, 0.8, frame.polar, -0.36), 1, frame.azimuthal, -0.48);
  const regular = fromCartesian(space, g, f);
  if (!regular) {
    throw new Error("Off-axis vector fixture required.");
  }
  const initial = walkerPenrose(space, g, photon, f);
  const motion = motionFromTangent(space, g, photon, 0);
  const bl: FourVector = [
    regular[0] - ((2 * g.point.radius - space.charge ** 2) / g.delta) * regular[1],
    regular[1],
    regular[2],
    regular[3] - (space.spin / g.delta) * regular[1],
  ];
  const reference = referenceGeodesic(
    space,
    { ...motion.constants, ...motion },
    6,
    0.9,
    0.035,
    1e-12,
    undefined,
    bl,
  );
  const path = advanceGeodesic(createGeodesic(space, g, photon, 0), -0.035, { tolerance: 1e-12 });
  const point = orbitPoint(path.path);
  const tangent = orbitTangent(path.path);
  const end = point && kerrGeometry(space, point);
  if (path.kind !== "complete" || !end || !tangent) {
    throw new Error("Regular transported endpoint required.");
  }
  expect(end.point.radius).toBeCloseTo(reference.state[0], 8);
  const [ft, fr, ftheta, fphi] = reference.state.slice(6);
  if (ft === undefined || fr === undefined || ftheta === undefined || fphi === undefined) {
    throw new Error("Missing transported reference vector.");
  }
  const sign = chartSign(end.point.chart);
  const transported = toCartesian(space, end, [
    ft + ((sign * (2 * end.point.radius - space.charge ** 2)) / end.delta) * fr,
    fr,
    ftheta,
    fphi + ((sign * space.spin) / end.delta) * fr,
  ]);
  expect(inner(end, transported, transported)).toBeCloseTo(1, 8);
  expect(inner(end, transported, tangent)).toBeCloseTo(0, 8);
  const final = walkerPenrose(space, end, tangent, transported);
  expect(final.real).toBeCloseTo(initial.real, 8);
  expect(final.imaginary).toBeCloseTo(initial.imaginary, 8);
});

test("zero-energy free fall crosses the bifurcation sphere instead of stalling at the horizon", () => {
  const space = { spin: 0, charge: 0 };
  const g = kerrGeometry(space, { radius: 1, inclination: 0, azimuth: 0, chart: "outgoing" });
  if (!g) {
    throw new Error("Regular white-hole fixture required.");
  }
  const initial = createGeodesic(space, g, principalFrame(g).velocity, 1);
  expect(initial.constants.energy === 0).toBe(true);
  // Cycloidal Schwarzschild solution r=1+cos(eta), tau=eta+sin(eta), from eta=-pi/2 to pi/2.
  const elapsed = Math.PI + 2;
  const result = advanceGeodesic(initial, elapsed, { parameter: "proper", tolerance: 1e-12 });
  expect(result.kind).toBe("complete");
  expect(result.path.block).toEqual({ kind: "black-hole", universe: 0 });
  expect(orbitPoint(result.path)?.radius).toBeCloseTo(1, 8);
  const velocity = orbitTangent(result.path);
  expect(velocity?.[0]).toBeCloseTo(2, 8);
  expect(velocity?.[3]).toBeCloseTo(-1, 8);
  const reverse = advanceGeodesic(result.path, -elapsed, { parameter: "proper", tolerance: 1e-12 });
  expect(reverse.kind).toBe("complete");
  expect(reverse.path.block).toEqual(initial.block);
  expect(orbitPoint(reverse.path)?.radius).toBeCloseTo(1, 8);
  const rotating = { spin: 0.5, charge: 0.25 };
  const start = kerrGeometry(rotating, {
    radius: 1,
    inclination: Math.PI / 2,
    azimuth: 0,
    chart: "outgoing",
  });
  if (!start) {
    throw new Error("Regular rotating bifurcation fixture required.");
  }
  const cycle = {
    ...createGeodesic(rotating, start, observerPhoton(principalFrame(start), [1, 0, 0]), 0),
    constants: { energy: 0, angularMomentum: 0, carter: 1, massSquared: 0 },
    state: {
      radial: [1, Math.sqrt(1 - rotating.spin ** 2 - rotating.charge ** 2), 0, 0] as const,
      direction: [1, 0, 0] as const,
      angular: [0, 1, 0] as const,
    },
  };
  // E=L=0 gives r=1+d*sin(γ), θ=π/2+γ with this fixture's Mino parameter γ; BL t/φ stay fixed.
  const periodic = advanceGeodesic(cycle, 2 * Math.PI, { tolerance: 1e-12 });
  expect(periodic.kind).toBe("complete");
  expect(periodic.path.block).toEqual({ kind: "white-hole", universe: 1 });
  const regular = regularOrbit(periodic.path);
  expect(regular?.state.radial[0]).toBeCloseTo(1, 8);
  expect(regular?.state.radial[2]).toBeCloseTo(0, 8);
  expect(regular?.state.direction[0]).toBeCloseTo(1, 8);
  expect(regular?.state.direction[1]).toBeCloseTo(0, 8);
  expect(regular?.state.direction[2]).toBeCloseTo(0, 8);
});

test("charged radial free fall measures proper time and emerges through the next white-hole block", () => {
  const space = { spin: 0, charge: 0.8 };
  const g = kerrGeometry(space, { radius: 3, inclination: 1, azimuth: 0, chart: "ingoing" });
  if (!g) {
    throw new Error("Regular fixture required.");
  }
  const initial = createGeodesic(space, g, principalFrame(g).velocity, 1);
  const result = advanceGeodesic(initial, 6, { parameter: "proper", tolerance: 1e-12 });
  expect(result.kind).toBe("complete");
  expect(result.path.block).toEqual({ kind: "exterior", universe: 1, side: 1 });
  expect(result.path.state.radial[3]).toBeCloseTo(6, 10);
  const point = orbitPoint(result.path);
  const tangent = orbitTangent(result.path);
  const end = point && kerrGeometry(space, point);
  if (!end || !tangent) {
    throw new Error("Regular endpoint required.");
  }
  expect(inner(end, tangent, tangent)).toBeCloseTo(-1, 9);
  const back = advanceGeodesic(result.path, -6, { parameter: "proper", tolerance: 1e-12 });
  expect(back.kind).toBe("complete");
  expect(back.path.block).toEqual(initial.block);
  expect(orbitPoint(back.path)?.radius).toBeCloseTo(3, 8);
});

test("regular metric agrees with the Boyer–Lindquist line element under its Jacobian", () => {
  const tangent: FourVector = [0.8, -0.2, 0.13, 0.04];
  for (const space of [
    { spin: 0.7, charge: 0.2 },
    { spin: 1.2, charge: -0.6 },
  ]) {
    for (const radius of [6, 1, 0.1, 0, -2]) {
      for (const chart of ["ingoing", "outgoing"] satisfies KerrChart[]) {
        const inclination = 0.8;
        const g = kerrGeometry(space, { radius, inclination, azimuth: 0.7, chart });
        if (!g) {
          throw new Error("Expected a regular chart point.");
        }
        const a = space.spin;
        const sine2 = Math.sin(inclination) ** 2;
        const [t, r, theta, phi] = tangent;
        const expected =
          (-g.delta * (t - a * sine2 * phi) ** 2 +
            sine2 * ((radius ** 2 + a ** 2) * phi - a * t) ** 2) /
            g.sigma +
          g.sigma * ((r * r) / g.delta + theta * theta);
        const sign = chart === "ingoing" ? 1 : -1;
        const regular: FourVector = [
          t + ((sign * (2 * radius - space.charge ** 2)) / g.delta) * r,
          r,
          theta,
          phi + ((sign * a) / g.delta) * r,
        ];
        const cartesian = toCartesian(space, g, regular);
        expect(inner(g, cartesian, cartesian)).toBeCloseTo(expected, 11);
        const restored = fromCartesian(space, g, cartesian);
        if (!restored) {
          throw new Error("Expected an invertible off-axis Jacobian.");
        }
        for (let i = 0; i < 4; i++) {
          expect(restored[i]).toBeCloseTo(regular[i] ?? NaN, 12);
          expect(raise(g, lower(g, cartesian))[i]).toBeCloseTo(cartesian[i] ?? NaN, 11);
        }
      }
    }
  }
});

test("boosted observers and photons retain local normalization across horizons and on the axis", () => {
  const space = { spin: 0.8, charge: 0.3 };
  const structure = horizons(space);
  if (structure.kind !== "pair") {
    throw new Error("Expected a subextremal fixture.");
  }
  for (const radius of [30, structure.outer, 1, structure.inner, 0, -3]) {
    for (const inclination of [0, 0.7, Math.PI]) {
      const g = kerrGeometry(space, { radius, inclination, azimuth: 0.3, chart: "ingoing" });
      if (!g) {
        throw new Error("Expected a regular fixture.");
      }
      const frame = boostFrame(principalFrame(g), [0.2, -0.4, 0.3]);
      if (!frame) {
        throw new Error("Expected a timelike observer.");
      }
      const tetrad = [frame.velocity, frame.radial, frame.polar, frame.azimuthal];
      for (const [i, left] of tetrad.entries()) {
        for (const [j, right] of tetrad.entries()) {
          expect(inner(g, left, right)).toBeCloseTo(i === j ? (i === 0 ? -1 : 1) : 0, 11);
        }
      }
      const photon = observerPhoton(frame, [0, 0.6, 0.8]);
      expect(inner(g, photon, photon)).toBeCloseTo(0, 11);
      expect(measuredFrequency(g, photon, frame.velocity)).toBeCloseTo(1, 11);
      const motion = motionFromTangent(space, g, photon, 0);
      expect(motion.radialVelocity ** 2).toBeCloseTo(
        motionRadialPotential(space, motion.constants, radius),
        8,
      );
    }
  }
});

test("custom coordinate velocities use the light cone, and static observers stop at the ergosurface", () => {
  const space = { spin: 0.8, charge: 0 };
  const outer = kerrGeometry(space, { radius: 10, inclination: 1, azimuth: 0, chart: "ingoing" });
  const ergo = kerrGeometry(space, {
    radius: 2,
    inclination: Math.PI / 2,
    azimuth: 0,
    chart: "ingoing",
  });
  if (!outer || !ergo) {
    throw new Error("Expected regular geometry.");
  }
  expect(staticObserver(ergo)).toBeUndefined();
  const observer = coordinateObserver(outer, [-0.15, 0.1, 0.03]);
  if (!observer) {
    throw new Error("Expected a timelike coordinate velocity.");
  }
  expect(observer.velocity.slice(1).map((x) => x / observer.velocity[0])).toEqual([
    expect.closeTo(-0.15, 12),
    expect.closeTo(0.1, 12),
    expect.closeTo(0.03, 12),
  ]);
  expect(coordinateObserver(outer, [2, 0, 0])).toBeUndefined();
});

// These regular exterior/negative-radius charts complement the explicit horizon/axis families above.
test("generated observer boosts preserve the light cone, measured energy and metric inverse", () => {
  const component = fc.double({ min: -0.4, max: 0.4, noNaN: true });
  fc.assert(
    fc.property(
      fc.double({ min: -1.2, max: 1.2, noNaN: true }),
      fc.double({ min: -0.8, max: 0.8, noNaN: true }),
      fc.oneof(
        fc.double({ min: 3, max: 30, noNaN: true }),
        fc.double({ min: -8, max: -2, noNaN: true }),
      ),
      fc.double({ min: 0, max: Math.PI, noNaN: true }),
      fc.tuple(component, component, component),
      fc.double({ min: -Math.PI, max: Math.PI, noNaN: true }),
      (spin, charge, radius, inclination, velocity, angle) => {
        const g = kerrGeometry(
          { spin, charge },
          { radius, inclination, azimuth: angle, chart: "ingoing" },
        );
        const frame = g && boostFrame(principalFrame(g), velocity);
        if (!g || !frame) {
          throw new Error("Generated regular timelike observer rejected.");
        }
        const p = observerPhoton(frame, [Math.cos(angle), Math.sin(angle), 0]);
        expect(inner(g, frame.velocity, frame.velocity)).toBeCloseTo(-1, 10);
        expect(inner(g, p, p)).toBeCloseTo(0, 10);
        expect(measuredFrequency(g, p, frame.velocity)).toBeCloseTo(1, 10);
        for (const [index, value] of raise(g, lower(g, p)).entries()) {
          expect(value).toBeCloseTo(p[index] ?? NaN, 10);
        }
      },
    ),
  );
});
