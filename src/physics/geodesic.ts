import {
  chartPrimitives,
  chartSign,
  crossExtremalHorizon,
  crossHorizon,
  initialBlock,
} from "./atlas.ts";
import type { SpacetimeBlock } from "./atlas.ts";
import { combine, horizons, kerrGeometry } from "./geometry.ts";
import type { FourVector, KerrChart, KerrGeometry, KerrPoint } from "./geometry.ts";
import { motionFromTangent } from "./motion.ts";
import type { MotionConstants } from "./motion.ts";
import type { Spacetime } from "./spacetime.ts";
import { cross, linear } from "./vector.ts";
import type { Vec3 } from "./vector.ts";
import type { PlasmaProfile } from "./plasma.ts";

/** Polynomial coordinate used for radial motion; neither choice changes the spacetime block. */
export type RadialChart = "radius" | "inverse";

/** Reduced data: radial coordinate, its Mino derivative, chart time (or log amplitude), proper time. */
export interface OrbitState {
  readonly radial: FourVector;
  readonly direction: Vec3;
  readonly angular: Vec3;
}

/** Corotating logarithmic Kruskal amplitude at one nondegenerate bifurcation surface. */
interface BifurcationPatch {
  readonly horizon: "outer" | "inner";
  readonly radius: number;
  readonly other: number;
  readonly kappa: number;
  readonly omega: number;
}

/** A neutral timelike/null path with a regular angular representation. */
export interface Geodesic {
  readonly space: Spacetime;
  readonly constants: MotionConstants;
  readonly chart: KerrChart;
  readonly radialChart: RadialChart;
  readonly state: OrbitState;
  readonly block: SpacetimeBlock;
  readonly plasma: PlasmaProfile | null;
  readonly bifurcation: BifurcationPatch | null;
}

/** Initialize physical motion from a Cartesian tangent; time is v or u in the selected null chart. */
export function createGeodesic(
  space: Spacetime,
  geometry: KerrGeometry,
  tangent: FourVector,
  massSquared: 0 | 1,
  time = 0,
  block = initialBlock(space, geometry.point.radius, geometry.point.chart),
  plasma: PlasmaProfile | null = null,
): Geodesic {
  if (plasma && massSquared !== 0) {
    throw new RangeError("A material ray profile cannot describe a massive observer.");
  }
  const initial = motionFromTangent(space, geometry, tangent, massSquared);
  const radius = geometry.point.radius;
  const inverse = Math.abs(radius) > 1;
  const sign = chartSign(geometry.point.chart);
  return {
    space,
    plasma,
    bifurcation: null,
    block,
    constants: initial.constants,
    chart: geometry.point.chart,
    radialChart: inverse ? "inverse" : "radius",
    state: {
      radial: [
        inverse ? 1 / radius : radius,
        inverse ? -initial.radialVelocity / (radius * radius) : initial.radialVelocity,
        time,
        0,
      ],
      direction: [
        sign * geometry.principal[1],
        sign * geometry.principal[2],
        sign * geometry.principal[3],
      ],
      angular: initial.angular,
    },
  };
}

/** Advance the atlas across the horizons crossed by one monotone radial segment. */
function continuationBlock(path: Geodesic, end: OrbitState, step: number): SpacetimeBlock {
  if (path.block.kind === "disconnected") {
    return path.block;
  }
  if (path.bifurcation) {
    if (
      path.state.radial[1] === 0 ||
      Math.sign(path.state.radial[1]) === Math.sign(end.radial[1])
    ) {
      return path.block;
    }
    const block = path.block;
    if (block.kind !== "black-hole" && block.kind !== "white-hole") {
      throw new RangeError("A zero-energy bifurcation joins two dynamical blocks.");
    }
    return {
      kind: block.kind === "black-hole" ? "white-hole" : "black-hole",
      universe:
        block.universe +
        (path.bifurcation.horizon === "outer" ? 0 : block.kind === "black-hole" ? 1 : -1),
    };
  }
  const structure = horizons(path.space);
  let crossings: { readonly radius: number; readonly horizon: "inner" | "outer" | "extremal" }[];
  switch (structure.kind) {
    case "none":
      return path.block;
    case "single":
      crossings = [{ radius: structure.outer, horizon: "outer" }];
      break;
    case "extremal":
      crossings = [{ radius: structure.radius, horizon: "extremal" }];
      break;
    case "pair":
      crossings = [
        { radius: structure.inner, horizon: "inner" },
        { radius: structure.outer, horizon: "outer" },
      ];
      break;
  }
  const before = path.state.radial[0];
  const after = end.radial[0];
  const radialSign =
    (after - before) * step * (path.radialChart === "inverse" ? -1 : 1) > 0 ? 1 : -1;
  const coordinate = (radius: number) => (path.radialChart === "inverse" ? 1 / radius : radius);
  crossings = crossings
    .filter(({ radius, horizon }) => {
      const at = coordinate(radius);
      if (before === at) {
        if (after === at) {
          return false;
        }
        const outside = path.radialChart === "inverse" ? after < at : after > at;
        const stationary = horizon === "inner" ? !outside : outside;
        return stationary !== (path.block.kind === (horizon === "inner" ? "interior" : "exterior"));
      }
      return Math.sign(before - at) !== Math.sign(after - at);
    })
    .toSorted(
      (left, right) =>
        (coordinate(left.radius) - coordinate(right.radius)) * Math.sign(after - before),
    );
  let block: SpacetimeBlock = path.block;
  for (const { radius, horizon } of crossings) {
    const p =
      path.constants.energy * (radius * radius + path.space.spin ** 2) -
      path.space.spin * path.constants.angularMomentum;
    const side = p >= 0 ? 1 : -1;
    block =
      horizon === "extremal"
        ? crossExtremalHorizon(block, radialSign, side)
        : crossHorizon(block, horizon, radialSign, side);
  }
  return block;
}

/** Rotate both angular vectors between a corotating patch and its regular null chart. */
function rotateOrbitState(state: OrbitState, angle: number, time: number): OrbitState {
  const cosine = Math.cos(angle),
    sine = Math.sin(angle);
  const rotate = (v: Vec3): Vec3 => [
    cosine * v[0] - sine * v[1],
    sine * v[0] + cosine * v[1],
    v[2],
  ];
  return {
    radial: [state.radial[0], state.radial[1], time, state.radial[3]],
    direction: rotate(state.direction),
    angular: rotate(state.angular),
  };
}

/** Recover a finite ingoing/outgoing chart after traversing a bifurcation patch; undefined on its sphere. */
export function regularOrbit(path: Geodesic): Geodesic | undefined {
  const patch = path.bifurcation;
  if (!patch) {
    return path;
  }
  const velocity = path.state.radial[1];
  if (velocity === 0) {
    return undefined;
  }
  const time =
    (path.state.radial[2] + Math.log(Math.abs(velocity))) / (chartSign(path.chart) * patch.kappa);
  return {
    ...path,
    bifurcation: null,
    state: rotateOrbitState(path.state, patch.omega * time, time),
  };
}

/** Enter the nearest regular bifurcation patch for the exact E=0, aL=0 family. */
function conditionBifurcation(path: Geodesic): Geodesic | undefined {
  const { energy, angularMomentum } = path.constants;
  if (energy !== 0 || path.space.spin * angularMomentum !== 0) {
    return undefined;
  }
  const structure = horizons(path.space);
  if (structure.kind !== "single" && structure.kind !== "pair") {
    return undefined;
  }
  const coordinate = path.state.radial[0];
  if (coordinate === 0) {
    return undefined;
  }
  const r = path.radialChart === "inverse" ? 1 / coordinate : coordinate;
  const outer = structure.kind === "single" || r >= 1;
  const radius = outer ? structure.outer : structure.inner;
  const other = outer ? (structure.kind === "single" ? 0 : structure.inner) : structure.outer;
  if (path.bifurcation?.radius === radius) {
    return path;
  }
  const regular = regularOrbit(path);
  if (!regular) {
    return undefined;
  }
  const velocity =
    regular.radialChart === "inverse" ? -regular.state.radial[1] * r * r : regular.state.radial[1];
  if (velocity === 0) {
    return undefined;
  }
  const squared = radius * radius + path.space.spin ** 2;
  const patch: BifurcationPatch = {
    horizon: outer ? "outer" : "inner",
    radius,
    other,
    kappa: (radius - 1) / squared,
    omega: path.space.spin / squared,
  };
  const time = regular.state.radial[2];
  const amplitude = chartSign(regular.chart) * patch.kappa * time - Math.log(Math.abs(velocity));
  const state = rotateOrbitState(regular.state, -patch.omega * time, amplitude);
  return {
    ...regular,
    radialChart: "radius",
    bifurcation: patch,
    state: { ...state, radial: [r, velocity, amplitude, state.radial[3]] },
  };
}

/** Resolve a finite point without introducing an azimuth singularity into the equations of motion. */
export function orbitPoint(input: Geodesic): KerrPoint | undefined {
  const path = regularOrbit(input);
  if (!path) {
    return undefined;
  }
  const coordinate = path.state.radial[0];
  if (path.radialChart === "inverse" && coordinate === 0) {
    return undefined;
  }
  const n = path.state.direction;
  return {
    radius: path.radialChart === "inverse" ? 1 / coordinate : coordinate,
    inclination: Math.atan2(Math.hypot(n[0], n[1]), n[2]),
    azimuth: Math.atan2(n[1], n[0]),
    chart: path.chart,
  };
}

/** Horizon-regular separated equations on the sphere; no division by sin(theta) or Δ. */
export function orbitDerivative(path: Geodesic, state: OrbitState): OrbitState | undefined {
  const { energy: e, angularMomentum: l, carter: c, massSquared: m } = path.constants;
  const { spin: a, charge: q } = path.space;
  const [x, velocity] = state.radial;
  const sign = chartSign(path.chart);
  const n = state.direction;
  const k = (l - a * e) ** 2 + c;
  const constant = a * (a * e - l);
  const squared = a * a + q * q;
  const quadratic = 2 * e * constant - k - m * squared;
  const central = constant * constant - squared * k;
  const amplitude = path.plasma?.amplitude ?? 0;
  const scaleSquared = path.plasma?.scaleSquared ?? 1;
  const patch = path.bifurcation;
  if (patch) {
    const r2 = x * x;
    const denominator = r2 + scaleSquared;
    const fr = (amplitude * r2) / denominator;
    const profileDerivative = (2 * amplitude * x * scaleSquared) / (denominator * denominator);
    const b = k + m * r2 + fr;
    if (!(b > 0)) {
      return undefined;
    }
    const bDerivative = 2 * m * x + profileDerivative;
    const delta = (x - patch.radius) * (x - patch.other);
    const acceleration = -(x - 1) * b - (delta * bDerivative) / 2;
    const rotation = (-patch.omega * sign * (x + patch.radius) * velocity) / (x - patch.other);
    const amplitudeRate =
      -velocity *
      ((1 - patch.kappa * (x + patch.radius)) / (x - patch.other) + bDerivative / (2 * b));
    const axial: Vec3 = [0, 0, 1];
    return {
      radial: [velocity, acceleration, amplitudeRate, m === 0 ? 0 : r2 + a * a * n[2] * n[2]],
      direction: linear(cross(state.angular, n), 1, cross(axial, n), rotation),
      angular: linear(cross(n, axial), -a * a * m * n[2], cross(axial, state.angular), rotation),
    };
  }
  let acceleration: number;
  let drag: number;
  let nullTime: number;
  let proper: number;
  if (path.radialChart === "inverse") {
    const denominator = e + constant * x * x + sign * velocity;
    if (denominator === 0 || (m !== 0 && x === 0)) {
      return undefined;
    }
    acceleration = m + quadratic * x + 3 * k * x * x + 2 * central * x * x * x;
    const profileDenominator = 1 + scaleSquared * x * x;
    const fr = amplitude / profileDenominator;
    const delta = 1 - 2 * x + squared * x * x;
    acceleration -=
      x * (1 - 3 * x + 2 * squared * x * x) * fr -
      (amplitude * scaleSquared * x ** 3 * delta) / profileDenominator ** 2;
    drag = ((k + fr) * x * x + m) / denominator;
    nullTime =
      a * (l - a * e * (1 - n[2] * n[2])) +
      ((1 + a * a * x * x) * (k + fr + (m === 0 ? 0 : m / (x * x)))) / denominator;
    proper = m === 0 ? 0 : 1 / (x * x) + a * a * n[2] * n[2];
  } else {
    const denominator = e * x * x + constant - sign * velocity;
    const delta = x * x - 2 * x + squared;
    if (denominator === 0 && delta === 0) {
      return undefined;
    }
    acceleration = 2 * (e * e - m) * x * x * x + 3 * m * x * x + quadratic * x + k;
    const profileDenominator = x * x + scaleSquared;
    const fr = (amplitude * x * x) / profileDenominator;
    acceleration -= (x - 1) * fr + (delta * amplitude * x * scaleSquared) / profileDenominator ** 2;
    // The rationalized form has 0/0 at a zero of P on a principal null ray.
    drag =
      denominator === 0
        ? (e * x * x + constant + sign * velocity) / delta
        : (k + fr + m * x * x) / denominator;
    nullTime = a * (l - a * e * (1 - n[2] * n[2])) + (x * x + a * a) * drag;
    proper = m === 0 ? 0 : x * x + a * a * n[2] * n[2];
  }
  const rotation = a * (drag - e);
  const axial: Vec3 = [0, 0, 1];
  return {
    radial: [velocity, acceleration, nullTime, proper],
    direction: linear(cross(state.angular, n), 1, cross(axial, n), rotation),
    angular: linear(
      cross(n, axial),
      a * a * (e * e - m) * n[2],
      cross(axial, state.angular),
      rotation,
    ),
  };
}

/** Reconstruct the physical Cartesian tangent at a finite regular point. */
export function orbitTangent(input: Geodesic): FourVector | undefined {
  const path = regularOrbit(input);
  if (!path) {
    return undefined;
  }
  const point = orbitPoint(path);
  const geometry = point && kerrGeometry(path.space, point);
  const derivative = orbitDerivative(path, path.state);
  if (!point || !geometry || !derivative) {
    return undefined;
  }
  const radial =
    path.radialChart === "inverse"
      ? -path.state.radial[1] * point.radius * point.radius
      : path.state.radial[1];
  const sign = chartSign(path.chart);
  const spatial = linear(
    linear(path.state.direction, radial, derivative.direction, point.radius),
    1,
    cross([0, 0, 1], derivative.direction),
    sign * path.space.spin,
  );
  return [
    (derivative.radial[2] - sign * radial) / geometry.sigma,
    spatial[0] / geometry.sigma,
    spatial[1] / geometry.sigma,
    spatial[2] / geometry.sigma,
  ];
}

/** Change regular spacetime charts in an overlap; this is not a horizon crossing. */
export function changeOrbitChart(path: Geodesic): Geodesic | undefined {
  const patch = path.bifurcation;
  if (patch) {
    const r = path.state.radial[0];
    const k =
      (path.constants.angularMomentum - path.space.spin * path.constants.energy) ** 2 +
      path.constants.carter;
    const fr = path.plasma
      ? (path.plasma.amplitude * r * r) / (r * r + path.plasma.scaleSquared)
      : 0;
    const b = k + path.constants.massSquared * r * r + fr;
    if (!(b > 0) || r === patch.other) {
      return undefined;
    }
    const logarithm = Math.log(Math.abs(r - patch.other));
    const horizonScale = patch.radius * patch.radius + path.space.spin ** 2;
    const otherScale = patch.other * patch.other + path.space.spin ** 2;
    const amplitude =
      -path.state.radial[2] +
      2 * patch.kappa * r -
      (1 + otherScale / horizonScale) * logarithm -
      Math.log(b);
    const angle = 2 * chartSign(path.chart) * patch.omega * (r + 2 * logarithm);
    return {
      ...path,
      chart: path.chart === "ingoing" ? "outgoing" : "ingoing",
      state: rotateOrbitState(path.state, angle, amplitude),
    };
  }
  const point = orbitPoint(path);
  const primitives = point && chartPrimitives(path.space, point.radius);
  if (!primitives) {
    return undefined;
  }
  const sign = chartSign(path.chart);
  const angle = -2 * sign * primitives[0];
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const rotate = (v: Vec3): Vec3 => [
    cosine * v[0] - sine * v[1],
    sine * v[0] + cosine * v[1],
    v[2],
  ];
  const [r, velocity, time, proper] = path.state.radial;
  return {
    ...path,
    chart: path.chart === "ingoing" ? "outgoing" : "ingoing",
    state: {
      radial: [r, velocity, time - 2 * sign * primitives[1], proper],
      direction: rotate(path.state.direction),
      angular: rotate(path.state.angular),
    },
  };
}

/** Choose well-conditioned coordinates at the current point, leaving physical data unchanged. */
export function conditionOrbit(path: Geodesic): Geodesic | undefined {
  const bifurcation = conditionBifurcation(path);
  if (bifurcation) {
    return bifurcation;
  }
  let current = path;
  const [x, velocity, time, proper] = current.state.radial;
  if (Math.abs(x) > 1) {
    current = {
      ...current,
      radialChart: current.radialChart === "radius" ? "inverse" : "radius",
      state: { ...current.state, radial: [1 / x, -velocity / (x * x), time, proper] },
    };
  }
  const [r, v] = current.state.radial;
  const { energy: e, angularMomentum: l } = current.constants;
  const a = current.space.spin;
  const p =
    current.radialChart === "radius" ? e * (r * r + a * a) - a * l : e + a * (a * e - l) * r * r;
  const motion = current.radialChart === "radius" ? v : -v;
  return chartSign(current.chart) * p * motion > 0 ? changeOrbitChart(current) : current;
}

/** Combine one RK stage in local buffers, avoiding intermediate vector records for every coefficient. */
function weightedState(
  base: OrbitState,
  derivatives: readonly OrbitState[],
  weights: readonly number[],
  step: number,
): OrbitState {
  const radial: [number, number, number, number] = [...base.radial];
  const direction: [number, number, number] = [...base.direction];
  const angular: [number, number, number] = [...base.angular];
  for (let i = 0; i < weights.length; i++) {
    const weight = weights[i];
    const derivative = derivatives[i];
    if (weight === undefined || !derivative) {
      throw new Error("Incomplete Runge–Kutta stage.");
    }
    if (weight === 0) {
      continue;
    }
    const scale = step * weight;
    for (const axis of [0, 1, 2] as const) {
      radial[axis] += scale * derivative.radial[axis];
      direction[axis] += scale * derivative.direction[axis];
      angular[axis] += scale * derivative.angular[axis];
    }
    radial[3] += scale * derivative.radial[3];
  }
  return { radial, direction, angular };
}

/** Dormand–Prince 5(4) pair; coefficients define the numerical method, not physical tuning. */
const stages = [
  [1 / 5],
  [3 / 40, 9 / 40],
  [44 / 45, -56 / 15, 32 / 9],
  [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
  [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
  [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
] as const;
const lowerOrder = [
  5179 / 57600,
  0,
  7571 / 16695,
  393 / 640,
  -92097 / 339200,
  187 / 2100,
  1 / 40,
] as const;

/** One embedded step; undefined means that a stage left the current regular chart. */
export function orbitStep(
  path: Geodesic,
  step: number,
  parameter: "mino" | "proper" = "mino",
): { readonly state: OrbitState; readonly error: number } | undefined {
  const evaluate = (state: OrbitState): OrbitState | undefined => {
    const derivative = orbitDerivative(path, state);
    if (!derivative || parameter === "mino") {
      return derivative;
    }
    const rate = derivative.radial[3];
    if (!(rate > 0)) {
      return undefined;
    }
    return {
      radial: combine(derivative.radial, 1 / rate, derivative.radial, 0),
      direction: linear(derivative.direction, 1 / rate, derivative.direction, 0),
      angular: linear(derivative.angular, 1 / rate, derivative.angular, 0),
    };
  };
  const first = evaluate(path.state);
  if (!first) {
    return undefined;
  }
  const derivatives = [first];
  let state = path.state;
  for (const weights of stages) {
    state = weightedState(path.state, derivatives, weights, step);
    const derivative = evaluate(state);
    if (!derivative) {
      return undefined;
    }
    derivatives.push(derivative);
  }
  const comparison = weightedState(path.state, derivatives, lowerOrder, step);
  let error = 0;
  for (const key of ["radial", "direction", "angular"] as const) {
    for (const [i, value] of state[key].entries()) {
      const other = comparison[key][i];
      if (other === undefined || !Number.isFinite(value + other)) {
        return undefined;
      }
      error = Math.max(
        error,
        Math.abs(value - other) / (key === "direction" ? 1 : 1 + Math.abs(value)),
      );
    }
  }
  return { state, error };
}

/** Continue a finite signed Mino interval, or proper time for a massive path. */
export function advanceGeodesic(
  initial: Geodesic,
  duration: number,
  options: {
    readonly tolerance?: number;
    readonly maxSteps?: number;
    readonly parameter?: "mino" | "proper";
  } = {},
): {
  readonly kind: "complete" | "unresolved" | "infinity" | "singularity";
  readonly path: Geodesic;
  readonly elapsed: number;
} {
  const { tolerance = 1e-10, maxSteps = 10000, parameter = "mino" } = options;
  if (
    !Number.isFinite(duration) ||
    !(tolerance > 0) ||
    !Number.isFinite(tolerance) ||
    !Number.isSafeInteger(maxSteps) ||
    maxSteps < 1 ||
    (parameter === "proper" && initial.constants.massSquared !== 1)
  ) {
    throw new RangeError(
      "Geodesic work requires a finite interval, positive tolerance, and positive step budget.",
    );
  }
  let path = initial;
  let elapsed = 0;
  const direction = Math.sign(duration);
  let step =
    direction * Math.min(Math.abs(duration), 0.01 / Math.max(1, Math.hypot(...path.state.angular)));
  for (let attempt = 0; attempt < maxSteps && direction * (duration - elapsed) > 0; attempt++) {
    const conditioned = conditionOrbit(path);
    if (!conditioned) {
      break;
    }
    path = conditioned;
    step = direction * Math.min(Math.abs(step), Math.abs(duration - elapsed));
    if (elapsed + step === elapsed) {
      break;
    }
    let trial = orbitStep(path, step, parameter);
    if (
      trial &&
      trial.error <= tolerance &&
      path.state.radial[1] !== 0 &&
      Math.sign(path.state.radial[1]) !== Math.sign(trial.state.radial[1])
    ) {
      // End just after the bracketed turn so subsequent boundary searches see one radial leg.
      let left = 0;
      let right = 1;
      for (let iteration = 0; iteration < 60; iteration++) {
        const middle = (left + right) / 2;
        if (middle === left || middle === right) {
          break;
        }
        const candidate = orbitStep(path, step * middle, parameter);
        if (!candidate) {
          return { kind: "unresolved", path, elapsed };
        }
        if (Math.sign(path.state.radial[1]) === Math.sign(candidate.state.radial[1])) {
          left = middle;
        } else {
          right = middle;
          trial = candidate;
        }
      }
      step *= right;
    }
    if (trial && trial.error <= tolerance) {
      const boundary =
        path.radialChart === "inverse"
          ? "infinity"
          : path.space.spin === 0 ||
              (path.state.direction[2] === 0 &&
                path.state.angular[0] === 0 &&
                path.state.angular[1] === 0)
            ? "singularity"
            : undefined;
      if (boundary && path.state.radial[0] * trial.state.radial[0] <= 0) {
        // Infinity is a conformal endpoint, not an overlap with the negative-r sheet.
        let left = 0;
        let right = step;
        let endpoint = trial.state;
        for (let iteration = 0; iteration < 60; iteration++) {
          const middle = (left + right) / 2;
          if (middle === left || middle === right) {
            break;
          }
          const candidate = orbitStep(path, middle, parameter);
          if (!candidate) {
            return { kind: "unresolved", path, elapsed };
          }
          if (path.state.radial[0] * candidate.state.radial[0] > 0) {
            left = middle;
          } else {
            right = middle;
            endpoint = candidate.state;
          }
        }
        return {
          kind: boundary,
          path: {
            ...path,
            block: continuationBlock(path, endpoint, right),
            state: {
              ...endpoint,
              radial: [0, endpoint.radial[1], endpoint.radial[2], endpoint.radial[3]],
            },
          },
          elapsed: elapsed + right,
        };
      }
      const bifurcationTurn =
        path.bifurcation &&
        path.state.radial[1] !== 0 &&
        Math.sign(path.state.radial[1]) !== Math.sign(trial.state.radial[1]);
      path = { ...path, block: continuationBlock(path, trial.state, step), state: trial.state };
      if (bifurcationTurn) {
        const changed = changeOrbitChart(path);
        if (!changed) {
          return { kind: "unresolved", path, elapsed };
        }
        path = changed;
      }
      elapsed += step;
    }
    step *= trial
      ? trial.error === 0
        ? 2
        : Math.min(2, Math.max(0.1, 0.9 * (tolerance / trial.error) ** 0.2))
      : 0.25;
  }
  return { kind: elapsed === duration ? "complete" : "unresolved", path, elapsed };
}
