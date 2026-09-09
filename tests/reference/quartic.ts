import { carlsonRF, jacobi } from "./elliptic.ts";

/** Coefficients of c0 + c1 x + c2 x² + c3 x³ + c4 x⁴. */
export type Quartic = readonly [number, number, number, number, number];

interface Weierstrass {
  readonly g2: number;
  readonly g3: number;
  readonly realRoot: number;
  readonly scale: number;
  readonly parameter: number;
  readonly threeReal: boolean;
}

export interface QuarticPath {
  readonly initial: number;
  readonly velocity: number;
  readonly f: number;
  readonly first: number;
  readonly second: number;
  readonly third: number;
  readonly fourth: number;
  readonly elliptic: Weierstrass;
}

/**
 * Prepare the Biermann-Weierstrass solution of x'² = f(x), x'' = f'(x)/2.
 * The initial velocity retains direction through simple turning points.
 * See Cieślik, Hackmann & Mach (2023), arXiv:2305.07771, Section II D.
 */
export function prepareQuartic(
  coefficients: Quartic,
  initial: number,
  velocity: number,
): QuarticPath {
  const [c0, c1, c2, c3, c4] = coefficients;
  const x = initial;
  const f = (((c4 * x + c3) * x + c2) * x + c1) * x + c0;
  const g2 = c4 * c0 - (c3 * c1) / 4 + (c2 * c2) / 12;
  const g3 =
    (c4 * c2 * c0) / 6 +
    (c3 * c2 * c1) / 48 -
    c2 ** 3 / 216 -
    (c4 * c1 * c1) / 16 -
    (c3 * c3 * c0) / 16;
  return {
    initial,
    velocity,
    f,
    first: ((4 * c4 * x + 3 * c3) * x + 2 * c2) * x + c1,
    second: (12 * c4 * x + 6 * c3) * x + 2 * c2,
    third: 24 * c4 * x + 6 * c3,
    fourth: 24 * c4,
    elliptic: prepareWeierstrass(g2, g3),
  };
}

function prepareWeierstrass(g2: number, g3: number): Weierstrass {
  const discriminant = g2 ** 3 - 27 * g3 * g3;
  if (g2 > 0 && discriminant >= 0) {
    const amplitude = Math.sqrt(g2 / 3);
    // The exact cosine lies in [-1,1] in this discriminant branch.
    const cosine = Math.max(-1, Math.min(1, (3 * Math.sqrt(3) * g3) / g2 ** 1.5));
    const angle = Math.acos(cosine) / 3;
    const e1 = amplitude * Math.cos(angle);
    const e3 = amplitude * Math.cos(angle + (2 * Math.PI) / 3);
    const e2 = -e1 - e3;
    return {
      g2,
      g3,
      realRoot: e3,
      scale: e1 - e3,
      parameter: (e2 - e3) / (e1 - e3),
      threeReal: true,
    };
  }
  const rootDisc = Math.sqrt(Math.max(0, (g3 * g3) / 64 - g2 ** 3 / 1728));
  const dominant = Math.cbrt(g3 / 8 + (g3 < 0 ? -rootDisc : rootDisc));
  const e = dominant === 0 ? 0 : dominant + g2 / (12 * dominant);
  const scale = Math.sqrt(Math.max(0, 3 * e * e - g2 / 4));
  return {
    g2,
    g3,
    realRoot: e,
    scale,
    parameter: scale === 0 ? 0 : 0.5 - (3 * e) / (4 * scale),
    threeReal: false,
  };
}

/** Return t² wp(t) and t³ wp'(t); this scaling removes the pole at t = 0. */
function scaledWeierstrass(prepared: Weierstrass, t: number): readonly [number, number] {
  const { g2, g3, realRoot: e, scale, parameter, threeReal } = prepared;
  const t2 = t * t;
  const g2t4 = g2 * t2 * t2;
  const g3t6 = g3 * t2 ** 3;
  if (Math.abs(g2t4) + Math.abs(g3t6) < 1e-8) {
    return [1 + g2t4 / 20 + g3t6 / 28, -2 + g2t4 / 10 + g3t6 / 7];
  }
  const rootScale = Math.sqrt(scale);
  if (threeReal) {
    const { sn, cn, dn } = jacobi(rootScale * t, parameter);
    return [(e + scale / (sn * sn)) * t2, ((-2 * rootScale ** 3 * cn * dn) / sn ** 3) * t2 * t];
  }
  const { sn, cn, dn } = jacobi(rootScale * t, parameter);
  const quotient = cn / (sn * dn);
  return [
    (e + scale * quotient * quotient) * t2,
    ((-2 * rootScale ** 3 * cn * (1 - parameter + parameter * cn ** 4)) / (sn * dn) ** 3) * t2 * t,
  ];
}

/**
 * Evaluate a prepared real trajectory. Undefined denotes a singular rational
 * representation, not a physical endpoint. Callers must resolve event/domain
 * boundaries before using a continuation beyond the current exterior segment.
 */
export function evaluateQuartic(path: QuarticPath, parameter: number): number | undefined {
  if (parameter === 0) {
    return path.initial;
  }
  const [w, derivative] = scaledWeierstrass(path.elliptic, parameter);
  const t2 = parameter * parameter;
  const b = w - (path.second * t2) / 24;
  const denominator = 2 * b * b - (path.f * path.fourth * t2 * t2) / 48;
  if (denominator === 0) {
    return undefined;
  }
  const numerator =
    -path.velocity * derivative * parameter +
    (path.first * b * t2) / 2 +
    (path.f * path.third * t2 * t2) / 24;
  const result = path.initial + numerator / denominator;
  return Number.isFinite(result) ? result : undefined;
}

/**
 * Invert a real quartic path at a specified point and velocity, returning its first
 * positive candidate phase. Undefined denotes an unreachable or unresolved point.
 * The Abel addition identity is derived in docs/numerics.md. Callers still need
 * endpoint residual checks near repeated roots and rational singularities.
 */
export function quarticArrival(
  path: QuarticPath,
  target: number,
  targetVelocity: number,
): number | undefined {
  const delta = target - path.initial;
  if (delta === 0) {
    return undefined;
  }
  const combined = path.f + path.velocity * targetVelocity + (path.first * delta) / 2;
  const wp = combined / (2 * delta * delta) + path.second / 24;
  const targetFirst =
    path.first +
    path.second * delta +
    (path.third * delta * delta) / 2 +
    (path.fourth * delta ** 3) / 6;
  const derivative =
    ((path.velocity * targetFirst + path.first * targetVelocity) * delta -
      4 * combined * targetVelocity) /
    (4 * delta ** 3);
  const { realRoot, scale, parameter: m, threeReal } = path.elliptic;
  if (!Number.isFinite(wp) || !Number.isFinite(derivative)) {
    return undefined;
  }
  if (scale === 0) {
    return wp > 0 && derivative < 0 ? 1 / Math.sqrt(wp) : undefined;
  }
  let sine2: number;
  const minimum = threeReal ? realRoot + scale : realRoot;
  const uncertainty =
    32 *
    Number.EPSILON *
    (Math.abs(combined / (2 * delta * delta)) + Math.abs(path.second / 24) + Math.abs(minimum));
  let time: number;
  if (Math.abs(wp - minimum) <= uncertainty && m < 1) {
    const curvature = 6 * minimum * minimum - path.elliptic.g2 / 2;
    if (!(curvature > 0)) {
      return undefined;
    }
    time = carlsonRF(0, 1 - m, 1) / Math.sqrt(scale) + derivative / curvature;
  } else {
    if (threeReal) {
      sine2 = scale / (wp - realRoot);
    } else {
      const difference = wp - realRoot;
      if (difference < 0) {
        return undefined;
      }
      const discriminant = Math.hypot(
        difference - scale,
        2 * Math.sqrt(scale) * Math.sqrt(difference) * Math.sqrt(1 - m),
      );
      sine2 = (2 * scale) / (scale + difference + discriminant);
    }
    if (!(sine2 > 0 && sine2 <= 1) || !(m <= 1)) {
      return undefined;
    }
    const rootScale = Math.sqrt(scale);
    if (1 - m * sine2 <= 0) {
      return undefined;
    }
    const first = (Math.sqrt(sine2) * carlsonRF(1 - sine2, 1 - m * sine2, 1)) / rootScale;
    if (derivative <= 0) {
      time = first;
    } else {
      if (m === 1) {
        return undefined;
      }
      time = (2 * carlsonRF(0, 1 - m, 1)) / rootScale - first;
    }
  }
  if (targetVelocity === 0) {
    return undefined;
  }
  for (let iteration = 0; iteration < 4; iteration++) {
    const value = evaluateQuartic(path, time);
    if (value === undefined) {
      return undefined;
    }
    time -= (value - target) / targetVelocity;
    if (!(time > 0) || !Number.isFinite(time)) {
      return undefined;
    }
  }
  const residual = evaluateQuartic(path, time);
  return residual !== undefined && Math.abs(residual - target) < 1e-10 ? time : undefined;
}
