import type { Spacetime } from "../../src/physics/spacetime.ts";

/** Future Killing data and separated launch velocities supplied to the independent metric oracle. */
interface ReferencePhoton {
  readonly energy: number;
  readonly angularMomentum: number;
  readonly radialVelocity: number;
  readonly polarVelocity: number;
}

/** Value and forward-mode r/θ derivatives; analytic differentiation still incurs binary64 rounding. */
type Dual = readonly [number, number, number];
interface RefractiveProfile {
  readonly amplitude: number;
  readonly scaleSquared: number;
}
const constant = (x: number): Dual => [x, 0, 0];
const add = (x: Dual, y: Dual): Dual => [x[0] + y[0], x[1] + y[1], x[2] + y[2]];
const scale = (x: Dual, k: number): Dual => [x[0] * k, x[1] * k, x[2] * k];
const mul = (x: Dual, y: Dual): Dual => [
  x[0] * y[0],
  x[1] * y[0] + x[0] * y[1],
  x[2] * y[0] + x[0] * y[2],
];
const inverse = (x: Dual): Dual => [1 / x[0], -x[1] / x[0] ** 2, -x[2] / x[0] ** 2];

/** Backward path and parallel-vector state; momenta retain their future-directed convention. */
export type ReferenceState = readonly [
  r: number,
  theta: number,
  phi: number,
  t: number,
  pr: number,
  ptheta: number,
  ft: number,
  fr: number,
  ftheta: number,
  fphi: number,
];

/**
 * Independent metric Hamiltonian oracle in exterior Boyer-Lindquist coordinates.
 * Expand the line element, invert its t/phi block, and differentiate using dual
 * numbers. No production trajectory equations are used.
 */
function hamiltonian(
  space: Spacetime,
  energy: number,
  l: number,
  state: ReferenceState,
  plasma?: RefractiveProfile,
) {
  const a = space.spin;
  const r: Dual = [state[0], 1, 0];
  const sin: Dual = [Math.sin(state[1]), 0, Math.cos(state[1])];
  const sin2 = mul(sin, sin);
  const r2a2 = add(mul(r, r), constant(a * a));
  const sigma = add(r2a2, scale(sin2, -a * a));
  const delta = add(add(r2a2, scale(r, -2)), constant(space.charge ** 2));
  const invSigma = inverse(sigma);
  const tt = mul(add(scale(delta, -1), scale(sin2, a * a)), invSigma);
  const tp = mul(mul(scale(add(delta, scale(r2a2, -1)), a), sin2), invSigma);
  const pp = mul(mul(add(mul(r2a2, r2a2), scale(mul(delta, sin2), -a * a)), sin2), invSigma);
  const determinant = add(mul(tt, pp), scale(mul(tp, tp), -1));
  const invDet = inverse(determinant);
  const itt = mul(pp, invDet);
  const itp = scale(mul(tp, invDet), -1);
  const ipp = mul(tt, invDet);
  const irr = mul(delta, invSigma);
  const terms = [
    scale(itt, energy ** 2),
    scale(itp, -2 * energy * l),
    scale(ipp, l * l),
    scale(irr, state[4] ** 2),
    scale(invSigma, state[5] ** 2),
  ];
  if (plasma) {
    const r2 = mul(r, r);
    terms.push(
      scale(
        mul(mul(r2, inverse(add(r2, constant(plasma.scaleSquared)))), invSigma),
        plasma.amplitude,
      ),
    );
  }
  const h = scale(terms.reduce(add, constant(0)), 0.5);
  const s = sigma[0];
  const velocity = [
    -s * (-itt[0] * energy + itp[0] * l),
    -s * irr[0] * state[4],
    -s * invSigma[0] * state[5],
    -s * (-itp[0] * energy + ipp[0] * l),
  ];
  const polarization = state.slice(6);
  const transport = [0, 0, 0, 0];
  if (polarization.some((value) => value !== 0)) {
    const zero = constant(0);
    const metric = [
      tt,
      zero,
      zero,
      tp,
      zero,
      inverse(irr),
      zero,
      zero,
      zero,
      zero,
      sigma,
      zero,
      tp,
      zero,
      zero,
      pp,
    ];
    const inverseMetric = [
      itt,
      zero,
      zero,
      itp,
      zero,
      irr,
      zero,
      zero,
      zero,
      zero,
      invSigma,
      zero,
      itp,
      zero,
      zero,
      ipp,
    ];
    const partial = (row: number, column: number, coordinate: number) =>
      coordinate === 1 || coordinate === 2 ? (metric[row * 4 + column]?.[coordinate] ?? 0) : 0;
    // Levi-Civita transport from differentiated metric components, independent of Killing tensors.
    for (let mu = 0; mu < 4; mu++) {
      let rate = 0;
      for (let nu = 0; nu < 4; nu++) {
        for (let rho = 0; rho < 4; rho++) {
          for (let sigmaIndex = 0; sigmaIndex < 4; sigmaIndex++) {
            rate -=
              0.5 *
              (inverseMetric[mu * 4 + sigmaIndex]?.[0] ?? 0) *
              (partial(sigmaIndex, rho, nu) +
                partial(sigmaIndex, nu, rho) -
                partial(nu, rho, sigmaIndex)) *
              (velocity[nu] ?? 0) *
              (polarization[rho] ?? 0);
          }
        }
      }
      transport[mu] = rate;
    }
  }
  const derivative: ReferenceState = [
    -s * irr[0] * state[4],
    -s * invSigma[0] * state[5],
    -s * (-itp[0] * energy + ipp[0] * l),
    -s * (-itt[0] * energy + itp[0] * l),
    s * h[1],
    s * h[2],
    transport[0] ?? 0,
    transport[1] ?? 0,
    transport[2] ?? 0,
    transport[3] ?? 0,
  ];
  return {
    derivative,
    residual: Math.abs(2 * h[0]) / terms.reduce((sum, term) => sum + Math.abs(term[0]), 0),
  };
}

function offset(y: ReferenceState, d: ReferenceState, h: number): ReferenceState {
  return [
    y[0] + h * d[0],
    y[1] + h * d[1],
    y[2] + h * d[2],
    y[3] + h * d[3],
    y[4] + h * d[4],
    y[5] + h * d[5],
    y[6] + h * d[6],
    y[7] + h * d[7],
    y[8] + h * d[8],
    y[9] + h * d[9],
  ];
}

/**
 * Integrate a backward exterior reference path with step-doubled RK4.
 *
 * @remarks
 * Uses an independently differentiated BL metric, away from coordinate poles.
 * Momentum remains future-directed. A residual and tighter-tolerance comparison
 * provide complementary evidence, not a certified global error bound.
 *
 * @param r - Initial positive exterior BL radius in M.
 * @param theta - Initial polar angle strictly between 0 and π radians.
 * @param end - Nonnegative magnitude of the backward Mino interval.
 * @param polarization - Optional initial BL tangent components `(fᵗ,fʳ,fθ,fφ)`.
 * @returns Endpoint state and maximum sampled Hamiltonian residual.
 * @throws RangeError - If interval, tolerance, or initial chart checks fail.
 * @throws Error - If arithmetic leaves the chart, a step stagnates, or attempts are exhausted.
 */
export function referenceGeodesic(
  space: Spacetime,
  photon: ReferencePhoton,
  r: number,
  theta: number,
  end: number,
  tolerance = 1e-11,
  plasma?: RefractiveProfile,
  polarization: readonly [number, number, number, number] = [0, 0, 0, 0],
) {
  if (
    !Number.isFinite(end) ||
    end < 0 ||
    !Number.isFinite(tolerance) ||
    tolerance <= 0 ||
    !Number.isFinite(r) ||
    !Number.isFinite(theta) ||
    theta <= 0 ||
    theta >= Math.PI
  ) {
    throw new RangeError(
      "The Hamiltonian reference requires a finite nonnegative path in its regular chart.",
    );
  }
  const delta = r * r - 2 * r + space.spin ** 2 + space.charge ** 2;
  if (!(delta > 0) || !(r > 1)) {
    throw new RangeError("The Hamiltonian reference requires an exterior observer.");
  }
  let state: ReferenceState = [
    r,
    theta,
    0,
    0,
    photon.radialVelocity / delta,
    photon.polarVelocity,
    ...polarization,
  ];
  const derivative = (y: ReferenceState) =>
    hamiltonian(space, photon.energy, photon.angularMomentum, y, plasma).derivative;
  const step = (y: ReferenceState, h: number): ReferenceState => {
    const k1 = derivative(y);
    const k2 = derivative(offset(y, k1, h / 2));
    const k3 = derivative(offset(y, k2, h / 2));
    const k4 = derivative(offset(y, k3, h));
    return offset(offset(offset(offset(y, k1, h / 6), k2, h / 3), k3, h / 3), k4, h / 6);
  };
  let time = 0;
  let h = Math.min(0.001, end);
  let maxResidual = 0;
  for (let attempt = 0; attempt < 100000 && time < end; attempt++) {
    h = Math.min(h, end - time);
    if (time + h === time) {
      throw new Error("Reference step underflow.");
    }
    const full = step(state, h);
    const half = step(step(state, h / 2), h / 2);
    let error = 0;
    for (let i = 0; i < 10; i++) {
      const x = full[i];
      const y = half[i];
      if (x === undefined || y === undefined || !Number.isFinite(x + y)) {
        throw new Error("Reference left its regular coordinate domain.");
      }
      error = Math.max(error, Math.abs(x - y) / (15 * tolerance * (1 + Math.abs(y))));
    }
    if (error <= 1) {
      state = half;
      time += h;
      maxResidual = Math.max(
        maxResidual,
        hamiltonian(space, photon.energy, photon.angularMomentum, state, plasma).residual,
      );
    }
    h *= error === 0 ? 2 : Math.min(2, Math.max(0.1, 0.9 * error ** -0.2));
  }
  if (time !== end) {
    throw new Error("Reference exhausted its integration budget.");
  }
  return { state, maxResidual };
}
