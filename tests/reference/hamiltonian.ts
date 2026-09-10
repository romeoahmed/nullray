import type { Spacetime } from "../../src/physics/spacetime.ts";
import type { Photon } from "../../src/physics/photon.ts";

/** Value and exact forward-mode derivatives with respect to r and theta. */
type Dual = readonly [number, number, number];
const constant = (x: number): Dual => [x, 0, 0];
const add = (x: Dual, y: Dual): Dual => [x[0] + y[0], x[1] + y[1], x[2] + y[2]];
const scale = (x: Dual, k: number): Dual => [x[0] * k, x[1] * k, x[2] * k];
const mul = (x: Dual, y: Dual): Dual => [
  x[0] * y[0],
  x[1] * y[0] + x[0] * y[1],
  x[2] * y[0] + x[0] * y[2],
];
const inverse = (x: Dual): Dual => [1 / x[0], -x[1] / x[0] ** 2, -x[2] / x[0] ** 2];

/** Backward path state; momenta retain their future-directed convention. */
export type ReferenceState = readonly [
  r: number,
  theta: number,
  phi: number,
  t: number,
  pr: number,
  ptheta: number,
];

/**
 * Independent metric Hamiltonian oracle in exterior Boyer-Lindquist coordinates.
 * Expand the line element, invert its t/phi block, and differentiate using dual
 * numbers. No separated potentials or elliptic production code are used.
 */
function hamiltonian(space: Spacetime, energy: number, l: number, state: ReferenceState) {
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
  const h = scale(terms.reduce(add, constant(0)), 0.5);
  const s = sigma[0];
  const derivative: ReferenceState = [
    -s * irr[0] * state[4],
    -s * invSigma[0] * state[5],
    -s * (-itp[0] * energy + ipp[0] * l),
    -s * (-itt[0] * energy + itp[0] * l),
    s * h[1],
    s * h[2],
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
  ];
}

/**
 * Step-doubled RK4 in the regular exterior chart, up to a finite Mino time.
 * Throws on invalid inputs, coordinate singularities or exhausted work; the
 * returned residual and a second tolerance run are independent accuracy evidence.
 */
export function referenceGeodesic(
  space: Spacetime,
  photon: Photon,
  r: number,
  theta: number,
  end: number,
  tolerance = 1e-11,
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
  let state: ReferenceState = [r, theta, 0, 0, photon.radialVelocity / delta, photon.polarVelocity];
  const derivative = (y: ReferenceState) =>
    hamiltonian(space, photon.energy, photon.angularMomentum, y).derivative;
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
    for (let i = 0; i < 6; i++) {
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
        hamiltonian(space, photon.energy, photon.angularMomentum, state).residual,
      );
    }
    h *= error === 0 ? 2 : Math.min(2, Math.max(0.1, 0.9 * error ** -0.2));
  }
  if (time !== end) {
    throw new Error("Reference exhausted its integration budget.");
  }
  return { state, maxResidual };
}
