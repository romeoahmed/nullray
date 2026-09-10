import { cross, dot } from "../../src/physics/vector.ts";
import type { Vec3 } from "../../src/physics/vector.ts";
import { referenceSky } from "./sky.ts";

/**
 * Binary64 separated propagation and converged parameter differences for one image.
 * The supplied chart locates rays; it is not an optical or flux oracle. Callers
 * must validate that chart separately and keep it independent of GPU root results.
 */
export function referenceStellarImage(
  frame: Float32Array,
  source: Vec3,
  initial: readonly [number, number],
  project: (phase: number, logarithm: number) => readonly [number, number],
) {
  const norm = Math.hypot(...source);
  const target: Vec3 = [source[0] / norm, source[1] / norm, source[2] / norm];
  let axis = 0;
  for (let index = 1; index < 3; index++) {
    if (Math.abs(target[index] ?? NaN) > Math.abs(target[axis] ?? NaN)) {
      axis = index;
    }
  }
  const first = (axis + 1) % 3,
    second = (axis + 2) % 3;
  const chart = [
    (target[first] ?? NaN) / (target[axis] ?? NaN),
    (target[second] ?? NaN) / (target[axis] ?? NaN),
  ];
  const map = (phase: number, logarithm: number) => {
    const pixel = project(phase, logarithm);
    const sky = referenceSky(pixel[0] ?? NaN, pixel[1] ?? NaN, frame);
    return {
      phase,
      logarithm,
      pixel,
      ...sky,
      residual: [
        (sky.direction[first] ?? NaN) - (chart[0] ?? NaN) * (sky.direction[axis] ?? NaN),
        (sky.direction[second] ?? NaN) - (chart[1] ?? NaN) * (sky.direction[axis] ?? NaN),
      ],
      chord: Math.hypot(
        sky.direction[0] - target[0],
        sky.direction[1] - target[1],
        sky.direction[2] - target[2],
      ),
    };
  };
  let [phase, logarithm] = initial;
  const iterations = [];
  for (let iteration = 0; iteration < 8; iteration++) {
    const center = map(phase, logarithm);
    iterations.push(center);
    const h = 1e-4;
    const pa = map(phase + h, logarithm),
      pb = map(phase - h, logarithm);
    const la = map(phase, logarithm + h),
      lb = map(phase, logarithm - h);
    const u = pa.residual.map((value, index) => (value - (pb.residual[index] ?? NaN)) / (2 * h));
    const v = la.residual.map((value, index) => (value - (lb.residual[index] ?? NaN)) / (2 * h));
    const ux = u[0] ?? NaN,
      uy = u[1] ?? NaN,
      vx = v[0] ?? NaN,
      vy = v[1] ?? NaN;
    const determinant = ux * vy - uy * vx;
    if (!Number.isFinite(determinant) || determinant === 0) {
      throw new Error("Independent critical image Jacobian is singular");
    }
    const rx = center.residual[0] ?? NaN,
      ry = center.residual[1] ?? NaN;
    phase -= (rx * vy - ry * vx) / determinant;
    logarithm -= (ux * ry - uy * rx) / determinant;
  }
  const center = iterations.toSorted((a, b) => a.chord - b.chord)[0];
  if (!center) {
    throw new Error("Missing critical reference root");
  }
  const areas = [0.002, 0.001, 0.0005].map((h) => {
    const k = h * 10;
    const a = map(center.phase + h, center.logarithm),
      b = map(center.phase - h, center.logarithm);
    const c = map(center.phase, center.logarithm + k),
      d = map(center.phase, center.logarithm - k);
    const u: Vec3 = [
      (a.direction[0] - b.direction[0]) / (2 * h),
      (a.direction[1] - b.direction[1]) / (2 * h),
      (a.direction[2] - b.direction[2]) / (2 * h),
    ];
    const v: Vec3 = [
      (c.direction[0] - d.direction[0]) / (2 * k),
      (c.direction[1] - d.direction[1]) / (2 * k),
      (c.direction[2] - d.direction[2]) / (2 * k),
    ];
    const pu = a.pixel.map((value, index) => (value - (b.pixel[index] ?? NaN)) / (2 * h));
    const pv = c.pixel.map((value, index) => (value - (d.pixel[index] ?? NaN)) / (2 * k));
    const screenArea = Math.abs((pu[0] ?? NaN) * (pv[1] ?? NaN) - (pu[1] ?? NaN) * (pv[0] ?? NaN));
    return { h, k, jacobian: Math.abs(dot(center.direction, cross(u, v))) / screenArea };
  });
  const jacobian = areas[2]?.jacobian ?? NaN;
  return {
    center,
    iterations,
    areas,
    jacobian,
    areaConvergence: Math.abs(jacobian / (areas[1]?.jacobian ?? NaN) - 1),
  };
}
