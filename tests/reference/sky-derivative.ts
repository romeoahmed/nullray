import { cross, dot } from "../../src/physics/vector.ts";
import type { Vec3 } from "../../src/physics/vector.ts";
import { referenceSky } from "./sky.ts";

/** Centered binary64 screen difference at an explicit step in physical pixels. */
export function referenceSkyGradient(
  x: number,
  y: number,
  axis: number,
  step: number,
  frame: Float32Array,
): Vec3 {
  const before = referenceSky(
    x - (axis === 0 ? step : 0),
    y - (axis === 1 ? step : 0),
    frame,
  ).direction;
  const after = referenceSky(
    x + (axis === 0 ? step : 0),
    y + (axis === 1 ? step : 0),
    frame,
  ).direction;
  return [
    (after[0] - before[0]) / (2 * step),
    (after[1] - before[1]) / (2 * step),
    (after[2] - before[2]) / (2 * step),
  ];
}

/** Independent directional differences avoid subtracting nearly parallel screen gradients for area. */
export function referenceSkyArea(
  x: number,
  y: number,
  frame: Float32Array,
  screen: readonly [Vec3, Vec3],
) {
  let component = 0;
  for (let axis = 1; axis < 3; axis++) {
    if (
      Math.hypot(screen[0][axis] ?? NaN, screen[1][axis] ?? NaN) >
      Math.hypot(screen[0][component] ?? NaN, screen[1][component] ?? NaN)
    ) {
      component = axis;
    }
  }
  const sx = screen[0][component] ?? NaN,
    sy = screen[1][component] ?? NaN;
  const norm = Math.hypot(sx, sy);
  if (!(norm > 0)) {
    throw new Error("Degenerate reference screen gradient");
  }
  const bx = sx / norm,
    by = sy / norm;
  const direction = referenceSky(x, y, frame).direction;
  const gradients = [
    [bx, by],
    [-by, bx],
  ].map(([vx = NaN, vy = NaN], axis) =>
    (axis === 0 ? [2 ** -19, 2 ** -18] : [0.001, 0.002]).map((step): Vec3 => {
      const before = referenceSky(x - step * vx, y - step * vy, frame).direction;
      const after = referenceSky(x + step * vx, y + step * vy, frame).direction;
      return [
        (after[0] - before[0]) / (2 * step),
        (after[1] - before[1]) / (2 * step),
        (after[2] - before[2]) / (2 * step),
      ];
    }),
  );
  const areas = [0, 1].map((level) => {
    const first = gradients[0]?.[level],
      second = gradients[1]?.[level];
    if (!first || !second) {
      throw new Error("Missing directional reference");
    }
    return Math.abs(dot(direction, cross(first, second)));
  });
  const fine = areas[0] ?? NaN,
    coarse = areas[1] ?? NaN;
  return { area: fine, convergence: Math.abs(fine / coarse - 1), basis: [bx, by], areas };
}
