import { record, vector3 } from "./decode.ts";
import { cross, dot, normalize } from "../physics/vector.ts";
import type { Vec3 } from "../physics/vector.ts";

/** Camera axes in the observer's orthonormal (r, theta, phi) frame. */
export interface Camera {
  readonly forward: Vec3;
  readonly up: Vec3;
  readonly right: Vec3;
}

export const initialCamera: Camera = {
  forward: [-1, 0, 0],
  up: [0, -1, 0],
  right: [0, 0, 1],
};

/**
 * Validate forward/up hints and construct a fresh orthonormal camera basis.
 *
 * @returns Normalized axes, or `undefined` for malformed, nonfinite, zero, or
 * collinear hints. An externally supplied right axis is ignored and recomputed.
 */
export function createCamera(value: unknown): Camera | undefined {
  if (!record(value) || !vector3(value.forward) || !vector3(value.up)) {
    return undefined;
  }
  const forwardLength = Math.hypot(...value.forward);
  const upLength = Math.hypot(...value.up);
  if (
    !(forwardLength > 0) ||
    !(upLength > 0) ||
    !Number.isFinite(forwardLength) ||
    !Number.isFinite(upLength)
  ) {
    return undefined;
  }
  const forward = normalize(value.forward);
  const upHint = normalize(value.up);
  const perpendicular = cross(forward, upHint);
  if (!(Math.hypot(...perpendicular) > 0)) {
    return undefined;
  }
  const right = normalize(perpendicular);
  return { forward, right, up: normalize(cross(right, forward)) };
}

/**
 * Yaw then pitch in the current local camera frame, avoiding a fixed world-up pole.
 *
 * @param camera - Orthonormal axes in the observer tetrad.
 * @param horizontal - Finite yaw toward camera right, in radians.
 * @param vertical - Finite pitch toward negative camera up, in radians.
 * @returns New axes to commit through scene validation before tracing photons.
 */
export function turnCamera(camera: Camera, horizontal: number, vertical: number): Camera {
  const yawCosine = Math.cos(horizontal);
  const yawSine = Math.sin(horizontal);
  const pitchCosine = Math.cos(vertical);
  const pitchSine = Math.sin(vertical);
  const yaw: Vec3 = [
    camera.forward[0] * yawCosine + camera.right[0] * yawSine,
    camera.forward[1] * yawCosine + camera.right[1] * yawSine,
    camera.forward[2] * yawCosine + camera.right[2] * yawSine,
  ];
  const forward: Vec3 = [
    yaw[0] * pitchCosine - camera.up[0] * pitchSine,
    yaw[1] * pitchCosine - camera.up[1] * pitchSine,
    yaw[2] * pitchCosine - camera.up[2] * pitchSine,
  ];
  const up: Vec3 = [
    camera.up[0] * pitchCosine + yaw[0] * pitchSine,
    camera.up[1] * pitchCosine + yaw[1] * pitchSine,
    camera.up[2] * pitchCosine + yaw[2] * pitchSine,
  ];
  return { forward, up, right: normalize(cross(forward, up)) };
}

/** Resolve a vector into an orthonormal frame. */
export function components(value: Vec3, axes: readonly [Vec3, Vec3, Vec3]): Vec3 {
  return [dot(value, axes[0]), dot(value, axes[1]), dot(value, axes[2])];
}
