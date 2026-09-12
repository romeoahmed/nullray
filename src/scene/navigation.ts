import { components } from "./camera.ts";
import type { Observer, Scene, SceneInput } from "./scene.ts";
import { normalize } from "../physics/vector.ts";
import type { Vec3 } from "../physics/vector.ts";

/** Spherical placement chart used for navigation, not a spatial embedding of the metric. */
export function navigationAxes(observer: Observer): readonly [Vec3, Vec3, Vec3] {
  const sine = Math.sin(observer.inclination);
  const cosine = Math.cos(observer.inclination);
  const longitudeSine = Math.sin(observer.azimuth);
  const longitudeCosine = Math.cos(observer.azimuth);
  return [
    [sine * longitudeCosine, sine * longitudeSine, cosine],
    [cosine * longitudeCosine, cosine * longitudeSine, -sine],
    [-longitudeSine, longitudeCosine, 0],
  ];
}

/** Compose a vector from components in a basis. */
export function fromComponents(value: Vec3, axes: readonly [Vec3, Vec3, Vec3]): Vec3 {
  return [
    value[0] * axes[0][0] + value[1] * axes[1][0] + value[2] * axes[2][0],
    value[0] * axes[0][1] + value[1] * axes[1][1] + value[2] * axes[2][1],
    value[0] * axes[0][2] + value[1] * axes[1][2] + value[2] * axes[2][2],
  ];
}

/**
 * Translate placement along camera axes while preserving its navigation-chart orientation.
 *
 * @remarks
 * The spherical placement chart is not a physical worldline. Radius retains
 * its sign and is capped at 200 M; a zero or nonrepresentable result leaves
 * placement unchanged. Commit the returned inputs through scene validation.
 *
 * @param scene - Validated observer placement and camera.
 * @param motion - Finite right/up/forward weights; magnitude does not scale displacement.
 * @param distance - Signed placement distance in M.
 * @returns Updated inputs, or the original scene for a no-op.
 * @throws RangeError - If distance is nonfinite or a nonzero motion cannot be normalized.
 */
export function translateCamera(scene: Scene, motion: Vec3, distance: number): SceneInput {
  if (!Number.isFinite(distance)) {
    throw new RangeError("Navigation distance must be finite.");
  }
  if (distance === 0 || motion.every((value) => value === 0)) {
    return scene;
  }
  const { observer, camera } = scene;
  const oldAxes = navigationAxes(observer);
  const local = fromComponents(normalize(motion), [camera.right, camera.up, camera.forward]);
  const delta = fromComponents(local, oldAxes);
  const radial = oldAxes[0];
  const position: Vec3 = [
    observer.radius * radial[0] + distance * delta[0],
    observer.radius * radial[1] + distance * delta[1],
    observer.radius * radial[2] + distance * delta[2],
  ];
  const radius = Math.hypot(...position);
  const side = observer.radius < 0 ? -1 : 1;
  if (!(radius > 0) || !Number.isFinite(radius)) {
    return scene;
  }
  const next = {
    ...observer,
    radius: side * Math.min(200, radius),
    inclination: Math.atan2(Math.hypot(position[0], position[1]), side * position[2]),
    azimuth:
      position[0] === 0 && position[1] === 0
        ? observer.azimuth
        : Math.atan2(side * position[1], side * position[0]),
  };
  const axes = navigationAxes(next);
  return {
    ...scene,
    observer: next,
    camera: {
      forward: components(fromComponents(camera.forward, oldAxes), axes),
      up: components(fromComponents(camera.up, oldAxes), axes),
    },
  };
}
