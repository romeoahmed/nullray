import { components } from "./camera.ts";
import type { Observer, Scene, SceneInput } from "./scene.ts";
import { normalize } from "../physics/vector.ts";
import type { Vec3 } from "../physics/vector.ts";
import { outerHorizon } from "../physics/spacetime.ts";

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
 * Move in camera right/up/forward directions while preserving orientation in the placement chart.
 * @param scene - Validated stationary observer placement and camera frame.
 * @param motion - Relative right/up/forward weights; only the direction determines displacement.
 * @param distance - Signed displacement in the spherical placement chart, in gravitational radii.
 * @returns Scene inputs to validate with createScene; this chart is not a physical worldline.
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
  if (!(radius > 0) || !Number.isFinite(radius)) {
    return scene;
  }
  const next = {
    ...observer,
    radius: Math.max(outerHorizon(scene.space) + 0.05, Math.min(200, radius)),
    inclination: Math.atan2(Math.hypot(position[0], position[1]), position[2]),
    azimuth:
      position[0] === 0 && position[1] === 0
        ? observer.azimuth
        : Math.atan2(position[1], position[0]),
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
