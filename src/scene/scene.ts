import { createCamera, initialCamera } from "./camera.ts";
import type { Camera } from "./camera.ts";
import { isco, outerHorizon } from "../physics/spacetime.ts";
import type { Spacetime } from "../physics/spacetime.ts";

/** Stationary ZAMO placement and perspective; no physical observer velocity is implied. */
export interface Observer {
  /** Boyer-Lindquist radius in gravitational radii. */
  readonly radius: number;
  /** Colatitude from the north spin axis, in radians. */
  readonly inclination: number;
  /** Boyer–Lindquist longitude in radians; construction wraps it into [0, 2 pi). */
  readonly azimuth: number;
  /** Vertical field of view, in radians. */
  readonly fieldOfView: number;
}

/** Typed physical inputs; URL and storage decoders validate unknown structures before construction. */
export interface SceneInput {
  readonly space: Spacetime;
  readonly observer: Observer;
  readonly camera?: Pick<Camera, "forward" | "up">;
}

const validated = Symbol("Scene");

/** A coherent optical snapshot; construct through createScene. */
export interface Scene extends SceneInput {
  readonly camera: Camera;
  readonly [validated]: true;
  readonly diskInner: number;
  readonly diskOuter: number;
}

/** Atomic scene construction: errors never expose a partially prepared geometry. */
export type SceneResult =
  | { readonly ok: true; readonly value: Scene }
  | { readonly ok: false; readonly error: string };

/**
 * Validate physical inputs and their GPU representation before deriving emitter geometry.
 * @param input - Physical inputs with optional camera axes; their values are validated here.
 * @param previous - Validated scene whose ISCO can be reused when spin and charge are unchanged.
 * @returns A fresh coherent scene, or an error preserving both inputs unchanged.
 */
export function createScene(input: SceneInput, previous?: Scene): SceneResult {
  const space = { spin: Math.fround(input.space.spin), charge: Math.fround(input.space.charge) };
  const observer = { ...input.observer, radius: Math.fround(input.observer.radius) };
  if (![...Object.values(space), ...Object.values(observer)].every(Number.isFinite)) {
    return { ok: false, error: "Scene parameters must be finite numbers." };
  }
  if (
    input.space.spin ** 2 + input.space.charge ** 2 >= 1 ||
    space.spin ** 2 + space.charge ** 2 >= 1
  ) {
    return { ok: false, error: "Spin and charge must satisfy a² + q² < 1." };
  }
  if (observer.radius <= outerHorizon(space)) {
    return { ok: false, error: "The observer must remain outside the event horizon." };
  }
  if (observer.inclination < 0 || observer.inclination > Math.PI) {
    return { ok: false, error: "Inclination must lie between the north and south poles." };
  }
  if (!(Math.fround(observer.fieldOfView) > 0 && Math.fround(observer.fieldOfView) < Math.PI)) {
    return { ok: false, error: "Field of view must lie between 0 and 180 degrees." };
  }
  const camera = createCamera(input.camera ?? initialCamera);
  if (!camera) {
    return { ok: false, error: "Camera axes must be finite, nonzero, and independent." };
  }
  // Camera updates reuse the same circular-emitter problem; no global mutable cache is needed.
  const minimumInner =
    previous && previous.space.spin === space.spin && previous.space.charge === space.charge
      ? previous.diskInner
      : isco(space, 1);
  let diskInner = Math.fround(minimumInner);
  if (diskInner < minimumInner) {
    // Quantize toward the stable side of the ISCO, never into the plunging region.
    const bits = new DataView(new ArrayBuffer(4));
    bits.setFloat32(0, diskInner);
    bits.setUint32(0, bits.getUint32(0) + 1);
    diskInner = bits.getFloat32(0);
  }
  const diskOuter = 64;
  if (
    Math.fround(observer.inclination) === Math.fround(Math.PI / 2) &&
    observer.radius >= diskInner &&
    observer.radius <= diskOuter
  ) {
    return { ok: false, error: "The observer cannot lie on the emitting disk surface." };
  }
  return {
    ok: true,
    value: {
      [validated]: true,
      space,
      camera,
      observer: {
        ...observer,
        azimuth: ((observer.azimuth % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI),
      },
      diskInner,
      diskOuter,
    },
  };
}

const initial = createScene({
  space: { spin: 0.7, charge: 0.2 },
  observer: { radius: 30, inclination: 1.2, azimuth: 0, fieldOfView: Math.PI / 3 },
});
if (!initial.ok) {
  throw new Error(initial.error);
}
export const initialScene: Scene = initial.value;
