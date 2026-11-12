import { horizonRadii } from "../physics/geometry.ts";
import type { KerrChart } from "../physics/geometry.ts";
import type { Jet } from "../physics/jet.ts";
import { heatingProfile, plasmaCutoff, plasmaProfile } from "../physics/plasma.ts";
import type { Plasma } from "../physics/plasma.ts";
import { isco } from "../physics/spacetime.ts";
import type { Spacetime } from "../physics/spacetime.ts";
import { createCamera, initialCamera } from "./camera.ts";
import type { Camera } from "./camera.ts";
import type { Result } from "./decode.ts";
import { decodeJet } from "./jet.ts";
import { decodePlasma } from "./plasma.ts";
import { decodePhysicalObserver, prepareScene } from "./preparation.ts";
import type { PhysicalObserver, PreparedScene } from "./preparation.ts";

/**
 * Placement or free-fall launch coordinates plus vertical camera field of view.
 *
 * @remarks
 * Radius is signed in M; inclination, null-chart azimuth, and field of view are
 * radians. `motion` alone supplies physical velocity. Omitted chart/motion
 * select ingoing coordinates and a circular ZAMO.
 */
export interface ObserverInput {
  readonly radius: number;
  readonly inclination: number;
  readonly azimuth: number;
  readonly fieldOfView: number;
  readonly chart?: KerrChart;
  readonly motion?: PhysicalObserver;
}

/** Complete observer inputs in units M and radians. Free fall is evaluated at the requested proper time. */
export interface Observer extends ObserverInput {
  readonly chart: KerrChart;
  readonly motion: PhysicalObserver;
}

/** Positive-radius zero-torque annulus repeated in each exterior source domain, in units M. */
export interface Disk {
  readonly inner: number;
  readonly outer: number;
}

/** Pure physical inputs. Omitted source and observer choices receive documented starting defaults. */
export interface SceneInput {
  readonly space: Spacetime;
  readonly observer: ObserverInput;
  readonly camera?: Pick<Camera, "forward" | "up">;
  readonly disk?: Disk;
  readonly jet?: Jet | null;
  readonly plasma?: Plasma | null;
}

/**
 * {@link SceneInput} whose nested records are still undecoded.
 *
 * @remarks
 * Scalars must already be numbers. {@link createScene} decodes chart, motion,
 * camera, jet and plasma itself, so typed inputs and parsed links share one path.
 */
export interface SceneSource {
  readonly space: Spacetime;
  readonly observer: Pick<ObserverInput, "radius" | "inclination" | "azimuth" | "fieldOfView"> & {
    readonly chart?: unknown;
    readonly motion?: unknown;
  };
  readonly camera?: unknown;
  readonly disk?: Disk;
  readonly jet?: unknown;
  readonly plasma?: unknown;
}

const validated = Symbol("Scene");

/**
 * Validated physical inputs and their coupled preparation.
 *
 * @remarks
 * Construct through {@link createScene}; the brand is not runtime validation.
 * Prepared buffers may be shared with an earlier scene and must remain unmodified.
 */
export interface Scene extends SceneInput {
  readonly camera: Camera;
  readonly observer: Observer;
  readonly disk: Disk;
  readonly jet: Jet | null;
  readonly plasma: Plasma | null;
  readonly prepared: PreparedScene;
  readonly [validated]: true;
}

export type SceneResult = Result<Scene>;

/** Copy the physical inputs without prepared buffers, for messages and serialization. */
export function sceneInput({ space, observer, camera, disk, jet, plasma }: Scene): SceneInput {
  return { space, observer, camera, disk, jet, plasma };
}

/** Round a nonnegative radius outward to f32 so an inner edge does not round below its input. */
function outwardFloat(value: number): number {
  const rounded = Math.fround(value);
  if (rounded >= value) {
    return rounded;
  }
  const bytes = new DataView(new ArrayBuffer(4));
  bytes.setFloat32(0, rounded);
  bytes.setUint32(0, bytes.getUint32(0) + 1);
  return bytes.getFloat32(0);
}

/**
 * Validate coupled physical inputs and prepare an atomic scene replacement.
 *
 * @remarks
 * Supports naked/interior placement within the implemented optical domain.
 * Acceptance does not certify complete ray coverage for every detector pixel.
 *
 * @param input - Typed inputs, or parsed data whose scalars are known to be numbers.
 * @param previous - Optional accepted scene for reusing unchanged observer/profile work.
 * @returns A complete scene or a domain error; neither input nor previous buffers are mutated.
 */
export function createScene(input: SceneSource, previous?: Scene): SceneResult {
  const space = { spin: Math.fround(input.space.spin), charge: Math.fround(input.space.charge) };
  // Defaults replace only omitted fields; an explicit null stays invalid.
  const { chart = "ingoing", motion: motionInput = { kind: "zamo" } } = input.observer;
  const { camera: cameraInput = initialCamera } = input;
  const motion = decodePhysicalObserver(motionInput);
  if (!motion || (chart !== "ingoing" && chart !== "outgoing")) {
    return { ok: false, error: "Invalid physical observer model or coordinate chart." };
  }
  const { radius, inclination, azimuth, fieldOfView } = input.observer;
  if (
    ![...Object.values(space), radius, inclination, azimuth, fieldOfView].every(Number.isFinite)
  ) {
    return { ok: false, error: "Scene parameters must be finite numbers." };
  }
  if (inclination < 0 || inclination > Math.PI) {
    return { ok: false, error: "Inclination must lie between the north and south poles." };
  }
  if (!(Math.fround(fieldOfView) > 0 && Math.fround(fieldOfView) < Math.PI)) {
    return { ok: false, error: "Field of view must lie between 0 and 180 degrees." };
  }
  const observer: Observer = {
    radius: Math.fround(radius),
    inclination,
    fieldOfView,
    chart,
    motion,
    azimuth: ((azimuth % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI),
  };
  const jet = decodeJet(input.jet ?? null);
  if (jet === undefined) {
    return { ok: false, error: "Invalid jet prescription." };
  }
  const plasma = decodePlasma(input.plasma ?? null);
  if (plasma === undefined) {
    return { ok: false, error: "Invalid plasma density, radius or observing frequency." };
  }
  const heat = plasma && heatingProfile(plasma);
  if (heat) {
    const omega = heat.frequency / 6;
    const a = Math.abs(space.spin);
    const timelikeBound =
      1 -
      2 / heat.inner -
      (4 * a * omega) / heat.inner -
      (heat.outer ** 2 + a * a + (2 * a * a) / heat.inner) * omega * omega;
    if (!(heat.inner > 2 && heat.outer > heat.inner && timelikeBound > 0)) {
      return {
        ok: false,
        error:
          "The heated annulus requires a larger radius so its pattern remains outside the ergoregion and subluminal.",
      };
    }
  }
  if (jet && jet.inner <= (horizonRadii(space).at(-1) ?? 0)) {
    return { ok: false, error: "The jet launch surface must lie outside the outer horizon." };
  }
  const camera = createCamera(cameraInput);
  if (!camera) {
    return { ok: false, error: "Camera axes must be finite, nonzero, and independent." };
  }
  const subextremal = space.spin ** 2 + space.charge ** 2 < 1;
  const disk = {
    inner: outwardFloat(input.disk?.inner ?? (subextremal ? isco(space, 1) : 16)),
    outer: Math.fround(input.disk?.outer ?? 32),
  };
  if (!(disk.inner > 0 && disk.outer > disk.inner) || !Number.isFinite(disk.outer)) {
    return { ok: false, error: "The emitting annulus requires finite ordered positive radii." };
  }
  const prepared = prepareScene(space, observer, disk, previous);
  if (!prepared.ok) {
    return prepared;
  }
  if (plasma && !(Math.fround(plasmaCutoff(plasmaProfile(plasma), prepared.value.geometry)) < 1)) {
    return { ok: false, error: "The observing frequency must exceed the local plasma cutoff." };
  }
  return {
    ok: true,
    value: {
      [validated]: true,
      space,
      observer,
      camera,
      disk,
      jet,
      plasma,
      prepared: prepared.value,
    },
  };
}

const initial = createScene({
  space: { spin: 0, charge: 0 },
  observer: { radius: 65, inclination: 1.51, azimuth: 1.45, fieldOfView: 0.34 },
  disk: { inner: 6, outer: 14 },
  camera: { forward: [-1, 0, 0], up: [0, -Math.cos(0.16), Math.sin(0.16)] },
});
if (!initial.ok) {
  throw new Error(initial.error);
}
export const initialScene: Scene = initial.value;
