import type { Result } from "./decode.ts";
import { decodeJet } from "./jet.ts";
import { decodePlasma } from "./plasma.ts";
import { heatingProfile, plasmaCutoff, plasmaProfile } from "../physics/plasma.ts";
import type { Plasma } from "../physics/plasma.ts";
import type { Jet } from "../physics/jet.ts";
import { horizons } from "../physics/geometry.ts";
import { createCamera, initialCamera } from "./camera.ts";
import type { Camera } from "./camera.ts";
import { isco } from "../physics/spacetime.ts";
import type { Spacetime } from "../physics/spacetime.ts";
import type { KerrChart } from "../physics/geometry.ts";
import { decodePhysicalObserver, prepareScene } from "./preparation.ts";
import type { PhysicalObserver, PreparedScene } from "./preparation.ts";

/** Observer placement or free-fall launch point; navigation does not supply physical velocity. */
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

const validated = Symbol("Scene");

/** Coherent geometry, observer and source preparation; construct through createScene. */
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

/** Round a nonnegative source radius toward the stable side of a marginal orbit. */
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

/** Validate and prepare one atomic physical snapshot, including naked and interior geometries. */
export function createScene(input: SceneInput, previous?: Scene): SceneResult {
  const space = { spin: Math.fround(input.space.spin), charge: Math.fround(input.space.charge) };
  const motion = decodePhysicalObserver(input.observer.motion ?? { kind: "zamo" });
  const chart = input.observer.chart ?? "ingoing";
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
  if (jet) {
    const structure = horizons(space);
    const outer =
      structure.kind === "pair" || structure.kind === "single"
        ? structure.outer
        : structure.kind === "extremal"
          ? structure.radius
          : 0;
    if (jet.inner <= outer) {
      return { ok: false, error: "The jet launch surface must lie outside the outer horizon." };
    }
  }
  const camera = createCamera(input.camera ?? initialCamera);
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
  observer: { radius: 65, inclination: 1.535, azimuth: 1.45, fieldOfView: 0.28 },
  disk: { inner: 6, outer: 14 },
});
if (!initial.ok) {
  throw new Error(initial.error);
}
export const initialScene: Scene = initial.value;
