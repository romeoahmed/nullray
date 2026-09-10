import { record } from "./decode.ts";
import type { Result } from "./decode.ts";
import {
  advanceGeodesic,
  createGeodesic,
  orbitPoint,
  orbitTangent,
  regularOrbit,
} from "../physics/geodesic.ts";
import { horizons, kerrGeometry } from "../physics/geometry.ts";
import type { KerrGeometry } from "../physics/geometry.ts";
import {
  boostFrame,
  comovingFrame,
  coordinateObserver,
  principalFrame,
  staticObserver,
  zamoObserver,
} from "../physics/observer.ts";
import type { ObserverFrame } from "../physics/observer.ts";
import { initialBlock } from "../physics/atlas.ts";
import type { SpacetimeBlock } from "../physics/atlas.ts";
import type { Spacetime } from "../physics/spacetime.ts";
import type { Vec3 } from "../physics/vector.ts";
import type { Disk, Observer, Scene } from "./scene.ts";
import { createDiskProfile } from "../physics/disk.ts";
import type { DiskProfile } from "../physics/disk.ts";

/** Physical observer choice; free fall starts with a measured local velocity and uses proper time. */
export type PhysicalObserver =
  | { readonly kind: "regular" | "static" | "zamo" }
  | { readonly kind: "custom"; readonly velocity: Vec3 }
  | { readonly kind: "freefall"; readonly velocity: Vec3; readonly properTime: number };

/** Finite endpoint of a physical observer, independent of the camera and emitting sources. */
interface PreparedObserver {
  readonly geometry: KerrGeometry;
  readonly frame: ObserverFrame;
  readonly block: SpacetimeBlock;
  /** Change in null-chart time along the observer worldline, including coordinate transitions. */
  readonly timeOffset: number;
}

/** Derived scene data; source inputs and ownership remain outside the GPU layer. */
export interface PreparedScene extends PreparedObserver {
  readonly disk: DiskProfile;
}

const fail = (error: string) => ({ ok: false, error }) as const;

/** Decode unknown view inputs without trusting a cast or an object brand. */
export function decodePhysicalObserver(input: unknown): PhysicalObserver | undefined {
  if (!record(input)) {
    return undefined;
  }
  const { kind, velocity, properTime } = input;
  if (kind === "regular" || kind === "static" || kind === "zamo") {
    return { kind };
  }
  if (
    (kind !== "custom" && kind !== "freefall") ||
    !Array.isArray(velocity) ||
    velocity.length !== 3
  ) {
    return undefined;
  }
  const values: readonly unknown[] = velocity;
  const [x, y, z] = values;
  if (
    typeof x !== "number" ||
    typeof y !== "number" ||
    typeof z !== "number" ||
    ![x, y, z].every((value) => Number.isFinite(Math.fround(value)))
  ) {
    return undefined;
  }
  const vector: Vec3 = [Math.fround(x), Math.fround(y), Math.fround(z)];
  if (kind === "custom") {
    return { kind, velocity: vector };
  }
  if (typeof properTime !== "number" || !Number.isFinite(properTime) || properTime < 0) {
    return undefined;
  }
  return { kind, velocity: vector, properTime };
}

/** Prepare one physical worldline; navigation and field of view cannot supply observer velocity. */
function prepareObserver(space: Spacetime, placement: Observer): Result<PreparedObserver> {
  let geometry = kerrGeometry(space, placement);
  if (
    !geometry ||
    !(Math.fround(geometry.sigma) > 0) ||
    !Number.isFinite(Math.fround(geometry.sigma)) ||
    !Number.isFinite(Math.fround(geometry.factor))
  ) {
    return fail(
      "The observer lies on a curvature singularity or outside the representable geometry.",
    );
  }
  let block = initialBlock(space, placement.radius, placement.chart);
  let frame: ObserverFrame | undefined;
  let timeOffset = 0;
  const observer = placement.motion;
  switch (observer.kind) {
    case "regular":
      frame = principalFrame(geometry);
      break;
    case "static":
      frame = staticObserver(geometry);
      break;
    case "zamo":
      frame = zamoObserver(geometry, space.spin);
      break;
    case "custom":
      frame = coordinateObserver(geometry, observer.velocity);
      break;
    case "freefall": {
      const initial = boostFrame(principalFrame(geometry), observer.velocity);
      if (!initial) {
        return fail("The launch velocity must have magnitude below the speed of light.");
      }
      const result = advanceGeodesic(
        createGeodesic(space, geometry, initial.velocity, 1),
        observer.properTime,
        { parameter: "proper" },
      );
      if (result.kind !== "complete") {
        return fail(
          result.kind === "singularity"
            ? "The free-falling observer reaches a curvature singularity before this proper time."
            : "The requested worldline is numerically unresolved.",
        );
      }
      const regular = regularOrbit(result.path);
      const point = regular && orbitPoint(regular);
      const tangent = regular && orbitTangent(regular);
      const end = point && kerrGeometry(space, point);
      if (!end || !tangent || !regular) {
        return fail("The worldline has no regular finite endpoint.");
      }
      geometry = end;
      frame = comovingFrame(geometry, tangent);
      block = result.path.block;
      timeOffset = regular.state.radial[2];
      break;
    }
    default:
      return fail("Unknown observer model.");
  }
  if (
    !frame ||
    ![frame.velocity, frame.radial, frame.polar, frame.azimuthal]
      .flat()
      .every((value) => Number.isFinite(Math.fround(value)))
  ) {
    return fail(
      observer.kind === "static" || observer.kind === "zamo"
        ? "This stationary observer requires a timelike circular worldline."
        : "This coordinate velocity is outside the observer's light cone.",
    );
  }
  if (
    !Number.isFinite(Math.fround(timeOffset)) ||
    !(Math.fround(geometry.sigma) > 0) ||
    !Number.isFinite(Math.fround(geometry.sigma)) ||
    !Number.isFinite(Math.fround(geometry.factor))
  ) {
    return fail("The observer endpoint exceeds the representable geometry or clock.");
  }
  return { ok: true, value: { geometry, frame, block, timeOffset } };
}

/** Compare only inputs that change the worldline; camera axes and field of view are independent. */
function sameObserver(left: Observer, right: Observer): boolean {
  if (
    left.radius !== right.radius ||
    left.inclination !== right.inclination ||
    left.azimuth !== right.azimuth ||
    left.chart !== right.chart
  ) {
    return false;
  }
  const a = left.motion,
    b = right.motion;
  if (a.kind !== b.kind) {
    return false;
  }
  if (a.kind !== "custom" && a.kind !== "freefall") {
    return true;
  }
  if (b.kind !== "custom" && b.kind !== "freefall") {
    return false;
  }
  return (
    a.velocity.every((value, axis) => value === b.velocity[axis]) &&
    (a.kind !== "freefall" || (b.kind === "freefall" && a.properTime === b.properTime))
  );
}

/** Atomically prepare observer and sources, reusing unchanged physical inputs from a prior scene. */
export function prepareScene(
  space: Spacetime,
  placement: Observer,
  source: Disk,
  previous?: Scene,
): Result<PreparedScene> {
  const reusable =
    previous?.space.spin === space.spin && previous.space.charge === space.charge
      ? previous
      : undefined;
  const observer =
    reusable && sameObserver(placement, reusable.observer)
      ? { ok: true as const, value: reusable.prepared }
      : prepareObserver(space, placement);
  if (!observer.ok) {
    return observer;
  }
  const { geometry } = observer.value;
  const structure = horizons(space);
  const outer =
    structure.kind === "pair" || structure.kind === "single"
      ? structure.outer
      : structure.kind === "extremal"
        ? structure.radius
        : 0;
  if (!(source.inner > outer)) {
    return fail("The emitting annulus must lie in the positive-radius exterior.");
  }
  const { radius, inclination } = geometry.point;
  if (
    Math.fround(inclination) === Math.fround(Math.PI / 2) &&
    radius >= source.inner &&
    radius <= source.outer
  ) {
    return fail("The observer cannot lie on the emitting disk surface.");
  }
  try {
    const disk =
      reusable?.disk.inner === source.inner && reusable.disk.outer === source.outer
        ? reusable.prepared.disk
        : createDiskProfile(space, source.inner, source.outer);
    return { ok: true, value: { ...observer.value, disk } };
  } catch (error) {
    return fail(
      error instanceof Error
        ? error.message
        : "The emitting annulus is outside the circular-orbit domain.",
    );
  }
}
