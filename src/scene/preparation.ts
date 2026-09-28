import { initialBlock } from "../physics/atlas.ts";
import type { SpacetimeBlock } from "../physics/atlas.ts";
import { createDiskProfile } from "../physics/disk.ts";
import type { DiskProfile } from "../physics/disk.ts";
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
import type { Spacetime } from "../physics/spacetime.ts";
import type { Vec3 } from "../physics/vector.ts";
import { finite, record, vector3 } from "./decode.ts";
import type { Result } from "./decode.ts";
import type { Disk, Observer, Scene } from "./scene.ts";

/**
 * Observer-motion inputs, independent of navigation and source playback.
 *
 * @remarks
 * Custom velocity means KS coordinate `d(X,Y,Z)/dT`. Free-fall velocity is
 * measured in the launch reference tetrad in units of c; its nonnegative
 * proper-time interval is in M. The two velocity conventions are not interchangeable.
 */
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

/**
 * Prepared observer endpoint and disk profile, with no GPU resources.
 *
 * @remarks
 * Unchanged data may be shared with a previous scene. Treat its typed arrays
 * as borrowed, read-only storage even though their elements are mutable in TypeScript.
 */
export interface PreparedScene extends PreparedObserver {
  readonly disk: DiskProfile;
}

const fail = (error: string) => ({ ok: false, error }) as const;

/**
 * Decode and quantize observer-motion fields before metric validation.
 *
 * @returns A recognized input record, or `undefined` for invalid fields.
 * Timelikeness and free-fall reachability are checked during scene preparation.
 */
export function decodePhysicalObserver(input: unknown): PhysicalObserver | undefined {
  if (!record(input)) {
    return undefined;
  }
  const { kind, velocity, properTime } = input;
  if (kind === "regular" || kind === "static" || kind === "zamo") {
    return { kind };
  }
  if ((kind !== "custom" && kind !== "freefall") || !vector3(velocity)) {
    return undefined;
  }
  const [x, y, z] = velocity;
  if (!velocity.every((value) => Number.isFinite(Math.fround(value)))) {
    return undefined;
  }
  const vector: Vec3 = [Math.fround(x), Math.fround(y), Math.fround(z)];
  if (kind === "custom") {
    return { kind, velocity: vector };
  }
  if (!finite(properTime) || properTime < 0) {
    return undefined;
  }
  return { kind, velocity: vector, properTime };
}

/** Select a local frame or advance a free-fall worldline without mutating its launch data. */
function observerEndpoint(space: Spacetime, geometry: KerrGeometry, placement: Observer) {
  const observer = placement.motion;
  const local = (frame: ObserverFrame | undefined) =>
    ({
      ok: true,
      value: {
        geometry,
        frame,
        block: initialBlock(space, placement.radius, placement.chart),
        timeOffset: 0,
      },
    }) as const;
  switch (observer.kind) {
    case "regular":
      return local(principalFrame(geometry));
    case "static":
      return local(staticObserver(geometry));
    case "zamo":
      return local(zamoObserver(geometry, space.spin));
    case "custom":
      return local(coordinateObserver(geometry, observer.velocity));
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
      return {
        ok: true,
        value: {
          geometry: end,
          frame: comovingFrame(end, tangent),
          block: result.path.block,
          timeOffset: regular.state.radial[2],
        },
      } as const;
    }
    default:
      return fail("Unknown observer model.");
  }
}

/** Prepare and validate one complete endpoint; camera controls cannot supply observer velocity. */
function prepareObserver(space: Spacetime, placement: Observer): Result<PreparedObserver> {
  const launch = kerrGeometry(space, placement);
  if (
    !launch ||
    !(Math.fround(launch.sigma) > 0) ||
    !Number.isFinite(Math.fround(launch.sigma)) ||
    !Number.isFinite(Math.fround(launch.factor))
  ) {
    return fail(
      "The observer lies on a curvature singularity or outside the representable geometry.",
    );
  }
  const endpoint = observerEndpoint(space, launch, placement);
  if (!endpoint.ok) {
    return endpoint;
  }
  const { geometry, frame, block, timeOffset } = endpoint.value;
  if (
    !frame ||
    ![frame.velocity, frame.radial, frame.polar, frame.azimuthal].every((axis) =>
      axis.every((value) => Number.isFinite(Math.fround(value))),
    )
  ) {
    return fail(
      placement.motion.kind === "static" || placement.motion.kind === "zamo"
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
  switch (a.kind) {
    case "regular":
    case "static":
    case "zamo":
      return a.kind === b.kind;
    case "custom":
      return b.kind === "custom" && a.velocity.every((value, axis) => value === b.velocity[axis]);
    case "freefall":
      return (
        b.kind === "freefall" &&
        a.properTime === b.properTime &&
        a.velocity.every((value, axis) => value === b.velocity[axis])
      );
    default:
      throw new Error("Unknown observer model.", { cause: a satisfies never });
  }
}

/**
 * Prepare a coupled observer and disk, reusing unchanged data from a prior scene.
 *
 * @remarks
 * Requires structurally valid, quantized scene inputs. Reuse preserves an
 * unchanged free-fall endpoint independently of camera edits. Returned arrays
 * may alias the previous scene and must not be mutated by consumers.
 *
 * @returns A complete prepared value, or an error for unsupported observer/source data.
 */
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
