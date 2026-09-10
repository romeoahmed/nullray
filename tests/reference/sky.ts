import { photonFromLocal } from "../../src/physics/photon.ts";
import { traceVisibility } from "./visibility.ts";
import { evaluateGeodesic } from "./geodesic.ts";
import { integrateTransport } from "./transport.ts";
import { normalize } from "../../src/physics/vector.ts";
import type { Vec3 } from "../../src/physics/vector.ts";
import { equatorCrossings } from "./equator.ts";

/** Binary64 optical map evaluated from the same quantized frame as the GPU. */
export function referenceRay(x: number, y: number, frame: Float32Array) {
  const value = (index: number) => frame[index] ?? NaN;
  const space = { spin: value(8), charge: value(9) };
  const radius = value(4);
  const inclination = value(5) === Math.fround(Math.PI) ? Math.PI : value(5);
  const sx = ((x + 0.5 + value(2) - value(0) / 2) / value(1)) * value(6);
  const sy = ((y + 0.5 + value(3) - value(1) / 2) / value(1)) * value(6);
  const component = (axis: number) =>
    value(12 + axis) - sy * value(16 + axis) + sx * value(20 + axis);
  const source = normalize([component(0), component(1), component(2)]);
  const photon = photonFromLocal(space, radius, inclination, source);
  const { path, outcome } = traceVisibility(space, photon, radius, inclination, {
    inner: value(10),
    outer: value(11),
  });
  const axis = inclination === 0 || inclination === Math.PI;
  const launchAzimuth =
    value(7) +
    (axis && Math.hypot(source[1], source[2]) > 0
      ? Math.atan2(source[2], (inclination === 0 ? 1 : -1) * source[1])
      : 0);
  return { space, photon, path, outcome, launchAzimuth };
}

/** Unit sky direction and unnormalized Killing energy; rejects nonescaping or unresolved rays. */
export function referenceSky(x: number, y: number, frame: Float32Array) {
  const { space, photon, path, outcome, launchAzimuth } = referenceRay(x, y, frame);
  if (outcome.kind !== "sky") {
    throw new Error("The reference pixel must escape.");
  }
  const mapped = integrateTransport(space, photon, path, outcome.time, false, 1e-11);
  const mu = evaluateGeodesic(path, outcome.time)?.cosineTheta;
  if (mapped.kind !== "resolved" || mu === undefined) {
    throw new Error("Unresolved reference sky.");
  }
  const radial = Math.sqrt(1 - mu * mu);
  const direction: Vec3 = [
    radial * Math.cos(launchAzimuth + mapped.azimuth),
    radial * Math.sin(launchAzimuth + mapped.azimuth),
    mu,
  ];
  return { direction, energy: photon.energy };
}

/**
 * Complete physical endpoint for identical f32 camera words. Reference transport
 * must converge twice; no recorded GPU output is used as an accuracy oracle.
 */
export function referenceEndpoint(x: number, y: number, frame: Float32Array) {
  const ray = referenceRay(x, y, frame);
  const { space, photon, path, outcome, launchAzimuth } = ray;
  if (outcome.kind === "unresolved") {
    throw new Error(`Unresolved reference endpoint: ${outcome.reason}`);
  }
  if (outcome.kind === "captured") {
    return { kind: "captured" } as const;
  }
  const disk = outcome.kind === "disk";
  const coarse = integrateTransport(space, photon, path, outcome.time, disk, 2e-9, 65536);
  const fine = integrateTransport(space, photon, path, outcome.time, disk, 1e-9, 65536);
  if (
    coarse.kind !== "resolved" ||
    fine.kind !== "resolved" ||
    Math.abs(fine.azimuth - coarse.azimuth) > 1e-7 ||
    Math.abs(fine.coordinateTime - coarse.coordinateTime) >
      1e-7 * (1 + Math.abs(fine.coordinateTime))
  ) {
    throw new Error("Reference endpoint transport did not converge.");
  }
  const equator = equatorCrossings(
    space,
    photon,
    frame[5] === Math.fround(Math.PI) ? Math.PI : (frame[5] ?? NaN),
  );
  const crossings =
    equator.kind === "crossings" && outcome.time >= equator.first
      ? 1 + Math.floor((outcome.time - equator.first) / equator.spacing)
      : 0;
  const azimuth = launchAzimuth + fine.azimuth;
  if (outcome.kind === "sky") {
    const mu = evaluateGeodesic(path, outcome.time)?.cosineTheta;
    if (mu === undefined || Math.abs(mu) > 1) {
      throw new Error("Invalid reference latitude");
    }
    const radial = Math.sqrt((1 - mu) * (1 + mu));
    return {
      kind: "sky",
      direction: [radial * Math.cos(azimuth), radial * Math.sin(azimuth), mu] as Vec3,
      energy: photon.energy,
      tag: 1 + crossings,
    } as const;
  }
  const { radius: r } = outcome;
  const { spin: a, charge: q } = space;
  const orbital = Math.sqrt(r - q * q);
  const omega = orbital / (r * r + a * orbital);
  // Normalize the circular four-velocity directly against the equatorial metric,
  // independently of production's cancelled inverse-radius frequency formula.
  const tt = -(1 - (2 * r - q * q) / (r * r));
  const tp = (-a * (2 * r - q * q)) / (r * r);
  const pp = r * r + a * a + (a * a * (2 * r - q * q)) / (r * r);
  const norm = -(tt + 2 * omega * tp + omega * omega * pp);
  const frequency = Math.sqrt(norm) / (photon.energy - omega * photon.angularMomentum);
  if (!(frequency > 0)) {
    throw new Error("Invalid reference circular emitter");
  }
  return {
    kind: "disk",
    radius: r,
    azimuth,
    frequency,
    delay: fine.coordinateTime,
    tag: -3 - outcome.order,
  } as const;
}
