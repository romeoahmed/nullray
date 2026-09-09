import { photonFromLocal } from "../../src/physics/photon.ts";
import { traceVisibility } from "./visibility.ts";
import { evaluateGeodesic } from "./geodesic.ts";
import { integrateTransport } from "./transport.ts";
import { normalize } from "../../src/physics/vector.ts";
import type { Vec3 } from "../../src/physics/vector.ts";

/** Fresh quantized 96-byte frame; callers may adjust it without mutating shared reference inputs. */
export const createReferenceFrame = () =>
  new Float32Array([
    1280, 720, 0, 0, 18, 1.2, 1.2, 0, 0.7, 0.2, 3.304466962814331, 64, -1, 0, 0, 0, 0, -1, 0, 0, 0,
    0, 1, 0,
  ]);

/** Binary64 optical map evaluated from the same quantized frame as the GPU. */
export function referenceRay(x: number, y: number, frame: Float32Array = createReferenceFrame()) {
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
export function referenceSky(x: number, y: number, frame: Float32Array = createReferenceFrame()) {
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
