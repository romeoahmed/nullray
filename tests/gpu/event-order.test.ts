import { expect, test } from "vitest";
import { createDiskProfile } from "../../src/physics/disk.ts";
import { createScene } from "../../src/model/scene.ts";
import { normalize, cross, dot } from "../../src/physics/vector.ts";
import type { Vec3 } from "../../src/physics/vector.ts";
import { traceDirections } from "./visibility-probe.ts";
import launchRoundoff from "../fixtures/launch-roundoff.json";
import { photonFromLocal } from "../../src/physics/photon.ts";
import { traceVisibility } from "../reference/visibility.ts";

test("critical launch remains an escaping ray under the quantized source contract", async () => {
  const { space, radius, inclination } = launchRoundoff;
  const scene = createScene({
    space,
    observer: { radius, inclination, azimuth: 0, fieldOfView: 1 },
  });
  if (!scene.ok) {
    throw new Error(scene.error);
  }
  const [x, y, z] = launchRoundoff.direction;
  if (x === undefined || y === undefined || z === undefined) {
    throw new Error("Missing critical launch direction.");
  }
  const local: Vec3 = [x, y, z];
  const photon = photonFromLocal(space, radius, inclination, local);
  const reference = traceVisibility(space, photon, radius, inclination, {
    inner: scene.value.diskInner,
    outer: scene.value.diskOuter,
  });
  expect(reference.outcome.kind).toBe("sky");
  const endpoint = await traceDirections([local], inclination, scene.value);
  expect(endpoint[3]).toBeGreaterThan(0);
});

const direction = (mu: number, phi: number): Vec3 => {
  const radial = Math.sqrt(1 - mu * mu);
  return [radial * Math.cos(phi), radial * Math.sin(phi), mu];
};

test.for([
  { spin: 0.7, charge: 0.2, radius: 18, inclination: 1.2, width: 1280, height: 720 },
  { spin: 0.7, charge: 0.2, radius: 18, inclination: 1.2, width: 128, height: 72 },
  { spin: -0.8, charge: 0.3, radius: 30, inclination: 2.2, width: 128, height: 72 },
  { spin: 0, charge: 0, radius: 30, inclination: 0, width: 128, height: 72 },
])(
  "direct event order matches scanned image at $spin / $charge / $width × $height",
  async (parameters) => {
    const scene = createScene({
      space: parameters,
      observer: { ...parameters, azimuth: 0, fieldOfView: 2 * Math.atan(0.6) },
    });
    if (!scene.ok) {
      throw new Error(scene.error);
    }
    const profile = createDiskProfile(
      scene.value.space,
      scene.value.diskInner,
      scene.value.diskOuter,
    );
    const bolometric = (r: number, g: number) => {
      const coordinate =
        (Math.log(r / scene.value.diskInner) /
          Math.log(scene.value.diskOuter / scene.value.diskInner)) *
        (profile.temperature.length - 1);
      const lower = Math.min(Math.floor(coordinate), profile.temperature.length - 2);
      const weight = coordinate - lower;
      return (
        (g *
          ((profile.temperature[lower] ?? 0) * (1 - weight) +
            (profile.temperature[lower + 1] ?? 0) * weight)) **
        4
      );
    };
    const directions: Vec3[] = [];
    const { width, height } = parameters;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        directions.push(
          normalize([
            -1,
            ((y + 0.5 - height / 2) * 1.2) / height,
            ((x + 0.5 - width / 2) * 1.2) / height,
          ]),
        );
      }
    }
    const data = await traceDirections(directions, parameters.inclination, scene.value, true);
    const failures: unknown[] = [];
    let maximumAngle = 0;
    let maximumInverseRadius = 0;
    let radiusCase: unknown;
    let maximumCriticalFluxError = 0;
    for (let index = 0; index < directions.length; index++) {
      const at = (offset: number) => data[index * 8 + offset] ?? Number.NaN;
      if (at(3) !== at(7)) {
        failures.push({
          index,
          direction: directions[index],
          direct: Array.from(data.subarray(index * 8, index * 8 + 4)),
          scanned: Array.from(data.subarray(index * 8 + 4, index * 8 + 8)),
        });
      } else if (at(3) > 0) {
        const direct = direction(at(0), at(1));
        const scanned = direction(at(4), at(5));
        maximumAngle = Math.max(
          maximumAngle,
          Math.atan2(Math.hypot(...cross(direct, scanned)), dot(direct, scanned)),
        );
      } else if (at(3) <= -3) {
        maximumAngle = Math.max(
          maximumAngle,
          Math.abs(Math.atan2(Math.sin(at(1) - at(5)), Math.cos(at(1) - at(5)))),
        );
        const error =
          Math.abs(1 / at(0) - 1 / at(4)) / (2e-6 + 2e-5 * Math.max(1 / at(0), 1 / at(4)));
        if (at(3) <= -5) {
          const directFlux = bolometric(at(0), at(2));
          const scannedFlux = bolometric(at(4), at(6));
          maximumCriticalFluxError = Math.max(
            maximumCriticalFluxError,
            Math.abs(directFlux - scannedFlux) / (1e-6 + Math.max(directFlux, scannedFlux)),
          );
        } else if (error > maximumInverseRadius) {
          maximumInverseRadius = error;
          radiusCase = {
            index,
            direction: directions[index],
            direct: Array.from(data.subarray(index * 8, index * 8 + 4)),
            scanned: Array.from(data.subarray(index * 8 + 4, index * 8 + 8)),
          };
        }
      }
    }
    expect(failures.length, JSON.stringify(failures.slice(0, 10))).toBe(0);
    expect(maximumAngle).toBeLessThan(5e-5);
    expect(maximumCriticalFluxError).toBeLessThan(0.005);
    expect(maximumInverseRadius, JSON.stringify(radiusCase)).toBeLessThan(1);
  },
);
