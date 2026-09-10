import { createCamera, initialCamera } from "../../src/scene/camera.ts";
import type { Scene } from "../../src/scene/scene.ts";
import cameraFixture from "../fixtures/camera-roundoff.json" with { type: "json" };
import stellarFixture from "../fixtures/schwarzschild-stellar-images.json" with { type: "json" };

/** Fresh f32 input words; fixtures own the physical scene rather than the reference solver. */
export function createTestFrame(values: readonly number[] = cameraFixture.frame): Float32Array {
  return new Float32Array(values);
}

/** Quantize an explicit physical scene to the production six-vec4 input layout. */
export function sceneFrame(scene: Scene, width: number, height: number): Float32Array {
  return new Float32Array([
    width,
    height,
    0,
    0,
    scene.observer.radius,
    scene.observer.inclination,
    2 * Math.tan(scene.observer.fieldOfView / 2),
    scene.observer.azimuth,
    scene.space.spin,
    scene.space.charge,
    scene.diskInner,
    scene.diskOuter,
    ...scene.camera.forward,
    0,
    ...scene.camera.up,
    0,
    ...scene.camera.right,
    0,
  ]);
}

/** Quantized polar camera for the independently generated Schwarzschild image fixture. */
export function stellarImageFrame(camera: unknown = initialCamera): Float32Array {
  const axes = createCamera(camera);
  if (!axes) {
    throw new Error("Invalid test camera");
  }
  const { width, height, radius, zoom, diskInner, diskOuter } = stellarFixture.frame;
  return new Float32Array([
    width,
    height,
    0,
    0,
    radius,
    0,
    zoom,
    0,
    0,
    0,
    diskInner,
    diskOuter,
    ...axes.forward,
    0,
    ...axes.up,
    0,
    ...axes.right,
    0,
  ]);
}
