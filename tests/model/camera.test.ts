import { expect, test } from "vitest";
import * as fc from "fast-check";
import { createCamera, initialCamera, turnCamera } from "../../src/model/camera.ts";
import { fromComponents, navigationAxes, translateCamera } from "../../src/model/navigation.ts";
import { createScene, initialScene } from "../../src/model/scene.ts";
import { dot, cross } from "../../src/physics/vector.ts";
import { decodeView, encodeView } from "../../src/model/view.ts";
import { initialAppearance } from "../../src/model/appearance.ts";

test("camera turns preserve handedness, orthogonality, and full-sphere orientation", () => {
  fc.assert(
    fc.property(
      fc.double({ min: -Math.PI, max: Math.PI, noNaN: true }),
      fc.double({ min: -Math.PI, max: Math.PI, noNaN: true }),
      (horizontal, vertical) => {
        const camera = turnCamera(initialCamera, horizontal, vertical);
        expect(dot(camera.forward, camera.up)).toBeCloseTo(0, 14);
        expect(dot(camera.forward, camera.right)).toBeCloseTo(0, 14);
        expect(dot(camera.up, camera.right)).toBeCloseTo(0, 14);
        expect(dot(cross(camera.forward, camera.up), camera.right)).toBeCloseTo(1, 14);
        for (const axis of [camera.forward, camera.up, camera.right]) {
          expect(Math.hypot(...axis)).toBeCloseTo(1, 14);
        }
      },
    ),
  );
  expect(turnCamera(initialCamera, Math.PI / 2, 0).forward[2]).toBeCloseTo(1, 14);
  expect(turnCamera(initialCamera, Math.PI, 0).forward[0]).toBeCloseTo(1, 14);
  expect(turnCamera(initialCamera, 0, Math.PI / 2).forward[1]).toBeCloseTo(1, 14);
});

test("camera inputs reject degenerate axes without excluding steep or polar views", () => {
  for (const value of [
    null,
    {},
    { forward: [0, 0, 0], up: [1, 0, 0] },
    { forward: [1, 0, 0], up: [-1, 0, 0] },
    { forward: [1, 0, 0], up: [0, NaN, 0] },
    { forward: [1, "0", 0], up: [0, 1, 0] },
    { forward: [1, 0], up: [0, 1, 0] },
  ]) {
    expect(createCamera(value)).toBeUndefined();
  }
  expect(createCamera({ forward: [1e308, 0, 0], up: [0, 1e308, 0] })).toBeDefined();
  expect(createCamera({ forward: [1, 0, 0], up: [1, 1e-100, 0] })).toBeDefined();
});

test("free translations preserve chart orientation across poles and longitude seams", () => {
  for (const inclination of [0, 1e-8, Math.PI / 2, Math.PI - 1e-8, Math.PI]) {
    for (const azimuth of [0, 2 * Math.PI - 1e-8]) {
      const initial = createScene({
        ...initialScene,
        // Keep equatorial chart checks outside the emitting annulus.
        observer: {
          ...initialScene.observer,
          radius: initialScene.diskOuter + 16,
          inclination,
          azimuth,
        },
        camera: turnCamera(initialCamera, 0.7, -0.4),
      });
      if (!initial.ok) {
        throw new Error(initial.error);
      }
      const moved = createScene(translateCamera(initial.value, [1, 0.5, -0.3], 0.4));
      if (!moved.ok) {
        throw new Error(moved.error);
      }
      const oldAxes = navigationAxes(initial.value.observer);
      const axes = navigationAxes(moved.value.observer);
      for (const key of ["forward", "up"] as const) {
        const before = fromComponents(initial.value.camera[key], oldAxes);
        const after = fromComponents(moved.value.camera[key], axes);
        for (let i = 0; i < 3; i++) {
          expect(after[i]).toBeCloseTo(before[i] ?? NaN, 13);
        }
      }
    }
  }
});

test("free view links retain camera orientation, navigation, and display state", () => {
  const scene = createScene({ ...initialScene, camera: turnCamera(initialCamera, 1.3, -0.7) });
  if (!scene.ok) {
    throw new Error(scene.error);
  }
  const view = {
    scene: scene.value,
    appearance: initialAppearance,
    bloom: 0.2,
    exposureEV: 1,
    time: 4,
    diagnostic: "image",
    display: "auto",
    navigation: "free",
  } as const;
  const result = decodeView(encodeView(view));
  if (result.kind !== "view") {
    throw new Error("Missing restored camera.");
  }
  expect(result.value.navigation).toBe("free");
  expect(result.value.scene.observer).toEqual(view.scene.observer);
  for (const key of ["forward", "up", "right"] as const) {
    for (let i = 0; i < 3; i++) {
      expect(result.value.scene.camera[key][i]).toBeCloseTo(view.scene.camera[key][i] ?? NaN, 14);
    }
  }
});
