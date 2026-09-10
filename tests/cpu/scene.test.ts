import { describe, expect, test } from "vitest";
import * as fc from "fast-check";
import { createScene, initialScene } from "../../src/scene/scene.ts";
import { outerHorizon } from "../../src/physics/spacetime.ts";
import { createCamera, initialCamera, turnCamera } from "../../src/scene/camera.ts";
import { fromComponents, navigationAxes, translateCamera } from "../../src/scene/navigation.ts";
import { dot, cross } from "../../src/physics/vector.ts";
import { decodeView, encodeView } from "../../src/scene/view.ts";
import { initialAppearance } from "../../src/scene/appearance.ts";
import { initialPlasma } from "../../src/scene/plasma.ts";

const equatorial = (radius: number, inclination = Math.PI / 2) =>
  createScene({
    ...initialScene,
    observer: { ...initialScene.observer, radius, inclination },
  });

describe("Coupled scene validation", () => {
  test("camera and source edits preserve a free-fall endpoint while physical time advances it", () => {
    const prepared = createScene({
      ...initialScene,
      observer: {
        ...initialScene.observer,
        motion: { kind: "freefall", velocity: [0, 0, 0], properTime: 4 },
      },
    });
    if (!prepared.ok) {
      throw new Error(prepared.error);
    }
    const scene = prepared.value;
    const edits = createScene(
      {
        ...scene,
        camera: turnCamera(scene.camera, 0.5, 0.2),
        observer: { ...scene.observer, fieldOfView: 0.8 },
        disk: { ...scene.disk, outer: scene.disk.outer * 2 },
      },
      scene,
    );
    if (!edits.ok) {
      throw new Error(edits.error);
    }
    expect(edits.value.prepared.geometry).toEqual(scene.prepared.geometry);
    expect(edits.value.prepared.frame).toEqual(scene.prepared.frame);
    expect(edits.value.prepared.timeOffset).toBe(scene.prepared.timeOffset);
    expect(edits.value.disk.outer).toBe(scene.disk.outer * 2);
    const later = createScene(
      {
        ...scene,
        observer: {
          ...scene.observer,
          motion: { kind: "freefall", velocity: [0, 0, 0], properTime: 5 },
        },
      },
      scene,
    );
    if (!later.ok) {
      throw new Error(later.error);
    }
    expect(later.value.prepared.geometry.point.radius).toBeLessThan(
      scene.prepared.geometry.point.radius,
    );
    expect(later.value.prepared.timeOffset).toBeGreaterThan(scene.prepared.timeOffset);
  });
  test("scene construction retains a valid coupled spacetime after GPU quantization", () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0, max: 0.99, noNaN: true }),
        fc.double({ min: -Math.PI, max: Math.PI, noNaN: true }),
        fc.double({ min: 2.1, max: 200, noNaN: true }),
        (magnitude, angle, radius) => {
          const result = createScene({
            camera: initialScene.camera,
            space: { spin: magnitude * Math.cos(angle), charge: magnitude * Math.sin(angle) },
            observer: { ...initialScene.observer, radius },
          });
          expect(result.ok).toBe(true);
          if (!result.ok) {
            throw new Error(result.error);
          }
          const {
            space,
            observer,
            disk: { inner: diskInner, outer: diskOuter },
          } = result.value;
          expect(space.spin ** 2 + space.charge ** 2).toBeLessThan(1);
          expect(observer.radius).toBeGreaterThan(outerHorizon(space));
          expect(diskInner).toBeGreaterThan(outerHorizon(space));
          expect(diskInner).toBeLessThan(diskOuter);
          expect(Math.fround(observer.radius)).toBe(observer.radius);
        },
      ),
    );
  });

  test("scene construction rejects invalid inputs and boundaries erased by quantization", () => {
    expect(createScene({ ...initialScene, plasma: initialPlasma }).ok).toBe(true);
    expect(
      createScene({ ...initialScene, plasma: { ...initialPlasma, frequencyGHz: 0.1 } }).ok,
    ).toBe(false);
    for (const space of [
      { spin: Infinity, charge: 0 },
      { spin: 0, charge: NaN },
      { spin: NaN, charge: 0 },
    ]) {
      expect(createScene({ ...initialScene, space }).ok).toBe(false);
    }
    for (const radius of [0, 2, 2 + 1e-9, Infinity]) {
      expect(
        createScene({
          ...initialScene,
          space: { spin: 0, charge: 0 },
          observer: { ...initialScene.observer, radius },
        }).ok,
      ).toBe(false);
    }
    for (const fieldOfView of [0, 1e-99, Math.PI, Infinity]) {
      expect(
        createScene({ ...initialScene, observer: { ...initialScene.observer, fieldOfView } }).ok,
      ).toBe(false);
    }
    for (const inclination of [0, Math.PI]) {
      expect(
        createScene({ ...initialScene, observer: { ...initialScene.observer, inclination } }).ok,
      ).toBe(true);
    }
  });

  test("scene construction excludes the emitting surface after quantization without excluding its hole", () => {
    const middle = (initialScene.disk.inner + initialScene.disk.outer) / 2;
    for (const radius of [
      initialScene.disk.inner,
      middle,
      initialScene.disk.outer,
      initialScene.disk.outer + 1e-8,
    ]) {
      expect(equatorial(radius)).toEqual({
        ok: false,
        error: "The observer cannot lie on the emitting disk surface.",
      });
    }
    expect(equatorial(middle, Math.PI / 2 + 1e-8).ok).toBe(false);
    expect(equatorial((outerHorizon(initialScene.space) + initialScene.disk.inner) / 2).ok).toBe(
      true,
    );
    expect(equatorial(initialScene.disk.outer + 1e-4).ok).toBe(true);
    expect(equatorial(middle, Math.PI / 2 - 1e-4).ok).toBe(true);
    expect(equatorial(middle, Math.PI / 2 + 1e-4).ok).toBe(true);
  });
});

describe("Camera and navigation", () => {
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
            radius: initialScene.disk.outer + 16,
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
    const scene = createScene({
      ...initialScene,
      camera: turnCamera(initialCamera, 1.3, -0.7),
      plasma: initialPlasma,
    });
    if (!scene.ok) {
      throw new Error(scene.error);
    }
    const view = {
      scene: scene.value,
      appearance: initialAppearance,
      bloom: 0.2,
      exposureEV: 1,
      whiteBalance: 5000,
      time: 4,
      diagnostic: "image",
      analyzer: null,
      display: "auto",
      navigation: "free",
    } as const;
    const result = decodeView(encodeView(view));
    if (result.kind !== "view") {
      throw new Error("Missing restored camera.");
    }
    expect(result.value.navigation).toBe("free");
    expect(result.value.scene.observer).toEqual(view.scene.observer);
    expect(result.value.scene.plasma).toEqual(view.scene.plasma);
    for (const key of ["forward", "up", "right"] as const) {
      for (let i = 0; i < 3; i++) {
        expect(result.value.scene.camera[key][i]).toBeCloseTo(view.scene.camera[key][i] ?? NaN, 14);
      }
    }
  });
});
