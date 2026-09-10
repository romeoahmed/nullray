import { describe, expect, test } from "vitest";
import * as fc from "fast-check";
import { initialSession, transition } from "../../src/scene/session.ts";
import type { Action, Session } from "../../src/scene/session.ts";
import { encodeView, decodeView } from "../../src/scene/view.ts";
import { createAppearance, initialAppearance } from "../../src/scene/appearance.ts";
import { initialScene } from "../../src/scene/scene.ts";

function apply(session: Session, action: Action): Session {
  const result = transition(session, action);
  if (!result.ok) {
    throw new Error(result.error);
  }
  return result.value;
}

describe("Session transitions", () => {
  test("invalid coupled geometry leaves the complete session untouched", () => {
    const before = encodeView(initialSession.view);
    const result = transition(initialSession, {
      type: "scene",
      value: {
        ...initialSession.view.scene,
        space: { spin: 0.9, charge: 0.9 },
      },
    });
    expect(result.ok).toBe(false);
    expect(encodeView(initialSession.view)).toBe(before);
  });

  test("presentation edits preserve physical inputs and survive a view round trip", () => {
    fc.assert(
      fc.property(
        fc.double({ min: -6, max: 6, noNaN: true }),
        fc.double({ min: 0, max: 1, noNaN: true }),
        (exposure, bloom) => {
          const changed = apply(apply(initialSession, { type: "exposure", value: exposure }), {
            type: "bloom",
            value: bloom,
          });
          expect(changed.view.scene).toEqual(initialSession.view.scene);
          expect(changed.view.appearance).toEqual(initialSession.view.appearance);
          const restored = decodeView(encodeView(changed.view));
          expect(restored.kind).toBe("view");
          if (restored.kind === "view") {
            expect(restored.value.scene).toEqual(changed.view.scene);
            expect(restored.value.appearance).toEqual(changed.view.appearance);
            // JSON normalizes -0; its exposure and bloom semantics are identical to +0.
            expect(restored.value.exposureEV === exposure).toBe(true);
            expect(restored.value.bloom === bloom).toBe(true);
          }
        },
      ),
      {
        examples: [
          [-0, 0],
          [0, -0],
        ],
      },
    );
  });

  test("photography freezes motion, diagnostics stop refinement, and restored views stay paused", () => {
    const photo = apply(initialSession, { type: "refine" });
    expect(photo.motion).toBe("refining");
    const diagnostic = apply(photo, { type: "diagnostic", value: "frequency" });
    expect(diagnostic.motion).toBe("paused");
    expect(transition(diagnostic, { type: "refine" }).ok).toBe(false);
    const restored = apply(diagnostic, {
      type: "restore",
      value: { ...initialSession.view, time: 123 },
    });
    expect(restored.motion).toBe("paused");
    expect(restored.view.time).toBe(123);
    expect(apply(restored, { type: "toggle-motion" }).motion).toBe("playing");
  });
});

describe("Portable views", () => {
  const snapshot = {
    scene: initialScene,
    appearance: initialAppearance,
    time: 123.5,
    exposureEV: -1.2,
    bloom: 0.17,
    navigation: "orbit",
    display: "sdr",
    diagnostic: "image",
  } as const;

  test("view links retain camera, physical inputs, display, and emission epoch", () => {
    fc.assert(
      fc.property(
        fc.double({ min: -6, max: 6, noNaN: true }),
        fc.double({ min: 0, max: 100000, noNaN: true }),
        (exposureEV, time) => {
          const view = { ...snapshot, exposureEV, time };
          // JSON canonicalizes signed zero; both signs have the same exposure/epoch meaning.
          expect(decodeView(encodeView(view))).toEqual({
            kind: "view",
            value: { ...view, exposureEV: exposureEV + 0, time: time + 0 },
          });
        },
      ),
      {
        examples: [
          [-0, 0],
          [0, -0],
        ],
      },
    );
  });

  test("view links reject malformed, missing, unknown-version, and nonphysical inputs atomically", () => {
    const valid = JSON.parse(decodeURIComponent(encodeView(snapshot).slice(6))) as unknown;
    if (typeof valid !== "object" || valid === null) {
      throw new Error("Missing encoded view.");
    }
    const bad = [
      null,
      [],
      {},
      { ...valid, version: 2 },
      { ...valid, version: 4 },
      { ...valid, camera: undefined },
      { ...valid, navigation: undefined },
      { ...valid, appearance: undefined },
      { ...valid, camera: { forward: [1, 0, 0], up: [1, 0, 0] } },
      { ...valid, navigation: "unknown" },
      { ...valid, bloom: null },
      { ...valid, bloom: -0.01 },
      { ...valid, bloom: 1.01 },
      { ...valid, exposureEV: "1" },
      { ...valid, appearance: { ...initialAppearance, skyBrightness: -1 } },
      { ...valid, appearance: null },
      { ...valid, time: 1e308 },
      { ...valid, display: "unknown" },
      { ...valid, diagnostic: false },
      { ...valid, space: { spin: 1, charge: 0.2 } },
      { ...valid, observer: { ...initialScene.observer, radius: 1 } },
    ];
    for (const value of bad) {
      expect(decodeView(`#view=${encodeURIComponent(JSON.stringify(value))}`).kind).toBe("invalid");
    }
    for (const fragment of ["#view=%", "#view={", "#unknown", `#view=${"x".repeat(4096)}`]) {
      expect(decodeView(fragment).kind).toBe("invalid");
    }
    expect(decodeView("")).toEqual({ kind: "empty" });
    fc.assert(
      fc.property(fc.string(), (text) => {
        expect(() => decodeView(text)).not.toThrow();
      }),
    );
  });

  test("view links preserve custom source appearance", () => {
    const appearance = createAppearance({
      diskTemperature: 12000,
      diskStructure: 0,
      skyBrightness: 2,
    });
    if (!appearance.ok) {
      throw new Error(appearance.error);
    }
    const view = { ...snapshot, appearance: appearance.value };
    expect(decodeView(encodeView(view))).toEqual({ kind: "view", value: view });
  });

  test("source appearance rejects untrusted values and permits a dark sky and steady disk", () => {
    for (const value of [
      null,
      [],
      {},
      "warm",
      { ...initialAppearance, diskTemperature: 0 },
      { ...initialAppearance, diskTemperature: 30001 },
      { ...initialAppearance, diskStructure: 1.1 },
      { ...initialAppearance, skyBrightness: 8.1 },
      { ...initialAppearance, skyBrightness: Number.NaN },
      { ...initialAppearance, diskStructure: "0.5" },
    ]) {
      expect(createAppearance(value).ok).toBe(false);
    }
    expect(createAppearance({ diskTemperature: 1000, diskStructure: 0, skyBrightness: 0 }).ok).toBe(
      true,
    );
  });
});
