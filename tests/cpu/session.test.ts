import { describe, expect, test } from "vitest";
import * as fc from "fast-check";
import { initialSession, transition } from "../../src/scene/session.ts";
import type { Action, Session } from "../../src/scene/session.ts";
import { encodeView, decodeView } from "../../src/scene/view.ts";
import { createAppearance, initialAppearance } from "../../src/scene/appearance.ts";
import { initialScene } from "../../src/scene/scene.ts";

test("appearance boundaries survive quantization and repeated view round trips", () => {
  const result = createAppearance({
    ...initialAppearance,
    diskThickness: 0.2,
    diskOpticalDepth: 100,
  });
  if (!result.ok) {
    throw new Error(result.error);
  }
  expect(createAppearance(result.value)).toEqual(result);
  for (const diskThickness of [1e-100, -1, Infinity]) {
    expect(createAppearance({ ...initialAppearance, diskThickness }).ok).toBe(false);
  }
  expect(createAppearance({ ...initialAppearance, diskOpticalDepth: 1e-100 }).ok).toBe(false);
  const view = { ...initialSession.view, appearance: result.value };
  const encoded = encodeView(view);
  const restored = decodeView(encoded);
  if (restored.kind !== "view") {
    throw new Error("Expected a restored boundary appearance.");
  }
  expect(encodeView(restored.value)).toBe(encoded);
});

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
        space: { spin: NaN, charge: 0.9 },
      },
    });
    expect(result.ok).toBe(false);
    expect(encodeView(initialSession.view)).toBe(before);
  });

  test("presentation action sequences retain physical state, last writes, and input immutability", () => {
    const controls = fc.record({
      exposure: fc.double({ min: -6, max: 6, noNaN: true }),
      bloom: fc.double({ min: 0, max: 1, noNaN: true }),
      whiteBalance: fc.integer({ min: 2500, max: 12000 }),
    });
    fc.assert(
      fc.property(fc.array(controls, { minLength: 1, maxLength: 20 }), (edits) => {
        let state = initialSession;
        for (const edit of edits) {
          const before = encodeView(state.view);
          const next = apply(
            apply(apply(state, { type: "exposure", value: edit.exposure }), {
              type: "bloom",
              value: edit.bloom,
            }),
            { type: "white-balance", value: edit.whiteBalance },
          );
          expect(encodeView(state.view)).toBe(before);
          expect(next.view.scene).toBe(initialSession.view.scene);
          expect(next.view.appearance).toBe(initialSession.view.appearance);
          expect(next.motion).toBe(state.motion);
          expect(next.view).toMatchObject({
            exposureEV: edit.exposure,
            bloom: edit.bloom,
            whiteBalance: edit.whiteBalance,
          });
          state = next;
        }
      }),
      { numRuns: 40 },
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
    whiteBalance: 4800,
    bloom: 0.17,
    navigation: "orbit",
    display: "sdr",
    diagnostic: "image",
    analyzer: null,
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
    const valid: unknown = JSON.parse(decodeURIComponent(encodeView(snapshot).slice(6)));
    if (typeof valid !== "object" || valid === null) {
      throw new Error("Missing encoded view.");
    }
    const bad = [
      null,
      [],
      {},
      { ...valid, version: 2 },
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
      { ...valid, space: { spin: "1", charge: 0.2 } },
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
      fc.property(
        fc.oneof(
          fc.string(),
          fc.jsonValue().map((value) => `#view=${encodeURIComponent(JSON.stringify(value))}`),
        ),
        (text) => {
          expect(() => decodeView(text)).not.toThrow();
        },
      ),
    );
  });

  test("view links preserve custom source appearance", () => {
    const appearance = createAppearance({
      ...initialAppearance,
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
      { ...initialAppearance, skyBrightness: 64.1 },
      { ...initialAppearance, skyBrightness: Number.NaN },
      { ...initialAppearance, diskStructure: "0.5" },
    ]) {
      expect(createAppearance(value).ok).toBe(false);
    }
    expect(
      createAppearance({
        ...initialAppearance,
        diskTemperature: 1000,
        diskStructure: 0,
        skyBrightness: 0,
      }).ok,
    ).toBe(true);
  });
});

test("native resolution is the default and adaptive playback is an explicit choice", () => {
  expect(initialSession.resolution).toBe(1);
  for (const value of ["auto", 0.25, 0.5, 0.75, 1] as const) {
    expect(apply(initialSession, { type: "resolution", value }).resolution).toBe(value);
  }
  for (const value of [0, -1, 1.01, NaN, Infinity]) {
    expect(transition(initialSession, { type: "resolution", value }).ok).toBe(false);
  }
});

test("display actions and persisted values agree at closed and half-open boundaries", () => {
  const state = Object.freeze({
    ...initialSession,
    view: Object.freeze({ ...initialSession.view }),
  });
  const before = encodeView(state.view);
  for (const [type, field, minimum, maximum] of [
    ["exposure", "exposureEV", -6, 6],
    ["white-balance", "whiteBalance", 2500, 12000],
    ["bloom", "bloom", 0, 1],
  ] as const) {
    for (const value of [minimum, maximum]) {
      const next = apply(state, { type, value });
      expect(next.view[field]).toBe(value);
      expect(next.view.scene).toBe(state.view.scene);
      expect(next.view.appearance).toBe(state.view.appearance);
      expect(decodeView(encodeView(next.view))).toEqual({ kind: "view", value: next.view });
    }
    for (const value of [minimum - 1, maximum + 1, NaN, Infinity, -Infinity]) {
      expect(transition(state, { type, value }).ok).toBe(false);
      expect(decodeView(encodeView({ ...state.view, [field]: value })).kind).toBe("invalid");
    }
  }
  for (const value of [null, 0, Math.PI / 2]) {
    const next = apply(state, { type: "analyzer", value });
    expect(decodeView(encodeView(next.view))).toEqual({ kind: "view", value: next.view });
  }
  for (const value of [-1, Math.PI, Infinity, NaN]) {
    expect(transition(state, { type: "analyzer", value }).ok).toBe(false);
    // JSON encodes nonfinite numbers as null, which intentionally bypasses the analyzer.
    if (Number.isFinite(value)) {
      expect(decodeView(encodeView({ ...state.view, analyzer: value })).kind).toBe("invalid");
    }
  }
  expect(encodeView(state.view)).toBe(before);
});
