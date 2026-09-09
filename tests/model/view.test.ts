import { expect, test } from "vitest";
import * as fc from "fast-check";
import { createAppearance, initialAppearance } from "../../src/model/appearance.ts";
import { initialScene } from "../../src/model/scene.ts";
import { decodeView, encodeView } from "../../src/model/view.ts";

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
    { ...valid, version: 5 },
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

test("legacy views migrate their fixed source appearance and new views preserve custom light", () => {
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
  const versionThree = {
    version: 3,
    ...snapshot,
    space: initialScene.space,
    observer: initialScene.observer,
  };
  expect(decodeView(`#view=${encodeURIComponent(JSON.stringify(versionThree))}`)).toEqual({
    kind: "view",
    value: snapshot,
  });
  const legacy = {
    version: 1,
    space: initialScene.space,
    observer: initialScene.observer,
    time: snapshot.time,
    exposureEV: snapshot.exposureEV,
    display: snapshot.display,
    diagnostic: snapshot.diagnostic,
  };
  expect(decodeView(`#view=${encodeURIComponent(JSON.stringify(legacy))}`)).toEqual({
    kind: "view",
    value: { ...snapshot, bloom: 0 },
  });
  expect(
    decodeView(
      `#view=${encodeURIComponent(JSON.stringify({ ...legacy, version: 2, appearance: appearance.value }))}`,
    ),
  ).toEqual({
    kind: "view",
    value: { ...snapshot, appearance: appearance.value, bloom: 0 },
  });
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
