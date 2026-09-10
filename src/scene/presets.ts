import { initialJet } from "./jet.ts";
import { initialPlasma } from "./plasma.ts";
import { createAppearance } from "./appearance.ts";
import type { SourceAppearance } from "./appearance.ts";
import { createScene, initialScene } from "./scene.ts";
import type { SceneInput } from "./scene.ts";
import type { SavedView } from "./view.ts";
import { initialView } from "./view.ts";

interface Preset {
  readonly name: string;
  readonly description: string;
  readonly view: SavedView;
}

/** Build curated views through the same coupled scene validation as interactive controls. */
function preset(
  name: string,
  description: string,
  input: SceneInput,
  appearance: Partial<SourceAppearance> = {},
  display: Partial<Pick<SavedView, "exposureEV" | "whiteBalance" | "bloom">> = {},
): Preset {
  const scene = createScene(input);
  if (!scene.ok) {
    throw new Error(`${name}: ${scene.error}`);
  }
  const light = createAppearance({ ...initialView.appearance, ...appearance });
  if (!light.ok) {
    throw new Error(`${name}: ${light.error}`);
  }
  return {
    name,
    description,
    view: { ...initialView, ...display, scene: scene.value, appearance: light.value },
  };
}

/** Six distinct starting points; uncommon coordinate experiments remain available through controls. */
export const presets: readonly Preset[] = [
  preset("Classic disk", "Warm light folded around a dark horizon", initialScene),
  preset(
    "Blue accretion flow",
    "A hot disk and a luminous polar outflow",
    {
      ...initialScene,
      space: { spin: 0.7, charge: 0 },
      disk: { inner: 4, outer: 18 },
      jet: { ...initialJet, outer: 80, gammaMin: 1600, density: 8e4, openingAngle: 0.16 },
      observer: { ...initialScene.observer, radius: 55, inclination: 1.18, fieldOfView: 0.64 },
      camera: { forward: [-1, 0, 0], up: [0, -Math.cos(-0.45), Math.sin(-0.45)] },
    },
    { diskTemperature: 12000, diskThickness: 0.045, diskOpticalDepth: 0.22, skyBrightness: 6 },
    { exposureEV: 0.5, whiteBalance: 3200, bloom: 0.18 },
  ),
  preset("Relativistic jets", "Twin streams seen across the rotation axis", {
    ...initialScene,
    jet: initialJet,
    observer: { ...initialScene.observer, radius: 50, inclination: 0.7, fieldOfView: 1.2 },
  }),
  preset("Between horizons", "Observe light from the dynamical interior", {
    ...initialScene,
    space: { spin: 0.7, charge: 0.2 },
    observer: { ...initialScene.observer, radius: 1, motion: { kind: "regular" } },
  }),
  preset("Naked singularity", "Explore geometry without an event horizon", {
    space: { spin: 1.2, charge: 0.3 },
    observer: { ...initialScene.observer, radius: 8, motion: { kind: "regular" } },
  }),
  preset(
    "Thermal refraction",
    "Follow radio rays through a heated plasma",
    {
      ...initialScene,
      plasma: { ...initialPlasma, heating: 6, density: 3e12, frequencyGHz: 20 },
    },
    {},
    { exposureEV: -1, bloom: 0.04 },
  ),
];
