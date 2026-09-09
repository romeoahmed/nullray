import { createScene, initialScene } from "./scene.ts";
import type { SceneInput } from "./scene.ts";
import type { SavedView } from "./view.ts";
import { initialView } from "./view.ts";

interface Preset {
  readonly name: string;
  readonly view: SavedView;
}

/** Build curated views through the same coupled scene validation as interactive controls. */
function preset(name: string, input: SceneInput): Preset {
  const scene = createScene(input);
  if (!scene.ok) {
    throw new Error(`${name}: ${scene.error}`);
  }
  return {
    name,
    view: { ...initialView, scene: scene.value },
  };
}

export const presets: readonly Preset[] = [
  preset("Classic disk", initialScene),
  preset("Wide sky", {
    ...initialScene,
    observer: { ...initialScene.observer, radius: 80, fieldOfView: Math.PI / 2 },
  }),
  preset("Polar view", {
    ...initialScene,
    observer: { ...initialScene.observer, inclination: 0 },
  }),
  preset("Non-spinning", {
    ...initialScene,
    space: { spin: 0, charge: 0 },
  }),
];
