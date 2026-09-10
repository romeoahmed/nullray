import type { Result } from "./decode.ts";
import { createAppearance } from "./appearance.ts";
import { initialCamera } from "./camera.ts";
import { createScene, initialScene } from "./scene.ts";
import type { SceneInput } from "./scene.ts";
import type { SavedView } from "./view.ts";
import { initialView } from "./view.ts";

/** Playback intent; photographic refinement freezes the source epoch. */
export type Motion = "playing" | "paused" | "refining";

/** Immutable user intent. Completed frame time and GPU progress belong to the renderer. */
export interface Session {
  readonly view: SavedView;
  readonly motion: Motion;
  readonly resolution: number;
}

export const initialSession: Session = {
  view: initialView,
  motion: "playing",
  resolution: 0.5,
};

/** Typed adapter intent. Restored views must already have passed decodeView or scene construction. */
export type Action =
  | { readonly type: "scene"; readonly value: SceneInput }
  | { readonly type: "analyzer"; readonly value: number | null }
  | { readonly type: "appearance"; readonly value: unknown }
  | { readonly type: "exposure" | "white-balance" | "bloom" | "resolution"; readonly value: number }
  | { readonly type: "display"; readonly value: SavedView["display"] }
  | { readonly type: "diagnostic"; readonly value: SavedView["diagnostic"] }
  | { readonly type: "navigation"; readonly value: SavedView["navigation"] }
  | { readonly type: "restore"; readonly value: SavedView }
  | { readonly type: "toggle-motion" | "refine" | "reset-camera" };

/** Complete session replacement, or a validation error with no state change. */
export type Transition = Result<Session>;

/** Apply one complete action; rejected input cannot partially alter coupled parameters. */
export function transition(state: Session, action: Action): Transition {
  const { view } = state;
  switch (action.type) {
    case "scene": {
      const scene = createScene(action.value, view.scene);
      return scene.ok
        ? { ok: true, value: { ...state, view: { ...view, scene: scene.value } } }
        : scene;
    }
    case "analyzer": {
      const { value } = action;
      if (value !== null && (!Number.isFinite(value) || value < 0 || value >= Math.PI)) {
        return { ok: false, error: "Analyzer angle must lie between 0 and 180 degrees." };
      }
      return { ok: true, value: { ...state, view: { ...view, analyzer: value } } };
    }
    case "appearance": {
      const appearance = createAppearance(action.value);
      return appearance.ok
        ? { ok: true, value: { ...state, view: { ...view, appearance: appearance.value } } }
        : appearance;
    }
    case "exposure":
    case "white-balance":
    case "bloom":
    case "resolution": {
      const { value } = action;
      const valid =
        Number.isFinite(value) &&
        (action.type === "exposure"
          ? value >= -6 && value <= 6
          : action.type === "white-balance"
            ? value >= 2500 && value <= 12000
            : action.type === "bloom"
              ? value >= 0 && value <= 1
              : value > 0 && value <= 1);
      if (!valid) {
        return { ok: false, error: "This control value is outside its range." };
      }
      return {
        ok: true,
        value:
          action.type === "resolution"
            ? { ...state, resolution: value }
            : {
                ...state,
                view: {
                  ...view,
                  [action.type === "exposure"
                    ? "exposureEV"
                    : action.type === "white-balance"
                      ? "whiteBalance"
                      : "bloom"]: value,
                },
              },
      };
    }
    case "display":
      return { ok: true, value: { ...state, view: { ...view, display: action.value } } };
    case "diagnostic":
      return {
        ok: true,
        value: {
          ...state,
          motion: state.motion === "refining" ? "paused" : state.motion,
          view: { ...view, diagnostic: action.value },
        },
      };
    case "navigation": {
      const scene =
        action.value === "orbit"
          ? createScene({ ...view.scene, camera: initialCamera }, view.scene)
          : { ok: true as const, value: view.scene };
      return scene.ok
        ? {
            ok: true,
            value: {
              ...state,
              view: {
                ...view,
                navigation: action.value,
                scene: scene.value,
              },
            },
          }
        : scene;
    }
    case "restore":
      return { ok: true, value: { ...state, view: action.value, motion: "paused" } };
    case "toggle-motion":
      return {
        ok: true,
        value: { ...state, motion: state.motion === "playing" ? "paused" : "playing" },
      };
    case "refine":
      return view.diagnostic === "image"
        ? { ok: true, value: { ...state, motion: "refining" } }
        : { ok: false, error: "Choose the image view before refining a photograph." };
    case "reset-camera": {
      const scene = createScene(
        { ...view.scene, observer: initialScene.observer, camera: initialCamera },
        view.scene,
      );
      return scene.ok
        ? {
            ok: true,
            value: { ...state, view: { ...view, scene: scene.value, navigation: "orbit" } },
          }
        : scene;
    }
    default:
      throw new Error("Unknown session action.", { cause: action satisfies never });
  }
}
