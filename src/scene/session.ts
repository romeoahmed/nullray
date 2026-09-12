import { mapResult } from "./decode.ts";
import { analyzerValue, displayControls, displayValue } from "./presentation.ts";
import type { Result } from "./decode.ts";
import { createAppearance } from "./appearance.ts";
import { initialCamera } from "./camera.ts";
import { createScene, initialScene } from "./scene.ts";
import type { SceneInput } from "./scene.ts";
import type { SavedView } from "./view.ts";
import { initialView } from "./view.ts";

/** Playback intent; photographic refinement freezes the source epoch. */
export type Motion = "playing" | "paused" | "refining";

/** Explicit linear raster scale in (0, 1], or adaptive playback with native paused detail. */
export type Resolution = number | "auto";

/** Readonly user intent; prepared scene buffers may be shared. Live epoch/progress belong to the worker. */
export interface Session {
  readonly view: SavedView;
  readonly motion: Motion;
  readonly resolution: Resolution;
}

export const initialSession: Session = {
  view: initialView,
  motion: "playing",
  resolution: 1,
};

/** Typed adapter intent. Restored views must already have passed decodeView or scene construction. */
export type Action =
  | { readonly type: "scene"; readonly value: SceneInput }
  | { readonly type: "analyzer"; readonly value: number | null }
  | { readonly type: "appearance"; readonly value: unknown }
  | { readonly type: keyof typeof displayControls; readonly value: number }
  | { readonly type: "resolution"; readonly value: Resolution }
  | { readonly type: "display"; readonly value: SavedView["display"] }
  | { readonly type: "diagnostic"; readonly value: SavedView["diagnostic"] }
  | { readonly type: "navigation"; readonly value: SavedView["navigation"] }
  | { readonly type: "restore"; readonly value: SavedView }
  | { readonly type: "toggle-motion" | "refine" | "reset-camera" };

/** Complete session replacement, or a validation error with no state change. */
export type Transition = Result<Session>;

/**
 * Apply one typed action as an atomic session replacement.
 *
 * @remarks
 * Restored views and enum choices must already be decoded. Physical updates
 * use coupled scene validation; display edits preserve preparation and history
 * eligibility. The input session is never mutated.
 *
 * @returns The accepted session, or a domain error with no partial state change.
 * @throws Error - If an unrecognized action reaches the exhaustive boundary.
 */
export function transition(state: Session, action: Action): Transition {
  const { view } = state;
  const replaceView = (patch: Partial<SavedView>): Session => ({
    ...state,
    view: { ...view, ...patch },
  });
  switch (action.type) {
    case "scene": {
      return mapResult(createScene(action.value, view.scene), (scene) => replaceView({ scene }));
    }
    case "analyzer": {
      const { value } = action;
      if (!analyzerValue(value)) {
        return { ok: false, error: "Analyzer angle must lie between 0 and 180 degrees." };
      }
      return { ok: true, value: replaceView({ analyzer: value }) };
    }
    case "appearance": {
      return mapResult(createAppearance(action.value), (appearance) => replaceView({ appearance }));
    }
    case "resolution": {
      const { value } = action;
      if (value !== "auto" && (!Number.isFinite(value) || value <= 0 || value > 1)) {
        return {
          ok: false,
          error: "Resolution must be Auto or a scale greater than zero and at most one.",
        };
      }
      return { ok: true, value: { ...state, resolution: value } };
    }
    case "exposure":
    case "white-balance":
    case "bloom": {
      const { value } = action;
      if (!displayValue(action.type, value)) {
        return { ok: false, error: "This control value is outside its range." };
      }
      return { ok: true, value: replaceView({ [displayControls[action.type].field]: value }) };
    }
    case "display":
      return { ok: true, value: replaceView({ display: action.value }) };
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
      return mapResult(scene, (value) => replaceView({ navigation: action.value, scene: value }));
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
      return mapResult(scene, (value) => replaceView({ scene: value, navigation: "orbit" }));
    }
    default:
      throw new Error("Unknown session action.", { cause: action satisfies never });
  }
}
