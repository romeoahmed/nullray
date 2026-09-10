import type { SourceAppearance } from "../scene/appearance.ts";
import type { SceneInput } from "../scene/scene.ts";
import type { Motion } from "../scene/session.ts";
import type { SavedView } from "../scene/view.ts";
import type { Coverage } from "../gpu/coverage.ts";

/** Messages stay within this application's dedicated worker; no GPU objects cross back. */
export type RenderRequest =
  | { readonly type: "initialize"; readonly canvas: OffscreenCanvas }
  | {
      readonly type: "update";
      readonly revision: number;
      readonly scene?: SceneInput;
      readonly appearance?: SourceAppearance;
      readonly presentation: Pick<SavedView, "exposureEV" | "bloom" | "diagnostic">;
      readonly motion: Motion;
      readonly resolution: number;
      readonly hdr: boolean;
      readonly visible: boolean;
      readonly width: number;
      readonly height: number;
      /** A restore starts a new emission epoch; ordinary controls retain the worker clock. */
      readonly time?: number;
    }
  | { readonly type: "export"; readonly revision: number }
  | { readonly type: "retry" | "dispose" };

/** Completed GPU work tagged with the intent revision that produced it. */
export interface CompletedFrame {
  readonly type: "frame";
  readonly revision: number;
  readonly time: number;
  readonly samples: number;
  readonly width: number;
  readonly height: number;
  readonly coverage?: Coverage;
}

/** Worker replies; revision-tagged results are discarded after a newer client update. */
export type RenderEvent =
  | CompletedFrame
  | { readonly type: "starting" | "disposed" }
  | { readonly type: "error"; readonly message: string }
  | { readonly type: "exported"; readonly revision: number; readonly blob: Blob }
  | { readonly type: "export-error"; readonly revision: number; readonly message: string };
