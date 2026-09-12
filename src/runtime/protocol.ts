import type { SourceAppearance } from "../scene/appearance.ts";
import type { SceneInput } from "../scene/scene.ts";
import type { Motion, Resolution } from "../scene/session.ts";
import type { Presentation } from "../scene/presentation.ts";
import type { Coverage } from "../gpu/imaging/coverage.ts";
import type { RayPath } from "../physics/ray-path.ts";

/**
 * Internal messages to the dedicated render worker.
 *
 * @remarks
 * Updates carry complete presentation/runtime intent but may omit unchanged
 * scene/appearance inputs. Width/height are native device-pixel dimensions;
 * inspection points are normalized image coordinates. Only initialization
 * transfers an OffscreenCanvas; no GPU handles cross the boundary.
 */
export type RenderRequest =
  | { readonly type: "initialize"; readonly canvas: OffscreenCanvas }
  | {
      readonly type: "update";
      readonly revision: number;
      readonly scene?: SceneInput;
      readonly appearance?: SourceAppearance;
      readonly presentation: Presentation;
      readonly motion: Motion;
      readonly resolution: Resolution;
      readonly hdr: boolean;
      readonly visible: boolean;
      readonly width: number;
      readonly height: number;
      /** A restore starts a new emission epoch; ordinary controls retain the worker clock. */
      readonly time?: number;
    }
  | { readonly type: "export"; readonly revision: number }
  | {
      readonly type: "inspect";
      readonly revision: number;
      readonly point: readonly [number, number];
    }
  | { readonly type: "retry" | "dispose" };

/**
 * Completion feedback for one intent revision after submitted GPU work settles.
 *
 * @remarks
 * Width/height describe the actual raster, not requested native dimensions.
 * Time is the source epoch in M; samples count the retained sequence. Coverage
 * is optional and reports missing work rather than physical accuracy.
 */
export interface CompletedFrame {
  readonly type: "frame";
  readonly revision: number;
  readonly time: number;
  readonly samples: number;
  /** Target for this image sequence; stationary previews stop when they reach it. */
  readonly targetSamples: number;
  readonly width: number;
  readonly height: number;
  readonly coverage?: Coverage;
}

/** Worker replies; revision-tagged results are discarded after a newer client update. */
export type RenderEvent =
  | { readonly type: "ray-path"; readonly revision: number; readonly path: RayPath }
  | { readonly type: "inspect-error"; readonly revision: number; readonly message: string }
  | CompletedFrame
  | { readonly type: "starting" | "disposed" }
  | { readonly type: "error"; readonly message: string }
  | { readonly type: "exported"; readonly revision: number; readonly blob: Blob }
  | { readonly type: "export-error"; readonly revision: number; readonly message: string };
