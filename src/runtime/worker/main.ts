import { createRenderer } from "../../gpu/renderer.ts";
import type { Renderer } from "../../gpu/renderer.ts";
import type { RenderEvent, RenderRequest } from "../protocol.ts";
import type { Scene } from "../../scene/scene.ts";
import type { SourceAppearance } from "../../scene/appearance.ts";
import { createScene } from "../../scene/scene.ts";
import { createAppearance } from "../../scene/appearance.ts";
import { initialSession } from "../../scene/session.ts";

import { sampleCounts } from "../../gpu/imaging/sampling.ts";
import { initialPixelBudget, resolutionScale, nextPixelBudget } from "./resolution.ts";

/** Diagnostics retain categorical samples; ordinary paused images settle automatically. */
const samplingMode = (
  motion: RenderIntent["motion"],
  view: RenderIntent["presentation"]["diagnostic"],
) =>
  motion === "refining"
    ? "photograph"
    : motion === "paused" && view === "image"
      ? "settle"
      : "live";

const port = self as DedicatedWorkerGlobalScope;
const lifetime = new AbortController();
let phase: "idle" | "starting" | "ready" | "failed" | "disposed" = "idle";
let renderer: Renderer | undefined;
let canvas: OffscreenCanvas | undefined;
type RenderIntent = Extract<RenderRequest, { type: "update" }> & {
  readonly scene: Scene;
  readonly appearance: SourceAppearance;
};

let intent: RenderIntent = {
  type: "update",
  scene: initialSession.view.scene,
  appearance: initialSession.view.appearance,
  presentation: {
    exposureEV: initialSession.view.exposureEV,
    whiteBalance: initialSession.view.whiteBalance,
    bloom: initialSession.view.bloom,
    diagnostic: initialSession.view.diagnostic,
    analyzer: initialSession.view.analyzer,
  },
  motion: "paused",
  resolution: initialSession.resolution,
  hdr: false,
  visible: true,
  width: 0,
  height: 0,
  revision: 0,
};
let initialization: AbortController | undefined;
let initializationTail = Promise.resolve();

/** Prepare source inputs on the worker and commit one complete rendering intent. */
function prepareIntent(
  current: RenderIntent,
  request: Extract<RenderRequest, { type: "update" }>,
): RenderIntent {
  const nextScene = request.scene
    ? createScene(request.scene, current.scene)
    : ({ ok: true, value: current.scene } as const);
  const nextAppearance = request.appearance
    ? createAppearance(request.appearance)
    : ({ ok: true, value: current.appearance } as const);
  if (!nextScene.ok) {
    throw new Error(nextScene.error);
  }
  if (!nextAppearance.ok) {
    throw new Error(nextAppearance.error);
  }
  return { ...request, scene: nextScene.value, appearance: nextAppearance.value };
}

let completedRevision = -1;
let samples = 0;
let pixelBudget = initialPixelBudget;
let liveFrames = 0;
let liveMilliseconds = 0;
let time = 0;
let previousTick: number | undefined;
let scheduled = 0;
let pending = false;
let requestedInspection: Extract<RenderRequest, { type: "inspect" }> | undefined;

const emit = (event: RenderEvent) => port.postMessage(event);
const message = (error: unknown) => (error instanceof Error ? error.message : "Rendering failed.");

function fail(error: unknown) {
  if (phase === "disposed") {
    return;
  }
  phase = "failed";
  initialization?.abort();
  initialization = undefined;
  const old = renderer;
  renderer = undefined;
  old?.dispose();
  cancelAnimationFrame(scheduled);
  scheduled = 0;
  emit({ type: "error", message: message(error) });
}

function schedule() {
  const { revision, motion, visible, width, height } = intent;
  if (phase !== "ready" || pending || scheduled || !visible || !width || !height) {
    return;
  }
  if (
    revision !== completedRevision ||
    requestedInspection?.revision === revision ||
    motion === "playing" ||
    samples < sampleCounts[samplingMode(motion, intent.presentation.diagnostic)]
  ) {
    scheduled = requestAnimationFrame((now) => {
      void draw(now);
    });
  }
}

/** Keep at most one frame completion pending; incoming controls may replace the next intent. */
async function draw(now: number) {
  const {
    scene,
    appearance,
    presentation,
    motion,
    resolution,
    hdr,
    visible,
    width,
    height,
    revision,
  } = intent;
  scheduled = 0;
  const submitted = renderer;
  if (!submitted || phase !== "ready" || pending || !visible || !width || !height) {
    return;
  }
  pending = true;
  const submittedRevision = revision;
  if (motion === "playing" && previousTick !== undefined) {
    time += Math.min(0.25, (now - previousTick) / 1000) * 10;
  }
  previousTick = now;
  const submittedTime = time;
  const inspect =
    requestedInspection?.revision === submittedRevision ? requestedInspection : undefined;
  if (inspect) {
    requestedInspection = undefined;
  }
  try {
    submitted.resize(width, height);
    const started = performance.now();
    const count = submitted.render(scene, {
      appearance,
      exposureEV: presentation.exposureEV,
      whiteBalance: presentation.whiteBalance,
      bloom: presentation.bloom,
      view: presentation.diagnostic,
      analyzer: presentation.analyzer,
      resolutionScale: resolutionScale(
        resolution,
        motion === "playing",
        width,
        height,
        pixelBudget,
      ),
      hdr,
      time: submittedTime,
      sampling: samplingMode(motion, presentation.diagnostic),
    });
    await submitted.finished();
    if (renderer !== submitted || intent.revision !== submittedRevision || phase !== "ready") {
      return;
    }
    if (motion === "playing" && resolution === "auto") {
      liveMilliseconds += performance.now() - started;
      liveFrames++;
      if (liveFrames === 8) {
        pixelBudget = nextPixelBudget(
          pixelBudget,
          (canvas?.width ?? 0) * (canvas?.height ?? 0),
          liveMilliseconds / liveFrames,
        );
        liveFrames = 0;
        liveMilliseconds = 0;
      }
    } else {
      liveFrames = 0;
      liveMilliseconds = 0;
    }
    if (inspect) {
      try {
        const path = await submitted.inspect(inspect.point);
        if (renderer === submitted && intent.revision === submittedRevision && phase === "ready") {
          emit({ type: "ray-path", revision: submittedRevision, path });
        }
      } catch (error) {
        if (renderer === submitted && phase === "ready") {
          emit({ type: "inspect-error", revision: submittedRevision, message: message(error) });
        }
      }
      if (renderer !== submitted || intent.revision !== submittedRevision || phase !== "ready") {
        return;
      }
    }
    const coverage = count === 64 ? await submitted.coverage() : undefined;
    if (renderer !== submitted || intent.revision !== submittedRevision || phase !== "ready") {
      return;
    }
    samples = count;
    completedRevision = submittedRevision;
    emit({
      type: "frame",
      revision: submittedRevision,
      time: submittedTime,
      samples: count,
      targetSamples: sampleCounts[samplingMode(motion, presentation.diagnostic)],
      width: canvas?.width ?? 0,
      height: canvas?.height ?? 0,
      ...(coverage ? { coverage } : {}),
    });
  } catch (error) {
    if (renderer === submitted) {
      fail(error);
    }
  } finally {
    pending = false;
    schedule();
  }
}

async function initialize() {
  if (!canvas || phase === "starting" || phase === "disposed") {
    return;
  }
  phase = "starting";
  const attempt = new AbortController();
  initialization = attempt;
  const preceding = initializationTail;
  const completion = Promise.withResolvers<void>();
  initializationTail = completion.promise;
  emit({ type: "starting" });
  try {
    // A superseded initializer must release the shared canvas context before
    // another initializer can acquire it and eventually configure a new device.
    await preceding;
    if (lifetime.signal.aborted || initialization !== attempt) {
      return;
    }
    const value = await createRenderer(canvas, AbortSignal.any([lifetime.signal, attempt.signal]));
    if (lifetime.signal.aborted || initialization !== attempt) {
      value.dispose();
      return;
    }
    initialization = undefined;
    renderer = value;
    phase = "ready";
    samples = 0;
    pixelBudget = initialPixelBudget;
    liveFrames = 0;
    liveMilliseconds = 0;
    completedRevision = -1;
    previousTick = undefined;
    void value.lost.then(() => {
      if (renderer === value) {
        fail(new Error("The graphics device was lost. Retry to restore this view."));
      }
      return undefined;
    });
    schedule();
  } catch (error) {
    if (initialization === attempt) {
      fail(error);
    }
  } finally {
    completion.resolve();
  }
}

async function exportPhoto(requestedRevision: number) {
  const submitted = renderer;
  try {
    if (
      !submitted ||
      pending ||
      samples !== 64 ||
      requestedRevision !== completedRevision ||
      intent.revision !== completedRevision
    ) {
      throw new Error("Wait for the current photograph to finish before saving it.");
    }
    const blob = await submitted.exportPNG();
    if (phase !== "disposed" && renderer === submitted && intent.revision === requestedRevision) {
      emit({ type: "exported", revision: requestedRevision, blob });
    }
  } catch (error) {
    if (phase !== "disposed") {
      emit({ type: "export-error", revision: requestedRevision, message: message(error) });
    }
  }
}

port.addEventListener("message", ({ data }: MessageEvent<RenderRequest>) => {
  if (phase === "disposed") {
    return;
  }
  try {
    switch (data.type) {
      case "initialize":
        if (canvas) {
          throw new Error("The render surface is already owned by this worker.");
        }
        canvas = data.canvas;
        void initialize();
        break;
      case "update": {
        const next = prepareIntent(intent, data);
        if (next.time !== undefined) {
          time = next.time;
          previousTick = undefined;
        }
        if (intent.motion !== next.motion || intent.visible !== next.visible) {
          previousTick = undefined;
        }
        if (
          intent.resolution !== next.resolution ||
          intent.motion !== next.motion ||
          intent.visible !== next.visible ||
          intent.width !== next.width ||
          intent.height !== next.height
        ) {
          liveFrames = 0;
          liveMilliseconds = 0;
        }
        intent = next;
        if (requestedInspection?.revision !== intent.revision) {
          requestedInspection = undefined;
        }
        schedule();
        break;
      }
      case "retry":
        if (phase === "failed") {
          void initialize();
        }
        break;
      case "export":
        void exportPhoto(data.revision);
        break;
      case "inspect":
        if (data.revision === intent.revision) {
          requestedInspection = data;
          schedule();
        }
        break;
      case "dispose": {
        phase = "disposed";
        lifetime.abort();
        cancelAnimationFrame(scheduled);
        const old = renderer;
        renderer = undefined;
        old?.dispose();
        emit({ type: "disposed" });
        port.close();
        break;
      }
    }
  } catch (error) {
    fail(error);
  }
});
