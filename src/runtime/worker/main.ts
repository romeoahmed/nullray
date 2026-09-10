import { createRenderer } from "../../gpu/renderer.ts";
import type { Renderer } from "../../gpu/renderer.ts";
import type { RenderEvent, RenderRequest } from "../protocol.ts";
import type { Scene } from "../../scene/scene.ts";
import type { SourceAppearance } from "../../scene/appearance.ts";
import { createScene } from "../../scene/scene.ts";
import { createAppearance } from "../../scene/appearance.ts";
import { initialSession } from "../../scene/session.ts";

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
    bloom: initialSession.view.bloom,
    diagnostic: initialSession.view.diagnostic,
  },
  motion: "paused",
  resolution: 1,
  hdr: false,
  visible: true,
  width: 0,
  height: 0,
  revision: 0,
};
let initialization: AbortController | undefined;
let initializationTail = Promise.resolve();

/** Reconstruct clone-lost brands and commit a complete validated rendering intent. */
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
let time = 0;
let previousTick: number | undefined;
let scheduled = 0;
let pending = false;

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
    motion === "playing" ||
    (motion === "refining" && samples < 64)
  ) {
    scheduled = requestAnimationFrame((now) => {
      void draw(now);
    });
  }
}

/** At most one submitted frame; incoming controls replace intent while it completes. */
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
  try {
    submitted.resize(width, height);
    const count = submitted.render(scene, {
      appearance,
      exposureEV: presentation.exposureEV,
      bloom: presentation.bloom,
      view: presentation.diagnostic,
      resolutionScale: resolution,
      hdr,
      time: submittedTime,
      refine: motion === "refining",
    });
    await submitted.finished();
    if (renderer !== submitted || intent.revision !== submittedRevision || phase !== "ready") {
      return;
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
        intent = next;
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
