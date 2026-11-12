import { sampleCounts } from "../../gpu/imaging/sampling.ts";
import { createRenderer } from "../../gpu/renderer.ts";
import type { Renderer } from "../../gpu/renderer.ts";
import type { SourceAppearance } from "../../scene/appearance.ts";
import { createAppearance } from "../../scene/appearance.ts";
import { presentationOf } from "../../scene/presentation.ts";
import type { Scene } from "../../scene/scene.ts";
import { createScene } from "../../scene/scene.ts";
import { initialSession } from "../../scene/session.ts";
import type { RenderEvent, RenderRequest } from "../protocol.ts";
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
  presentation: presentationOf(initialSession.view),
  motion: "paused",
  resolution: initialSession.resolution,
  hdr: false,
  visible: true,
  width: 0,
  height: 0,
  revision: 0,
};
/** Failure, retry and disposal each start a new generation; older attempts discard their result. */
let generation = 0;
let initializing = Promise.resolve();

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
  generation++;
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
  const sampling = samplingMode(motion, presentation.diagnostic);
  const stale = () =>
    renderer !== submitted || intent.revision !== submittedRevision || phase !== "ready";
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
      presentation,
      resolutionScale: resolutionScale(
        resolution,
        motion === "playing",
        width,
        height,
        pixelBudget,
      ),
      hdr,
      time: submittedTime,
      sampling,
    });
    await submitted.finished();
    if (stale()) {
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
        if (!stale()) {
          emit({ type: "ray-path", revision: submittedRevision, path });
        }
      } catch (error) {
        if (renderer === submitted && phase === "ready") {
          emit({ type: "inspect-error", revision: submittedRevision, message: message(error) });
        }
      }
      if (stale()) {
        return;
      }
    }
    const coverage = count === 64 ? await submitted.coverage() : undefined;
    if (stale()) {
      return;
    }
    samples = count;
    completedRevision = submittedRevision;
    emit({
      type: "frame",
      revision: submittedRevision,
      time: submittedTime,
      samples: count,
      targetSamples: sampleCounts[sampling],
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

function initialize() {
  if (!canvas || phase === "starting" || phase === "disposed") {
    return;
  }
  phase = "starting";
  const surface = canvas;
  const attempt = ++generation;
  emit({ type: "starting" });
  // Attempts run one after another: a superseded attempt must release the canvas
  // context before the next one configures a new device on it.
  initializing = initializing.then(async () => {
    if (attempt !== generation) {
      return;
    }
    try {
      const value = await createRenderer(surface);
      if (attempt !== generation) {
        value.dispose();
        return;
      }
      renderer = value;
      phase = "ready";
      samples = 0;
      pixelBudget = initialPixelBudget;
      liveFrames = 0;
      liveMilliseconds = 0;
      completedRevision = -1;
      previousTick = undefined;
      void value.failure.then((error) => {
        if (renderer === value) {
          fail(error);
        }
        return undefined;
      });
      schedule();
    } catch (error) {
      if (attempt === generation) {
        fail(error);
      }
    }
    return undefined;
  });
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
        canvas = data.canvas;
        initialize();
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
          initialize();
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
        generation++;
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
