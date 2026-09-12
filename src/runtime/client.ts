import type { Session } from "../scene/session.ts";
import type { RenderEvent, RenderRequest } from "./protocol.ts";

/**
 * Transfer a fresh canvas to a dedicated worker and coalesce accepted UI intent.
 *
 * @param canvas - Canvas that has never been transferred; transfer cannot be undone.
 * @param receive - Synchronous UI callback for current replies and lifecycle failures.
 * @returns A client owning the worker; dispose it before remounting with a fresh canvas.
 * Initialization/transfer failures throw. No GPU objects return through the protocol.
 */
export function createRenderClient(
  canvas: HTMLCanvasElement,
  receive: (event: RenderEvent) => void,
) {
  using owned = new DisposableStack();
  const worker = new Worker(new URL("./worker/main.ts", import.meta.url), {
    type: "module",
  });
  owned.defer(() => worker.terminate());
  const offscreen = canvas.transferControlToOffscreen();
  let phase: "active" | "failed" | "disposed" = "active";
  let revision = 0;
  let scheduled = 0;
  let previous: Session | undefined;
  let requested:
    | {
        readonly session: Session;
        readonly hdr: boolean;
        readonly visible: boolean;
        readonly width: number;
        readonly height: number;
        readonly time?: number;
      }
    | undefined;
  const send = (request: RenderRequest, transfer: Transferable[] = []) =>
    worker.postMessage(request, transfer);
  worker.addEventListener("message", ({ data }: MessageEvent<RenderEvent>) => {
    if (data.type === "disposed") {
      lifetime.dispose();
      return;
    }
    if (phase === "active" && (!("revision" in data) || data.revision === revision)) {
      receive(data);
    }
  });
  worker.addEventListener("error", (error) => {
    lifetime.dispose();
    if (phase !== "active") {
      return;
    }
    phase = "failed";
    cancelAnimationFrame(scheduled);
    scheduled = 0;
    requested = undefined;
    receive({ type: "error", message: error.message || "The render worker stopped." });
  });
  send({ type: "initialize", canvas: offscreen }, [offscreen]);
  const lifetime = owned.move();

  function flush() {
    scheduled = 0;
    if (!requested || phase !== "active") {
      return;
    }
    const { session, hdr, visible, width, height, time } = requested;
    const { space, observer, camera, disk, jet, plasma } = session.view.scene;
    send({
      type: "update",
      revision,
      ...(previous?.view.scene !== session.view.scene
        ? { scene: { space, observer, camera, disk, jet, plasma } }
        : {}),
      ...(previous?.view.appearance !== session.view.appearance
        ? { appearance: session.view.appearance }
        : {}),
      presentation: {
        exposureEV: session.view.exposureEV,
        whiteBalance: session.view.whiteBalance,
        bloom: session.view.bloom,
        diagnostic: session.view.diagnostic,
        analyzer: session.view.analyzer,
      },
      motion: session.motion,
      resolution: session.resolution,
      hdr,
      visible,
      width,
      height,
      ...(time === undefined ? {} : { time }),
    });
    previous = session;
    requested = undefined;
  }

  return {
    /** Flush pending intent, then inspect a normalized image point; the default is the center. */
    inspect(point: readonly [number, number] = [0.5, 0.5]) {
      if (phase === "active") {
        cancelAnimationFrame(scheduled);
        flush();
        send({ type: "inspect", revision, point });
      }
    },
    /**
     * Coalesce a session update while preserving a pending explicit epoch restore.
     *
     * @param width - Nonnegative integer content width in device pixels.
     * @param height - Nonnegative integer content height in device pixels.
     * @param time - Optional restored source epoch in M; omission retains the worker clock.
     * @returns The new revision, or the unchanged revision after failure/disposal.
     */
    update(
      session: Session,
      hdr: boolean,
      visible: boolean,
      width: number,
      height: number,
      time?: number,
    ) {
      if (phase !== "active") {
        return revision;
      }
      revision++;
      const epoch = time ?? requested?.time;
      requested = {
        session,
        hdr,
        visible,
        width,
        height,
        ...(epoch === undefined ? {} : { time: epoch }),
      };
      if (!visible || width === 0 || height === 0) {
        cancelAnimationFrame(scheduled);
        flush();
      } else if (!scheduled) {
        scheduled = requestAnimationFrame(flush);
      }
      return revision;
    },
    /** Request a snapshot of the current completed revision; stale replies are ignored. */
    exportPhoto() {
      if (phase === "active") {
        send({ type: "export", revision });
      }
    },
    /**
     * Request device reinitialization while the worker is alive.
     *
     * @returns False after a worker crash, requiring a new canvas/client. True only
     * means the request was sent; it does not certify successful reinitialization.
     * A disposed client returns false and sends no request.
     */
    retry() {
      if (phase === "active") {
        send({ type: "retry" });
      }
      return phase === "active";
    },
    /** Stop scheduling and let the worker release its resources before termination. */
    dispose() {
      if (phase === "disposed") {
        return;
      }
      const active = phase === "active";
      phase = "disposed";
      cancelAnimationFrame(scheduled);
      scheduled = 0;
      requested = undefined;
      previous = undefined;
      if (active) {
        send({ type: "dispose" });
      } else {
        lifetime.dispose();
      }
    },
  };
}
