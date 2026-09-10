import type { Session } from "../scene/session.ts";
import type { RenderEvent, RenderRequest } from "./protocol.ts";

/** Transfer one fresh canvas and coalesce UI updates without exposing GPU lifetime to the DOM. */
export function createRenderClient(
  canvas: HTMLCanvasElement,
  receive: (event: RenderEvent) => void,
) {
  const offscreen = canvas.transferControlToOffscreen();
  const worker = new Worker(new URL("./worker/main.ts", import.meta.url), {
    type: "module",
  });
  let disposed = false;
  let failed = false;
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
      worker.terminate();
      return;
    }
    if (!disposed && (!("revision" in data) || data.revision === revision)) {
      receive(data);
    }
  });
  worker.addEventListener("error", (error) => {
    failed = true;
    cancelAnimationFrame(scheduled);
    scheduled = 0;
    requested = undefined;
    if (disposed) {
      worker.terminate();
    } else {
      receive({ type: "error", message: error.message || "The render worker stopped." });
    }
  });
  try {
    send({ type: "initialize", canvas: offscreen }, [offscreen]);
  } catch (error) {
    worker.terminate();
    throw error;
  }

  function flush() {
    scheduled = 0;
    if (!requested || disposed) {
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
    /** Inspect the central ray after the latest coalesced scene reaches the worker. */
    inspect(point: readonly [number, number] = [0.5, 0.5]) {
      if (!disposed && !failed) {
        cancelAnimationFrame(scheduled);
        flush();
        send({ type: "inspect", revision, point });
      }
    },
    /** Queue the latest snapshot, preserving an explicit restore epoch across coalesced controls. */
    update(
      session: Session,
      hdr: boolean,
      visible: boolean,
      width: number,
      height: number,
      time?: number,
    ) {
      if (disposed || failed) {
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
      if (!disposed) {
        send({ type: "export", revision });
      }
    },
    /** Retry recoverable device failure; false means the crashed worker requires a fresh canvas. */
    retry() {
      if (!disposed && !failed) {
        send({ type: "retry" });
      }
      return !failed;
    },
    /** Stop scheduling and let the worker release its resources before termination. */
    dispose() {
      if (disposed) {
        return;
      }
      disposed = true;
      cancelAnimationFrame(scheduled);
      if (failed) {
        worker.terminate();
      } else {
        send({ type: "dispose" });
      }
    },
  };
}
