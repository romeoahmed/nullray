import { createRenderClient } from "../runtime/client.ts";
import type { CompletedFrame, RenderEvent } from "../runtime/protocol.ts";
import { presets } from "../scene/presets.ts";
import { initialSession, transition } from "../scene/session.ts";
import type { Action, Session } from "../scene/session.ts";
import { decodeView, encodeView } from "../scene/view.ts";
import type { SavedView } from "../scene/view.ts";
import { bindAudio } from "./audio.ts";
import { bindViews } from "./bookmarks.ts";
import { bindControls } from "./controls.ts";
import { element } from "./elements.ts";
import { bindNavigation } from "./navigation.ts";
import { bindReadouts } from "./readouts.ts";
import { drawTopology } from "./topology.ts";

/**
 * Mount UI subscriptions and a worker-backed canvas with one cleanup owner.
 *
 * @param root - Application markup; its canvas is replaced before permanent transfer.
 * @param restored - Accepted session for hot remount, otherwise restore the current URL.
 * @returns Cleanup that detaches subscriptions, requests worker disposal, and
 * returns the session with the latest observed epoch. Worker termination is asynchronous.
 */
export function mountApp(root: HTMLElement, restored?: Session): () => Session {
  using owned = new DisposableStack();
  const subscriptions = new AbortController();
  const { signal } = subscriptions;
  owned.defer(() => subscriptions.abort());
  owned.defer(bindAudio(root));
  const button = (id: string) => element(root, `#${id}`, HTMLButtonElement);
  const text = (id: string) => element(root, `#${id}`, HTMLElement);
  const original = element(root, "canvas", HTMLCanvasElement);
  // An OffscreenCanvas transfer is permanent. A remount gets a fresh surface.
  const canvas = original.cloneNode(false);
  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new Error("Missing canvas.");
  }
  original.replaceWith(canvas);
  const status = text("status"),
    feedback = text("feedback"),
    detail = text("render-detail");
  const observation = text("observation-state");
  const focusView = button("focus-view");
  let imageOnly = false;
  const focusSurfaces = root.querySelectorAll<HTMLElement>(
    ".wordmark, .header-actions > button:not(#focus-view), .observation, .bottom-ui, .telemetry",
  );
  const setImageOnly = (value: boolean) => {
    imageOnly = value;
    for (const panel of root.querySelectorAll<HTMLElement>("[popover]:popover-open")) {
      panel.hidePopover();
    }
    root.classList.toggle("image-only", value);
    for (const surface of focusSurfaces) {
      surface.inert = value;
    }
    focusView.setAttribute("aria-pressed", String(value));
    focusView.setAttribute("aria-label", value ? "Show controls" : "Hide controls");
    focusView.textContent = value ? "Controls" : "◩";
    canvas.focus({ preventScroll: true });
  };
  focusView.addEventListener("click", () => setImageOnly(!imageOnly), { signal });
  owned.defer(() => {
    root.classList.remove("image-only");
    focusView.setAttribute("aria-pressed", "false");
    focusView.setAttribute("aria-label", "Hide controls");
    focusView.textContent = "◩";
    for (const surface of focusSurfaces) {
      surface.inert = false;
    }
  });
  const settings = text("controls"),
    settingsFeedback = text("settings-feedback"),
    feedbackHome = text("feedback-home");
  settings.addEventListener(
    "toggle",
    (event) => {
      (event.newState === "open" ? settingsFeedback : feedbackHome).append(feedback);
    },
    { signal },
  );
  const motion = button("toggle-motion"),
    photo = button("photograph"),
    save = button("export-photo");
  const retry = button("retry"),
    photoPanel = text("photo-status"),
    photoLabel = text("photo-label");
  const coverage = text("sampling-status"),
    exportStatus = text("export-status");
  const progress = element(root, "#photo-progress", HTMLProgressElement);
  const syncReadouts = bindReadouts(root);
  const topologyMap = element(root, "#topology-map", SVGSVGElement);
  const inspectRay = button("inspect-ray");
  const pickRay = button("pick-ray");
  let pickingRay = false;
  owned.defer(() => canvas.style.removeProperty("cursor"));
  const rayStatus = text("ray-status");
  const shareDialog = element(root, "#view-dialog", HTMLDialogElement);
  const shareInput = element(root, "#view-link", HTMLInputElement);
  const copyStatus = text("copy-status");
  const highRange = matchMedia("(dynamic-range: high)");
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  let session: Session = restored ?? {
    ...initialSession,
    motion: reducedMotion.matches ? "paused" : "playing",
  };
  let revision = 0;
  let completed: CompletedFrame | undefined;
  let initialized = false;
  let exporting = false;
  let width = 0,
    height = 0;
  let photoURL: string | undefined;
  owned.defer(() => {
    if (photoURL) {
      URL.revokeObjectURL(photoURL);
    }
  });
  const percent = new Intl.NumberFormat("en", { style: "percent", maximumSignificantDigits: 2 });
  const hdr = () =>
    session.view.display === "hdr" || (session.view.display === "auto" && highRange.matches);
  const snapshot = (): SavedView => ({
    ...session.view,
    time: completed?.time ?? session.view.time,
  });

  function syncStatus() {
    const { view } = session;
    const frame = completed?.revision === revision ? completed : undefined;
    motion.textContent = session.motion === "playing" ? "Pause" : "Play";
    motion.setAttribute(
      "aria-label",
      session.motion === "playing" ? "Pause motion" : "Resume motion",
    );
    motion.setAttribute("aria-pressed", String(session.motion === "playing"));
    photo.disabled = view.diagnostic !== "image";
    photo.setAttribute("aria-pressed", String(session.motion === "refining"));
    photoPanel.hidden = session.motion !== "refining";
    const samples = frame?.samples ?? 0;
    const state = !initialized
      ? "Starting"
      : session.motion === "playing"
        ? "Live"
        : session.motion === "refining"
          ? samples === 64
            ? "Photograph complete"
            : "Photographing"
          : frame && samples >= frame.targetSamples
            ? "Paused · image settled"
            : "Paused · settling";
    observation.textContent = state;
    progress.value = samples;
    photoLabel.textContent =
      !frame && completed?.samples === 64
        ? "Updating photograph…"
        : samples === 64
          ? "Photograph ready"
          : `Refining · ${samples} / 64`;
    save.disabled = samples !== 64 || !frame?.coverage || exporting;
    if (frame?.coverage) {
      const value = frame.coverage;
      coverage.textContent =
        value.unresolvedSamples === 0
          ? "All sampled rays resolved."
          : `${value.unresolvedPixels.toLocaleString("en")} pixels contain unresolved samples (${percent.format(value.unresolvedSamples / (value.pixels * value.samplesPerPixel))} of samples).`;
    } else {
      coverage.textContent =
        completed?.samples === 64
          ? "Applying changes to the photograph…"
          : "64 samples per pixel. Changing the scene restarts.";
    }
    if (initialized) {
      const summary = `Kerr–Newman · a ${view.scene.space.spin.toFixed(2)} · q ${view.scene.space.charge.toFixed(2)}`;
      if (status.textContent !== summary) {
        status.textContent = summary;
      }
    }
    syncReadouts(view, hdr());
  }

  function receive(event: RenderEvent) {
    switch (event.type) {
      case "ray-path": {
        drawTopology(topologyMap, session.view.scene, event.path);
        // Collapse adjacent repeats only: a later return to a domain remains part of the route.
        const domains = event.path.points
          .map(
            (point) =>
              `${point.block} ${point.universe}${point.side === 0 ? "" : `/${point.side}`}`,
          )
          .filter((domain, index, route) => domain !== route[index - 1]);
        rayStatus.textContent = `${event.path.kind} · ${event.path.equatorialCrossings} equatorial crossings · ${domains.join(" → ")}. Green: accepted steps, clipped to the map.`;
        break;
      }
      case "inspect-error":
        rayStatus.textContent = event.message;
        break;
      case "starting":
        initialized = false;
        completed = undefined;
        syncStatus();
        retry.hidden = true;
        status.textContent = "Starting…";
        break;
      case "frame":
        completed = event;
        initialized = true;
        retry.hidden = true;
        detail.textContent = `${event.width.toLocaleString("en")} × ${event.height.toLocaleString("en")} pixels`;
        syncStatus();
        break;
      case "error":
        initialized = false;
        completed = undefined;
        exporting = false;
        syncStatus();
        status.textContent = event.message;
        observation.textContent = "Stopped";
        retry.hidden = false;
        save.disabled = true;
        break;
      case "exported": {
        exporting = false;
        if (photoURL) {
          URL.revokeObjectURL(photoURL);
        }
        photoURL = URL.createObjectURL(event.blob);
        const link = document.createElement("a");
        link.href = photoURL;
        link.download = "nullray.png";
        link.click();
        exportStatus.textContent = "Saved as an SDR Display P3 PNG.";
        syncStatus();
        break;
      }
      case "export-error":
        exporting = false;
        exportStatus.textContent = event.message;
        syncStatus();
        break;
      case "disposed":
        break;
    }
  }
  const client = owned.adopt(createRenderClient(canvas, receive), (value) => value.dispose());
  inspectRay.addEventListener(
    "click",
    () => {
      rayStatus.textContent = "Tracing the center ray…";
      client.inspect();
    },
    { signal },
  );
  pickRay.addEventListener(
    "click",
    () => {
      pickingRay = true;
      canvas.style.cursor = "crosshair";
      rayStatus.textContent = "Click the image to pick a ray.";
      settings.hidePopover();
    },
    { signal },
  );
  canvas.addEventListener(
    "pointerdown",
    (event) => {
      if (!pickingRay || event.button !== 0) {
        return;
      }
      event.preventDefault();
      event.stopImmediatePropagation();
      pickingRay = false;
      canvas.style.removeProperty("cursor");
      const bounds = canvas.getBoundingClientRect();
      client.inspect([
        (event.clientX - bounds.left) / bounds.width,
        (event.clientY - bounds.top) / bounds.height,
      ]);
      rayStatus.textContent = "Tracing the picked ray…";
      settings.showPopover();
    },
    { signal, capture: true },
  );
  document.addEventListener(
    "keydown",
    (event) => {
      if (event.key === "Escape" && imageOnly) {
        setImageOnly(false);
      }
      if (event.key === "Escape" && pickingRay) {
        pickingRay = false;
        canvas.style.removeProperty("cursor");
        rayStatus.textContent = "Ray selection cancelled.";
      }
    },
    { signal },
  );
  const update = (time?: number) => {
    revision = client.update(session, hdr(), !document.hidden, width, height, time);
    exporting = false;
    exportStatus.textContent = "";
    syncStatus();
  };
  const dispatch = (action: Action) => {
    const next = transition(session, action);
    feedback.textContent = next.ok ? "" : next.error;
    if (next.ok) {
      session = next.value;
      if (action.type === "restore") {
        completed = undefined;
      }
      update(action.type === "restore" ? action.value.time : undefined);
    }
    syncControls();
  };
  const syncControls = bindControls(root, () => session, dispatch, signal);
  bindNavigation(
    canvas,
    () => session.view.scene,
    (value) => dispatch({ type: "scene", value }),
    () => session.view.navigation,
    () => dispatch({ type: "reset-camera" }),
    signal,
  );
  for (const [id, type] of [
    ["toggle-motion", "toggle-motion"],
    ["photograph", "refine"],
    ["reset-view", "reset-camera"],
  ] as const) {
    button(id).addEventListener("click", () => dispatch({ type }), { signal });
  }
  canvas.addEventListener(
    "keydown",
    (event) => {
      if (event.ctrlKey || event.metaKey || event.altKey) {
        return;
      }
      if (event.key.toLowerCase() === "h" && !event.repeat) {
        event.preventDefault();
        setImageOnly(!imageOnly);
      }
      if (event.code === "Space" && !event.repeat) {
        event.preventDefault();
        dispatch({ type: "toggle-motion" });
      }
    },
    { signal },
  );
  retry.addEventListener(
    "click",
    () => {
      if (!client.retry()) {
        history.replaceState(null, "", encodeView(snapshot()));
        location.reload();
      }
    },
    { signal },
  );
  save.addEventListener(
    "click",
    () => {
      if (save.disabled) {
        return;
      }
      exporting = true;
      exportStatus.textContent = "Preparing PNG…";
      syncStatus();
      client.exportPhoto();
    },
    { signal },
  );

  function openView(fragment: string) {
    const decoded = decodeView(fragment);
    if (decoded.kind === "invalid") {
      feedback.textContent = decoded.error;
      return;
    }
    if (decoded.kind === "view") {
      if (location.hash !== fragment) {
        history.pushState(null, "", fragment);
      }
      dispatch({ type: "restore", value: decoded.value });
    }
  }
  const presetControls = text("presets");
  presetControls.replaceChildren(
    ...presets.map(({ name, description, view }, index) => {
      const preset = document.createElement("button");
      preset.type = "button";
      const number = document.createElement("span");
      number.className = "preset-number";
      number.textContent = String(index + 1).padStart(2, "0");
      number.ariaHidden = "true";
      const title = document.createElement("strong");
      title.textContent = name;
      const caption = document.createElement("small");
      caption.textContent = description;
      preset.append(number, title, caption);
      preset.addEventListener(
        "click",
        () => {
          openView(encodeView(view));
          text("views-panel").hidePopover();
          canvas.focus({ preventScroll: true });
        },
        { signal },
      );
      return preset;
    }),
  );
  bindViews(root, snapshot, openView, signal);
  button("share-view").addEventListener(
    "click",
    () => {
      const url = new URL(location.href);
      url.hash = encodeView(snapshot());
      shareInput.value = url.href;
      copyStatus.textContent = "";
    },
    { signal },
  );
  shareDialog.addEventListener(
    "toggle",
    (event) => {
      if (event.newState === "open") {
        shareInput.select();
      }
    },
    { signal },
  );
  button("copy-view").addEventListener(
    "click",
    () => {
      const link = shareInput.value;
      void navigator.clipboard.writeText(link).then(
        () => {
          if (!signal.aborted && shareInput.value === link) {
            copyStatus.textContent = "Link copied.";
          }
          return undefined;
        },
        () => {
          if (!signal.aborted && shareInput.value === link) {
            copyStatus.textContent = "Select the link and copy it manually.";
          }
          return undefined;
        },
      );
    },
    { signal },
  );
  window.addEventListener("hashchange", () => openView(location.hash), { signal });
  highRange.addEventListener("change", () => update(), { signal });
  document.addEventListener("visibilitychange", () => update(), { signal });
  const resize = new ResizeObserver(([entry]) => {
    const box = entry?.devicePixelContentBoxSize[0];
    if (box && (width !== box.inlineSize || height !== box.blockSize)) {
      width = box.inlineSize;
      height = box.blockSize;
      update();
    }
  });
  owned.defer(() => resize.disconnect());
  resize.observe(canvas, { box: "device-pixel-content-box" });
  syncControls();
  if (!restored) {
    openView(location.hash);
  }
  update(session.view.time);
  const lifetime = owned.move();
  return () => {
    const saved = { ...session, view: snapshot() };
    lifetime.dispose();
    return saved;
  };
}
