import { turnCamera } from "../scene/camera.ts";
import { translateCamera } from "../scene/navigation.ts";
import type { SceneInput, Scene } from "../scene/scene.ts";

/**
 * Bind pointer, wheel, and keyboard placement controls under an abort signal.
 *
 * @remarks
 * Free mode changes camera orientation/placement; orbit mode changes spherical
 * placement. The caller owns the canvas and cancels listeners with the signal.
 *
 * @param change - Commits proposed inputs through scene validation; navigation adds no physical velocity.
 * @param getScene - Reads the latest accepted scene for each input event.
 */
export function bindNavigation(
  canvas: HTMLCanvasElement,
  getScene: () => Scene,
  change: (scene: SceneInput) => void,
  getMode: () => "orbit" | "free",
  reset: () => void,
  signal: AbortSignal,
): void {
  const options = { signal };
  const pointers = new Map<number, readonly [number, number]>();
  const orbit = (horizontal: number, vertical: number) => {
    const scene = getScene();
    if (getMode() === "free") {
      change({ ...scene, camera: turnCamera(scene.camera, horizontal, vertical) });
    } else {
      change({
        ...scene,
        observer: {
          ...scene.observer,
          azimuth: scene.observer.azimuth - horizontal,
          inclination: Math.max(0, Math.min(Math.PI, scene.observer.inclination + vertical)),
        },
      });
    }
  };
  const zoom = (logScale: number) => {
    const scene = getScene();
    const { observer } = scene;
    if (getMode() === "free") {
      change(translateCamera(scene, [0, 0, 1], -logScale * Math.max(1, Math.abs(observer.radius))));
    } else {
      change({
        ...scene,
        observer: {
          ...observer,
          radius: Math.max(-200, Math.min(200, observer.radius * Math.exp(logScale))),
        },
      });
    }
  };
  canvas.addEventListener(
    "pointerdown",
    (event) => {
      if (event.button !== 0 || pointers.size >= 2) {
        return;
      }
      pointers.set(event.pointerId, [event.clientX, event.clientY]);
      canvas.setPointerCapture(event.pointerId);
      canvas.focus({ preventScroll: true });
    },
    options,
  );
  canvas.addEventListener(
    "pointermove",
    (event) => {
      const previous = pointers.get(event.pointerId);
      if (!previous) {
        return;
      }
      const other = pointers.entries().find(([id]) => id !== event.pointerId)?.[1];
      pointers.set(event.pointerId, [event.clientX, event.clientY]);
      if (other) {
        const before = Math.hypot(previous[0] - other[0], previous[1] - other[1]);
        const after = Math.hypot(event.clientX - other[0], event.clientY - other[1]);
        if (before > 0 && after > 0) {
          zoom(Math.log(before / after));
        }
      } else {
        orbit((event.clientX - previous[0]) * 0.005, (event.clientY - previous[1]) * 0.005);
      }
    },
    options,
  );
  const release = ({ pointerId }: PointerEvent) => {
    pointers.delete(pointerId);
    if (canvas.hasPointerCapture(pointerId)) {
      canvas.releasePointerCapture(pointerId);
    }
  };
  canvas.addEventListener("pointerup", release, options);
  canvas.addEventListener("pointercancel", release, options);
  canvas.addEventListener(
    "lostpointercapture",
    ({ pointerId }) => pointers.delete(pointerId),
    options,
  );
  signal.addEventListener(
    "abort",
    () => {
      for (const id of pointers.keys()) {
        if (canvas.hasPointerCapture(id)) {
          canvas.releasePointerCapture(id);
        }
      }
      pointers.clear();
    },
    { once: true },
  );
  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const scale =
        event.deltaMode === WheelEvent.DOM_DELTA_LINE
          ? 16
          : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
            ? canvas.clientHeight
            : 1;
      zoom(Math.max(-0.5, Math.min(0.5, event.deltaY * scale * 0.001)));
    },
    { signal, passive: false },
  );
  const pressed = new Set<string>();
  let movement = 0;
  let previousTime: number | undefined;
  const stop = () => {
    pressed.clear();
    cancelAnimationFrame(movement);
    movement = 0;
    previousTime = undefined;
  };
  const advance = (now: number) => {
    movement = 0;
    if (getMode() !== "free" || pressed.size === 0) {
      stop();
      return;
    }
    const elapsed = previousTime === undefined ? 0 : Math.min(0.05, (now - previousTime) / 1000);
    previousTime = now;
    const scene = getScene();
    const speed = 0.35 * Math.max(1, Math.abs(scene.observer.radius));
    const right = Number(pressed.has("d")) - Number(pressed.has("a"));
    const up = Number(pressed.has("e")) - Number(pressed.has("q"));
    const forward = Number(pressed.has("w")) - Number(pressed.has("s"));
    if (elapsed > 0 && (right !== 0 || up !== 0 || forward !== 0)) {
      change(translateCamera(scene, [right, up, forward], elapsed * speed));
    }
    movement = requestAnimationFrame(advance);
  };
  canvas.addEventListener(
    "keyup",
    (event) => {
      pressed.delete(event.key.toLowerCase());
      if (pressed.size === 0) {
        stop();
      }
    },
    options,
  );
  canvas.addEventListener("blur", stop, options);
  window.addEventListener("blur", stop, options);
  document.addEventListener(
    "visibilitychange",
    () => {
      if (document.hidden) {
        stop();
      }
    },
    options,
  );
  signal.addEventListener("abort", stop, { once: true });
  canvas.addEventListener(
    "keydown",
    (event) => {
      if (event.ctrlKey || event.metaKey || event.altKey) {
        return;
      }
      const key = event.key.toLowerCase();
      if (getMode() === "free" && ["w", "a", "s", "d", "q", "e"].includes(key)) {
        event.preventDefault();
        pressed.add(key);
        if (!movement) {
          movement = requestAnimationFrame(advance);
        }
        return;
      }
      switch (event.key) {
        case "ArrowLeft":
          orbit(-0.04, 0);
          break;
        case "ArrowRight":
          orbit(0.04, 0);
          break;
        case "ArrowUp":
          orbit(0, -0.04);
          break;
        case "ArrowDown":
          orbit(0, 0.04);
          break;
        case "+":
        case "=":
          zoom(-0.1);
          break;
        case "-":
          zoom(0.1);
          break;
        case "r":
        case "R":
        case "Home":
          reset();
          break;
        default:
          return;
      }
      event.preventDefault();
    },
    options,
  );
}
