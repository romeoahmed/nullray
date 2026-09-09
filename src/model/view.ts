import { createCamera, initialCamera } from "./camera.ts";
import { createAppearance, initialAppearance } from "./appearance.ts";
import type { SourceAppearance } from "./appearance.ts";
import { createScene, initialScene } from "./scene.ts";
import type { Scene } from "./scene.ts";

/** Versioned visual snapshot. Opening a link freezes emission at the recorded epoch. */
export interface SavedView {
  readonly scene: Scene;
  readonly appearance: SourceAppearance;
  readonly exposureEV: number;
  readonly bloom: number;
  readonly navigation: "orbit" | "free";
  readonly display: "auto" | "hdr" | "sdr";
  /** Observer coordinate epoch in geometric time units; nonnegative and finite in f32. */
  readonly time: number;
  readonly diagnostic: "image" | "frequency" | "order";
}

/** Shared starting view for the session and curated presets. */
export const initialView: SavedView = {
  scene: initialScene,
  appearance: initialAppearance,
  exposureEV: 0,
  bloom: 0.08,
  navigation: "orbit",
  display: "auto",
  diagnostic: "image",
  time: 0,
};

/** Distinguish an absent view from an invalid fragment and a fully validated snapshot. */
export type ViewResult =
  | { readonly kind: "empty" }
  | { readonly kind: "invalid"; readonly error: string }
  | { readonly kind: "view"; readonly value: SavedView };

const record = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);
const finite = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

/** Encode only source inputs; derived disk geometry is recomputed when a view is opened. */
export function encodeView(view: SavedView): string {
  return `#view=${encodeURIComponent(
    JSON.stringify({
      version: 4,
      camera: { forward: view.scene.camera.forward, up: view.scene.camera.up },
      navigation: view.navigation,
      bloom: view.bloom,
      appearance: view.appearance,
      space: view.scene.space,
      observer: view.scene.observer,
      exposureEV: view.exposureEV,
      display: view.display,
      time: view.time,
      diagnostic: view.diagnostic,
    }),
  )}`;
}

/** Treat URL content as unknown and validate the complete snapshot before applying any of it. */
export function decodeView(fragment: string): ViewResult {
  if (!fragment || fragment === "#") {
    return { kind: "empty" };
  }
  const invalid: ViewResult = {
    kind: "invalid",
    error: "This view link is invalid or uses an unsupported version.",
  };
  if (!fragment.startsWith("#view=") || fragment.length > 4096) {
    return invalid;
  }
  let data: unknown;
  try {
    data = JSON.parse(decodeURIComponent(fragment.slice(6)));
  } catch {
    return invalid;
  }
  if (
    !record(data) ||
    (data.version !== 1 && data.version !== 2 && data.version !== 3 && data.version !== 4) ||
    !record(data.space) ||
    !record(data.observer)
  ) {
    return invalid;
  }
  const { spin, charge } = data.space;
  const { radius, inclination, azimuth, fieldOfView } = data.observer;
  const { exposureEV, display, time, diagnostic } = data;
  const bloom = data.version >= 3 ? data.bloom : 0;
  const navigation = data.version === 4 ? data.navigation : "orbit";
  const camera = data.version === 4 ? createCamera(data.camera) : initialCamera;
  if (
    !finite(spin) ||
    !finite(charge) ||
    !finite(radius) ||
    !finite(inclination) ||
    !finite(azimuth) ||
    !finite(fieldOfView) ||
    !camera ||
    (navigation !== "orbit" && navigation !== "free") ||
    !finite(bloom) ||
    bloom < 0 ||
    bloom > 1 ||
    !finite(exposureEV) ||
    !finite(time) ||
    exposureEV < -6 ||
    exposureEV > 6 ||
    time < 0 ||
    !Number.isFinite(Math.fround(time)) ||
    (display !== "auto" && display !== "hdr" && display !== "sdr") ||
    (diagnostic !== "image" && diagnostic !== "frequency" && diagnostic !== "order")
  ) {
    return invalid;
  }
  const appearance = createAppearance(data.version === 1 ? initialAppearance : data.appearance);
  if (!appearance.ok) {
    return invalid;
  }
  const scene = createScene({
    camera,
    space: { spin, charge },
    observer: { radius, inclination, azimuth, fieldOfView },
  });
  if (!scene.ok) {
    return { kind: "invalid", error: `Cannot open this view: ${scene.error}` };
  }
  return {
    kind: "view",
    value: {
      scene: scene.value,
      appearance: appearance.value,
      exposureEV,
      bloom,
      navigation,
      display,
      time,
      diagnostic,
    },
  };
}
