import { finite, record } from "./decode.ts";
import { decodeJet } from "./jet.ts";
import { decodePlasma } from "./plasma.ts";
import { createCamera } from "./camera.ts";
import { createAppearance, initialAppearance } from "./appearance.ts";
import type { SourceAppearance } from "./appearance.ts";
import { createScene, initialScene } from "./scene.ts";
import type { Scene } from "./scene.ts";
import { decodePhysicalObserver } from "./preparation.ts";

/** Versioned visual snapshot. Opening a link freezes emission at the recorded epoch. */
export interface SavedView {
  readonly scene: Scene;
  readonly appearance: SourceAppearance;
  readonly exposureEV: number;
  /** Photographic neutral blackbody in kelvin; independent of source temperature. */
  readonly whiteBalance: number;
  readonly bloom: number;
  readonly navigation: "orbit" | "free";
  readonly display: "auto" | "hdr" | "sdr";
  /** Observer coordinate epoch in geometric time units; nonnegative and finite in f32. */
  readonly time: number;
  readonly analyzer: number | null;
  readonly diagnostic: "image" | "frequency" | "order" | "domain" | "polarization" | "angle";
}

/** Shared starting view for the session and curated presets. */
export const initialView: SavedView = {
  scene: initialScene,
  appearance: initialAppearance,
  exposureEV: 6,
  whiteBalance: 4500,
  bloom: 0.3,
  navigation: "orbit",
  display: "auto",
  diagnostic: "image",
  analyzer: null,
  time: 0,
};

/** Distinguish an absent view from an invalid fragment and a fully validated snapshot. */
export type ViewResult =
  | { readonly kind: "empty" }
  | { readonly kind: "invalid"; readonly error: string }
  | { readonly kind: "view"; readonly value: SavedView };

/** Encode only source inputs; derived disk geometry is recomputed when a view is opened. */
export function encodeView(view: SavedView): string {
  return `#view=${encodeURIComponent(
    JSON.stringify({
      version: 1,
      camera: { forward: view.scene.camera.forward, up: view.scene.camera.up },
      navigation: view.navigation,
      bloom: view.bloom,
      appearance: view.appearance,
      space: view.scene.space,
      observer: view.scene.observer,
      disk: view.scene.disk,
      jet: view.scene.jet,
      plasma: view.scene.plasma,
      exposureEV: view.exposureEV,
      whiteBalance: view.whiteBalance,
      display: view.display,
      time: view.time,
      diagnostic: view.diagnostic,
      analyzer: view.analyzer,
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
    data.version !== 1 ||
    !record(data.space) ||
    !record(data.observer) ||
    !record(data.disk)
  ) {
    return invalid;
  }
  const { spin, charge } = data.space;
  const { radius, inclination, azimuth, fieldOfView, chart } = data.observer;
  const motion = decodePhysicalObserver(data.observer.motion);
  const { inner, outer } = data.disk;
  const jet = decodeJet(data.jet);
  const plasma = decodePlasma(data.plasma);
  const { exposureEV, whiteBalance, display, time, diagnostic, bloom, navigation, analyzer } = data;
  const camera = createCamera(data.camera);
  if (
    jet === undefined ||
    plasma === undefined ||
    !finite(spin) ||
    !finite(charge) ||
    !finite(radius) ||
    !finite(inclination) ||
    !finite(azimuth) ||
    !finite(fieldOfView) ||
    !motion ||
    (chart !== "ingoing" && chart !== "outgoing") ||
    !finite(inner) ||
    !finite(outer) ||
    !camera ||
    (navigation !== "orbit" && navigation !== "free") ||
    !finite(bloom) ||
    bloom < 0 ||
    bloom > 1 ||
    (analyzer !== null && (!finite(analyzer) || analyzer < 0 || analyzer >= Math.PI)) ||
    !finite(exposureEV) ||
    !finite(whiteBalance) ||
    whiteBalance < 2500 ||
    whiteBalance > 12000 ||
    !finite(time) ||
    exposureEV < -6 ||
    exposureEV > 6 ||
    time < 0 ||
    !Number.isFinite(Math.fround(time)) ||
    (display !== "auto" && display !== "hdr" && display !== "sdr") ||
    (diagnostic !== "image" &&
      diagnostic !== "frequency" &&
      diagnostic !== "order" &&
      diagnostic !== "domain" &&
      diagnostic !== "polarization" &&
      diagnostic !== "angle")
  ) {
    return invalid;
  }
  const appearance = createAppearance(data.appearance);
  if (!appearance.ok) {
    return invalid;
  }
  const scene = createScene({
    camera,
    space: { spin, charge },
    observer: { radius, inclination, azimuth, fieldOfView, chart, motion },
    disk: { inner, outer },
    jet,
    plasma,
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
      whiteBalance,
      bloom,
      navigation,
      display,
      time,
      diagnostic,
      analyzer,
    },
  };
}
