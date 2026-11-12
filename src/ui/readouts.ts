import { horizons, stationaryLimits } from "../physics/geometry.ts";
import type { Horizons } from "../physics/geometry.ts";
import type { SavedView } from "../scene/view.ts";
import { element } from "./elements.ts";
import { drawTopology } from "./topology.ts";

const diagnosticDescriptions = {
  frequency: "Red: blueshift · Blue: redshift · Green: unchanged frequency",
  order: "Hue cycles with the number of equatorial crossings along the backward ray.",
  polarization:
    "Linear polarization fraction: black 0%, white 100% before exposure. Stokes intensities are averaged before measurement.",
  angle:
    "Detector orientation: red 0°, green 60°, blue 120°. Brightness follows linear polarization degree.",
  domain:
    "Hue identifies the exterior universe; bright colors show disk hits, dim colors the sky. Purple: negative-radius sky · Dark gray: singularity · Blue-gray: source-free characteristic.",
  image:
    "Inspect views mark unresolved samples in pink. Photographs report missing samples separately.",
} satisfies Record<SavedView["diagnostic"], string>;

function observationName({ scene }: SavedView): string {
  switch (scene.prepared.block.kind) {
    case "black-hole":
    case "interior":
      return "Black-hole interior";
    case "white-hole":
      return "White-hole region";
    case "naked":
      return "Horizonless geometry";
    case "exterior":
    case "disconnected":
      return scene.plasma
        ? "Refractive field"
        : scene.jet
          ? "Disk & polar outflow"
          : "Thermal disk";
  }
}

function describeHorizons(structure: Horizons): string {
  switch (structure.kind) {
    case "pair":
      return `Horizons r− ${structure.inner.toFixed(3)}, r+ ${structure.outer.toFixed(3)}`;
    case "extremal":
      return "Extremal horizon r = 1";
    case "single":
      return "Horizon r = 2";
    case "none":
      return "Naked singularity · no horizons";
  }
}

/**
 * Bind view descriptions independently of per-frame progress telemetry.
 *
 * @returns A synchronizer that compares view identity/HDR state and updates the
 * borrowed DOM. Scene changes clear the previous selected-ray diagram.
 */
export function bindReadouts(root: HTMLElement): (view: SavedView, hdr: boolean) => void {
  const text = (id: string) => element(root, `#${id}`, HTMLElement);
  const displayStatus = text("display-status");
  const navigationHint = text("navigation-hint");
  const diagnosticStatus = text("diagnostic-status");
  const rayStatus = text("ray-status");
  const topologyMap = element(root, "#topology-map", SVGSVGElement);
  let mappedScene: SavedView["scene"] | undefined;
  let previous: SavedView | undefined;
  let previousHDR: boolean | undefined;
  return (view, hdr) => {
    if (previous === view && previousHDR === hdr) {
      return;
    }
    previous = view;
    previousHDR = hdr;
    displayStatus.textContent = hdr ? "HDR · extended highlights" : "SDR · standard highlights";
    navigationHint.textContent =
      view.navigation === "free"
        ? "Drag to look · WASD to move · Q/E down/up · R to reset"
        : "Drag to orbit · Scroll or pinch to approach · Arrow keys to turn";
    const { space, prepared } = view.scene;
    text("observation-name").textContent = observationName(view);
    if (mappedScene !== view.scene) {
      drawTopology(topologyMap, view.scene);
      rayStatus.textContent = "Trace the center or pick an image ray to inspect its source domain.";
      mappedScene = view.scene;
    }
    const structure = horizons(space);
    const limits = stationaryLimits(space, prepared.geometry.point.inclination);
    const horizonText = describeHorizons(structure);
    const block = prepared.block;
    const blockText =
      block.kind === "naked"
        ? "horizonless block"
        : block.kind === "disconnected"
          ? "disconnected negative-radius component"
          : `${block.kind} · universe ${block.universe}${"side" in block ? ` · side ${block.side}` : ""}`;
    text("topology-status").textContent =
      `${horizonText}. ${limits ? `Stationary limits at this latitude: ${limits[0].toFixed(3)}, ${limits[1].toFixed(3)}. ` : "No stationary limit at this latitude. "}Observer r = ${prepared.geometry.point.radius.toFixed(3)} · ${blockText}. These surfaces do not emit light.`;
    diagnosticStatus.textContent = diagnosticDescriptions[view.diagnostic];
  };
}
