import type { SpacetimeBlock } from "../../physics/atlas.ts";
import { heatingProfile, plasmaProfile } from "../../physics/plasma.ts";
import type { Scene } from "../../scene/scene.ts";
import type { OpticalSettings } from "./engine.ts";

/** Uniform block representation shared with the native horizon atlas. */
function blockWords(block: SpacetimeBlock): readonly [number, number, number, number] {
  switch (block.kind) {
    case "exterior":
      return [0, block.universe, block.side, 0];
    case "black-hole":
      return [1, block.universe, 0, 0];
    case "interior":
      return [2, block.universe, block.side, 0];
    case "white-hole":
      return [3, block.universe, 0, 0];
    case "naked":
      return [4, 0, 1, 0];
    case "disconnected":
      return [5, 0, 1, 0];
    default:
      throw new Error("Unknown spacetime block.");
  }
}

/**
 * Write the 288-byte optical uniform ABI into caller-owned storage.
 *
 * @remarks
 * Both views are mutated and must alias. Inputs are already validated; this
 * function does not upload to the GPU. Field order matches imaging/frame.wgsl.
 *
 * @param values - f32 view covering 72 lanes at the start of the frame storage.
 * @param words - i32 view covering the same bytes and byte offset for block metadata.
 * @param jetScale - Prepared dimensionless jet-table frequency scale.
 */
export function writeOpticalFrame(
  values: Float32Array,
  words: Int32Array,
  scene: Scene,
  { appearance, time, view, jitter }: OpticalSettings,
  jetScale: number,
): void {
  const { geometry: g, frame, block } = scene.prepared;
  values.set([scene.space.spin, scene.space.charge, scene.disk.inner, scene.disk.outer]);
  values.set(
    [g.point.radius, g.point.inclination, g.point.azimuth, g.point.chart === "ingoing" ? 1 : -1],
    4,
  );
  values.set([...frame.velocity, ...frame.radial, ...frame.polar, ...frame.azimuthal], 8);
  values.set([...scene.camera.right, 0, ...scene.camera.up, 0, ...scene.camera.forward, 0], 24);
  values.set(
    [
      time + scene.prepared.timeOffset,
      appearance.diskTemperature,
      appearance.diskStructure,
      appearance.skyBrightness,
    ],
    36,
  );
  words.set(blockWords(block), 40);
  values.set(
    [
      jitter[0],
      jitter[1],
      Math.tan(scene.observer.fieldOfView / 2),
      view === "frequency" ? 1 : view === "order" ? 2 : view === "domain" ? 3 : 0,
    ],
    44,
  );
  // f32 optical tolerance and a bounded dispatch; material quadrature uses its own physical scales.
  values.set([4e-5, 2048, 0, 0], 48);
  if (scene.jet) {
    values.set(
      [scene.jet.inner, scene.jet.outer, Math.cos(scene.jet.openingAngle), scene.jet.speed],
      52,
    );
    values.set([jetScale, 0, 0, 1], 56);
  } else {
    values.fill(0, 52, 60);
  }
  if (scene.plasma) {
    const profile = plasmaProfile(scene.plasma);
    values.set(
      [profile.amplitude, profile.scaleSquared, scene.plasma.frequencyGHz * 0.04799243073366221, 1],
      60,
    );
  } else {
    values.set([0, 1, 0, 0], 60);
  }
  const heat = scene.plasma && heatingProfile(scene.plasma);
  values.set(heat ? [heat.inner, heat.outer, heat.contrast, heat.frequency] : [0, 0, 0, 0], 64);
  values.set(
    [
      appearance.diskThickness,
      appearance.diskOpticalDepth,
      appearance.otherUniverses,
      scene.prepared.disk.columnNormalization,
    ],
    68,
  );
}
