import type { Resolution } from "../../scene/session.ts";

/** Initial pixel budget for explicit Auto playback; native/fixed choices bypass it. */
export const initialPixelBudget = 640 * 360;

/**
 * Adjust Auto's pixel budget from a completed timing window.
 *
 * @param pixels - Positive rendered pixel count, used as the adaptation baseline.
 * @param milliseconds - Mean render/submission-completion wall time, not a GPU timestamp.
 * @returns The previous budget for rejected timing or the dead band; otherwise a bounded budget.
 * The worker controls the measurement window and excludes fixed-resolution frames.
 */
export function nextPixelBudget(budget: number, pixels: number, milliseconds: number): number {
  if (!Number.isFinite(milliseconds) || milliseconds <= 0 || pixels <= 0) {
    return budget;
  }
  // A broad dead band avoids resizing around normal scheduling noise. This is
  // a responsiveness policy, not a GPU timer or a guaranteed frame rate.
  if (milliseconds >= 24 && milliseconds <= 42) {
    return budget;
  }
  const ratio = Math.max(0.75, Math.min(1.25, 33 / milliseconds));
  return Math.max(160 * 90, Math.min(1920 * 1080, pixels * ratio));
}

/** Fixed choices remain exact; only Auto playback adapts, and Auto pauses at native detail. */
export function resolutionScale(
  resolution: Resolution,
  playing: boolean,
  width: number,
  height: number,
  budget: number,
): number {
  if (resolution !== "auto") {
    return resolution;
  }
  return playing ? Math.min(1, Math.sqrt(budget / Math.max(1, width * height))) : 1;
}
