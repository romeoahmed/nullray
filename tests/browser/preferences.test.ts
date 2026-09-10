import { expect, test } from "vitest";
import { page } from "vitest/browser";
import markup from "../../index.html?raw";
import { bindViews } from "../../src/ui/bookmarks.ts";
import { decodeView } from "../../src/scene/view.ts";
import { initialAppearance } from "../../src/scene/appearance.ts";
import { initialScene } from "../../src/scene/scene.ts";
import { bindAudio } from "../../src/ui/audio.ts";
import { bindControls } from "../../src/ui/controls.ts";
import { initialSession } from "../../src/scene/session.ts";
import type { Action } from "../../src/scene/session.ts";

test("one navigation choice dispatches once and disposed controls stop dispatching", async ({
  onTestFinished,
}) => {
  const template = new DOMParser().parseFromString(markup, "text/html");
  const root = template.querySelector<HTMLElement>("#app");
  if (!root) {
    throw new Error("Missing interface.");
  }
  document.body.append(root);
  const subscription = new AbortController();
  onTestFinished(() => {
    subscription.abort();
    root.remove();
  });
  const actions: Action[] = [];
  bindControls(
    root,
    () => initialSession,
    (action) => actions.push(action),
    subscription.signal,
  )();
  root.querySelector<HTMLElement>("#controls")?.showPopover();
  await page.getByText("Observer Position · motion · camera", { exact: true }).click();
  const navigation = page.getByRole("combobox", { name: "Navigation", exact: true });
  await navigation.selectOptions("free");
  expect(actions).toEqual([{ type: "navigation", value: "free" }]);
  subscription.abort();
  await navigation.selectOptions("orbit");
  expect(actions).toHaveLength(1);
});

const storageKey = "nullray.views.v1";

/** Silent PCM avoids a test soundtrack while exercising native decoding and playback. */
function silentAudio(seconds: number): File {
  const samples = Math.round(seconds * 8000);
  const bytes = new ArrayBuffer(44 + samples * 2);
  const data = new DataView(bytes);
  const label = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++) {
      data.setUint8(offset + i, value.charCodeAt(i));
    }
  };
  label(0, "RIFF");
  data.setUint32(4, bytes.byteLength - 8, true);
  label(8, "WAVEfmt ");
  data.setUint32(16, 16, true);
  data.setUint16(20, 1, true);
  data.setUint16(22, 1, true);
  data.setUint32(24, 8000, true);
  data.setUint32(28, 16000, true);
  data.setUint16(32, 2, true);
  data.setUint16(34, 16, true);
  label(36, "data");
  data.setUint32(40, samples * 2, true);
  return new File([bytes], "silence.wav", { type: "audio/wav" });
}

test("local audio waits for a gesture, plays to completion, and supports replacement and remount", async ({
  onTestFinished,
}) => {
  const template = new DOMParser().parseFromString(markup, "text/html");
  const root = template.querySelector<HTMLElement>("#app");
  if (!root) {
    throw new Error("Missing interface.");
  }
  document.body.append(root);
  let dispose = bindAudio(root);
  onTestFinished(() => {
    dispose();
    root.remove();
  });
  const panel = root.querySelector<HTMLElement>("#controls");
  panel?.showPopover();
  await page.getByText("Audio Local soundtrack", { exact: true }).click();
  const media = root.querySelector<HTMLAudioElement>("#audio-player");
  if (!media) {
    throw new Error("Missing native audio player.");
  }
  const file = page.getByLabelText("Audio file", { exact: true });
  const clear = page.getByRole("button", { name: "Clear audio", exact: true });
  await expect.element(clear).toBeDisabled();
  // A real user gesture calls the public media API; native shadow controls are browser-owned.
  const play = document.createElement("button");
  play.textContent = "Start test soundtrack";
  play.addEventListener("click", () => {
    void media.play();
  });
  media.after(play);
  await file.upload(silentAudio(0.3));
  await expect.poll(() => media.readyState).toBeGreaterThanOrEqual(1);
  expect(media.paused).toBe(true);
  expect(media.controls).toBe(true);
  await page.getByRole("button", { name: "Start test soundtrack" }).click();
  await expect.poll(() => media.ended).toBe(true);
  await expect.element(page.getByText("silence.wav", { exact: true })).toBeVisible();
  const previousURL = media.src;
  await file.upload(silentAudio(10));
  await expect.poll(() => media.duration).toBe(10);
  expect(media.src).not.toBe(previousURL);
  expect(media.paused).toBe(true);
  await page.getByRole("button", { name: "Start test soundtrack" }).click();
  await expect.poll(() => media.paused).toBe(false);
  await clear.click();
  expect(media.paused).toBe(true);
  expect(media.hasAttribute("src")).toBe(false);
  expect(media.hidden).toBe(true);
  await expect.element(clear).toBeDisabled();
  dispose();
  dispose = bindAudio(root);
  await file.upload(silentAudio(0.3));
  await expect.poll(() => media.readyState).toBeGreaterThanOrEqual(1);
  await page.getByRole("button", { name: "Start test soundtrack" }).click();
  await expect.poll(() => media.ended).toBe(true);
});

test("saved views persist across mounts, reject duplicates, restore snapshots, and undo deletion", async ({
  onTestFinished,
}) => {
  const previous = localStorage.getItem(storageKey);
  localStorage.removeItem(storageKey);
  const template = new DOMParser().parseFromString(markup, "text/html");
  const root = template.querySelector<HTMLElement>("#app");
  if (!root) {
    throw new Error("Missing interface.");
  }
  document.body.append(root);
  let subscription = new AbortController();
  let restored = "";
  const snapshot = {
    scene: initialScene,
    appearance: initialAppearance,
    time: 37,
    exposureEV: -2,
    whiteBalance: 4800,
    bloom: 0.2,
    navigation: "orbit",
    display: "sdr",
    diagnostic: "image",
    analyzer: null,
  } as const;
  const bind = () =>
    bindViews(
      root,
      () => snapshot,
      (fragment) => {
        restored = fragment;
      },
      subscription.signal,
    );
  bind();
  onTestFinished(() => {
    subscription.abort();
    root.remove();
    if (previous === null) {
      localStorage.removeItem(storageKey);
    } else {
      localStorage.setItem(storageKey, previous);
    }
  });
  await page.getByRole("button", { name: "Views", exact: true }).click();
  await page.getByText("Saved views On this browser", { exact: true }).click();
  const name = page.getByRole("textbox", { name: "Name this view", exact: true });
  const save = page.getByRole("button", { name: "Save current view", exact: true });
  await name.fill("Wide view");
  await save.click();
  await expect.element(page.getByText("Saved “Wide view” on this browser.")).toBeVisible();
  const persisted = localStorage.getItem(storageKey);
  await name.fill("Wide view");
  await save.click();
  expect(localStorage.getItem(storageKey)).toBe(persisted);
  await expect
    .element(page.getByText("That name is already saved. Choose another name."))
    .toBeVisible();
  subscription.abort();
  subscription = new AbortController();
  bind();
  await page.getByRole("button", { name: "Open saved view", exact: true }).click();
  expect(decodeView(restored)).toEqual({ kind: "view", value: snapshot });
  await page.getByRole("button", { name: "Delete saved view", exact: true }).click();
  await expect
    .element(page.getByRole("button", { name: "Open saved view", exact: true }))
    .toBeDisabled();
  await page.getByRole("button", { name: "Undo deletion", exact: true }).click();
  expect(localStorage.getItem(storageKey)).toBe(persisted);
  localStorage.setItem(storageKey, "malformed");
  await name.fill("Another view");
  await save.click();
  expect(localStorage.getItem(storageKey)).toBe("malformed");
  await expect
    .element(
      page.getByText("Saved views could not be read or written. Use Share view to keep a link."),
    )
    .toBeVisible();
});
