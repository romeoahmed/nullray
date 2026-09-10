import { expect, test } from "vitest";
import { page } from "vitest/browser";
import markup from "../../index.html?raw";
import { bindViews } from "../../src/ui/bookmarks.ts";
import { decodeView } from "../../src/scene/view.ts";
import { initialAppearance } from "../../src/scene/appearance.ts";
import { initialScene } from "../../src/scene/scene.ts";
import { bindAudio } from "../../src/ui/audio.ts";

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
  const play = page.getByRole("button", { name: "Play audio", exact: true });
  const file = page.getByLabelText("Audio file", { exact: true });
  await expect.element(play).toBeDisabled();
  await file.upload(silentAudio(0.3));
  await expect.element(play).toBeEnabled();
  await play.click();
  await expect
    .element(page.getByRole("button", { name: "Pause audio", exact: true }))
    .toBeVisible();
  await expect.element(play).toBeVisible();
  await expect.element(page.getByText("silence.wav", { exact: true })).toBeVisible();
  await file.upload(silentAudio(10));
  await play.click();
  await page.getByRole("button", { name: "Pause audio", exact: true }).click();
  await expect.element(play).toBeVisible();
  const volume = page.getByRole("slider", { name: "Volume", exact: true });
  await volume.fill("0.25");
  await expect.element(page.getByText("25%", { exact: true })).toBeVisible();
  await play.click();
  await page.getByRole("button", { name: "Clear audio", exact: true }).click();
  await expect.element(play).toBeDisabled();
  dispose();
  dispose = bindAudio(root);
  await file.upload(silentAudio(0.3));
  await play.click();
  await expect
    .element(page.getByRole("button", { name: "Pause audio", exact: true }))
    .toBeVisible();
  await expect.element(play).toBeVisible();
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
    bloom: 0.2,
    navigation: "orbit",
    display: "sdr",
    diagnostic: "image",
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
