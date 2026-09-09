import { expect, test } from "vitest";
import { page } from "vitest/browser";
import markup from "../../index.html?raw";
import { bindViews } from "../../src/explorer/bookmarks.ts";
import { decodeView } from "../../src/model/view.ts";
import { initialAppearance } from "../../src/model/appearance.ts";
import { initialScene } from "../../src/model/scene.ts";

const storageKey = "nullray.views.v1";

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
