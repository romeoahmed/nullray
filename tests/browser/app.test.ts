import { expect, test } from "vitest";
import { page, userEvent } from "vitest/browser";
import markup from "../../index.html?raw";
import { mountApp } from "../../src/ui/app.ts";
import { initialSession } from "../../src/scene/session.ts";
import { decodeView } from "../../src/scene/view.ts";
import { element } from "../../src/ui/elements.ts";
import "../../src/ui/style.css";

function createRoot() {
  const template = new DOMParser().parseFromString(markup, "text/html");
  const root = element(template, "#app", HTMLElement);
  document.body.append(root);
  return root;
}
const button = (name: string) => page.getByRole("button", { name, exact: true });
const select = (name: string) => page.getByRole("combobox", { name, exact: true });
const slider = (name: string) => page.getByRole("slider", { name, exact: true });

/** Capture settled native panel transitions for visual review. */
async function screenshot(root: HTMLElement, name: string) {
  // Native popover removal cancels its exit transition; cancellation also settles the visual state.
  await Promise.allSettled(
    root.getAnimations({ subtree: true }).map((animation) => animation.finished),
  );
  await page.screenshot({ path: `../../test-results/${name}.png` });
}

test("desktop and landscape panels preserve keyboard focus and stay within the viewport", async ({
  onTestFinished,
}) => {
  await page.viewport(1440, 900);
  const root = createRoot();
  const dispose = mountApp(root, { ...initialSession, motion: "paused" });
  onTestFinished(() => {
    dispose();
    root.remove();
  });
  await expect.element(page.getByText(/^Kerr–Newman ·/)).toBeVisible();
  await expect.element(page.getByText("Paused · image settled", { exact: true })).toBeVisible();
  await screenshot(root, "explorer-desktop");
  await button("Hide controls").click();
  await expect.element(element(root, "#toggle-controls", HTMLButtonElement)).not.toBeVisible();
  await expect.element(button("Show controls")).toBeVisible();
  await userEvent.keyboard("{Escape}");
  await expect.element(button("Settings")).toBeVisible();
  await expect
    .element(page.getByRole("button", { name: "Hide controls", exact: true }))
    .toHaveAttribute("aria-pressed", "false");
  await button("Settings").click();
  await expect.element(slider("Charge")).toBeVisible();
  await screenshot(root, "explorer-desktop-settings");
  await userEvent.keyboard("{Escape}");
  await expect.element(button("Settings")).toHaveFocus();
  await button("Share view").click();
  await expect.element(page.getByRole("textbox", { name: "Link", exact: true })).toHaveFocus();
  await button("Close view link").click();
  await expect.element(button("Share view")).toHaveFocus();
  await page.viewport(844, 390);
  await button("Settings").click();
  await page.getByText("Observer Position · motion · camera", { exact: true }).click();
  const panel = element(root, "#controls", HTMLElement);
  const bounds = panel.getBoundingClientRect();
  expect(bounds.left).toBeGreaterThanOrEqual(0);
  expect(bounds.top).toBeGreaterThanOrEqual(0);
  expect(bounds.right).toBeLessThanOrEqual(window.innerWidth);
  expect(bounds.bottom).toBeLessThanOrEqual(window.innerHeight);
  expect(panel.scrollWidth).toBeLessThanOrEqual(panel.clientWidth);
  await screenshot(root, "explorer-landscape");
}, 30_000);

test("mobile controls keep the canvas accessible and restore a shared scene atomically", async ({
  onTestFinished,
}) => {
  await page.viewport(390, 844);
  const initialHash = location.hash;
  const root = createRoot();
  const dispose = mountApp(root, { ...initialSession, motion: "paused", resolution: 1 });
  onTestFinished(() => {
    dispose();
    root.remove();
    history.replaceState(null, "", initialHash || location.pathname);
  });
  await expect.element(page.getByText(/^Kerr–Newman ·/)).toBeVisible();
  const canvas = element(root, "canvas", HTMLCanvasElement);
  await expect.poll(() => canvas.width).toBe(Math.round(390 * devicePixelRatio));
  expect(document.documentElement.scrollWidth).toBeLessThanOrEqual(window.innerWidth);
  await screenshot(root, "explorer-mobile");
  await button("Settings").click();
  await screenshot(root, "explorer-mobile-settings");
  await page.getByText("Image Color · detail · highlights", { exact: true }).click();
  await expect.element(select("Exploration detail")).toHaveValue("1");
  await select("Exploration detail").selectOptions("0.25");
  await expect.poll(() => canvas.width).toBe(Math.floor(390 * devicePixelRatio * 0.25));
  await select("Exploration detail").selectOptions("auto");
  await expect.poll(() => canvas.width).toBe(Math.round(390 * devicePixelRatio));
  await select("Exploration detail").selectOptions("0.5");
  await expect.poll(() => canvas.width).toBe(Math.floor(390 * devicePixelRatio * 0.5));
  await page.getByText("Observer Position · motion · camera", { exact: true }).click();
  await select("Navigation").selectOptions("free");
  await button("Close settings").click();
  const distance = element(root, "#distance", HTMLInputElement);
  const initialDistance = distance.valueAsNumber;
  canvas.focus();
  await userEvent.keyboard("{w>}");
  await expect.poll(() => distance.valueAsNumber).toBeLessThan(initialDistance);
  window.dispatchEvent(new Event("blur"));
  const stopped = distance.value;
  await new Promise<void>((resolve) =>
    requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
  );
  expect(distance.value).toBe(stopped);
  await userEvent.keyboard("{/w}");
  await button("Reset view").click();
  await button("Settings").click();
  const disk = initialSession.view.scene.disk;
  const materialRadius = String((disk.inner + disk.outer) / 2);
  distance.value = materialRadius;
  distance.dispatchEvent(new Event("input", { bubbles: true }));
  const inclination = element(root, "#inclination", HTMLInputElement);
  const before = inclination.value;
  inclination.value = "90";
  inclination.dispatchEvent(new Event("input", { bubbles: true }));
  await expect
    .element(
      page.getByText("The observer cannot lie on the emitting disk surface.", { exact: true }),
    )
    .toBeVisible();
  expect(inclination.value).toBe(before);
  const feedback = element(root, "#feedback", HTMLElement);
  const feedbackBounds = feedback.getBoundingClientRect();
  expect(
    document.elementFromPoint(
      feedbackBounds.x + feedbackBounds.width / 2,
      feedbackBounds.y + feedbackBounds.height / 2,
    ),
  ).toBe(feedback);
  await slider("Exposure").click();
  await userEvent.keyboard("{End}");
  await expect.element(slider("Exposure")).toHaveValue("6");
  await button("Close settings").click();
  await button("Share view").click();
  await expect.element(page.getByRole("dialog", { name: "View link" })).toBeVisible();
  const link = new URL(element(root, "#view-link", HTMLInputElement).value);
  const saved = decodeView(link.hash);
  expect(saved.kind).toBe("view");
  if (saved.kind === "view") {
    expect(saved.value.exposureEV).toBe(6);
  }
  await userEvent.keyboard("{Escape}");
  await button("Views").click();
  await page.getByRole("button", { name: /Classic disk/ }).click();
  expect(element(root, "#views-panel", HTMLElement).matches(":popover-open")).toBe(false);
  expect(document.activeElement).toBe(canvas);
  await expect.poll(() => distance.value).toBe(String(initialSession.view.scene.observer.radius));
  location.hash = link.hash;
  await expect.poll(() => distance.value).toBe(materialRadius);
  expect(element(root, "#exposure", HTMLInputElement).value).toBe("6");
  location.hash = "#view=invalid";
  await expect
    .element(page.getByText("This view link is invalid or uses an unsupported version."))
    .toBeVisible();
  expect(distance.value).toBe(materialRadius);
  await button("Settings").click();
  const panel = element(root, "#controls", HTMLElement);
  expect(panel.scrollWidth).toBeLessThanOrEqual(panel.clientWidth);
  const bounds = panel.getBoundingClientRect();
  expect(bounds.left).toBeGreaterThanOrEqual(0);
  expect(bounds.right).toBeLessThanOrEqual(window.innerWidth);
  for (const width of [320, 390]) {
    // Each viewport change completes before measuring its native layout.
    // oxlint-disable-next-line no-await-in-loop
    await page.viewport(width, 844);
    for (const control of root.querySelectorAll<HTMLButtonElement>(".toolbar button")) {
      expect(control.scrollWidth).toBeLessThanOrEqual(control.clientWidth);
      expect(control.getBoundingClientRect().width).toBeGreaterThanOrEqual(44);
    }
    expect(
      element(root, ".toolbar", HTMLElement).getBoundingClientRect().right,
    ).toBeLessThanOrEqual(width);
  }
  await screenshot(root, "explorer-mobile-feedback");
}, 30_000);

test("photographs finish at native resolution and presentation edits preserve completed samples", async ({
  onTestFinished,
}) => {
  await page.viewport(240, 180);
  const root = createRoot();
  const dispose = mountApp(root, { ...initialSession, motion: "paused", resolution: 0.5 });
  onTestFinished(() => {
    dispose();
    root.remove();
  });
  const canvas = element(root, "canvas", HTMLCanvasElement);
  await expect.poll(() => canvas.width).toBe(Math.floor(240 * devicePixelRatio * 0.5));
  await button("Refine still image").click();
  await expect.element(page.getByText("Photograph ready", { exact: true })).toBeVisible();
  await expect.element(button("Save photograph")).toBeEnabled();
  expect(canvas.width).toBe(Math.round(240 * devicePixelRatio));
  expect(element(root, "#sampling-status", HTMLElement).textContent).toMatch(
    /All sampled rays resolved|pixels contain unresolved/,
  );
  const exposure = element(root, "#exposure", HTMLInputElement);
  exposure.value = "2";
  exposure.dispatchEvent(new Event("input", { bubbles: true }));
  await expect.element(button("Save photograph")).toBeEnabled();
  const progress = element(root, "#photo-progress", HTMLProgressElement);
  expect(progress.value).toBe(progress.max);
}, 30_000);

test("remount preserves the paused scene and resized surfaces resume rendering", async ({
  onTestFinished,
}) => {
  await page.viewport(320, 240);
  const root = createRoot();
  let dispose = mountApp(root, { ...initialSession, motion: "paused" });
  onTestFinished(() => {
    dispose();
    root.remove();
  });
  await expect.element(page.getByText(/^Kerr–Newman ·/)).toBeVisible();
  const saved = dispose();
  dispose = mountApp(root, saved);
  const canvas = element(root, "canvas", HTMLCanvasElement);
  await expect
    .poll(() => canvas.width)
    .toBe(
      Math.floor(320 * devicePixelRatio * (saved.resolution === "auto" ? 1 : saved.resolution)),
    );
  await expect.element(button("Resume motion")).toBeVisible();
  canvas.style.display = "none";
  await new Promise<void>((resolve) =>
    requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
  );
  canvas.style.width = "224px";
  canvas.style.display = "";
  await expect
    .poll(() => canvas.width)
    .toBe(
      Math.floor(224 * devicePixelRatio * (saved.resolution === "auto" ? 1 : saved.resolution)),
    );
  await page.viewport(480, 320);
  canvas.style.width = "100%";
  await expect
    .poll(() => canvas.width)
    .toBe(
      Math.floor(480 * devicePixelRatio * (saved.resolution === "auto" ? 1 : saved.resolution)),
    );
});
