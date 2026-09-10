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
  await page.screenshot({ path: "../../test-results/explorer-desktop.png" });
  await button("Settings").click();
  await expect.element(slider("Charge")).toBeVisible();
  await expect.element(slider("Spin")).toHaveAttribute("aria-valuetext", "0.70");
  await page.screenshot({ path: "../../test-results/explorer-desktop-settings.png" });
  await userEvent.keyboard("{Escape}");
  await expect.element(button("Settings")).toHaveFocus();
  await button("Share view").click();
  await expect.element(page.getByRole("textbox", { name: "Link", exact: true })).toHaveFocus();
  await button("Close view link").click();
  await expect.element(button("Share view")).toHaveFocus();
  await page.viewport(844, 390);
  await button("Settings").click();
  await page.getByText("Camera Position & perspective", { exact: true }).click();
  const panel = element(root, "#controls", HTMLElement);
  const bounds = panel.getBoundingClientRect();
  expect(bounds.left).toBeGreaterThanOrEqual(0);
  expect(bounds.top).toBeGreaterThanOrEqual(0);
  expect(bounds.right).toBeLessThanOrEqual(window.innerWidth);
  expect(bounds.bottom).toBeLessThanOrEqual(window.innerHeight);
  expect(panel.scrollWidth).toBeLessThanOrEqual(panel.clientWidth);
  await page.screenshot({ path: "../../test-results/explorer-landscape.png" });
}, 30_000);

test("mobile controls keep the canvas accessible and restore a shared scene atomically", async ({
  onTestFinished,
}) => {
  await page.viewport(390, 844);
  const initialHash = location.hash;
  const root = createRoot();
  const dispose = mountApp(root, { ...initialSession, motion: "paused" });
  onTestFinished(() => {
    dispose();
    root.remove();
    history.replaceState(null, "", initialHash || location.pathname);
  });
  await expect.element(page.getByText(/^Kerr–Newman ·/)).toBeVisible();
  const canvas = element(root, "canvas", HTMLCanvasElement);
  await expect.poll(() => canvas.width).toBe(Math.round(390 * devicePixelRatio));
  expect(document.documentElement.scrollWidth).toBeLessThanOrEqual(window.innerWidth);
  await button("Settings").click();
  await page.getByText("Image Detail & highlights", { exact: true }).click();
  await select("Exploration resolution").selectOptions("0.5");
  await expect.poll(() => canvas.width).toBe(Math.floor(390 * devicePixelRatio * 0.5));
  await page.getByText("Camera Position & perspective", { exact: true }).click();
  await select("Navigation").selectOptions("free");
  await button("Close settings").click();
  canvas.focus();
  await userEvent.keyboard("{w>}");
  const distance = element(root, "#distance", HTMLInputElement);
  await expect.poll(() => distance.valueAsNumber).toBeLessThan(29.9);
  window.dispatchEvent(new Event("blur"));
  const stopped = distance.value;
  await new Promise<void>((resolve) =>
    requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
  );
  expect(distance.value).toBe(stopped);
  await userEvent.keyboard("{/w}");
  await button("Reset view").click();
  await button("Settings").click();
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
  await button("Wide sky").click();
  await expect.poll(() => distance.value).toBe("80");
  location.hash = link.hash;
  await expect.poll(() => distance.value).toBe("30");
  expect(element(root, "#exposure", HTMLInputElement).value).toBe("6");
  location.hash = "#view=invalid";
  await expect
    .element(page.getByText("This view link is invalid or uses an unsupported version."))
    .toBeVisible();
  expect(distance.value).toBe("30");
  await button("Close views").click();
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
  await page.screenshot({ path: "../../test-results/explorer-mobile.png" });
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
  await expect.element(button("Save PNG (SDR)")).toBeEnabled();
  expect(canvas.width).toBe(Math.round(240 * devicePixelRatio));
  expect(element(root, "#sampling-status", HTMLElement).textContent).toMatch(
    /All sampled rays resolved|pixels contain unresolved/,
  );
  const exposure = element(root, "#exposure", HTMLInputElement);
  exposure.value = "2";
  exposure.dispatchEvent(new Event("input", { bubbles: true }));
  await expect.element(button("Save PNG (SDR)")).toBeEnabled();
  expect(element(root, "#photo-progress", HTMLProgressElement).value).toBe(64);
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
  await expect.poll(() => canvas.width).toBe(Math.round(320 * devicePixelRatio));
  await expect.element(button("Resume motion")).toBeVisible();
  canvas.style.display = "none";
  await new Promise<void>((resolve) =>
    requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
  );
  canvas.style.width = "224px";
  canvas.style.display = "";
  await expect.poll(() => canvas.width).toBe(Math.round(224 * devicePixelRatio));
  await page.viewport(480, 320);
  canvas.style.width = "100%";
  await expect.poll(() => canvas.width).toBe(Math.round(480 * devicePixelRatio));
});
