import { expect, test } from "vitest";
import { page, server } from "vitest/browser";
import markup from "../../index.html?raw";
import { mountApp } from "../../src/ui/app.ts";
import { initialSession } from "../../src/scene/session.ts";
import { element } from "../../src/ui/elements.ts";
import { presets } from "../../src/scene/presets.ts";
import "../../src/ui/style.css";

test.for(presets.filter((preset) => ["Classic disk", "Blue accretion flow"].includes(preset.name)))(
  "$name completes a fixed-epoch visual acceptance photograph",
  { timeout: 45_000 },
  async (preset, { onTestFinished }) => {
    await page.viewport(960, 600);
    const root = element(new DOMParser().parseFromString(markup, "text/html"), "#app", HTMLElement);
    document.body.append(root);
    const dispose = mountApp(root, { ...initialSession, view: preset.view, motion: "refining" });
    onTestFinished(() => {
      dispose();
      root.remove();
    });
    await expect
      .element(page.getByText("Photograph ready", { exact: true }), { timeout: 30_000 })
      .toBeVisible();
    const canvas = element(root, "canvas", HTMLCanvasElement);
    const coverage = element(root, "#sampling-status", HTMLElement).textContent;
    const name = preset.name === "Classic disk" ? "warm" : "blue";
    await server.commands.writeFile(
      `test-results/acceptance-${name}.json`,
      JSON.stringify(
        {
          view: preset.view,
          width: canvas.width,
          height: canvas.height,
          samples: element(root, "#photo-progress", HTMLProgressElement).max,
          coverage,
          browser: navigator.userAgent,
        },
        null,
        2,
      ),
    );
    await page.getByRole("button", { name: "Hide controls", exact: true }).click();
    await Promise.allSettled(
      root.getAnimations({ subtree: true }).map((animation) => animation.finished),
    );
    await page.screenshot({ path: `../../test-results/acceptance-${name}.png` });
  },
);
