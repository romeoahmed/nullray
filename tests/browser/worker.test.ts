import { expect, test } from "vitest";
import type { RenderEvent, RenderRequest } from "../../src/runtime/protocol.ts";
import { initialSession } from "../../src/scene/session.ts";

test("retry releases a superseded initializer before reusing its canvas context", async ({
  onTestFinished,
}) => {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 48;
  document.body.append(canvas);
  const worker = new Worker(new URL("../../src/runtime/worker/main.ts", import.meta.url), {
    type: "module",
  });
  onTestFinished(() => {
    worker.terminate();
    canvas.remove();
  });
  const events: RenderEvent[] = [];
  worker.addEventListener("message", ({ data }: MessageEvent<RenderEvent>) => events.push(data));
  const send = (message: RenderRequest, transfer: Transferable[] = []) =>
    worker.postMessage(message, transfer);
  const offscreen = canvas.transferControlToOffscreen();
  send({ type: "initialize", canvas: offscreen }, [offscreen]);
  await expect.poll(() => events.some((event) => event.type === "starting")).toBe(true);
  const view = initialSession.view;
  const update = {
    type: "update",
    revision: 1,
    scene: view.scene,
    appearance: view.appearance,
    presentation: {
      exposureEV: view.exposureEV,
      whiteBalance: view.whiteBalance,
      bloom: view.bloom,
      diagnostic: view.diagnostic,
      analyzer: view.analyzer,
    },
    motion: "paused",
    resolution: 1,
    hdr: false,
    visible: true,
    width: 64,
    height: 48,
  } as const satisfies RenderRequest;
  send({ ...update, scene: { ...view.scene, space: { spin: Number.NaN, charge: 1 } } });
  send({ type: "retry" });
  send({ ...update, revision: 2 });
  await expect
    .poll(() => events.some((event) => event.type === "frame" && event.revision === 2), {
      timeout: 10_000,
    })
    .toBe(true);
  expect(events.filter((event) => event.type === "error")).toHaveLength(1);
  send({ ...update, revision: 3, presentation: { ...update.presentation, exposureEV: 1 } });
  await expect
    .poll(() => events.some((event) => event.type === "frame" && event.revision === 3))
    .toBe(true);
  expect(events.filter((event) => event.type === "error")).toHaveLength(1);
  send({ type: "dispose" });
  await expect.poll(() => events.some((event) => event.type === "disposed")).toBe(true);
}, 30_000);
