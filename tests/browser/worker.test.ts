import { createRenderClient } from "../../src/runtime/client.ts";
import { expect, test } from "vitest";
import type { RenderEvent, RenderRequest } from "../../src/runtime/protocol.ts";
import { initialSession } from "../../src/scene/session.ts";

test("worker retries initialization, preserves settled display state, and exports only the current photograph", async ({
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
  await expect
    .poll(() =>
      events.some(
        (event) =>
          event.type === "frame" && event.revision === 2 && event.samples === event.targetSamples,
      ),
    )
    .toBe(true);
  const { scene: _scene, appearance: _appearance, ...displayOnly } = update;
  send({ ...displayOnly, revision: 3, presentation: { ...update.presentation, exposureEV: 1 } });
  await expect
    .poll(() => events.some((event) => event.type === "frame" && event.revision === 3))
    .toBe(true);
  expect(events.filter((event) => event.type === "error")).toHaveLength(1);
  const displayUpdate = events.find((event) => event.type === "frame" && event.revision === 3);
  if (displayUpdate?.type !== "frame") {
    throw new Error("Missing displayed frame.");
  }
  expect(displayUpdate.samples).toBe(displayUpdate.targetSamples);
  expect(displayUpdate.samples).toBeGreaterThan(1);
  send({ ...update, revision: 4, time: 12 });
  await expect
    .poll(() => events.some((event) => event.type === "frame" && event.revision === 4))
    .toBe(true);
  expect(events.find((event) => event.type === "frame" && event.revision === 4)).toMatchObject({
    samples: 1,
    time: 12,
  });
  send({ ...displayOnly, revision: 5, motion: "playing", width: 1024, height: 768 });
  await expect
    .poll(() => events.some((event) => event.type === "frame" && event.revision === 5))
    .toBe(true);
  const live = events.find((event) => event.type === "frame" && event.revision === 5);
  expect(live?.type).toBe("frame");
  if (live?.type === "frame") {
    expect([live.width, live.height]).toEqual([1024, 768]);
    expect(live).toMatchObject({ samples: 1, targetSamples: 1 });
  }
  send({
    ...displayOnly,
    revision: 6,
    motion: "playing",
    width: 1024,
    height: 768,
    resolution: "auto",
  });
  await expect
    .poll(() => events.some((event) => event.type === "frame" && event.revision === 6))
    .toBe(true);
  const adaptive = events.find((event) => event.type === "frame" && event.revision === 6);
  expect(adaptive?.type).toBe("frame");
  if (adaptive?.type === "frame") {
    expect(adaptive.width * adaptive.height).toBeLessThan(1024 * 768);
  }
  send({ ...displayOnly, revision: 7, motion: "refining" });
  await expect
    .poll(
      () =>
        events.some(
          (event) =>
            event.type === "frame" &&
            event.revision === 7 &&
            event.samples === event.targetSamples &&
            event.samples > 1,
        ),
      { timeout: 10_000 },
    )
    .toBe(true);
  send({ type: "export", revision: 6 });
  await expect
    .poll(() => events.some((event) => event.type === "export-error" && event.revision === 6))
    .toBe(true);
  send({ type: "export", revision: 7 });
  await expect
    .poll(() => events.some((event) => event.type === "exported" && event.revision === 7), {
      timeout: 10_000,
    })
    .toBe(true);
  const exported = events.find((event) => event.type === "exported" && event.revision === 7);
  expect(exported?.type).toBe("exported");
  if (exported?.type === "exported") {
    expect(exported.blob.type).toBe("image/png");
    const image = await createImageBitmap(exported.blob);
    expect([image.width, image.height]).toEqual([64, 48]);
    image.close();
  }
  send({ type: "dispose" });
  await expect.poll(() => events.some((event) => event.type === "disposed")).toBe(true);
}, 30_000);

test("client coalesces revisions and restored epochs, then closes its public lifecycle", async ({
  onTestFinished,
}) => {
  const canvas = document.createElement("canvas");
  document.body.append(canvas);
  const events: RenderEvent[] = [];
  const client = createRenderClient(canvas, (event) => events.push(event));
  onTestFinished(() => {
    client.dispose();
    canvas.remove();
  });
  const session = { ...initialSession, motion: "paused" } as const;
  client.update(session, false, true, 32, 24, 7);
  const revision = client.update(
    { ...session, view: { ...session.view, exposureEV: 1 } },
    false,
    true,
    32,
    24,
  );
  await expect
    .poll(() => events.some((event) => event.type === "frame" && event.revision === revision), {
      timeout: 10_000,
    })
    .toBe(true);
  const frames = events.filter((event) => event.type === "frame");
  expect(frames.length).toBeGreaterThan(0);
  expect(frames.every((event) => event.revision === revision && event.time === 7)).toBe(true);
  expect(events.some((event) => event.type === "error")).toBe(false);
  client.dispose();
  const count = events.length;
  expect(client.retry()).toBe(false);
  expect(client.update(session, false, true, 64, 48, 9)).toBe(revision);
  client.inspect();
  client.exportPhoto();
  client.dispose();
  await new Promise<void>((resolve) =>
    requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
  );
  expect(events).toHaveLength(count);
}, 15_000);
