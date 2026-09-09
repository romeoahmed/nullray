import { expect, test } from "vitest";
import { createRenderer } from "../../src/render/renderer.ts";
import { initialScene } from "../../src/model/scene.ts";

test("aborted renderer initialization releases the canvas for a fresh owner", async () => {
  const canvas = new OffscreenCanvas(1, 1);
  const attempt = new AbortController();
  const pending = createRenderer(canvas, attempt.signal);
  const reason = new Error("Superseded renderer");
  attempt.abort(reason);
  await expect(pending).rejects.toBe(reason);
  await expect(createRenderer(canvas, attempt.signal)).rejects.toBe(reason);

  using owned = new DisposableStack();
  const renderer = owned.adopt(await createRenderer(canvas), (value) => value.dispose());
  renderer.resize(16, 9);
  expect(
    renderer.render(initialScene, {
      exposureEV: 0,
      hdr: false,
      time: 0,
      refine: true,
      view: "image",
    }),
  ).toBe(1);
  await renderer.finished();
  expect([canvas.width, canvas.height]).toEqual([16, 9]);
  expect((await renderer.coverage()).pixels).toBe(144);
  renderer.dispose();
  expect(() => renderer.coverage()).toThrow("No refined image");
  expect(() => renderer.dispose()).not.toThrow();
});
