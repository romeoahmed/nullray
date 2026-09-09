import { expect, test } from "vitest";
import { createRenderer } from "../../src/render/renderer.ts";
import { initialScene } from "../../src/model/scene.ts";

test("exploration resolution changes preserve native photographic dimensions and history", async () => {
  const canvas = new OffscreenCanvas(1, 1);
  using owned = new DisposableStack();
  const renderer = owned.adopt(await createRenderer(canvas), (value) => value.dispose());
  const settings = { exposureEV: 0, hdr: false, time: 0, refine: false, view: "image" as const };
  renderer.resize(32, 18);
  renderer.render(initialScene, { ...settings, resolutionScale: 0.5 });
  await renderer.finished();
  expect([canvas.width, canvas.height]).toEqual([16, 9]);
  expect(renderer.render(initialScene, { ...settings, refine: true, resolutionScale: 0.5 })).toBe(
    1,
  );
  await renderer.finished();
  expect([canvas.width, canvas.height]).toEqual([32, 18]);
  expect(renderer.render(initialScene, { ...settings, refine: true, resolutionScale: 0.75 })).toBe(
    2,
  );
  await renderer.finished();
  expect([canvas.width, canvas.height]).toEqual([32, 18]);
  renderer.render(initialScene, { ...settings, resolutionScale: 0.75 });
  await renderer.finished();
  expect([canvas.width, canvas.height]).toEqual([24, 13]);
  renderer.resize(40, 24);
  renderer.render(initialScene, { ...settings, resolutionScale: 0.5 });
  await renderer.finished();
  expect([canvas.width, canvas.height]).toEqual([20, 12]);
  for (const resolutionScale of [0, -1, 1.1, NaN, Infinity]) {
    expect(() => renderer.render(initialScene, { ...settings, resolutionScale })).toThrow(
      RangeError,
    );
  }
});
