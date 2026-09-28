import { expect, test } from "vitest";

import { requestDevice, watchDevice } from "../../src/gpu/device.ts";

test("uncaptured native GPU validation errors reach the device owner", async () => {
  using owned = new DisposableStack();
  const device = owned.adopt(await requestDevice(), (value) => value.destroy());
  const health = owned.use(watchDevice(device));
  // Deliberately invalid native usage; no error scope consumes the event.
  device
    .createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.STORAGE })
    .destroy();
  // Flush the native GPU timeline, as production does after each submission.
  await device.queue.onSubmittedWorkDone();
  const failure = await health.failure;
  expect(failure).toBeInstanceOf(Error);
  expect(failure.message).toMatch(/GPU execution failed:/);
  device.destroy();
  expect(await health.failure).toBe(failure);
});

test("native device loss reaches the device owner without a rejected promise", async () => {
  using owned = new DisposableStack();
  const device = owned.adopt(await requestDevice(), (value) => value.destroy());
  const health = owned.use(watchDevice(device));
  device.destroy();
  expect((await health.failure).message).toMatch(/graphics device was lost/);
});

test("disposing the subscription before native teardown suppresses intentional loss", async () => {
  using owned = new DisposableStack();
  const device = owned.adopt(await requestDevice(), (value) => value.destroy());
  const health = owned.use(watchDevice(device));
  let reported = false;
  void health.failure.then(() => {
    reported = true;
    return undefined;
  });
  health[Symbol.dispose]();
  device.destroy();
  await device.lost;
  expect(reported).toBe(false);
});
