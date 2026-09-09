import { expect, test } from "vitest";
import { createCoverageReader } from "../../src/render/coverage.ts";
import { requestDevice } from "../../src/render/device.ts";

test("coverage reduction counts partial workgroups and all 64 sample weights", async () => {
  const device = await requestDevice();
  using owned = new DisposableStack();
  owned.defer(() => device.destroy());
  const readCoverage = await createCoverageReader(device);
  const width = 17;
  const height = 19;
  const data = new Float32Array(width * height * 4);
  let unresolvedPixels = 0;
  let unresolvedSamples = 0;
  for (let pixel = 0; pixel < width * height; pixel++) {
    const failed = Math.min(64, (pixel % 65) + (pixel % 16) / 16);
    data[pixel * 4 + 3] = failed / 64;
    unresolvedPixels += Number(failed > 0);
    unresolvedSamples += failed;
  }
  const history = owned.adopt(
    device.createTexture({
      size: [width, height],
      format: "rgba32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    }),
    (value) => value.destroy(),
  );
  device.queue.writeTexture({ texture: history }, data, { bytesPerRow: width * 16 }, [
    width,
    height,
  ]);
  expect(await readCoverage(history, 64)).toEqual({
    pixels: width * height,
    samplesPerPixel: 64,
    unresolvedPixels,
    unresolvedSamples,
  });
});
