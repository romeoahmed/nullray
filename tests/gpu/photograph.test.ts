import { expect, test } from "vitest";
import { createPhotoExporter } from "../../src/render/photograph.ts";
import { compileShader, requestDevice } from "../../src/render/device.ts";
import presentation from "../../src/render/shaders/present.wgsl?raw";

const decode = (value: number) => {
  const channel = value / 255;
  return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
};

test.each([0, 0.5, 1])(
  "PNG export preserves bloom %s, SDR exposure, P3 colors, and padded-row orientation",
  async (strength) => {
    const device = await requestDevice();
    using resources = new DisposableStack();
    resources.defer(() => device.destroy());
    const radiance = resources.adopt(
      device.createTexture({
        size: [7, 3],
        format: "rgba16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      }),
      (value) => value.destroy(),
    );
    const scattered = resources.adopt(
      device.createTexture({
        size: [7, 3],
        format: "rgba16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      }),
      (value) => value.destroy(),
    );
    const values = new Float16Array(7 * 3 * 4);
    for (let y = 0; y < 3; y++) {
      for (let x = 0; x < 7; x++) {
        values.set(
          y === 0 ? [16, 16, 16, 1] : y === 1 ? [4, 0, 0, 1] : [0, 0, 0, 1],
          (y * 7 + x) * 4,
        );
      }
    }
    device.queue.writeTexture({ texture: radiance }, values, { bytesPerRow: 7 * 8 }, [7, 3]);
    const exportPhoto = await createPhotoExporter(
      device,
      await compileShader(device, presentation, "photo-test"),
    );
    device.queue.writeTexture(
      { texture: scattered },
      values.map((value, index) => (index % 4 === 3 ? value : value * 0.5)),
      { bytesPerRow: 7 * 8 },
      [7, 3],
    );
    const pending = exportPhoto(radiance, -2, scattered, strength);
    // A later upload must not alter an export that has already been submitted.
    device.queue.writeTexture(
      { texture: radiance },
      new Float16Array(values.length),
      { bytesPerRow: 7 * 8 },
      [7, 3],
    );
    device.queue.writeTexture(
      { texture: scattered },
      new Float16Array(values.length),
      { bytesPerRow: 7 * 8 },
      [7, 3],
    );
    const blob = await pending;
    expect(blob.type).toBe("image/png");
    const bitmap = await createImageBitmap(blob);
    resources.defer(() => bitmap.close());
    expect([bitmap.width, bitmap.height]).toEqual([7, 3]);
    const canvas = new OffscreenCanvas(7, 3);
    const context = canvas.getContext("2d", { colorSpace: "display-p3", willReadFrequently: true });
    if (!context) {
      throw new Error("Missing P3 decoding context.");
    }
    context.drawImage(bitmap, 0, 0);
    const decoded = context.getImageData(0, 0, 7, 3, { colorSpace: "display-p3" }).data;

    const gain = 1 - 0.5 * strength;
    for (let x = 0; x < 7; x++) {
      for (let c = 0; c < 3; c++) {
        expect(
          Math.abs(decode(decoded[x * 4 + c] ?? Number.NaN) - (4 * gain) / (1 + 4 * gain)),
        ).toBeLessThan(0.006);
        expect(decoded[(14 + x) * 4 + c]).toBe(0);
      }
      const red = decode(decoded[(7 + x) * 4] ?? Number.NaN);
      const green = decode(decoded[(7 + x) * 4 + 1] ?? Number.NaN);
      const blue = decode(decoded[(7 + x) * 4 + 2] ?? Number.NaN);
      expect(Math.abs(red - (0.82246197 * gain) / (1 + 0.82246197 * gain))).toBeLessThan(0.006);
      expect(Math.abs(green / red - 0.0403596)).toBeLessThan(0.002);
      expect(Math.abs(blue / red - 0.0207701)).toBeLessThan(0.002);
    }
    expect(
      Array.from(decoded)
        .filter((_, index) => index % 4 === 3)
        .every((alpha) => alpha === 255),
    ).toBe(true);
  },
);
