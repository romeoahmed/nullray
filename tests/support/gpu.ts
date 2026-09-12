import { requestDevice } from "../../src/gpu/device.ts";
import { test as baseTest } from "vitest";

/** Check all queued error scopes and unexpected loss after a completed workload; call once, before disposal. */
export function checkDevice(device: GPUDevice): () => Promise<void> {
  const filters = ["validation", "out-of-memory", "internal"] as const;
  for (const filter of filters) {
    device.pushErrorScope(filter);
  }
  let lost: GPUDeviceLostInfo | undefined;
  void device.lost.then((info) => {
    lost = info;
    return undefined;
  });
  return async () => {
    await device.queue.onSubmittedWorkDone();
    const errors = await Promise.all(filters.map(() => device.popErrorScope()));
    const failure = errors.find((error) => error !== null);
    if (failure || lost) {
      throw new Error(`GPU execution failed: ${failure?.message ?? lost?.message}`);
    }
  };
}

/** Real per-test device; cleanup checks submitted work even when an assertion fails. */
export const test = baseTest.extend("device", async ({ task }, { onCleanup }) => {
  const device = await requestDevice();
  device.label = task.name;
  const check = checkDevice(device);
  onCleanup(async () => {
    try {
      await check();
    } finally {
      device.destroy();
    }
  });
  return device;
});

/**
 * Copy one submitted f16/f32 RGBA texture subresource into packed f32 rows.
 *
 * @param texture - Borrowed texture with COPY_SRC usage, retained through readback.
 * @param mipLevel - Existing mip level; bounds remain the caller's responsibility.
 * @param layer - Existing array layer, or zero for an ordinary 2D image.
 * @returns Caller-owned pixels without GPU row padding; staging is released on settlement.
 * @throws Error - If the texture format is unsupported; native readback errors also reject.
 */
export async function readPixels(
  device: GPUDevice,
  texture: GPUTexture,
  mipLevel = 0,
  layer = 0,
): Promise<Float32Array> {
  const width = Math.max(1, texture.width >> mipLevel);
  const height = Math.max(1, texture.height >> mipLevel);
  if (texture.format !== "rgba16float" && texture.format !== "rgba32float") {
    throw new Error(`Unsupported reference readback format: ${texture.format}`);
  }
  const bytes = texture.format === "rgba16float" ? 2 : 4;
  const stride = Math.ceil((width * 4 * bytes) / 256) * 256;
  using owned = new DisposableStack();
  const staging = owned.adopt(
    device.createBuffer({
      size: stride * height,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    }),
    (value) => value.destroy(),
  );
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture, mipLevel, origin: [0, 0, layer] },
    { buffer: staging, bytesPerRow: stride },
    [width, height],
  );
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const data =
    bytes === 2
      ? new Float16Array(staging.getMappedRange())
      : new Float32Array(staging.getMappedRange());
  const result = new Float32Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    result.set(
      data.subarray((y * stride) / bytes, (y * stride) / bytes + width * 4),
      y * width * 4,
    );
  }
  staging.unmap();
  return result;
}
