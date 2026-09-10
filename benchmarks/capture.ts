/** Read a completed RGBA16F image after timing; rows retain WebGPU's required padding. */
export async function captureRadiance(device: GPUDevice, image: GPUTexture): Promise<Float32Array> {
  const stride = Math.ceil((image.width * 8) / 256) * 256;
  using owned = new DisposableStack();
  const staging = owned.adopt(
    device.createBuffer({
      size: stride * image.height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
    (value) => value.destroy(),
  );
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture: image }, { buffer: staging, bytesPerRow: stride }, [
    image.width,
    image.height,
  ]);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const source = new Float16Array(staging.getMappedRange());
  const result = new Float32Array(image.width * image.height * 4);
  for (let y = 0; y < image.height; y++) {
    result.set(
      source.subarray((y * stride) / 2, (y * stride) / 2 + image.width * 4),
      y * image.width * 4,
    );
  }
  staging.unmap();
  return result;
}

/** Record image quality evidence without imposing a performance acceptance threshold. */
export function imageCoverage(pixels: Float32Array) {
  let unresolvedWeight = 0;
  let radiance = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    const [r = NaN, g = NaN, b = NaN, weight = NaN] = pixels.subarray(i, i + 4);
    if (![r, g, b, weight].every(Number.isFinite) || weight < 0 || weight > 1) {
      throw new Error("The rendered benchmark image contains invalid data.");
    }
    unresolvedWeight += 1 - weight;
    radiance += Math.abs(r) + Math.abs(g) + Math.abs(b);
  }
  return { pixels: pixels.length / 4, unresolvedWeight, radiance };
}
