import { blackbodyWhiteBalance } from "../../physics/radiation.ts";

/**
 * Reuse the presentation shader to snapshot SDR Display P3 pixels into a tagged PNG.
 * @returns An exporter that submits the snapshot before awaiting readback and owns its temporary resources.
 */
export async function createPhotoExporter(device: GPUDevice, module: GPUShaderModule) {
  const pipeline = await device.createRenderPipelineAsync({
    layout: "auto",
    vertex: { module, entryPoint: "vertex" },
    fragment: { module, entryPoint: "fragment", targets: [{ format: "rgba8unorm" }] },
  });
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  return async (
    radiance: GPUTexture,
    exposureEV: number,
    bloom = radiance,
    strength = 0,
    whiteBalance?: number,
    photographic = true,
  ): Promise<Blob> => {
    if (!Number.isFinite(strength) || strength < 0 || strength > 1) {
      throw new RangeError("Bloom must be between zero and one.");
    }
    if (!Number.isFinite(exposureEV) || exposureEV < -6 || exposureEV > 6) {
      throw new RangeError("Photo exposure must be between −6 and +6 EV.");
    }
    const { width, height } = radiance;
    const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
    if (bytesPerRow * height > device.limits.maxBufferSize) {
      throw new RangeError("This image exceeds the device's photo readback capacity.");
    }
    using resources = new DisposableStack();
    const output = resources.adopt(
      device.createTexture({
        label: "SDR photo",
        size: [width, height],
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      }),
      (value) => value.destroy(),
    );
    const uniform = resources.adopt(
      device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      (value) => value.destroy(),
    );
    const readback = resources.adopt(
      device.createBuffer({
        size: bytesPerRow * height,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      (value) => value.destroy(),
    );
    const calibration =
      whiteBalance === undefined ? [1, 1, 1] : blackbodyWhiteBalance(whiteBalance);
    device.queue.writeBuffer(
      uniform,
      0,
      new Float32Array([2 ** exposureEV, 1, strength, 0, ...calibration, photographic ? 1 : 0]),
    );
    const bindings = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: radiance.createView() },
        { binding: 1, resource: sampler },
        { binding: 2, resource: { buffer: uniform } },
        { binding: 3, resource: bloom.createView() },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        { view: output.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] },
      ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.draw(3);
    pass.end();
    encoder.copyTextureToBuffer({ texture: output }, { buffer: readback, bytesPerRow }, [
      width,
      height,
    ]);
    // Submit before the first await: later scene changes cannot replace this snapshot.
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const mapped = readback.getMappedRange();
    const pixels = new Uint8ClampedArray(width * height * 4);
    for (let row = 0; row < height; row++) {
      pixels.set(new Uint8ClampedArray(mapped, row * bytesPerRow, width * 4), row * width * 4);
    }
    readback.unmap();
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("2d", { colorSpace: "display-p3" });
    if (!context) {
      throw new Error("Display P3 photo encoding is unavailable.");
    }
    context.putImageData(new ImageData(pixels, width, height, { colorSpace: "display-p3" }), 0, 0);
    return canvas.convertToBlob({ type: "image/png" });
  };
}
