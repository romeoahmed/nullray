import source from "./shaders/coverage.wgsl?raw";
import { compileShader } from "./device.ts";

/** Image-wide missing weights; affected pixels and missing samples measure different quantities. */
export interface Coverage {
  readonly pixels: number;
  readonly samplesPerPixel: number;
  readonly unresolvedPixels: number;
  /** Sum of missing sample weights, in multiples of 1/16. */
  readonly unresolvedSamples: number;
}

/**
 * Read unresolved weights from a submitted photographic history without copying its image.
 * @returns A reader that owns temporary readback buffers and borrows the history for each call.
 */
export async function createCoverageReader(device: GPUDevice) {
  const module = await compileShader(device, source, "sampling coverage");
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module } });
  return async (history: GPUTexture, samplesPerPixel: number): Promise<Coverage> => {
    if (!Number.isSafeInteger(samplesPerPixel) || samplesPerPixel < 1 || samplesPerPixel > 64) {
      throw new RangeError("Coverage readback requires 1–64 accumulated samples.");
    }
    using owned = new DisposableStack();
    const buffer = (size: number, usage: GPUBufferUsageFlags) =>
      owned.adopt(device.createBuffer({ size, usage }), (value) => value.destroy());
    const columns = Math.ceil(history.width / 16);
    const rows = Math.ceil(history.height / 16);
    const bytes = columns * rows * 8;
    const output = buffer(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const staging = buffer(bytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const uniform = buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(uniform, 0, new Uint32Array([samplesPerPixel, 0, 0, 0]));
    const bindings = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: history.createView() },
        { binding: 1, resource: { buffer: uniform } },
        { binding: 2, resource: { buffer: output } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.dispatchWorkgroups(columns, rows);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const counts = new Uint32Array(staging.getMappedRange());
    let unresolvedPixels = 0;
    let unresolvedSamples = 0;
    for (let index = 0; index < counts.length; index += 2) {
      unresolvedPixels += counts[index] ?? 0;
      unresolvedSamples += counts[index + 1] ?? 0;
    }
    staging.unmap();
    return {
      pixels: history.width * history.height,
      samplesPerPixel,
      unresolvedPixels,
      unresolvedSamples: unresolvedSamples / 16,
    };
  };
}
