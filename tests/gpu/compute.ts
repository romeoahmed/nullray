import { compileShader, requestDevice } from "../../src/gpu/device.ts";
import { test as baseTest } from "vitest";

/** Native lazy fixture: every requesting test owns a real, independently disposed device. */
export const test = baseTest.extend("device", async ({ task }, { onCleanup }) => {
  const device = await requestDevice();
  device.label = task.name;
  device.pushErrorScope("validation");
  onCleanup(async () => {
    try {
      const error = await device.popErrorScope();
      if (error) {
        throw new Error(`GPU validation failed: ${error.message}`);
      }
    } finally {
      device.destroy();
    }
  });
  return device;
});

/** Read an owned copy of a floating-point RGBA texture, including padded and one-pixel rows. */
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

interface ComputeResources {
  readonly entries: readonly GPUBindGroupEntry[];
  dispose(): void;
}

/**
 * Execute a storage-buffer kernel and return an owned copy of its f32 output.
 * Uniform or storage input/output occupy bindings 0/1; extra storage inputs follow from binding 2.
 * Each call owns a real device, and both success and failure release its buffers.
 */
export async function computeReadback(
  source: string,
  inputData: Float32Array,
  outputFloats: number,
  workgroups: number,
  extraInputs: readonly Float32Array[] = [],
  entryPoint?: string,
  resources?: (device: GPUDevice) => ComputeResources | Promise<ComputeResources>,
): Promise<Float32Array> {
  const device = await requestDevice();
  device.pushErrorScope("validation");
  using owned = new DisposableStack();
  owned.defer(() => device.destroy());
  const prepared = resources && owned.adopt(await resources(device), (value) => value.dispose());
  const buffer = (size: number, usage: GPUBufferUsageFlags) =>
    owned.adopt(device.createBuffer({ size, usage }), (value) => value.destroy());
  const module = await compileShader(device, source, "compute-reference-test");
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: entryPoint ? { module, entryPoint } : { module },
  });
  const size = outputFloats * Float32Array.BYTES_PER_ELEMENT;
  const input = buffer(
    inputData.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  const output = buffer(size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const staging = buffer(size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(input, 0, inputData);
  const extraBindings = extraInputs.map((data, index) => {
    const storage = buffer(data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(storage, 0, data);
    return { binding: index + 2, resource: { buffer: storage } };
  });
  const bindings = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: output } },
      ...extraBindings,
      ...(prepared?.entries ?? []),
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindings);
  pass.dispatchWorkgroups(workgroups);
  pass.end();
  encoder.copyBufferToBuffer(output, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange()).slice();
  staging.unmap();
  const validation = await device.popErrorScope();
  if (validation) {
    throw new Error(`GPU reference execution failed: ${validation.message}`);
  }
  return result;
}
