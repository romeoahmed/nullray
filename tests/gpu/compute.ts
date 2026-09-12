import { compileShader, requestDevice } from "../../src/gpu/device.ts";
import { checkDevice } from "../support/gpu.ts";
export { test, readPixels } from "../support/gpu.ts";

interface ComputeResources {
  readonly entries: readonly GPUBindGroupEntry[];
  dispose(): void;
}

/**
 * Execute a test kernel on a fresh real device and copy its f32 storage output.
 *
 * @remarks
 * Input/output bind at 0/1. Extra storage inputs start at 2; resource callbacks
 * must choose nonconflicting bindings. This helper owns the device and all
 * acquired buffers, including callback resources, through success or failure.
 *
 * @param inputData - Borrowed uniform/storage data copied before dispatch.
 * @param outputFloats - Positive output count in f32 elements.
 * @param workgroups - Dispatch count on the x axis; y/z remain one.
 * @param resources - Optional factory transferring ownership of returned GPU resources.
 * @returns A caller-owned output copy. Compilation, validation, or readback failures reject.
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
  const check = checkDevice(device);
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
  await check();
  return result;
}
