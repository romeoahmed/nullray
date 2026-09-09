import { compileShader, requestDevice } from "../../src/render/device.ts";

/**
 * Execute a storage-buffer kernel and return an owned copy of its f32 output.
 * Input/output occupy bindings 0/1; extra inputs follow from binding 2.
 * Each call owns a real device, and both success and failure release its buffers.
 */
export async function computeReadback(
  source: string,
  inputData: Float32Array,
  outputFloats: number,
  workgroups: number,
  extraInputs: readonly Float32Array[] = [],
): Promise<Float32Array> {
  const device = await requestDevice();
  device.pushErrorScope("validation");
  using owned = new DisposableStack();
  owned.defer(() => device.destroy());
  const buffer = (size: number, usage: GPUBufferUsageFlags) =>
    owned.adopt(device.createBuffer({ size, usage }), (value) => value.destroy());
  const module = await compileShader(device, source, "compute-reference-test");
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module },
  });
  const size = outputFloats * Float32Array.BYTES_PER_ELEMENT;
  const input = buffer(inputData.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
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
