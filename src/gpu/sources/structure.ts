import { compileShader } from "../device.ts";
import source from "../wgsl/sources/noise.wgsl?raw";

/**
 * Owner of the generated lattice texture with sampler bindings borrowed by consumers.
 *
 * @remarks
 * Call `dispose` after dependent work is submitted; consumers must not destroy
 * the borrowed texture independently. The generating device is not owned.
 */
export interface StructureField {
  readonly texture: GPUTexture;
  readonly sampler: GPUSampler;
  dispose(): void;
}

/**
 * Generate and submit a deterministic 64³ scalar lattice on the borrowed device.
 *
 * @returns The field owner after submission, without waiting for GPU completion.
 * Subsequent work on the same queue observes the upload/generation order.
 */
export async function createStructureField(device: GPUDevice): Promise<StructureField> {
  const module = await compileShader(device, source, "material lattice");
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module } });
  using owned = new DisposableStack();
  const size = 64;
  const texture = owned.adopt(
    device.createTexture({
      label: "material structure",
      dimension: "3d",
      size: [size, size, size],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_SRC,
    }),
    (value) => value.destroy(),
  );
  const sampler = device.createSampler({
    addressModeU: "repeat",
    addressModeV: "repeat",
    addressModeW: "repeat",
    minFilter: "linear",
    magFilter: "linear",
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass({ label: "material lattice" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(
    0,
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: texture.createView() }],
    }),
  );
  pass.dispatchWorkgroups(size / 4, size / 4, size / 4);
  pass.end();
  device.queue.submit([encoder.finish()]);
  const lifetime = owned.move();
  return { texture, sampler, dispose: () => lifetime.dispose() };
}
