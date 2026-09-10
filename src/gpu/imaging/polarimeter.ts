import source from "../wgsl/passes/polarimeter.wgsl?raw";
import { compileShader } from "../device.ts";
import type { OpticalImage } from "../optics/engine.ts";

/** Detector-owned output and uniforms; changing analyzer angle reuses physical Stokes images. */
export async function createPolarimeter(device: GPUDevice) {
  const module = await compileShader(device, source, "polarimeter");
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module } });
  const owned = new DisposableStack();
  const uniform = owned.adopt(
    device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
    (value) => value.destroy(),
  );
  let target: GPUTexture | undefined;
  const parameters = new Float32Array(4);
  let bound: Pick<OpticalImage, "radiance" | "q" | "u"> | undefined;
  let bindings: GPUBindGroup | undefined;
  owned.defer(() => target?.destroy());
  return {
    encode(
      encoder: GPUCommandEncoder,
      image: Pick<OpticalImage, "radiance" | "q" | "u">,
      analyzer: number | null,
      diagnostic: "image" | "polarization" | "angle",
    ): GPUTexture {
      if (owned.disposed) {
        throw new Error("The polarimeter is disposed.");
      }
      if (analyzer === null && diagnostic === "image") {
        return image.radiance;
      }
      const { width, height } = image.radiance;
      if (!target || target.width !== width || target.height !== height) {
        const next = device.createTexture({
          label: "polarimeter",
          size: [width, height],
          format: "rgba16float",
          usage:
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_SRC,
        });
        target?.destroy();
        target = next;
      }
      const angle = 2 * (analyzer ?? 0);
      parameters.set([
        Math.cos(angle),
        Math.sin(angle),
        diagnostic === "polarization" ? 2 : diagnostic === "angle" ? 3 : 1,
        0,
      ]);
      device.queue.writeBuffer(uniform, 0, parameters);
      if (
        !bindings ||
        bound?.radiance !== image.radiance ||
        bound.q !== image.q ||
        bound.u !== image.u
      ) {
        bindings = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: image.radiance.createView() },
            { binding: 1, resource: image.q.createView() },
            { binding: 2, resource: image.u.createView() },
            { binding: 3, resource: target.createView() },
            { binding: 4, resource: { buffer: uniform } },
          ],
        });
        bound = image;
      }
      const pass = encoder.beginComputePass({ label: "polarimeter" });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindings);
      pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
      pass.end();
      return target;
    },
    dispose() {
      owned.dispose();
    },
  };
}
