import source from "../wgsl/passes/accumulate.wgsl?raw";
import { compileShader } from "../device.ts";
import { createCoverageReader } from "./coverage.ts";

/**
 * Create a lazy f32 mean history and f16 output, borrowing the device.
 *
 * @remarks
 * The returned owner handles one Stokes component. Index zero resets the sequence;
 * missing weight remains in history alpha and is not renormalized away.
 */
export async function createAccumulation(device: GPUDevice) {
  const module = await compileShader(device, source, "radiance accumulation");
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module } });
  const readCoverage = await createCoverageReader(device);
  using owned = new DisposableStack();
  const uniform = owned.adopt(
    device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    (value) => value.destroy(),
  );
  const parameters = new Uint32Array(4);
  let samples = 0;
  let boundSource: GPUTexture | undefined;
  let bindings: GPUBindGroup | undefined;
  let target:
    | {
        readonly resources: DisposableStack;
        readonly width: number;
        readonly height: number;
        readonly history: GPUTexture;
        readonly output: GPUTexture;
      }
    | undefined;
  owned.defer(() => target?.resources.dispose());
  const lifetime = owned.move();
  return {
    /**
     * Encode the next sample, with integer index 0–63; zero starts a new sequence.
     *
     * @remarks
     * Submit before another encode updates the shared uniform. Resizing requires
     * index zero. The returned texture is borrowed, overwritten on later encodes,
     * and destroyed on resize/disposal. Invalid sequencing throws RangeError.
     */
    encode(encoder: GPUCommandEncoder, radiance: GPUTexture, index: number): GPUTexture {
      if (lifetime.disposed) {
        throw new Error("Photographic history has been disposed.");
      }
      if (
        !Number.isInteger(index) ||
        index < 0 ||
        index >= 64 ||
        (index !== 0 && index !== samples)
      ) {
        throw new RangeError(
          "Photographic samples must start at zero and advance sequentially through 63.",
        );
      }
      const { width, height } = radiance;
      if (index !== 0 && (!target || target.width !== width || target.height !== height)) {
        throw new RangeError("Resizing photographic history requires a new sample sequence.");
      }
      if (!target || target.width !== width || target.height !== height) {
        using resources = new DisposableStack();
        const texture = (format: GPUTextureFormat) =>
          resources.adopt(
            device.createTexture({
              size: [width, height],
              format,
              usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.COPY_SRC,
            }),
            (value) => value.destroy(),
          );
        const previous = target;
        target = {
          width,
          height,
          history: texture("rgba32float"),
          output: texture("rgba16float"),
          resources: resources.move(),
        };
        previous?.resources.dispose();
        bindings = undefined;
      }
      parameters[0] = index;
      device.queue.writeBuffer(uniform, 0, parameters);
      if (!bindings || boundSource !== radiance) {
        bindings = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: radiance.createView() },
            { binding: 1, resource: target.history.createView() },
            { binding: 2, resource: target.output.createView() },
            { binding: 3, resource: { buffer: uniform } },
          ],
        });
        boundSource = radiance;
      }
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindings);
      pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
      pass.end();
      samples = index + 1;
      return target.output;
    },
    /** Submit a readback after the caller has submitted all encoded history writes. */
    coverage() {
      if (!target || samples === 0 || lifetime.disposed) {
        throw new Error("No photographic history is available.");
      }
      return readCoverage(target.history, samples);
    },
    /** Release history and its uniforms; the caller retains the device. */
    dispose() {
      lifetime.dispose();
    },
  };
}
