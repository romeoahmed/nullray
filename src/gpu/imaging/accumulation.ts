import { compileShader } from "../device.ts";
import source from "../wgsl/passes/accumulate.wgsl?raw";
import { createCoverageReader } from "./coverage.ts";

/**
 * Compile shared accumulation/coverage pipelines and return a factory for independent histories.
 *
 * @remarks
 * The factory borrows the device; each history owns its uniforms and lazy f32/f16
 * targets. Dispose every history before the device. Index zero resets only that
 * history; missing weight is never renormalized away.
 */
export async function createAccumulationFactory(device: GPUDevice) {
  const module = await compileShader(device, source, "radiance accumulation");
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module } });
  const readCoverage = await createCoverageReader(device);
  return function createHistory() {
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
    owned.defer(() => {
      target?.resources.dispose();
      target = undefined;
    });
    const lifetime = owned.move();
    return {
      /**
       * Encode the next sample, with integer index 0–63; zero starts a new sequence.
       *
       * @remarks
       * Submit before another encode updates the shared uniform. Resizing requires
       * index zero. The returned texture is borrowed, overwritten on later encodes,
       * and destroyed on resize/disposal.
       */
      encode(encoder: GPUCommandEncoder, radiance: GPUTexture, index: number): GPUTexture {
        const { width, height } = radiance;
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
        if (!target) {
          throw new Error("No photographic history is available.");
        }
        return readCoverage(target.history, samples);
      },
      /** Release history and its uniforms; the caller retains the device. */
      dispose() {
        lifetime.dispose();
      },
    };
  };
}
