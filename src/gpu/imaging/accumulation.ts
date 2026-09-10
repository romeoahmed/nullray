import source from "../wgsl/passes/accumulate.wgsl?raw";
import { compileShader } from "../device.ts";
import { createCoverageReader } from "./coverage.ts";

/**
 * Equal-weight Halton points over the pixel box, centered at the pixel origin.
 * @param index - Zero-based sample index supplied by the bounded photographic sequence.
 * @returns Horizontal and vertical offsets in pixel units.
 */
export function pixelJitter(index: number): readonly [number, number] {
  const radicalInverse = (base: number) => {
    let integer = index + 1;
    let fraction = 1 / base;
    let value = 0;
    while (integer > 0) {
      value += (integer % base) * fraction;
      integer = Math.floor(integer / base);
      fraction /= base;
    }
    return value - 0.5;
  };
  return [radicalInverse(2), radicalInverse(3)];
}

/** Lazily allocated photographic history; index zero resets the running mean. */
export async function createAccumulation(device: GPUDevice) {
  const module = await compileShader(device, source, "radiance accumulation");
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module } });
  const readCoverage = await createCoverageReader(device);
  const owned = new DisposableStack();
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
  return {
    /** Encode one sample into owned history; borrowed output remains valid until resize/disposal. */
    encode(encoder: GPUCommandEncoder, radiance: GPUTexture, index: number): GPUTexture {
      if (owned.disposed) {
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
      if (!target || samples === 0 || owned.disposed) {
        throw new Error("No photographic history is available.");
      }
      return readCoverage(target.history, samples);
    },
    /** Release history and its uniforms; the caller retains the device. */
    dispose() {
      owned.dispose();
    },
  };
}
