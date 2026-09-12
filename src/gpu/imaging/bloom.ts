import fullscreen from "../wgsl/imaging/fullscreen.wgsl?raw";
import source from "../wgsl/passes/bloom.wgsl?raw";
import { compileShader } from "../device.ts";

/**
 * Create a glare pyramid with normalized kernels and equal weight per spatial octave.
 *
 * @remarks
 * Borrows the device and owns size-dependent targets/uniforms. Constant images
 * are preserved within rounding; clamped edges do not guarantee total image-flux conservation.
 */
export async function createBloom(device: GPUDevice) {
  const module = await compileShader(device, `${fullscreen}\n${source}`, "bloom");
  const pipeline = (entryPoint: string) =>
    device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module, entryPoint: "vertex" },
      fragment: { module, entryPoint, targets: [{ format: "rgba16float" }] },
    });
  const [downsample, upsample] = await Promise.all([pipeline("downsample"), pipeline("upsample")]);
  const sampler = device.createSampler({ minFilter: "linear", magFilter: "linear" });
  const lifetime = new DisposableStack();
  type Level = {
    readonly down: GPUTexture;
    readonly up: GPUTexture;
    readonly blend: GPUBuffer;
  };
  let target:
    | {
        readonly width: number;
        readonly height: number;
        readonly levels: readonly Level[];
        readonly resources: DisposableStack;
      }
    | undefined;
  lifetime.defer(() => target?.resources.dispose());

  function createTarget(width: number, height: number) {
    using resources = new DisposableStack();
    const texture = (w: number, h: number) =>
      resources.adopt(
        device.createTexture({
          label: "bloom radiance",
          size: [w, h],
          format: "rgba16float",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.RENDER_ATTACHMENT |
            GPUTextureUsage.COPY_SRC,
        }),
        (value) => value.destroy(),
      );
    const levels: Level[] = [];
    // A fractional last octave changes the halo continuously as the viewport is resized.
    const span = Math.max(1, Math.log2(height * 0.08));
    let w = width;
    let h = height;
    for (let level = 0; level < Math.ceil(span); level++) {
      w = Math.max(1, Math.ceil(w / 2));
      h = Math.max(1, Math.ceil(h / 2));
      const blend = resources.adopt(
        device.createBuffer({
          label: "bloom scale weight",
          size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
        (value) => value.destroy(),
      );
      device.queue.writeBuffer(
        blend,
        0,
        new Float32Array([Math.min(1, 1 / (span - level)), 0, 0, 0]),
      );
      levels.push({ down: texture(w, h), up: texture(w, h), blend });
      if (w === 1 && h === 1) {
        break;
      }
    }
    return { width, height, levels, resources: resources.move() };
  }
  function preparePass(
    input: GPUTexture,
    output: GPUTexture,
    detail?: Pick<Level, "down" | "blend">,
  ) {
    const selected = detail ? upsample : downsample;
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: input.createView() },
      { binding: 1, resource: sampler },
    ];
    if (detail) {
      entries.push(
        { binding: 2, resource: detail.down.createView() },
        { binding: 3, resource: { buffer: detail.blend } },
      );
    }
    return {
      pipeline: selected,
      bindings: device.createBindGroup({ layout: selected.getBindGroupLayout(0), entries }),
      output: output.createView(),
    };
  }
  let input: GPUTexture | undefined;
  let output: GPUTexture | undefined;
  let passes: ReturnType<typeof preparePass>[] = [];
  return {
    /**
     * Encode filtering of a borrowed radiance texture without submitting.
     *
     * @returns Borrowed output, overwritten by subsequent encodes and destroyed on
     * resize/disposal. Keep inputs alive until submitted work completes.
     */
    encode(
      encoder: GPUCommandEncoder,
      radiance: GPUTexture,
      timestamps?: { readonly querySet: GPUQuerySet; readonly begin: number; readonly end: number },
    ): GPUTexture {
      if (lifetime.disposed) {
        throw new Error("Bloom resources have been disposed.");
      }
      if (target?.width !== radiance.width || target.height !== radiance.height) {
        const next = createTarget(radiance.width, radiance.height);
        target?.resources.dispose();
        target = next;
        input = undefined;
      }
      const { levels } = target;
      if (input !== radiance) {
        passes = [];
        let result = radiance;
        for (const { down } of levels) {
          passes.push(preparePass(result, down));
          result = down;
        }
        for (let index = levels.length - 2; index >= 0; index--) {
          const level = levels[index];
          if (level) {
            passes.push(preparePass(result, level.up, level));
            result = level.up;
          }
        }
        input = radiance;
        output = result;
      }
      for (const [index, prepared] of passes.entries()) {
        const descriptor: GPURenderPassDescriptor = {
          colorAttachments: [{ view: prepared.output, loadOp: "clear", storeOp: "store" }],
        };
        if (timestamps && (index === 0 || index === passes.length - 1)) {
          descriptor.timestampWrites = {
            querySet: timestamps.querySet,
            ...(index === 0 ? { beginningOfPassWriteIndex: timestamps.begin } : {}),
            ...(index === passes.length - 1 ? { endOfPassWriteIndex: timestamps.end } : {}),
          };
        }
        const pass = encoder.beginRenderPass(descriptor);
        pass.setPipeline(prepared.pipeline);
        pass.setBindGroup(0, prepared.bindings);
        pass.draw(3);
        pass.end();
      }
      if (!output) {
        throw new Error("Missing bloom output.");
      }
      return output;
    },
    /** Release the filter's textures; the caller retains the device. */
    dispose() {
      lifetime.dispose();
    },
  };
}
