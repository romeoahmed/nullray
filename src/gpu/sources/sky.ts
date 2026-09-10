import { createStars, decodeFaintStars } from "../../physics/sky.ts";
import { createStarTree } from "../../physics/stars.ts";
import { compileShader } from "../device.ts";
import type { StructureField } from "./structure.ts";
import fieldSource from "../wgsl/sources/field.wgsl?raw";
import skySource from "../wgsl/sources/sky.wgsl?raw";
import faintStarsURL from "../../data/faint-stars.bin?url";

/** Decode the bundled catalogue into one caller-owned spectral source tree. */
export async function createCelestialData(): Promise<Float32Array<ArrayBuffer>> {
  const response = await fetch(faintStarsURL);
  if (!response.ok) {
    throw new Error(`Could not load the stellar catalogue: ${response.status}.`);
  }
  const faint = decodeFaintStars(await response.arrayBuffer());
  return createStarTree([...createStars(), ...faint]);
}

/** Generate a spectral cube and its solid-angle-weighted mips on the GPU. The caller owns the result. */
export async function createSkyTexture(
  device: GPUDevice,
  field: StructureField,
  size = 256,
): Promise<GPUTexture> {
  if (!Number.isSafeInteger(size) || size < 1 || size > 1024 || (size & (size - 1)) !== 0) {
    throw new RangeError("Sky face size must be a power of two between 1 and 1024.");
  }
  const module = await compileShader(device, `${fieldSource}\n${skySource}`, "spectral sky");
  const reductionLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { format: "rgba32float", viewDimension: "2d-array" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { format: "rgba16float", viewDimension: "2d-array" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float", viewDimension: "2d-array" },
      },
    ],
  });
  const [generate, reduce] = await Promise.all([
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "generate_sky" },
    }),
    device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [reductionLayout] }),
      compute: { module, entryPoint: "reduce_sky" },
    }),
  ]);
  using scratch = new DisposableStack();
  const mipLevelCount = Math.log2(size) + 1;
  const descriptor = {
    size: [size, size, 6],
    mipLevelCount,
    usage:
      GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  };
  const work = scratch.adopt(
    device.createTexture({ ...descriptor, label: "sky integration", format: "rgba32float" }),
    (value) => value.destroy(),
  );
  const output = device.createTexture({
    ...descriptor,
    label: "celestial spectra",
    format: "rgba16float",
  });
  try {
    const encoder = device.createCommandEncoder();
    for (let level = 0; level < mipLevelCount; level++) {
      const view = { dimension: "2d-array", baseMipLevel: level, mipLevelCount: 1 } as const;
      const pipeline = level === 0 ? generate : reduce;
      const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: work.createView(view) },
        { binding: 1, resource: output.createView(view) },
      ];
      if (level === 0) {
        entries.push({ binding: 20, resource: field.texture.createView() });
      } else {
        entries.push({
          binding: 2,
          resource: work.createView({ ...view, baseMipLevel: level - 1 }),
        });
      }
      const pass = encoder.beginComputePass({
        label: level === 0 ? "spectral sky" : "sky solid-angle reduction",
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(
        0,
        device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }),
      );
      const groups = Math.ceil((size >> level) / 8);
      pass.dispatchWorkgroups(groups, groups, 6);
      pass.end();
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return output;
  } catch (error) {
    output.destroy();
    throw error;
  }
}
