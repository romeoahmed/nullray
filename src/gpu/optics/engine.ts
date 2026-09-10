import { opticalSource, starsSource } from "./shaders.ts";
import { createJetTable } from "../../physics/synchrotron.ts";
import { createAtmosphereTable } from "../../physics/atmosphere.ts";
import { compileShader } from "../device.ts";
import { createBlackbodyTable } from "../../physics/radiation.ts";
import { createCelestialData, createSkyTexture } from "../sources/sky.ts";
import { createStructureField } from "../sources/structure.ts";
import type { Scene } from "../../scene/scene.ts";
import type { SourceAppearance } from "../../scene/appearance.ts";
import type { RayPath, RayPoint } from "../../physics/ray-path.ts";
import { writeOpticalFrame } from "./frame.ts";

/** Stored radiance and integer source-domain data, owned by the optical renderer. */
export interface OpticalImage {
  readonly radiance: GPUTexture;
  readonly q: GPUTexture;
  readonly u: GPUTexture;
  readonly domains: GPUTexture;
}

/** One geometric sample. Camera and source inputs have already passed scene validation. */
export interface OpticalSettings {
  readonly appearance: SourceAppearance;
  readonly time: number;
  readonly view: "image" | "frequency" | "order" | "domain";
  readonly jitter: readonly [number, number];
}

export interface Optics {
  /** Read a normalized image point (default centre) for the last encoded frame; callers serialize readbacks and frame changes. */
  inspect(point?: readonly [number, number]): Promise<RayPath>;
  encode(
    encoder: GPUCommandEncoder,
    scene: Scene,
    width: number,
    height: number,
    settings: OpticalSettings,
  ): OpticalImage;
  dispose(): void;
}

/** Own the native analytic-extension pipeline and all of its source/target storage. */
export async function createOptics(device: GPUDevice): Promise<Optics> {
  using owned = new DisposableStack();
  const [module, celestial] = await Promise.all([
    compileShader(device, opticalSource, "optical flow"),
    createCelestialData(),
  ]);
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "render_image" },
  });
  const starsModule = await compileShader(device, starsSource, "stellar footprints");
  const starsLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { viewDimension: "cube" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
      {
        binding: 13,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: "write-only", format: "rgba16float" },
      },
      ...[14, 15, 16].map((binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float" as const },
      })),
      { binding: 17, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "sint" } },
      { binding: 18, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ],
  });
  const starsPipeline = await device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [starsLayout] }),
    compute: { module: starsModule, entryPoint: "composite_stars" },
  });
  const inspectionPipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "inspect_ray" },
  });
  const buffer = (size: number, usage: GPUBufferUsageFlags) =>
    owned.adopt(device.createBuffer({ size, usage }), (value) => value.destroy());
  const uniform = buffer(288, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const pathBytes = 16 + 4097 * 32;
  const pathBuffer = buffer(pathBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const atmosphereData = createAtmosphereTable();
  const atmosphere = buffer(
    atmosphereData.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  device.queue.writeBuffer(atmosphere, 0, atmosphereData);
  const table = createBlackbodyTable();
  const blackbody = buffer(table.data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(blackbody, 0, table.data);
  const catalogue = celestial;
  const stars = buffer(catalogue.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(stars, 0, catalogue);
  const disk = buffer(512 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const jetTable = buffer(256 * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const field = owned.adopt(await createStructureField(device), (value) => value.dispose());
  const sky = owned.adopt(await createSkyTexture(device, field), (value) => value.destroy());
  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    mipmapFilter: "linear",
    maxAnisotropy: 8,
  });
  const values = new Float32Array(72);
  const words = new Int32Array(values.buffer);
  let target:
    | {
        readonly image: OpticalImage;
        readonly pathBindings: GPUBindGroup;
        readonly skyBindings: GPUBindGroup;
      }
    | undefined;
  let previousDisk: Float32Array | undefined;
  let jetKey: readonly number[] = [];
  let jetScale = 1;
  let targetLifetime = new DisposableStack();
  owned.defer(() => targetLifetime.dispose());
  const inspectionBindings = device.createBindGroup({
    layout: inspectionPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 2, resource: { buffer: blackbody } },
      { binding: 5, resource: { buffer: disk } },
      { binding: 7, resource: { buffer: atmosphere } },
      { binding: 10, resource: { buffer: pathBuffer } },
      { binding: 19, resource: { buffer: jetTable } },
      { binding: 20, resource: field.texture.createView() },
      { binding: 21, resource: field.sampler },
    ],
  });
  const lifetime = owned.move();
  return {
    async inspect(point = [0.5, 0.5]) {
      if (lifetime.disposed || !target) {
        throw new Error("Render a frame before inspecting a ray.");
      }
      if (!point.every((value) => Number.isFinite(value) && value >= 0 && value <= 1)) {
        throw new RangeError("The detector point must lie inside the image.");
      }
      values[50] = ((point[0] - 0.5) * target.image.radiance.width) / target.image.radiance.height;
      values[51] = point[1] - 0.5;
      device.queue.writeBuffer(uniform, 0, values);
      using readback = new DisposableStack();
      const staging = readback.adopt(
        device.createBuffer({
          size: pathBytes,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        }),
        (value) => value.destroy(),
      );
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass({ label: "selected ray inspection" });
      pass.setPipeline(inspectionPipeline);
      pass.setBindGroup(0, inspectionBindings);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.copyBufferToBuffer(pathBuffer, 0, staging, 0, pathBytes);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const floats = new Float32Array(staging.getMappedRange());
      const integers = new Int32Array(floats.buffer);
      const count = integers[0] ?? 0;
      const kind = (["unresolved", "infinity", "disk", "singularity", "source-free"] as const)[
        integers[1] ?? 0
      ];
      if (!kind || count < 1 || count > 4097) {
        throw new Error("Invalid GPU ray path.");
      }
      const points: RayPoint[] = [];
      for (let i = 0; i < count; i++) {
        const offset = 4 + i * 8;
        const coordinate = floats[offset] ?? NaN;
        const block = (
          ["exterior", "black-hole", "interior", "white-hole", "naked", "disconnected"] as const
        )[integers[offset + 4] ?? -1];
        if (!block) {
          throw new Error("Invalid GPU ray domain.");
        }
        points.push({
          radius:
            floats[offset + 3] === 0
              ? coordinate
              : coordinate === 0
                ? (integers[2] ?? 0) * Infinity
                : 1 / coordinate,
          inclination: Math.acos(Math.max(-1, Math.min(1, floats[offset + 1] ?? NaN))),
          time: floats[offset + 2] ?? NaN,
          block,
          universe: integers[offset + 5] ?? 0,
          side: integers[offset + 6] ?? 0,
          chart:
            integers[offset + 7] === 2
              ? "bifurcation-ingoing"
              : integers[offset + 7] === -2
                ? "bifurcation-outgoing"
                : integers[offset + 7] === 1
                  ? "ingoing"
                  : "outgoing",
        });
      }
      return { kind, points, equatorialCrossings: integers[3] ?? 0 };
    },
    encode(encoder, scene, width, height, { appearance, time, view, jitter }) {
      if (lifetime.disposed) {
        throw new Error("The optical renderer is disposed.");
      }
      if (
        !target ||
        target.image.radiance.width !== width ||
        target.image.radiance.height !== height
      ) {
        using next = new DisposableStack();
        const texture = (label: string, format: GPUTextureFormat) =>
          next.adopt(
            device.createTexture({
              label,
              size: [width, height],
              format,
              usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,
            }),
            (value) => value.destroy(),
          );
        const image: OpticalImage = {
          radiance: texture("radiance", "rgba16float"),
          q: texture("Stokes Q", "rgba16float"),
          u: texture("Stokes U", "rgba16float"),
          domains: texture("source domains", "rgba32sint"),
        };
        const foreground = texture("foreground radiance", "rgba16float");
        const arrivals = texture("asymptotic directions and energy", "rgba32float");
        const transmissions = texture("material transmission", "rgba16float");
        const pathBindings = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uniform } },
            { binding: 1, resource: foreground.createView() },
            { binding: 2, resource: { buffer: blackbody } },
            { binding: 3, resource: sky.createView({ dimension: "cube" }) },
            { binding: 4, resource: sampler },
            { binding: 5, resource: { buffer: disk } },
            { binding: 6, resource: image.domains.createView() },
            { binding: 7, resource: { buffer: atmosphere } },
            { binding: 8, resource: image.q.createView() },
            { binding: 9, resource: image.u.createView() },
            { binding: 10, resource: { buffer: pathBuffer } },
            { binding: 19, resource: { buffer: jetTable } },
            { binding: 11, resource: arrivals.createView() },
            { binding: 12, resource: transmissions.createView() },
            { binding: 20, resource: field.texture.createView() },
            { binding: 21, resource: field.sampler },
          ],
        });
        const skyBindings = device.createBindGroup({
          layout: starsLayout,
          entries: [
            { binding: 0, resource: { buffer: uniform } },
            { binding: 2, resource: { buffer: blackbody } },
            { binding: 3, resource: sky.createView({ dimension: "cube" }) },
            { binding: 4, resource: sampler },
            { binding: 13, resource: image.radiance.createView() },
            { binding: 14, resource: foreground.createView() },
            { binding: 15, resource: arrivals.createView() },
            { binding: 16, resource: transmissions.createView() },
            { binding: 17, resource: image.domains.createView() },
            { binding: 18, resource: { buffer: stars } },
          ],
        });
        target = { image, pathBindings, skyBindings };
        targetLifetime.dispose();
        targetLifetime = next.move();
      }
      if (previousDisk !== scene.prepared.disk.temperature) {
        previousDisk = scene.prepared.disk.temperature;
        device.queue.writeBuffer(disk, 0, scene.prepared.disk.temperature);
      }
      if (scene.jet) {
        const key = [
          scene.jet.density,
          scene.jet.field,
          scene.jet.massSolar,
          scene.jet.gammaMin,
          scene.plasma?.frequencyGHz ?? 0,
        ];
        if (key.some((value, index) => value !== jetKey[index])) {
          const spectrum = createJetTable(scene.jet, scene.plasma?.frequencyGHz);
          jetScale = spectrum.scale;
          device.queue.writeBuffer(jetTable, 0, spectrum.data);
          jetKey = key;
        }
      }
      writeOpticalFrame(values, words, scene, { appearance, time, view, jitter }, jetScale);
      device.queue.writeBuffer(uniform, 0, values);
      const pass = encoder.beginComputePass({ label: "optical flow" });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, target.pathBindings);
      pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
      pass.end();
      const backgroundPass = encoder.beginComputePass({ label: "stellar beam footprints" });
      backgroundPass.setPipeline(starsPipeline);
      backgroundPass.setBindGroup(0, target.skyBindings);
      backgroundPass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
      backgroundPass.end();
      return target.image;
    },
    dispose() {
      lifetime.dispose();
    },
  };
}
