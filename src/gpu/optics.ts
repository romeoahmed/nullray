import { createStellarSearch } from "./stellar-search.ts";
import type { StellarRaster, StellarSearchStage } from "./stellar-search.ts";
import { opticalSources } from "./shaders.ts";
export { opticalSources } from "./shaders.ts";
import { observerFrameBytes, writeObserverFrame } from "./observer.ts";
import { initialAppearance } from "../scene/appearance.ts";
import type { SourceAppearance } from "../scene/appearance.ts";
import type { Scene } from "../scene/scene.ts";
import { createBlackbodyTable } from "../physics/radiation.ts";
import { createDiskProfile } from "../physics/disk.ts";
import { createBeamQueue } from "./beams.ts";
import type { BeamQueue } from "./beams.ts";
import { compileShader } from "./device.ts";
import { createSkyTexture, createCelestialData } from "./sky.ts";
import type { CelestialData } from "./sky.ts";

/** Spectral image or inspection of retained physical endpoints. */
export type OpticalView = "image" | "frequency" | "order";

/** Borrowed optical targets and work counters, owned by Optics until resize or disposal. */
export interface OpticalImage {
  /** RGB is linear radiance; alpha records the resolved sampling fraction. */
  readonly radiance: GPUTexture;
  /** Source-search candidates, failures, mixed cells and overflow; not a completeness certificate. */
  readonly stellarStatistics: GPUBuffer;
  /** Sky: (cos theta, phi, E, positive branch). Disk: (r, phi, frequency ratio, -3-order); 0 is capture and -2 is unresolved. */
  readonly endpoints: GPUTexture;
  /** Disk coordinate-time displacement from the observer, in geometric time units. */
  readonly emissionTime: GPUTexture;
  /** Three u32 counters: candidate beams, differentiated rays, successful repairs. */
  readonly beamStatistics: GPUBuffer;
  readonly beamCapacity: number;
  /** Candidate pixels, unresolved geometry samples, missing derivatives, other radiation failures. */
  readonly edgeStatistics: GPUBuffer;
  readonly edgeCapacity: number;
  readonly edgeBeamStatistics: GPUBuffer;
  readonly edgeBeamCapacity: number;
}

/** Stable GPU pass names for optional benchmark timestamp instrumentation. */
export type OpticalStage =
  | StellarSearchStage
  | "bin-stars"
  | "geometry"
  | "find-beams"
  | "repair-beams"
  | "find-edges"
  | "sample-edges"
  | "find-edge-beams"
  | "repair-edge-beams"
  | "shade-edges"
  | "sky";

/** Device-local pipeline owner; encode mutates its reusable targets and submits no command buffer. */
export interface Optics {
  /** Encode geometry and spectral filtering. Returned textures remain owned until resize/disposal. */
  encode(
    encoder: GPUCommandEncoder,
    scene: Scene,
    width: number,
    height: number,
    options?: {
      readonly time?: number;
      readonly appearance?: SourceAppearance;
      /** Reuse source images across jitter (default); "geometry" requires unchanged scene/jitter. "none" forces all optics. */
      readonly reuse?: "none" | "sources" | "geometry";
      readonly antialias?: boolean;
      /** Offset from each pixel center, in physical image pixels. */
      readonly jitter?: readonly [number, number];
      readonly view?: OpticalView;
      /** Optional paired timestamps per pass; no callback or readback runs in normal rendering. */
      readonly profile?: (stage: OpticalStage) => GPUComputePassTimestampWrites;
    },
  ): OpticalImage;
  /** Release all owned targets and buffers; the caller retains the device. */
  dispose(): void;
}

/**
 * Device-local optical resources, shared by the application and GPU timing workload.
 * Optional prepared celestial data is borrowed until initialization resolves;
 * uploads copy it without mutation or transfer of its CPU buffers.
 */
export async function createOptics(
  device: GPUDevice,
  initialization: {
    readonly signal?: AbortSignal | undefined;
    readonly celestial?: CelestialData;
  } = {},
): Promise<Optics> {
  const { signal } = initialization;
  signal?.throwIfAborted();
  using owned = new DisposableStack();
  let target:
    | (OpticalImage & {
        readonly width: number;
        readonly height: number;
        readonly geometry: GPUBindGroup;
        readonly beams: BeamQueue;
        readonly edgeBeams: BeamQueue;
        readonly findEdges: GPUBindGroup;
        readonly sampleEdges: GPUBindGroup;
        readonly shadeEdges: GPUBindGroup;
        readonly sky: GPUBindGroup;
        readonly stellarRaster: StellarRaster;
        readonly resources: DisposableStack;
      })
    | undefined;
  const destroyTarget = () => {
    target?.resources.dispose();
    target = undefined;
  };
  owned.defer(destroyTarget);
  const [geometryModule, beamModule, edgeBeamModule, edgeModule, skyModule] = await Promise.all([
    compileShader(device, opticalSources.geometry, "optical geometry"),
    compileShader(device, opticalSources.beam, "local sky beams"),
    compileShader(device, opticalSources.edgeBeam, "boundary sky beams"),
    compileShader(device, opticalSources.edge, "boundary samples"),
    compileShader(device, opticalSources.sky, "spectral sky"),
  ]);
  const [
    geometry,
    findBeams,
    beam,
    findEdgeBeams,
    repairEdgeBeams,
    findEdges,
    sampleEdges,
    sky,
    shadeEdges,
  ] = await Promise.all([
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: geometryModule },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: beamModule, entryPoint: "find_beams" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: beamModule, entryPoint: "repair_beams" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: edgeBeamModule, entryPoint: "find_edge_beams" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: edgeBeamModule, entryPoint: "repair_edge_beams" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: edgeModule, entryPoint: "find_edges" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: edgeModule, entryPoint: "sample_edges" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: skyModule, entryPoint: "resolve_sky" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: skyModule, entryPoint: "shade_edges" },
    }),
  ]);
  signal?.throwIfAborted();
  const celestialData = initialization.celestial ?? createCelestialData();
  const celestial = owned.adopt(createSkyTexture(device, celestialData.levels), (texture) =>
    texture.destroy(),
  );
  const celestialView = celestial.createView({ dimension: "cube" });
  const starData = celestialData.stars;
  const stellarSearch = owned.adopt(await createStellarSearch(device, starData), (value) =>
    value.dispose(),
  );
  signal?.throwIfAborted();
  const stars = owned.adopt(
    device.createBuffer({
      label: "point-source tree",
      size: starData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    (buffer) => buffer.destroy(),
  );
  device.queue.writeBuffer(stars, 0, starData);

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    mipmapFilter: "linear",
    maxAnisotropy: 4,
  });
  const uniform = owned.adopt(
    device.createBuffer({
      label: "frame",
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
    (buffer) => buffer.destroy(),
  );
  const table = createBlackbodyTable();
  const observerFrame = owned.adopt(
    device.createBuffer({
      label: "binary64 observer metric",
      size: observerFrameBytes,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
    (buffer) => buffer.destroy(),
  );
  const observerValues = new Float64Array(observerFrameBytes / 8);
  const blackbody = owned.adopt(
    device.createBuffer({
      label: "blackbody",
      size: table.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    (buffer) => buffer.destroy(),
  );
  device.queue.writeBuffer(blackbody, 0, table.data);
  const radiationUniform = owned.adopt(
    device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    (buffer) => buffer.destroy(),
  );
  const radiationFrame = new Float32Array(12);
  const diskProfile = owned.adopt(
    device.createBuffer({
      size: 512 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    (value) => value.destroy(),
  );
  let profileScene: Scene | undefined;
  const frame = new Float32Array(24);
  let previousAntialias = false;
  const lifetime = owned.move();
  return {
    encode(encoder, scene, width, height, options = {}) {
      const {
        profile,
        time = 0,
        appearance = initialAppearance,
        reuse = "sources",
        antialias = false,
        jitter = [0, 0],
        view = "image",
      } = options;
      const trace = reuse !== "geometry";
      const resized = !target || target.width !== width || target.height !== height;
      if (lifetime.disposed) {
        throw new Error("The optical renderer has been disposed.");
      }
      if (
        !Number.isSafeInteger(width) ||
        !Number.isSafeInteger(height) ||
        width < 1 ||
        height < 1 ||
        Math.max(width, height) > device.limits.maxTextureDimension2D
      ) {
        throw new RangeError(
          "Optical dimensions must be positive integers within the device limit.",
        );
      }
      if (!target || target.width !== width || target.height !== height) {
        destroyTarget();
        using resources = new DisposableStack();
        const endpoints = resources.adopt(
          device.createTexture({
            label: "optical endpoints",
            size: [width, height],
            format: "rgba32float",
            usage:
              GPUTextureUsage.STORAGE_BINDING |
              GPUTextureUsage.TEXTURE_BINDING |
              GPUTextureUsage.COPY_SRC,
          }),
          (texture) => texture.destroy(),
        );
        const radiance = resources.adopt(
          device.createTexture({
            label: "scene radiance",
            size: [width, height],
            format: "rgba16float",
            usage:
              GPUTextureUsage.STORAGE_BINDING |
              GPUTextureUsage.TEXTURE_BINDING |
              GPUTextureUsage.COPY_SRC,
          }),
          (texture) => texture.destroy(),
        );
        const emissionTime = resources.adopt(
          device.createTexture({
            size: [width, height],
            format: "r32float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
          }),
          (texture) => texture.destroy(),
        );
        const beams = createBeamQueue(device, resources, uniform, observerFrame, endpoints, {
          find: findBeams,
          repair: beam,
        });
        // Varying disk interiors occupy area, not only boundary length. Allow
        // 1/32 of the raster to retain 16 samples, capped at 2^17 pixels (56 MiB
        // for endpoint, delay, and radiance atlases before row padding).
        const edgeCapacity = Math.min(
          width * height,
          Math.max(32768, Math.min(131072, Math.ceil((width * height) / 32))),
        );
        const edgeBeamCapacity = Math.max(4096, Math.min(16384, 3 * (width + height)));
        const edgeColumns = Math.ceil(Math.sqrt(edgeCapacity));
        const edgeRows = Math.ceil(edgeCapacity / edgeColumns);
        const edgeIndex = resources.adopt(
          device.createTexture({
            size: [width, height],
            format: "r32uint",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
          }),
          (value) => value.destroy(),
        );
        const edgeSamples = resources.adopt(
          device.createTexture({
            size: [edgeColumns * 4, edgeRows * 4],
            format: "rgba32float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
          }),
          (value) => value.destroy(),
        );
        const edgeRadiance = resources.adopt(
          device.createTexture({
            size: [edgeColumns * 4, edgeRows * 4],
            format: "rgba16float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
          }),
          (value) => value.destroy(),
        );
        const edgeTimes = resources.adopt(
          device.createTexture({
            size: [edgeColumns * 4, edgeRows * 4],
            format: "r32float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
          }),
          (value) => value.destroy(),
        );
        const edgeStatistics = resources.adopt(
          device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
          }),
          (value) => value.destroy(),
        );
        const edgePixels = resources.adopt(
          device.createBuffer({ size: edgeCapacity * 8, usage: GPUBufferUsage.STORAGE }),
          (value) => value.destroy(),
        );
        const edgeBeams = createBeamQueue(
          device,
          resources,
          uniform,
          observerFrame,
          edgeSamples,
          {
            find: findEdgeBeams,
            repair: repairEdgeBeams,
          },
          {
            capacity: edgeBeamCapacity,
            find: [
              { binding: 7, resource: { buffer: edgePixels } },
              { binding: 8, resource: { buffer: edgeStatistics } },
            ],
            repair: [{ binding: 7, resource: { buffer: edgePixels } }],
          },
        );
        const geometryBindings = device.createBindGroup({
          layout: geometry.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uniform } },
            { binding: 1, resource: endpoints.createView() },
            { binding: 2, resource: emissionTime.createView() },
            { binding: 3, resource: { buffer: observerFrame } },
          ],
        });
        const stellarRaster = resources.adopt(stellarSearch.createRaster(width, height), (value) =>
          value.dispose(),
        );
        const skyBindings = device.createBindGroup({
          layout: sky.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: endpoints.createView() },
            { binding: 1, resource: radiance.createView() },
            { binding: 2, resource: { buffer: blackbody } },
            { binding: 3, resource: celestialView },
            { binding: 4, resource: sampler },
            { binding: 5, resource: { buffer: radiationUniform } },
            { binding: 6, resource: { buffer: diskProfile } },
            { binding: 7, resource: emissionTime.createView() },
            { binding: 8, resource: { buffer: stars } },
            { binding: 9, resource: beams.index.createView() },
            { binding: 10, resource: { buffer: beams.beams } },
            { binding: 11, resource: edgeIndex.createView() },
            { binding: 15, resource: edgeRadiance.createView() },
            { binding: 19, resource: { buffer: uniform } },
            { binding: 20, resource: { buffer: stellarSearch.partition } },
            { binding: 21, resource: { buffer: stellarSearch.refined } },
            { binding: 23, resource: { buffer: stellarRaster.heads } },
            { binding: 24, resource: { buffer: stellarRaster.links } },
          ],
        });
        target = {
          width,
          height,
          stellarRaster,
          stellarStatistics: stellarSearch.statistics,
          endpoints,
          emissionTime,
          radiance,
          geometry: geometryBindings,
          beams,
          edgeBeams,
          beamCapacity: beams.capacity,
          beamStatistics: beams.statistics,
          edgeBeamCapacity: edgeBeams.capacity,
          edgeBeamStatistics: edgeBeams.statistics,
          edgeCapacity,
          edgeStatistics,
          findEdges: device.createBindGroup({
            layout: findEdges.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: uniform } },
              { binding: 1, resource: endpoints.createView() },
              { binding: 2, resource: edgeIndex.createView() },
              { binding: 3, resource: { buffer: edgeStatistics } },
              { binding: 4, resource: { buffer: edgePixels } },
              { binding: 7, resource: emissionTime.createView() },
            ],
          }),
          sampleEdges: device.createBindGroup({
            layout: sampleEdges.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: uniform } },
              { binding: 3, resource: { buffer: edgeStatistics } },
              { binding: 4, resource: { buffer: edgePixels } },
              { binding: 5, resource: edgeSamples.createView() },
              { binding: 6, resource: edgeTimes.createView() },
              { binding: 8, resource: { buffer: observerFrame } },
            ],
          }),
          shadeEdges: device.createBindGroup({
            layout: shadeEdges.getBindGroupLayout(0),
            entries: [
              { binding: 2, resource: { buffer: blackbody } },
              { binding: 3, resource: celestialView },
              { binding: 4, resource: sampler },
              { binding: 5, resource: { buffer: radiationUniform } },
              { binding: 6, resource: { buffer: diskProfile } },
              { binding: 8, resource: { buffer: stars } },
              { binding: 12, resource: edgeSamples.createView() },
              { binding: 13, resource: edgeTimes.createView() },
              { binding: 14, resource: edgeRadiance.createView() },
              { binding: 16, resource: { buffer: edgeStatistics } },
              { binding: 17, resource: edgeBeams.index.createView() },
              { binding: 18, resource: { buffer: edgeBeams.beams } },
              { binding: 19, resource: { buffer: uniform } },
              { binding: 20, resource: { buffer: stellarSearch.partition } },
              { binding: 25, resource: { buffer: edgePixels } },
            ],
          }),
          sky: skyBindings,
          resources: resources.move(),
        };
      }
      const { space, observer } = scene;
      if (
        !profileScene ||
        profileScene.space.spin !== space.spin ||
        profileScene.space.charge !== space.charge ||
        profileScene.diskInner !== scene.diskInner ||
        profileScene.diskOuter !== scene.diskOuter
      ) {
        device.queue.writeBuffer(
          diskProfile,
          0,
          createDiskProfile(space, scene.diskInner, scene.diskOuter).temperature,
        );
        profileScene = scene;
      }
      const emissionEpoch = Math.floor(time / 24);
      radiationFrame.set([
        space.spin,
        space.charge,
        scene.diskInner,
        time - 24 * emissionEpoch,
        appearance.diskTemperature,
        appearance.diskStructure,
        appearance.skyBrightness,
        scene.diskOuter,
        view === "image" ? 0 : view === "frequency" ? 1 : 2,
        antialias ? 1 : 0,
        target.edgeCapacity,
        emissionEpoch,
      ]);
      device.queue.writeBuffer(radiationUniform, 0, radiationFrame);
      const passDescriptor = (stage: OpticalStage): GPUComputePassDescriptor =>
        profile ? { timestampWrites: profile(stage) } : {};
      const rebuildEdges = antialias && (trace || resized || !previousAntialias);
      if (rebuildEdges) {
        encoder.clearBuffer(target.edgeStatistics);
        encoder.clearBuffer(target.edgeBeamStatistics);
      } else if (antialias) {
        encoder.clearBuffer(target.edgeStatistics, 4, 12);
      }
      if (trace || resized) {
        encoder.clearBuffer(target.beamStatistics);
        frame.set([
          width,
          height,
          jitter[0],
          jitter[1],
          observer.radius,
          observer.inclination,
          2 * Math.tan(observer.fieldOfView / 2),
          observer.azimuth,
          space.spin,
          space.charge,
          scene.diskInner,
          scene.diskOuter,
          ...scene.camera.forward,
          0,
          ...scene.camera.up,
          0,
          ...scene.camera.right,
          0,
        ]);
        device.queue.writeBuffer(uniform, 0, frame);
        writeObserverFrame(frame, observerValues);
        device.queue.writeBuffer(observerFrame, 0, observerValues);
        stellarSearch.encode(encoder, frame, passDescriptor, reuse !== "none");
        target.stellarRaster.encode(encoder, passDescriptor("bin-stars"));
        const geometryPass = encoder.beginComputePass(passDescriptor("geometry"));
        geometryPass.setPipeline(geometry);
        geometryPass.setBindGroup(0, target.geometry);
        geometryPass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        geometryPass.end();
        target.beams.encode(
          encoder,
          profile
            ? {
                find: profile("find-beams"),
                repair: profile("repair-beams"),
              }
            : undefined,
        );
      }
      if (rebuildEdges) {
        const findPass = encoder.beginComputePass(passDescriptor("find-edges"));
        findPass.setPipeline(findEdges);
        findPass.setBindGroup(0, target.findEdges);
        findPass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        findPass.end();
        const samplePass = encoder.beginComputePass(passDescriptor("sample-edges"));
        samplePass.setPipeline(sampleEdges);
        samplePass.setBindGroup(0, target.sampleEdges);
        samplePass.dispatchWorkgroups(Math.ceil((target.edgeCapacity * 16) / 64));
        samplePass.end();
        target.edgeBeams.encode(
          encoder,
          profile
            ? {
                find: profile("find-edge-beams"),
                repair: profile("repair-edge-beams"),
              }
            : undefined,
        );
      }
      if (antialias) {
        const shadePass = encoder.beginComputePass(passDescriptor("shade-edges"));
        shadePass.setPipeline(shadeEdges);
        shadePass.setBindGroup(0, target.shadeEdges);
        shadePass.dispatchWorkgroups(Math.ceil((target.edgeCapacity * 16) / 64));
        shadePass.end();
      }
      previousAntialias = antialias;
      const skyPass = encoder.beginComputePass(passDescriptor("sky"));
      skyPass.setPipeline(sky);
      skyPass.setBindGroup(0, target.sky);
      skyPass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
      skyPass.end();
      return target;
    },
    dispose() {
      lifetime.dispose();
    },
  };
}
