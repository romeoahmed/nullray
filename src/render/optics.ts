import derivative from "./shaders/derivative.wgsl?raw";
import differential from "./shaders/differential.wgsl?raw";
import imageSource from "./shaders/stellar-image.wgsl?raw";
import elliptic from "./shaders/elliptic.wgsl?raw";
import arrival from "./shaders/arrival.wgsl?raw";
import equator from "./shaders/equator.wgsl?raw";
import transport from "./shaders/transport.wgsl?raw";
import radiation from "./shaders/radiation.wgsl?raw";
import orbit from "./shaders/orbit.wgsl?raw";
import tracing from "./shaders/trace.wgsl?raw";
import beamSource from "./shaders/beam.wgsl?raw";
import edgeBeamSource from "./shaders/edge-beams.wgsl?raw";
import edgeSource from "./shaders/edges.wgsl?raw";
import repairSource from "./shaders/repair.wgsl?raw";
import geometrySource from "./shaders/geometry.wgsl?raw";
import diagnostic from "./shaders/diagnostic.wgsl?raw";
import starsSource from "./shaders/stars.wgsl?raw";
import skySource from "./shaders/sky.wgsl?raw";
import { initialAppearance } from "../model/appearance.ts";
import type { SourceAppearance } from "../model/appearance.ts";
import type { Scene } from "../model/scene.ts";
import { createBlackbodyTable } from "../physics/radiation.ts";
import { createDiskProfile } from "../physics/disk.ts";
import { createBeamQueue } from "./beams.ts";
import type { BeamQueue } from "./beams.ts";
import { compileShader } from "./device.ts";
import { createSkyTexture, createCelestialData } from "./sky.ts";

/** Exact shader modules used by the renderer and benchmark source fingerprints. */
const raySource = `${elliptic}\n${equator}\n${arrival}\n${transport}\n${orbit}\n${tracing}`;
const derivativeSource = `${raySource}\n${derivative}\n${differential}\n${imageSource}`;
export const opticalSources = {
  ray: raySource,
  geometry: `${raySource}\n${geometrySource}`,
  beam: `enable subgroups;\n${derivativeSource}\n${beamSource}\n${repairSource}`,
  edgeBeam: `enable subgroups;\n${derivativeSource}\n${beamSource}\n${repairSource}\n${edgeBeamSource}`,
  edge: `enable subgroups;\n${raySource}\n${edgeSource}`,
  sky: `${radiation}\n${starsSource}\n${beamSource}\n${diagnostic}\n${skySource}`,
} as const;

/** Spectral image or inspection of retained physical endpoints. */
export type OpticalView = "image" | "frequency" | "order";

/** Borrowed optical targets and work counters, owned by Optics until resize or disposal. */
export interface OpticalImage {
  /** RGB is linear radiance; alpha records the resolved sampling fraction. */
  readonly radiance: GPUTexture;
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

/** Caller-owned query set and indices enclosing the encoded optical passes. */
export interface OpticalTimestamps {
  readonly querySet: GPUQuerySet;
  readonly begin: number;
  readonly end: number;
}

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
      /** False reuses geometry; the caller must keep scene and jitter unchanged. Resize still retraces. */
      readonly trace?: boolean;
      readonly antialias?: boolean;
      /** Offset from each pixel center, in physical image pixels. */
      readonly jitter?: readonly [number, number];
      readonly view?: OpticalView;
      readonly timestamps?: OpticalTimestamps;
    },
  ): OpticalImage;
  /** Release all owned targets and buffers; the caller retains the device. */
  dispose(): void;
}

/** Device-local optical resources, shared by the application and GPU timing workload. */
export async function createOptics(device: GPUDevice, signal?: AbortSignal): Promise<Optics> {
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
  const celestialData = createCelestialData();
  const celestial = owned.adopt(createSkyTexture(device, celestialData.levels), (texture) =>
    texture.destroy(),
  );
  const celestialView = celestial.createView({ dimension: "cube" });
  const starData = celestialData.stars;
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
        timestamps,
        time = 0,
        appearance = initialAppearance,
        trace = true,
        antialias = false,
        jitter = [0, 0],
        view = "image",
      } = options;
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
        const beams = createBeamQueue(device, resources, uniform, endpoints, {
          find: findBeams,
          repair: beam,
        });
        // Boundary work grows with image length; cap its atlases independently of full-frame storage.
        const edgeCapacity = Math.min(
          width * height,
          Math.max(32768, Math.min(65536, 8 * (width + height))),
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
          ],
        });
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
          ],
        });
        target = {
          width,
          height,
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
        const geometryPass = encoder.beginComputePass(
          timestamps
            ? {
                timestampWrites: {
                  querySet: timestamps.querySet,
                  beginningOfPassWriteIndex: timestamps.begin,
                },
              }
            : {},
        );
        geometryPass.setPipeline(geometry);
        geometryPass.setBindGroup(0, target.geometry);
        geometryPass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        geometryPass.end();
        target.beams.encode(encoder);
      }
      if (rebuildEdges) {
        const findPass = encoder.beginComputePass(
          timestamps && !trace && !resized
            ? {
                timestampWrites: {
                  querySet: timestamps.querySet,
                  beginningOfPassWriteIndex: timestamps.begin,
                },
              }
            : {},
        );
        findPass.setPipeline(findEdges);
        findPass.setBindGroup(0, target.findEdges);
        findPass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        findPass.end();
        const samplePass = encoder.beginComputePass();
        samplePass.setPipeline(sampleEdges);
        samplePass.setBindGroup(0, target.sampleEdges);
        samplePass.dispatchWorkgroups(Math.ceil((target.edgeCapacity * 16) / 64));
        samplePass.end();
        target.edgeBeams.encode(encoder);
      }
      if (antialias) {
        const shadePass = encoder.beginComputePass(
          timestamps && !trace && !resized && !rebuildEdges
            ? {
                timestampWrites: {
                  querySet: timestamps.querySet,
                  beginningOfPassWriteIndex: timestamps.begin,
                },
              }
            : {},
        );
        shadePass.setPipeline(shadeEdges);
        shadePass.setBindGroup(0, target.shadeEdges);
        shadePass.dispatchWorkgroups(Math.ceil((target.edgeCapacity * 16) / 64));
        shadePass.end();
      }
      previousAntialias = antialias;
      const skyPass = encoder.beginComputePass(
        timestamps
          ? {
              timestampWrites: {
                querySet: timestamps.querySet,
                ...(!trace && !resized && !antialias
                  ? { beginningOfPassWriteIndex: timestamps.begin }
                  : {}),
                endOfPassWriteIndex: timestamps.end,
              },
            }
          : {},
      );
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
