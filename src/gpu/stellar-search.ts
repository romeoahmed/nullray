import work from "./wgsl/lensing/stellar-work.wgsl?raw";
import composition from "./wgsl/imaging/stellar-raster.wgsl?raw";
import ownership from "./wgsl/imaging/stellar-partition.wgsl?raw";
import radiation from "./wgsl/imaging/radiation.wgsl?raw";
import stars from "./wgsl/imaging/source-tree.wgsl?raw";
import patch from "./wgsl/imaging/stellar-patch.wgsl?raw";
import beam from "./wgsl/lensing/beam.wgsl?raw";
import beamProjection from "./wgsl/lensing/beam-projection.wgsl?raw";
import projection from "./wgsl/lensing/projection64.wgsl?raw";
import criticalUpdate from "./wgsl/lensing/critical-step64.wgsl?raw";
import criticalDerivative from "./wgsl/lensing/critical-tangent64.wgsl?raw";
import critical from "./wgsl/lensing/critical64.wgsl?raw";
import search from "./wgsl/lensing/stellar-search.wgsl?raw";
import refinement from "./wgsl/lensing/stellar-refine.wgsl?raw";
import { opticalSources } from "./shaders.ts";
import { criticalDirection } from "../physics/critical.ts";
import { createBlackbodyTable } from "../physics/radiation.ts";
import { compileShader } from "./device.ts";
import { observerFrameBytes, writeObserverFrame } from "./observer.ts";

/** Executed modules retained in acceptance fingerprints. */
export const stellarSearchSource = `${opticalSources.derivative}\n${ownership}\n${radiation}\n${stars}\n${patch}\n${beam}\n${beamProjection}\n${projection}\n${critical}\n${criticalDerivative}\n${criticalUpdate}\n${search}\n${work}\n${refinement}`;

/** Finite search work, not a claim that all physical images lie inside this strip. */
export interface StellarSearchOptions {
  readonly columns: number;
  readonly rows: number;
  readonly capacity: number;
  readonly maximumOffset: number;
  readonly minimumOffset: number;
}

/** Stable GPU pass names for optical profiling. */
export type StellarSearchStage =
  | "sample-stars"
  | "classify-stars"
  | "count-stars"
  | "scan-star-blocks"
  | "allocate-stars"
  | "write-stars"
  | "prepare-mixed-stars"
  | "sample-mixed-stars"
  | "prepare-stars"
  | "prepare-star-rays"
  | "refine-stars";

/** Caller-owned detector lists borrowing confirmed images from their parent search. */
export interface StellarRaster {
  readonly heads: GPUBuffer;
  readonly links: GPUBuffer;
  encode(encoder: GPUCommandEncoder, descriptor?: GPUComputePassDescriptor): void;
  dispose(): void;
}

/** Physical image storage remains owned by the search until disposal. */
export interface StellarSearch {
  readonly partition: GPUBuffer;
  /** The caller disposes the raster before disposing its parent search. */
  createRaster(width: number, height: number): StellarRaster;
  readonly samples: GPUBuffer;
  readonly seeds: GPUBuffer;
  readonly refined: GPUBuffer;
  readonly statistics: GPUBuffer;
  /** Reuse physical images across detector jitter; return whether the optical search ran. */
  encode(
    encoder: GPUCommandEncoder,
    frame: Float32Array,
    passDescriptor?: (stage: StellarSearchStage) => GPUComputePassDescriptor,
    reuse?: boolean,
  ): boolean;
  dispose(): void;
}

/**
 * Own the GPU critical strip and source-directed candidate queue.
 * Prepared star-tree data is copied during initialization; the caller retains its CPU buffers.
 */
export async function createStellarSearch(
  device: GPUDevice,
  starData: Float32Array<ArrayBuffer>,
  options: StellarSearchOptions = {
    columns: 256,
    rows: 128,
    capacity: 32768,
    maximumOffset: 0.25,
    minimumOffset: 1e-8,
  },
): Promise<StellarSearch> {
  const { columns, rows, capacity, maximumOffset, minimumOffset } = options;
  if (
    starData.byteLength < 32 ||
    starData.byteLength % 32 !== 0 ||
    starData.byteLength > device.limits.maxStorageBufferBindingSize
  ) {
    throw new RangeError(
      "The prepared stellar tree must contain complete nodes within the device storage limit.",
    );
  }
  if (
    ![columns, rows, capacity].every(Number.isSafeInteger) ||
    columns < 3 ||
    rows < 1 ||
    capacity < 1 ||
    !Number.isFinite(maximumOffset) ||
    !(minimumOffset >= 2 ** -126 && maximumOffset > minimumOffset && maximumOffset <= 1e30) ||
    Math.max(columns * (rows + 1) * 32, columns * rows * 160, capacity * 1456) >
      device.limits.maxStorageBufferBindingSize ||
    capacity > device.limits.maxComputeWorkgroupsPerDimension ||
    Math.ceil((columns * (rows + 1)) / 64) > device.limits.maxComputeWorkgroupsPerDimension ||
    Math.ceil((columns * rows * 5) / 64) > device.limits.maxComputeWorkgroupsPerDimension
  ) {
    throw new RangeError("Stellar search dimensions and offsets must fit the device work limits.");
  }
  using owned = new DisposableStack();
  const rasterModule = await compileShader(
    device,
    `${ownership}\n${composition}`,
    "stellar detector bins",
  );
  const rasterPipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: rasterModule, entryPoint: "bin_stellar_images" },
  });
  const module = await compileShader(device, stellarSearchSource, "stellar image search");
  const [
    samplePipeline,
    countPipeline,
    scanPipeline,
    allocatePipeline,
    writePipeline,
    dispatchPipeline,
    preparePipeline,
    refinePipeline,
    mixedDispatchPipeline,
    mixedSamplePipeline,
    classifyPipeline,
  ] = await Promise.all([
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "sample_critical_strip" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "count_stellar_seeds" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "scan_stellar_seed_blocks" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "allocate_stellar_seed_blocks" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "write_stellar_seeds" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "prepare_stellar_refinement" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "prepare_stellar_rays" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "refine_stellar_images" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "prepare_mixed_stellar_cells" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "sample_mixed_stellar_cells" },
    }),
    device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "classify_stellar_cells" },
    }),
  ]);
  const makeBuffer = (size: number, usage: GPUBufferUsageFlags, label: string) =>
    owned.adopt(device.createBuffer({ size, usage, label }), (value) => value.destroy());
  const uniform = makeBuffer(
    96,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    "stellar search frame",
  );
  const observer = makeBuffer(
    observerFrameBytes,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    "stellar search observer",
  );
  const columnBuffer = makeBuffer(
    columns * 64,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    "critical curve columns",
  );
  const partition = makeBuffer(
    columns * 16,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    "stellar image ownership",
  );
  const samples = makeBuffer(
    columns * (rows + 1) * 32,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    "critical strip samples",
  );
  const mixedCells = makeBuffer(columns * rows * 4, GPUBufferUsage.STORAGE, "mixed stellar cells");
  const mixedSlots = makeBuffer(
    columns * rows * 4,
    GPUBufferUsage.STORAGE,
    "mixed stellar cell slots",
  );
  const mixedSamples = makeBuffer(
    columns * rows * 160,
    GPUBufferUsage.STORAGE,
    "mixed stellar midpoint samples",
  );
  const seeds = makeBuffer(
    capacity * 32,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    "stellar image candidates",
  );
  const statistics = makeBuffer(
    32,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    "stellar search work",
  );
  const refined = makeBuffer(
    capacity * 64,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    "refined stellar images",
  );
  // Host-shareable differentiated paths are laid out explicitly in stellar-work.wgsl.
  const imageWork = makeBuffer(capacity * 1456, GPUBufferUsage.STORAGE, "stellar prepared rays");
  const iterationBuffer = makeBuffer(
    32 * 256,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    "stellar iteration indices",
  );
  const iterationValues = new Uint32Array(32 * 64);
  for (let iteration = 0; iteration < 32; iteration++) {
    iterationValues.set([iteration, 32, 8, 8], iteration * 64);
  }
  device.queue.writeBuffer(iterationBuffer, 0, iterationValues);
  const dispatch = makeBuffer(
    32,
    GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    "stellar refinement dispatch",
  );
  const trials = makeBuffer(8 + capacity * 8, GPUBufferUsage.STORAGE, "unfinished stellar roots");
  const table = createBlackbodyTable();
  const blackbody = makeBuffer(
    table.data.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    "stellar spectral table",
  );
  device.queue.writeBuffer(blackbody, 0, table.data);
  const sourceBuffer = makeBuffer(
    starData.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    "stellar search sources",
  );
  device.queue.writeBuffer(sourceBuffer, 0, starData);
  const settings = makeBuffer(
    16,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    "stellar search settings",
  );
  const settingsWords = new Uint32Array([columns, rows, 0, 0]);
  new Float32Array(settingsWords.buffer).set([Math.log(maximumOffset), Math.log(minimumOffset)], 2);
  device.queue.writeBuffer(settings, 0, settingsWords);
  const sampleGroup = device.createBindGroup({
    layout: samplePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: observer } },
      { binding: 3, resource: { buffer: columnBuffer } },
      { binding: 4, resource: { buffer: samples } },
      { binding: 7, resource: { buffer: statistics } },
      { binding: 8, resource: { buffer: settings } },
    ],
  });
  const seedOffsets = makeBuffer(
    (columns * rows + Math.ceil((columns * rows) / 256)) * 8,
    GPUBufferUsage.STORAGE,
    "deterministic stellar seed offsets",
  );
  const seedBindings: GPUBindGroupEntry[] = [
    { binding: 4, resource: { buffer: samples } },
    { binding: 5, resource: { buffer: sourceBuffer } },
    { binding: 6, resource: { buffer: seeds } },
    { binding: 7, resource: { buffer: statistics } },
    { binding: 8, resource: { buffer: settings } },
    { binding: 14, resource: { buffer: mixedSamples } },
    { binding: 15, resource: { buffer: mixedSlots } },
    { binding: 16, resource: { buffer: seedOffsets } },
  ];
  const countGroup = device.createBindGroup({
    layout: countPipeline.getBindGroupLayout(0),
    entries: seedBindings,
  });
  const writeGroup = device.createBindGroup({
    layout: writePipeline.getBindGroupLayout(0),
    entries: seedBindings,
  });
  const scanGroup = device.createBindGroup({
    layout: scanPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 8, resource: { buffer: settings } },
      { binding: 16, resource: { buffer: seedOffsets } },
    ],
  });
  const allocateGroup = device.createBindGroup({
    layout: allocatePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 6, resource: { buffer: seeds } },
      { binding: 7, resource: { buffer: statistics } },
      { binding: 8, resource: { buffer: settings } },
      { binding: 16, resource: { buffer: seedOffsets } },
    ],
  });
  const classifyGroup = device.createBindGroup({
    layout: classifyPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 4, resource: { buffer: samples } },
      { binding: 7, resource: { buffer: statistics } },
      { binding: 8, resource: { buffer: settings } },
      { binding: 13, resource: { buffer: mixedCells } },
      { binding: 15, resource: { buffer: mixedSlots } },
    ],
  });
  const dispatchGroups = Array.from({ length: 32 }, (_, iteration) =>
    device.createBindGroup({
      layout: dispatchPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 7, resource: { buffer: statistics } },
        { binding: 9, resource: { buffer: refined } },
        { binding: 10, resource: { buffer: dispatch } },
        { binding: 12, resource: { buffer: iterationBuffer, offset: iteration * 256, size: 16 } },
        { binding: 17, resource: { buffer: trials } },
      ],
    }),
  );
  const mixedDispatchGroup = device.createBindGroup({
    layout: mixedDispatchPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 7, resource: { buffer: statistics } },
      { binding: 10, resource: { buffer: dispatch } },
      { binding: 13, resource: { buffer: mixedCells } },
    ],
  });
  const mixedSampleGroup = device.createBindGroup({
    layout: mixedSamplePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: observer } },
      { binding: 3, resource: { buffer: columnBuffer } },
      { binding: 7, resource: { buffer: statistics } },
      { binding: 8, resource: { buffer: settings } },
      { binding: 13, resource: { buffer: mixedCells } },
      { binding: 14, resource: { buffer: mixedSamples } },
    ],
  });
  const prepareGroups = Array.from({ length: 32 }, (_, iteration) =>
    device.createBindGroup({
      layout: preparePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: observer } },
        { binding: 6, resource: { buffer: seeds } },
        { binding: 7, resource: { buffer: statistics } },
        { binding: 8, resource: { buffer: settings } },
        { binding: 9, resource: { buffer: refined } },
        { binding: 11, resource: { buffer: imageWork } },
        { binding: 12, resource: { buffer: iterationBuffer, offset: iteration * 256, size: 16 } },
        { binding: 17, resource: { buffer: trials } },
      ],
    }),
  );
  const refineGroups = Array.from({ length: 32 }, (_, iteration) =>
    device.createBindGroup({
      layout: refinePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 2, resource: { buffer: blackbody } },
        { binding: 5, resource: { buffer: sourceBuffer } },
        { binding: 7, resource: { buffer: statistics } },
        { binding: 9, resource: { buffer: refined } },
        { binding: 11, resource: { buffer: imageWork } },
        { binding: 12, resource: { buffer: iterationBuffer, offset: iteration * 256, size: 16 } },
        { binding: 17, resource: { buffer: trials } },
      ],
    }),
  );
  const columnValues = new Float64Array(columns * 8);
  const partitionValues = new Float32Array(columns * 4);
  const partitionPoints: number[][] = [];
  const observerValues = new Float64Array(observerFrameBytes / 8);
  let previousFrame: Float32Array | undefined;
  const lifetime = owned.move();
  return {
    partition,
    createRaster(width, height) {
      if (
        !Number.isSafeInteger(width) ||
        !Number.isSafeInteger(height) ||
        width < 1 ||
        height < 1 ||
        width * height * 4 > device.limits.maxStorageBufferBindingSize
      ) {
        throw new RangeError("Stellar detector dimensions must fit the device storage limit.");
      }
      if (lifetime.disposed) {
        throw new Error("The stellar search has been disposed.");
      }
      using resources = new DisposableStack();
      const heads = resources.adopt(
        device.createBuffer({
          size: width * height * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          label: "stellar detector heads",
        }),
        (value) => value.destroy(),
      );
      const links = resources.adopt(
        device.createBuffer({
          size: capacity * 4 * 16,
          usage: GPUBufferUsage.STORAGE,
          label: "stellar detector links",
        }),
        (value) => value.destroy(),
      );
      const bindings = device.createBindGroup({
        layout: rasterPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 19, resource: { buffer: uniform } },
          { binding: 20, resource: { buffer: partition } },
          { binding: 21, resource: { buffer: refined } },
          { binding: 22, resource: { buffer: statistics } },
          { binding: 23, resource: { buffer: heads } },
          { binding: 24, resource: { buffer: links } },
        ],
      });
      const rasterLifetime = resources.move();
      return {
        heads,
        links,
        encode(encoder, descriptor) {
          if (lifetime.disposed || rasterLifetime.disposed) {
            throw new Error("The stellar raster has been disposed.");
          }
          encoder.clearBuffer(heads);
          const pass = encoder.beginComputePass(descriptor);
          pass.setPipeline(rasterPipeline);
          pass.setBindGroup(0, bindings);
          pass.dispatchWorkgroups(Math.ceil(capacity / 64));
          pass.end();
        },
        dispose() {
          rasterLifetime.dispose();
        },
      };
    },
    samples,
    seeds,
    refined,
    statistics,
    encode(encoder, frame, passDescriptor, reuse = true) {
      if (lifetime.disposed) {
        throw new Error("The stellar search has been disposed.");
      }
      if (frame.length !== 24 || !frame.every(Number.isFinite)) {
        throw new RangeError("Stellar search requires one finite quantized optical frame.");
      }
      device.queue.writeBuffer(uniform, 0, frame);
      const changed =
        !reuse ||
        !previousFrame ||
        frame.some(
          (value, index) => index !== 2 && index !== 3 && value !== previousFrame?.[index],
        );
      if (!changed) {
        return false;
      }
      const space = { spin: frame[8] ?? NaN, charge: frame[9] ?? NaN };
      const radius = frame[4] ?? NaN;
      const inclination = frame[5] === Math.fround(Math.PI) ? Math.PI : (frame[5] ?? NaN);
      partitionPoints.length = 0;
      let partitionValid = true;
      for (let column = 0; column < 2 * columns; column++) {
        const point = criticalDirection(space, radius, inclination, (Math.PI * column) / columns);
        if (point.kind === "point") {
          const tangent = Math.hypot(point.direction[1], point.direction[2]);
          if (!(tangent > 0)) {
            partitionValid = false;
          }
          if (column % 2 === 0) {
            partitionPoints.push([
              Math.atan2(point.direction[2], point.direction[1]),
              point.direction[0] / tangent,
              0,
              1,
            ]);
          }
        } else {
          partitionValid = false;
        }
        columnValues.set(
          point.kind === "point" ? [...point.direction, 1] : [0, 0, 0, 0],
          column * 4,
        );
      }
      partitionValues.fill(0);
      if (partitionValid) {
        partitionPoints.sort((first, second) => (first[0] ?? 0) - (second[0] ?? 0));
        for (const [index, point] of partitionPoints.entries()) {
          partitionValues.set(point, index * 4);
        }
      }
      device.queue.writeBuffer(partition, 0, partitionValues);
      writeObserverFrame(frame, observerValues);
      device.queue.writeBuffer(observer, 0, observerValues);
      device.queue.writeBuffer(columnBuffer, 0, columnValues);
      encoder.clearBuffer(statistics);
      const sampling = encoder.beginComputePass(passDescriptor?.("sample-stars"));
      sampling.setPipeline(samplePipeline);
      sampling.setBindGroup(0, sampleGroup);
      sampling.dispatchWorkgroups(Math.ceil((columns * (rows + 1)) / 64));
      sampling.end();
      const classifying = encoder.beginComputePass(passDescriptor?.("classify-stars"));
      classifying.setPipeline(classifyPipeline);
      classifying.setBindGroup(0, classifyGroup);
      classifying.dispatchWorkgroups(Math.ceil((columns * rows) / 64));
      classifying.end();
      const mixedDispatchPass = encoder.beginComputePass(passDescriptor?.("prepare-mixed-stars"));
      mixedDispatchPass.setPipeline(mixedDispatchPipeline);
      mixedDispatchPass.setBindGroup(0, mixedDispatchGroup);
      mixedDispatchPass.dispatchWorkgroups(1);
      mixedDispatchPass.end();
      const mixedSampling = encoder.beginComputePass(passDescriptor?.("sample-mixed-stars"));
      mixedSampling.setPipeline(mixedSamplePipeline);
      mixedSampling.setBindGroup(0, mixedSampleGroup);
      mixedSampling.dispatchWorkgroupsIndirect(dispatch, 16);
      mixedSampling.end();
      for (const [pipeline, bindings, groups, stage] of [
        [countPipeline, countGroup, Math.ceil((columns * rows) / 64), "count-stars"],
        [scanPipeline, scanGroup, Math.ceil((columns * rows) / 256), "scan-star-blocks"],
        [allocatePipeline, allocateGroup, 1, "allocate-stars"],
        [writePipeline, writeGroup, Math.ceil((columns * rows) / 64), "write-stars"],
      ] as const) {
        const pass = encoder.beginComputePass(passDescriptor?.(stage));
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindings);
        pass.dispatchWorkgroups(groups);
        pass.end();
      }
      for (let iteration = 0; iteration < 32; iteration++) {
        const dispatchGroup = dispatchGroups[iteration];
        const prepareGroup = prepareGroups[iteration];
        const refineGroup = refineGroups[iteration];
        if (!dispatchGroup || !prepareGroup || !refineGroup) {
          throw new Error("Missing stellar iteration bindings.");
        }
        const preparing = encoder.beginComputePass(passDescriptor?.("prepare-stars"));
        preparing.setPipeline(dispatchPipeline);
        preparing.setBindGroup(0, dispatchGroup);
        preparing.dispatchWorkgroups(1);
        preparing.end();
        const preparingRays = encoder.beginComputePass(passDescriptor?.("prepare-star-rays"));
        preparingRays.setPipeline(preparePipeline);
        preparingRays.setBindGroup(0, prepareGroup);
        preparingRays.dispatchWorkgroupsIndirect(dispatch, 16);
        preparingRays.end();
        const refining = encoder.beginComputePass(passDescriptor?.("refine-stars"));
        refining.setPipeline(refinePipeline);
        refining.setBindGroup(0, refineGroup);
        refining.dispatchWorkgroupsIndirect(dispatch, 0);
        refining.end();
      }
      previousFrame = frame.slice();
      return true;
    },
    dispose() {
      lifetime.dispose();
    },
  };
}
