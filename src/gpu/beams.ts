/** Bounded derivative work owned by the enclosing optical target's disposable stack. */
export interface BeamQueue {
  readonly capacity: number;
  readonly index: GPUTexture;
  readonly beams: GPUBuffer;
  readonly statistics: GPUBuffer;
  /** Encode candidate selection and one cooperative workgroup per retained ray. Clear statistics first. */
  encode(
    encoder: GPUCommandEncoder,
    timestamps?: {
      readonly find: GPUComputePassTimestampWrites;
      readonly repair: GPUComputePassTimestampWrites;
    },
  ): void;
}

/** Allocate one bounded derivative queue in its optical target's resource lifetime. */
export function createBeamQueue(
  device: GPUDevice,
  resources: DisposableStack,
  uniform: GPUBuffer,
  observerFrame: GPUBuffer,
  map: GPUTexture,
  pipelines: {
    readonly find: GPUComputePipeline;
    readonly repair: GPUComputePipeline;
  },
  extra: {
    readonly capacity?: number;
    readonly find?: readonly GPUBindGroupEntry[];
    readonly repair?: readonly GPUBindGroupEntry[];
  } = {},
): BeamQueue {
  const capacity = Math.min(map.width * map.height, extra.capacity ?? 4096);
  const buffer = (size: number, usage: GPUBufferUsageFlags = GPUBufferUsage.STORAGE) =>
    resources.adopt(device.createBuffer({ size, usage }), (value) => value.destroy());
  const index = resources.adopt(
    device.createTexture({
      size: [map.width, map.height],
      format: "r32uint",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    }),
    (value) => value.destroy(),
  );
  const statistics = buffer(
    16,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  );
  const pixels = buffer(capacity * 8);
  const beams = buffer(capacity * 32);
  const find = device.createBindGroup({
    layout: pipelines.find.getBindGroupLayout(0),
    entries: [
      { binding: 1, resource: map.createView() },
      { binding: 2, resource: index.createView() },
      { binding: 4, resource: { buffer: statistics } },
      { binding: 5, resource: { buffer: pixels } },
      ...(extra.find ?? []),
    ],
  });
  const repairEntries: GPUBindGroupEntry[] = [
    { binding: 1, resource: map.createView() },
    { binding: 2, resource: index.createView() },
    { binding: 3, resource: { buffer: beams } },
    { binding: 4, resource: { buffer: statistics } },
    { binding: 5, resource: { buffer: pixels } },
    { binding: 0, resource: { buffer: uniform } },
    { binding: 6, resource: { buffer: observerFrame } },
    ...(extra.repair ?? []),
  ];
  const repair = device.createBindGroup({
    layout: pipelines.repair.getBindGroupLayout(0),
    entries: repairEntries,
  });
  return {
    capacity,
    index,
    beams,
    statistics,
    encode(encoder, timestamps) {
      const findPass = encoder.beginComputePass(
        timestamps ? { timestampWrites: timestamps.find } : {},
      );
      findPass.setPipeline(pipelines.find);
      findPass.setBindGroup(0, find);
      findPass.dispatchWorkgroups(Math.ceil(map.width / 8), Math.ceil(map.height / 8));
      findPass.end();
      const repairPass = encoder.beginComputePass(
        timestamps ? { timestampWrites: timestamps.repair } : {},
      );
      repairPass.setPipeline(pipelines.repair);
      repairPass.setBindGroup(0, repair);
      repairPass.dispatchWorkgroups(capacity);
      repairPass.end();
    },
  };
}
