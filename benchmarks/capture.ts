import type { OpticalImage } from "../src/gpu/optics.ts";

/** Copy retained benchmark outputs after timing, including arbitrary-width texture row padding. */
export async function captureOpticalImage(
  device: GPUDevice,
  image: OpticalImage,
  filtered?: GPUTexture,
) {
  const layouts = [
    { texture: image.radiance, texelBytes: 8 },
    { texture: image.endpoints, texelBytes: 16 },
    ...(filtered ? [{ texture: filtered, texelBytes: 8 }] : []),
  ];
  let bytes = 0;
  const copies = layouts.map(({ texture, texelBytes }) => {
    const stride = Math.ceil((texture.width * texelBytes) / 256) * 256;
    const offset = bytes;
    bytes += stride * texture.height;
    return { texture, texelBytes, stride, offset };
  });
  const statisticsOffset = bytes;
  using owned = new DisposableStack();
  const staging = owned.adopt(
    device.createBuffer({
      size: bytes + 80,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
    (value) => value.destroy(),
  );
  const encoder = device.createCommandEncoder();
  for (const { texture, stride, offset } of copies) {
    encoder.copyTextureToBuffer({ texture }, { buffer: staging, bytesPerRow: stride, offset }, [
      texture.width,
      texture.height,
    ]);
  }
  for (const [index, buffer] of [
    image.beamStatistics,
    image.edgeStatistics,
    image.edgeBeamStatistics,
    image.stellarStatistics,
  ].entries()) {
    encoder.copyBufferToBuffer(
      buffer,
      0,
      staging,
      statisticsOffset + index * 16,
      index === 3 ? 32 : 16,
    );
  }
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const mapped = staging.getMappedRange();
  const packed = copies.map(({ texture, texelBytes, stride, offset }) => {
    const rowBytes = texture.width * texelBytes;
    const result = new Uint8Array(rowBytes * texture.height);
    for (let y = 0; y < texture.height; y++) {
      result.set(new Uint8Array(mapped, offset + y * stride, rowBytes), y * rowBytes);
    }
    return result.buffer;
  });
  const statistics = new Uint32Array(mapped, statisticsOffset, 20).slice();
  staging.unmap();
  const [radiance, endpoints, bloom] = packed;
  if (!radiance || !endpoints) {
    throw new Error("The benchmark capture is incomplete.");
  }
  return {
    radiance: new Float16Array(radiance),
    endpoints: new Float32Array(endpoints),
    filtered: bloom ? new Float16Array(bloom) : undefined,
    beamStatistics: statistics.subarray(0, 3),
    edgeStatistics: statistics.subarray(4, 8),
    edgeBeamStatistics: statistics.subarray(8, 11),
    stellarStatistics: Array.from(statistics.subarray(12, 20)),
  };
}

/** Describe incomplete output without treating its render time as successful image quality. */
export function imageCoverage(radiance: Float16Array, endpoints: Float32Array, width: number) {
  const geometryFailures: Record<string, number> = {};
  const unresolvedCoverage: { x: number; y: number; resolvedFraction: number }[] = [];
  if (!radiance.every(Number.isFinite) || !endpoints.every(Number.isFinite)) {
    throw new Error("The benchmark produced nonfinite radiance or endpoint records.");
  }
  for (let index = 0; index < radiance.length / 4; index++) {
    if (endpoints[index * 4 + 3] === -2) {
      const stage = String(endpoints[index * 4]);
      geometryFailures[stage] = (geometryFailures[stage] ?? 0) + 1;
    }
    const resolvedFraction = radiance[index * 4 + 3] ?? NaN;
    if (resolvedFraction < 0 || resolvedFraction > 1) {
      throw new Error("The benchmark produced an invalid resolved sample fraction.");
    }
    if (resolvedFraction < 1) {
      unresolvedCoverage.push({
        x: index % width,
        y: Math.floor(index / width),
        resolvedFraction,
      });
    }
  }
  return { geometryFailures, unresolvedCoverage, unresolvedPixels: unresolvedCoverage.length };
}
