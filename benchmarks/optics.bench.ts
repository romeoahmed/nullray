import { requiredFeatures } from "../src/render/device.ts";
import { createBloom } from "../src/render/bloom.ts";
import bloomSource from "../src/render/shaders/bloom.wgsl?raw";
import { expect, test } from "vitest";
import { server } from "vitest/browser";
import { createOptics, opticalSources } from "../src/render/optics.ts";
import type { OpticalImage } from "../src/render/optics.ts";
import { createBlackbodyTable } from "../src/physics/radiation.ts";
import { brightStars } from "../src/data/bright-stars.ts";
import { createStarTree } from "../src/physics/stars.ts";
import { createStars } from "../src/physics/sky.ts";
import { initialAppearance } from "../src/model/appearance.ts";
import { createScene } from "../src/model/scene.ts";

const hex = (bytes: ArrayBuffer) => new Uint8Array(bytes).toHex();

test.for([
  { name: "optics", width: 128, height: 72, trace: true },
  { name: "retrograde", width: 128, height: 72, trace: true, spin: -0.7, charge: 0.2 },
  { name: "charged", width: 128, height: 72, trace: true, spin: 0.2, charge: 0.9 },
  { name: "optics-360p", width: 640, height: 360, trace: true },
  { name: "optics-720p", width: 1280, height: 720, trace: true },
  { name: "radiation", width: 1280, height: 720, trace: false },
  { name: "bloom", width: 1280, height: 720, trace: false },
])("$name at $width × $height, GPU timestamps", async (workload, { bench, skip, annotate }) => {
  if (!navigator.gpu) {
    throw new Error("WebGPU is required for this benchmark.");
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    throw new Error("No WebGPU adapter is available.");
  }
  if (!adapter.features.has("timestamp-query")) {
    skip("GPU timestamps unavailable; no substitute CPU timing is reported.");
  }
  const device = await adapter.requestDevice({
    requiredFeatures: [...requiredFeatures, "timestamp-query"],
  });
  device.pushErrorScope("validation");
  using owned = new DisposableStack();
  owned.defer(() => device.destroy());
  const buffer = (size: number, usage: GPUBufferUsageFlags) =>
    owned.adopt(device.createBuffer({ size, usage }), (value) => value.destroy());
  const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
  const bloom = owned.adopt(await createBloom(device), (value) => value.dispose());
  const { width, height, trace } = workload;
  const scene = createScene({
    space: { spin: workload.spin ?? 0.7, charge: workload.charge ?? 0.2 },
    observer: { radius: 18, inclination: 1.2, fieldOfView: 2 * Math.atan(0.6), azimuth: 0 },
  });
  if (!scene.ok) {
    throw new Error(scene.error);
  }
  const querySet = owned.adopt(device.createQuerySet({ type: "timestamp", count: 2 }), (value) =>
    value.destroy(),
  );
  const resolve = buffer(16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
  const readback = buffer(16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const colorBytes = width * height * 8;
  const pixels = buffer(colorBytes * 3 + 48, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  let discarded = 0;
  let image: OpticalImage | undefined;
  let bloomImage: GPUTexture | undefined;
  if (!trace) {
    const preparation = device.createCommandEncoder();
    image = optics.encode(preparation, scene.value, width, height, { antialias: true });
    device.queue.submit([preparation.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  await bench(
    `${workload.name} / ${width * height} pixels / GPU ms`,
    { writeResult: `test-results/bench/gpu-${workload.name}.json` },
    async () => {
      // Timestamp resets and timer quantization can produce unusable deltas.
      for (let attempt = 0; attempt < 64; attempt++) {
        const encoder = device.createCommandEncoder();
        if (workload.name === "bloom" && image) {
          bloomImage = bloom.encode(encoder, image.radiance, { querySet, begin: 0, end: 1 });
        } else {
          image = optics.encode(encoder, scene.value, width, height, {
            trace,
            antialias: true,
            timestamps: { querySet, begin: 0, end: 1 },
          });
        }
        encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
        encoder.copyBufferToBuffer(resolve, 0, readback, 0, 16);
        device.queue.submit([encoder.finish()]);
        // Timestamp retries reuse one staging buffer and must remain serial.
        // oxlint-disable-next-line no-await-in-loop
        await readback.mapAsync(GPUMapMode.READ);
        const times = new BigUint64Array(readback.getMappedRange());
        const begin = times[0];
        const end = times[1];
        readback.unmap();
        if (begin !== undefined && end !== undefined && begin > 0n && end > begin) {
          return { overriddenDuration: Number(end - begin) / 1e6 };
        }
        discarded++;
      }
      const validation = await device.popErrorScope();
      throw new Error(
        validation?.message ?? "GPU timestamp measurement unavailable after 64 discarded samples.",
      );
    },
  ).run({ iterations: 64, time: 1000, warmupIterations: 16, warmupTime: 250 });
  if (!image) {
    throw new Error("The benchmark did not produce an optical image.");
  }
  const encoder = device.createCommandEncoder();
  const bloomStride = bloomImage ? Math.ceil((bloomImage.width * 8) / 256) * 256 : 0;
  const bloomReadback = bloomImage
    ? buffer(bloomStride * bloomImage.height, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ)
    : undefined;
  if (bloomImage && bloomReadback) {
    encoder.copyTextureToBuffer(
      { texture: bloomImage },
      { buffer: bloomReadback, bytesPerRow: bloomStride },
      [bloomImage.width, bloomImage.height],
    );
  }
  encoder.copyTextureToBuffer(
    { texture: image.radiance },
    { buffer: pixels, bytesPerRow: width * 8 },
    [width, height],
  );
  encoder.copyTextureToBuffer(
    { texture: image.endpoints },
    { buffer: pixels, offset: colorBytes, bytesPerRow: width * 16 },
    [width, height],
  );
  encoder.copyBufferToBuffer(image.beamStatistics, 0, pixels, colorBytes * 3, 16);
  encoder.copyBufferToBuffer(image.edgeStatistics, 0, pixels, colorBytes * 3 + 16, 16);
  encoder.copyBufferToBuffer(image.edgeBeamStatistics, 0, pixels, colorBytes * 3 + 32, 16);
  device.queue.submit([encoder.finish()]);
  await pixels.mapAsync(GPUMapMode.READ);
  const mappedPixels = pixels.getMappedRange();
  const values = new Float16Array(mappedPixels, 0, width * height * 4);
  const endpoints = new Float32Array(mappedPixels, colorBytes, width * height * 4);
  let unresolved = 0;
  const unresolvedCoverage: { x: number; y: number; resolvedFraction: number }[] = [];
  const geometryFailures: Record<string, number> = {};
  for (let i = 0; i < width * height; i++) {
    const red = values[i * 4] ?? Number.NaN;
    const green = values[i * 4 + 1] ?? Number.NaN;
    const blue = values[i * 4 + 2] ?? Number.NaN;
    expect(Number.isFinite(red + green + blue)).toBe(true);
    if (endpoints[i * 4 + 3] === -2) {
      const stage = String(endpoints[i * 4]);
      geometryFailures[stage] = (geometryFailures[stage] ?? 0) + 1;
    }
    if ((values[i * 4 + 3] ?? 0) < 1) {
      unresolved++;
      unresolvedCoverage.push({
        x: i % width,
        y: Math.floor(i / width),
        resolvedFraction: values[i * 4 + 3] ?? 0,
      });
    }
  }
  const endpointData = endpoints.slice();
  const radianceData = values.slice();
  const repairStatistics = new Uint32Array(mappedPixels, colorBytes * 3, 3).slice();
  const edgeStatistics = new Uint32Array(mappedPixels, colorBytes * 3 + 16, 4).slice();
  const edgeCandidates = edgeStatistics[0] ?? 0;
  const edgeBeamStatistics = new Uint32Array(mappedPixels, colorBytes * 3 + 32, 3).slice();
  pixels.unmap();
  let filteredImage: { width: number; height: number; sha256: string; file: string } | undefined;
  if (bloomImage && bloomReadback) {
    await bloomReadback.mapAsync(GPUMapMode.READ);
    const mapped = new Float16Array(bloomReadback.getMappedRange());
    const packed = new Float16Array(bloomImage.width * bloomImage.height * 4);
    for (let y = 0; y < bloomImage.height; y++) {
      packed.set(
        mapped.subarray((y * bloomStride) / 2, (y * bloomStride) / 2 + bloomImage.width * 4),
        y * bloomImage.width * 4,
      );
    }
    bloomReadback.unmap();
    expect(packed.every(Number.isFinite)).toBe(true);
    filteredImage = {
      width: bloomImage.width,
      height: bloomImage.height,
      sha256: hex(await crypto.subtle.digest("SHA-256", packed)),
      file: `gpu-${workload.name}-filtered.f16`,
    };
    await server.commands.writeFile(
      `test-results/bench/${filteredImage.file}`,
      new Uint8Array(packed.buffer).toBase64(),
      "base64",
    );
  }
  const [digest, tableDigest, endpointDigest, radianceDigest, starDigest] = await Promise.all([
    crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(
        `${opticalSources.geometry}\n${opticalSources.beam}\n${opticalSources.edge}\n${opticalSources.edgeBeam}\n${opticalSources.sky}`,
      ),
    ),
    crypto.subtle.digest("SHA-256", createBlackbodyTable().data.slice()),
    crypto.subtle.digest("SHA-256", endpointData),
    crypto.subtle.digest("SHA-256", radianceData),
    crypto.subtle.digest("SHA-256", createStarTree(createStars())),
  ]);
  // Retain exact readbacks outside measured callbacks for equal-quality comparisons.
  {
    await server.commands.writeFile(
      `test-results/bench/gpu-${workload.name}-radiance.f16`,
      new Uint8Array(radianceData.buffer).toBase64(),
      "base64",
    );
    await server.commands.writeFile(
      `test-results/bench/gpu-${workload.name}-endpoints.f32`,
      new Uint8Array(endpointData.buffer).toBase64(),
      "base64",
    );
  }
  const metadata = {
    schema: 11,
    requiredFeatures: [...requiredFeatures, "timestamp-query"],
    enabledFeatures: [...device.features],
    filteredImage,
    bloomSHA256: hex(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(bloomSource))),
    sourceSHA256: hex(digest),
    blackbodySHA256: hex(tableDigest),
    endpointSHA256: hex(endpointDigest),
    radianceSHA256: hex(radianceDigest),
    adapter: {
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      device: adapter.info.device,
      description: adapter.info.description,
    },
    browser: navigator.userAgent,
    scene: scene.value,
    appearance: initialAppearance,
    width,
    height,
    sampling: {
      minimumIterations: 64,
      minimumTimeMs: 1000,
      warmupIterations: 16,
      warmupTimeMs: 250,
    },
    primarySamplesPerPixel: 1,
    beamRepairs: {
      capacity: image.beamCapacity,
      candidates: repairStatistics[0],
      additionalRays: repairStatistics[1],
      repaired: repairStatistics[2],
      cached: !trace,
    },
    boundarySamples: {
      capacity: image.edgeCapacity,
      candidates: edgeCandidates,
      additionalRays: Math.min(edgeCandidates, image.edgeCapacity) * 16,
      overflowPixels: Math.max(0, edgeCandidates - image.edgeCapacity),
      cached: !trace,
      geometryFailures: edgeStatistics[1],
      derivativeFailures: edgeStatistics[2],
      radiationFailures: edgeStatistics[3],
    },
    boundaryBeamRepairs: {
      capacity: image.edgeBeamCapacity,
      candidates: edgeBeamStatistics[0],
      additionalRays: edgeBeamStatistics[1],
      repaired: edgeBeamStatistics[2],
      cached: !trace,
    },
    sky: {
      faceSize: 128,
      stars: brightStars.length,
      catalogue: "HYG 4.1, V <= 6.5",
      starTreeSHA256: hex(starDigest),
      pointSourceTree: true,
      mipLevels: 8,
      anisotropy: 4,
    },
    scope:
      workload.name === "bloom"
        ? "production six-scale bloom filter on retained radiance; excludes optics, presentation and submission"
        : trace
          ? "production geometry + beam refinement + boundary quadrature + radiation; excludes presentation and submission"
          : "production radiation only, retained geometry; excludes preparation, presentation and submission",
    unresolvedPixels: unresolved,
    unresolvedCoverage,
    geometryFailures,
    totalPixels: width * height,
    readbacks: {
      radiance: `gpu-${workload.name}-radiance.f16`,
      endpoints: `gpu-${workload.name}-endpoints.f32`,
      layout: "row-major RGBA, little-endian binary16 radiance and binary32 endpoints",
    },
    discardedTimestampSamples: discarded,
  };
  expect(await device.popErrorScope()).toBeNull();
  const context = JSON.stringify(metadata, null, 2);
  await server.commands.writeFile(`test-results/bench/gpu-${workload.name}-context.json`, context);
  await annotate("GPU benchmark context", {
    body: context,
    bodyEncoding: "utf-8",
    contentType: "application/json",
  });
});
