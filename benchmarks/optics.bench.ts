import { sourceFingerprint, benchmarkFingerprint } from "./source.ts";
import { captureOpticalImage, imageCoverage } from "./capture.ts";
import { requiredFeatures } from "../src/gpu/device.ts";
import { createBloom } from "../src/gpu/bloom.ts";
import bloomSource from "../src/gpu/wgsl/passes/bloom.wgsl?raw";
import { expect, test } from "vitest";
import { server } from "vitest/browser";
import { createOptics } from "../src/gpu/optics.ts";
import type { OpticalImage, OpticalStage } from "../src/gpu/optics.ts";
import { createBlackbodyTable } from "../src/physics/radiation.ts";
import { brightStars } from "../src/data/bright-stars.ts";
import { createStarTree } from "../src/physics/stars.ts";
import { createStars } from "../src/physics/sky.ts";
import { initialAppearance } from "../src/scene/appearance.ts";
import { createScene } from "../src/scene/scene.ts";

const hex = (bytes: ArrayBuffer) => new Uint8Array(bytes).toHex();

interface Workload {
  readonly name: string;
  readonly width: number;
  readonly height: number;
  readonly reuse: "none" | "sources" | "geometry";
  readonly spin?: number;
  readonly charge?: number;
  readonly radius?: number;
}
const workloads: readonly Workload[] = [
  { name: "full-frame", width: 1280, height: 720, reuse: "none" },
  { name: "view-sampling", width: 1280, height: 720, reuse: "sources" },
  { name: "lighting", width: 1280, height: 720, reuse: "geometry" },
  { name: "bloom", width: 1280, height: 720, reuse: "geometry" },
  { name: "retrograde", width: 640, height: 360, reuse: "none", spin: -0.7, charge: 0.2 },
  { name: "charged", width: 640, height: 360, reuse: "none", spin: 0.2, charge: 0.9 },
  {
    name: "near-extremal",
    width: 640,
    height: 360,
    reuse: "none",
    spin: Math.fround(1 - 2 ** -24),
    charge: 0,
  },
  { name: "distant", width: 640, height: 360, reuse: "none", radius: 200 },
];

test.for(workloads)(
  "$name at $width × $height, GPU timestamps",
  { timeout: 180000 },
  async (workload, { bench, skip, annotate }) => {
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
    const optics = await createOptics(device);
    owned.defer(() => optics.dispose());
    const bloom = owned.adopt(await createBloom(device), (value) => value.dispose());
    const { width, height, reuse } = workload;
    const trace = reuse !== "geometry";
    const scene = createScene({
      space: { spin: workload.spin ?? 0.7, charge: workload.charge ?? 0.2 },
      observer: {
        radius: workload.radius ?? 18,
        inclination: 1.2,
        fieldOfView: 2 * Math.atan(0.6),
        azimuth: 0,
      },
    });
    if (!scene.ok) {
      throw new Error(scene.error);
    }
    const queryCount = workload.name === "bloom" ? 2 : 256;
    const stageSamples: Partial<Record<OpticalStage, number[]>> = {};
    const querySet = owned.adopt(
      device.createQuerySet({ type: "timestamp", count: queryCount }),
      (value) => value.destroy(),
    );
    const resolve = buffer(queryCount * 8, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
    const readback = buffer(queryCount * 8 + 32, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const stellarWorkSamples: number[][] = [];
    let discarded = 0;
    const discardedIntervals: string[][] = [];
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
          const stages: OpticalStage[] = [];
          if (workload.name === "bloom" && image) {
            bloomImage = bloom.encode(encoder, image.radiance, { querySet, begin: 0, end: 1 });
          } else {
            image = optics.encode(encoder, scene.value, width, height, {
              reuse,
              antialias: true,
              profile: (stage: OpticalStage) => {
                const index = stages.length * 2;
                if (index + 1 >= queryCount) {
                  throw new Error("Optical stages exceed the timestamp query capacity.");
                }
                stages.push(stage);
                return {
                  querySet,
                  beginningOfPassWriteIndex: index,
                  endOfPassWriteIndex: index + 1,
                };
              },
            });
          }
          const usedQueries = workload.name === "bloom" ? 2 : stages.length * 2;
          encoder.resolveQuerySet(querySet, 0, usedQueries, resolve, 0);
          encoder.copyBufferToBuffer(resolve, 0, readback, 0, usedQueries * 8);
          if (image) {
            encoder.copyBufferToBuffer(image.stellarStatistics, 0, readback, queryCount * 8, 32);
          }
          device.queue.submit([encoder.finish()]);
          // Timestamp retries reuse one staging buffer and must remain serial.
          // oxlint-disable-next-line no-await-in-loop
          await readback.mapAsync(GPUMapMode.READ);
          const mappedTiming = readback.getMappedRange();
          const times = new BigUint64Array(mappedTiming, 0, usedQueries).slice();
          if (image) {
            stellarWorkSamples.push(Array.from(new Uint32Array(mappedTiming, queryCount * 8, 8)));
          }
          const begin = times[0];
          const end = times[usedQueries - 1];
          readback.unmap();
          const validStages = stages.every((_, i) => {
            const start = times[2 * i],
              finish = times[2 * i + 1];
            return start !== undefined && finish !== undefined && start > 0n && finish >= start;
          });
          if (
            begin !== undefined &&
            end !== undefined &&
            begin > 0n &&
            end > begin &&
            validStages
          ) {
            const totals = new Map<OpticalStage, number>();
            stages.forEach((stage, i) => {
              totals.set(
                stage,
                (totals.get(stage) ?? 0) +
                  Number((times[2 * i + 1] ?? 0n) - (times[2 * i] ?? 0n)) / 1e6,
              );
            });
            for (const [stage, elapsed] of totals) {
              (stageSamples[stage] ??= []).push(elapsed);
            }
            return { overriddenDuration: Number(end - begin) / 1e6 };
          }
          discarded++;
          if (discardedIntervals.length < 8) {
            discardedIntervals.push(Array.from(times, String));
          }
        }
        const validation = await device.popErrorScope();
        throw new Error(
          validation?.message ??
            `GPU timestamp measurement unavailable after 64 discarded samples: ${JSON.stringify(discardedIntervals)}`,
        );
      },
    ).run({ iterations: 64, time: 1000, warmupIterations: 16, warmupTime: 250 });
    if (!image) {
      throw new Error("The benchmark did not produce an optical image.");
    }
    const capture = await captureOpticalImage(device, image, bloomImage);
    const {
      radiance: radianceData,
      endpoints: endpointData,
      beamStatistics: repairStatistics,
      edgeStatistics,
      edgeBeamStatistics,
      stellarStatistics,
    } = capture;
    const { unresolvedPixels, unresolvedCoverage, geometryFailures } = imageCoverage(
      radianceData,
      endpointData,
      width,
    );
    const edgeCandidates = edgeStatistics[0] ?? 0;
    let filteredImage: { width: number; height: number; sha256: string; file: string } | undefined;
    if (bloomImage && capture.filtered) {
      expect(capture.filtered.every(Number.isFinite)).toBe(true);
      filteredImage = {
        width: bloomImage.width,
        height: bloomImage.height,
        sha256: hex(await crypto.subtle.digest("SHA-256", capture.filtered)),
        file: `gpu-${workload.name}-filtered.f16`,
      };
      await server.commands.writeFile(
        `test-results/bench/${filteredImage.file}`,
        new Uint8Array(capture.filtered.buffer).toBase64(),
        "base64",
      );
    }
    const [digest, tableDigest, endpointDigest, radianceDigest, starDigest] = await Promise.all([
      sourceFingerprint(),
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
      schema: 1,
      reuse,
      stageSamplesMs: stageSamples,
      stageSampleScope:
        "Paired timestamps on every executed pass; totals include all refinement trials and warmup. The reported interval spans first pass start to final pass end. Instrumentation can alter scheduling.",
      requiredFeatures: [...requiredFeatures, "timestamp-query"],
      enabledFeatures: [...device.features],
      filteredImage,
      bloomSHA256: hex(
        await crypto.subtle.digest("SHA-256", new TextEncoder().encode(bloomSource)),
      ),
      sourceSHA256: hex(digest),
      harnessSHA256: hex(await benchmarkFingerprint()),
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
            ? reuse === "none"
              ? "uncached physical stellar search + binning + geometry + beam refinement + boundary quadrature + radiation; excludes initialization, presentation and submission"
              : "cached physical stellar images; timed binning + geometry + beam refinement + boundary quadrature + radiation; excludes initialization, presentation and submission"
            : "production radiation only, retained geometry; excludes preparation, presentation and submission",
      unresolvedPixels,
      unresolvedCoverage,
      stellarStatistics,
      stellarWorkSamples,
      stellarWorkSampleScope: "Every submitted frame, including warmup and discarded timestamps.",
      geometryFailures,
      totalPixels: width * height,
      readbacks: {
        radiance: `gpu-${workload.name}-radiance.f16`,
        endpoints: `gpu-${workload.name}-endpoints.f32`,
        layout: "row-major RGBA, little-endian binary16 radiance and binary32 endpoints",
      },
      discardedTimestampSamples: discarded,
      discardedIntervals,
    };
    expect(await device.popErrorScope()).toBeNull();
    const context = JSON.stringify(metadata, null, 2);
    await server.commands.writeFile(
      `test-results/bench/gpu-${workload.name}-context.json`,
      context,
    );
    expect(new Set(stellarWorkSamples.map((sample) => JSON.stringify(sample))).size).toBe(1);
    await annotate("GPU benchmark context", {
      body: context,
      bodyEncoding: "utf-8",
      contentType: "application/json",
    });
  },
);
