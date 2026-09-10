import { test } from "./compute.ts";
import { describe, expect } from "vitest";
import { createOptics } from "../../src/gpu/optics.ts";
import { compileShader } from "../../src/gpu/device.ts";
import { createAppearance, initialAppearance } from "../../src/scene/appearance.ts";
import { initialScene, createScene } from "../../src/scene/scene.ts";
import { server } from "vitest/browser";
import distantSky from "../fixtures/distant-sky-arrival.json" with { type: "json" };
import { referenceRay, referenceSky } from "../reference/sky.ts";
import { presets } from "../../src/scene/presets.ts";
import { createPhotoExporter } from "../../src/gpu/photograph.ts";
import presentation from "../../src/gpu/wgsl/passes/present.wgsl?raw";
import { createRenderer } from "../../src/gpu/renderer.ts";
import { createAccumulation, pixelJitter } from "../../src/gpu/accumulation.ts";
import { sceneFrame } from "../support/frames.ts";
import { sourceFingerprint } from "../../benchmarks/source.ts";
import photographicBoundary from "../fixtures/photographic-boundaries.json" with { type: "json" };

const decode = (value: number) => {
  const channel = value / 255;
  return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
};

describe("Optical cache", () => {
  test("source animation changes disk light while retaining geometry and sky", async ({
    device,
  }) => {
    using owned = new DisposableStack();

    device.pushErrorScope("validation");
    const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
    const width = 32;
    const height = 24;
    const count = width * height;
    const colorBytes = count * 8;
    const staging = owned.adopt(
      device.createBuffer({
        size: colorBytes * 3 + 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      (value) => value.destroy(),
    );
    async function frame(time: number, trace: boolean, appearance = initialAppearance) {
      const encoder = device.createCommandEncoder();
      const image = optics.encode(encoder, initialScene, width, height, {
        time,
        reuse: trace ? "sources" : "geometry",
        appearance,
      });
      encoder.copyTextureToBuffer(
        { texture: image.radiance },
        { buffer: staging, bytesPerRow: width * 8 },
        [width, height],
      );
      encoder.copyBufferToBuffer(image.beamStatistics, 0, staging, colorBytes * 3, 16);
      encoder.copyTextureToBuffer(
        { texture: image.endpoints },
        { buffer: staging, offset: colorBytes, bytesPerRow: width * 16 },
        [width, height],
      );
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const mapped = staging.getMappedRange();
      const color = new Float16Array(mapped, 0, count * 4).slice();
      const endpoints = new Float32Array(mapped, colorBytes, count * 4).slice();
      const beams = new Uint32Array(mapped, colorBytes * 3, 3).slice();
      staging.unmap();
      return { color, endpoints, beams };
    }
    const first = await frame(0, true);
    const later = await frame(100, false);
    const repeated = await frame(100, false);
    expect(later.endpoints).toEqual(first.endpoints);
    expect(repeated.color).toEqual(later.color);
    expect(first.beams[0]).toBeGreaterThan(0);
    expect(later.beams).toEqual(first.beams);
    expect(repeated.beams).toEqual(first.beams);
    let disk = 0;
    let changed = 0;
    let sky = 0;
    for (let i = 0; i < count; i++) {
      const tag = first.endpoints[i * 4 + 3] ?? Number.NaN;
      const before = first.color[i * 4] ?? Number.NaN;
      const after = later.color[i * 4] ?? Number.NaN;
      expect(Number.isFinite(before + after)).toBe(true);
      const resolved = first.color[i * 4 + 3];
      expect([0, 1]).toContain(resolved);
      if (resolved === 0) {
        expect(tag).toBeGreaterThan(0);
        expect(before).toBe(0);
        expect(first.color[i * 4 + 1]).toBe(0);
      }
      if (tag <= -3) {
        disk++;
        if (Math.abs(before - after) > 0.001) {
          changed++;
        }
      } else if (tag > 0) {
        sky++;
        expect(after).toBe(before);
      }
    }
    expect(disk).toBeGreaterThan(20);
    expect(changed).toBeGreaterThan(disk / 2);
    expect(sky).toBeGreaterThan(0);
    const longTime = await frame(240_000_000, false);
    const longTimeStep = await frame(240_000_000.125, false);
    expect(longTimeStep.endpoints).toEqual(first.endpoints);
    expect(longTimeStep.beams).toEqual(first.beams);
    let longTimeChanges = 0;
    for (let index = 0; index < count; index++) {
      const tag = first.endpoints[index * 4 + 3] ?? NaN;
      if (tag <= -3 && longTime.color[index * 4] !== longTimeStep.color[index * 4]) {
        longTimeChanges++;
      } else if (tag > 0) {
        expect(longTimeStep.color.subarray(index * 4, index * 4 + 4)).toEqual(
          longTime.color.subarray(index * 4, index * 4 + 4),
        );
      }
    }
    expect(longTimeChanges).toBeGreaterThan(0);
    const bright = createAppearance({ ...initialAppearance, skyBrightness: 2 });
    const flat = createAppearance({ ...initialAppearance, diskStructure: 0 });
    const hot = createAppearance({
      ...initialAppearance,
      diskStructure: 0,
      diskTemperature: 10000,
    });
    const dark = createAppearance({ ...initialAppearance, skyBrightness: 0 });
    if (!bright.ok || !flat.ok || !dark.ok || !hot.ok) {
      throw new Error("Invalid source fixture.");
    }
    const brighterSky = await frame(0, false, bright.value);
    const steady = await frame(0, false, flat.value);
    const steadyLater = await frame(100, false, flat.value);
    const darkSky = await frame(0, false, dark.value);
    const hotter = await frame(100, false, hot.value);
    let heated = 0;
    expect(brighterSky.endpoints).toEqual(first.endpoints);
    expect(brighterSky.beams).toEqual(first.beams);
    expect(steadyLater.color).toEqual(steady.color);
    for (let i = 0; i < count; i++) {
      const tag = first.endpoints[i * 4 + 3] ?? Number.NaN;
      if (tag > 0) {
        expect(darkSky.color[i * 4 + 3]).toBe(1);
        expect(hotter.color.subarray(i * 4, i * 4 + 4)).toEqual(
          steadyLater.color.subarray(i * 4, i * 4 + 4),
        );
        for (let channel = 0; channel < 3; channel++) {
          expect(darkSky.color[i * 4 + channel]).toBe(0);
          if (first.color[i * 4 + 3] === 1) {
            const expected = 2 * (first.color[i * 4 + channel] ?? Number.NaN);
            expect(brighterSky.color[i * 4 + channel]).toBeCloseTo(expected, 4);
          }
        }
      } else if (tag <= -3) {
        if ((hotter.color[i * 4] ?? 0) > (steadyLater.color[i * 4] ?? 0)) {
          heated++;
        }
        // A visible point image's detector tent may overlap a disk-center pixel.
        // Subtract the opaque-disk baseline before checking source-flux scaling.
        expect(brighterSky.color[i * 4 + 3]).toBe(first.color[i * 4 + 3]);
        for (let channel = 0; channel < 3; channel++) {
          const base = darkSky.color[i * 4 + channel] ?? NaN;
          const normal = first.color[i * 4 + channel] ?? NaN;
          const brightValue = brighterSky.color[i * 4 + channel] ?? NaN;
          const residual = brightValue - 2 * normal + base;
          // Independent binary16 stores contribute at most half an ulp each.
          const rounding =
            0.0005 * (Math.abs(brightValue) + 2 * Math.abs(normal) + Math.abs(base)) + 2 ** -23;
          expect(Math.abs(residual)).toBeLessThanOrEqual(rounding);
        }
      }
    }

    expect(heated).toBeGreaterThan(20);
    expect(await device.popErrorScope()).toBeNull();
  });
});

describe("Interactive sampling", () => {
  test("near and distant views preserve resolved sky and report missing coverage", async ({
    device,
  }) => {
    using owned = new DisposableStack();

    device.pushErrorScope("validation");
    const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
    const width = 640;
    const height = 360;
    const colorBytes = width * height * 8;
    const staging = owned.adopt(
      device.createBuffer({
        size: colorBytes * 3,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      (value) => value.destroy(),
    );
    const records = [];
    const views = [6, 30, 160, 200].map((radius) => ({
      radius,
      fieldOfView: initialScene.observer.fieldOfView,
    }));
    views.push({ radius: 200, fieldOfView: distantSky.precisionSelection.fieldOfView });
    for (const { radius, fieldOfView } of views) {
      const scene = createScene({
        ...initialScene,
        observer: { ...initialScene.observer, radius, fieldOfView },
      });
      if (!scene.ok) {
        throw new Error(scene.error);
      }
      const encoder = device.createCommandEncoder();
      const image = optics.encode(encoder, scene.value, width, height, { antialias: true });
      encoder.copyTextureToBuffer(
        { texture: image.radiance },
        { buffer: staging, bytesPerRow: width * 8 },
        [width, height],
      );
      encoder.copyTextureToBuffer(
        { texture: image.endpoints },
        { buffer: staging, offset: colorBytes, bytesPerRow: width * 16 },
        [width, height],
      );
      device.queue.submit([encoder.finish()]);
      // Optical targets and staging storage are reused between complete scenes.
      // oxlint-disable-next-line no-await-in-loop
      await staging.mapAsync(GPUMapMode.READ);
      const mapped = staging.getMappedRange();
      const radiance = new Float16Array(mapped, 0, width * height * 4);
      const endpoints = new Float32Array(mapped, colorBytes, width * height * 4);
      const stages: Record<string, number> = {};
      const examples: Record<string, { x: number; y: number; endpoint: number[] }> = {};
      const orders: Record<string, number> = {};
      let missing = 0;
      let missingWeight = 0;
      let skyReferenceSamples = 0;
      let maximumSkyError = 0;
      for (let i = 0; i < width * height; i++) {
        const status = endpoints[i * 4 + 3] ?? 0;
        orders[status] = (orders[status] ?? 0) + 1;
        if (status === -2) {
          const stage = String(endpoints[i * 4]);
          stages[stage] = (stages[stage] ?? 0) + 1;
          examples[stage] ??= {
            x: i % width,
            y: Math.floor(i / width),
            endpoint: Array.from(endpoints.subarray(i * 4, i * 4 + 4)),
          };
        }
        const alpha = radiance[i * 4 + 3] ?? 0;
        missing += Number(alpha < 1);
        missingWeight += 1 - alpha;
      }
      // The distant frame previously rejected 9,330 ordinary sky rays at stage 15.
      // Preserve that regression separately from remaining critical-ray failures.
      if (radius >= 160) {
        expect.soft(stages["15"] ?? 0, `arrival uncertainty at radius ${radius}`).toBe(0);
      }
      if (radius >= 160) {
        const { observer, space, camera, diskInner, diskOuter } = scene.value;
        const frame = new Float32Array([
          width,
          height,
          0,
          0,
          observer.radius,
          observer.inclination,
          2 * Math.tan(observer.fieldOfView / 2),
          observer.azimuth,
          space.spin,
          space.charge,
          diskInner,
          diskOuter,
          ...camera.forward,
          0,
          ...camera.up,
          0,
          ...camera.right,
          0,
        ]);
        const samples =
          radius === 200 ? [distantSky.pixel, distantSky.halfPeriodCancellation.pixel] : [];
        if (fieldOfView === distantSky.precisionSelection.fieldOfView) {
          samples.push(distantSky.precisionSelection.pixel);
        }
        for (let y = 12; y < height; y += 40) {
          for (let x = 12; x < width; x += 40) {
            if (referenceRay(x, y, frame).outcome.kind === "sky") {
              samples.push([x, y]);
            }
          }
        }
        for (const pixel of samples) {
          const x = pixel[0] ?? NaN;
          const y = pixel[1] ?? NaN;
          const expected = referenceSky(x, y, frame);
          const offset = (y * width + x) * 4;
          expect(endpoints[offset + 3]).toBeGreaterThan(0);
          const mu = endpoints[offset] ?? NaN;
          const phi = endpoints[offset + 1] ?? NaN;
          const radial = Math.sqrt(1 - mu * mu);
          const actual = [radial * Math.cos(phi), radial * Math.sin(phi), mu];
          const error = Math.hypot(
            ...actual.map((value, axis) => value - (expected.direction[axis] ?? NaN)),
          );
          skyReferenceSamples++;
          maximumSkyError = Math.max(maximumSkyError, error);
          expect(error, `sky direction at radius ${radius}, pixel ${x},${y}`).toBeLessThan(5e-5);
        }
      }
      records.push({
        radius,
        fieldOfView,
        width,
        height,
        stages,
        examples,
        orders,
        missing,
        missingWeight,
        skyReferenceSamples,
        maximumSkyError,
      });
      staging.unmap();
    }
    await server.commands.writeFile(
      "test-results/exploration-endpoints.json",
      JSON.stringify(records, null, 2),
    );
    expect(await device.popErrorScope()).toBeNull();
  });
});

describe("Physical scenes", () => {
  test("presets and near/far exploration fit the optical work budget", async ({
    device,
    annotate,
  }) => {
    using owned = new DisposableStack();

    device.pushErrorScope("validation");
    const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
    const staging = owned.adopt(
      device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      (value) => value.destroy(),
    );
    const records: {
      name: string;
      boundary: number[];
      derivatives: number[];
      stellar: number[];
      capacity: number;
      derivativeCapacity: number;
    }[] = [];
    const close = createScene({
      ...initialScene,
      observer: { ...initialScene.observer, radius: 18, fieldOfView: 2 * Math.atan(0.6) },
    });
    if (!close.ok) {
      throw new Error(close.error);
    }
    const workloads = presets.flatMap(({ name, view }) => [
      { name: `${name} / native`, view, width: 2560, height: 1440 },
      { name: `${name} / 360p`, view, width: 640, height: 360 },
    ]);
    const closeWorkload = {
      name: "Close charged disk / 360p",
      view: { scene: close.value, appearance: initialAppearance, time: 0 },
      width: 640,
      height: 360,
    };
    const distanceWorkloads = [2, 6, 200].flatMap((radius) => {
      const scene = createScene({
        ...initialScene,
        observer: { ...initialScene.observer, radius },
      });
      if (!scene.ok) {
        throw new Error(scene.error);
      }
      return [640, 2560].map((width) => ({
        name: `Distance ${radius} / ${width}`,
        view: { scene: scene.value, appearance: initialAppearance, time: 0 },
        width,
        height: (width * 9) / 16,
      }));
    });
    for (const { name, view, width, height } of [
      ...workloads,
      closeWorkload,
      ...distanceWorkloads,
    ]) {
      const encoder = device.createCommandEncoder();
      const image = optics.encode(encoder, view.scene, width, height, {
        appearance: view.appearance,
        time: view.time,
        antialias: true,
      });
      encoder.copyBufferToBuffer(image.edgeStatistics, 0, staging, 0, 16);
      encoder.copyBufferToBuffer(image.edgeBeamStatistics, 0, staging, 16, 16);
      encoder.copyBufferToBuffer(image.stellarStatistics, 0, staging, 32, 32);
      device.queue.submit([encoder.finish()]);
      // Submissions reuse the optical target and staging buffer.
      // oxlint-disable-next-line no-await-in-loop
      await staging.mapAsync(GPUMapMode.READ);
      const counts = new Uint32Array(staging.getMappedRange()).slice();
      staging.unmap();
      records.push({
        name,
        boundary: Array.from(counts.subarray(0, 4)),
        derivatives: Array.from(counts.subarray(4, 7)),
        stellar: Array.from(counts.subarray(8, 16)),
        capacity: image.edgeCapacity,
        derivativeCapacity: image.edgeBeamCapacity,
      });
    }
    await annotate("Preset work budgets", {
      body: JSON.stringify(records),
      bodyEncoding: "utf-8",
      contentType: "application/json",
    });
    await server.commands.writeFile(
      "test-results/exploration-budgets.json",
      JSON.stringify(records, null, 2),
    );
    expect(await device.popErrorScope()).toBeNull();
    for (const record of records) {
      expect.soft(record.boundary[0], record.name).toBeLessThanOrEqual(record.capacity);
      expect
        .soft(record.derivatives[0], record.name)
        .toBeLessThanOrEqual(record.derivativeCapacity);
    }
  }, 30000);
});

describe("Still images", () => {
  test.for([0, 0.5, 1])(
    "PNG export preserves bloom %s, SDR exposure, P3 colors, and padded-row orientation",
    async (strength, { device }) => {
      using resources = new DisposableStack();

      const radiance = resources.adopt(
        device.createTexture({
          size: [7, 3],
          format: "rgba16float",
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        }),
        (value) => value.destroy(),
      );
      const scattered = resources.adopt(
        device.createTexture({
          size: [7, 3],
          format: "rgba16float",
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        }),
        (value) => value.destroy(),
      );
      const values = new Float16Array(7 * 3 * 4);
      for (let y = 0; y < 3; y++) {
        for (let x = 0; x < 7; x++) {
          values.set(
            y === 0 ? [16, 16, 16, 1] : y === 1 ? [4, 0, 0, 1] : [0, 0, 0, 1],
            (y * 7 + x) * 4,
          );
        }
      }
      device.queue.writeTexture({ texture: radiance }, values, { bytesPerRow: 7 * 8 }, [7, 3]);
      const exportPhoto = await createPhotoExporter(
        device,
        await compileShader(device, presentation, "photo-test"),
      );
      device.queue.writeTexture(
        { texture: scattered },
        values.map((value, index) => (index % 4 === 3 ? value : value * 0.5)),
        { bytesPerRow: 7 * 8 },
        [7, 3],
      );
      const pending = exportPhoto(radiance, -2, scattered, strength);
      // A later upload must not alter an export that has already been submitted.
      device.queue.writeTexture(
        { texture: radiance },
        new Float16Array(values.length),
        { bytesPerRow: 7 * 8 },
        [7, 3],
      );
      device.queue.writeTexture(
        { texture: scattered },
        new Float16Array(values.length),
        { bytesPerRow: 7 * 8 },
        [7, 3],
      );
      const blob = await pending;
      expect(blob.type).toBe("image/png");
      const bitmap = await createImageBitmap(blob);
      resources.defer(() => bitmap.close());
      expect([bitmap.width, bitmap.height]).toEqual([7, 3]);
      const canvas = new OffscreenCanvas(7, 3);
      const context = canvas.getContext("2d", {
        colorSpace: "display-p3",
        willReadFrequently: true,
      });
      if (!context) {
        throw new Error("Missing P3 decoding context.");
      }
      context.drawImage(bitmap, 0, 0);
      const decoded = context.getImageData(0, 0, 7, 3, { colorSpace: "display-p3" }).data;

      const gain = 1 - 0.5 * strength;
      for (let x = 0; x < 7; x++) {
        for (let c = 0; c < 3; c++) {
          expect(
            Math.abs(decode(decoded[x * 4 + c] ?? Number.NaN) - (4 * gain) / (1 + 4 * gain)),
          ).toBeLessThan(0.006);
          expect(decoded[(14 + x) * 4 + c]).toBe(0);
        }
        const red = decode(decoded[(7 + x) * 4] ?? Number.NaN);
        const green = decode(decoded[(7 + x) * 4 + 1] ?? Number.NaN);
        const blue = decode(decoded[(7 + x) * 4 + 2] ?? Number.NaN);
        expect(Math.abs(red - (0.82246197 * gain) / (1 + 0.82246197 * gain))).toBeLessThan(0.006);
        expect(Math.abs(green / red - 0.0403596)).toBeLessThan(0.002);
        expect(Math.abs(blue / red - 0.0207701)).toBeLessThan(0.002);
      }
      expect(
        Array.from(decoded)
          .filter((_, index) => index % 4 === 3)
          .every((alpha) => alpha === 255),
      ).toBe(true);
    },
  );
});

describe("Detector resolution", () => {
  test("exploration resolution changes preserve native photographic dimensions and history", async () => {
    const canvas = new OffscreenCanvas(1, 1);
    using owned = new DisposableStack();
    const renderer = owned.adopt(await createRenderer(canvas), (value) => value.dispose());
    const settings = { exposureEV: 0, hdr: false, time: 0, refine: false, view: "image" as const };
    renderer.resize(32, 18);
    renderer.render(initialScene, { ...settings, resolutionScale: 0.5 });
    await renderer.finished();
    expect([canvas.width, canvas.height]).toEqual([16, 9]);
    expect(renderer.render(initialScene, { ...settings, refine: true, resolutionScale: 0.5 })).toBe(
      1,
    );
    await renderer.finished();
    expect([canvas.width, canvas.height]).toEqual([32, 18]);
    expect(
      renderer.render(initialScene, { ...settings, refine: true, resolutionScale: 0.75 }),
    ).toBe(2);
    await renderer.finished();
    expect([canvas.width, canvas.height]).toEqual([32, 18]);
    renderer.render(initialScene, { ...settings, resolutionScale: 0.75 });
    await renderer.finished();
    expect([canvas.width, canvas.height]).toEqual([24, 13]);
    renderer.resize(40, 24);
    renderer.render(initialScene, { ...settings, resolutionScale: 0.5 });
    await renderer.finished();
    expect([canvas.width, canvas.height]).toEqual([20, 12]);
    for (const resolutionScale of [0, -1, 1.1, NaN, Infinity]) {
      expect(() => renderer.render(initialScene, { ...settings, resolutionScale })).toThrow(
        RangeError,
      );
    }
  });
});

describe("Resource ownership", () => {
  test("aborted renderer initialization releases the canvas for a fresh owner", async () => {
    const canvas = new OffscreenCanvas(1, 1);
    const attempt = new AbortController();
    const pending = createRenderer(canvas, attempt.signal);
    const reason = new Error("Superseded renderer");
    attempt.abort(reason);
    await expect(pending).rejects.toBe(reason);
    await expect(createRenderer(canvas, attempt.signal)).rejects.toBe(reason);

    using owned = new DisposableStack();
    const renderer = owned.adopt(await createRenderer(canvas), (value) => value.dispose());
    renderer.resize(16, 9);
    expect(
      renderer.render(initialScene, {
        exposureEV: 0,
        hdr: false,
        time: 0,
        refine: true,
        view: "image",
      }),
    ).toBe(1);
    await renderer.finished();
    expect([canvas.width, canvas.height]).toEqual([16, 9]);
    expect((await renderer.coverage()).pixels).toBe(144);
    renderer.dispose();
    expect(() => renderer.coverage()).toThrow("No refined image");
    expect(() => renderer.dispose()).not.toThrow();
  });
});

test("64 photographic samples preserve every unresolved weight at native portrait resolution", async ({
  device,
  annotate,
}) => {
  using owned = new DisposableStack();
  device.pushErrorScope("validation");
  const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
  const accumulation = owned.adopt(await createAccumulation(device), (value) => value.dispose());
  const capacity = 1024;
  const bytes = 16 + capacity * 32;
  const buffer = (size: number, usage: GPUBufferUsageFlags) =>
    owned.adopt(device.createBuffer({ size, usage }), (value) => value.destroy());
  const output = buffer(
    bytes,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  );
  const staging = buffer(bytes + 80, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const module = await compileShader(
    device,
    `
    struct Failure { pixel: vec4f, endpoint: vec4f }
    struct Failures { count: atomic<u32>, reserved: u32, padding: vec2u, records: array<Failure> }
    @group(0) @binding(0) var radiance: texture_2d<f32>;
    @group(0) @binding(1) var endpoints: texture_2d<f32>;
    @group(0) @binding(2) var<storage, read_write> failures: Failures;
    @compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id: vec3u) {
      if(any(id.xy >= textureDimensions(radiance))) { return; }
      let color = textureLoad(radiance, vec2i(id.xy), 0);
      if(color.a == 1.0) { return; }
      let index = atomicAdd(&failures.count, 1u);
      if(index >= arrayLength(&failures.records)) { return; }
      failures.records[index] = Failure(vec4f(vec2f(id.xy), color.a, 0.0), textureLoad(endpoints, vec2i(id.xy), 0));
    }
  `,
    "photographic audit",
  );
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module } });
  const width = photographicBoundary.frame[0],
    height = photographicBoundary.frame[1];
  if (!width || !height) {
    throw new Error("The photographic fixture requires image dimensions.");
  }
  const scene = createScene(photographicBoundary.scene);
  if (!scene.ok) {
    throw new Error(scene.error);
  }
  expect(Array.from(sceneFrame(scene.value, width, height))).toEqual(photographicBoundary.frame);
  const records = [];
  for (let sample = 0; sample < photographicBoundary.samples; sample++) {
    const encoder = device.createCommandEncoder();
    const jitter = pixelJitter(sample);
    const image = optics.encode(encoder, scene.value, width, height, {
      antialias: true,
      jitter,
      time: photographicBoundary.time,
    });
    accumulation.encode(encoder, image.radiance, sample);
    encoder.clearBuffer(output);
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: image.radiance.createView() },
        { binding: 1, resource: image.endpoints.createView() },
        { binding: 2, resource: { buffer: output } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.end();
    encoder.copyBufferToBuffer(output, 0, staging, 0, bytes);
    encoder.copyBufferToBuffer(image.beamStatistics, 0, staging, bytes, 16);
    encoder.copyBufferToBuffer(image.edgeStatistics, 0, staging, bytes + 16, 16);
    encoder.copyBufferToBuffer(image.edgeBeamStatistics, 0, staging, bytes + 32, 16);
    encoder.copyBufferToBuffer(image.stellarStatistics, 0, staging, bytes + 48, 32);
    device.queue.submit([encoder.finish()]);
    // oxlint-disable-next-line no-await-in-loop
    await staging.mapAsync(GPUMapMode.READ);
    const mapped = staging.getMappedRange();
    const count = new Uint32Array(mapped, 0, 1)[0] ?? NaN;
    expect(count).toBeLessThan(capacity);
    const values = new Float32Array(mapped, 16, count * 8);
    const failures = Array.from({ length: count }, (_, index) =>
      Array.from(values.subarray(index * 8, index * 8 + 8)),
    ).toSorted((a, b) => (a[1] ?? 0) - (b[1] ?? 0) || (a[0] ?? 0) - (b[0] ?? 0));
    const statistics = Array.from(new Uint32Array(mapped, bytes, 20));
    records.push({ sample, jitter, failures, statistics });
    staging.unmap();
  }
  const coverage = await accumulation.coverage();
  const uniquePixels = new Set(
    records.flatMap(({ failures }) => failures.map(([x, y]) => `${x},${y}`)),
  );
  const missingWeight = records.reduce(
    (sum, { failures }) =>
      sum + failures.reduce((weight, failure) => weight + 1 - (failure[2] ?? NaN), 0),
    0,
  );
  expect(coverage.unresolvedPixels).toBe(uniquePixels.size);
  expect(coverage.unresolvedSamples).toBe(missingWeight);
  expect(coverage.samplesPerPixel).toBe(photographicBoundary.samples);
  await annotate("Photographic coverage survey; this is not a zero-failure optical gate", {
    body: JSON.stringify(coverage),
    bodyEncoding: "utf-8",
    contentType: "application/json",
  });
  expect(await device.popErrorScope()).toBeNull();
  await server.commands.writeFile(
    "test-results/photo-audit.json",
    JSON.stringify(
      {
        schema: 1,
        frame: Array.from(sceneFrame(scene.value, width, height)),
        time: photographicBoundary.time,
        browser: navigator.userAgent,
        sourceSHA256: new Uint8Array(await sourceFingerprint()).toHex(),
        coverage,
        records,
      },
      null,
      2,
    ),
  );
}, 120000);
