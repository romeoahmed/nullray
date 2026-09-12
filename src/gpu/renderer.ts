import fullscreen from "./wgsl/imaging/fullscreen.wgsl?raw";
import { createBloom } from "./imaging/bloom.ts";
import { createPhotoExporter } from "./imaging/photograph.ts";
import presentation from "./wgsl/passes/present.wgsl?raw";
import { compileShader, requestDevice } from "./device.ts";
import { initialAppearance } from "../scene/appearance.ts";
import type { SourceAppearance } from "../scene/appearance.ts";
import type { Scene } from "../scene/scene.ts";
import { createOptics } from "./optics/engine.ts";
import type { OpticalImage } from "./optics/engine.ts";
import type { Presentation } from "../scene/presentation.ts";
import { displayValue } from "../scene/presentation.ts";
import { createPolarimeter } from "./imaging/polarimeter.ts";
import { createAccumulation } from "./imaging/accumulation.ts";
import type { Coverage } from "./imaging/coverage.ts";
import type { RayPath } from "../physics/ray-path.ts";
import { blackbodyWhiteBalance } from "../physics/radiation.ts";
import type { Vec3 } from "../physics/vector.ts";

import { sampleCounts, sampleOffset } from "./imaging/sampling.ts";
import type { SamplingMode } from "./imaging/sampling.ts";

const opticalView = (value: Presentation["diagnostic"]) =>
  value === "polarization" || value === "angle" ? "image" : value;

/**
 * Device/canvas owner for serial optical, detector, and presentation work.
 *
 * @remarks
 * The worker schedules submissions and rejects stale replies. Returned promises
 * must settle before replacing resources used by inspection or readback.
 */
export interface Renderer {
  /** Inspect the last optical frame while the worker holds its submission lock. */
  inspect(point?: readonly [number, number]): Promise<RayPath>;
  readonly lost: Promise<GPUDeviceLostInfo>;
  /**
   * Record nonnegative integer device-pixel dimensions; zero suspends rendering.
   *
   * @remarks
   * A uniform reduction enforces the device texture limit. Invalid dimensions
   * throw RangeError. Raster allocation occurs on the next render.
   */
  resize(width: number, height: number): void;
  /** Resolve after submitted GPU work completes, allowing bounded frame scheduling. */
  finished(): Promise<void>;
  /**
   * Read missing weights for the current non-live sample sequence.
   *
   * @remarks
   * Submit its history writes first. Throws if no accumulated image is available;
   * the result reports missing samples, not an image-error bound.
   */
  coverage(): Promise<Coverage>;
  /**
   * Snapshot the completed 64-sample photograph as an SDR Display P3 PNG.
   *
   * @remarks
   * Throws if the active renderer has no completed photograph. The asynchronous result
   * contains 8-bit display pixels, not HDR radiance or coverage metadata.
   */
  exportPNG(): Promise<Blob>;
  /** Submit one frame and return its current sample count; unchanged completed images reuse their history. */
  render(scene: Scene, settings: RenderSettings): number;
  /** Release the device and canvas context; safe to call repeatedly. */
  dispose(): void;
}

/** Optical/source controls and display choices for one submitted frame. */
export interface RenderSettings {
  readonly appearance?: SourceAppearance;
  /** Scale per raster axis in (0, 1], default 1; Photograph overrides it with native pixels. */
  readonly resolutionScale?: number;
  readonly exposureEV: number;
  readonly whiteBalance?: number;
  readonly bloom?: number;
  readonly hdr: boolean;
  /** Observer epoch in geometric time units; navigation does not advance this clock. */
  readonly time: number;
  readonly sampling: SamplingMode;
  readonly view: Presentation["diagnostic"];
  readonly analyzer?: number | null;
}

/**
 * Acquire a device and own an OffscreenCanvas rendering context.
 *
 * @param canvas - Transferred canvas whose previous renderer has already been disposed.
 * @param signal - Optional cancellation checked before acquisition and before publication.
 * @returns The renderer owner; initialization failure or observed cancellation rejects
 * and releases acquired resources. Dispose the owner when the worker replaces it.
 */
export async function createRenderer(
  canvas: OffscreenCanvas,
  signal?: AbortSignal,
): Promise<Renderer> {
  signal?.throwIfAborted();
  const device = await requestDevice();
  using owned = new DisposableStack();
  owned.defer(() => device.destroy());
  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new Error("Unable to create a WebGPU canvas.");
  }
  owned.defer(() => context.unconfigure());
  const format = "rgba16float";
  const displayModule = await compileShader(
    device,
    `${fullscreen}\n${presentation}`,
    "presentation",
  );
  const display = await device.createRenderPipelineAsync({
    layout: "auto",
    vertex: { module: displayModule, entryPoint: "vertex" },
    fragment: { module: displayModule, entryPoint: "fragment", targets: [{ format }] },
  });
  const bloomFilter = owned.adopt(await createBloom(device), (value) => value.dispose());
  const exportPhoto = await createPhotoExporter(device, displayModule);
  const optics = owned.adopt(await createOptics(device), (value) => value.dispose());
  const accumulation = owned.adopt(await createAccumulation(device), (value) => value.dispose());
  const qAccumulation = owned.adopt(await createAccumulation(device), (value) => value.dispose());
  const uAccumulation = owned.adopt(await createAccumulation(device), (value) => value.dispose());
  const polarimeter = owned.adopt(await createPolarimeter(device), (value) => value.dispose());
  const displayUniform = owned.adopt(
    device.createBuffer({
      label: "display",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
    (buffer) => buffer.destroy(),
  );
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  const displayFrame = new Float32Array(8);
  let calibration: Vec3 = [1, 1, 1];
  let calibratedTemperature: number | undefined;
  let width = 0;
  let height = 0;
  let requestedWidth = 0;
  let requestedHeight = 0;
  let outputHDR: boolean | undefined;
  let previous:
    | {
        readonly scene: Scene;
        readonly appearance: SourceAppearance;
        readonly time: number;
        readonly exposureEV: number;
        readonly whiteBalance: number | undefined;
        readonly bloom: number;
        readonly sampling: SamplingMode;
        readonly view: Presentation["diagnostic"];
        readonly analyzer?: number | null;
      }
    | undefined;
  let scattered: GPUTexture | undefined;
  let samples = 0;
  let coverage: Promise<Coverage> | undefined;
  let displayed: GPUTexture | undefined;
  let boundDisplay: GPUTexture | undefined;
  let boundBloom: GPUTexture | undefined;
  let displayBindings: GPUBindGroup | undefined;
  let stokes: Pick<OpticalImage, "radiance" | "q" | "u"> | undefined;
  signal?.throwIfAborted();
  const lifetime = owned.move();
  return {
    inspect(point) {
      return optics.inspect(point);
    },
    lost: device.lost,
    finished() {
      return device.queue.onSubmittedWorkDone();
    },
    coverage() {
      if (previous?.sampling === "live" || !previous || samples === 0 || lifetime.disposed) {
        throw new Error("No refined image is available.");
      }
      coverage ??= accumulation.coverage();
      return coverage;
    },
    exportPNG() {
      if (
        !displayed ||
        previous?.sampling !== "photograph" ||
        samples !== 64 ||
        lifetime.disposed
      ) {
        throw new Error("Refine a still image before exporting it.");
      }
      return exportPhoto(
        displayed,
        previous.exposureEV,
        scattered ?? displayed,
        previous.bloom,
        previous.whiteBalance,
        previous.view === "image" && !previous.scene.plasma,
      );
    },
    resize(pixelWidth, pixelHeight) {
      if (
        !Number.isSafeInteger(pixelWidth) ||
        !Number.isSafeInteger(pixelHeight) ||
        pixelWidth < 0 ||
        pixelHeight < 0
      ) {
        throw new RangeError("Canvas dimensions must be nonnegative integer device pixels.");
      }
      const scale = Math.min(
        1,
        device.limits.maxTextureDimension2D / Math.max(1, pixelWidth, pixelHeight),
      );
      requestedWidth = Math.floor(pixelWidth * scale);
      requestedHeight = Math.floor(pixelHeight * scale);
    },
    render(
      scene,
      {
        exposureEV,
        whiteBalance,
        hdr,
        time,
        sampling,
        view,
        analyzer = null,
        appearance = initialAppearance,
        bloom = 0,
        resolutionScale = 1,
      },
    ) {
      if (lifetime.disposed || requestedWidth === 0 || requestedHeight === 0) {
        return 0;
      }
      if (!displayValue("exposure", exposureEV)) {
        throw new RangeError("Exposure must be between −6 and +6 EV.");
      }
      if (!displayValue("bloom", bloom)) {
        throw new RangeError("Bloom must be between zero and one.");
      }
      const strength = view === "image" ? bloom : 0;
      const temperature = view === "image" && !scene.plasma ? whiteBalance : undefined;
      if (temperature !== calibratedTemperature) {
        calibration = temperature === undefined ? [1, 1, 1] : blackbodyWhiteBalance(temperature);
        calibratedTemperature = temperature;
      }
      if (outputHDR !== hdr) {
        context.configure({
          device,
          format,
          colorSpace: "display-p3",
          toneMapping: { mode: hdr ? "extended" : "standard" },
          alphaMode: "opaque",
        });
        if (context.getConfiguration()?.toneMapping?.mode !== (hdr ? "extended" : "standard")) {
          throw new Error("This browser does not support the requested display mode.");
        }
        outputHDR = hdr;
      }
      if (!Number.isFinite(resolutionScale) || resolutionScale <= 0 || resolutionScale > 1) {
        throw new RangeError(
          "Exploration resolution scale must be greater than zero and at most one.",
        );
      }
      const scale = sampling === "photograph" ? 1 : resolutionScale;
      const nextWidth = Math.max(1, Math.floor(requestedWidth * scale));
      const nextHeight = Math.max(1, Math.floor(requestedHeight * scale));
      const resized = width !== nextWidth || height !== nextHeight;
      if (resized) {
        width = nextWidth;
        height = nextHeight;
        canvas.width = width;
        canvas.height = height;
      }
      displayFrame.set([
        2 ** exposureEV,
        hdr ? 4 : 1,
        strength,
        view === "image" ? 0 : 1,
        ...calibration,
        view === "image" && !scene.plasma ? 1 : 0,
      ]);
      device.queue.writeBuffer(displayUniform, 0, displayFrame);
      const encoder = device.createCommandEncoder();
      const invalidated =
        resized ||
        previous?.scene !== scene ||
        previous.appearance !== appearance ||
        previous.time !== time ||
        previous.sampling !== sampling ||
        opticalView(previous.view) !== opticalView(view);
      if (invalidated) {
        samples = 0;
        coverage = undefined;
      }
      const newSample = invalidated || samples < sampleCounts[sampling];
      if (newSample) {
        coverage = undefined;
        const image = optics.encode(encoder, scene, width, height, {
          time,
          view: opticalView(view),
          appearance,
          jitter: sampleOffset(sampling, samples),
        });
        stokes =
          sampling !== "live"
            ? {
                radiance: accumulation.encode(encoder, image.radiance, samples),
                q: qAccumulation.encode(encoder, image.q, samples),
                u: uAccumulation.encode(encoder, image.u, samples),
              }
            : image;
        samples++;
        scattered = undefined;
      }
      const detectorChanged = previous?.analyzer !== analyzer || previous.view !== view;
      if (stokes && (newSample || detectorChanged)) {
        displayed = polarimeter.encode(
          encoder,
          stokes,
          view === "image" ? analyzer : null,
          view === "polarization" || view === "angle" ? view : "image",
        );
      }
      if (detectorChanged) {
        scattered = undefined;
      }
      if (displayed && strength > 0 && !scattered) {
        scattered = bloomFilter.encode(encoder, displayed);
      }
      if (!displayed) {
        throw new Error("No radiance is available for presentation.");
      }
      const bloomImage = scattered ?? displayed;
      if (!displayBindings || boundDisplay !== displayed || boundBloom !== bloomImage) {
        displayBindings = device.createBindGroup({
          layout: display.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: displayed.createView() },
            { binding: 1, resource: sampler },
            { binding: 2, resource: { buffer: displayUniform } },
            { binding: 3, resource: bloomImage.createView() },
          ],
        });
        boundDisplay = displayed;
        boundBloom = bloomImage;
      }
      const displayPass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: [0, 0, 0, 1],
          },
        ],
      });
      displayPass.setPipeline(display);
      displayPass.setBindGroup(0, displayBindings);
      displayPass.draw(3);
      displayPass.end();
      device.queue.submit([encoder.finish()]);
      previous = {
        scene,
        appearance,
        time,
        exposureEV,
        whiteBalance: temperature,
        bloom: strength,
        sampling,
        view,
        analyzer,
      };
      return samples;
    },
    dispose() {
      lifetime.dispose();
    },
  };
}
