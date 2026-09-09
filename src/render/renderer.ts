import { createBloom } from "./bloom.ts";
import { createPhotoExporter } from "./photograph.ts";
import presentation from "./shaders/present.wgsl?raw";
import { compileShader, requestDevice } from "./device.ts";
import { initialAppearance } from "../model/appearance.ts";
import type { SourceAppearance } from "../model/appearance.ts";
import type { Scene } from "../model/scene.ts";
import { createOptics } from "./optics.ts";
import type { OpticalView } from "./optics.ts";
import { createAccumulation, pixelJitter } from "./accumulation.ts";
import type { Coverage } from "./coverage.ts";

/** Device and canvas owner coordinating optical and photographic resources. */
export interface Renderer {
  readonly lost: Promise<GPUDeviceLostInfo>;
  /** Set the content-box size in physical display pixels. */
  resize(width: number, height: number): void;
  /** Resolve after submitted GPU work completes, allowing bounded frame scheduling. */
  finished(): Promise<void>;
  /** Read missing sample weights from the current refinement sequence. */
  coverage(): Promise<Coverage>;
  /** Snapshot the completed still image with SDR tone mapping and Display P3 encoding. */
  exportPNG(): Promise<Blob>;
  /** Submit one frame and return its photographic sample count, or zero during exploration. */
  render(scene: Scene, settings: RenderSettings): number;
  /** Release the device and canvas context; safe to call repeatedly. */
  dispose(): void;
}

/** Optical/source controls and display choices for one submitted frame. */
export interface RenderSettings {
  readonly appearance?: SourceAppearance;
  /** Linear exploration scale in (0, 1]; still-image refinement uses native pixels. */
  readonly resolutionScale?: number;
  readonly exposureEV: number;
  readonly bloom?: number;
  readonly hdr: boolean;
  /** Observer epoch in geometric time units; navigation does not advance this clock. */
  readonly time: number;
  readonly refine: boolean;
  readonly view: OpticalView;
}

/** Own one canvas context and its GPU resources; dispose its previous owner before initialization. */
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
  const displayModule = await compileShader(device, presentation, "presentation");
  const display = await device.createRenderPipelineAsync({
    layout: "auto",
    vertex: { module: displayModule, entryPoint: "vertex" },
    fragment: { module: displayModule, entryPoint: "fragment", targets: [{ format }] },
  });
  const bloomFilter = owned.adopt(await createBloom(device), (value) => value.dispose());
  const exportPhoto = await createPhotoExporter(device, displayModule);
  const optics = owned.adopt(await createOptics(device, signal), (value) => value.dispose());
  const accumulation = owned.adopt(await createAccumulation(device), (value) => value.dispose());
  const displayUniform = owned.adopt(
    device.createBuffer({
      label: "display",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
    (buffer) => buffer.destroy(),
  );
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  const displayFrame = new Float32Array(4);
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
        readonly bloom: number;
        readonly refine: boolean;
        readonly view: OpticalView;
      }
    | undefined;
  let scattered: GPUTexture | undefined;
  let samples = 0;
  let coverage: Promise<Coverage> | undefined;
  let displayed: GPUTexture | undefined;
  signal?.throwIfAborted();
  const lifetime = owned.move();
  return {
    lost: device.lost,
    finished() {
      return device.queue.onSubmittedWorkDone();
    },
    coverage() {
      if (!previous?.refine || samples === 0 || lifetime.disposed) {
        throw new Error("No refined image is available.");
      }
      coverage ??= accumulation.coverage();
      return coverage;
    },
    exportPNG() {
      if (!displayed || !previous?.refine || samples !== 64 || lifetime.disposed) {
        throw new Error("Refine a still image before exporting it.");
      }
      return exportPhoto(displayed, previous.exposureEV, scattered ?? displayed, previous.bloom);
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
        hdr,
        time,
        refine,
        view,
        appearance = initialAppearance,
        bloom = 0,
        resolutionScale = 1,
      },
    ) {
      if (lifetime.disposed || requestedWidth === 0 || requestedHeight === 0) {
        return 0;
      }
      if (!Number.isFinite(exposureEV) || exposureEV < -6 || exposureEV > 6) {
        throw new RangeError("Exposure must be between −6 and +6 EV.");
      }
      if (!Number.isFinite(bloom) || bloom < 0 || bloom > 1) {
        throw new RangeError("Bloom must be between zero and one.");
      }
      const strength = view === "image" ? bloom : 0;
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
      const scale = refine ? 1 : resolutionScale;
      const nextWidth = Math.max(1, Math.floor(requestedWidth * scale));
      const nextHeight = Math.max(1, Math.floor(requestedHeight * scale));
      const resized = width !== nextWidth || height !== nextHeight;
      if (resized) {
        width = nextWidth;
        height = nextHeight;
        canvas.width = width;
        canvas.height = height;
      }
      displayFrame.set([2 ** exposureEV, hdr ? 4 : 1, strength, 0]);
      device.queue.writeBuffer(displayUniform, 0, displayFrame);
      const encoder = device.createCommandEncoder();
      const invalidated =
        resized ||
        previous?.scene !== scene ||
        previous.appearance !== appearance ||
        previous.time !== time ||
        previous.refine !== refine ||
        previous.view !== view;
      if (invalidated) {
        samples = 0;
        coverage = undefined;
      }
      if (invalidated || (refine && samples < 64)) {
        coverage = undefined;
        const image = optics.encode(encoder, scene, width, height, {
          time,
          view,
          appearance,
          antialias: !refine && view === "image",
          trace: refine || resized || previous?.scene !== scene || previous.refine !== refine,
          jitter: refine ? pixelJitter(samples) : [0, 0],
        });
        const radiance = refine
          ? accumulation.encode(encoder, image.radiance, samples++)
          : image.radiance;
        displayed = radiance;
        scattered = undefined;
      }
      if (displayed && strength > 0 && !scattered) {
        scattered = bloomFilter.encode(encoder, displayed);
      }
      if (!displayed) {
        throw new Error("No radiance is available for presentation.");
      }
      const displayBindings = device.createBindGroup({
        layout: display.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: displayed.createView() },
          { binding: 1, resource: sampler },
          { binding: 2, resource: { buffer: displayUniform } },
          { binding: 3, resource: (scattered ?? displayed).createView() },
        ],
      });
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
      previous = { scene, appearance, time, exposureEV, bloom: strength, refine, view };
      return samples;
    },
    dispose() {
      lifetime.dispose();
    },
  };
}
