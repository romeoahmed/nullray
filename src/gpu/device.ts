// The current @types/web and @types/webworker feature unions omit tier 2. This
// assertion bridges only that declaration gap; the adapter is checked at runtime.
// oxlint-disable-next-line typescript/no-unsafe-type-assertion
const textureFormatsTier2 = "texture-formats-tier2" as GPUFeatureName;

/** Native features shared by rendering and comparable GPU benchmarks. */
export const requiredFeatures = [
  "subgroups",
  textureFormatsTier2,
] as const satisfies readonly GPUFeatureName[];

/** I/Q/U, source domains, asymptotic directions and transmission are written in one path pass. */
export const requiredLimits = { maxStorageTexturesPerShaderStage: 6 } as const;

/**
 * Request a device with the renderer's shared GPU features and storage limits.
 *
 * @returns A caller-owned device to destroy after dependent resources are released.
 * Rejects for missing capabilities, no adapter, or a failed native device request.
 * This is not a preflight for every browser API used by the application.
 */
export async function requestDevice(): Promise<GPUDevice> {
  if (!navigator.gpu) {
    throw new Error(
      "WebGPU is unavailable. Open Nullray in a current browser with WebGPU enabled.",
    );
  }
  if (!navigator.gpu.wgslLanguageFeatures.has("readonly_and_readwrite_storage_textures")) {
    throw new Error("Nullray requires WGSL read-write storage textures.");
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    throw new Error("No WebGPU adapter is available.");
  }
  const missing = requiredFeatures.filter((feature) => !adapter.features.has(feature));
  if (missing.length > 0) {
    throw new Error(`Missing required WebGPU features: ${missing.join(", ")}.`);
  }
  if (
    adapter.limits.maxStorageTexturesPerShaderStage <
    requiredLimits.maxStorageTexturesPerShaderStage
  ) {
    throw new Error("Nullray requires six storage textures per shader stage.");
  }
  return adapter.requestDevice({ requiredFeatures: [...requiredFeatures], requiredLimits });
}

/**
 * Observe the first uncaptured GPU error or device loss on a borrowed device.
 *
 * @returns A disposable subscription and a promise that resolves with the failure.
 * Dispose before destroying the device so intentional teardown is not reported.
 * Captured error scopes remain the caller's responsibility; the promise never rejects.
 */
export function watchDevice(device: GPUDevice) {
  const subscription = new AbortController();
  const { signal } = subscription;
  const { promise, resolve } = Promise.withResolvers<Error>();
  device.addEventListener(
    "uncapturederror",
    (event) => {
      event.preventDefault();
      resolve(new Error(`GPU execution failed: ${event.error.message}`));
    },
    { signal },
  );
  void device.lost.then((info) => {
    if (!signal.aborted) {
      resolve(
        new Error(`The graphics device was lost. Retry to restore this view. ${info.message}`),
      );
    }
    return undefined;
  });
  return {
    failure: promise,
    [Symbol.dispose]() {
      subscription.abort();
    },
  };
}

/**
 * Compile WGSL on a borrowed device and preserve source-located error messages.
 *
 * @returns A shader module when compilation reports no errors; pipeline validation
 * still occurs separately. Rejects on compilation errors or native API failures.
 */
export async function compileShader(
  device: GPUDevice,
  code: string,
  label: string,
): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ code, label });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length > 0) {
    throw new Error(
      errors
        .map((message) => `${label}:${message.lineNum}:${message.linePos} ${message.message}`)
        .join("\n"),
    );
  }
  return module;
}
