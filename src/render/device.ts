// TypeScript 7.0.2's DOM/WebWorker feature union predates tier 2. The assertion
// bridges that declaration gap only; adapter.features.has checks runtime support.
// oxlint-disable-next-line typescript/no-unsafe-type-assertion
const textureFormatsTier2 = "texture-formats-tier2" as GPUFeatureName;

/** Native features shared by rendering and comparable GPU benchmarks. */
export const requiredFeatures = [
  "subgroups",
  textureFormatsTier2,
] as const satisfies readonly GPUFeatureName[];

/** Request the renderer's mandatory native WebGPU capability. */
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
  return adapter.requestDevice({ requiredFeatures: [...requiredFeatures] });
}

/** Preserve shader diagnostics before attempting asynchronous pipeline creation. */
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
