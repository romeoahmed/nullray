import { sourceFingerprint, benchmarkFingerprint } from "./source.ts";

/** Environment of the device that actually executed the workload; hashes are computed outside timing. */
export async function gpuContext(device: GPUDevice) {
  const [source, harness] = await Promise.all([sourceFingerprint(), benchmarkFingerprint()]);
  return {
    schema: 1,
    recordedAt: new Date().toISOString(),
    browser: navigator.userAgent,
    adapter: {
      vendor: device.adapterInfo.vendor,
      architecture: device.adapterInfo.architecture,
      device: device.adapterInfo.device,
      description: device.adapterInfo.description,
    },
    enabledFeatures: [...device.features].toSorted(),
    sourceSHA256: new Uint8Array(source).toHex(),
    harnessSHA256: new Uint8Array(harness).toHex(),
    timing: {
      clock: "wall-clock",
      unit: "ms",
      completion: "GPUQueue.onSubmittedWorkDone",
      timestampQuery: false,
    },
  };
}
