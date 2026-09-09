import { expect } from "vitest";
import { opticalSources } from "../../src/render/optics.ts";
import { compileShader, requestDevice } from "../../src/render/device.ts";
import type { Vec3 } from "../../src/physics/vector.ts";
import type { Scene } from "../../src/model/scene.ts";
import scanReference from "../fixtures/radial-scan.wgsl?raw";

/** Probe the production classifier at prescribed local directions, independent of raster resolution. */
export async function traceDirections(
  directions: readonly Vec3[],
  inclination: number,
  scene?: Scene,
  compareScan = false,
) {
  const device = await requestDevice();
  using owned = new DisposableStack();
  owned.defer(() => device.destroy());
  device.pushErrorScope("validation");
  const buffer = (size: number, usage: GPUBufferUsageFlags) =>
    owned.adopt(device.createBuffer({ size, usage }), (value) => value.destroy());
  const frame = buffer(96, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(
    frame,
    0,
    new Float32Array([
      1,
      1,
      0,
      0,
      scene?.observer.radius ?? 30,
      inclination,
      1,
      scene?.observer.azimuth ?? 0,
      scene?.space.spin ?? 0,
      scene?.space.charge ?? 0,
      scene?.diskInner ?? 6,
      scene?.diskOuter ?? 64,
    ]),
  );
  const data = new Float32Array(directions.length * 4);
  for (const [index, direction] of directions.entries()) {
    data.set(direction, index * 4);
  }
  const input = buffer(data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const resultBytes = data.byteLength * (compareScan ? 2 : 1);
  const output = buffer(resultBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const readback = buffer(resultBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(input, 0, data);
  const emissionTime = owned.adopt(
    device.createTexture({
      size: [Math.min(1024, directions.length), Math.ceil(directions.length / 1024)],
      format: "r32float",
      usage: GPUTextureUsage.STORAGE_BINDING,
    }),
    (value) => value.destroy(),
  );
  const module = await compileShader(
    device,
    `${opticalSources.geometry}
${compareScan ? scanReference : ""}
    @group(0) @binding(3) var<storage, read> probe_directions: array<vec4f>;
    @group(0) @binding(4) var<storage, read_write> probe_results: array<vec4f>;
    @compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= arrayLength(&probe_directions)) { return; }
      let pixel = vec2i(i32(id.x % 1024u), i32(id.x / 1024u));
      ${
        compareScan
          ? `probe_results[2u * id.x] = trace_ray(frame, probe_directions[id.x].xyz).value;
      probe_results[2u * id.x + 1u] = scan_reference(probe_directions[id.x].xyz, pixel);`
          : `probe_results[id.x] = trace_ray(frame, probe_directions[id.x].xyz).value;`
      }
    }`,
    "production visibility probe",
  );
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "probe" },
  });
  const bindings = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: frame } },
      ...(compareScan ? [{ binding: 2, resource: emissionTime.createView() }] : []),
      { binding: 3, resource: { buffer: input } },
      { binding: 4, resource: { buffer: output } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindings);
  pass.dispatchWorkgroups(Math.ceil(directions.length / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, resultBytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(readback.getMappedRange()).slice();
  readback.unmap();
  expect(await device.popErrorScope()).toBeNull();
  return result;
}
