import { expect, test } from "vitest";
import { opticalSources } from "../../src/render/optics.ts";
import { compileShader, requestDevice } from "../../src/render/device.ts";
import { normalize } from "../../src/physics/vector.ts";
import beam from "../../src/render/shaders/beam.wgsl?raw";
import { computeReadback } from "./compute.ts";

test("beam area classification retains rank and parity at extreme finite magnifications", async () => {
  const input = new Float32Array([0, 1e-30, 1e-11, 1e-9, 1, 1e15, 1e25, 1e35]);
  const result = await computeReadback(
    `${beam}
    @group(0) @binding(0) var<storage, read> inputs: array<f32>;
    @group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
    @compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id: vec3u) {
      let scale = inputs[id.x];
      let n = vec3f(1.0, 0.0, 0.0);
      let dx = vec3f(0.0, scale, scale);
      let dy = vec3f(0.0, -scale, scale);
      outputs[id.x] = vec4f(f32(celestial_has_area(n, dx, dy)),
        f32(celestial_has_area(n, dx, -dy)),
        f32(celestial_has_area(n, dx, dx)),
        f32(celestial_has_area(n, dx, vec3f(0.0))));
    }`,
    input,
    input.length * 4,
    input.length,
  );
  for (let index = 0; index < input.length; index++) {
    const resolved = index >= 3 ? 1 : 0;
    expect(Array.from(result.subarray(index * 4, index * 4 + 4))).toEqual([
      resolved,
      resolved,
      0,
      0,
    ]);
  }
});

test("sky beams recover both derivatives across diagonal branch boundaries", async () => {
  const masks = [
    [
      [-1, 0],
      [1, 0],
      [0, -1],
      [0, 1],
    ],
    [
      [-1, 0],
      [1, 0],
      [-1, -1],
      [1, 1],
    ],
    [
      [0, -1],
      [0, 1],
      [-1, -1],
      [1, 1],
    ],
    [
      [-1, -1],
      [-1, 1],
      [1, -1],
      [1, 1],
    ],
    [
      [-1, -1],
      [1, 1],
    ],
    [],
  ];
  const width = masks.length * 3;
  const data = new Float32Array(width * 3 * 4);
  // An affine camera plane at x = 1 gives analytic tangent derivatives at its center.
  const dx = [0, 0.003, 0.001];
  const dy = [0, -0.001, 0.002];
  for (const [index, mask] of masks.entries()) {
    for (let y = -1; y <= 1; y++) {
      for (let x = -1; x <= 1; x++) {
        const sameBranch = (x === 0 && y === 0) || mask.some(([mx, my]) => x === mx && y === my);
        const direction = normalize([1, 0.003 * x - 0.001 * y, 0.001 * x + 0.002 * y]);
        data.set(
          [direction[2], Math.atan2(direction[1], direction[0]), 1, sameBranch ? 1 : -3],
          ((y + 1) * width + index * 3 + x + 1) * 4,
        );
      }
    }
  }
  const device = await requestDevice();
  using owned = new DisposableStack();
  owned.defer(() => device.destroy());
  const map = owned.adopt(
    device.createTexture({
      size: [width, 3],
      format: "rgba32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    }),
    (value) => value.destroy(),
  );
  device.queue.writeTexture({ texture: map }, data, { bytesPerRow: width * 16 }, [width, 3]);
  const buffer = (usage: GPUBufferUsageFlags) =>
    owned.adopt(
      device.createBuffer({
        size: masks.length * 32,
        usage,
      }),
      (value) => value.destroy(),
    );
  const output = buffer(GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const staging = buffer(GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const module = await compileShader(
    device,
    `${opticalSources.sky}
    @group(0) @binding(11) var<storage, read_write> gradients: array<vec4f>;
    @compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id: vec3u) {
      let pixel = vec2i(i32(id.x) * 3 + 1, 1);
      let center = textureLoad(optical_map, pixel, 0);
      let beam = celestial_beam(optical_map, pixel, center);
      gradients[id.x * 2u] = beam.dx;
      gradients[id.x * 2u + 1u] = beam.dy;
    }`,
    "branch-gradient-test",
  );
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "probe" },
  });
  const bindings = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: map.createView() },
      { binding: 11, resource: { buffer: output } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindings);
  pass.dispatchWorkgroups(masks.length);
  pass.end();
  encoder.copyBufferToBuffer(output, 0, staging, 0, output.size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange()).slice();
  staging.unmap();
  for (let index = 0; index < masks.length; index++) {
    for (const [axis, expected] of [dx, dy].entries()) {
      for (let component = 0; component < 3; component++) {
        const target = index < 4 ? (expected[component] ?? NaN) : 0;
        expect(Math.abs((result[index * 8 + axis * 4 + component] ?? NaN) - target)).toBeLessThan(
          1e-7,
        );
      }
    }
  }
});
