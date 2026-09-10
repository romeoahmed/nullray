import { describe, expect } from "vitest";
import { blackbodyXYZ, createBlackbodyTable } from "../../src/physics/radiation.ts";
import { createStarTree } from "../../src/physics/stars.ts";
import { cross, dot, normalize } from "../../src/physics/vector.ts";
import type { Vec3 } from "../../src/physics/vector.ts";
import radiation from "../../src/gpu/wgsl/imaging/radiation.wgsl?raw";
import stars from "../../src/gpu/wgsl/imaging/source-tree.wgsl?raw";
import patchSource from "../../src/gpu/wgsl/imaging/stellar-patch.wgsl?raw";
import { test, computeReadback } from "./compute.ts";
import ownership from "../../src/gpu/wgsl/imaging/stellar-partition.wgsl?raw";
import composition from "../../src/gpu/wgsl/imaging/stellar-raster.wgsl?raw";
import { compileShader } from "../../src/gpu/device.ts";

const luminance = (rgb: Float32Array) =>
  0.2126390059 * (rgb[0] ?? NaN) + 0.7151686788 * (rgb[1] ?? NaN) + 0.0721923154 * (rgb[2] ?? NaN);

/** Exact derivative of the independently constructed homogeneous bilinear map. */
function samplePatch(corners: readonly Vec3[], u: number, v: number) {
  const corner = (index: number, axis: number) => corners[index]?.[axis] ?? NaN;
  const value = [0, 1, 2].map(
    (axis) =>
      (1 - u) * (1 - v) * corner(0, axis) +
      u * (1 - v) * corner(1, axis) +
      (1 - u) * v * corner(2, axis) +
      u * v * corner(3, axis),
  );
  const direction = normalize([value[0] ?? NaN, value[1] ?? NaN, value[2] ?? NaN]);
  const dx = (axis: number) =>
    (1 - v) * (corner(1, axis) - corner(0, axis)) + v * (corner(3, axis) - corner(2, axis));
  const dy = (axis: number) =>
    (1 - u) * (corner(2, axis) - corner(0, axis)) + u * (corner(3, axis) - corner(1, axis));
  const jacobian =
    Math.abs(dot(direction, cross([dx(0), dx(1), dx(2)], [dy(0), dy(1), dy(2)]))) /
    dot(direction, [value[0] ?? NaN, value[1] ?? NaN, value[2] ?? NaN]) ** 2;
  return { direction, jacobian };
}

function foldedPatch(scale: number): readonly Vec3[] {
  // Equal off-diagonal corners make exchanging u and v an exact symmetry.
  return [
    [0, 0, 1],
    normalize([scale, 0, 1]),
    normalize([scale, 0, 1]),
    normalize([2 * scale, scale, 1]),
  ].map((value): Vec3 => [
    Math.fround(value[0] ?? NaN),
    Math.fround(value[1] ?? NaN),
    Math.fround(value[2] ?? NaN),
  ]);
}

describe("Finite detector patches", () => {
  // These local-map checks integrate the entire map; production ownership is checked through resolve_sky.
  const unpartitioned = `struct StellarFootprint { origin: vec2f, scale: f32, }
fn stellar_filtered_image(position: vec2f, footprint: StellarFootprint) -> bool { return false; }
`;

  const source = `${unpartitioned}\n${radiation}\n${stars}\n${patchSource}
@group(0) @binding(0) var<storage, read> patches: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> colors: array<vec4f>;
@group(0) @binding(3) var<storage, read> star_nodes: array<StarNode>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  let base = 5u * id.x;
  let cell = StellarPatch(patches[base], patches[base+1u], patches[base+2u], patches[base+3u]);
  colors[id.x] = point_stars_patch(cell, patches[base+4u].xy, StellarFootprint(vec2f(0.0), 0.0));
}`;

  test.for([0.02, 0.5, 2])(
    "folded bilinear patch at scale %s retains both opposite-parity images and their flux",
    async (scale) => {
      const corners = foldedPatch(scale);
      const sample = samplePatch(corners, 0.2, 0.8);
      const beams: number[] = [];
      for (const y of [-1, 0]) {
        for (const x of [-1, 0]) {
          for (const corner of corners) {
            beams.push(...corner, 1);
          }
          beams.push(x, y, 0, 0);
        }
      }
      const output = await computeReadback(source, new Float32Array(beams), 16, 4, [
        createBlackbodyTable().data,
        createStarTree([{ direction: sample.direction, temperature: 6500, flux: 1e-5 }]),
      ]);
      let total = 0;
      for (let index = 0; index < 4; index++) {
        expect(output[index * 4 + 3]).toBe(1);
        total += luminance(output.subarray(index * 4, index * 4 + 3));
      }
      expect(Math.abs(total / (2e-5 / sample.jacobian) - 1)).toBeLessThan(0.002);
    },
  );

  test.for(
    [0, 0.7, 1.5].flatMap((angle) =>
      [
        [0, 0],
        ...[-0.83, -0.3, 0.3, 0.5, 0.77].flatMap((phase) => [
          [0, phase],
          [phase, 0],
        ]),
      ].map(([sourceX = NaN, sourceY = NaN]) => ({ angle, sourceX, sourceY })),
    ),
  )(
    "shared-grid ownership at ($sourceX,$sourceY), rotation $angle",
    async ({ angle, sourceX, sourceY }) => {
      const gridDirection = (x: number, y: number): Vec3 => {
        const direction = normalize([0.01 * x, 0.01 * y, 1]);
        return [
          Math.fround(Math.cos(angle) * direction[0] + Math.sin(angle) * direction[2]),
          Math.fround(direction[1]),
          Math.fround(-Math.sin(angle) * direction[0] + Math.cos(angle) * direction[2]),
        ];
      };
      const beams: number[] = [];
      const lowerX = Math.floor(sourceX),
        lowerY = Math.floor(sourceY);
      const corners = [0, 1].flatMap((y) =>
        [0, 1].map((x) => gridDirection(lowerX + x, lowerY + y)),
      );
      const sample = samplePatch(corners, sourceX - lowerX, sourceY - lowerY);
      for (const y of [-1, 0]) {
        for (const x of [-1, 0]) {
          for (const dy of [0, 1]) {
            for (const dx of [0, 1]) {
              beams.push(...gridDirection(x + dx, y + dy), 1);
            }
          }
          beams.push(x, y, 0, 0);
        }
      }
      const output = await computeReadback(source, new Float32Array(beams), 16, 4, [
        createBlackbodyTable().data,
        createStarTree([{ direction: sample.direction, temperature: 6500, flux: 1e-5 }]),
      ]);
      let total = 0;
      for (let index = 0; index < 4; index++) {
        expect(output[index * 4 + 3]).toBe(1);
        total += luminance(output.subarray(index * 4, index * 4 + 3));
      }
      const reference =
        (1e-5 * (1 - Math.abs(sourceX)) * (1 - Math.abs(sourceY))) / sample.jacobian;
      expect(Math.abs(total / reference - 1), JSON.stringify(Array.from(output))).toBeLessThan(
        0.002,
      );
    },
  );

  test("a point source on the bilinear fold remains explicitly unresolved", async () => {
    const corners = foldedPatch(0.02);
    const sample = samplePatch(corners, 0.5, 0.5);
    const output = await computeReadback(
      source,
      new Float32Array([
        ...corners.flatMap((corner) => [corner[0], corner[1], corner[2], 1]),
        0,
        0,
        0,
        0,
      ]),
      4,
      1,
      [
        createBlackbodyTable().data,
        createStarTree([{ direction: sample.direction, temperature: 6500, flux: 1e-5 }]),
      ],
    );
    expect(Array.from(output)).toEqual([0, 0, 0, 0]);
  });

  test("each folded image uses its own interpolated Killing energy", async () => {
    const corners = foldedPatch(0.2);
    const sample = samplePatch(corners, 0.2, 0.8);
    const energies = [0.8, 0.9, 0.6, 1.2].map(Math.fround);
    const beams: number[] = [];
    for (const y of [-1, 0]) {
      for (const x of [-1, 0]) {
        for (const [index, corner] of corners.entries()) {
          beams.push(...corner, energies[index] ?? NaN);
        }
        beams.push(x, y, 0, 0);
      }
    }
    const output = await computeReadback(source, new Float32Array(beams), 16, 4, [
      createBlackbodyTable().data,
      createStarTree([{ direction: sample.direction, temperature: 6500, flux: 1e-5 }]),
    ]);
    const reference = [
      [0.2, 0.8],
      [0.8, 0.2],
    ].reduce((sum, [u = NaN, v = NaN]) => {
      const energy =
        (1 - u) * (1 - v) * (energies[0] ?? NaN) +
        u * (1 - v) * (energies[1] ?? NaN) +
        (1 - u) * v * (energies[2] ?? NaN) +
        u * v * (energies[3] ?? NaN);
      return (
        sum + (1e-5 * blackbodyXYZ(6500 / energy)[1]) / blackbodyXYZ(6500)[1] / sample.jacobian
      );
    }, 0);
    let total = 0;
    for (let index = 0; index < 4; index++) {
      expect(output[index * 4 + 3]).toBe(1);
      total += luminance(output.subarray(index * 4, index * 4 + 3));
    }
    expect(Math.abs(total / reference - 1)).toBeLessThan(0.002);
  });

  test.for([1e-9, 0.001, 1, 1e6])("regular patch flux stays finite at scale %s", async (scale) => {
    const corners: Vec3[] = [
      [-1, -1],
      [1, -1],
      [-1, 1],
      [1, 1],
    ].map(([x = NaN, y = NaN]) => {
      const direction = normalize([scale * x, scale * y, 1]);
      return [Math.fround(direction[0]), Math.fround(direction[1]), Math.fround(direction[2])];
    });
    const sample = samplePatch(corners, 0.2, 0.3);
    const beams: number[] = [];
    for (const y of [-1, 0]) {
      for (const x of [-1, 0]) {
        for (const corner of corners) {
          beams.push(...corner, 1);
        }
        beams.push(x, y, 0, 0);
      }
    }
    const output = await computeReadback(source, new Float32Array(beams), 16, 4, [
      createBlackbodyTable().data,
      createStarTree([{ direction: sample.direction, temperature: 6500, flux: 1e-5 }]),
    ]);
    expect(output.every(Number.isFinite)).toBe(true);
    let total = 0;
    for (let index = 0; index < 4; index++) {
      expect(output[index * 4 + 3]).toBe(1);
      total += luminance(output.subarray(index * 4, index * 4 + 3));
    }
    expect(Math.abs(total / (1e-5 / sample.jacobian) - 1)).toBeLessThan(0.002);
  });

  test("fold roots survive chart changes, parity, and motion toward the caustic", async () => {
    const cases = [0.02, 0.5, 2].flatMap((scale) =>
      [0, 0.7, 1.5, 2.8].flatMap((angle) =>
        [0.2, 0.35, 0.49].map((u) => {
          const corners = foldedPatch(scale).map(([x, y, z]): Vec3 => [
            Math.fround(Math.cos(angle) * x + Math.sin(angle) * z),
            y,
            Math.fround(-Math.sin(angle) * x + Math.cos(angle) * z),
          ]);
          const sample = samplePatch(corners, u, 1 - u);
          return { scale, angle, u, corners, direction: sample.direction };
        }),
      ),
    );
    const inputs: number[] = [];
    for (const item of cases) {
      for (const corner of item.corners) {
        inputs.push(...corner, 1);
      }
      inputs.push(...item.direction, 0);
    }
    const output = await computeReadback(
      `${unpartitioned}\n${radiation}\n${stars}\n${patchSource}
@group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
@group(0) @binding(3) var<storage, read> star_nodes: array<StarNode>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  let offset = id.x * 5u;
  let cell = StellarPatch(inputs[offset], inputs[offset+1u], inputs[offset+2u], inputs[offset+3u]);
  let roots = stellar_patch_roots(cell, inputs[offset+4u].xyz);
  outputs[id.x*2u] = vec4f(roots.positions[0], roots.positions[1]);
  outputs[id.x*2u+1u] = vec4f(f32(roots.count), f32(roots.valid), 0.0, 0.0);
}`,
      new Float32Array(inputs),
      cases.length * 8,
      cases.length,
    );
    for (const [index, item] of cases.entries()) {
      const values = output.subarray(index * 8, index * 8 + 8);
      const label = JSON.stringify({
        scale: item.scale,
        angle: item.angle,
        u: item.u,
        values: Array.from(values),
      });
      expect(values[4], label).toBe(2);
      expect(values[5], label).toBe(1);
      const roots = [
        [values[0] ?? NaN, values[1] ?? NaN],
        [values[2] ?? NaN, values[3] ?? NaN],
      ].toSorted((a, b) => (a[0] ?? NaN) - (b[0] ?? NaN));
      expect(
        Math.hypot((roots[0]?.[0] ?? NaN) - item.u, (roots[0]?.[1] ?? NaN) - (1 - item.u)),
        label,
      ).toBeLessThan(2e-4);
      expect(
        Math.hypot((roots[1]?.[0] ?? NaN) - (1 - item.u), (roots[1]?.[1] ?? NaN) - item.u),
        label,
      ).toBeLessThan(2e-4);
    }
  });

  test("shared-edge orientation agrees with exact binary32 products through cancellation and underflow", async () => {
    const cases: number[][] = [];
    for (const scale of [1e-42, 1e-30, 1e-20, 1e-19, 1e-16, 1e-10, 1]) {
      for (let step = 0; step < 16; step++) {
        const x = Math.fround(scale * (1 + step * 2 ** -23));
        const y = Math.fround(scale * (1 - step * 2 ** -23));
        const center = Math.fround(scale);
        cases.push(
          [x, center, center, y],
          [center, x, y, center],
          [-x, center, center, -y],
          [x, center, -center, -y],
          [x, y, x, y],
          [0, x, y, 0],
        );
      }
    }
    const output = await computeReadback(
      `${unpartitioned}\n${radiation}\n${stars}\n${patchSource}
@group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(3) var<storage, read> star_nodes: array<StarNode>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
  let value = inputs[id.x];
  outputs[id.x] = f32(stellar_orientation(value.xy, value.zw));
}`,
      new Float32Array(cases.flat()),
      cases.length,
      cases.length,
    );
    for (const [index, values] of cases.entries()) {
      // Two f32 significands multiply exactly in binary64. Cancellation also
      // retains their determinant sign; all products are normal binary64 values.
      const determinant =
        (values[0] ?? NaN) * (values[3] ?? NaN) - (values[1] ?? NaN) * (values[2] ?? NaN);
      const sign = determinant === 0 ? 0 : determinant > 0 ? 1 : -1;
      expect(output[index], JSON.stringify(values)).toBe(sign);
    }
  });
});

describe("Image ownership", () => {
  test("ordinary and critical images share a detector pixel without duplicate or missing flux", async ({
    device,
  }) => {
    using owned = new DisposableStack();

    device.pushErrorScope("validation");
    const makeBuffer = (size: number, usage: GPUBufferUsageFlags) =>
      owned.adopt(device.createBuffer({ size, usage }), (value) => value.destroy());
    const first = normalize([0.012, 0.02, 1]);
    const second = normalize([0.065, 0.02, 1]);
    const tree = createStarTree(
      [first, second].map((direction) => ({ direction, temperature: 6500, flux: 1e-5 })),
    );
    let sourceIndex = -1;
    for (let index = 0; index < tree.length / 8; index++) {
      if ((tree[index * 8 + 3] ?? 0) < 0 && Math.abs((tree[index * 8] ?? NaN) - first[0]) < 1e-7) {
        sourceIndex = index;
      }
    }
    if (sourceIndex < 0) {
      throw new Error("Missing source leaf.");
    }
    const module = await compileShader(
      device,
      `${ownership}\n${radiation}\n${stars}\n${composition}
@group(0) @binding(0) var<storage, read_write> prepared_images: array<StellarImage>;
@group(0) @binding(1) var<storage, read_write> colors: array<vec4f>;
@group(0) @binding(3) var<storage, read> star_nodes: array<StarNode>;
@compute @workgroup_size(1) fn prepare() {
  // The normalized affine map has J(X,Y) = 0.01 / (1 + 0.01*(X²+Y²))^(3/2).
  let area = 0.01 / pow(1.0 + 0.01 * (0.12 * 0.12 + 0.2 * 0.2), 1.5);
  let flux = star_nodes[${sourceIndex}u].upper.x * blackbody_radiance(6500.0).rgb / area;
  prepared_images[0] = StellarImage(vec4f(12.0,8.0,0.12,0.2),vec4f(flux,1.0),vec4f(0.0),vec4f(0.0));
}
@compute @workgroup_size(1) fn probe() {
  let direction = vec3f(0.0,0.0,1.0);
  let dx = vec3f(0.1,0.0,0.0);
  let dy = vec3f(0.0,0.1,0.0);
  colors[0] = point_stars(direction,dx,dy,1.0,StellarFootprint(vec2f(12.0,8.0),0.0));
  let ordinary = point_stars(direction,dx,dy,1.0,StellarFootprint(vec2f(12.0,8.0),1.0));
  colors[1] = vec4f(ordinary.rgb + stellar_detector_flux(vec2i(12,8)),ordinary.a);
  colors[2] = vec4f(select(0.0,1.0,stellar_search_owns(vec2f(12.12,8.2))),
    select(0.0,1.0,stellar_search_owns(vec2f(12.65,8.2))),
    select(0.0,1.0,stellar_search_owns(vec2f(12.0,8.0))),0.0);
}`,
      "stellar ownership and composition",
    );
    const pipelines = await Promise.all(
      ["prepare", "bin_stellar_images", "probe"].map((entryPoint) =>
        device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint } }),
      ),
    );
    const table = createBlackbodyTable();
    const blackbody = makeBuffer(
      table.data.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );
    const sources = makeBuffer(tree.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const frame = makeBuffer(96, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const partition = makeBuffer(64, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const images = makeBuffer(64, GPUBufferUsage.STORAGE);
    const counts = makeBuffer(32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const heads = makeBuffer(16 * 16 * 4, GPUBufferUsage.STORAGE);
    const links = makeBuffer(64, GPUBufferUsage.STORAGE);
    const output = makeBuffer(48, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const readback = makeBuffer(48, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(blackbody, 0, table.data);
    device.queue.writeBuffer(sources, 0, tree);
    device.queue.writeBuffer(
      frame,
      0,
      new Float32Array([
        16, 16, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0,
      ]),
    );
    device.queue.writeBuffer(
      partition,
      0,
      new Float32Array(
        [-Math.PI, -Math.PI / 2, 0, Math.PI / 2].flatMap((angle) => [angle, -Math.sqrt(3), 0, 1]),
      ),
    );
    device.queue.writeBuffer(counts, 0, new Uint32Array([1]));
    const resources = new Map([
      [0, images],
      [1, output],
      [2, blackbody],
      [3, sources],
      [19, frame],
      [20, partition],
      [21, images],
      [22, counts],
      [23, heads],
      [24, links],
    ]);
    const stages = [
      [0, 2, 3],
      [19, 20, 21, 22, 23, 24],
      [1, 2, 3, 19, 20, 21, 23, 24],
    ];
    const encoder = device.createCommandEncoder();
    for (const [index, pipeline] of pipelines.entries()) {
      const entries = (stages[index] ?? []).map((binding) => {
        const buffer = resources.get(binding);
        if (!buffer) {
          throw new Error("Missing test resource.");
        }
        return { binding, resource: { buffer } };
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(
        0,
        device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }),
      );
      pass.dispatchWorkgroups(1);
      pass.end();
    }
    encoder.copyBufferToBuffer(output, 0, readback, 0, 48);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(readback.getMappedRange()).slice();
    readback.unmap();
    expect(await device.popErrorScope()).toBeNull();
    expect(Array.from(values.subarray(8, 11))).toEqual([1, 0, 0]);
    expect(values[3]).toBe(1);
    expect(values[7]).toBe(1);
    for (let channel = 0; channel < 3; channel++) {
      expect(Math.abs((values[channel + 4] ?? NaN) / (values[channel] ?? NaN) - 1)).toBeLessThan(
        2e-5,
      );
    }
  });
});
