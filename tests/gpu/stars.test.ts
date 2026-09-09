import { expect, test } from "vitest";
import { server } from "vitest/browser";
import { createBlackbodyTable } from "../../src/physics/radiation.ts";
import radiationSource from "../../src/render/shaders/radiation.wgsl?raw";
import starsSource from "../../src/render/shaders/stars.wgsl?raw";
import { createStarTree } from "../../src/physics/stars.ts";
import { cross, dot, normalize } from "../../src/physics/vector.ts";
import type { Vec3 } from "../../src/physics/vector.ts";
import { computeReadback } from "./compute.ts";

const source = `${radiationSource}
${starsSource}
@group(0) @binding(0) var<storage, read> beams: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> colors: array<vec4f>;
@group(0) @binding(3) var<storage, read> star_nodes: array<StarNode>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&colors)) { return; }
  colors[id.x] = point_stars(beams[3u * id.x].xyz, beams[3u * id.x + 1u].xyz, beams[3u * id.x + 2u].xyz, beams[3u * id.x].w);
}`;

test("point-source flux is stable through subpixel motion, shear, parity, and cube-edge directions", async () => {
  const directions: readonly Vec3[] = [[0, 0, 1], normalize([1, 1, 1]), normalize([-1, 0.2, 1])];
  const beams: number[] = [];
  const areas: number[] = [];
  const groups: number[] = [];
  // Each isolated patch observes one source; all three sources share a tree.
  const catalogue = directions.map((direction) => ({
    direction,
    temperature: 6500,
    flux: 1e-5,
  }));
  for (const direction of directions) {
    const horizontal = normalize(cross(direction, [0, 1, 0]));
    const vertical = cross(direction, horizontal);
    for (const scale of [0.001, 0.01]) {
      for (const stretch of [-3, 0.3, 1]) {
        for (const phase of [-0.49, -0.2, 0, 0.23, 0.49]) {
          groups.push(areas.length);
          const position = (x: number, y: number): Vec3 => {
            const component = (axis: 0 | 1 | 2) =>
              direction[axis] +
              scale * ((x + 0.35 * y) * horizontal[axis] + stretch * y * vertical[axis]);
            return normalize([component(0), component(1), component(2)]);
          };
          for (let y = -3; y <= 3; y++) {
            for (let x = -3; x <= 3; x++) {
              const center = position(x + phase, y + phase);
              const left = position(x + phase - 1, y + phase);
              const right = position(x + phase + 1, y + phase);
              const before = position(x + phase, y + phase - 1);
              const after = position(x + phase, y + phase + 1);
              const dx: Vec3 = [
                (right[0] - left[0]) / 2,
                (right[1] - left[1]) / 2,
                (right[2] - left[2]) / 2,
              ];
              const dy: Vec3 = [
                (after[0] - before[0]) / 2,
                (after[1] - before[1]) / 2,
                (after[2] - before[2]) / 2,
              ];
              beams.push(...center, 1, ...dx, 0, ...dy, 0);
              areas.push(Math.abs(dot(center, cross(dx, dy))));
            }
          }
        }
      }
    }
  }
  const result = await computeReadback(
    source,
    new Float32Array(beams),
    areas.length * 4,
    Math.ceil(areas.length / 64),
    [createBlackbodyTable().data, createStarTree(catalogue)],
  );
  for (const start of groups) {
    let flux = 0;
    for (let index = start; index < start + 49; index++) {
      expect(result[index * 4 + 3]).toBe(1);
      flux +=
        (0.2126390059 * (result[index * 4] ?? Number.NaN) +
          0.7151686788 * (result[index * 4 + 1] ?? Number.NaN) +
          0.0721923154 * (result[index * 4 + 2] ?? Number.NaN)) *
        (areas[index] ?? Number.NaN);
    }
    expect(Math.abs(flux / 1e-5 - 1)).toBeLessThan(0.002);
  }
});

test("stellar spectral range failures remain unresolved before unsafe frequency division", async () => {
  const energies = [1, 1e-6, 0, -1];
  const result = await computeReadback(
    source,
    new Float32Array(energies.flatMap((energy) => [0, 0, 1, energy, 0.01, 0, 0, 0, 0, 0.01, 0, 0])),
    energies.length * 4,
    1,
    [
      createBlackbodyTable().data,
      createStarTree([{ direction: [0, 0, 1], temperature: 6500, flux: 1e-5 }]),
    ],
  );
  expect(result.every(Number.isFinite)).toBe(true);
  expect(Array.from(result).filter((_, index) => index % 4 === 3)).toEqual([1, 0, 0, 0]);
});

test("stellar inversion retains finite coverage across large isotropic beam scales", async () => {
  const scales = [1e-11, 1e-9, 1, 1e15, 1e25].map(Math.fround);
  const result = await computeReadback(
    source,
    new Float32Array(scales.flatMap((scale) => [0, 0, 1, 1, scale, 0, 0, 0, 0, scale, 0, 0])),
    scales.length * 4,
    1,
    [
      createBlackbodyTable().data,
      createStarTree([{ direction: [0, 0, 1], temperature: 6500, flux: 1e-5 }]),
    ],
  );
  expect(result.every(Number.isFinite)).toBe(true);
  expect(result[3]).toBe(0);
  for (const [index, scale] of scales.entries()) {
    if (index === 0) {
      continue;
    }
    expect(result[index * 4 + 3]).toBe(1);
    const luminance =
      0.2126390059 * (result[index * 4] ?? NaN) +
      0.7151686788 * (result[index * 4 + 1] ?? NaN) +
      0.0721923154 * (result[index * 4 + 2] ?? NaN);
    if (scale > 1e20) {
      expect(luminance).toBe(0);
    } else {
      expect(Math.abs((luminance * scale * scale) / 1e-5 - 1)).toBeLessThan(0.002);
    }
  }
});

test.for([0.2, 0.75, 1.5])(
  "folded stellar map with image radius %s",
  async (radius, { annotate }) => {
    const scale = 0.001;
    const flux = 1e-5;
    const sourceDirection = normalize([scale * radius * radius, 0, 1]);
    const tree = createStarTree([{ direction: sourceDirection, temperature: 6500, flux }]);
    // Exact roots of the uploaded source's gnomonic coordinate under u = scale * x².
    const sourceU = Math.fround(sourceDirection[0]) / Math.fround(sourceDirection[2]);
    const root = Math.sqrt(sourceU / scale);
    const expected = (flux * (1 + sourceU * sourceU) ** 1.5) / (scale * scale * root);
    const beams: number[] = [];
    const groups = [];
    for (const subdivisions of [1, 2, 4, 8, 16, 32]) {
      for (const phase of [0.125, 0.375]) {
        const start = beams.length / 12;
        const step = 1 / subdivisions;
        for (let iy = -3 * subdivisions; iy < 3 * subdivisions; iy++) {
          for (let ix = -3 * subdivisions; ix < 3 * subdivisions; ix++) {
            const x = (ix + phase) * step;
            const y = (iy + phase) * step;
            const vector: Vec3 = [scale * x * x, scale * y, 1];
            const length = Math.hypot(...vector);
            const direction = normalize(vector);
            const derivative = (raw: Vec3): Vec3 => {
              const projection = dot(direction, raw);
              return [
                ((raw[0] - direction[0] * projection) * step) / length,
                ((raw[1] - direction[1] * projection) * step) / length,
                ((raw[2] - direction[2] * projection) * step) / length,
              ];
            };
            beams.push(
              ...direction,
              1,
              ...derivative([2 * scale * x, 0, 0]),
              0,
              ...derivative([0, scale, 0]),
              0,
            );
          }
        }
        groups.push({ subdivisions, phase, start, end: beams.length / 12 });
      }
    }
    const result = await computeReadback(
      source,
      new Float32Array(beams),
      beams.length / 3,
      Math.ceil(beams.length / 12 / 64),
      [createBlackbodyTable().data, tree],
    );
    const records = groups.map(({ subdivisions, phase, start, end }) => {
      let measured = 0;
      for (let index = start; index < end; index++) {
        expect(result[index * 4 + 3]).toBe(1);
        measured +=
          (0.2126390059 * (result[index * 4] ?? NaN) +
            0.7151686788 * (result[index * 4 + 1] ?? NaN) +
            0.0721923154 * (result[index * 4 + 2] ?? NaN)) /
          subdivisions ** 2;
      }
      return { subdivisions, phase, measured, relativeError: Math.abs(measured / expected - 1) };
    });
    const body = JSON.stringify({ radius, sourceU, expected, records }, null, 2);
    await server.commands.writeFile(`test-results/stellar-fold-${radius}.json`, body);
    await annotate("Analytic two-image fold integration", {
      body,
      bodyEncoding: "utf-8",
      contentType: "application/json",
    });
    for (const record of records.filter((value) => value.subdivisions === 32)) {
      expect(record.relativeError).toBeLessThan(0.02);
    }
  },
);
