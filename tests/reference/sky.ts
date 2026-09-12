import { dot, normalize } from "../../src/physics/vector.ts";
import type { Vec3 } from "../../src/physics/vector.ts";
import { createNoiseVolume, sampleNoise } from "./structure.ts";

/** Power-of-two encoding keeps dim spectral coefficients above half-float underflow. */
export const skyCoefficientScale = 1024;

export interface SkyLevel {
  readonly size: number;
  /** Six RGBA faces in WebGPU cube order; RGB hold spectral coefficients × skyCoefficientScale, not colors. */
  readonly data: Float16Array<ArrayBuffer>;
}

/** Face normal, horizontal axis, vertical axis in WebGPU's cube sampling convention. */
const faces: readonly (readonly [Vec3, Vec3, Vec3])[] = [
  [
    [1, 0, 0],
    [0, 0, -1],
    [0, -1, 0],
  ],
  [
    [-1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
  ],
  [
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
  ],
  [
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, -1],
  ],
  [
    [0, 0, 1],
    [1, 0, 0],
    [0, -1, 0],
  ],
  [
    [0, 0, -1],
    [-1, 0, 0],
    [0, -1, 0],
  ],
];

/** Unit direction in a WebGPU cube face chart; u and v locate the projected face. */
export function cubeDirection(face: number, u: number, v: number): Vec3 {
  const axes = faces[face];
  if (!axes) {
    throw new RangeError("A cube face index must lie between zero and five.");
  }
  const [normal, horizontal, vertical] = axes;
  return normalize([
    normal[0] + u * horizontal[0] + v * vertical[0],
    normal[1] + u * horizontal[1] + v * vertical[1],
    normal[2] + u * horizontal[2] + v * vertical[2],
  ]);
}

const solidAnglePrimitive = (u: number, v: number) =>
  Math.atan2(u * v, Math.sqrt(1 + u * u + v * v));

/** Closed-form cube-texel solid angle from projected corners, evaluated in binary64. */
export function cubeTexelSolidAngle(size: number, x: number, y: number): number {
  const left = (2 * x) / size - 1;
  const right = (2 * (x + 1)) / size - 1;
  const top = (2 * y) / size - 1;
  const bottom = (2 * (y + 1)) / size - 1;
  return (
    solidAnglePrimitive(right, bottom) -
    solidAnglePrimitive(left, bottom) -
    solidAnglePrimitive(right, top) +
    solidAnglePrimitive(left, top)
  );
}

/**
 * Generate the diffuse spectral cube and its solid-angle-weighted reference mips.
 *
 * @remarks
 * Geometry/integration use binary64; working arrays and stored mip coefficients
 * are deliberately quantized to match the production storage contract.
 * Catalogue point sources are evaluated separately.
 *
 * @param size - Power-of-two face size from 1 through 1024 texels.
 * @returns Caller-owned levels in WebGPU cube-face order.
 * @throws RangeError - If the face size is unsupported.
 */
export function createSkyMap(size = 256): readonly SkyLevel[] {
  if (!Number.isSafeInteger(size) || size < 1 || size > 1024 || (size & (size - 1)) !== 0) {
    throw new RangeError("Sky face size must be a power of two between 1 and 1024.");
  }
  let current = new Float32Array(size * size * 6 * 4);
  // J2000 north Galactic pole; see the source-frame contract in docs/physics/emission.md.
  const poleRA = (192.8594812065348 * Math.PI) / 180;
  const poleDec = (27.12825118085622 * Math.PI) / 180;
  const galacticNormal: Vec3 = [
    Math.cos(poleDec) * Math.cos(poleRA),
    Math.cos(poleDec) * Math.sin(poleRA),
    Math.sin(poleDec),
  ];
  const noise = createNoiseVolume();
  for (let face = 0; face < 6; face++) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const direction = cubeDirection(
          face,
          (2 * (x + 0.5)) / size - 1,
          (2 * (y + 0.5)) / size - 1,
        );
        const latitude = dot(direction, galacticNormal);
        const [nx, ny, nz] = direction;
        const warp = sampleNoise(noise, 5 * nx, 5 * ny, 5 * nz);
        let clouds = 0;
        let frequency = 12;
        let weight = 0.6;
        for (let octave = 0; octave < 5; octave++) {
          clouds +=
            weight * sampleNoise(noise, frequency * nx + warp, frequency * ny, frequency * nz);
          frequency *= 2;
          weight *= 0.6;
        }
        const band = Math.exp(-0.5 * (latitude / 0.13) ** 2);
        const dust = Math.exp(
          -3 * Math.exp(2 * clouds) * Math.exp(-0.5 * ((latitude + 0.025 * warp) / 0.035) ** 2),
        );
        const stars = band * Math.exp(2.4 * clouds);
        const offset = ((face * size + y) * size + x) * 4;
        current[offset] = 0.0012 * stars * Math.sqrt(dust);
        current[offset + 1] = 0.000008 + 0.0018 * stars * dust;
        current[offset + 2] = 0.00012 * stars * dust * dust;
      }
    }
  }
  const levels: SkyLevel[] = [];
  // Cube faces share the same projected texel areas; evaluate each grid once.
  const areas = (width: number) =>
    Float64Array.from({ length: width * width }, (_, index) =>
      cubeTexelSolidAngle(width, index % width, Math.floor(index / width)),
    );
  let texelAreas = areas(size);
  let width = size;
  while (true) {
    const data = Float16Array.from(current, (value) => value * skyCoefficientScale);
    if (!data.every(Number.isFinite)) {
      throw new RangeError("Sky coefficients exceed half-float storage range.");
    }
    levels.push({ size: width, data });
    if (width === 1) {
      return levels;
    }
    const nextWidth = width / 2;
    const next = new Float32Array(nextWidth * nextWidth * 6 * 4);
    const nextAreas = areas(nextWidth);
    for (let face = 0; face < 6; face++) {
      for (let y = 0; y < nextWidth; y++) {
        for (let x = 0; x < nextWidth; x++) {
          const parentArea = nextAreas[y * nextWidth + x] ?? NaN;
          for (let dy = 0; dy < 2; dy++) {
            for (let dx = 0; dx < 2; dx++) {
              const childArea = texelAreas[(2 * y + dy) * width + 2 * x + dx] ?? NaN;
              const child = ((face * width + 2 * y + dy) * width + 2 * x + dx) * 4;
              const parent = ((face * nextWidth + y) * nextWidth + x) * 4;
              for (let channel = 0; channel < 3; channel++) {
                next[parent + channel] =
                  (next[parent + channel] ?? 0) +
                  ((current[child + channel] ?? 0) * childArea) / parentArea;
              }
            }
          }
        }
      }
    }
    current = next;
    texelAreas = nextAreas;
    width = nextWidth;
  }
}
