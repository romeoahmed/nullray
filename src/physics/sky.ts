import { brightStars } from "../data/bright-stars.ts";
import { dot, normalize } from "./vector.ts";
import type { Vec3 } from "./vector.ts";

/** Power-of-two encoding keeps dim spectral coefficients above half-float underflow. */
export const skyCoefficientScale = 1024;

/** Spectral populations stored as independent, linearly filterable coefficients. */
export const skyTemperatures = [4500, 6500, 12000] as const;

export interface Star {
  readonly direction: Vec3;
  /** Blackbody color proxy in kelvin; not a measured effective temperature. */
  readonly temperature: number;
  /** Integrated source luminance over solid angle at the common scene normalization. */
  readonly flux: number;
}

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

/** Exact solid angle of a cube texel, from the projected rectangle's four corners. */
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
 * HYG bright stars in J2000 equatorial axes (+x at RA 0, +z at the north pole).
 * Ballesteros (2012), equation 14, maps B−V to a blackbody color proxy.
 * Missing colors use 6500 K. V magnitudes approximate relative CIE Y flux;
 * the zero-magnitude scene normalization is artistic, not absolute photometry.
 */
export function createStars(): readonly Star[] {
  return brightStars.map(([, ra, dec, magnitude, colorIndex]) => {
    const radial = Math.cos(dec);
    const temperature =
      colorIndex === null
        ? 6500
        : 4600 * (1 / (0.92 * colorIndex + 1.7) + 1 / (0.92 * colorIndex + 0.62));
    return {
      direction: [radial * Math.cos(ra), radial * Math.sin(ra), Math.sin(dec)],
      temperature,
      flux: 4e-5 * 10 ** (-0.4 * magnitude),
    };
  });
}

/** Diffuse spectral radiance with solid-angle-weighted mip levels. Point stars are separate. */
export function createSkyMap(size = 128): readonly SkyLevel[] {
  if (!Number.isSafeInteger(size) || size < 1 || size > 1024 || (size & (size - 1)) !== 0) {
    throw new RangeError("Sky face size must be a power of two between 1 and 1024.");
  }
  let current = new Float32Array(size * size * 6 * 4);
  // J2000 north Galactic pole; see the source-frame contract in docs/physics.md.
  const poleRA = (192.8594812065348 * Math.PI) / 180;
  const poleDec = (27.12825118085622 * Math.PI) / 180;
  const galacticNormal: Vec3 = [
    Math.cos(poleDec) * Math.cos(poleRA),
    Math.cos(poleDec) * Math.sin(poleRA),
    Math.sin(poleDec),
  ];
  for (let face = 0; face < 6; face++) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const direction = cubeDirection(
          face,
          (2 * (x + 0.5)) / size - 1,
          (2 * (y + 0.5)) / size - 1,
        );
        const latitude = dot(direction, galacticNormal);
        const band =
          Math.exp((-latitude * latitude) / 0.018) *
          (1 - 0.7 * Math.exp((-latitude * latitude) / 0.0008));
        const clouds =
          0.75 + 0.25 * Math.sin(12 * direction[0] + 8 * direction[1] - 7 * direction[2]);
        const offset = ((face * size + y) * size + x) * 4;
        current[offset] = 0.0002 * band * clouds;
        current[offset + 1] = 0.0004 + 0.001 * band * clouds;
        current[offset + 2] = 0.000015 * band;
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
