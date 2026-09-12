/** Periodic scalar lattice; bytes are normalized to [-1, 1] when sampled. */
export interface NoiseVolume {
  readonly size: number;
  readonly data: Uint8Array<ArrayBuffer>;
}

const fade = (t: number) => t * t * t * (t * (6 * t - 15) + 10);

/** Fresh deterministic source data, independent of clocks and rendering state. */
export function createNoiseVolume(): NoiseVolume {
  const size = 64;
  const data = new Uint8Array(size ** 3);
  for (let index = 0; index < data.length; index++) {
    let value = Math.imul(index ^ 0x6a09e667, 0x9e3779b1);
    value ^= value >>> 16;
    value = Math.imul(value, 0x85ebca6b);
    value ^= value >>> 13;
    value = Math.imul(value, 0xc2b2ae35);
    data[index] = (value ^ (value >>> 16)) >>> 24;
  }
  return { size, data };
}

/** Evaluate the ideal quintic lattice interpolant in binary64; the borrowed byte grid is unchanged. */
export function sampleNoise(volume: NoiseVolume, x: number, y: number, z: number): number {
  const { size, data } = volume;
  const ix = Math.floor(x),
    iy = Math.floor(y),
    iz = Math.floor(z);
  const u = fade(x - ix),
    v = fade(y - iy),
    w = fade(z - iz);
  let sum = 0;
  for (let dz = 0; dz < 2; dz++) {
    for (let dy = 0; dy < 2; dy++) {
      for (let dx = 0; dx < 2; dx++) {
        const index =
          (((iz + dz) & (size - 1)) * size + ((iy + dy) & (size - 1))) * size +
          ((ix + dx) & (size - 1));
        const weight = (dx === 0 ? 1 - u : u) * (dy === 0 ? 1 - v : v) * (dz === 0 ? 1 - w : w);
        sum += ((data[index] ?? 128) / 127.5 - 1) * weight;
      }
    }
  }
  return sum;
}
