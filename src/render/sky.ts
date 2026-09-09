import { createSkyMap, createStars } from "../physics/sky.ts";
import { createStarTree } from "../physics/stars.ts";
import type { SkyLevel } from "../physics/sky.ts";

/** Fresh worker-local CPU storage for one diffuse sky and its spectral point-source tree. */
export interface CelestialData {
  readonly levels: readonly SkyLevel[];
  readonly stars: Float32Array<ArrayBuffer>;
}

/** Build source data on the render worker; buffers stay with their GPU owner. */
export function createCelestialData(): CelestialData {
  return { levels: createSkyMap(), stars: createStarTree(createStars()) };
}

/** Upload spectral coefficients and their solid-angle-weighted mip chain. Caller owns the texture. */
export function createSkyTexture(device: GPUDevice, levels: readonly SkyLevel[]): GPUTexture {
  const first = levels[0];
  if (!first) {
    throw new RangeError("The celestial cube must contain a base mip level.");
  }
  const texture = device.createTexture({
    label: "celestial spectra",
    size: [first.size, first.size, 6],
    mipLevelCount: levels.length,
    format: "rgba16float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  try {
    for (const [mipLevel, level] of levels.entries()) {
      device.queue.writeTexture(
        { texture, mipLevel },
        level.data,
        { bytesPerRow: level.size * 8, rowsPerImage: level.size },
        [level.size, level.size, 6],
      );
    }
    return texture;
  } catch (error) {
    texture.destroy();
    throw error;
  }
}
