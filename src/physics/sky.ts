import { brightStars } from "../data/bright-stars.ts";
import type { Vec3 } from "./vector.ts";

/**
 * Point source in the fixed J2000 equatorial sky; no distance or proper motion.
 */
export interface Star {
  /** Source direction with +x at RA zero and +z at the north celestial pole. */
  readonly direction: Vec3;
  /** Blackbody color proxy in kelvin; not a measured effective temperature. */
  readonly temperature: number;
  /** Integrated source luminance over solid angle at the common scene normalization. */
  readonly flux: number;
}

/**
 * Convert the bundled bright HYG catalogue into caller-owned source records.
 *
 * @remarks
 * Ballesteros (2012), equation 14, maps B−V to a blackbody proxy; absent colors
 * use 6500 K. V magnitudes approximate relative CIE Y flux with an artistic
 * zero point. Dataset attribution and selection are recorded in NOTICE.md.
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

/**
 * Decode the borrowed HYG extension buffer into fresh source records.
 *
 * @remarks
 * Each 16-byte record stores little-endian f32 right ascension and declination
 * in radians, apparent V magnitude, and blackbody-proxy temperature in kelvin.
 *
 * @returns Caller-owned sources; the input buffer is not retained or mutated.
 * @throws RangeError - If record length, coordinates, magnitude, or temperature fail validation.
 */
export function decodeFaintStars(data: ArrayBuffer): readonly Star[] {
  if (data.byteLength % 16 !== 0 || data.byteLength > 200_000 * 16) {
    throw new RangeError("Invalid faint-star catalogue length.");
  }
  const view = new DataView(data);
  const stars: Star[] = [];
  for (let offset = 0; offset < data.byteLength; offset += 16) {
    const ra = view.getFloat32(offset, true);
    const dec = view.getFloat32(offset + 4, true);
    const magnitude = view.getFloat32(offset + 8, true);
    const temperature = view.getFloat32(offset + 12, true);
    if (
      ![ra, dec, magnitude, temperature].every(Number.isFinite) ||
      ra < 0 ||
      ra > 2 * Math.PI ||
      Math.abs(dec) > Math.PI / 2 ||
      magnitude <= 6.5 ||
      magnitude > 10 ||
      temperature < 1000 ||
      temperature > 50000
    ) {
      throw new RangeError("Invalid faint-star catalogue record.");
    }
    const radial = Math.cos(dec);
    stars.push({
      direction: [radial * Math.cos(ra), radial * Math.sin(ra), Math.sin(dec)],
      temperature,
      flux: 4e-5 * 10 ** (-0.4 * magnitude),
    });
  }
  return stars;
}
