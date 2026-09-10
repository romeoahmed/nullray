import { cross, dot, normalize } from "../../src/physics/vector.ts";
import type { Vec3 } from "../../src/physics/vector.ts";

/** Invert the actual quantized launch basis; f32 axes need not remain orthonormal. */
export function projectStellarDirection(frame: Float32Array, direction: Vec3) {
  const axis = (start: number): Vec3 => [
    frame[start] ?? NaN,
    frame[start + 1] ?? NaN,
    frame[start + 2] ?? NaN,
  ];
  const forward = axis(12),
    up = axis(16),
    right = axis(20);
  const determinant = dot(forward, cross(right, up));
  const depth = dot(direction, cross(right, up)) / determinant;
  const horizontal = dot(direction, cross(up, forward)) / determinant;
  const vertical = dot(direction, cross(forward, right)) / determinant;
  const scale = (frame[6] ?? NaN) / (frame[1] ?? NaN);
  return {
    pixel: [
      ((frame[0] ?? NaN) - 1) / 2 + horizontal / depth / scale,
      ((frame[1] ?? NaN) - 1) / 2 - vertical / depth / scale,
    ] as const,
    depth,
    determinant,
    scale,
  };
}

/** Reproject independent physical images and transform their solid-angle Jacobians. */
export function reprojectStellarImages<
  T extends { readonly pixel: readonly number[]; readonly jacobian: number },
>(frame: Float32Array, original: Float32Array, images: readonly T[]) {
  const width = original[0] ?? NaN,
    height = original[1] ?? NaN,
    zoom = original[6] ?? NaN;
  const originalScale = zoom / height;
  return images.map((sample) => {
    const sx = ((sample.pixel[0] ?? NaN) - (width - 1) / 2) * originalScale;
    const sy = ((sample.pixel[1] ?? NaN) - (height - 1) / 2) * originalScale;
    const direction = normalize([
      (original[12] ?? NaN) - sy * (original[16] ?? NaN) + sx * (original[20] ?? NaN),
      (original[13] ?? NaN) - sy * (original[17] ?? NaN) + sx * (original[21] ?? NaN),
      (original[14] ?? NaN) - sy * (original[18] ?? NaN) + sx * (original[22] ?? NaN),
    ]);
    const projected = projectStellarDirection(frame, direction);
    const sourceProjection = projectStellarDirection(original, direction);
    const ratio =
      Math.abs(projected.determinant / sourceProjection.determinant) *
      (projected.scale / originalScale) ** 2 *
      (projected.depth / sourceProjection.depth) ** 3;
    return {
      ...sample,
      pixel: projected.pixel,
      jacobian: sample.jacobian * ratio,
    };
  });
}
