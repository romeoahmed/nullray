/**
 * Carlson's symmetric integral RF(x, y, z), for nonnegative real arguments
 * with at most one zero. Duplication and the local series follow DLMF 19.36.
 * Arguments are scaled before iteration to protect the arithmetic range.
 */
export function carlsonRF(x: number, y: number, z: number): number {
  const scale = Math.max(x, y, z);
  if (
    ![x, y, z].every(Number.isFinite) ||
    Math.min(x, y, z) < 0 ||
    scale === 0 ||
    (x === 0 && y === 0) ||
    (x === 0 && z === 0) ||
    (y === 0 && z === 0)
  ) {
    throw new RangeError("RF requires finite nonnegative arguments and at most one zero.");
  }
  let a = x / scale;
  let b = y / scale;
  let c = z / scale;
  for (let iteration = 0; iteration < 64; iteration++) {
    const mean = (a + b + c) / 3;
    const dx = (mean - a) / mean;
    const dy = (mean - b) / mean;
    const dz = (mean - c) / mean;
    if (Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) < 0.0025) {
      const e2 = dx * dy - dz * dz;
      const e3 = dx * dy * dz;
      return (
        (1 + (e2 / 24 - 0.1 - (3 * e3) / 44) * e2 + e3 / 14) / Math.sqrt(mean) / Math.sqrt(scale)
      );
    }
    const sa = Math.sqrt(a);
    const sb = Math.sqrt(b);
    const sc = Math.sqrt(c);
    const lambda = sa * sb + sa * sc + sb * sc;
    a = (a + lambda) / 4;
    b = (b + lambda) / 4;
    c = (c + lambda) / 4;
  }
  throw new RangeError("RF duplication did not converge.");
}

export interface Jacobi {
  readonly sn: number;
  readonly cn: number;
  readonly dn: number;
}

/**
 * Real Jacobi functions in parameter convention m = k², including m < 0.
 * Descending Landen transformations invert the arithmetic-geometric mean.
 * The trigonometric and hyperbolic limits are evaluated directly.
 */
export function jacobi(u: number, m: number): Jacobi {
  if (!Number.isFinite(u) || !Number.isFinite(m) || m > 1) {
    throw new RangeError("Jacobi evaluation requires finite u and m <= 1.");
  }
  if (m < 0) {
    const scale = Math.sqrt(1 - m);
    const transformedPhase = u * scale;
    const transformedParameter = m / (m - 1);
    if (!Number.isFinite(transformedPhase) || transformedParameter >= 1) {
      throw new RangeError("The negative-parameter transformation exceeds binary64 resolution.");
    }
    const v = jacobiPositive(transformedPhase, transformedParameter);
    return { sn: v.sn / (scale * v.dn), cn: v.cn / v.dn, dn: 1 / v.dn };
  }
  return jacobiPositive(u, m);
}

function jacobiPositive(u: number, m: number): Jacobi {
  if (m === 0) {
    return { sn: Math.sin(u), cn: Math.cos(u), dn: 1 };
  }
  if (m === 1) {
    const sech = 1 / Math.cosh(u);
    return { sn: Math.tanh(u), cn: sech, dn: sech };
  }
  let a = 1;
  let b = Math.sqrt(1 - m);
  let multiplier = 1;
  const ratios: number[] = [];
  for (let iteration = 0; iteration < 24; iteration++) {
    const nextA = (a + b) / 2;
    const ratio = (a - b) / (a + b);
    if (Math.abs(ratio) < 2e-16) {
      break;
    }
    ratios.push(ratio);
    b = Math.sqrt(a * b);
    a = nextA;
    multiplier *= 2;
  }
  // Reduce by the full real period before multiplying the phase during descent.
  const period = (2 * Math.PI) / a;
  let phase = multiplier * a * (u % period);
  for (let i = ratios.length - 1; i >= 0; i--) {
    const ratio = ratios[i];
    if (ratio === undefined) {
      throw new Error("Missing Landen stage.");
    }
    phase = (phase + Math.asin(ratio * Math.sin(phase))) / 2;
  }
  const sn = Math.sin(phase);
  const cn = Math.cos(phase);
  return { sn, cn, dn: Math.sqrt(1 - m * sn * sn) };
}
