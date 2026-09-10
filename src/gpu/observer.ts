/** Binary64 observer data shared by selected rays; layout matches Observer64 in WGSL. */
export const observerFrameBytes = 96;

/**
 * Write the stationary observer metric from an already quantized optical frame.
 * Preserve operation order: the GPU launch consumes these IEEE words directly.
 * The destination contains twelve binary64 values and is mutated in place.
 */
export function writeObserverFrame(frame: Float32Array, destination: Float64Array): void {
  const radius = frame[4] ?? NaN;
  const theta = frame[5] === Math.fround(Math.PI) ? Math.PI : (frame[5] ?? NaN);
  const spin = frame[8] ?? NaN;
  const charge = frame[9] ?? NaN;
  const sine = theta === 0 || theta === Math.PI ? 0 : Math.sin(theta);
  const cosine = Math.cos(theta);
  const rr = radius * radius;
  const aa = spin * spin;
  const qq = charge * charge;
  const cc = cosine * cosine;
  const sigma = rr + aa * cc;
  const delta = rr - 2 * radius + aa + qq;
  const rraa = rr + aa;
  const bigA = rraa * rraa - aa * delta * (sine * sine);
  destination.set([
    sine,
    cosine,
    rr,
    aa,
    cc,
    sigma,
    bigA,
    Math.sqrt((sigma * delta) / bigA),
    (spin * (2 * radius - qq)) / bigA,
    Math.sqrt(bigA / sigma),
    Math.sqrt(sigma),
    Math.sqrt(delta * sigma),
  ]);
}
