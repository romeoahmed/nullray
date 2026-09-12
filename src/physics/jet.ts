/** Prescribed bipolar outflow; radii in M, angle in radians, speed in units c. */
export interface Jet {
  readonly inner: number;
  readonly outer: number;
  /** Outer half-width angle in radians; the inner width scales approximately as |z|^0.6. */
  readonly openingAngle: number;
  /** Core speed relative to the local circular ZAMO, in units of c. */
  readonly speed: number;
  /** Launch electron-density scale in cm⁻³. */
  readonly density: number;
  /** RMS isotropically tangled magnetic-field scale in gauss. */
  readonly field: number;
  /** Black-hole mass in solar masses, used to convert M to emissivity path length. */
  readonly massSolar: number;
  /** Low-energy cutoff of the power-law electron distribution, gamma >= gammaMin. */
  readonly gammaMin: number;
}
