/** Prescribed bipolar outflow; radii in M, angle in radians, speed in units c. */
export interface Jet {
  readonly inner: number;
  readonly outer: number;
  /** Width at outer radius is outer * tan(openingAngle); the inner flow collimates parabolically. */
  readonly openingAngle: number;
  readonly speed: number;
  /** Launch density scale in cm⁻³, and rms tangled magnetic field in gauss. */
  readonly density: number;
  readonly field: number;
  /** Converts geometrical path length to physical emissivity length. */
  readonly massSolar: number;
  /** Low-energy cutoff of the power-law electron distribution, gamma >= gammaMin. */
  readonly gammaMin: number;
}
