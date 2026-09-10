/** Shared host binary64 metric; twelve consecutive IEEE word pairs (96 bytes). */
struct Observer64 {
  sine: Soft64,
  cosine: Soft64,
  radius2: Soft64,
  spin2: Soft64,
  cosine2: Soft64,
  sigma: Soft64,
  big_a: Soft64,
  lapse: Soft64,
  dragging: Soft64,
  azimuth_scale: Soft64,
  polar_scale: Soft64,
  radial_scale: Soft64,
}
