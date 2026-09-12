/**
 * Conservative Milne atmosphere rows `(μ, I(μ)/I(0), polarization percent)`.
 *
 * @remarks
 * Selected h = 0 data from Silant'ev, Alekseeva and Ananjevskaja (2019), Table 4.
 * These are tabulated atmosphere solutions, not measured fluid-state data.
 * Attribution is recorded in NOTICE.md.
 *
 * @see https://doi.org/10.1093/mnras/stz123
 */
export const milne: readonly (readonly [number, number, number])[] = [
  [0, 1, 11.713],
  [0.01, 1.036, 10.878],
  [0.02, 1.066, 10.295],
  [0.03, 1.094, 9.805],
  [0.04, 1.12, 9.374],
  [0.05, 1.146, 8.986],
  [0.06, 1.17, 8.631],
  [0.07, 1.194, 8.304],
  [0.08, 1.218, 8],
  [0.09, 1.241, 7.716],
  [0.1, 1.264, 7.449],
  [0.15, 1.375, 6.312],
  [0.2, 1.483, 5.41],
  [0.25, 1.587, 4.667],
  [0.3, 1.69, 4.041],
  [0.35, 1.791, 3.502],
  [0.4, 1.892, 3.033],
  [0.45, 1.991, 2.619],
  [0.5, 2.091, 2.252],
  [0.55, 2.189, 1.923],
  [0.6, 2.287, 1.627],
  [0.65, 2.385, 1.358],
  [0.7, 2.483, 1.113],
  [0.75, 2.58, 0.888],
  [0.8, 2.677, 0.682],
  [0.85, 2.774, 0.492],
  [0.9, 2.87, 0.316],
  [1, 3.063, 0],
];
