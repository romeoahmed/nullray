# Third-party attribution

Nullray's original code and documentation are licensed under [MIT](LICENSE). The material identified below retains its stated terms; the root license does not replace them.

## Example photographs

[Classic disk](docs/media/classic-disk.png) and [Blue accretion flow](docs/media/blue-accretion-flow.png) are rendered with Nullray, © 2026 Romeo Ahmed, and distributed under [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/). They use the HYG catalogue and CIE colour-matching data credited below. The renderings apply Nullray's source, spectral and display models to those data.

## Photographic tone mapping

The highlight shoulder in [present.wgsl](src/gpu/wgsl/passes/present.wgsl) adapts the peak-compression and neutral-highlight construction of the [Khronos PBR Neutral tone mapper](https://github.com/KhronosGroup/ToneMapping/tree/main/PBR_Neutral), copyright 2024 The Khronos Group, Inc., under [Apache-2.0](licenses/Khronos-Apache-2.0.txt). Nullray omits the surface-reflection offset, changes the compression/desaturation parameters and applies the shoulder to Display P3 with an explicit SDR/HDR content peak. These changes do not implement the complete PBR Neutral transform or imply Khronos endorsement.

## CIE 1931 standard colorimetric observer

[src/data/cie1931.ts](src/data/cie1931.ts) contains the colour-matching functions of the CIE 1931 standard colorimetric observer, published by the International Commission on Illumination (CIE), Vienna, Austria, 2019. Dataset DOI: [10.25039/CIE.DS.xvudnb9b](https://doi.org/10.25039/CIE.DS.xvudnb9b). Original source: CIE 018:2019, Table 6.

The data is licensed under [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/). This license applies to the included data and its adaptations. The conversion preserves all decimal values and makes the 360–830 nm wavelength column implicit in the array index. It does not imply CIE endorsement of Nullray.

Original file: `CIE_xyz_1931_2deg.csv`. SHA-256: `fa663e3535a7e0763a745993a1f0a192eb0275ac46ad2d1befd7626841e713c1`. See the [dataset page](https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer) and its linked metadata for provenance.

## HYG star catalogue

[src/data/bright-stars.ts](src/data/bright-stars.ts) and [src/data/faint-stars.bin](src/data/faint-stars.bin) adapt David Nash / Astronexus's [HYG Database 4.4](https://codeberg.org/astronexus/hyg), licensed under [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/). That license applies to the included catalogue and its adaptations. No endorsement is implied. The upstream repository has moved from GitHub to Codeberg.

### Pinned source

The source is [`data/hyg/CURRENT/hyg_v44.csv.gz`](https://codeberg.org/astronexus/hyg/src/commit/53e3df311869e813ace5f1ad2ec4ce909f13256c/data/hyg/CURRENT/hyg_v44.csv.gz) at commit `53e3df311869e813ace5f1ad2ec4ce909f13256c`, dated 2026-07-12. Import uses the original HYG catalogue, not the separate AT-HYG/HYGLike dataset in the same repository. The [upstream dataset notes](https://codeberg.org/astronexus/hyg/src/commit/53e3df311869e813ace5f1ad2ec4ce909f13256c/data/hyg/README.md) describe the version's duplicate cleanup, labeling and astrometric changes; the [dataset license](https://codeberg.org/astronexus/hyg/src/commit/53e3df311869e813ace5f1ad2ec4ce909f13256c/data/hyg/CURRENT/LICENSE) records CC BY-SA 4.0.

| Artifact                  |      Bytes | SHA-256                                                            |
| ------------------------- | ---------: | ------------------------------------------------------------------ |
| Upstream gzip             | 13,636,362 | `00b349893b9a53106dd488d8371e8d2fa586043e500bb3cdb8bff3931682197d` |
| Decompressed CSV          | 33,929,696 | `ea0e1699a1e7d48daa197b5fe567fcf447420e367f37619862ee1d6f077599cd` |
| Bundled faint-star binary |  1,586,352 | `ff74992bccb2773567a2d901b64ad58de42c3eee1fadc742a6d30c3b886cd886` |

### Selection and conversion

Read the gzip as UTF-8 CSV with its header and preserve source row order. Exclude the Sun (`id = 0`) and select finite apparent visual magnitudes $m_V\le10$. This retains 108,067 sources from 119,614 CSV data rows, including the excluded Solar row in the upstream count. Use `rarad` and `decrad` directly: they are right ascension and declination in radians in the catalogue's J2000 frame.

- Bright sources: retain the 8,920 rows with $m_V\le6.5$ as `[id, rarad, decrad, mag, ci]`. Parse numeric fields without additional coordinate rounding; 40 absent B−V color indices remain `null`.
- Faint sources: retain the 99,147 rows with $6.5<m_V\le10$. Each 16-byte record stores little-endian f32 `rarad`, `decrad`, `mag` and a derived blackbody-proxy temperature in kelvin. For a present color index, evaluate the [Ballesteros approximation](docs/physics/emission.md#distant-sky) in binary64 before the single f32 storage rounding. The 319 absent colors use 6500 K. The binary has no header, IDs or padding.

The catalogue supplies observed directions and relative visual magnitudes. The renderer's blackbody color approximation, missing-color assumption, scene normalization and diffuse background are described in [the emission model](docs/physics/emission.md#distant-sky); they are not additional catalogue measurements.

## Milne scattering atmosphere

[src/data/milne.ts](src/data/milne.ts) records selected numerical results for the nonmagnetic conservative atmosphere (h = 0) from Table 4 of N. A. Silant’ev, G. A. Alekseeva and Yu. K. Ananjevskaja, [“Radiation intensity and polarization in an atmosphere with a chaotic magnetic field”](https://doi.org/10.1093/mnras/stz123), MNRAS 484, 2786–2792 (2019). The paper is © 2019 The Author(s), published by Oxford University Press on behalf of the Royal Astronomical Society. The numerical intensity and polarization values at the selected angular nodes are retained; Nullray interpolates Stokes intensities and normalizes the hemispheric flux. No article prose or implementation code is included.
