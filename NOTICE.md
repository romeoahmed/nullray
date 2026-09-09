# Third-party data

## CIE 1931 standard colorimetric observer

[src/data/cie1931.ts](src/data/cie1931.ts) contains the colour-matching functions of the CIE 1931 standard colorimetric observer, published by the International Commission on Illumination (CIE), Vienna, Austria, 2019. Dataset DOI: [10.25039/CIE.DS.xvudnb9b](https://doi.org/10.25039/CIE.DS.xvudnb9b). Original source: CIE 018:2019, Table 6.

The data is licensed under [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/). This license applies to the included data and its adaptations. The conversion preserves all decimal values and makes the 360–830 nm wavelength column implicit in the array index. It does not imply CIE endorsement of Nullray.

Original file: `CIE_xyz_1931_2deg.csv`. SHA-256: `fa663e3535a7e0763a745993a1f0a192eb0275ac46ad2d1befd7626841e713c1`. The checksum was verified before conversion. See the [dataset page](https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer) and its linked metadata for provenance.

## HYG bright-star catalogue

[src/data/bright-stars.ts](src/data/bright-stars.ts) adapts David Nash / Astronexus's [HYG Database 4.1](https://github.com/astronexus/HYG-Database/tree/main/hyg), licensed under [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/). That license applies to the included catalogue and its adaptations. No endorsement is implied.

Original file: [`hygdata_v41.csv`](https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/CURRENT/hygdata_v41.csv). SHA-256: `d9f69fd86bbf90a4e4d52b4c5c53eacfa6dfc0bfdef85bfd94f095e0bebe4ebd`. The conversion selects finite apparent visual magnitudes at most 6.5 and excludes the Sun (`id = 0`), retaining 8,920 records as `[id, rarad, decrad, mag, ci]`. Right ascension and declination are radians in the catalogue's J2000 frame. Forty absent color indices remain `null`. Values are parsed as numbers without coordinate rounding. Version 4.1 is pinned to this verified archive; [the author's revision notes](https://www.astronexus.com/projects/hyg-details) state that 4.2 changes names only.

The catalogue supplies observed directions and relative visual magnitudes. The renderer's blackbody color approximation, missing-color assumption, scene normalization, and diffuse background are described in [physics](docs/physics.md#spectral-transfer-and-display); they are not additional catalogue measurements.
