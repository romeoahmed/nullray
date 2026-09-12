# Implementation and coverage ledger

This page maps production capabilities to executable checks and open coverage. A listed test is not evidence of a completed run; passing finite cases does not establish domain completeness.

## Capability status

| Capability         | Implemented scope                                                                                                             | Limit on the claim                                                                  |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Kerr–Newman optics | Horizon-regular neutral ray tracing, signed radius, charged circular emitters and physical observers                          | Finite f32 integration with unresolved work; not a field-equation solver            |
| Horizon topology   | Horizon/stationary-limit classification, ring detection in supported events, block continuation and signed infinity endpoints | No complete maximal-atlas certification or generic simultaneous-event proof         |
| Other domains      | Interior/white-hole/exterior-copy bookkeeping and negative-radius ends with independent illumination                          | Mathematical source boundaries; not an astrophysical traversal prediction           |
| Polarization       | Vacuum WP endpoint transport, Milne I/Q/U and linear analyzers                                                                | Degenerate screens incomplete; no magnetized or plasma polarization transport       |
| Disk and jets      | Neutral thin-disk flux shape, prescribed finite atmosphere and synchrotron outflow                                            | No GRMHD, dynamical launching, multiple scattering or self-absorption               |
| Refraction         | Stationary separable density and a time-dependent heated annulus at one frequency                                             | Monochrome preview; no catalogue stars or broadband dispersion                      |
| Image formation    | Direct quadrature, diffuse mips, local stellar footprints and retained missing weights                                        | Nonlinear stellar flux, missed subpixel images and extreme f16 saturation unbounded |
| Interaction        | Native/fixed/Auto raster modes, worker scheduling, bookmarks, inspection and PNG export                                       | Native speed and physical-display HDR depend on the environment                     |

## Executable contracts

| Area                        | Checks                                                                                                                                         | Meaning                                                                                                                                                                                        |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Geometry and observers      | [CPU relativity](../../tests/cpu/relativity.test.ts)                                                                                           | Metric/BL agreement, orthonormal frames, independent Hamiltonian trajectories, generated light-cone/metric-inverse properties, proper-time motion, analytic principal and bifurcation families |
| Native trajectories         | [GPU relativity](../../tests/gpu/relativity.test.ts)                                                                                           | Quantized observer/Carter/WP data, plasma trajectories, image radiance, source-domain continuation and the Schwarzschild critical boundary                                                     |
| Coupled input state         | [Scene](../../tests/cpu/scene.test.ts), [session](../../tests/cpu/session.test.ts)                                                             | Atomic validation, camera/worldline independence, navigation, appearance boundaries and schema-1 round trips                                                                                   |
| Spectral sources            | [CPU sources](../../tests/cpu/sources.test.ts), [GPU radiation](../../tests/gpu/radiation.test.ts)                                             | Charged emitters, disk flux, CIE integration, frequency shifts, finite-cutoff synchrotron and material continuity                                                                              |
| GPU-generated sky and stars | [GPU sources](../../tests/gpu/sources.test.ts)                                                                                                 | Scalar lattice agreement, independent diffuse cube/mip flux and stellar tree queries versus direct sums                                                                                        |
| Source exclusion            | [Barriers](../../tests/gpu/barriers.test.ts)                                                                                                   | Generated and boundary radial signs against exact dyadic arithmetic; distinct source-free, disk, singular and asymptotic outcomes                                                              |
| Heated refraction           | [Heating](../../tests/gpu/heating.test.ts)                                                                                                     | Cartesian gradients versus independent finite differences in both regular charts; production transfer changes with heating and epoch                                                           |
| Photograph and display      | [Sampling](../../tests/gpu/sampling.test.ts), [imaging](../../tests/gpu/imaging.test.ts)                                                       | Generated HDR direct-sum/partial-weight properties, coverage reduction, bloom, color conversion and local stellar gradients                                                                    |
| Sampling and raster policy  | [Quadrature](../../tests/cpu/sampling.test.ts), [resolution](../../tests/cpu/resolution.test.ts), [worker](../../tests/browser/worker.test.ts) | Pixel coverage/affine integration, exact fixed scales, explicit Auto policy, bounded settling and native export                                                                                |
| Browser ownership           | [App](../../tests/browser/app.test.ts), [preferences](../../tests/browser/preferences.test.ts), [worker](../../tests/browser/worker.test.ts)   | Controls, focus, local persistence/media, canvas remount, worker lifetime, current-revision export and photographic state                                                                      |

The [inner-horizon](../../tests/regressions/inner-horizon-reflection.json) and [material-tail](../../tests/regressions/material-empty-tail.json) fixtures preserve concrete failures with inputs/provenance. Checks assert physical destination and retained light, not screenshots or timings.

Analytic cases include zero-energy Schwarzschild free fall and rotating charged bifurcation motion. Generic polarization checks integrate the connection independently. Analyzer complements recover stored I; thin jet tests check density scaling.

## Open coverage

- General horizon-critical families beyond the exact implemented $E=0$, $aL=0$ continuation; direct observer placement on bifurcation spheres; simultaneous or ambiguous events.
- Exact horizon intersections beyond accepted-step brackets, critical-image positions, image derivatives and full combined-domain convergence.
- Nonlinear stellar-image completeness and caustic-resolved integrated flux. Local tree/kernel checks do not prove these properties.
- General degenerate vacuum polarization screens, polarization in refracting plasma, broadband refraction, magnetized transfer, self-absorption and fluid evolution.
- Broader source/observer combinations, time-dependent image convergence and physical-display HDR behavior.
- Structured-disk column normalization, proper-height accuracy away from the equator, thin/volume boundary differences, jet and surface-noise aliasing.
- Mixed thermal/jet normalization in the plasma preview; it is not a calibrated brightness-temperature image.
- Extreme-radiance saturation and Stokes consistency through separate f16 clamps. A resolved sample is not necessarily unsaturated.

Budget exhaustion stays unresolved. Darkness, proven source exclusion and singularity termination are distinct. [Validation methods](methods.md) defines the evidence needed to change this ledger.

[Visual tests](../../tests/visual/acceptance.test.ts) check capture completion and save artifacts; they do not provide a pixel-similarity gate or replace manual review.
