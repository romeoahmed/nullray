# Coverage

This ledger identifies checks that exercise the production implementation and the limits of their conclusions. A test's presence is not evidence that it ran; the recorded verification run must identify its source and environment.

## Executable contracts

| Area                        | Checks                                                                                                                              | Meaning                                                                                                                                        |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Geometry and observers      | [CPU relativity](../tests/cpu/relativity.test.ts)                                                                                   | Metric/BL agreement, orthonormal frames, independent Hamiltonian trajectories, proper-time motion, analytic principal and bifurcation families |
| Native trajectories         | [GPU relativity](../tests/gpu/relativity.test.ts)                                                                                   | Quantized observer/Carter/WP data, plasma trajectories, image radiance, source-domain continuation and the Schwarzschild critical boundary     |
| Coupled input state         | [Scene](../tests/cpu/scene.test.ts), [session](../tests/cpu/session.test.ts)                                                        | Atomic validation, camera/worldline independence, navigation, appearance boundaries and schema-1 round trips                                   |
| Spectral sources            | [CPU sources](../tests/cpu/sources.test.ts), [GPU radiation](../tests/gpu/radiation.test.ts)                                        | Charged emitters, disk flux, CIE integration, frequency shifts, finite-cutoff synchrotron and material continuity                              |
| GPU-generated sky and stars | [GPU sources](../tests/gpu/sources.test.ts)                                                                                         | Scalar lattice agreement, independent diffuse cube/mip flux and stellar tree queries versus direct sums                                        |
| Source exclusion            | [Barriers](../tests/gpu/barriers.test.ts)                                                                                           | Certified radial signs against exact dyadic arithmetic; distinct source-free, disk, singular and asymptotic outcomes                           |
| Heated refraction           | [Heating](../tests/gpu/heating.test.ts)                                                                                             | Cartesian gradients versus independent finite differences in both regular charts; production transfer changes with heating and epoch           |
| Photograph and display      | [Sampling](../tests/gpu/sampling.test.ts), [imaging](../tests/gpu/imaging.test.ts)                                                  | HDR means, missing weights, coverage reduction, bloom, color conversion and local stellar gradients                                            |
| Browser ownership           | [App](../tests/browser/app.test.ts), [preferences](../tests/browser/preferences.test.ts), [worker](../tests/browser/worker.test.ts) | Controls, focus, local persistence/media, canvas remount, worker lifetime and photographic state                                               |

The [inner-horizon regression](../tests/regressions/inner-horizon-reflection.json) retains detector inputs and a failing shader identity for an incorrect return to the illuminated exterior. The [material-tail regression](../tests/regressions/material-empty-tail.json) preserves rays whose foreground light was discarded after leaving all material. Their checks assert physical destination and retained radiation, not a frozen screenshot or elapsed-time threshold.

Analytic zero-energy families include Schwarzschild cycloidal free fall and a rotating charged orbit through both bifurcation spheres. The generic polarization reference integrates the metric connection independently before comparing Walker–Penrose contractions. Complementary analyzers recover stored intensity, and thin jet radiance is checked for electron-density scaling.

## Verification record

The 2026-09-11 local verification passed strict Oxlint/TypeScript checks and the Vite production build, 40 CPU tests, 26 actual GPU tests and 8 browser tests. Five CPU benchmarks and seven GPU/startup benchmarks completed. Browser execution used Chromium 153 on macOS with the reported Apple `metal-3` adapter; the adapter did not expose a specific device name. Desktop (1440 × 900), portrait (390 × 844) and landscape (844 × 390) screenshots were inspected, and toolbar layout was checked at widths of 320 and 390 pixels.

The 64-pixel-face diffuse cube's base-level relative RMS coefficient error against the independent reference was 0.0508%; all tested mip flux checks passed. The complete 64³ scalar lattice matched its integer reference, and stellar tree queries agreed with direct spectral-kernel sums for the tested regular, duplicate and anisotropic source cases.

At 320 × 180 and one sample per pixel, mean submitted-frame wall times were 22.73 ms (disk), 24.09 ms (jet), 37.31 ms (blue flow), 12.00 ms (plasma), 5.90 ms (interior) and 34.88 ms (naked). Each workload used one warmup and eight measured frames with reused resources; all six recorded zero unresolved weight. The interior workload was dark. These samples do not certify the wider scene domain. Existing-device optical initialization averaged 65.32 ms across five samples after one warmup, including catalogue preparation and GPU source generation. This is neither cold launch nor a universal performance claim.

Local evidence is written under `test-results/`: `bench/` contains timings, complete workload inputs, browser/adapter information, coverage and source/harness SHA-256 identities; `sky-*.json` contains diffuse-source errors and mip flux; `explorer-*.png` contains visual review captures. These generated artifacts are not committed fixtures. Benchmark workloads and initialization responsibilities changed during the rewrite, so these measurements establish the current implementation without claiming a general speedup from older runs.

## Open coverage

- General horizon-critical families beyond the exact implemented $E=0$, $aL=0$ continuation; direct observer placement on bifurcation spheres; simultaneous or ambiguous events.
- Exact horizon intersections beyond accepted-step brackets, critical-image positions, image derivatives and full combined-domain convergence.
- Nonlinear stellar-image completeness and caustic-resolved integrated flux. Local tree/kernel checks do not prove these properties.
- General degenerate vacuum polarization screens, polarization in refracting plasma, broadband refraction, magnetized transfer, self-absorption and fluid evolution.
- Broader source/observer combinations, time-dependent image convergence and physical-display HDR behavior.

Numerical budget exhaustion remains unresolved. Image darkness, successful source exclusion and singularity termination are different outcomes. [Validation](validation.md) defines the evidence required before changing these status statements.
