# Coverage and current limits

Nullray remains an optical prototype. Executable checks establish the cases below; they do not establish a complete joint spacetime, observer, camera, and source domain. [Validation](validation.md) defines the independent references and acceptance criteria. [Numerics](numerics.md) owns formulas and work budgets.

## Executable evidence

| Family                   | Executable evidence                                                                                                                                   | Boundary of the claim                                                                                                                    |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Coupled scene and camera | [CPU scene](../tests/cpu/scene.test.ts), [session](../tests/cpu/session.test.ts)                                                                      | Quantized spacetime, exterior placement, camera/navigation invariants, atomic state changes, strict version-1 view parsing               |
| Ordinary trajectories    | [CPU geodesics](../tests/cpu/geodesics.test.ts), [GPU events](../tests/gpu/ray-events.test.ts)                                                        | Hamiltonian segment agreement, event ordering, turning points, zero/negative energy, polar and Schwarzschild limits                      |
| Thermal sources          | [CPU sources](../tests/cpu/sources.test.ts), [GPU radiation](../tests/gpu/radiation.test.ts)                                                          | Neutral Kerr–Newman circular emitters, ISCO and disk profiles, spectra, frequency and emission-clock transfer                            |
| Elliptic arithmetic      | [CPU elliptic](../tests/cpu/elliptic.test.ts), [GPU elliptic](../tests/gpu/elliptic.test.ts), [binary64](../tests/gpu/arithmetic.test.ts)             | Known limits, near-degenerate roots, exact-integer arithmetic comparisons; no unrestricted long-phase guarantee                          |
| Critical rays            | [Local geometry](../tests/gpu/local-geometry.test.ts), [critical endpoints](../tests/gpu/critical-rays.test.ts)                                       | Actual production primary and boundary entry points compared with converged quantized-input references                                   |
| Sky derivatives          | [Differentials](../tests/gpu/ray-differentials.test.ts), [beams](../tests/gpu/sky-beams.test.ts)                                                      | Cartesian derivatives and independently converged solid-angle areas, including fractional boundary samples                               |
| Critical curves          | [CPU lensing](../tests/cpu/lensing.test.ts), [GPU curves](../tests/gpu/critical-curve.test.ts)                                                        | Vacuum double-root/null invariants, both axes, charged and near-extremal families, reciprocal-camera projection and tangents             |
| Stellar light            | [Filtering](../tests/gpu/stellar-filtering.test.ts), [sources](../tests/gpu/sources.test.ts), [physical images](../tests/gpu/stellar-lensing.test.ts) | Folded bilinear pairs, shared edges, spectral shifts, independent known-source positions and flux; incomplete physical image enumeration |
| Sampling and output      | [Sampling](../tests/gpu/sampling.test.ts), [imaging](../tests/gpu/imaging.test.ts), [rendering](../tests/gpu/rendering.test.ts)                       | Failure weights, partial workgroups, 64-sample history, normalized bloom, display transforms and padded PNG snapshots                    |
| Browser lifetime         | [Application](../tests/browser/app.test.ts), [preferences](../tests/browser/preferences.test.ts), [worker](../tests/browser/worker.test.ts)           | Native controls, local views/audio, motion cleanup, remount/retry, responsive geometry and photographic state                            |

Tests are grouped by behavior: six CPU files, fifteen GPU files, and three browser files. Strict optical acceptance runs inside `pnpm test:gpu`; no separate optional acceptance project exists. Assertions retain quantity-specific tolerances and real GPU execution. Failed, skipped, or missing prerequisites are reported separately.

## Retained counterexamples

Fixtures preserve quantized inputs, independent reference settings, and the measured failing versions. Historical paths inside provenance identify those versions and need not match today's directories.

| Fixture                                                                                                              | Preserved failure                                                                                      |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [Critical primary rays](../tests/fixtures/critical-primary-rays.json)                                                | Critical f32 launch/path conditioning                                                                  |
| [Boundary coverage](../tests/fixtures/boundary-coverage.json)                                                        | 59 incomplete pixels and all 944 boundary subpixels                                                    |
| [Critical derivatives](../tests/fixtures/critical-derivative-rays.json)                                              | Nearly parallel gradients and area cancellation                                                        |
| [Critical stellar images](../tests/fixtures/critical-stellar-images.json)                                            | Twenty-five charged-spacetime roots: deep, near-polar, sub-spacing, stalled, and mixed-cell families   |
| [Schwarzschild images](../tests/fixtures/schwarzschild-stellar-images.json)                                          | Six high-precision images, two opaque and four visible                                                 |
| [Distant sky arrival](../tests/fixtures/distant-sky-arrival.json)                                                    | Arrival-bracket cancellation and distant precision selection                                           |
| [Critical event order](../tests/fixtures/critical-event-order.json)                                                  | Higher-order opacity disagreement independently checked with a null Hamiltonian                        |
| [Outer-disk arrival](../tests/fixtures/outer-disk-arrival.json)                                                      | Distant inverse-radius conditioning                                                                    |
| [Nearly radial arrival](../tests/fixtures/nearly-radial-arrival.json)                                                | Near-radial boundary conditioning                                                                      |
| [Jacobi dn](../tests/fixtures/jacobi-dn.json)                                                                        | Cancellation at the limiting modulus                                                                   |
| [Launch rounding](../tests/fixtures/launch-roundoff.json), [camera rounding](../tests/fixtures/camera-roundoff.json) | Expression-context sensitivity in quantized launches                                                   |
| [Boundary beams](../tests/fixtures/boundary-beams.json)                                                              | Thin-branch derivative repair                                                                          |
| [Photographic boundaries](../tests/fixtures/photographic-boundaries.json)                                            | Open native-portrait failures: arrival brackets, outer-disk destination ties, and critical derivatives |

## Image acceptance

The production primary gate checks three retained critical pixels. Its sky-direction target is $5\times10^{-5}$; the retained mixed-precision result is approximately $2.12\times10^{-6}$. Disk azimuth, inverse radius, and delay have separate criteria. The boundary gate checks every destination and endpoint quantity for all 944 retained subpixels. Passing these finite corpora does not prove the entire precision-selection band.

Ten retained sky samples test differentiated preparation, cooperative transport, and restored Cartesian beams. Gradient and solid-angle area errors must each remain below 0.2%; the area reference independently converges below $10^{-5}$. The reference chooses its own well-conditioned directional basis. Convergence of two almost parallel gradients alone is insufficient evidence for their small determinant.

The Schwarzschild source-image gate renders production 640 × 360 frames with one 6500 K source and a dark diffuse sky, subtracting disk-only renders while retaining opacity. Primary, boundary, detector-jitter, yaw/pitch, and roll variants must recover integrated pair flux within 0.2%. A separate gate checks each of the four visible images in three camera frames so a bright image cannot conceal a missing faint companion. The renderer receives the source and frame, never the reference image positions. The oracle uses an independently integrated planar orbit and implicit image Jacobian.

Twenty-five retained Kerr–Newman images use independent binary64 source roots and converged parameter differences. Every matching image must lie within 0.001 physical pixels and 0.2% flux. The reference itself must meet a $10^{-7}$ source chord residual and $10^{-4}$ area-refinement criterion. One physical midpoint subdivision recovers ten images previously missed in mixed cells. These checks cover their retained roots; they do not prove uniqueness or image completeness elsewhere.

Bilinear cell tests retain opposite-parity pairs, canonical shared-edge ownership, chart rotations, seams, and spectral differences between images. Affine beam tests evaluate the Jacobian at the image position. Both are local interpolation models and can miss physical structure between sampled rays.

## Repeatability and finite queues

Source candidates are allocated in canonical coarse-cell/source/root order. A count pass, two-word unsigned prefix scan, and write pass make the retained subset independent of workgroup scheduling when the queue fills. The default capacity remains 32768. Carry propagation preserves offsets above $2^{32}$; diagnostic totals saturate instead of wrapping. Compact active queues schedule unfinished candidates while preserving their canonical state/output index. Overflow is counted explicitly; deterministic truncation does not recover omitted images or bound their flux. Fresh-device checks exercise complete searches, forced overflow, a partial block, and totals beyond the u32 range.

## Remaining release work

| Requirement                     | Current limit                                                                                                           | Evidence needed to close it                                                        |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Complete source-aware filtering | One physical subdivision level; still-mixed children and failed roots remain                                            | Adaptive source coverage across thin branches/caustics with controlled error       |
| Root ownership                  | Conforming interpolation edges; Newton roots may leave their seed cell                                                  | Unique physical-image ownership without losing nearby same-branch images           |
| Omitted light                   | Finite radial strip, grid, candidate and iteration budgets                                                              | A defensible omitted-flux bound, including unresolved and higher-order images      |
| Combined optical domain         | Finite critical, polar, charged, near-extremal and distant corpora; open disk-boundary ties and derivative cancellation | Joint endpoint/derivative/source validation, including nonpositive-C axis handling |
| Responsive exploration          | Full physical source search dominates changed views                                                                     | Measured reductions at unchanged position/flux criteria and coverage               |
| Visual convergence              | Selective 4 × 4 exploration and 64 jittered photographic samples                                                        | Moving-star, fine-branch, and native-resolution convergence evidence               |

Sampled-ray alpha and photographic coverage count missing detector samples. They do not include every missing point-source image. Zero unresolved pixels therefore cannot certify stellar completeness. A higher iteration budget cannot repair a biased endpoint, and a smaller time with missing images is not an equal-quality improvement.

## Visual acceptance

Visual inspection covered the default rendered scene, bright-background panel legibility, native select menus, mobile portrait at 390 × 844 CSS pixels, and landscape at 844 × 390. The mobile About panel now retains the same edge spacing as Settings. Inspection is evidence of appearance and interaction, not an independent geodesic reference.

| Requirement               | Current conclusion                                                                                                                                                                                                                                                                                                 |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Disk realism              | Eight bounded source modes and lower default exposure/bloom give more variation, but smooth bands and strong highlights remain. The source is prescribed thermal emission, not a turbulent plasma simulation.                                                                                                      |
| Photon ring               | Thin higher-order disk arcs are visible and retained order/endpoint cases pass. A complete, converged photon ring is not established.                                                                                                                                                                              |
| Performance               | The full 720p solve remains hundreds of milliseconds. Cached-source timing does not describe a moving camera. See measurements below.                                                                                                                                                                              |
| Doppler effect            | Gravitational and emitter Doppler shifts are implemented and independently checked. Their visual asymmetry does not certify disk realism.                                                                                                                                                                          |
| Antialiasing              | Selective 4 × 4 quadrature and 64-sample photographs preserve missing weights. Fine arcs, moving stars, and large magnifications lack a general convergence guarantee.                                                                                                                                             |
| Unresolved color          | The deterministic 1388 × 1674, 64-sample survey at $r_o=30$, field of view $\pi/3$, and epoch zero reports **68 affected pixels and 4.375 missing sample weights**. This is not the same recorded scene/epoch/browser state as the earlier report of 14, so the counts are not a controlled regression comparison. |
| Lensed stellar arcs/rings | Known-source position and flux gates pass, but the default background's visible arcs remain weak. Point-source enumeration and extended diffuse structure are incomplete.                                                                                                                                          |

The photograph survey retains every failing frame, jitter and detector pixel in `test-results/photo-audit.json`. The checked-in [minimized corpus](../tests/fixtures/photographic-boundaries.json) includes destination disagreements at the outer disk, explicit unbracketed arrivals, and failed critical derivatives. Forcing integer preparation does not resolve all of them. Raw quadrature refinement also exposed persistent area bias in some critical rays; increasing budgets or loosening convergence alone would not establish accuracy. These cases remain open, and the coverage-accounting test is not a zero-failure optical gate.

## Measurements

September 10 measurements used Apple M5, Metal-3, Chromium 153, and Node 26.8.1. Renderer previews were closed and GPU workloads ran serially; other system load was uncontrolled. The production fingerprint is `882ae81f6b14a3e0e5a93c13f2181e9ca35507578c46b9063ddd3cb9936ef0cc`, and the harness fingerprint is `fa8942edf80b0cbe9bb05fe2d4aded77c06ff425cb17abf726995f4e05e7b8b6`.

Artifacts under `test-results/bench/` include complete production/harness and data fingerprints, physical inputs, device/browser identity, work counters, paired GPU timestamps, and exact image readbacks. Generated reports are local measurements, not checked-in physical references. [Tooling](tooling.md#benchmarks) defines each timed scope. At least 64 measured samples follow at least 16 warmups, with minimum timing windows also enforced for short workloads. No timestamp intervals were discarded.

| Workload            | Mean GPU time | Relative margin of error | Unresolved pixels | Candidates / overflow / failed refinement |
| ------------------- | ------------: | -----------------------: | ----------------: | ----------------------------------------- |
| Full frame, 720p    |    265.060 ms |                   0.494% |                 0 | 8565 / 0 / 57                             |
| View sampling, 720p |     69.173 ms |                   0.317% |                 0 | 8565 / 0 / 57                             |
| Lighting, 720p      |     10.923 ms |                   0.997% |                 0 | 8565 / 0 / 57                             |
| Bloom, 720p         |      0.606 ms |                   7.710% |                 0 | Reused optical frame                      |
| Retrograde, 360p    |    420.133 ms |                   1.112% |                 2 | 52490 / 19722 / 34                        |
| Charged, 360p       |    224.116 ms |                   0.250% |                 0 | 16972 / 0 / 14                            |
| Near-extremal, 360p |     58.252 ms |                   1.435% |               556 | 0 / 0 / 0                                 |
| Distant, 360p       |    233.422 ms |                   0.573% |                 0 | 7901 / 0 / 24                             |

The near-extremal workload finds no candidates, has 1522 failed sampled stellar endpoints, and leaves 556 detector pixels incomplete; its timing is not an accuracy success. Retrograde search overflows by 19722 candidates. Zero detector failures in the charged or distant workload does not establish complete stellar images. The broader preset survey also records unresolved derivatives in the non-spinning views; a queue-capacity pass is not a domain-accuracy pass.

For a controlled comparison, the frozen solver from the start of this review received only the current disk field and its work-selection bound, plus the identical current benchmark harness. Its full-frame and cached-source means are 329.240 and 85.052 ms; current means are 265.060 and 69.173 ms. The comparator fingerprint is `34967734c657a148a31be9a8417495084fab2513a7e6749c0df142f4f574ec99`. Its setup and readbacks are retained in `test-results/bench/equal-source-baseline/`. Older 563.8/84.0 ms and 305.6/79.6 ms measurements belong to earlier code/source workloads; they are not the current matched comparator.

Both versions retain 8565 candidates, 1787 mixed parents, 4066 mixed children, zero overflow, identical detector destination tags, and identical sampled alpha. Failed refinements increase from **46 to 57**. RGB relative $L_1$ difference is $2.05\times10^{-7}$ over the whole image and $2.73\times10^{-5}$ over sky-center pixels. These small differences and passing retained-source gates do **not** prove unchanged stellar completeness; the lower time is reported without claiming a certified equal-quality speedup. Exact comparisons are in `test-results/bench/equal-source-comparison.json`.

Existing-device initialization averages 138.88 ms over five samples after one warmup, with 16.41% relative margin. It includes source preparation, shader/pipeline creation, uploads, completion, and disposal; caches are uncontrolled. This is not cold launch. The short bloom timing also has a wider margin than the full optical workloads.

All **190 GPU tests, 74 CPU tests, seven browser tests, five CPU preparation benchmarks, and nine GPU/startup benchmarks** passed in this run. Static checks, production build, repository formatting, and all 144 relative documentation links pass. These results leave the physical and visual requirements above open.
