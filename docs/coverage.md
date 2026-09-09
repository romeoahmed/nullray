# Coverage and current limits

Nullray is an optical prototype. The current checks establish particular mathematical, shader, and interaction behaviors; they do not establish a complete support domain. Known counterexamples remain recorded even when the regression suite passes.

## Executable evidence

| Family                         | Evidence                                                                                                                                                                          | Scope and limit                                                                                                                         |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Ordinary exterior trajectories | [CPU geodesics](../tests/physics/geodesic.test.ts)                                                                                                                                | Generated short segments and explicit limits compared with independent Hamiltonian integration; not full endpoint coverage              |
| Circular disk                  | [Geometry](../tests/physics/geometry.test.ts), [disk profile](../tests/physics/disk.test.ts)                                                                                      | Neutral charged-orbit normalization/ISCO, five signed Kerr flux references, charged grid convergence                                    |
| Equator timing                 | [CPU](../tests/physics/equator.test.ts), [GPU](../tests/gpu/equator.test.ts)                                                                                                      | Periodic transverse crossings and quantized ordinary cases; not all separatrices                                                        |
| Radial arrival                 | [CPU](../tests/physics/arrival.test.ts), [GPU](../tests/gpu/arrival.test.ts)                                                                                                      | Direct inverse phase versus scanned reference, including zero/negative-energy horizon paths                                             |
| Opaque event order             | [Event order](../tests/gpu/event-order.test.ts), [visibility](../tests/gpu/visibility.test.ts)                                                                                    | Frozen-scan image comparisons, independently checked higher-order and outer-disk cases, Schwarzschild critical impact and frequency law |
| Azimuth and delay              | [CPU](../tests/physics/transport.test.ts), [GPU](../tests/gpu/transport.test.ts)                                                                                                  | Ordinary/axis paths, Schwarzschild deflection, prescribed critical-emission phase criterion                                             |
| Axis camera limits             | [Axis](../tests/physics/axis.test.ts), [GPU camera](../tests/gpu/camera.test.ts)                                                                                                  | Regular-coordinate limits, meridian/orientation plumbing; not complete nonpositive-C axis transport                                     |
| Elliptic conditioning          | [GPU elliptic](../tests/gpu/elliptic.test.ts), [differentiation](../tests/gpu/differential.test.ts)                                                                               | Elementary limits, quantized Jacobi cases, analytic derivative identities; not arbitrary long-phase accuracy                            |
| Local sky mapping              | [Sky Jacobian](../tests/gpu/sky-jacobian.test.ts), [beam repair](../tests/gpu/beam-repair.test.ts)                                                                                | Same quantized camera inputs, converged finer binary64 differences, branch/coordinate checks                                            |
| Stellar and diffuse flux       | [Stars](../tests/gpu/stars.test.ts), [sky](../tests/gpu/sky.test.ts), [CPU sky](../tests/physics/sky.test.ts)                                                                     | Motion/shear/parity, spectral shifts, cube seams, mip flux, catalogue integrity; local affine flux is not caustic integration           |
| Disk structure and filtering   | [Emission clock](../tests/gpu/emission-clock.test.ts), [edge sampling](../tests/gpu/edge-sampling.test.ts)                                                                        | Retarded/cohort continuity, fixed-quadrature comparisons, disk-interior sampling, failure weights                                       |
| Bounded sampling               | [Presets](../tests/gpu/presets.test.ts), [coverage](../tests/gpu/coverage.test.ts)                                                                                                | Four preset queue budgets at 2560 × 1440; integer reduction across partial workgroups and all 64 failure weights                        |
| Output                         | [Accumulation](../tests/gpu/accumulation.test.ts), [bloom](../tests/gpu/bloom.test.ts), [presentation](../tests/gpu/presentation.test.ts), [PNG](../tests/gpu/photograph.test.ts) | History, normalized filtering, color/exposure, padded rows, orientation and submitted snapshot behavior                                 |
| Resource lifetime              | [GPU lifecycle](../tests/gpu/lifecycle.test.ts), [worker integration](../tests/browser/worker.test.ts)                                                                            | Aborted initialization, context reuse, disposal, and worker replacement on actual WebGPU                                                |
| User intent                    | [Session](../tests/model/session.test.ts), [view](../tests/model/view.test.ts), [browser](../tests/browser/app.test.ts), [bookmarks](../tests/browser/views.test.ts)              | Atomic rejection, native controls, mobile geometry, frozen restoration, real worker/remount, photographic completion, persistence/undo  |

Test files own their exact tolerances and generator distributions. Broadly, ordinary GPU angular transport uses a 5e−5 absolute criterion, while time, inverse radius, source flux, and Jacobians have separate checks. Do not replace them with one universal epsilon. The old shader's bytes are a comparison record, not an independent physical oracle.

## Retained counterexamples

| Fixture                                                               | What it preserves                                                               |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| [critical-primary-rays](../tests/fixtures/critical-primary-rays.json) | Known critical f32 endpoint errors, reference settings, and measured versions   |
| [critical-event-order](../tests/fixtures/critical-event-order.json)   | Higher-order event disagreement checked against an independent null Hamiltonian |
| [outer-disk-arrival](../tests/fixtures/outer-disk-arrival.json)       | Inverse-radius conditioning at a distant disk endpoint                          |
| [nearly-radial-arrival](../tests/fixtures/nearly-radial-arrival.json) | Near-radial boundary conditioning                                               |
| [jacobi-dn](../tests/fixtures/jacobi-dn.json)                         | Cancellation near the Jacobi limiting modulus                                   |
| [launch-roundoff](../tests/fixtures/launch-roundoff.json)             | Critical launch sensitivity to one-ULP Carter changes after struct extraction   |
| [camera-roundoff](../tests/fixtures/camera-roundoff.json)             | Quantized camera-expression sensitivity and affected pixels                     |
| [boundary-beams](../tests/fixtures/boundary-beams.json)               | Thin-branch sky derivative cases                                                |
| [radial-scan](../tests/fixtures/radial-scan.wgsl)                     | Frozen validation-only comparator for direct radial event inversion             |

Historical source paths inside fixture metadata name the measured version and are deliberately not rewritten as current file paths. The active code and executable tests use the current module layout.

The critical-primary corpus includes a sky direction-vector error of about **0.00575**, well above ordinary angular acceptance. Independent Hamiltonian comparisons validate finite reference segments, not direct integration to infinity or the GPU endpoint. No change in this cleanup claims to close that error.

Exact Schwarzschild spherical launch remains explicitly unresolved. Wider quadrature budgets do not by themselves correct nearly repeated-root path error. Nonpositive-C axis transport, complete near-horizon/extremal coverage, event ties, and nonlinear stellar multiplicity remain open.

## Image acceptance

The renderer retains first-hit opacity, image-order diagnostics, neutral-emitter Doppler/gravitational transfer, source spectra, and missing sample weights. Exploration uses selective 4 × 4 sampling; photography uses 64 jittered samples. The stellar filter still assumes a locally affine map. A source image can be missed between sampled rays, and refinement can average a biased endpoint.

Desktop and mobile browser checks cover the native-popover interface, readable settings, responsive canvas dimensions, exposure/source control behavior, paused view restoration, and photographic state. Visual checks complement the numerical corpus; they do not certify all photon-ring orders or moving stellar-image convergence.

## Measurements

Production benchmark artifacts are generated by `pnpm bench` and `pnpm bench:gpu` under `test-results/bench/`. Each run records its own hardware/runtime, source/data fingerprints, sampling, timed scope, queue use, unresolved coverage, and image readbacks. CPU work measures actual scene/disk/sky/thermal preparation. GPU timing separates complete optics, retained-geometry radiation, bloom, and initialization.

### September 9, 2026 measurement

Apple M5, Metal-3, Chromium 153; GPU run started at 19:22 China Standard Time. The shader fingerprint is `63e1e966108315e08308e0e17de44e6f1de4a53d3a26ea2cb414d02db50d9a31`. These are absolute measurements after the resource-lifetime cleanup, with serial GPU workloads and no task-owned preview running; other system load was not controlled. No cleanup speedup is inferred from this snapshot.

| Workload                       | Mean       | Relative margin of error | Samples | Pixels with missing coverage |
| ------------------------------ | ---------- | ------------------------ | ------- | ---------------------------- |
| Complete optics, 128 × 72      | 38.145 ms  | ±1.89%                   | 64      | 2                            |
| Retrograde optics, 128 × 72    | 54.304 ms  | ±1.03%                   | 64      | 4                            |
| High-charge optics, 128 × 72   | 41.956 ms  | ±1.50%                   | 64      | 1                            |
| Complete optics, 640 × 360     | 79.762 ms  | ±0.62%                   | 64      | 20                           |
| Complete optics, 1280 × 720    | 110.480 ms | ±1.26%                   | 64      | 64                           |
| Cached radiation, 1280 × 720   | 6.760 ms   | ±0.67%                   | 148     | 64                           |
| Bloom, 1280 × 720              | 0.411 ms   | ±1.82%                   | 2,433   | 64                           |
| Existing-device initialization | 58.420 ms  | ±9.16%                   | 5       | —                            |

Spinning workloads use $a=0.7,q=0.2$, observer radius 18, inclination 1.2, and a field of view of 2 atan(0.6) radians; sampling is one primary ray plus production selective refinement. Retrograde uses $a=-0.7,q=0.2$; high-charge uses $a=0.2,q=0.9$, at the same camera and field of view. Each optical run has at least 64 samples and 1000 ms of measured work after at least 16 warmup samples and 250 ms. Initialization uses one warmup and five wall-clock samples on an existing device. No invalid timestamps were discarded and no boundary queue overflowed.

The 128 × 72 spinning, retrograde, high-charge, 360p and 720p images retain missing weights of 0.125, 0.3125, 0.0625, 1.5625 and 4.8125 full-pixel equivalents respectively. Two 720p primary endpoints remain unresolved at disk transport stage 8; all missing coverage stays visible. Cached-radiation and bloom rows report their retained input image's coverage. Bloom's exact filtered output is saved separately. These counts describe work completion, not an independent image-error bound.

CPU preparation ran on the same M5 with Node.js 26.8.1. The measured means were scene update 0.303 µs, disk profile 30.855 µs, star-tree preparation 23.847 ms, diffuse cube 27.350 ms and thermal table 5.864 ms. Relative margins of error were 0.41%, 0.28%, 1.74%, 2.23%, 0.48%. This CPU run overlapped browser interaction validation, so it is not an isolated performance comparison. Each workload consumes the prepared data, with serialization/hash work outside the timed callback.

Requested and actually enabled GPU features are recorded separately. Exact radiance/endpoint readbacks, filtered bloom data, sample distributions, source/data fingerprints and full context remain in the generated benchmark directory. No whole-frame performance guarantee or closed critical-ray accuracy claim follows from this snapshot.
