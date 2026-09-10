# Validation

A plausible black-hole image is not evidence of correct branch selection, frequency transfer, or pixel integration. Validation separates mathematical identities, independent reference paths, shader execution, and visible interaction. The [coverage ledger](coverage.md) states what the current evidence does and does not establish.

## Independent references

Use known limits and symmetries, the binary64 separated solver in `tests/reference/`, and independent metric-Hamiltonian integration. The latter evolves

$$
H=\tfrac12 g^{\mu\nu}p_\mu p_\nu,\qquad
\frac{dx^\mu}{d\lambda}=\frac{\partial H}{\partial p_\mu},\qquad
\frac{dp_\mu}{d\lambda}=-\frac{\partial H}{\partial x^\mu}.
$$

Integrate backward using the same future-directed initial covector. Refine physical events, not only integration steps; an accepted step must not skip an opaque disk crossing. Check endpoint coordinates, frequency, and event order as well as the null residual. Tighten reference tolerances and demonstrate convergence before treating a result as an oracle. A translation of the production elliptic formulas is useful but not independent.

Compare CPU and GPU on identical quantized inputs. Separate camera/input rounding from optical solver error. Use higher-precision fixtures for difficult special functions and critical rays; record their precision and provenance. Preserve concrete counterexamples even when the current implementation cannot meet the target.

Reference functions require explicit frames; fixture construction belongs to `tests/support/`. The separated binary64 reference uses adaptive Simpson transport, distinct from production Gaussian quadrature, and metric-Hamiltonian checks supply a different trajectory method. The charged stellar-root reference accepts a coordinate chart from its caller; the production critical chart is a parameterization, independently checked elsewhere, rather than an independent orbit oracle. Fixtures use schema 1 and retain concrete inputs and provenance; regenerated endpoint checks do not accept historical GPU outputs as physical truth.

## Executable suites

| Project   | Scope                                                                                                                                           |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `cpu`     | Scene/view/session validation, camera geometry, circular emitters, thermal profiles, special functions, trajectory references, properties       |
| `gpu`     | Actual WGSL, buffer layouts, endpoint order, frequency transfer, local derivatives, source filtering, accumulation, coverage, presentation, PNG |
| `browser` | Native controls, mobile layout, motion cleanup, worker/remount behavior, snapshots, local persistence, photographic completion                  |

`pnpm test:gpu` includes the [critical primary endpoint gate](../tests/gpu/critical-rays.test.ts). It uses full production frames and texture readback because adding diagnostic outputs to a copied shader can change binary32 rounding. It requires sky unit-vector error below $5\times10^{-5}$, disk azimuth error below $5\times10^{-5}$ radians, and inverse-radius error below $2\times10^{-6}+2\times10^{-5}|1/r_{\mathrm{ref}}|$. These targets are independent of the measured counterexample errors. The retained corpus now passes through the mixed-precision production path; assertions are neither skipped nor marked as expected failures. Run it for optical precision changes and before any release claim. It remains an ordinary failing assertion when its target is missed; grouping it with related GPU cases prevents an optional acceptance command from being forgotten.

[Vitest projects](https://vitest.dev/guide/projects) share one runner. Browser Mode uses [Playwright](https://playwright.dev/docs/test-assertions) and real GPU execution. Mocks cannot establish shader validity, canvas ownership, or physical results. GPU files run serially. Tests are grouped by behavior—ray events, differentials, lensing, filtering, sampling, and rendering—rather than one file per implementation module. Shared GPU fixtures use Vitest’s context builder and cleanup callbacks. Resources are disposed even on failure; assertions do not depend on incidental source text or private scheduling counts.

Following the [Vitest test-review guidance](https://main.vitest.dev/guide/learn/writing-tests-with-ai), assertions check behavior and independent outcomes rather than mocks or private call sequences. Use accessible controls and visible outcomes for interaction tests. A small real image can validate photography state, while separate native-size inspection assesses appearance. Tests must not force expensive whole-frame refinement merely to check a label. Visual review covers desktop/mobile panels, focus, occlusion, legibility over bright imagery, pointer/touch interaction, and diagnostic visibility.

The GPU suite also runs [critical derivative checks](../tests/gpu/ray-differentials.test.ts) against binary64 central differences at steps $2^{-19}$ and $2^{-18}$ physical pixels. Reference gradients must agree within $10^{-5}$ relative error; Cartesian gradients and the solid-angle Jacobian must each agree within 0.002. It executes the production selection, cooperative transport and restored Cartesian beam functions. The [binary64 preparation gate](../tests/gpu/ray-differentials.test.ts) also checks derivatives before final screen-basis restoration. Both gates pass ten retained sky samples, including fractional boundary samples. They use independently oriented binary64 directional differences for area, with an explicit $10^{-5}$ relative area-convergence requirement; convergence of nearly parallel screen gradients alone is insufficient. Endpoint acceptance passes independently. These finite checks do not certify the entire precision-selection band or complete nonlinear stellar filtering.

The [boundary precision gate](../tests/gpu/critical-rays.test.ts) executes the production boundary entry point for 944 subpixels from 59 previously incomplete pixels. It checks every destination, sky unit vector, disk inverse radius, azimuth, and propagation time against converged binary64 references. Sky and azimuth targets remain $5\times10^{-5}$; relative disk-time error must be below $5\times10^{-5}$. The event-order suite additionally samples four full viewport scenes against a converged binary64 endpoint oracle, checking destination, disk order, sky direction, radius, azimuth, frequency, and delay for every selected ray. It has no frozen GPU implementation as a comparator.

The [Schwarzschild stellar-image fixture](../tests/fixtures/schwarzschild-stellar-images.json) uses an independent planar cubic orbit with Legendre integrals and an implicit image Jacobian. Its [generator](../tests/reference/schwarzschild-images.py) requires Python and mpmath 1.3.0 only for regeneration: run `python tests/reference/schwarzschild-images.py 100 > images.json`. All six image records, excluding residual strings, round identically in separate 80- and 100-digit runs. The [CPU checks](../tests/cpu/lensing.test.ts) compare sky directions, first opaque disk hits, energy, and refined radial differences against this oracle. The [GPU checks](../tests/gpu/stellar-lensing.test.ts) use integer pixel bases plus binary32 fractional jitter and compare the selected production ray functions against CPU references on those exact inputs. These checks validate known rays; they do not enumerate images or integrate their flux. The separate [production flux gate](../tests/gpu/stellar-lensing.test.ts) injects prepared celestial data through `createOptics`, renders the full frame through primary, boundary, and jittered sampling, and subtracts an opaque-disk-only baseline. It integrates a detector window around each visible pair and requires relative luminance error below 0.002 and fully resolved output in both renders. The target comes from the independent image positions, Jacobians, detector tent, and shifted source spectrum. Rotated-camera references independently invert the quantized camera basis and transform the oracle Jacobian by the ratio of observer solid-angle densities; the renderer receives no reference image positions. The [search-level check](../tests/gpu/stellar-lensing.test.ts) also checks each of the four visible images separately in three camera frames, using the same independent reference transformation and 0.002 relative flux limit. This individual gate prevents brighter images from hiding losses in the faint pair. These primary, boundary, jitter, and rotated-camera variants establish the retained source cases, not general source-aware convergence. See [measured image acceptance](coverage.md#image-acceptance).

The [Kerr–Newman image corpus](../tests/fixtures/critical-stellar-images.json) retains twenty-five roots across deep, near-polar, sub-spacing, stalled, and mixed-cell families. Its [reference](../tests/reference/stellar-image.ts) solves the source equation with binary64 orbit integration and three levels of parameter differences. Production search receives the quantized source and frame, and must recover each retained image within 0.001 pixels and 0.002 relative flux. Reference source residual and area-refinement error must independently be below $10^{-7}$ and $10^{-4}$. Matching uses image position as well as branch label; GPU results never define the target root or Jacobian.

The [photographic survey](../tests/gpu/rendering.test.ts) traces the full 1388 × 1674, 64-sample sequence from an [explicit scene](../tests/fixtures/photographic-boundaries.json), checks the final coverage reduction against every missing per-frame weight, and writes `test-results/photo-audit.json`. The report includes the quantized frame, jitter, source fingerprint, browser, failures, and work counters. This checks truthful coverage accounting; it deliberately does not call remaining optical failures successful endpoints. Its minimized boundary observations are open counterexamples, separately identified in the coverage ledger.

## Properties and difficult families

[fast-check](https://fast-check.dev/docs/core-blocks/arbitraries/) supplies generators and shrinking. Generate valid coupled spin/charge and local photon directions by construction; unconstrained random constants often describe impossible rays. Use finite binary32 inputs for GPU comparisons and binary64 inputs for reference research.

| Property                        | Essential condition                                                                       |
| ------------------------------- | ----------------------------------------------------------------------------------------- |
| Charge-sign symmetry            | Vacuum geometry and neutral emitters                                                      |
| Momentum scaling                | Positive scale; scale Carter constant quadratically and parameters consistently           |
| Schwarzschild rotation symmetry | Rotate observer and physical directions together                                          |
| Spin reflection                 | Transform azimuth, momentum, observer, and orbit orientation together                     |
| Root residual                   | Evaluate the original scaled polynomial; repeated roots need position/branch evidence too |
| Null constraint                 | Scale-aware metric contraction; not a substitute for event accuracy                       |
| Thermal transfer                | Shift the spectrum once; preserve source normalization                                    |
| Stellar flux                    | Motion, shear, parity, seams, and multiplicity; no second magnification factor            |
| View restoration                | Full atomic validation; rejected values preserve the existing state                       |

Deliberately sample near-horizon, near-extremal, zero/negative-energy, near-coplanar, polar-axis, turning-point, and nearly repeated-root families. Uniform random pixels miss narrow critical structures. Preserve a difficult family's defining constraints when shrinking. Do not fuzz nonfinite GPU arithmetic as a portable NaN detector; validate that input at the CPU boundary.

Retained fixtures include scene/camera or photon constants, exact sample position, seed/path where applicable, source/reference versions, expected uncertainty, and observed failure. Keep executable minimized failures. Historical timing dumps and abandoned implementations do not belong in the active fixture corpus.

## Pixel and image acceptance

Check image order and source destination before comparing coordinates or colors. Compare sky endpoints as unit vectors across seams and poles. Frequency uses endpoint contractions. Disk radius, azimuth, and delay require separate tolerances; the coordinate time to infinity is never a sky observable.

An output pixel estimates a finite footprint. Compare successive sampling levels and an independent reference where practical. A passing quadrature difference estimate is not a global error bound. A resolved center ray does not certify a caustic-crossing stellar image, and 64 averaged samples do not repair a biased endpoint.

Report unresolved pixels and unresolved sample weights separately. Never remove failed samples from the normalization or paint them as a successful physical result. Check queue overflow, partial workgroups, half-float range failures, and branch changes. Fixed capacity and iteration limits are work policies, not new physical exclusions.

## Performance

First reduce mathematical work: exact limiting cases, shared path preparation, analytic event ordering, and source-aware sampling. Then examine memory layout, dispatch organization, resource reuse, and UI isolation. Relaxing tolerances or rendering fewer pixels is a quality change, not an equal-quality solver improvement.

Record hardware, browser, production/data and harness hashes, scene, resolution, sample count, error/unresolved coverage, and timed scope. Retain image readbacks for comparisons. The optical benchmark records stellar work counters for every submitted frame, including warmup and rejected timestamp samples, and rejects variation on identical inputs. A final-frame snapshot alone can miss intermittent candidate loss. The fresh-device stellar repeatability regression compares seed/result pairs in canonical allocation order, including forced queue overflow and a partial scan block. Warm caches and cold launch are different workloads. GPU timestamps exclude CPU submission and may reset or quantize; reject invalid deltas explicitly. A software adapter can validate behavior but does not establish hardware performance.

Do not convert one benchmark into a general frame-rate claim. Strongly lensed regions and different camera placements can have very different cost. The active [benchmark commands](tooling.md#commands) measure actual production workloads.

## Checks by change type

| Change                                   | Required checks                                                                               |
| ---------------------------------------- | --------------------------------------------------------------------------------------------- |
| Optical formulas, precision, event logic | Independent reference/limit tests, relevant properties, real GPU comparisons, coverage record |
| WGSL or GPU layout                       | Real browser compilation/execution, output/layout checks, relevant references                 |
| Source spectra or disk emission          | Spectral/orbit/profile references, transfer tests, GPU image checks                           |
| Sampling or performance                  | Equal-quality readbacks, work/coverage counts, relevant image convergence, GPU timing         |
| Worker/resources                         | Browser lifecycle and state behavior, disposal/error paths, build                             |
| UI/navigation                            | Browser interaction, camera/scene invariants, desktop/mobile visual inspection                |
| Tool configuration                       | Static checks and actual affected commands                                                    |
| Documentation only                       | Formatting, relative links/anchors, equations, commands, status labels                        |

Run relevant checks once after the final change. Broaden or repeat only when new edits, failures, or unresolved concerns justify it. Report missing prerequisites; never label an unrun or empty GPU job as passed.
