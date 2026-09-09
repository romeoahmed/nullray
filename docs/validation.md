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

## Executable suites

| Project   | Scope                                                                                                                                           |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `cpu`     | Scene/view/session validation, camera geometry, circular emitters, thermal profiles, special functions, trajectory references, properties       |
| `gpu`     | Actual WGSL, buffer layouts, endpoint order, frequency transfer, local derivatives, source filtering, accumulation, coverage, presentation, PNG |
| `browser` | Native controls, mobile layout, motion cleanup, worker/remount behavior, snapshots, local persistence, photographic completion                  |

[Vitest projects](https://vitest.dev/guide/projects) share one runner. Browser Mode uses [Playwright](https://playwright.dev/docs/test-assertions) and real GPU execution. Mocks cannot establish shader validity, canvas ownership, or physical results. GPU files run serially. Resources are disposed even on failure; assertions do not depend on incidental source text or private scheduling counts.

Use accessible controls and visible outcomes for interaction tests. A small real image can validate photography state, while separate native-size inspection assesses appearance. Tests must not force expensive whole-frame refinement merely to check a label. Visual review covers desktop/mobile panels, focus, occlusion, legibility over bright imagery, pointer/touch interaction, and diagnostic visibility.

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

Record hardware, browser, source/data hashes, scene, resolution, sample count, error/unresolved coverage, and timed scope. Retain image readbacks for comparisons. Warm caches and cold launch are different workloads. GPU timestamps exclude CPU submission and may reset or quantize; reject invalid deltas explicitly. A software adapter can validate behavior but does not establish hardware performance.

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
