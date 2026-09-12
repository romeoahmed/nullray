# Validation and measurement protocol

Checks establish bounded contracts. [Coverage](coverage.md) maps them to production behavior and unverified domains.

## Checks by change type

| Change                         | Required evidence                                                                              |
| ------------------------------ | ---------------------------------------------------------------------------------------------- |
| Scene/domain validation        | Relevant CPU behavior and properties; static checks                                            |
| Optical or numerical equations | Independent reference or analytic limit, invariants, explicit tested domain                    |
| WGSL or GPU ABI                | Actual browser GPU compilation and execution on the shared quantized inputs                    |
| Worker, canvas, interaction    | Browser behavior, revisions, retry/disposal and focus/lifetime checks                          |
| Image formation or display     | Quantitative output/missing-weight checks and visual inspection                                |
| Documentation only             | Formatting, relative links and anchors, formula consistency, commands and honest status labels |

Use [development commands](../engineering/development.md#setup-and-commands). Required GPU checks need native execution; report missing prerequisites and skipped checks.

## Small, complementary test layers

Prefer observable results. Use focused white-box probes for physical invariants, ABI packing and numerical boundaries, paired with production image/inspection checks. Formula mirrors establish implementation agreement, not independent physical truth.

Do not freeze incidental counts, allocation sizes, ordering or error text. Exact assertions suit catalogue provenance, energy signs, schema discriminants and specified storage limits. Do not replace meaningful references with “finite/nonzero” checks.

Use [fast-check](https://fast-check.dev/docs/core-blocks/runners/) on bounded domains and short state sequences. Construct valid coupled inputs; add explicit axis, horizon, signed-zero and invalid-boundary cases. Preserve shrinking, seed/path and concrete counterexamples for [replay](https://fast-check.dev/docs/tutorials/quick-start/read-test-reports/). GPU properties reuse resources but reset each case and read back completed sequences.

[Browser assertions](https://playwright.dev/docs/best-practices) use accessible roles and awaited outcomes. Check revision/lifetime behavior at the worker boundary; sleeps and private call counts do not prove GPU completion. Always clean up mounts, persistence changes, workers and devices.

[GPU support](../../tests/support/gpu.ts) owns devices, error scopes and readback. Device loss and unavailable hardware fail explicitly. `test:ui` covers interaction; `test:visual` produces manual-review captures. Passing capture completion does not approve pixels. Lint and Vitest CI policy reject focused tests.

## Reference independence

[hamiltonian.ts](../../tests/reference/hamiltonian.ts) independently differentiates the BL metric and evolves polarization through its connection. Its coordinate domain excludes singular BL boundaries; tighter tolerance cannot certify horizon crossings. Analytic principal/bifurcation families and reversible block histories provide complementary evidence.

[Polarization](../../tests/reference/polarization.ts), [sky](../../tests/reference/sky.ts), [structure](../../tests/reference/structure.ts) and [bloom](../../tests/reference/bloom.ts) references compare observable quantities. Shared formulas/constants limit their independence.

Compare the same quantized inputs using quantity-specific tolerances. Separate metric, trajectory, endpoint, spectral, material, pixel and display errors. Account for sensitivity rather than using a universal epsilon.

Include relevant $a=q=0$, signed-spin, charged, zero/negative-energy, horizon and axis limits. Do not exclude failures with unexplained clamps. Preserve minimized cases and replay provenance.

## Regression fixtures

The two [regression fixtures](../../tests/regressions/) retain wrong inner-horizon reflection and lost foreground emission after an empty material tail. Keep the original camera explicit rather than inheriting changing presets. Assert destinations and retained radiation.

A fixture file or reference-only test does not validate production unless its result is compared.

## Image and interface acceptance

Inspect shadow/source edges, orientation, continuity, missing regions, stars and highlights. Diagnose optical errors in linear radiance. Record dimensions, epoch, physical inputs, sample sequence, destinations and missing weight.

Run `pnpm test:visual`, then inspect warm/blue images and their JSON sidecars under `test-results/acceptance-*`. Review open panels, focus, long values and refinement at desktop/narrow sizes, plus jet, interior/naked and plasma views. Apply [visual priorities](../design/experience.md); static SDR images do not validate motion or physical HDR.

## Performance

Benchmark a specific comparison. Ignored schema-1 artifacts in `test-results/bench/` record executing hardware/browser, source/harness/dependency identities, inputs, raster, samples, epoch and coverage. Source hashing includes the binary catalogue; harness hashing includes shared GPU/readback support.

| Harness                                        | Measured scope                                                                                                                                  | Excluded scope                                                      |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| [CPU](../../benchmarks/cpu.bench.ts)           | Five production workloads: scene update, free-fall camera reuse, disk table, thermal table, stellar tree                                        | Metadata and output hashing                                         |
| [Optics](../../benchmarks/optics.bench.ts)     | Six reused 320×180, single-sample optical workloads; 1 warmup + 8 measurements; waits for submission completion                                 | Initialization, readback, bloom, presentation                       |
| [Startup](../../benchmarks/startup.bench.ts)   | Optical pipeline/source construction, uploads, GPU generation, completion and disposal on an existing device; 1 warmup + 5 measurements         | Device acquisition and a genuinely cold browser/driver launch       |
| [Renderer](../../benchmarks/renderer.bench.ts) | Warm/blue live, 16-sample settle, 64-sample photo and display-only sequences; configured resources, 1 warmup + 3 measurements; waits each frame | Initialization, worker messages, rAF pacing, readback, PNG encoding |

Renderer live/settle/display workloads use 160×90; photographs use 320×180. Different rasters or sample counts do not establish equal-quality speedup. Inspect recorded inputs rather than assuming benchmark names match UI presets.

Compare final linear image error and missing weight at comparable quality, including the whole affected pipeline. Separate initialization and steady state. Fewer rays or scalar radiance sums are not image-error measures.

### Timing interpretation

Current [Vitest benchmarks](https://vitest.dev/guide/benchmarking) measure wall-clock milliseconds. GPU callbacks wait for `queue.onSubmittedWorkDone()`, including encoding/submission/completion notification. Setup, readback inspection, hashing and report writes remain outside timing. Renderer sequence checks prevent cached or incomplete work appearing faster. Run GPU workloads serially.

Pass-level [timestamps](https://www.w3.org/TR/webgpu/#timestamp) require optional `timestamp-query` and `timestampWrites` on actual passes. Resolve unsigned 64-bit nanoseconds and subtract integers before unit conversion. Precision is reduced; counter resets invalidate negative deltas, and zero deltas do not mean zero work. Keep readback outside measured passes and retain end-to-end wall time. Current reports use `timestampQuery: false`.

Fixed GPU runs are smoke measurements, not confidence or tail-latency estimates. Preserve their raw samples; CPU artifacts retain summary statistics. Select artifacts by matching identities and time, since ignored output can contain obsolete experiments. Longer/interleaved runs need a concrete comparison. No current harness measures worker/rAF pacing or PNG latency.
