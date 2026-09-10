# Validation

Verification follows the physical or user-visible contract. Tests should survive a better internal algorithm. Avoid assertions that freeze queue layouts, tuning values, shader decomposition or an abandoned implementation.

## Checks by change type

| Change                         | Required evidence                                                           |
| ------------------------------ | --------------------------------------------------------------------------- |
| Domain or scene validation     | Relevant CPU behavior/property checks and static checks                     |
| Optical or numerical equations | Independent reference or analytic limit, invariants, explicit tested domain |
| WGSL or GPU layout             | Actual browser GPU execution on quantized inputs                            |
| Worker, canvas or interaction  | Browser behavior and lifetime checks                                        |
| Image formation or display     | Rendered output, missing-weight behavior and visual inspection              |
| Documentation only             | Formatting, relative links, equations, commands and status accuracy         |

## Physical comparisons

The binary64 [metric-Hamiltonian reference](../tests/reference/hamiltonian.ts) expands and differentiates the BL metric independently of the production separated flow. Ordinary exterior segments can be compared there; horizon crossings need regular-coordinate invariants or independent analytic families. Proper-time normalization, reversible block history and principal null rays provide complementary checks.

Compare the same quantized initial data. Choose tolerances for the observable being compared, and distinguish input sensitivity from integration error. A normalized residual does not establish angular image accuracy. A successful image does not establish completeness of unresolved rays or missing stellar images.

Small semantic cases are preferable to exhaustive assertions about private stages. Retain minimized failures that still exercise the current physical contract, including complete inputs, random seed/shrink path when applicable, and reference provenance. The two deterministic image regressions in `tests/regressions/` encode discovered failures; they do not set performance gates. Catalogue counts and physical constants describe source data, not an internal algorithm. Reference-only code is not production validation unless the production calculation is actually compared with it.

## GPU and images

`pnpm test:gpu` compiles and executes native WGSL. Mocks, shader parsing and installed dependencies are not substitutes. Missing browser or GPU prerequisites are reported separately from passing checks.

Inspect the rendered result for continuity, source boundaries, image orientation, unresolved regions and display behavior. Use physically meaningful scenes, including limiting cases. Compare radiance before display transforms when diagnosing optical differences. Exposure cannot fix missing flux or a wrong ray branch.

[Coverage](coverage.md) records remaining critical-image, derivative and stellar-flux validation. Local kernel agreement does not certify complete images near caustics or source-domain boundaries.

## Performance

Benchmarks are comparative measurements. A simpler or more robust algorithm may justify a measured slowdown. Record hardware, source and harness fingerprints, dimensions, samples and unresolved output. Keep initialization, rendering and readback scopes clear. Compare images or quantities at comparable quality before claiming a speedup.

During a coherent implementation, use source review and static checks. Run the relevant complete suites and browser visual review together after development. Fix failures found in that verification phase and rerun affected checks. Benchmarks are optional comparative investigations, never a routine loop or speed gate. Commands and platform requirements are in [Tooling](tooling.md).
