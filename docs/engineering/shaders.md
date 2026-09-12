# WGSL implementation

[WebGPU](https://www.w3.org/TR/webgpu/) defines resource usage and dispatch; [WGSL](https://www.w3.org/TR/WGSL/) defines shader arithmetic, layout and synchronization. Physical equations remain in [physics](../physics/spacetime.md) and [numerics](../numerics/geodesics.md).

## Module assembly

TypeScript owners concatenate WGSL fragments into modules. Shared declarations become ordinary module-scope declarations; `enable` and `requires` directives precede them. An isolated fragment may reference symbols supplied by its owner.

| Directory under `src/gpu/wgsl/` | Responsibility                                                                                                                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `geodesics/`                    | Metric/WP contractions, circular-emitter normalization, separated/canonical flow, charts, dense output, events and conservative source exclusion |
| `imaging/`                      | Frame ABI, lookup tables, matter structure/transfer and shared fullscreen triangle                                                               |
| `sources/`                      | Noise lattice, sky generation/mips and stellar-tree traversal                                                                                    |
| `passes/`                       | Image/inspection, celestial composition, history, coverage, polarimetry, bloom and presentation                                                  |

Production uploads the host observer frame. Image and inspection share detector-ray initialization. GPU probes use the production helpers and the same quantized inputs; do not maintain alternate implementations solely for tests.

## Parallel execution and memory

Optical rays use independent 8×8 invocations with local adaptive state. Inspection uses one invocation for an ordered trajectory. Pass boundaries make completed neighboring endpoints available to celestial filtering.

Coverage uses subgroup reduction/election, workgroup atomics and a barrier. Edge lanes contribute zero and reach every collective. Assume neither subgroup width nor lane/workgroup correspondence. [Workgroup variables are zero-initialized](https://www.w3.org/TR/WGSL/#var-decls); no extra clearing loop is needed.

| Host-visible data          | Layout                                                                              |
| -------------------------- | ----------------------------------------------------------------------------------- |
| Optical frame              | 288 bytes; 16-byte vector slots, column-major `mat4x4f`, aliased i32 block identity |
| Catalogue node             | 32 bytes; two `vec4f` records with stackless escape indices                         |
| Inspection                 | 16-byte header, then 32-byte points                                                 |
| Spectral/atmosphere tables | `vec4f` storage records; disk profile uses scalar storage stride                    |
| Photographic history       | Separate rgba32float I/Q/U means; rgba16float output; integer domain metadata       |

Follow [alignment and size](https://www.w3.org/TR/WGSL/#alignment-and-size) and [address-space constraints](https://www.w3.org/TR/WGSL/#address-space-layout-constraints). `vec3f` has size 12 and alignment 16. Local orbit structs are not uploaded ABIs. Dynamic uniform-offset alignment does not determine structure-member padding.

## Arithmetic and features

Optics remain f32. Hardware filtering serves the material lattice, celestial mips and bloom; explicit f32 sky interpolation limits rounding differences amplified by exponential dust. Half-float storage has separate [saturation limits](../physics/transport.md#storage-and-missing-light).

- `extractBits` reads raw exponent fields. `frexp` is unsuitable for subnormal/nonfinite guards because those results are indeterminate.
- Guard divisions and roots with branches; `select` evaluates both value arguments.
- WGSL [`fma`](https://www.w3.org/TR/WGSL/#fma-builtin) does not guarantee single rounding. Preserve the stable stellar determinant, factored horizons and outward interval bounds unless numerical checks justify a change.
- Cubic `smoothstep` cannot replace the lattice's quintic fade while preserving second-derivative continuity.

[device.ts](../../src/gpu/device.ts) owns required device features and language checks. Device features must be requested; language features are queried through `navigator.gpu.wgslLanguageFeatures`. Optional f16 arithmetic, pointer extensions or subgroup size control need a concrete use and validation. Availability alone is not a reason to add a requirement.

## Formatting

No formatter is configured. Oxfmt/Oxlint do not support WGSL. The consulted W3C specifications do not designate a formatter.

[wgsl-analyzer](https://github.com/wgsl-analyzer/wgsl-analyzer) is a community editor-tooling candidate. Its Node/WASM derivative [@wasm-fmt/wgslfmt](https://github.com/wasm-fmt/wgslfmt) 0.1.0 failed a local idempotence check: repeatedly formatting `fn f() { var x = 1; x += 1; }` grew spaces before `+=`. That version was not adopted; this result does not establish the behavior of newer analyzer releases. [Tint's WGSL writer](https://dawn.googlesource.com/dawn/+/refs/heads/main/src/tint/lang/wgsl/writer/) generates compiler output rather than preserving authored source/comments.

Before adopting a formatter, check idempotence, comments/directives, operator grouping, extension syntax and assembled modules. Compile and execute affected GPU paths. Formatting alone is not numerical validation, and no shader DSL is needed solely for formatting.
