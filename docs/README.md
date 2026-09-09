# Nullray

Nullray is a TypeScript and native WebGPU/WGSL black-hole renderer. It combines semi-analytic Kerr–Newman light paths, a neutral thermal thin disk, spectral stars and diffuse sky, and interactive camera and photographic controls.

**Current status: optical prototype.** The application renders on a dedicated worker, supports orbit/free exploration, source and display controls, native-resolution photography, presets, view links, and local bookmarks. CPU references and real browser GPU tests exercise the model. Critical primary-ray precision and complete nonlinear stellar-image filtering remain unresolved; a passing suite is not a released support-domain claim.

## Read the model

| Document                        | Responsibility                                                                             |
| ------------------------------- | ------------------------------------------------------------------------------------------ |
| [Physics](physics.md)           | Units, metric, observer, geodesics, source boundaries, emission, spectral transfer         |
| [Numerics](numerics.md)         | Elliptic paths, ordered events, quadrature, derivatives, pixel integration, failure policy |
| [Architecture](architecture.md) | Functional model, render worker, ownership, invalidation, interaction                      |
| [Tooling](tooling.md)           | Runtime baseline, configuration, commands, benchmarks                                      |
| [Validation](validation.md)     | Independent evidence, test design, acceptance, performance interpretation                  |
| [Coverage](coverage.md)         | Current executable evidence, concrete gaps, measured workloads                             |
| [Agent guide](../AGENTS.md)     | Operational instructions for repository work                                               |
| [Attribution](../NOTICE.md)     | Data sources, licenses, and representation changes                                         |

Read physics and numerics before changing an optical calculation. Read architecture, tooling, and validation before implementation work.

## Run

Use Node.js 26 or newer and pnpm:

```sh
pnpm install --frozen-lockfile
pnpm dev
```

The browser baseline is intentionally modern: WebGPU with subgroups and texture formats tier 2, OffscreenCanvas in a dedicated worker, Display P3, native popovers, anchor positioning, subgrid, and OKLCH. There is no legacy renderer or polyfill stack. See [capability requirements and commands](tooling.md).

## Direction

The intended experience is a readable, responsive observatory: convincing thermal emission, stable fine stars, resolved lensed disk images, and direct exploration. Physical observer motion remains separate from navigation. Disk fluctuations and brightness are prescribed appearance choices, not GRMHD predictions.

The next numerical acceptance work is concentrated in primary-ray conditioning near critical roots, nonlinear stellar image enumeration/integration, and source-aware convergence over the combined camera/spacetime domain. Performance improvements must retain image error and unresolved coverage evidence. More samples or a plausible screenshot alone do not close these requirements.
