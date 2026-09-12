# Working on Nullray

Nullray is an unreleased TypeScript + native WebGPU/WGSL Kerr–Newman optical prototype. Start with [docs/README.md](docs/README.md) and [coverage](docs/validation/coverage.md). Before optical changes, read [spacetime conventions](docs/physics/spacetime.md), the relevant source/transport model and [numerics](docs/numerics/geodesics.md). For runtime work, read [architecture](docs/engineering/architecture.md).

## Commands

Use Node.js 26+ and pnpm with the existing lockfile. [package.json](package.json) owns scripts; [development](docs/engineering/development.md) explains configuration and browser prerequisites.

```sh
pnpm install --frozen-lockfile
pnpm dev
pnpm check
pnpm build
pnpm test
pnpm test:gpu
pnpm test:ui
pnpm test:visual
pnpm fmt:check
```

`build` includes static checks. `test` is CPU-only; GPU, UI and visual projects use real Chromium. Install a missing browser with `pnpm exec playwright install chromium`. Report unavailable adapters and skipped required checks. Mocks and empty runs do not validate GPU code.

For documentation-only edits, run `pnpm exec oxfmt --check README.md docs AGENTS.md NOTICE.md`; check links, anchors, formulas, commands and status. Choose other checks from [the validation matrix](docs/validation/methods.md#checks-by-change-type). Benchmark only for a concrete comparison.

## Implementation

- Keep validated inputs in `src/scene/`, pure calculations in `src/physics/`, GPU ownership in `src/gpu/`, worker scheduling in `src/runtime/`, and browser interaction in `src/ui/`.
- Prefer readonly records, explicit inputs, discriminated unions and exhaustive handling. Validate external data as `unknown`; casts and brands do not validate it. Keep loops and local mutation in numerical hot paths; document writes to caller-owned storage.
- Use native ESM and type-only imports under the existing compiler/environment boundaries. Edit unshipped schema-1 formats directly and remove obsolete branches.
- Route coupled physical changes and camera placement through scene validation. Navigation does not imply observer velocity; camera-only edits must reuse the free-fall endpoint.
- Give resources explicit owners and cleanup. Reject stale asynchronous results; preserve revisions and bounded worker submissions. Canvas transfer is permanent, so remount with a fresh element.
- Native pixels are the default. Honor fixed scales; only explicit Auto adapts playback. Raster policy must not change optical validity or tolerance.
- Write English declaration contracts and local algorithm reasoning using the [comment conventions](docs/engineering/development.md#comments-and-declaration-documentation). Keep WGSL explicit; Oxfmt/Oxlint do not validate it.

## Numerical invariants

- Preserve units, signature, Carter convention and future photon momentum with backward integration. Retain zero/negative Killing energy; do not normalize all data by it.
- Distinguish chart changes, block transitions, signed infinity, emission, singularity, proven source exclusion and unresolved work. Budget exhaustion or arbitrary NaNs must not become physical success.
- Use neutral Kerr–Newman emitters for charged geometry. Separate source prescriptions, frequency transport and display transforms; do not double-apply frequency or lensing amplification.
- Preserve ordered boundaries and missing sample weights. Settling and photographs trace direct rays. Reconstruction needs an end-to-end benefit at comparable image error without losing source domains or unfinished work.
- Guard divisions and roots before evaluation; WGSL `select` is not lazy. Formula rearrangements and reduced precision require numerical verification. Optical arithmetic remains f32.
- Request the shared GPU requirements. Subgroup code must assume neither fixed width nor lane/workgroup correspondence; follow [WGSL layout and arithmetic contracts](docs/engineering/shaders.md).

## Verification and delivery

Use independent references, shared quantized inputs and quantity-specific tolerances. WGSL/ABI changes need actual GPU execution; worker/UI changes need browser lifetime checks. Retain minimized failures and replay provenance. Complete coherent implementation before concentrated runtime and visual validation; stop unnecessary servers and watchers.

Preserve unrelated staged and working-tree changes. Avoid incidental dependency upgrades or formatting. Check [NOTICE](NOTICE.md) and licenses before adapting external material. Keep equations in their owning document and status in coverage; update affected links. Report changes, checks and remaining limits. Do not claim complete maximal extension, critical-image or nonlinear stellar-flux coverage without evidence.
