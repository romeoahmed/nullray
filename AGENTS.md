# Working on Nullray

## Project and status

Nullray is a TypeScript + native WebGPU/WGSL black-hole renderer. The intended core is a validated semi-analytic Kerr–Newman optical solver with a thin thermal disk, filtered HDR stars, and interactive exploration.

Start at [docs/README.md](docs/README.md). Read [physics](docs/physics.md) and [numerics](docs/numerics.md) before changing optical calculations; read [architecture](docs/architecture.md), [tooling](docs/tooling.md), and [validation](docs/validation.md) for implementation work.

The source is an optical prototype. `model/` and `physics/` own pure inputs and preparation; `render/` owns GPU execution in a dedicated worker; `explorer/` owns browser interaction. Binary64 trajectory references live in `tests/reference/`, outside production. Check the coverage ledger before making status claims.

## Commands

Use pnpm and the existing lockfile. Inspect `package.json` for current scripts and runtime requirements.

```sh
pnpm install --frozen-lockfile
pnpm dev
pnpm build
pnpm preview
```

The build runs `pnpm check` and Vite. `pnpm test` runs CPU tests, `pnpm test:gpu` executes WGSL, and `pnpm test:ui` checks browser/worker interaction. Benchmarks use `pnpm bench` and `pnpm bench:gpu`. For documentation-only changes:

```sh
pnpm exec oxfmt --check docs AGENTS.md
```

See [tooling](docs/tooling.md#commands) for command details. An installed dependency or a run with no tests is not validation.

If a required check cannot run, report the missing prerequisite and complete independent checks.

## Implementation conventions

- Prefer pure functions, readonly domain values, discriminated unions, explicit inputs, and exhaustive handling.
- Keep local mutation and direct loops in numerical hot paths. Clearly name operations that mutate caller-owned buffers.
- Use native ESM and type-only imports. Enable the documented strict compiler policy when implementing the toolchain.
- Validate untrusted data as `unknown`; casts and brands are not runtime validation.
- Keep coupled physical parameters valid through one construction/update path.
- Keep GPU handles, DOM access, clocks, randomness, and persistence outside the domain core.
- Route camera changes through scene validation; all optical samples use the same local camera basis. Navigation never implies physical observer velocity.
- Give GPU resources a clear owner and explicit cleanup. Reject stale asynchronous results after replacement/disposal.
- Request the shared required GPU features; photographic history needs texture formats tier 2 and WGSL read/write storage textures.
- Keep rendering on its dedicated worker; canvas transfer is permanent, so remount with a fresh element. Preserve input revisions and bounded scheduling.
- Keep WGSL explicit and close to reviewed formulas. Avoid an additional runtime framework or shader DSL without concrete benefit.
- Document declaration contracts with JSDoc blocks: units, domains, ownership, and failure meaning. Keep local algorithm reasoning in line comments; do not duplicate types.
- Use Oxfmt/Oxlint for their supported languages. Do not assume they validate or format WGSL.
- Keep changes scoped and preserve unrelated user work. Do not update dependencies or reformat unrelated files incidentally.

## Physical and numerical invariants

- Use the units, signature, Carter-constant convention, and backward-tracing orientation in [physics](docs/physics.md).
- Retain zero/negative Killing-energy cases; do not divide all photon data by energy unconditionally.
- Distinguish capture, escape, emission hits, invalid inputs, and unresolved numerical work.
- Never turn exhausted budgets or arbitrary NaNs into successful physical results.
- Do not hide branch errors by broad clamping, fixed epsilon injection, or unexplained parameter exclusions.
- Use the corresponding Kerr–Newman emitter model when charge is nonzero.
- Keep optical, spectral-transfer, and display assumptions separate. Do not double-apply frequency or lensing amplification.
- Apply the documented source-boundary and disk-surface conventions; order events along the backward path.
- Guard unsafe WGSL arithmetic before evaluation; do not rely on NaN propagation or `select` as a lazy guard.
- Treat formula rearrangements and reduced precision as numerical changes requiring verification.
- Use native subgroup operations without assuming lane/workgroup correspondence or a fixed subgroup size. Optical calculations remain f32.

## Verification

Choose checks appropriate to the change using [the validation matrix](docs/validation.md#checks-by-change-type).

- Numerical work needs independent references, relevant properties, and explicit supported domains.
- WGSL and GPU layout work needs actual browser GPU execution; mocks are insufficient.
- Compare CPU and GPU on the same quantized inputs and use quantity-specific tolerances.
- Save concrete minimized failures alongside seed/path and version metadata.
- Measure performance at comparable image quality and record hardware, sampling, error, and unresolved cases.
- For documentation-only edits, check formatting, relative links, formulas, commands, and status labels. Do not add artificial runtime tests.
- Report what changed, what was checked, and any remaining limitation. Never label an unrun or skipped required GPU check as passed.

## Documentation

Write documentation and code comments in English. Use descriptive Markdown headings, language-tagged fences, and relative repository links. Keep equations and decisions in their owning documents and link to them instead of duplicating long explanations.

Cite primary sources for physical equations and platform behavior. Preserve attribution and check licenses before adapting code or assets. Distinguish accepted design, experiments, implemented behavior, and measurements.

Update this guide and the relevant design documents when commands, physical contracts, supported domains, or architectural boundaries change. Keep this file operational and concise; detailed derivations belong in `docs/`.
