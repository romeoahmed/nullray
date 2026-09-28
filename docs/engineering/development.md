# Development

Use Node.js 26+ and pnpm with the existing lockfile. [package.json](../../package.json) owns scripts and runtime requirements. The application uses native APIs without polyfills.

## Setup and commands

| Command                          | Purpose                                             |
| -------------------------------- | --------------------------------------------------- |
| `pnpm install --frozen-lockfile` | Install recorded dependencies                       |
| `pnpm dev` / `pnpm preview`      | Serve source / built output                         |
| `pnpm check`                     | Typed Oxlint, Node/worker TypeScript and CSS checks |
| `pnpm check:css`                 | Stylelint with the standard CSS configuration       |
| `pnpm build`                     | Static checks and production bundle                 |
| `pnpm test`                      | CPU behavior and physical references                |
| `pnpm test:gpu`                  | Native GPU execution                                |
| `pnpm test:ui`                   | Browser and worker interaction                      |
| `pnpm test:visual`               | Warm/blue photographs for manual review             |
| `pnpm bench` / `pnpm bench:gpu`  | CPU / GPU measurements                              |
| `pnpm fmt:check`                 | Check supported source/document formatting          |
| `pnpm exec oxfmt <paths>`        | Format selected files                               |

Browser projects use Vitest's [Playwright provider](https://vitest.dev/config/browser/playwright) and full Chromium in [modern headless mode](https://playwright.dev/docs/browsers#chromium-new-headless-mode). Install a missing binary with `pnpm exec playwright install chromium`. GPU checks require a usable adapter with the application's features.

## Test configuration conventions

[vitest.config.ts](../../vitest.config.ts) discovers `tests/{cpu,gpu,browser,visual}/**/*.test.ts`. Flat `benchmarks/cpu*.bench.ts` files run in Node; remaining flat benchmark files run in the GPU project. Matching files need no configuration edit. Benchmark scripts use project-name wildcards to include Vitest's [benchmark sibling projects](https://vitest.dev/config/benchmark).

The test configuration [merges the application's Vite configuration](https://vitest.dev/config/) so browser tests use the same Lightning CSS transformer. Inline projects [inherit root policy](https://vitest.dev/guide/projects). CPU tests use [native Node imports](https://vitest.dev/config/experimental#experimental-vitemodulerunner); browser projects use Vite transforms. Sequence groups run CPU, GPU, browser and visual projects in order, and browser files run serially to avoid competing GPU workloads. Separate Vitest processes still need coordination.

`pnpm exec vitest run` selects all four projects, including visual captures. Use the narrower scripts routinely; inspect discovery with `pnpm exec vitest list --filesOnly`. [Validation](../validation/methods.md) owns test design and measurement protocol.

## Compiler and build

The [root TypeScript config](../../tsconfig.json) enables strict native ESM, unchecked-index awareness, exact optional fields, erasable syntax, type-only imports and no emission. Environment globals are separated:

| Config                                           | Scope                                                                         |
| ------------------------------------------------ | ----------------------------------------------------------------------------- |
| [App](../../tsconfig.app.json)                   | Browser code, browser/GPU/visual tests and GPU benchmarks; bundler resolution |
| [Node](../../tsconfig.node.json)                 | Config files, CPU tests/references and CPU benchmarks; NodeNext resolution    |
| [Worker](../../src/runtime/worker/tsconfig.json) | Dedicated-worker entry and worker globals                                     |

Browser code must not acquire Node globals, and Node/worker code must not acquire DOM globals. The Node project also checks native ESM `.config.mjs` files through JSDoc, without a configuration transpiler. `@types/web` and `@types/webworker` supply platform declarations. A narrow assertion bridges the missing `texture-formats-tier2` literal; runtime checks establish support.

[Vite](https://vite.dev/guide/features) builds raw WGSL imports and module workers with ESNext output, Lightning CSS and no module-preload polyfill. Use `new Worker(new URL(..., import.meta.url), { type: "module" })` so Vite owns the worker graph.

## Lint and formatting

[.oxlintrc.json](../../.oxlintrc.json) owns typed linting, TypeScript diagnostics, failure on warnings and unused-suppression reporting. Its explicit plugin list [replaces Oxlint defaults](https://oxc.rs/docs/guide/usage/linter/config.html#configure-plugins). Rules enforce unsafe-value, promise and exhaustive-switch contracts; environment overrides match Node and worker ownership. Vitest APIs are imported.

[.oxfmtrc.json](../../.oxfmtrc.json) enables [Oxfmt import sorting](https://oxc.rs/docs/guide/usage/formatter/sorting) and otherwise uses the defaults. Side-effect imports retain their order. Oxfmt and Oxlint respect `.gitignore` during discovery but accept explicitly named ignored files. Add tool-specific exclusions only when needed.

[Stylelint](https://stylelint.io/user-guide/get-started/) checks `src/**/*.css` in both `check:css` and `check`, and therefore in `build`. [Its ESM configuration](../../stylelint.config.mjs) extends the official standard config, checks custom-property references against the shared theme, rejects warnings, and reports invalid, unnecessary or unexplained disable comments. Oxfmt owns whitespace; Stylelint owns CSS semantics and notation. Global descending-specificity ordering is disabled because it cannot infer independent DOM components or mutually exclusive states. Native nesting, cascade layers, relative colors and logical properties need no custom syntax or compatibility plugins. Run `pnpm check:css --fix` for supported fixes, then format the edited files with Oxfmt.

These tools do not format or validate WGSL; see [shader tooling](shaders.md#formatting).

## Value and resource conventions

Use small functions with explicit inputs, readonly results and discriminated errors. Validate external values without coercion before scene construction. Readonly does not freeze arrays. [presentation.ts](../../src/scene/presentation.ts) owns finite view choices and ranges shared by decoders, controls and session actions.

Use native collection methods for value transformations; retain loops, typed-array writes and local caches in quadrature, integration and packing. Changing summation order or adding allocations is a numerical/performance change, not merely style. These conventions follow [TypeScript's functional guide](https://www.typescriptlang.org/docs/handbook/typescript-in-5-minutes-func.html).

Pass state explicitly to control transforms and synchronize all controls from one accepted session snapshot. Return complete endpoint records from observer branches, then validate the common result. Prefer exhaustive switches for domain variants and small named functions for descriptions; avoid currying or generic composition layers that obscure inputs or inference. Follow native [ECMAScript collection semantics](https://tc39.es/ecma262/multipage/indexed-collections.html): `map` transforms entries, `filter` selects them, and neither substitutes for validation of sparse external arrays.

Keep effects in resource owners. `DisposableStack` releases partial acquisitions; `move()` transfers the completed lifetime. Build replacements before publishing them. Parallelize independent work only when failure cannot orphan resources; serialize shared GPU submissions and canvas ownership. [Architecture](architecture.md) defines lifecycle and reuse.

## Comments and declaration documentation

Use `/** ... */` for declaration contracts and `//` for local reasoning. Explain units, domains, coordinate/frame conventions, mutation/aliasing, ownership and failure meaning where types cannot. Self-explanatory helpers need no tag checklist.

Use [TSDoc summaries and remarks](https://tsdoc.org/pages/tags/remarks/), `@param name - meaning`, and `@returns` for borrowed storage, sentinels or completion semantics. Do not duplicate TypeScript types or modifiers in tags. Describe actionable exceptions with `@throws` and asynchronous rejection where relevant; these are informational contracts. Link symbols with `{@link symbol}` and sources with `@see`. Preserve licenses, compiler directives and justified suppressions.

WGSL comments are not parsed as TSDoc. Document packed lanes, normalization, arithmetic prerequisites and validity flags. Keep derivations in physics/numerics. Oxlint's JSDoc rules check supported tags and descriptions, including `remarks`; they are not a complete API-documentation parser.

## Native browser baseline

The application requires a secure context, WebGPU, `subgroups`, `texture-formats-tier2`, six storage textures per stage, and WGSL `readonly_and_readwrite_storage_textures`. [device.ts](../../src/gpu/device.ts) checks the shared GPU requirements. See [WebGPU texture tiers](https://www.w3.org/TR/webgpu/#texture-formats-tier2) and [WGSL](https://www.w3.org/TR/WGSL/).

Other requirements include dedicated-worker OffscreenCanvas/animation frames, Display P3 with `rgba16float` presentation, device-pixel ResizeObserver sizes, native popovers/dialog commands, exclusive disclosures, CSS nesting/layers, subgrid and OKLCH. ECMAScript APIs include `DisposableStack`, `Float16Array`, iterator helpers and `Promise.withResolvers`. No fallback layer emulates them; device validation is not a complete browser-API preflight.

For documentation-only edits, run `pnpm exec oxfmt --check README.md docs AGENTS.md NOTICE.md` and verify links, anchors, formulas, commands and status. Runtime tests are needed only if behavior changes. No speed gate or complete CI workflow is configured.
