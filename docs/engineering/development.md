# Development

Use Node.js 26+ and pnpm with the existing lockfile. The application uses native browser APIs without polyfills.

## Commands

| Command                          | Purpose                                         |
| -------------------------------- | ----------------------------------------------- |
| `pnpm install --frozen-lockfile` | Install recorded dependencies                   |
| `pnpm dev` / `pnpm preview`      | Serve source / built output                     |
| `pnpm check`                     | Oxlint (typed), Node and worker `tsc`, CSS lint |
| `pnpm check:css`                 | Stylelint only                                  |
| `pnpm build`                     | `check`, then the production bundle             |
| `pnpm test`                      | CPU behavior and physical references            |
| `pnpm test:gpu`                  | Native GPU execution in Chromium                |
| `pnpm test:ui`                   | Browser and worker interaction                  |
| `pnpm test:visual`               | Warm/blue photographs for manual review         |
| `pnpm bench` / `pnpm bench:gpu`  | CPU / GPU measurements                          |
| `pnpm fmt` / `pnpm fmt:check`    | Format / check formatting                       |

Browser projects run in headless Chromium through Vitest's [Playwright provider](https://vitest.dev/config/browser/playwright). Install a missing binary with `pnpm exec playwright install chromium`. GPU checks need an adapter with the [required features](#native-browser-baseline).

For documentation-only edits, run `pnpm exec oxfmt --check README.md docs AGENTS.md NOTICE.md` and check links, anchors, formulas, commands and status. No speed gate or CI workflow is configured.

## Configuration

Tools run on their defaults; each file states only what the project needs.

| File                                         | What it adds to the defaults                                                                              |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| [vite.config.ts](../../vite.config.ts)       | `esnext` output, so native syntax such as `using` ships unlowered                                         |
| [vitest.config.ts](../../vitest.config.ts)   | Four projects discovered by directory: `tests/{cpu,gpu,browser,visual}/**/*.test.ts`                      |
| [tsconfig.json](../../tsconfig.json)         | Strict, erasable, type-only-import ESM without emission; three environment projects                       |
| [.oxlintrc.json](../../.oxlintrc.json)       | Type-aware linting with TypeScript diagnostics; correctness, suspicious and perf findings are errors      |
| [.oxfmtrc.json](../../.oxfmtrc.json)         | Import sorting                                                                                            |
| [.stylelintrc.json](../../.stylelintrc.json) | The standard config, custom-property checks against the theme, and whitespace rules left to the formatter |

The CPU project runs in Node with [native TypeScript imports](https://vitest.dev/config/experimental#experimental-vitemodulerunner). The three browser projects share one GPU, so they run in order with one file at a time. `benchmarks/cpu*.bench.ts` runs in Node; other benchmark files run in the GPU project. New files matching these patterns need no configuration change. `pnpm exec vitest run` selects all four projects, including visual captures.

TypeScript globals are separated by environment so browser code cannot use Node globals and worker code cannot use the DOM:

| Config                                           | Scope                                                     |
| ------------------------------------------------ | --------------------------------------------------------- |
| [App](../../tsconfig.app.json)                   | Browser code, browser/GPU/visual tests and GPU benchmarks |
| [Node](../../tsconfig.node.json)                 | Config files, CPU tests, references and CPU benchmarks    |
| [Worker](../../src/runtime/worker/tsconfig.json) | The dedicated-worker entry                                |

Create workers with `new Worker(new URL(..., import.meta.url), { type: "module" })` so Vite owns the worker graph. WGSL is imported with `?raw`; no tool here formats or validates it — see [WGSL](shaders.md).

## Code conventions

- **Validate at boundaries, trust inside.** URL fragments, storage and files are decoded as `unknown` in `src/scene/` and replaced atomically. Code behind that boundary receives typed, quantized values and does not re-check them. A guard inside `src/physics/` or `src/gpu/` must express a physical or numerical domain, not distrust of the caller.
- **Exhaustive switches need no `default`.** The compiler and linter reject a missing case.
- **Values.** Prefer small functions, explicit inputs, readonly records and discriminated results. Readonly does not freeze typed arrays; document writes to caller-owned storage.
- **Hot paths.** Keep loops, typed-array writes and local mutation in quadrature, integration and packing. Changing summation order or adding allocations is a numerical change.
- **Resources.** `DisposableStack` releases partial acquisitions and `move()` transfers a completed lifetime. Build a replacement before publishing it. Serialize GPU submissions and canvas ownership. [Architecture](architecture.md) defines lifetimes.
- **Shared ranges.** [presentation.ts](../../src/scene/presentation.ts) owns the choices and ranges used by decoders, controls and session actions.

## Comments

Use `/** ... */` for declaration contracts and `//` for local reasoning. State what types cannot: units, domains, coordinate and frame conventions, aliasing, ownership and the meaning of failure. Follow [TSDoc](https://tsdoc.org/) (`@remarks`, `@param name - meaning`, `@returns`, `@throws`); do not repeat types. Self-explanatory helpers need no comment.

In WGSL, document packed lanes, normalization, arithmetic prerequisites and validity flags. Derivations belong in the physics and numerics documents.

## Native browser baseline

The renderer needs a secure context and WebGPU with `subgroups`, `texture-formats-tier2`, six storage textures per stage and the WGSL language feature `readonly_and_readwrite_storage_textures`. [device.ts](../../src/gpu/device.ts) requests and checks these; see [WebGPU texture tiers](https://www.w3.org/TR/webgpu/#texture-formats-tier2) and [WGSL](https://www.w3.org/TR/WGSL/). `@types/web` does not yet declare the tier-2 feature name, so one assertion bridges that gap.

The interface also uses OffscreenCanvas in a dedicated worker, Display P3 `rgba16float` presentation, device-pixel `ResizeObserver` sizes, popovers and dialog invoker commands, exclusive `<details>`, CSS nesting, layers, subgrid and OKLCH relative colors. Script uses `DisposableStack`, `Float16Array`, iterator helpers and `Promise.withResolvers`. Nothing emulates a missing API.
