# Tooling and workflow

Use Node.js 26 or newer and global pnpm with the existing lockfile. The repository intentionally has no pinned `packageManager` field. A successful installation is not validation.

## Commands

| Command                          | Purpose                                                          |
| -------------------------------- | ---------------------------------------------------------------- |
| `pnpm install --frozen-lockfile` | Install the recorded dependency graph                            |
| `pnpm dev`                       | Vite development server                                          |
| `pnpm check`                     | Type-aware Oxlint, Node/config types, and dedicated-worker types |
| `pnpm build`                     | Static checks and production Vite bundle                         |
| `pnpm preview`                   | Serve the built output                                           |
| `pnpm test`                      | CPU model, physics, and reference tests                          |
| `pnpm test:gpu`                  | Actual browser WGSL and rendering tests                          |
| `pnpm test:ui`                   | Actual browser interaction and worker integration tests          |
| `pnpm test:watch`                | CPU watch mode                                                   |
| `pnpm bench`                     | Production CPU preparation benchmarks                            |
| `pnpm bench:gpu`                 | Optical GPU timestamps and initialization timings                |
| `pnpm format:check`              | Check formatting                                                 |
| `pnpm exec oxfmt <paths>`        | Format only intended files                                       |

Browser tests use a shared factory that creates independent Vitest Playwright options using installed Chromium with `channel: "chromium"`; this selects the full browser's modern headless mode. Fresh instance objects are necessary because Vitest assigns names while expanding projects. See [Vitest browser configuration](https://vitest.dev/config/browser/playwright) and [Playwright channels](https://playwright.dev/docs/browsers#chromium-new-headless-mode). When it is missing, install the matching browser with `pnpm exec playwright install chromium`. WebGPU checks must actually execute on a usable adapter. A missing prerequisite is a reported failure to validate, not a passed or silently skipped GPU job.

## Language and compiler boundaries

[TypeScript](https://www.typescriptlang.org/docs/handbook/intro.html) is configured for native ESM, ESNext, strict checking, unchecked-index awareness, exact optional fields, type-only imports, erasable syntax, and no emitted compiler output. Vite builds the application; type checking is separate.

The root `tsconfig.json` owns common compiler policy and references three environment projects; it includes no source files itself. `tsconfig.app.json` uses `module: Preserve` and its implied bundler resolution for browser code, UI/GPU tests, and GPU benchmarks. `tsconfig.node.json` uses `module: NodeNext` and Node globals for configuration, CPU tests, references, and CPU benchmarks. `src/runtime/worker/tsconfig.json` supplies dedicated-worker globals. Oxlint follows this solution graph; per-directory forwarding configs are unnecessary. Browser code cannot silently acquire `process`, and Node/worker code cannot acquire `document`.

DOM declarations come from `@types/web`; workers use `@types/webworker`, both with the ESNext language library. The [GPUWeb type integration guidance](https://github.com/gpuweb/types#integration-into-dom-type-definitions) identifies incomplete WebGPU declarations in TypeScript 7.0.2 and recommends `@types/web` 0.0.352 or newer. The standalone `@webgpu/types` package is therefore removed. The current declaration packages still omit `texture-formats-tier2`; one narrow assertion bridges that literal while an actual adapter-feature check remains mandatory.

Use readonly domain records and discriminated unions for state and outcomes. Validate external values as `unknown`. A cast cannot validate a view, create an orthonormal camera, or establish numerical support. Avoid frameworks, generic effect systems, or immutable copies inside hot loops without a demonstrated benefit.

The runtime directly uses modern [ECMAScript resource management](https://tc39.es/ecma262/multipage/control-abstraction-objects.html#sec-disposablestack-objects), iterator helpers, typed arrays, and other ESNext facilities. There are no polyfills. JavaScript numbers remain binary64; storing into a float typed array changes representation, not subsequent arithmetic semantics.

## Vite and Oxc

[Vite](https://vite.dev/guide/features) handles TypeScript and explicit `?raw` WGSL imports. Module workers use `new Worker(new URL(..., import.meta.url), { type: "module" })` so the build owns their dependencies. Lightning CSS processes the interface styles. The build targets ESNext and explicitly disables the [default module-preload polyfill](https://vite.dev/config/build-options.html#build-modulepreload); a syntax target alone does not disable it. Configuration stays at the repository root. ESNext output, disabling the preload polyfill, and the Lightning CSS transformer deliberately differ from Vite defaults; other build defaults are inherited. See the [build](https://vite.dev/config/build-options) and [shared](https://vite.dev/config/shared-options) options.

[Oxlint](https://oxc.rs/docs/guide/usage/linter/type-aware) checks correctness, suspicious code, performance pitfalls, unsafe values, promise ownership, exhaustive switches, and unused suppressions. Warnings fail the check. JSDoc rules check existing documentation without requiring redundant comments on every function. Use JSDoc blocks for declaration contracts: units, supported domains, ownership, and nonobvious formulas. TypeScript supplies type signatures; avoid redundant `@param` types and comments that merely repeat names. WGSL uses the same `/** ... */` documentation style, with ordinary line comments for local algorithm reasoning and tool directives. Oxlint checks TypeScript JSDoc, not WGSL comments.

[Oxfmt](https://oxc.rs/docs/guide/usage/formatter/config.html) formats its supported languages and Markdown. There is no formatter configuration file because the repository uses its defaults. It does not validate or format WGSL. `.gitignore` excludes generated output; do not duplicate those paths across tool configs. Do not update dependencies or reformat unrelated work incidentally.

## Native browser baseline

The application requires WebGPU, subgroup operations, the `texture-formats-tier2` device feature, WGSL `readonly_and_readwrite_storage_textures`, dedicated-worker OffscreenCanvas, worker animation frames, `rgba16float` presentation, Display P3 canvas support, native popovers, CSS anchor positioning, subgrid, OKLCH, and device-pixel ResizeObserver sizes. Required GPU features are requested through one shared list used by the application and benchmarks. Unsupported required capabilities produce an error. This project does not maintain a legacy renderer or CSS fallback stack.

Read/write `rgba32float` storage requires [texture formats tier 2](https://www.w3.org/TR/webgpu/#texture-formats-tier2); it is an optional adapter capability, not a guarantee from WebGPU availability alone. It permits one in-place photographic history texture. The shader declares its [read/write storage-texture language requirement](https://www.w3.org/TR/WGSL/#readonly_and_readwrite_storage_textures). Validate both requirements on the actual browser and adapter.

Use native WGSL vector, transcendental, bit, atomic, and subgroup operations where their contracts fit. Coverage uses integer `subgroupAdd` and an elected writer, with one workgroup barrier and no assumed subgroup size. Optical quantities remain f32. `f16`, experimental buffer views, or a new subgroup execution layout are not automatic accuracy or performance improvements. In particular, WGSL `fma` does not guarantee the ordered IEEE operations needed by an error-free two-float expansion. See [WGSL](https://www.w3.org/TR/WGSL/) and [WebGPU](https://www.w3.org/TR/webgpu/).

The current native features have concrete roles:

| Platform facility                                                                          | Use                                                                                              |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| `using`, `DisposableStack`, `move()`                                                       | Exception-safe acquisition and explicit lifetime transfer                                        |
| `Float16Array`                                                                             | Native half-float radiance uploads/readbacks; optical arithmetic stays f32                       |
| `Uint8Array.toHex()` / `toBase64()`                                                        | Benchmark fingerprints and binary test artifacts                                                 |
| Iterator helpers, `toSorted`, `Promise.withResolvers`                                      | Collection queries, nonmutating ordering, and worker request completion                          |
| HTML popovers, dialog commands, exclusive `details`, customizable native selects           | Native dismissal, focus, settings groups, and accessible option menus                            |
| CSS nesting/layers, relative OKLCH, subgrid, anchor fallbacks, `@starting-style`, `:has()` | Shared styles, color tokens, aligned controls, placement, transitions, and empty feedback layout |
| `interpolate-size`, `::details-content`, `transition-behavior: allow-discrete`             | Native intrinsic-size disclosure and popover entry/exit transitions                              |
| Dynamic viewport units, safe-area insets, reduced-motion/transparency preferences          | Responsive layouts and user preferences                                                          |

These facilities are used directly rather than polyfilled. The [ECMAScript specification](https://tc39.es/ecma262/) defines the language APIs, including [byte-to-hex conversion](https://tc39.es/ecma262/#sec-uint8array.prototype.tohex). [Architecture](architecture.md#interaction-and-presentation) links the HTML/CSS contracts. They complement the WebGPU/worker requirements above; they are not all recent additions to this codebase.

OKLCH describes interface colors, not physical radiance. HDR content is enabled through the canvas tone-mapping contract, while output headroom remains browser/OS/display dependent. [Physics](physics.md#spectral-transfer-and-display) owns the spectral and display transforms.

## Benchmarks

Benchmarks use the existing [Vitest benchmark API](https://vitest.dev/config/benchmark) and its Tinybench execution, rather than a second timing harness. Every workload consumes real production results. CPU preparation reports runtime, hardware, source fingerprints, inputs, and warmup. Startup timing includes source construction, shader/pipeline preparation, uploads, completion, and disposal on an existing device; it is not a cold browser launch.

Optical timing requires `timestamp-query`. Missing timestamp support is explicitly skipped without substituting CPU time. Every executed optical pass writes a beginning/end pair. The frame interval spans the first beginning to the final end; per-stage samples include warmup and are recorded separately from measured aggregate statistics. Instrumentation can alter scheduling. Readbacks, assertions, hashes, CPU preparation, queue submission, and presentation remain outside the timed interval. Invalid intervals are recorded and rejected.

Eight optical workloads distinguish useful reuse boundaries: `full-frame` forces physical image search and detector work; `view-sampling` reuses physical source images but traces detector geometry; `lighting` reuses geometry; `bloom` measures photographic filtering. These use 1280 × 720. `retrograde`, `charged`, `near-extremal`, and `distant` each force a full 640 × 360 solve to expose parameter-dependent cost and failures. Run `pnpm bench:gpu -t 'full-frame|view-sampling'` for a focused comparison. Startup is a separate existing-device workload.

Results under `test-results/bench/` use metadata schema 1 and include exact image readbacks, separate production TypeScript/WGSL and benchmark-harness fingerprints, data fingerprints, sample policy, adapter/browser identity, queue counts, unresolved coverage, and discarded timestamps. Texture capture handles padded rows for arbitrary widths after timing, and aggregate validation rejects nonfinite output or invalid sample fractions. Forcing a search reuses allocated resources; it does not recreate the renderer per frame. Compare equal inputs and image quality before claiming a speedup. Pause other renderer previews, serialize GPU workloads, and record any unverified concurrency. [Validation](validation.md#performance) defines interpretation and acceptance.
