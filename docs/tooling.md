# Tooling and workflow

Use Node.js 26 or newer and pnpm with the existing lockfile. The project targets native modern APIs and has no compatibility or polyfill layer. Installation alone is not validation.

## Commands

| Command                          | Purpose                                                             |
| -------------------------------- | ------------------------------------------------------------------- |
| `pnpm install --frozen-lockfile` | Install the recorded dependency graph                               |
| `pnpm dev`                       | Start Vite                                                          |
| `pnpm check`                     | Type-aware Oxlint plus Node/config and dedicated-worker type checks |
| `pnpm build`                     | Static checks and production Vite bundle                            |
| `pnpm preview`                   | Serve the built output                                              |
| `pnpm test`                      | CPU domain and physical-reference checks                            |
| `pnpm test:gpu`                  | Native browser GPU execution                                        |
| `pnpm test:ui`                   | Browser interaction and worker behavior                             |
| `pnpm test:watch`                | Watch CPU tests                                                     |
| `pnpm bench`                     | Production CPU preparation measurements                             |
| `pnpm bench:gpu`                 | Submitted optical-frame and initialization measurements             |
| `pnpm format:check`              | Check supported source/document formatting                          |
| `pnpm exec oxfmt <paths>`        | Format intended files                                               |

Browser projects use [Vitest's Playwright provider](https://vitest.dev/config/browser/playwright) with `channel: "chromium"`, selecting the full browser's [modern headless mode](https://playwright.dev/docs/browsers#chromium-new-headless-mode). Install a missing matching binary with `pnpm exec playwright install chromium`. A browser launch without an actual usable WebGPU adapter cannot validate WGSL. Report missing prerequisites instead of skipping a required check silently.

## Compiler and environment boundaries

The root TypeScript configuration defines strict native ESM policy: unchecked-index awareness, exact optional fields, type-only imports, erasable syntax and no compiler emission. Its solution references own separate environment globals:

- `tsconfig.app.json`: browser code, browser/GPU tests and GPU benchmarks, with `module: Preserve` and bundler resolution.
- `tsconfig.node.json`: configuration, CPU tests/references and CPU benchmarks, with `module: NodeNext` and Node globals.
- `src/runtime/worker/tsconfig.json`: the dedicated-worker entry and worker globals.

Browser code must not acquire `process`, and Node/worker code must not acquire `document`. DOM declarations use `@types/web`; worker declarations use `@types/webworker`. Follow [GPUWeb's declaration integration guidance](https://github.com/gpuweb/types#integration-into-dom-type-definitions). One narrow assertion covers the currently missing `texture-formats-tier2` literal; runtime adapter validation still establishes support.

[TypeScript's functional-language guide](https://www.typescriptlang.org/docs/handbook/typescript-in-5-minutes-func.html) supports the project's small structural records, explicit function inputs and discriminated unions. Readonly is a compile-time property, not deep immutability. Validate untrusted values as `unknown` and narrow them before construction. Keep direct mutation in numerical working buffers where copying has no semantic benefit.

## Build, lint and formatting

[Vite](https://vite.dev/guide/features) builds native TypeScript, explicit raw WGSL imports and module workers. ESNext output, Lightning CSS and disabled [module-preload polyfilling](https://vite.dev/config/build-options.html#build-modulepreload) are deliberate settings; other defaults are inherited. Worker URLs use `new Worker(new URL(..., import.meta.url), { type: "module" })` so Vite owns their dependency graph.

[Oxlint](https://oxc.rs/docs/guide/usage/linter/type-aware) checks correctness, unsafe values, promises, exhaustive handling and supported performance rules. Warnings and unused suppressions fail. JSDoc records units, domains, ownership and failure meaning, while local comments explain algorithm choices. Do not duplicate a TypeScript signature in prose.

[Oxfmt](https://oxc.rs/docs/guide/usage/formatter/config.html) uses default formatting for supported languages and Markdown. Neither tool validates or formats WGSL; actual GPU compilation/execution is required. Generated outputs are excluded through `.gitignore`. Keep dependency changes and unrelated formatting outside a focused task.

## Native browser baseline

The application requires WebGPU, subgroup operations, six storage textures per stage, `texture-formats-tier2`, WGSL `readonly_and_readwrite_storage_textures`, dedicated-worker OffscreenCanvas/animation frames, Display P3 and `rgba16float` presentation. [The device constructor](../src/gpu/device.ts) checks shared requirements and reports missing support.

[Texture formats tier 2](https://www.w3.org/TR/webgpu/#texture-formats-tier2) enables the in-place f32 photographic history. The shader declares its [read/write storage requirement](https://www.w3.org/TR/WGSL/#readonly_and_readwrite_storage_textures). Sky reduction reads one mip and writes another through single-mip views; distinct subresources obey WebGPU's usage-scope rules. Work textures stay alive through generation completion.

[WGSL](https://www.w3.org/TR/WGSL/) supplies native vectors, bit operations, transcendentals, atomics and subgroup reductions. Coverage reduces integers with `subgroupAdd` and `subgroupElect`, then synchronizes the workgroup. It assumes neither subgroup size nor lane/workgroup correspondence. Optical arithmetic stays f32. Reduced precision, `fma`, subgroup matrices or experimental extensions require a numerical and workload-specific reason; availability alone does not establish an improvement.

The browser also needs native popovers/dialog commands, exclusive disclosures, customizable selects, CSS nesting/layers, subgrid, OKLCH, anchor positioning and device-pixel ResizeObserver sizes. [HTML](https://html.spec.whatwg.org/multipage/) owns focus, dismissal and media behavior; [CSS specifications](https://www.w3.org/Style/CSS/Overview.en.html) define layout and user-preference behavior. UI color tokens do not represent physical radiance.

Local audio uses [native media controls](https://html.spec.whatwg.org/multipage/media.html#the-audio-element), because the product needs playback, seeking and volume without signal processing. Object URLs have explicit replacement/disposal lifetimes. A [Web Audio graph](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Best_practices) would be appropriate for processing or synthesis, which this soundtrack feature does not perform.

Native [ECMAScript](https://tc39.es/ecma262/) facilities have concrete roles: `DisposableStack` transfers ownership; `Float16Array` encodes storage without changing optical precision; `toSorted` orders independent values; map insertion helpers reuse exact local spectral calculations; iterator helpers handle collections; `Promise.withResolvers` represents worker requests; byte hex/base64 methods encode artifacts. The application does not emulate these APIs.

## Tests and measurements

Tests use [Vitest](https://vitest.dev/guide/), [fast-check](https://fast-check.dev/docs/) and actual [Playwright browser interactions](https://playwright.dev/docs/best-practices). Lazy device fixtures own and release a real GPU device. Property tests explore mathematical domains and preserve replay information on failure. Browser assertions use accessible roles and observable behavior rather than native controls' internal DOM.

Benchmarks use the installed Vitest benchmark context and Tinybench execution. Every callback produces a retained production result. CPU workloads cover scene updates, camera changes at prepared free-fall events, disk/thermal tables and the complete stellar tree. Output hashing and metadata are outside timing.

GPU initialization includes pipeline preparation, catalogue/tree construction, spectral tables, GPU noise/cube generation, uploads, completion and disposal on an existing device. One warmup and five measurements retain browser/driver caches; this is not cold launch. Optical workloads measure eight submitted frames after one warmup at 320×180 on reused resources. Readback, initialization, bloom and presentation remain outside frame timing.

`test-results/bench/` records schema-1 inputs, source and harness/dependency fingerprints, hardware, sampling and unresolved image weight. Source fingerprints include binary catalogue content. Timing distributions are comparative evidence with no speed threshold. Compare scope and image quality before interpreting a difference; [Validation](validation.md#performance) defines that policy.
