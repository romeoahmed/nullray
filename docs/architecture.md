# Architecture

Nullray separates user intent, physical preparation, GPU execution, and browser interaction. The renderer runs in a dedicated worker. There is no application framework, shader DSL, or runtime dependency.

## Module boundaries

| Location              | Owns                                                                                                      |
| --------------------- | --------------------------------------------------------------------------------------------------------- |
| `src/model/`          | Validated scenes, camera placement, source appearance, view codec, presets, immutable session transitions |
| `src/physics/`        | Metric and neutral circular orbits, disk flux, spectra, celestial data, vector and photon calculations    |
| `src/render/`         | Device resources, optical passes, accumulation, bloom, coverage, PNG encoding                             |
| `src/render/shaders/` | Explicit WGSL formulas and compute/render entry points                                                    |
| `src/render/worker/`  | OffscreenCanvas, renderer lifetime, simulation clock, bounded frame scheduling                            |
| `src/explorer/`       | DOM controls, pointer/keyboard navigation, local bookmarks, render-worker client, CSS                     |
| `src/data/`           | Attributed CIE and HYG source data                                                                        |
| `tests/reference/`    | Binary64 separated solver and independent Hamiltonian integrator; never imported by production            |
| `benchmarks/`         | Production preparation and GPU workloads                                                                  |

`model` and `physics` have no DOM, GPU handles, clock, randomness, or persistence. The DOM sends intent across [the render protocol](../src/render/protocol.ts); it does not manipulate GPU resources. The physics modules produce deterministic data from explicit inputs. Small immutable records express state; direct loops and local mutation remain appropriate for numerical arrays. Fresh typed arrays transfer ownership to their caller; readonly record fields prevent reassignment but do not freeze array elements. GPU encoders and DOM bindings are explicit effect boundaries, not domain operations. WGSL trace functions return values, while derivative helpers mutate only a caller-owned failure flag; no module-global derivative state is shared between evaluations.

## State and validation

[The initial view](../src/model/view.ts) owns shared scene, source, and display defaults used by presets and session initialization. [Session](../src/model/session.ts) combines a saved view, motion (`playing`, `paused`, or `refining`), and exploration resolution. Its exhaustive action union produces either a complete new session or a validation error. Invalid coupled parameters cannot partially change a scene.

[Scene construction](../src/model/scene.ts) validates the coupled spacetime, exterior observer, camera axes, and relevant f32 rounding before deriving disk geometry. Camera-only updates reuse the previous spacetime's ISCO. This is an explicit input dependency, not a global cache. Symbols mark constructed values inside one realm; structured clone does not preserve that brand, so the worker reconstructs scenes and appearances.

URLs and local storage are untrusted input. The [view codec](../src/model/view.ts) parses them as `unknown` and validates every field before restoring anything. The current version is 4; earlier versions have explicit defaults. A snapshot contains source inputs and emission epoch, not GPU handles or derived textures. Restoring a view pauses motion. The browser-local library limits names and collection size, rejects duplicates, preserves corrupt storage, and offers one deletion undo.

## Worker and resource lifetime

The application transfers a fresh canvas once. Remounting replaces that element because an OffscreenCanvas transfer cannot be reversed. Vite updates to the application module preserve its current session; a full page reload restores the URL view.

The client coalesces visible control changes into one animation-frame message. Hidden or zero-sized surfaces are sent immediately. Monotonic revisions reject obsolete frame and export responses. Geometry and appearance are cloned only when their identity changes; presentation controls travel separately.

The worker keeps at most one submitted frame outstanding. Incoming messages replace pending intent while the GPU finishes. It schedules another frame only for changed input, animation, or an unfinished photograph. Hidden and zero-sized surfaces stop scheduling. Animation advances ten geometric time units per second, caps each elapsed increment at 0.25 seconds, and resets its clock baseline across pause and visibility changes.

`DisposableStack` gives each device, render target, and temporary readback a clear owner. A lexical `using` stack unwinds resources on initialization failure, then `move()` transfers successful initialization to the returned owner. Application mounts use the same rule for event listeners, resize observation, the render client, and photograph URLs. Disposal aborts initialization, stops scheduling, destroys resources, and closes the worker. Device loss exposes retry with the same scene. A crashed worker requires a page reload; retry first preserves the current view in the URL. Async frame/export results are discarded after replacement or disposal.

[OffscreenCanvas](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas) and [dedicated workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API) provide the execution boundary. Moving scheduling off the main thread improves input isolation; it does not shorten a GPU optical solve.

## Render path

1. Trace primary rays to their first physical endpoint.
2. Find sky pixels without usable same-branch derivatives; repair retained candidates with differentiated rays.
3. During exploration, select mixed boundaries and varying disk pixels, trace a bounded 4 × 4 subpixel atlas, and repair its missing sky derivatives.
4. Evaluate disk and sky spectra from the retained endpoints and source controls.
5. For photography, accumulate 64 jittered samples in binary32 history at native resolution.
6. Filter linear radiance through normalized multiscale bloom and present it in Display P3.

The numerical meaning and limitations of these passes belong in [Numerics](numerics.md#pixel-integration). No pass converts exhausted work into capture or successful radiation.

Photography owns one read/write `rgba32float` history texture and one `rgba16float` output texture: 24 bytes per pixel, excluding optical targets. Each compute invocation reads and replaces its own history texel; ordered passes provide visibility between samples. This removes a second binary32 history allocation (126.6 MiB at 3840 × 2160). History stores mean linear radiance and missing sample weight. Coverage reduces integer sixteenth-sample units so partially resolved 4 × 4 quadrature is retained. Sample indices run sequentially from 0 through 63; index zero starts a new sequence, including after resize.

| Change                                | Recomputed work                                               |
| ------------------------------------- | ------------------------------------------------------------- |
| Camera, spacetime, raster dimensions  | Geometry, derivative queues, radiation, history, bloom        |
| Emission time or source appearance    | Radiation and history; exploration geometry stays cached      |
| Exposure, bloom strength, HDR mode    | Presentation; completed photographic samples remain available |
| New photographic sample               | Jittered geometry, radiation, accumulation, bloom             |
| Exploration resolution while refining | No photographic resize; photography remains native            |

The render worker builds source maps on the CPU and uploads them once per renderer. Star traversal uses a stackless 32-byte node: internal nodes carry bounds and an escape index; leaves reuse those lanes for direction and spectral data. The maximum catalogue size keeps indices exactly representable in f32. Queues retain bounded capacities and explicit overflow status. Workgroup dispatch order never determines physical ray order.

## Interaction and presentation

Static semantic markup lives in [index.html](../index.html). Settings, Views, and About use native popovers with CSS anchor positioning. Native details group secondary controls; subgrid aligns labels and values. The main view remains available for navigation, with noninteractive overlays using `pointer-events: none` and actual controls opting back in. CSS uses OKLCH, 44-pixel button targets, visible focus, dynamic viewport units, safe-area insets, and reduced-motion/transparency preferences.

Orbit mode turns observer position around the center; drag and arrows turn, wheel/pinch approaches. Free mode rotates the camera's own frame and translates with WASD/QE. Translation uses an explicit spherical placement chart, carries orientation between local bases, and returns through scene validation. It is not a spatial isometry, Kerr–Schild transform, or physical observer worldline. Navigation never adds aberration or Doppler velocity.

Photography freezes emission time. Coverage reports unresolved sample weights as well as affected pixels. PNG export submits a snapshot before awaiting readback, uses 256-byte row alignment, removes row padding, and encodes an SDR Display P3 image without DOM controls. It retains unresolved diagnostics. It does not certify convergence.

The [HTML popover contract](https://html.spec.whatwg.org/multipage/popover.html), [CSS anchors](https://developer.mozilla.org/en-US/docs/Web/CSS/Guides/Anchor_positioning), [subgrid](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_grid_layout/Subgrid), and [pointer events](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Properties/pointer-events) replace custom positioning and overlay behavior. Audio is not part of the current experience; Web Audio is not needed to describe or render this optical model.
