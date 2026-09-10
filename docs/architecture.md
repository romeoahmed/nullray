# Architecture

Nullray separates user intent, physical preparation, GPU execution, and browser interaction. The renderer runs in a dedicated worker. There is no application framework, shader DSL, or runtime dependency.

## Module boundaries

| Location           | Owns                                                                                                      |
| ------------------ | --------------------------------------------------------------------------------------------------------- |
| `src/scene/`       | Validated scenes, camera placement, source appearance, view codec, presets, immutable session transitions |
| `src/physics/`     | Metric and neutral circular orbits, disk flux, spectra, celestial data, vector and photon calculations    |
| `src/gpu/`         | Device resources, optical passes, accumulation, bloom, coverage, PNG encoding                             |
| `src/gpu/wgsl/`    | Explicit WGSL formulas and compute/render entry points                                                    |
| `src/runtime/`     | Typed client/worker protocol, OffscreenCanvas, renderer lifetime, clock, bounded scheduling               |
| `src/ui/`          | DOM controls, pointer/keyboard navigation, local bookmarks, optional local audio, CSS                     |
| `src/data/`        | Attributed CIE and HYG source data                                                                        |
| `tests/reference/` | Binary64 separated solver and independent Hamiltonian integrator; never imported by production            |
| `benchmarks/`      | Production preparation and GPU workloads                                                                  |

`scene` and `physics` have no DOM, GPU handles, clock, randomness, or persistence. The DOM sends intent across [the render protocol](../src/runtime/protocol.ts); it does not manipulate GPU resources. The physics modules produce deterministic data from explicit inputs. Small immutable records express state; direct loops and local mutation remain appropriate for numerical arrays. Fresh typed arrays transfer ownership to their caller; readonly record fields prevent reassignment but do not freeze array elements. GPU encoders and DOM bindings are explicit effect boundaries, not domain operations. WGSL trace functions return values, while derivative arithmetic helpers mutate only a caller-owned failure flag. Cooperative sky transport stores the prepared path and quadrature scratch in workgroup memory owned by one ray; no workgroup shares another ray's state.

WGSL modules are grouped by mathematical role: `math/` owns arithmetic and special functions, `geodesics/` owns launch and ordered ray transport, `lensing/` owns critical curves and source-image search, `imaging/` owns detector integration and radiation, and `passes/` owns entry points. [Shader assembly](../src/gpu/shaders.ts) makes dependencies explicit with raw imports; modules remain directly reviewable WGSL.

## State and validation

[The initial view](../src/scene/view.ts) owns shared scene, source, and display defaults used by presets and session initialization. [Session](../src/scene/session.ts) combines a saved view, motion (`playing`, `paused`, or `refining`), and exploration resolution. Its exhaustive action union produces either a complete new session or a validation error. Invalid coupled parameters cannot partially change a scene.

[Scene construction](../src/scene/scene.ts) validates the coupled spacetime, exterior observer, camera axes, and relevant f32 rounding before deriving disk geometry. Camera-only updates reuse the previous spacetime's ISCO. This is an explicit input dependency, not a global cache. Symbols mark constructed values inside one realm; structured clone does not preserve that brand, so the worker reconstructs scenes and appearances.

URLs and local storage are untrusted input. The [view codec](../src/scene/view.ts) parses them as `unknown` and validates every field before restoring anything. The sole format is version 1; every field is required and there are no migration branches. A snapshot contains source inputs and emission epoch, not GPU handles or derived textures. Restoring a view pauses motion. The browser-local library limits names and collection size, rejects duplicates, preserves corrupt storage, and offers one deletion undo.

## Worker and resource lifetime

The application transfers a fresh canvas once. Remounting replaces that element because an OffscreenCanvas transfer cannot be reversed. Vite updates to the application module preserve its current session; a full page reload restores the URL view.

The client coalesces visible control changes into one animation-frame message. Hidden or zero-sized surfaces are sent immediately. Monotonic revisions reject obsolete frame and export responses. Geometry and appearance are cloned only when their identity changes; presentation controls travel separately.

The worker keeps at most one submitted frame outstanding. Incoming messages replace pending intent while the GPU finishes. It schedules another frame only for changed input, animation, or an unfinished photograph. Hidden and zero-sized surfaces stop scheduling. Animation advances ten geometric time units per second, caps each elapsed increment at 0.25 seconds, and resets its clock baseline across pause and visibility changes.

`DisposableStack` gives each device, render target, and temporary readback a clear owner. A lexical `using` stack unwinds resources on initialization failure, then `move()` transfers successful initialization to the returned owner. Application mounts use the same rule for event listeners, resize observation, the render client, and photograph URLs. Disposal aborts initialization, stops scheduling, destroys resources, and closes the worker. Device loss exposes retry with the same scene. A crashed worker requires a page reload; retry first preserves the current view in the URL. Async frame/export results are discarded after replacement or disposal.

[OffscreenCanvas](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas) and [dedicated workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API) provide the execution boundary. Moving scheduling off the main thread improves input isolation; it does not shorten a GPU optical solve.

## Render path

1. Search and refine point-source images in a curved critical strip; bin confirmed images into detector tents.
2. Trace primary rays to their first physical endpoint.
3. Find sky pixels without usable same-branch derivatives; repair each retained ray with a cooperative workgroup.
4. During exploration, select mixed boundaries and varying disk pixels, trace a bounded 4 × 4 subpixel atlas, and repair its missing sky derivatives.
5. Evaluate disk and sky spectra from the retained endpoints and source controls; complete primary sky neighborhoods enumerate normalized bilinear stellar cells.
6. For photography, accumulate 64 jittered samples in binary32 history at native resolution.
7. Filter linear radiance through normalized multiscale bloom and present it in Display P3.

Primary and boundary sampling share the mixed-precision selection policy described in [Numerics](numerics.md#precision-and-failure-policy). A separate owned 96-byte uniform carries twelve binary64 observer metric values from the quantized frame. Selected rays prepare their camera launch, radial classification, and elliptic path with integer arithmetic before f32 event evaluation. The rounded path retains the elliptic parameter and its complement independently. Polar preparation follows the active analytic or general-quartic representation, so an unused general polar path is not solved or differentiated. Derivative repair uses the same selector and observer uniform. Selected derivatives are prepared in integer binary64, transported in a fixed orthonormal screen basis, and rotated back after Cartesian evaluation. Primary and boundary repair share one prepared-beam evaluation function that owns the 2048/16384 subdivision policy. The retained critical checks pass; complete critical-domain consistency remains unproven.

The stellar search owns its source data, critical columns, finite candidate/refinement buffers, and work counters. A coarse classification pass queues mixed cells before indirect midpoint sampling. Five midpoint records per possible coarse cell reserve 5 MiB at the default grid; two cell-index arrays retain queue membership and shared-edge ownership. Coarse transition cells and subdivided parents query the source tree after that sampling pass. A per-cell count, 256-lane integer prefix scan, and deterministic write pass allocate candidate ranges. The default two-word prefix storage is 257 KiB; overflow keeps the same canonical subset on repeated searches. Two alternating active-index queues use about 256 KiB at the default capacity. Up to 32 indirect preparation/transport stages consume unfinished candidates while retaining canonical per-candidate state between Newton iterations. The storage boundary uses a 1456-byte record per candidate, including the differentiated path, geometry derivatives, endpoint, preceding step, binary64 critical-circle/offset and centered-orbit words, and status; the default 32768 candidates allocate 45.5 MiB for these records. Each raster owns a head per pixel and four linked-list nodes per candidate. Physical image results are cached across detector jitter; pixel weights are rebuilt without retracing those images. The same image-position partition filters ordinary stellar contributions before the critical contribution is added after boundary averaging. Resize destroys the old raster lists with the other optical targets.

The numerical meaning and limitations of these passes belong in [Numerics](numerics.md#pixel-integration). No pass converts exhausted work into capture or successful radiation.

Photography owns one read/write `rgba32float` history texture and one `rgba16float` output texture: 24 bytes per pixel, excluding optical targets. Each compute invocation reads and replaces its own history texel; ordered passes provide visibility between samples. This removes a second binary32 history allocation (126.6 MiB at 3840 × 2160). History stores mean linear radiance and missing sample weight. Coverage reduces integer sixteenth-sample units so partially resolved 4 × 4 quadrature is retained. Sample indices run sequentially from 0 through 63; index zero starts a new sequence, including after resize.

| Change                                | Recomputed work                                               |
| ------------------------------------- | ------------------------------------------------------------- |
| Camera, spacetime, raster dimensions  | Geometry, derivative queues, radiation, history, bloom        |
| Emission time or source appearance    | Radiation and history; exploration geometry stays cached      |
| Exposure, bloom strength, HDR mode    | Presentation; completed photographic samples remain available |
| New photographic sample               | Jittered geometry, radiation, accumulation, bloom             |
| Exploration resolution while refining | No photographic resize; photography remains native            |

The render worker builds source maps on the CPU and uploads them once per renderer. `createOptics` also accepts prepared celestial data for controlled production-render acceptance; it borrows those CPU buffers until initialization resolves and copies them into owned GPU resources. The caller retains the CPU arrays. Star traversal uses a stackless 32-byte node: internal nodes carry bounds and an escape index; leaves reuse those lanes for direction and spectral data. The maximum catalogue size keeps indices exactly representable in f32. Queues retain bounded capacities and explicit overflow status. Workgroup dispatch order never determines physical ray order.

## Interaction and presentation

Static semantic markup lives in [index.html](../index.html). Settings, Views, and About use native popovers with CSS anchor positioning. Native details group secondary controls; subgrid aligns labels and values. The main view remains available for navigation, with noninteractive overlays using `pointer-events: none` and actual controls opting back in. CSS uses OKLCH, 44-pixel button targets, visible focus, dynamic viewport units, safe-area insets, and reduced-motion/transparency preferences.

Orbit mode turns observer position around the center; drag and arrows turn, wheel/pinch approaches. Free mode rotates the camera's own frame and translates with WASD/QE. Translation uses an explicit spherical placement chart, carries orientation between local bases, and returns through scene validation. It is not a spatial isometry, Kerr–Schild transform, or physical observer worldline. Navigation never adds aberration or Doppler velocity.

Photography freezes emission time. Coverage reports unresolved sample weights as well as affected pixels. PNG export submits a snapshot before awaiting readback, uses 256-byte row alignment, removes row padding, and encodes an SDR Display P3 image without DOM controls. It retains unresolved diagnostics. It does not certify convergence.

The [HTML popover contract](https://html.spec.whatwg.org/multipage/popover.html), [CSS anchors](https://developer.mozilla.org/en-US/docs/Web/CSS/Guides/Anchor_positioning), [subgrid](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_grid_layout/Subgrid), and [pointer events](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Properties/pointer-events) replace custom positioning and overlay behavior. Optional local audio uses a streaming media element connected once to a lazily created Web Audio gain graph. The play gesture starts both context resumption and media playback. Revision checks reject stale completion, volume changes use a short gain ramp, replacement revokes the preceding object URL, and disposal closes the context. Audio has no connection to the optical clock or saved view; no soundtrack is bundled.
