# Architecture

## Ownership boundaries

| Location             | Responsibility                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------ |
| `src/physics/`       | Pure geometry, physical frames, timelike flow, spectra and catalogue preparation           |
| `src/scene/`         | Validated inputs, coupled preparation, camera/navigation, actions and schema-1 persistence |
| `src/gpu/optics/`    | Optical pipeline ownership, frame ABI and shader composition                               |
| `src/gpu/sources/`   | Catalogue upload and GPU generation of material noise and the spectral sky                 |
| `src/gpu/imaging/`   | Stokes measurement, photographic history, coverage, bloom and export                       |
| `src/gpu/wgsl/`      | Explicit geodesic, material, source and image-pass equations                               |
| `src/runtime/`       | Dedicated worker boundary, input revisions, clocks and bounded scheduling                  |
| `src/ui/`            | DOM, navigation gestures, local bookmarks and local media ownership                        |
| `tests/reference/`   | Binary64 metric, polarization, procedural-sky and convolution references                   |
| `tests/regressions/` | Concrete physical failures with reproducing inputs and provenance                          |

The domain core has no GPU handles, browser objects, clocks, persistence or random state. Readonly records describe values; numerical loops mutate local working buffers. Returned typed arrays belong to their caller, and a readonly property does not freeze their elements. External data enters as `unknown` and passes runtime decoding. Type assertions and symbol brands do not validate data.

## A scene is one coupled value

[Construction](../src/scene/scene.ts) validates spacetime, placement, camera, observer motion and source support atomically. [Preparation](../src/scene/preparation.ts) produces the physical endpoint, orthonormal frame, chart time, spacetime block and disk profile. Source boundaries use representable radii before either host or GPU evaluates them.

Free fall advances a timelike geodesic from the supplied launch event by proper time. A singular, unsupported or unfinished worldline is a construction error. The camera rotates the endpoint frame; navigation does not create physical velocity. Changing only camera axes or field of view reuses the worldline. Source-radius edits reuse the observer and rebuild the disk; spacetime or worldline edits invalidate their physical dependencies. Reuse is explicit through a previous validated scene, without global caches.

[Views](../src/scene/view.ts) serialize inputs and appearance, detector settings and emission epoch. Derived fields, numerical state and GPU resources are excluded. The same schema-1 decoder serves URL restoration and the local bookmark library. A rejected action preserves the complete accepted session and resynchronizes controls.

## The worker owns rendering

The UI sends complete input revisions through [the protocol](../src/runtime/protocol.ts), and the worker constructs its own physical scene. Only one GPU completion is outstanding. Coalesced inputs keep work bounded; revision checks reject stale frame, inspection and export results.

Canvas transfer is permanent. Remount and retry use a fresh element. An aborted or replaced initialization cannot publish a renderer, and partial acquisition is released with native resource stacks. GPU objects remain inside the worker. Visibility, playback clocks and photographic progress never enter physical preparation.

## Optical and detector passes

[Optics](../src/gpu/optics/engine.ts) owns pipelines, source buffers and the current image targets. Its [frame writer](../src/gpu/optics/frame.ts) packs the explicit WGSL ABI. Bind groups and texture views are constructed with their targets and reused until resize. Disk uploads follow prepared-profile changes. Jet spectra are cached by density, magnetic field, mass, electron cutoff and observing frequency.

The GPU generates a periodic material lattice and all six diffuse-sky faces. Separate mip passes integrate sky coefficients by solid angle in f32, then store the sampled cube in f16. Catalogue positions and spectra remain a CPU-prepared stackless tree. Its Morton keys only choose spatial order; source positions retain f32 precision and coincident sources remain separate leaves.

The path pass integrates regular geodesics and ordered material transfer. It outputs I/Q/U, radiance coverage, source domains, asymptotic directions and transmission. A celestial pass uses neighboring endpoints within the same image branch to filter the diffuse sky and query stellar flux. A failed path keeps zero resolved weight; darkness alone cannot identify capture or source exclusion.

Photographs freeze the epoch and accumulate I/Q/U and missing sample weights over the same 64-sample sequence. The polarimeter acts on retained Stokes images. Bloom redistributes linear radiance; presentation applies camera calibration and the display transform. Exposure, white balance, analyzer and bloom changes reuse the completed physical image. [Physics](physics.md#spectral-transfer-and-display) owns these transforms.

Each owner releases its textures and buffers explicitly. Resize replaces only size-dependent resources. Temporary sky work textures live through submitted generation; temporary readback buffers live through mapping and decoding. No GPU handle is serialized or stored in scene state.

## Interaction and inspection

Native controls dispatch actions and synchronize from accepted state. Popovers, dialogs, exclusive disclosures and keyboard focus use HTML behavior. CSS layers separate tokens/base styles, layout, controls and accessibility preferences. The local soundtrack is a native audio element; the application owns its object URL, while the browser owns playback, seeking, volume and activation.

Selected-ray inspection invokes the production integrator with one detector direction. The worker serializes it with frame changes. The UI receives accepted radial/polar states, chart-tagged time, crossing count and source-domain history. The topology view clips distant radii and brackets horizon intersections between samples; it is not a proper-distance embedding.
