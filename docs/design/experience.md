# Controls and visual priorities

## Using the observatory

Choose a preset and edit physical placement, sources and detector settings. Startup plays at native pixels; restoring a preset or saved view pauses at its epoch. Pausing settles to 16 samples. Photograph freezes the epoch and accumulates 64 native samples before PNG export. [Resolution policy](../engineering/architecture.md#resolution-and-scheduling) defines fixed scales and opt-in Auto.

With canvas focus:

| Input               | Action                                                 |
| ------------------- | ------------------------------------------------------ |
| Drag / arrows       | Orbit placement, or turn the camera in free navigation |
| Wheel, pinch, + / − | Zoom in the active navigation mode                     |
| W/A/S/D, Q/E        | Translate along local camera axes in free mode         |
| Space               | Play/pause source time                                 |
| H                   | Toggle image-only mode                                 |
| Escape              | Restore controls or cancel ray selection               |
| R / Home            | Reset placement through scene validation               |

Navigation does not set physical observer velocity. Saved views and links retain inputs and epoch, not rendered pixels. Local audio stays on the device. Diagnostic colors are not emitted light.

## Composition

The warm view uses a shallow 6–14 M disk, a substantial silhouette, curved secondary images and approaching/receding brightness contrast. The blue view uses a broader 4–30 M atmosphere, spin 0.7 and extended bipolar jets. Material remains prescribed rather than fluid-simulated; more noise or bloom cannot supply missing fluid dynamics.

Six presets cover Classic disk, Blue accretion flow, Relativistic jets, Between horizons, Naked singularity and Thermal refraction. Additional variations belong in saved views.

[NPGS's volume shader](https://github.com/baopinshui/NPGS/blob/master/NPGS/Sources/Engine/Shaders/BlackHole_common.glsl) informed the separation of geometric and material sampling. No code or assets were adapted from this [GPL-3.0 project](https://github.com/baopinshui/NPGS/blob/master/LICENSE). Its artistic ring boosts and spectral-shift exponents are not Nullray's transfer law.

## Interface and acceptance

Keep the image dominant. Group physical controls separately from detector/display controls, show units beside values, and expose progress and missing weight during refinement. Native controls retain keyboard operation and focus restoration. Mobile panels must remain bounded and scrollable; primary touch targets should reach 44 CSS pixels. Respect reduced motion/transparency and forced colors.

Inspect desktop and narrow layouts with open panels, long values and keyboard focus. Review the warm/blue disk, jet/counterjet, interior/naked and plasma views. Record dimensions, epoch, source inputs, sample count and missing weight. [Validation](../validation/methods.md#image-and-interface-acceptance) defines the procedure; screenshots alone do not establish animated stability or physical-display HDR.

## Priorities and cost

1. Preserve the shadow, ordered source boundaries, curved disk images and frequency contrast. Fix missing work and discontinuities first.
2. Resolve and filter main-disk structure before adding fine noise. Check motion at fixed physical inputs.
3. Stabilize stellar footprints and measure critical-region flux error. Reference streaks are not automatically a correct lensing target.
4. Budget jets, broad glare and decoration after the main image. They must not hide numerical failures or consume its sampling budget.
5. Expand interior, polarization and plasma validation through explicit modes; avoid unmeasured diagnostic cost in ordinary frames.

[Image formation](../numerics/image-formation.md#where-approximation-belongs) identifies safe approximation boundaries; [coverage](../validation/coverage.md) lists unresolved physical and numerical work.
