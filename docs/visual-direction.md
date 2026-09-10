# Visual design

The photograph is the primary surface. Controls should make the physical choices legible while leaving room to see their consequences. Numerical and visual acceptance are complementary: an attractive image cannot establish a correct ray branch, and a passing invariant cannot establish readable controls.

## Scene composition

The warm reference family emphasizes a shallow luminous disk crossing a substantial dark silhouette, a fine inner arc, curved secondary images and an approaching/receding brightness difference. The default 6–14 M annulus and subdued sky serve that composition. Visible arcs must be interpreted from ray destinations; brightness alone does not establish a resolved photon subring.

The blue reference family opens the disk and adds irregular outer material and extended bipolar jets. Its 4–18 M atmosphere, spin 0.7, spectral temperature and camera calibration form a separate view. Concentrated roots, broader sheaths and soft boundaries keep the jets from reading as uniform shafts. The silhouette remains a compositional anchor.

Six presets cover distinct behavior: Classic disk, Blue accretion flow, Relativistic jets, Between horizons, Naked singularity and Thermal refraction. Each has a short description. More variations belong in user-saved views, not a growing preset list.

## Material and display

The [physical model](physics.md) combines a tapered Gaussian atmosphere, gray absorption, circular emitter motion and polarized Milne emission. Large clouds, finer filaments and vertical displacement have distinct roles. Retarded-time advection uses finite-lived cohorts. Jets are a prescribed synchrotron outflow, and the Galactic background combines a synthetic spectral cube with catalogue point sources.

The study of [NPGS's volume shader](https://github.com/baopinshui/NPGS/blob/master/NPGS/Sources/Engine/Shaders/BlackHole_common.glsl) informed the separation between geometric steps and material sampling. Its artistic ring boosts and spectral-shift exponents are not part of Nullray's transfer law. NPGS is [GPL-3.0](https://github.com/baopinshui/NPGS/blob/master/LICENSE); no code or assets are adapted from it.

Linear radiance, physical frequency shift, camera white balance and display compression remain separate. A common highlight shoulder retains color direction and approaches neutral at high exposure. Normalized bloom broadens glare without creating light. An image mask, altered source-domain identity or exposure adjustment cannot repair a wrong trajectory.

## Interface hierarchy

The masthead identifies the observatory and gives access to Settings and About. The lower toolbar carries exploration, refinement, sharing and saved views. Quiet telemetry reports the physical parameters and rendered resolution. A refinement card shows progress, export availability and unresolved sample coverage when relevant.

Settings group observer placement/motion, light, jets, image formation, local audio, plasma and inspection. Technical explanations stay beside the choice they clarify. Units distinguish radians/degrees, geometrized distance/time, GHz, kelvin and local velocity. Navigation and physical observer motion remain separate controls.

Dark translucent surfaces, restrained cyan accents and tabular numerical text provide a compact observatory character. Contrast and spacing carry hierarchy; decorative effects must not compete with the image. Native controls retain keyboard operation and focus restoration. Touch targets remain at least 44 CSS pixels. Small screens use bounded, scrollable panels, safe-area insets and a compact toolbar. Reduced motion, reduced transparency and forced colors have explicit styles.

## Visual acceptance

Inspect actual production captures at desktop and narrow/mobile sizes. Check panel bounds, long labels, numerical values, focus, feedback placement, open disclosures, preset cards and the photograph state. Review warm and blue images, jet/counterjet, a horizon/interior case and the refractive view. Verify that the shadow, source edges, stars and fine material remain continuous at the chosen sampling level.

Record dimensions, source inputs, epoch, sample count and missing weight alongside exported images. SDR exports are Display P3 PNGs. A screenshot or SDR export does not validate HDR headroom on a physical display. Still images do not certify animated stability, critical-image positions or nonlinear stellar flux; those remain in [Coverage](coverage.md).
