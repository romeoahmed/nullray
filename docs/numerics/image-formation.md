# Image formation and approximation policy

Pixels integrate detector area and path emission. Production uses direct point rays, finite material cells and local stellar footprints; it does not propagate a complete beam.

Implementation: [material march](../../src/gpu/wgsl/imaging/matter.wgsl), [celestial pass](../../src/gpu/wgsl/passes/stars.wgsl), [stellar kernel](../../src/gpu/wgsl/sources/stars.wgsl), [sampling](../../src/gpu/imaging/sampling.ts), [history](../../src/gpu/wgsl/passes/accumulate.wgsl), and [bloom](../../src/gpu/imaging/bloom.ts).

## Path and material sampling

Source tables and [transfer laws](../physics/transport.md) are separate from trajectory tolerance.

The path pass integrates disk/jet I/Q/U at retarded emission events. A celestial pass adds filtered diffuse light and stellar footprints. Subpixel quadrature averages linear samples and retains missing weight.

Accepted [geometric steps](geodesics.md#discrete-integration) supply cubic Hermite dense output to the material march. Endpoint and middle Bernstein derivative coefficients bound the cubic coordinate rates; material support and structure set cell distances. Direction renormalization is not an enclosure. Turn clipping precedes empty-interval culling.

Each interval uses two ordered midpoint cells with analytic attenuation. Sampling stops once the remaining segment is outside all matter. Exceeding 512 intervals per geometric segment reports unresolved work. Transmission below $10^{-4}$ terminates the ray, neglecting remaining emission; absolute error still depends on background brightness.

Material cells account for radial/angular motion, retarded time and age-dependent shear. Local photon/Jacobian estimates map spans into the warped lattice; quintic slope bounds control octave filtering. Fine scales fade between half and two lattice cells, skipping absent octaves. Filter each cohort before blending. This is a line-footprint approximation, not exact cell-average opacity or pixel-beam integration. Zero disk structure bypasses disk noise.

Heating uses canonical Cartesian momentum with oblate position and null time. It owns boundary departure before conversion back to separated flow. Material scales constrain Hamiltonian steps; each accepted clipped step uses one reintegrated midpoint transfer cell, unlike the separated dense march.

## Celestial source integration

A deterministic hash generates a periodic 64³ scalar lattice. Material uses hardware interpolation with quintic coordinate warp; one-time sky generation interpolates eight neighbors explicitly in f32. Sky coefficients remain spectral until transport.

Each cube texel's area is the sum of two spherical-triangle solid angles. For unnormalized corners $a,b,c$, each triangle uses

$$
\Omega=2\operatorname{atan2}\!\left(|a\cdot(b\times c)|,
|a||b||c|+(a\cdot b)|c|+(b\cdot c)|a|+(c\cdot a)|b|\right).
$$

On a cube face, the triple product reduces to squared texel side length, avoiding cancellation between corner primitives. Mips area-weight four children in f32 before independent f16 rounding. A binary64 rectangular-area reference checks coefficients and flux.

A 10-bit Morton ordering partitions sources into a preorder tree with bottom-up bounds and stackless escape indices. Leaves retain f32 directions, spectra and duplicates; ordering keys do not quantize stored positions. This serial host partition follows the spatial-prefix idea of [Karras](https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees), not the parallel GPU builder.

For detector increments $x,y$ in a source tangent plane, let $p=0.65^2$ and $s=10^{-12}$. Stellar covariance is $C=p(xx^T+yy^T)+sI$. Its determinant and the Gaussian quadratic form are evaluated as nonnegative sums:

$$
\det C=p^2(x\times y)^2+sp(|x|^2+|y|^2)+s^2,
$$

$$
d^TC^{-1}d=\frac{p[(d\times x)^2+(d\times y)^2]+s|d|^2}{\det C}.
$$

The 2D cross product is a scalar determinant. These nonnegative forms retain nearly rank-one footprints. Gaussian normalization is $2\pi\sqrt{\det C}$; radius-3.5 support in normalized coordinates omits $e^{-3.5^2/2}\approx0.22\%$ of the ideal 2D Gaussian. Direct binary64 sums check tree queries within this local model.

Neighbors must share source domain/crossing order and have chord distance at most 1/8. Consistent aligned one-sided slopes form a central derivative; otherwise use the smaller admissible slope. Isolated endpoints use unlensed width. These gradients also filter the diffuse cube; branch checks do not establish critical-image completeness.

Apply the [arithmetic rules](geodesics.md#arithmetic-rules) to these filters.

## Sampling and missing weight

For a pixel-box integral, each sample retains radiance $L_j$ and validity $v_j\in\{0,1\}$. At $N$ completed samples,

$$
\widehat L_N=\frac1N\sum_{j=1}^Nv_jL_j,\qquad
m_N=\frac1N\sum_{j=1}^N(1-v_j).
$$

Keep denominator $N$, not the successful-ray count. Zero-filled missing intensity underestimates nonnegative light; $m_N$ measures missing weight, not its unknown flux. Histories use the f32 recurrence $\bar x_N=\bar x_{N-1}+(x_N-\bar x_{N-1})/N$, storing $m_N$ in alpha. Resolved output alpha is $1-m_N$.

| Mode          | Pixel quadrature                                           | Epoch and work                             |
| ------------- | ---------------------------------------------------------- | ------------------------------------------ |
| Live          | One center ray                                             | No history across changing epoch or camera |
| Settled image | Sixteen centers of a 4×4 stratification                    | Frozen epoch; stop after 16                |
| Photograph    | First 64 Halton points in bases 2 and 3, shifted by $-1/2$ | Frozen epoch; native raster; stop after 64 |

Stratified centers integrate affine pixel radiance exactly in exact arithmetic. Finite Halton sampling is deterministic quadrature, not an unbiased random estimator or exact affine rule. Neither guarantees hitting a narrow critical image. Normal UI diagnostics use one sample.

Scene, source appearance, epoch, raster, diagnostic and sampling changes reset history. Display/analyzer edits reuse samples. No temporal reprojection crosses camera/source motion. Antialiasing cannot recover detail absent from a [reduced raster](../engineering/architecture.md#resolution-and-scheduling).

Coverage reports affected pixels and missing weight separately. Per-pixel reduction uses $\operatorname{round}(16Nm_N)$ integer units, then sums on the CPU; current binary 16/64-sample counts are representable apart from mean rounding. Magenta marks missing work, not emission. Zero missing weight does not establish trajectory accuracy or freedom from [f16 saturation](../physics/transport.md#storage-and-missing-light).

## Bloom

Bloom uses a normalized downsample/upsample tent pyramid. Octaves have equal weight; a fractional outer octave varies continuously with span $\max(1,\log_2(0.08H))$. The filter operates on linear light before display mapping.

It preserves constant input and has interior point-flux checks. Clamped boundaries, resampling and f16 rounding prevent a general exact-flux claim. Bloom cannot resolve aliased arcs.

## Where approximation belongs

| Operation                          | Implemented approximation                                                          | Boundary to preserve                                                         |
| ---------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Smooth spectra and radial profiles | Linear lookup interpolation in log temperature/radius coordinates                  | Do not extrapolate unsupported temperatures as valid radiation               |
| Geometric segments                 | Cubic Hermite dense output                                                         | Search ordered events; never interpolate a destination label across a branch |
| Disk material                      | Two midpoint cells, exponential attenuation, filtered fine octaves                 | Preserve foreground opacity and unfinished material work                     |
| Noise lattice                      | Quintic coordinate warp plus texture interpolation                                 | Filtering is a line-footprint estimate, not a full pixel-beam integral       |
| Diffuse sky                        | Solid-angle-weighted mips and local direction gradients                            | Filter surface brightness, not a pre-rasterized stellar catalogue            |
| Point stars                        | Locally linear Gaussian footprint and stackless exact tree query within that model | Domain/order agreement does not prove nonlinear caustic completeness         |
| Detector pixels                    | 16/64 direct subpixel rays                                                         | Keep missing weights; no averaging integer domain identities                 |
| Broad glare                        | Reduced-resolution normalized pyramid                                              | Keep it separate from physical radiance transport                            |

Surface disk structure uses zero material footprint and relies on pixel sampling; jet noise lacks equivalent octave filtering. Address main-disk and critical-edge coverage before adding finer detail.

No optical reconstruction is active. A half-resolution guide with branch checks and direct fallback was removed after a local slowdown; agreement between guide neighbors also cannot prove absence of subpixel images. Any replacement must improve end-to-end cost at comparable image error and missing weight. [Bruneton](https://arxiv.org/abs/2010.08735), [DNGR](https://arxiv.org/abs/1502.03808) and [AART](https://arxiv.org/abs/2211.07469) offer different source/sampling approaches whose guarantees do not transfer automatically.
