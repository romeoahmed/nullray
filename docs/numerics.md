# Numerical method

The production solver evaluates separated Kerr–Newman trajectories in WGSL f32. It combines real elliptic expressions with bounded numerical quadrature; “semi-analytic” does not mean exact arithmetic or complete branch coverage. [Physics](physics.md) defines the model, and [Coverage](coverage.md) records its tested limits.

## Potentials and parameterization

Keep future-directed photon constants and use positive backward Mino time $\tau=\gamma_{\rm obs}-\gamma$. Let $K=(L_z-aE)^2+\mathcal C$. Expanding the physical radial potential gives

$$
R(r)=E^2r^4+(a^2E^2-L_z^2-\mathcal C)r^2+2Kr-a^2\mathcal C-q^2K.
$$

The production radial coordinate is $u=1/r$, so infinity is a finite endpoint:

$$
\left(\frac{du}{d\tau}\right)^2
=P(u)=E^2+Bu^2+2Ku^3-Du^4,
\quad B=a^2E^2-L_z^2-\mathcal C,
\quad D=a^2\mathcal C+q^2K.
$$

For $\mu=\cos\theta$,

$$
\left(\frac{d\mu}{d\tau}\right)^2
=\mathcal C+B\mu^2-a^2E^2\mu^4.
$$

These forms remove divergent radius and explicit polar cotangents from trajectory evaluation. They do not regularize every azimuth chart. Neither divides by Killing energy; zero or negative $E$ remains meaningful. Initial velocities and turning-point continuation follow the backward convention throughout.

## Elliptic paths

The [scalar kernel](../src/render/shaders/elliptic.wgsl) evaluates the Biermann–Weierstrass quartic initial-value solution through real Jacobi functions. The construction follows [Cieślik, Hackmann, and Mach](https://arxiv.org/abs/2305.07771); Nullray applies the polynomial identity to its own charged radial and polar potentials. A binary64 implementation lives only in `tests/reference/`.

Path preparation retains polynomial derivatives, elliptic invariants, real-root representation, initial position, and orientation. Reduced-degree and zero-invariant limits have explicit branches. A turning point uses the acceleration $f'(x)/2$; an exactly repeated root is not forced through a fictitious simple turn.

Jacobi functions use the parameter $m=k^2$. AGM preparation is reused within scalar transport and differentiated paths. Carlson $R_F$ supports inverse phase evaluation; [DLMF 19.36](https://dlmf.nist.gov/19.36) describes symmetric-integral duplication. The kernel evaluates $\operatorname{dn}^2=(1-m)+m\operatorname{cn}^2$ to avoid subtractive cancellation near $m=1$. Domain and iteration failures stay explicit.

Small polynomial residuals do not establish root-position accuracy near repeated roots. Scaling, phase continuation, endpoint order, and independent reference checks are all necessary. The existing solver still has critical-ray precision counterexamples.

## Radial arrival and event ordering

[Arrival](../src/render/shaders/arrival.wgsl) first selects the accessible radial interval. The nonzero interior extrema of $P$ satisfy $B+3Ku-2Du^2=0$. A negative minimum establishes a turning barrier; a near-zero minimum within the kernel's term-scaled rounding uncertainty remains unresolved. Zero initial velocity uses acceleration for orientation.

For a prepared quartic $x'^2=f(x)$, define $d=x_b-x_0$, $v_0=x'(0)$, $v_b=x'(\tau_b)$, and $H=f(x_0)+v_0v_b+f'(x_0)d/2$. The addition identity gives

$$
\wp(\tau_b)=\frac{H}{2d^2}+\frac{f''(x_0)}{24},
\qquad
\wp'(\tau_b)=\frac{[v_0f'(x_b)+f'(x_0)v_b]d-4Hv_b}{4d^3}.
$$

The real Jacobi inverse and the sign of $\wp'$ select a phase seed. Near a half-period minimum, a local derivative expansion supplies the seed. The inverse is not accepted on its own: a bounded bracket expansion and bisection compare it with the forward path. Both bracket endpoints survive.

Disk crossings must precede the lower boundary-time bound. A crossing inside its uncertainty interval is unresolved. Sky evaluation uses the midpoint only after a 4e−5-radian phase-width check. A later uncertain horizon cannot invalidate an earlier disk hit already established. The frozen scan in the fixtures is a comparison method, never a runtime fallback.

For positive Carter constant, equator events use

$$
A=a^2E^2,\quad b=L_z^2+\mathcal C-A,\quad d=\sqrt{b^2+4A\mathcal C},
\quad \mu=\sqrt z\,\operatorname{cn}(\psi_0+\nu\tau,m),
$$

$$
z=\frac{2\mathcal C}{b+d}\ (b\ge0),\qquad
z=\frac{d-b}{2A}\ (b<0),\qquad
\nu=\sqrt d,\quad m=Az/d.
$$

At $L_z=0$, use $z=1$ exactly. The initial signed polar velocity supplies the Jacobi sine; `atan2` recovers its phase without the near-pole loss of `acos(cn)`. Equator zeros are spaced by $2K(m)/\nu$. Nonpositive $\mathcal C$ has no transverse equator crossings. Exactly coplanar overlap follows the explicit [surface convention](physics.md#disk-visibility), not an injected thickness.

## Azimuth and emission time

[Transport](../src/render/shaders/transport.wgsl) integrates rates along an already prepared path; it does not numerically step the geodesic. It combines the exterior radial azimuth terms before evaluation and omits the divergent coordinate time for sky rays.

For positive $\mathcal C$, write $\mu=\sqrt z\operatorname{cn}(v,m)$ and

$$
\epsilon=1-z=\frac{L_z^2z}{\mathcal C+a^2E^2z}.
$$

The unwrapped primitive $F(v)=\operatorname{atan2}(\operatorname{sn}v,\sqrt\epsilon\operatorname{cn}v)$ has derivative $\sqrt\epsilon\operatorname{dn}v/(1-\mu^2)$. Subtracting its analytic contribution leaves the regular polar rate

$$
r_{\rm pol}=\frac{L_zm\operatorname{sn}^2v}{(1+\operatorname{dn}v)(1-\mu^2)}.
$$

Evaluate the coefficient without dividing by $L_z$. At zero axial momentum the coordinate azimuth jumps by $\pi$ at a pole; the Cartesian direction stays continuous. Exact axis launches use the departing meridian and a right-continuous primitive. Nonpositive-C chart limits remain restricted.

For exact $a=0$ and positive-C motion, both frame dragging and this residual vanish. Sky azimuth is therefore the analytic primitive alone, for charged as well as uncharged spherical spacetimes. The production shader skips the zero quadrature; disk travel time still needs integration.

The remaining rates use composite Simpson refinement after $\tau=\tau_b\sin^2(\pi v/2)$. A prepared immutable kernel shares the path and AGM data. Reused trapezoidal sums avoid reevaluating old nodes; four interleaved accumulators shorten each scalar summation chain. Disk/time work permits 512 subintervals and sky-only work 1024. Phase-based initial grids reserve two refinement levels. The nominal estimates are 2e−5 absolute for azimuth and 2e−5 times $1+|\Delta t|$ for time; these are quadrature estimates, not complete endpoint error bounds. Exhaustion is unresolved.

## Sky derivatives

A locally smooth sky map needs derivatives with respect to physical screen pixels. Ordinary pixels use compatible same-branch neighbors, including a rank-checked two-dimensional stencil. [Forward differentiation](../src/render/shaders/differential.wgsl) repairs missing derivatives with one semi-analytic ray, carrying a value and two derivatives per scalar. It avoids finite differences between nearly equal endpoints without increasing primal f32 precision.

The differentiated quartic evaluator cancels the Weierstrass pole algebraically before evaluation. For a verified sky endpoint, $u(\tau_b,p)=0$ and $u'(\tau_b)=-E$ imply

$$
\partial_p\tau_b=\frac{\partial_pu}{E}.
$$

This division occurs only after establishing $E>0$ and sky destination. It is not global energy normalization. The positive-C polar primitive is differentiated with the signed root $L_z\sqrt{z/(\mathcal C+a^2E^2z)}$, avoiding an absolute-value derivative at zero momentum. At fixed $a=0$, its residual and screen derivatives vanish exactly.

In a regular endpoint chart, the source solid-angle Jacobian is

$$
J=|\mu_x\phi_y-\mu_y\phi_x|.
$$

Differentiated cosine-mapped Simpson quadrature compares all integrated components and the determinant change. It integrates the determinant's linear contraction alongside the components, shortening cancellation-prone operations. Four interleaved sums reduce serial rounding depth. It permits 2048 subintervals and an initial grid no larger than 256, requiring 1e−4 relative determinant agreement and component agreement within $10^{-5}(1+|v|)$. Endpoint poles, unsafe arithmetic, wrong branches, and exhausted estimates remain unresolved.

Stable differences between two resolved sky endpoints use rationalized latitude differences and trigonometric addition identities in [beam.wgsl](../src/render/shaders/beam.wgsl). They reduce arithmetic cancellation in neighbor stencils but cannot restore precision already lost in endpoint transport.

## Pixel integration

Beam repair and failure classification test the source solid-angle area after scaling both derivative vectors by their largest absolute component. The condition is $\sqrt{|n\cdot(\hat d_x\times\hat d_y)|}>10^{-10}/s$, with $d_i=s\hat d_i$; zero scale is rank deficient. This retains the existing $J>10^{-20}$ criterion without forming an overflowing squared scale or cross product.

The distant sky combines a solid-angle-weighted diffuse spectral mip chain with individual point sources. [The stellar filter](../src/render/shaders/stars.wgsl) applies a unit-integral tent in local image coordinates and divides source flux by $J$. This supplies lensing amplification once. It is a local affine approximation; it does not enumerate or integrate all images across a caustic. A folded-map test demonstrates the need for increasing spatial resolution.

Exploration selects mixed endpoint/image-order boundaries and varying disk pixels for 4 × 4 quadrature. Selection accounts for radial/azimuthal source variation, frequency, and retarded emission time. Every selected subpixel carries weight 1/16, including failures. Rank-deficient sky subpixels use a separate differentiated-ray queue. Refinement, queue overflow, and radiation failures remain distinct diagnostics.

The primary derivative queue has at most 4096 candidates. The boundary budget is $\min(WH,\max(32768,\min(65536,8(W+H))))$; its derivative budget is $\operatorname{clamp}(3(W+H),4096,16384)$, limited further by atlas size. These are bounded work/memory policies. Narrow features can lie between all primary samples, and fixed quadrature does not certify convergence near critical curves.

Photographs average 64 jittered samples in one read/write `rgba32float` history. Optical radiance alpha records the resolved fraction; history alpha records the averaged missing weight. The coverage pass reconstructs integer sixteenth-sample units, reduces them in subgroups/workgroups, and sums blocks on the CPU to avoid whole-image u32 overflow. It never rounds partial quadrature failures to whole samples or renormalizes away failed samples.

Frequency and image-order inspection operate on resolved endpoints and retain zero coverage for failed geometry. Frequency inspection uses $-\log_2 E$ for the sky and $\log_2 g$ for the disk. Diagnostic color never changes an unresolved endpoint into a successful photographic sample.

## Radiance filtering

Bloom convolves linear radiance with a separable $[1,2,1]/4$ tent at each pyramid level, then averages fine and reconstructed coarse levels. At fractional texel phase $f$, linear interpolation expands the one-dimensional coefficients to $[1-f,2-f,1+f,f]/4$. Pairing adjacent coefficients gives weights $(3-2f)/4$ and $(1+2f)/4; the two-dimensional product needs four hardware bilinear samples instead of nine. The pairing remains valid at arbitrary pixel phases and clamped image edges, including odd-sized and one-pixel targets. A direct binary64 convolution reference checks the implementation with binary16 rounding between levels. This is prescribed display scattering, not an additional gravitational amplification.

## Precision and failure policy

| Quantity                                      | Representation                                 |
| --------------------------------------------- | ---------------------------------------------- |
| CPU preparation and references                | ECMAScript binary64 numbers                    |
| Optical geometry, transport, derivatives      | WGSL f32                                       |
| Endpoint coordinates and photographic history | 32-bit texture components                      |
| Linear radiance and diffuse coefficients      | Half-float textures, with range checks/scaling |
| Work counts, queue indices, coverage          | Integer GPU values; binary64 host totals       |

Guard divisors, radicands, domain restrictions, and magnitudes before unsafe arithmetic. WGSL finite-math rules do not promise a useful NaN signal, and `select` is not lazy. Do not use f16 for optical paths. WGSL permits reassociation and an unfused `fma`; conventional error-free two-float transforms are therefore not a portable precision foundation. See [WGSL floating-point evaluation](https://www.w3.org/TR/WGSL/#floating-point-evaluation).

### Expression context and derivative ownership

Derivative helpers receive an explicit caller-owned failure flag. A rejected arithmetic operation marks that evaluation unresolved without contaminating independently owned evaluations. A pure value/status return trial lost an arrival gradient when composed with polar motion on Apple M5/macOS 26.6.2. Native replay of Dawn-generated Metal reproduced the loss with fast trigonometric functions; precise sine or cosine restored it. This historical investigation localized the observed backend behavior, without identifying a compiler optimization pass. Its generated replay directory is not a checked-in reference. The durable acceptance checks are [derivative ownership](../tests/gpu/differential.test.ts), [sky Jacobians](../tests/gpu/sky-jacobian.test.ts), and [beam repair](../tests/gpu/beam-repair.test.ts).

The [launch-roundoff fixture](../tests/fixtures/launch-roundoff.json) separately records a one-ULP Carter change after extracting launch intermediates into a returned struct. Complete execution changed from escape to unresolved disk transport while truncated probes agreed. This is expression-context sensitivity, not proof of a compiler defect. Launch expressions therefore remain inside the pure trace function. Treat future function extraction, expression reassociation, or precision changes as numerical work requiring complete GPU execution against the retained references.

Capture, escape, disk emission, invalid input, and unresolved numerical work are separate outcomes. Diagnostic stage codes help minimize failures; they are not physical destinations. Near-repeated roots, long phases, endpoint ties, nonpositive-C axis transport, and nonlinear stellar mapping require further acceptance work. The [critical-primary fixture](../tests/fixtures/critical-primary-rays.json) retains a known sky-vector error around 0.00575. A passing test suite does not erase that gap.
