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

The [scalar kernel](../src/gpu/wgsl/math/elliptic.wgsl) evaluates the Biermann–Weierstrass quartic initial-value solution through real Jacobi functions. The construction follows [Cieślik, Hackmann, and Mach](https://arxiv.org/abs/2305.07771); Nullray applies the polynomial identity to its own charged radial and polar potentials. The independent binary64 reference lives in `tests/reference/`; selected production paths use the integer arithmetic described below.

Path preparation retains polynomial derivatives, elliptic invariants, real-root representation, initial position, and orientation. Reduced-degree and zero-invariant limits have explicit branches. A turning point uses the acceleration $f'(x)/2$; an exactly repeated root is not forced through a fictitious simple turn.

Jacobi functions use the parameter $m=k^2$. AGM preparation is reused within scalar transport and differentiated paths. Carlson $R_F$ supports inverse phase evaluation; [DLMF 19.36](https://dlmf.nist.gov/19.36) describes symmetric-integral duplication. The kernel evaluates $\operatorname{dn}^2=(1-m)+m\operatorname{cn}^2$ to avoid subtractive cancellation near $m=1$. The prepared path preserves the complement $1-m$ separately, including when the rounded parameter equals one. Mixed-precision preparation computes that complement before rounding; reconstructing it from rounded $m$ loses critical phase accuracy. Domain and iteration failures stay explicit.

Exactly quadratic families avoid differentiation through a coalescing invariant-cubic pair. For $c_3=c_4=0$, preparation uses $e_3=c_2/12$, scale $-c_2/4$, and $m=0$ when $c_2<0$; for $c_2>0$, it uses $e_3=-c_2/6$, scale $c_2/4$, and $m=1$. The differentiated branch additionally requires both gradients of $c_3$ and $c_4$ to vanish. A transverse perturbation does not inherit this constant-modulus limit.

The scalar and differentiated motion kernels evaluate the corresponding elementary solutions directly, consistent with the [Jacobi limiting functions in DLMF 22.5(ii)](https://dlmf.nist.gov/22.5.ii). Write $A=f'(x_0)/f''(x_0)$, $\omega=\sqrt{|f''(x_0)|/2}$, and $B=v_0/\omega$. The harmonic solution is $(x_0-A)+A\cos(\omega\tau)+B\sin(\omega\tau)$; the hyperbolic solution replaces the cosine and sine by their hyperbolic forms. Near zero phase, half-angle squares preserve the small increment from $x_0$. The existing magnitude limits guard hyperbolic evaluation before exponentiation. [GPU checks](../tests/gpu/elliptic.test.ts) compare values and amplitude/frequency derivatives to the elementary functions over both families and reject a transverse cubic perturbation.

Small polynomial residuals do not establish root-position accuracy near repeated roots. Scaling, phase continuation, endpoint order, and independent reference checks are all necessary. The retained critical endpoint and derivative gates pass; those finite corpora do not establish the complete critical domain.

## Radial arrival and event ordering

[Arrival](../src/gpu/wgsl/geodesics/arrival.wgsl) first selects the accessible radial interval. The nonzero interior extrema of $P$ satisfy $B+3Ku-2Du^2=0$. A negative minimum establishes a turning barrier; a near-zero minimum within the kernel's term-scaled rounding uncertainty remains unresolved. Zero initial velocity uses acceleration for orientation. Selected paths classify extrema using integer binary64 coefficients before rounding. An extremum within $2^{-48}$ times its absolute polynomial term sum remains unresolved; extra precision does not replace the ambiguous-root guard.

For a prepared quartic $x'^2=f(x)$, define $d=x_b-x_0$, $v_0=x'(0)$, $v_b=x'(\tau_b)$, and $H=f(x_0)+v_0v_b+f'(x_0)d/2$. The addition identity gives

$$
\wp(\tau_b)=\frac{H}{2d^2}+\frac{f''(x_0)}{24},
\qquad
\wp'(\tau_b)=\frac{[v_0f'(x_b)+f'(x_0)v_b]d-4Hv_b}{4d^3}.
$$

The real Jacobi inverse and the sign of $\wp'$ select a phase seed. Near a half-period minimum, a local derivative expansion supplies the seed. Its rounding estimate sums the magnitudes of the terms in $H$ before cancellation; using the magnitude of their sum underestimates uncertainty for distant observers. The inverse is not accepted on its own: a bounded bracket expansion and bisection compare it with the forward path. Both bracket endpoints survive. The twelve bisection iterations stop early only when no binary32 midpoint lies strictly between them. A four-spacing early stop left enough phase error to stall a retained branch-8 stellar image; narrowing to adjacent representable times recovers it without changing the source-residual or flux gates. This does not remove uncertainty from the rounded forward path.

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

[Transport](../src/gpu/wgsl/geodesics/transport.wgsl) integrates rates along an already prepared path; it does not numerically step the geodesic. It combines the exterior radial azimuth terms before evaluation and omits the divergent coordinate time for sky rays.

For positive $\mathcal C$, write $\mu=\sqrt z\operatorname{cn}(v,m)$ and

$$
\epsilon=1-z=\frac{L_z^2z}{\mathcal C+a^2E^2z}.
$$

The unwrapped primitive $F(v)=\operatorname{atan2}(\operatorname{sn}v,\sqrt\epsilon\operatorname{cn}v)$ has derivative $\sqrt\epsilon\operatorname{dn}v/(1-\mu^2)$ with respect to $v$. Write $\nu=dv/d\tau$ and

$$
w=\sqrt{1+m\epsilon/z},\qquad
\nu\sqrt\epsilon=|L_z|/w.
$$

The [Jacobi identity](https://dlmf.nist.gov/22.6.E1) $\operatorname{dn}^2v=1-m\operatorname{sn}^2v$ gives

$$
1-\mu^2=\frac z m(w^2-\operatorname{dn}^2v),\qquad
\frac{L_z}{1-\mu^2}
=\frac{L_z\operatorname{dn}v}{w(1-\mu^2)}
+\frac{L_zm}{zw(w+\operatorname{dn}v)}.
$$

The first term integrates to $\operatorname{sgn}(L_z)F(v)$ with unit coefficient. The second is the smooth polar residual used by both scalar and differentiated transport. The implementation evaluates $w$ from $1+m\epsilon/z$ and never divides by $m$; the displayed factorization establishes the identity for nonzero $m$ and its implemented residual has a continuous zero-$m$ limit. The preceding subtraction used a larger coefficient and left narrow features in the residual's angular-momentum derivative near poles. On a retained near-polar image it exhausted 16384 subdivisions; the smooth form meets the unchanged derivative gates by 1024 nodes.

No coefficient divides by $L_z$. At zero axial momentum the coordinate azimuth jumps by $\pi$ at a pole; the Cartesian direction stays continuous. Exact axis launches use the departing meridian and a right-continuous primitive. Differentiation uses the signed root $L_z/(\nu w)$ to retain the transverse derivative at zero momentum. Nonpositive-C chart limits remain restricted.

For exact $a=0$ and positive-C motion, both frame dragging and this residual vanish. Sky azimuth is therefore the analytic primitive alone, for charged as well as uncharged spherical spacetimes. The production shader skips the zero quadrature; disk travel time still needs integration.

Sky directions interpret azimuth modulo $2\pi$. For nonzero axial angular momentum, the scalar sky transport evaluates the bounded polar angle directly, omitting integer full turns before adding the smooth azimuth integral. Forming the unwrapped angle first introduces avoidable binary32 rounding even though those turns cannot change a celestial direction. Disk transport retains the unwrapped primitive needed by its existing coordinate output. Zero-momentum axis handling retains its separate half-turn convention.

Sky endpoints with positive Carter constant evaluate $\mu=\sqrt{z_+}\operatorname{cn}(u\mid m)$ using the same phase, frequency, and modulus as equator events. The differentiated sky path uses this polar representation as well. Evaluating the endpoint through a separately rounded general quartic produced source-map plateaus that lost three retained high-order stellar images. Nonpositive-C paths retain the general quartic representation; the event solver still owns opacity and branch ordering. Preparation follows the active representation: scalar positive-C event paths keep only the initial latitude in the unused general-polar record, without preparing its quartic. Differentiated paths prepare the general polar quartic only when their analytic polar representation is disabled, including axis cases. Validity checks remain attached to the representation actually evaluated.

The remaining rates use composite eight-point [Gauss–Legendre quadrature](https://dlmf.nist.gov/3.5#v) after $\tau=\tau_b\sin^2(\pi v/2)$. The rule integrates degree-15 polynomials exactly in exact arithmetic. Its nodes and weights were derived from the degree-eight Legendre polynomial; they are stored as f32. A prepared kernel shares the path and AGM data, and four interleaved sums shorten scalar dependency chains. Native f32 disk/time work permits 512 nodes; selected disk paths and sky-only work permit 1024. Phase-based initial panels reserve two refinement levels. Two successive doubled grids must agree within 2e−5 absolute azimuth and 2e−5 times $1+|\Delta t|$ for time. The accepted value is the fine rule, without Richardson extrapolation. These are quadrature estimates, not complete endpoint error bounds; exhaustion remains unresolved.

## Sky derivatives

A locally smooth sky map needs derivatives with respect to physical screen pixels. Ordinary pixels use compatible same-branch neighbors, including a rank-checked two-dimensional stencil. [Forward differentiation](../src/gpu/wgsl/math/quartic-dual32.wgsl) repairs missing derivatives with one semi-analytic ray, carrying a value and two derivatives per scalar. It avoids finite differences between nearly equal endpoints. Selected critical rays also use binary64 path preparation and a fixed orthonormal derivative basis, as described under [precision](#expression-context-and-derivative-ownership).

The differentiated quartic evaluator cancels the Weierstrass pole algebraically before evaluation. For a verified sky endpoint, $u(\tau_b,p)=0$ and $u'(\tau_b)=-E$ imply

$$
\partial_p\tau_b=\frac{\partial_pu}{E}.
$$

This division occurs only after establishing $E>0$ and sky destination. It is not global energy normalization. The positive-C polar primitive is differentiated with the signed root $L_z\sqrt{z/(\mathcal C+a^2E^2z)}$, avoiding an absolute-value derivative at zero momentum. At fixed $a=0$, its residual and screen derivatives vanish exactly.

In a regular endpoint chart, the source solid-angle Jacobian is

$$
J=|\mu_x\phi_y-\mu_y\phi_x|.
$$

Differentiated transport uses the same cosine-mapped eight-point Gaussian rule. It integrates the determinant's linear contraction alongside the components. Two successive doubled grids must agree within 1e−4 of the final determinant and $10^{-5}(1+|v|)$ for every component. The budget is 2048 nodes for ordinary rays and 16384 for selected critical rays, with an initial grid no larger than 256. Endpoint poles, unsafe arithmetic, wrong branches, and exhausted estimates remain unresolved.

Each retained ray owns a 32-invocation workgroup. Lane zero prepares its path; [workgroupUniformLoad](https://www.w3.org/TR/WGSL/#workgroupUniformLoad-builtin) broadcasts the prepared values and uniform failure decisions. Each lane accumulates strided nodes privately. Native subgroup reductions combine the value, two derivatives, area contraction, and validity flag. Elected subgroup lanes write their own local-index scratch slots; lane zero combines those slots after a barrier. No fixed subgroup size or mapping to consecutive local invocations is assumed. The same rule and acceptance policy serve primary repair, boundary repair, and stellar refinement.

Stable differences between two resolved sky endpoints use rationalized latitude differences and trigonometric addition identities in [beam.wgsl](../src/gpu/wgsl/lensing/beam.wgsl). They reduce arithmetic cancellation in neighbor stencils but cannot restore precision already lost in endpoint transport.

## Pixel integration

Beam repair and failure classification test the source solid-angle area after scaling both derivative vectors by their largest absolute component. The condition is $\sqrt{|n\cdot(\hat d_x\times\hat d_y)|}>10^{-10}/s$, with $d_i=s\hat d_i$; zero scale is rank deficient. This retains the existing $J>10^{-20}$ criterion without forming an overflowing squared scale or cross product.

The distant sky combines a solid-angle-weighted diffuse spectral mip chain with individual point sources. A unit-integral detector tent weights each stellar image, and source flux divided by its source solid-angle Jacobian supplies magnification once. The primary and boundary paths currently use different local map approximations.

### Critical-curve preparation

The CPU [critical-direction preparer](../src/physics/critical.ts) supplies vacuum capture/escape seeds for the production stellar search. Disk opacity and finite-time endpoints still belong to the ray solver.

For this positive-Killing-energy spherical-orbit family only, write $\lambda=L_z/E$ and $K=[(L_z-aE)^2+\mathcal C]/E^2$. The double-root conditions $R=R'=0$ give $\sqrt K=2r\sqrt\Delta/(r-1)$; see [Wang, Lee, and Lin, equations 43–44](https://arxiv.org/html/2208.11906v1#S2.SS2). This normalization does not replace the unnormalized constants used for general photons.

Resolve the observer's polar-potential constraint with a periodic parameter $\chi$:

$$
\lambda=a\sin^2\theta+\sqrt K\sin\theta\cos\chi.
$$

Substitution into the double-root conditions gives the scalar equation

$$
(r^2+a^2\cos^2\theta)(r-1)-2r\Delta
-2ar\sqrt\Delta\sin\theta\cos\chi=0.
$$

The scalar solve retains $s=r-1$ directly, with $\Delta=s^2-(1-a^2-q^2)$ and exterior bracket $[\sqrt{1-a^2-q^2},3]$. This avoids losing near-horizon digits when subtracting one from an already rounded radius, and avoids feeding rounded horizon locations back into the quadratic. If the rounded lower endpoint lies inside the quadratic horizon, the next binary64 value supplies the exterior endpoint. The CPU uses bisection. The [GPU solver](../src/gpu/wgsl/lensing/critical64.wgsl) uses [Illinois interpolation](https://doi.org/10.1007/BF01934364): retaining the same endpoint twice halves its interpolation weight, while actual function signs continue to own the bracket. Endpoint-rounded proposals move one representable value inward. Both solvers stop only when no binary64 midpoint remains; the 128-iteration limit still reports unresolved work. The retained bracket width measures floating-point refinement, not a certified physical interval. No division by $a$ or $\sin\theta$ is needed, so the nonrotating and axis limits share the same equation. Scalar and differentiated critical directions retain $s$ through the small denominators.

The local ZAMO direction approaches the spherical radius from either side. Its radial component uses the factored potential

$$
R(r_o)/E^2=(r_o-r)^2\left[(r_o+r)^2-\frac{4r\Delta(r)}{(r-1)^2}\right],
$$

which preserves the zero radial component when observer and orbit radii coincide. Invalid input, exhausted or ill-conditioned arithmetic, and a prepared point are distinct results. A failed unit-vector check is unresolved rather than silently normalized. These seeds locate the vacuum separatrix; they neither enumerate stellar images nor certify a flux tail.

### Source-directed critical strip

[AART, section II.4](https://arxiv.org/html/2211.07469v3#S2.SS4), motivates geometry-adapted sampling through Kerr lensing bands defined by equatorial-crossing support. Its distant-observer equatorial-emission construction does not directly certify this renderer's finite-observer Kerr–Newman point-source search. Nullray therefore retains explicit source roots and error checks; an exponentially narrow band alone is not an omitted-flux bound for a magnified point source.

The [owned stellar search](../src/gpu/stellar-search.ts) samples 256 periodic critical columns and 128 logarithmic radial intervals. Its local directions are proportional to $(n_r+\delta,n_\theta,n_\phi)$, with $10^{-8}\le\delta\le0.25$. These are computational work bounds, not physical image-order cutoffs or a certified flux tail. Ordered opaque tracing determines every sample's endpoint. Same-branch cells query the source tree and invert their normalized bilinear sky map. Per-cell counts and a two-word unsigned integer prefix scan allocate canonical cell/source/root ranges before a second traversal writes at most 32768 retained candidates. Carry propagation preserves offsets beyond $2^{32}$; candidate and overflow diagnostic counters saturate at the maximum u32 value. Truncation therefore cannot wrap back into the retained range, and the retained subset does not depend on atomic scheduling. Mixed parents receive five physical midpoint samples and four quarter cells. Neighboring same-branch cells share these physical edge midpoints; their other edges and center retain homogeneous bilinear interpolation, so the coarse and fine source maps meet. Only the midpoint directions require the additional 256 critical columns. Still-mixed children remain unsearched. Mixed parents and children, failed projections/endpoints/inversions, overflow, and failed refinement remain separate work counters. This single subdivision level recovers ten independently retained images but does not bound omitted flux.

Each candidate is refined in $(\chi,\log\delta)$ with a fresh critical direction and a differentiated physical sky ray. Native sine/cosine values are normalized in integer binary64 before imposing the critical equation, so their phase rounding moves along the curve. Perspective projection uses the reciprocal basis of the actual quantized camera axes, with cross products and determinant sign evaluated in integer binary64. Component quantization does not retain exact orthonormality: dot-product projection introduced a direction round-trip error around $2.5\times10^{-8}$ in a yawed/pitched camera. The reciprocal transform inverts the implemented launch without changing the camera or its quantization. A [GPU round-trip check](../tests/gpu/critical-curve.test.ts) covers axis-aligned, yawed/pitched, and rolled cameras at radial offsets $10^{-3}$ and $10^{-8}$.

The perspective projection and final ray launch retain continuous binary64 screen coordinates; rounding a fractional pixel to binary32 already changed one second-order image's flux beyond the retained target. The Newton update uses derivatives in the prepared orthonormal screen basis. The critical-curve tangent follows implicit differentiation of the spherical-orbit equation, $dr_\mathrm{s}/d\psi=-F_\psi/F_r$, then binary64 differentiation of the local direction and reciprocal-camera projection. Parameter tangents rotate into the prepared basis before binary32 rounding. This avoids both a finite-difference chord across the curved separatrix and cancellation between nearly parallel restored Cartesian gradients. The final source solid-angle Jacobian is evaluated in the same orthonormal basis, where area is invariant under its rotation. A [GPU derivative check](../tests/gpu/critical-curve.test.ts) compares 192 tangents to converged independent directional differences. Bounded damped iterations limit work; the endpoint must retain its opaque branch and the source chord residual must be at most $5\times10^{-6}$ before spectral evaluation. The source-chart Jacobian must also predict a Newton displacement of at most 0.0005 physical pixels. The check scales the Jacobian before forming its determinant and compares cross-multiplied magnitudes, avoiding division by a small determinant. This first-order estimate is not a certified position bound or a proof of unique or complete image enumeration; high magnification can expose an endpoint-rounding floor and leave the candidate unresolved. Each iteration first prepares independent rays with one invocation per candidate, then transports each prepared derivative with a cooperative workgroup. Explicit host-shareable records replace boolean fields with integer flags at this storage boundary; the transport tolerances are retained. This avoids making 31 lanes wait while one lane computes a scalar critical direction and ray launch.

The accepted parameter state stores the Carter circle and positive radial offset as binary64 words. Updates rotate that circle and multiply the offset by the exponential of the logarithmic step; binary32 phase/log values serve only as diagnostics after initialization. The [bounded update kernel](../src/gpu/wgsl/lensing/critical-step64.wgsl) evaluates sine/cosine and exponential polynomials for steps within 0.25 radians and unit log scale; production clips steps more tightly to 0.1 and 0.5. This retains corrections below one binary32 parameter spacing. The [GPU update check](../tests/gpu/critical-curve.test.ts) compares 736 cases, including successive sub-spacing updates, with native binary64 references.

One differentiated transport supplies the sky value, residual, and Jacobian for each trial. Preparation verifies radial arrival, ordered disk opacity, and the retained branch without separately integrating a scalar sky endpoint. Its implicit critical tangent reuses the same solved orbit. Comparing a scalar residual with a separately rounded differentiated map previously introduced conflicting plateau decisions and repeated most of the path work.

The host encodes up to 32 trial stages. Two alternating compact index queues retain unfinished candidates; indirect preparation and transport dispatches consume only the active queue. Atomic append order changes scheduling, while canonical candidate indices preserve state ownership and output order. Each candidate permits eight accepted Newton corrections and up to eight successive halvings of a proposed step; the global 32-trial limit can end work first. A rejected trial preserves the preceding accepted parameters and Newton direction. A new nonconverged point must strictly reduce the same differentiated source residual. Convergence still requires both the angular and physical-pixel criteria above. Exhaustion, invalid derivatives, and branch changes remain unresolved. The coherent evaluator uses one physical map for residuals and derivatives. Fresh-device checks compare canonical retained records directly, including forced truncation and prefix totals beyond the u32 range.

The ordinary and critical filters use one [computational ownership function](../src/gpu/wgsl/imaging/stellar-partition.wgsl) at each image position. Sorted critical tangent angles interpolate $m=n_r/\sqrt{n_\theta^2+n_\phi^2}$. For a local screen direction $v$, the critical filter owns $-0.01\le(v_r/|v_\perp|-m)/\sqrt{1+m^2}<0.05$. The interpolant defines the partition; the physical solver still uses the curved critical family. Both filters evaluate the same partition, including the physical origin/scale of boundary subpixels. This avoids suppressing unrelated ordinary images merely because their detector tents overlap a critical image. A [shared-pixel GPU check](../tests/gpu/stellar-filtering.test.ts) combines one image from each path in a pixel whose center lies outside the critical partition and checks that their flux is neither lost nor doubled.

Confirmed critical images enter four per-pixel linked-list nodes, weighted by the unit-integral detector tent. Spectral resolve adds this light after boundary averaging and applies sky brightness once. Disk opacity belongs to the image's ray, so a visible image's detector tent may overlap a pixel whose center hits the disk. Output range checks occur after composition. Physical image positions and flux are reused across detector jitter; only their receiving pixels and weights change. Camera, spacetime, disk geometry, or raster changes invalidate the search.

The retained Schwarzschild source is found without supplying its image positions, and its four visible images meet the production flux gate through primary, boundary, and jittered composition. General source-aware subdivision, ambiguous-root ownership after Newton refinement, and a bounded omitted tail remain incomplete. Current work counters and sampled-ray alpha therefore do not certify completeness of stellar light.

### Primary stellar cells

A complete same-branch 3 × 3 primary neighborhood supplies four shared unit-pixel cells. The [stellar cell filter](../src/gpu/wgsl/imaging/stellar-patch.wgsl) normalizes the bilinear interpolation of each cell's four unit sky directions:

$$
\boldsymbol P(u,v)=(1-u)(1-v)\boldsymbol n_{00}+u(1-v)\boldsymbol n_{10}
 +(1-u)v\boldsymbol n_{01}+uv\boldsymbol n_{11},\qquad
\boldsymbol n(u,v)=\boldsymbol P/|\boldsymbol P|.
$$

A common open hemisphere excludes a zero homogeneous vector. A spherical cap bounds the normalized convex cell; direct chord differences avoid small-angle cancellation in $2(1-\cos\theta)$. Eight binary32 machine epsilons widen the chord query bound only. Each cell queries its own cap to avoid evaluating roots for sources belonging only to a distant part of the detector footprint.

Projecting perpendicular to the source direction gives two equations of the form $\boldsymbol A+\boldsymbol B u+(\boldsymbol C+\boldsymbol D u)v=0$. Eliminating $v$ yields

$$
(\boldsymbol B\mathbin\times\boldsymbol D)u^2+
(\boldsymbol A\mathbin\times\boldsymbol D+\boldsymbol B\mathbin\times\boldsymbol C)u+
\boldsymbol A\mathbin\times\boldsymbol C=0,
$$

where the cross product is the scalar two-dimensional determinant. Exchanging $\boldsymbol B$ and $\boldsymbol C$ gives the second elimination polynomial. Scaled quadratic solves retain both roots; linear equations are handled separately. A definitely negative discriminant excludes images. An arithmetic uncertainty band of $32\epsilon(b^2+4|ac|)$ rejects an uncertain discriminant without clamping it into a physical root, unless the other coordinate excludes the source independently.

Half-open $[0,1)^2$ ownership uses polynomial signs at canonical shared edges, rather than testing only the rounded root coordinates. The latter doubled some shared-edge images when an excluded root at 1 rounded down by one ulp. A filtered determinant computes edge orientation; cancellation and underflow use exact 48-bit integer products of the already rounded binary32 chart coordinates. Both adjacent cells therefore make the same edge decision. A zero edge determinant factors the boundary root before inversion. A two-equation residual pairs the scalar roots, with one-to-one matching on each coordinate; ambiguous pairing remains unresolved. These arithmetic screens are not an interval proof of physical image completeness.

Each retained image contributes at its own

$$
J=\frac{|\boldsymbol P\cdot(\partial_u\boldsymbol P\times\partial_v\boldsymbol P)|}{|\boldsymbol P|^3}.
$$

The implementation scales derivatives first and evaluates $1/\sqrt J$ only after enforcing $J>10^{-20}$. Killing energy is interpolated at the image position before shifting its source spectrum. A point on a singular fold remains explicitly unresolved. [Cell checks](../tests/gpu/stellar-filtering.test.ts) cover two opposite-parity images, shared-edge and vertex ownership under rotation, source-chart changes, spectral differences between images, and scale extremes. The [production sky check](../tests/gpu/sources.test.ts) also integrates a folded pair through endpoint storage and the actual spectral resolve.

This enumerates the images of the interpolated cell, not every image of the physical optical map. A thin branch or additional fold can lie between all retained rays. The separate critical search closes the retained [Schwarzschild multi-image gate](validation.md#independent-references); local cells alone do not.

### Incomplete and boundary neighborhoods

Incomplete primary neighborhoods and the 4 × 4 boundary atlas retain the [normalized affine filter](../src/gpu/wgsl/imaging/source-tree.wgsl). Its local model normalizes $\boldsymbol n_0+\boldsymbol d_xX+\boldsymbol d_yY$. If $h=\boldsymbol n_\star\cdot\boldsymbol n_0>0$, the density at the stellar image is $J_\star=J_0h^3$. The same projection supplies inversion and amplification; using only the center density underestimates wide-beam flux. The shader divides by two alignment-scaled beam factors and then by $h$, avoiding squared scales or cubed small cosines, and retains $J_\star>10^{-20}$. It cannot enumerate all images across a caustic. Neither local path interpolates endpoints across different visibility/image-order tags.

### Optical quadrature

Exploration selects mixed endpoint/image-order boundaries and varying disk pixels for 4 × 4 quadrature. Selection accounts for radial/azimuthal source variation, frequency, and retarded emission time. Every selected subpixel carries weight 1/16, including failures. Rank-deficient sky subpixels use a separate differentiated-ray queue. Refinement, queue overflow, and radiation failures remain distinct diagnostics.

For the [prescribed disk modes](physics.md#disk-emission), a geometry-only work indicator bounds each mode's change by $\min(2,m\Delta\psi+3k|\Delta\log r|)$ before summing its flux weight. Here $\Delta\psi$ includes wrapped longitude, differential rotation over the 24-unit cohort age, and retarded-time variation. Cohort replacement contributes at most $|\Delta t|/8$. A bound above one or a log-frequency difference above 0.03 selects quadrature. The per-mode cap follows from bounded sine amplitudes; it avoids treating one fine mode as arbitrarily large contrast. This is a selection heuristic, not a certified radiance-error bound.

The primary derivative queue has at most 4096 candidates. The boundary budget is $\min(WH,\max(32768,\min(131072,\lceil WH/32\rceil)))$; its derivative budget is $\operatorname{clamp}(3(W+H),4096,16384)$, limited further by atlas size. These are bounded work/memory policies. The boundary budget scales with area because disk-interior source variation can occupy an area rather than a curve. Its 131072-pixel cap requires 56 MiB for 16 endpoint/delay/radiance samples per pixel before row padding; overflow remains explicitly unresolved. Narrow features can lie between all primary samples, and fixed quadrature does not certify convergence near critical curves.

Photographs average 64 jittered samples in one read/write `rgba32float` history. Optical radiance alpha records the resolved fraction; history alpha records the averaged missing weight. The coverage pass reconstructs integer sixteenth-sample units, reduces them in subgroups/workgroups, and sums blocks on the CPU to avoid whole-image u32 overflow. It never rounds partial quadrature failures to whole samples or renormalizes away failed samples.

Frequency and image-order inspection operate on resolved endpoints and retain zero coverage for failed geometry. Frequency inspection uses $-\log_2 E$ for the sky and $\log_2 g$ for the disk. Diagnostic color never changes an unresolved endpoint into a successful photographic sample.

## Radiance filtering

Bloom convolves linear radiance with a separable $[1,2,1]/4$ tent at each pyramid level, then averages fine and reconstructed coarse levels. At fractional texel phase $f$, linear interpolation expands the one-dimensional coefficients to $[1-f,2-f,1+f,f]/4$. Pairing adjacent coefficients gives weights $(3-2f)/4$ and $(1+2f)/4; the two-dimensional product needs four hardware bilinear samples instead of nine. The pairing remains valid at arbitrary pixel phases and clamped image edges, including odd-sized and one-pixel targets. A direct binary64 convolution reference checks the implementation with binary16 rounding between levels. This is prescribed display scattering, not an additional gravitational amplification.

## Precision and failure policy

| Quantity                                      | Representation                                           |
| --------------------------------------------- | -------------------------------------------------------- |
| CPU preparation and references                | ECMAScript binary64 numbers                              |
| Optical geometry, transport, derivatives      | WGSL f32 with selected integer binary64 path preparation |
| Endpoint coordinates and photographic history | 32-bit texture components                                |
| Linear radiance and diffuse coefficients      | Half-float textures, with range checks/scaling           |
| Work counts, queue indices, coverage          | Integer GPU values; binary64 host totals                 |

Guard divisors, radicands, domain restrictions, and magnitudes before unsafe arithmetic. WGSL finite-math rules do not promise a useful NaN signal, and `select` is not lazy. Do not use f16 for optical paths. WGSL permits reassociation and an unfused `fma`; conventional error-free two-float transforms are therefore not a portable precision foundation. See [WGSL floating-point evaluation](https://www.w3.org/TR/WGSL/#floating-point-evaluation).

The [integer binary64 core](../src/gpu/wgsl/math/binary64.wgsl) prepares selected primary and boundary rays at higher precision. The selection screen scales the radial coefficients and tests $g_3<0$ together with $|g_2^3-27g_3^2|\le0.04(|g_2^3|+27g_3^2)$. For the real-root parameter $m=(e_2-e_3)/(e_1-e_3)$, the negative-$g_3$ repeated-root limit has $e_1=e_2$ and $m=1$; the positive-$g_3$ limit has $e_2=e_3$ and $m=0$. The latter previously selected almost every distant-view ray unnecessarily. This signed dimensionless band selects computational work; it is not a proven endpoint error bound or a new physical exclusion. Selected rays retain binary64 preparation through the elliptic parameters. Other rays use the scalar f32 solver. The observer metric is computed once from the quantized frame in host binary64 and stored in a separate 96-byte uniform, leaving the optical frame layout intact. Its twelve values retain sine/cosine, squared radius/spin/cosine, metric coefficients, lapse, dragging, and three tetrad scales. Primary, boundary, and differentiated launches share these IEEE words; screen-dependent products retain their original order.

The arithmetic stores IEEE words in `vec2u`, aligns significands with a sticky discarded-bit flag, and rounds to nearest even. Multiplication uses an exact base-$2^{16}$ product. Radix-2048 division computes five eleven-bit quotient digits in place of 55 single-bit steps. A normalized divisor has a high word of at least $2^{20}$; each high-word digit estimate is exact or one too large and is corrected against its full integer product. Square root seeds its first 24 bits with native `sqrt`, then exact integer-square comparisons correct the seed before 32 remaining digit-pair steps. Both operations retain three rounding bits and a sticky remainder; native estimate rounding never enters the accepted result. Arithmetic accepts normal finite numbers and zero; unsupported subnormal operands/results, overflow, and invalid operations propagate an integer sentinel without evaluating a floating NaN. Signed arithmetic zero is not an IEEE conformance claim. Separate bit conversions preserve finite binary32 values, including subnormals.

The [GPU arithmetic checks](../tests/gpu/arithmetic.test.ts) compare native binary64 results across retained edge cases, seeded exponent coverage, and cancellation pairs. The [camera launch](../src/gpu/wgsl/geodesics/launch64.wgsl) reconstructs the three critical fixture rays from quantized frames, with the observer's sine/cosine supplied in host binary64. Its direction and photon constants agree with the references within $2\times10^{-14}\max(1,|x|)$.

The [mixed-precision check](../tests/gpu/local-geometry.test.ts) additionally prepares quartic coefficients, derivatives, and elliptic parameters through [integer arithmetic](../src/gpu/wgsl/math/quartic64.wgsl). It scales the invariant cubic, refines its isolated signed root, and solves the remaining quadratic for the close pair. [Factored cubic families](../tests/gpu/elliptic.test.ts) check 106 cases across signs, scales, close/repeated real roots, complex pairs, and zero invariants. These finite families do not certify all near-degenerate inputs.

Prepared paths are rounded to f32 before [ordered tracing](../src/gpu/wgsl/geodesics/trace64.wgsl), which chooses the first opaque disk event or sky/horizon arrival without a supplied event order. One compute invocation carries each ray from its quantized frame to its endpoint and disk propagation time. On the three retained rays, the largest sky-vector error is $2.12\times10^{-6}$; disk inverse-radius, azimuth, and propagation-time errors also meet their targets. Launch checks include both axes and nearby inclinations. The production critical endpoint acceptance gate now passes. Broader endpoint coverage, the precision-selection policy, and derivative consistency still require joint-domain validation.

### Expression context and derivative ownership

[Differentiated binary64 preparation](../src/gpu/wgsl/lensing/sky-jacobian64.wgsl) uses the same critical-root selector as primary and boundary endpoints. It extends integer arithmetic through launch, polynomial invariants, and elliptic preparation. The isolated cubic root uses implicit differentiation, $dr=(r\,dg_2+dg_3)/(12r^2-g_2)$; the remaining pair follows from its quadratic factor. [Launch derivatives](../tests/gpu/local-geometry.test.ts) agree with converged screen differences, and [factored cubic derivatives](../tests/gpu/elliptic.test.ts) cover 45 scaled real/complex cases. A zero square-root argument remains unresolved for transverse repeated-root perturbations. The exact quadratic family uses the explicit limit described under [elliptic paths](#elliptic-paths).

Both derivative directions rotate into an orthonormal basis aligned with the radial modulus gradient before rounding to f32. A zero modulus gradient retains the identity basis. The basis is explicit path data, held fixed when applying the chain rule at that ray; Cartesian derivatives rotate back with integer arithmetic before their final f32 storage. Cooperative transport permits up to 16384 subdivisions on this path with unchanged convergence thresholds. Nine refinement levels allow an initial grid of 32 to reach that cap; ordinary rays remain capped at 2048. The [basis-level gate](../tests/gpu/ray-differentials.test.ts) and [production screen-beam gate](../tests/gpu/ray-differentials.test.ts) now cover ten retained sky samples. Maximum area relative errors are $1.26\times10^{-4}$ and $1.67\times10^{-4}$ respectively against the unchanged 0.002 target; screen gradient relative errors are below $3.05\times10^{-6}$.

The original area reference subtracted products of nearly parallel screen gradients. Convergence of each gradient alone was insufficient: that reference produced an apparent 0.89% area discrepancy. The corrected [area reference](../tests/reference/sky-derivative.ts) independently chooses an orthonormal basis from the binary64 sky gradient, takes directional differences, and checks the area itself at two step sizes. Its area convergence is $4.07\times10^{-6}$ on the difficult sample. The 0.2% acceptance threshold is unchanged; neither the GPU basis nor its measured area sets the reference value.

Derivative helpers receive an explicit caller-owned failure flag. A rejected arithmetic operation marks that evaluation unresolved without contaminating independently owned evaluations. A pure value/status return trial lost an arrival gradient when composed with polar motion on Apple M5/macOS 26.6.2. Native replay of Dawn-generated Metal reproduced the loss with fast trigonometric functions; precise sine or cosine restored it. This historical investigation localized the observed backend behavior, without identifying a compiler optimization pass. Its generated replay directory is not a checked-in reference. The durable acceptance checks are [derivative ownership](../tests/gpu/ray-differentials.test.ts), [sky Jacobians](../tests/gpu/ray-differentials.test.ts), and [beam repair](../tests/gpu/sky-beams.test.ts).

The [launch-roundoff fixture](../tests/fixtures/launch-roundoff.json) separately records a one-ULP Carter change after extracting launch intermediates into a returned struct. Complete execution changed from escape to unresolved disk transport while truncated probes agreed. This is expression-context sensitivity, not proof of a compiler defect. Launch expressions therefore remain inside the pure trace function. Treat future function extraction, expression reassociation, or precision changes as numerical work requiring complete GPU execution against the retained references.

Sharing the ordered-event function between f32 and mixed preparation also failed the [critical event-order reference](../tests/fixtures/critical-event-order.json) on Chromium 153/Apple Metal-3: inverse-radius error increased to $1.9703021\times10^{-5}$ against the unchanged $10^{-5}$ gate. Keeping equator and radial classification in the caller did not remove that regression. Restoring the separate trace functions restored the gate. The duplicated event bodies therefore remain a numerical integration constraint, not evidence that the two precision paths have different physical rules; future consolidation requires the same complete GPU checks.

Capture, escape, disk emission, invalid input, and unresolved numerical work are separate outcomes. Diagnostic stage codes help minimize failures; they are not physical destinations. Near-repeated roots, long phases, endpoint ties, nonpositive-C axis transport, and nonlinear stellar mapping require further acceptance work. The [photographic boundary corpus](../tests/fixtures/photographic-boundaries.json) additionally retains outer-disk destination disagreements, unbracketed arrivals, and critical derivative rejection at native portrait resolution. Integer preparation alone does not settle every endpoint tie or remove subsequent f32 area cancellation. The [critical-primary fixture](../tests/fixtures/critical-primary-rays.json) retains the historical sky-vector error around 0.00575; its three production endpoints now pass the original targets. This closes those counterexamples, not the complete critical support domain.
