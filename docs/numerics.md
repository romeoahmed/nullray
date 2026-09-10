# Numerical model

The equations and units are defined in [Physics](physics.md). This document records the current coordinate representation and algorithms. Numerical work limits control cost and return unfinished work; they do not define a physical capture boundary.

## Regular coordinates

The metric uses ingoing or outgoing Cartesian Kerr–Schild coordinates with sign $\epsilon=+1$ or $-1$. The rotating unit vector $\boldsymbol n$ gives the spatial position

$$
\boldsymbol X=r\boldsymbol n+\epsilon a\boldsymbol e_z\times\boldsymbol n.
$$

The null-chart time is $V=T+\epsilon r$. Chart and spacetime block are separate state. Coordinate transitions preserve the physical event; horizon crossings update the block history.

Direct radius resolves the regular part of $r=0$. Reciprocal radius $x=1/r$ resolves either asymptotic end. The implementation switches when the current coordinate exceeds unit magnitude, keeping the numerical radial coordinate bounded. Crossing reciprocal zero is an infinity endpoint, never a transition to negative radius.

## Pole-free angular motion

Store the angular direction $\boldsymbol n$ and canonical vector $\boldsymbol J$ rather than dividing by $\sin\theta$. For mass squared $m^2\in\{0,1\}$,

$$
\begin{aligned}
\boldsymbol n'&=\boldsymbol J\times\boldsymbol n+\Omega\boldsymbol e_z\times\boldsymbol n,\\
\boldsymbol J'&=a^2(E^2-m^2)n_z(\boldsymbol n\times\boldsymbol e_z)
 +\Omega\boldsymbol e_z\times\boldsymbol J.
\end{aligned}
$$

Primes are future Mino derivatives. The invariants are $|\boldsymbol n|=1$, $\boldsymbol J\cdot\boldsymbol n=0$, $J_z=L$ and $J^2-a^2(E^2-m^2)n_z^2=C+L^2$. The camera uses the same local frame at every sample, including either axis.

Let $K=(L-aE)^2+C$, $P=E(r^2+a^2)-aL$ and $v=r'$. The horizon-regular transport is

$$
D=\frac{P+\epsilon v}{\Delta}
 =\frac{K+m^2r^2}{P-\epsilon v},\qquad
\Omega=a(D-E),
$$

$$
V'=a\bigl[L-aE(1-n_z^2)\bigr]+(r^2+a^2)D.
$$

Choose the chart with the larger rationalized denominator before reaching a pole. The unreduced expression supplies the removable principal-ray limit when its denominator is regular. Neither expression may divide by zero.

## Radial flow and integration

For $b=a(aE-L)$, write the radial polynomial as

$$
R(r)=(E^2-m^2)r^4+2m^2r^3+A r^2+2Kr+B,
$$

where $A=2Eb-K-m^2(a^2+q^2)$ and $B=b^2-(a^2+q^2)K$. Evolve $r''=R'(r)/2$. In reciprocal radius the corresponding equation is

$$
x''=m^2+A x+3Kx^2+2Bx^3.
$$

These equations pass through ordinary radial turning points without manually reversing a square-root sign. They preserve the first integral in exact arithmetic; numerical drift still needs validation.

For the [separable plasma prescription](physics.md#plasma-refraction), subtract $\tfrac12(\Delta f_r)'$ from the radial acceleration. In reciprocal radius, subtract half the derivative of $x^2(1-2x+(a^2+q^2)x^2)A/(1+r_0^2x^2)$. The regular transport numerator uses $K+f_r$. Timelike observers retain their vacuum mass terms. The reference Hamiltonian differentiates the density term together with the inverse metric, independently of these separated equations.

The host uses Dormand–Prince 5(4). Timelike preparation may use proper time by dividing each stage by $\Sigma$. The GPU uses the four-evaluation Bogacki–Shampine 3(2) pair. Its ordinary radial endpoints are projected onto $v^2=R(x)$ in the active radius or reciprocal-radius coordinate. Scaled Newton corrections follow the normal to this constraint, correcting velocity on radial legs and coordinate at a simple turn. Correction size participates in step acceptance. The exact bifurcation patch retains its own regular equations. An uncorrectable degenerate gradient rejects the step. For geometries with horizons, the projection factors $\Delta$ using the same representable horizon pair as atlas transitions. The shared horizon representation keeps radial constraints and block transitions consistent near inner-horizon turns. General critical-family accuracy still requires independent image convergence. Embedded error estimates adapt the step. The current local tolerances are development settings, not a certificate of image error or a permanent performance requirement.

## Bifurcation continuation

A horizon with $P(r_h)=0$ can be a radial turning point whose continuation passes through the bifurcation sphere, not an ordinary crossing into a stationary exterior. The [Kruskal continuation of horizon-critical geodesics](https://doi.org/10.1007/s00023-023-01273-6) motivates a separate regular coordinate combination. The current implementation covers the exact family $E=0$, $aL=0$ at nondegenerate horizons. It does not use a near-zero threshold to change other trajectories.

For the selected horizon $r_h$ and other radial root $r_o$, write

$$
s_h=r_h^2+a^2,\qquad \kappa_h=\frac{r_h-1}{s_h},\qquad
\Omega_h=\frac a{s_h},\qquad B=K+m^2r^2+f_r(r).
$$

Here $R=-\Delta B$, $v=dr/d\gamma$, and $w$ is the current ingoing/outgoing null time with sign $\sigma=\pm1$. Instead of divergent $w$ and rotating longitude, evolve

$$
b=\sigma\kappa_h w-\log|v|,\qquad \psi=\varphi_w-\Omega_h w.
$$

Algebraic cancellation of the simple horizon factor gives

$$
\frac{db}{d\gamma}=-v\left[\frac{1-\kappa_h(r+r_h)}{r-r_o}+\frac{B'}{2B}\right].
$$

The pole-free angular-vector system uses the additional axial rotation
$-\sigma\Omega_h(r+r_h)v/(r-r_o)$. Both rates remain finite at the selected horizon. The nearest root selects the patch; the overlap midpoint is a coordinate choice, not a trajectory cutoff.

At a radial turn on that bifurcation surface, switch the null-chart sign using finite expressions:

$$
\begin{aligned}
b_{\rm new}&=-b+2\kappa_h r-\left(1+\frac{r_o^2+a^2}{s_h}\right)\log|r-r_o|-\log B,\\
\psi_{\rm new}&=\psi+2\sigma\Omega_h\left(r+2\log|r-r_o|\right).
\end{aligned}
$$

The outer bifurcation connects the black-hole and white-hole blocks of the same universe. At the inner bifurcation, black-hole block $n$ connects to white-hole block $n+1$. The inverse transition reverses that increment. CPU proper-time worldlines and GPU null paths use the same cancellation. An ordinary finite null-chart point can be recovered away from the exact bifurcation sphere; placing an observer exactly on that sphere still needs a separate coordinate interface.

## Ordered events

The image pass brackets equatorial crossings, coplanar annulus entry and radial endpoints in cubic dense output through accepted endpoint states and derivatives. Both separated and canonical paths search their accepted segment rather than reintegrating each bisection trial. It selects the earliest emitting or terminating event along the backward path. A regular horizon is not an opaque surface. When a step begins exactly on a horizon, the departing radial side is compared with the current block before updating it; a step boundary must neither erase nor duplicate a crossing. Disk hits require the specified annulus and a timelike emitter with positive measured photon frequency.

The source-free shortcut requires a strictly negative enclosure of the radial potential at a radius between the observer and the innermost emitting radius. The lower boundary is a negative-potential enclosure at zero for rotating geometries, or the absorbing zero-radius singularity when spin vanishes. Stored f32 constants are enclosed through guarded interval addition and multiplication, with outward representable neighbours and subnormal flush coverage. Positive reciprocals stay inside the [WGSL specified divisor range](https://www.w3.org/TR/WGSL/#floating-point-accuracy) and widen by four neighbours for division error. A vacuum-quartic stationary point supplies only a candidate radius; its location is never a proof without the independent negative-potential enclosure. Overflow or an uncertain sign leaves the ordinary ray solver active. This certifies source exclusion for the stored constants, not full trajectory or image convergence. Recorded source-free paths show only a finite prefix. Detector-normalized vacuum photons with exactly zero reduced Carter data and radial/angular motion are also resolved as source-free on the exactly representable Schwarzschild or extremal horizons, when all emitters lie outside. The inspector records their fixed radius and latitude once: the reduced constants do not retain the affine scale of a horizon generator. Nearby rays are not snapped onto that family.

A rotating ray may cross $r=0$ away from the equatorial ring. For zero spin, the entire zero-radius set is singular. Infinity, singularity and unresolved work remain distinct. Integer metadata carries source-domain identity independently of radiance.

An accepted step whose endpoint radial velocities bracket a turn is shortened by bisection to the far side of that turn. Cubic Hermite dense output from the accepted endpoint states and derivatives locates the turn without reintegrating at every bisection trial. This separates the incoming and outgoing radial legs for horizon bookkeeping, source searches and material clipping. Multiple turns with equal-sign endpoint velocities, simultaneous events, the remaining horizon-critical families and complete near-critical image coverage still need further work. The implemented bifurcation continuation does not certify the complete maximal atlas.

## Radiation and sampling

Source preparation computes a neutral Kerr–Newman zero-torque disk profile and a CIE-integrated thermal lookup. A thermal frequency shift is applied by shifting temperature; an additional lensing or frequency multiplier must not be applied to that already shifted specific radiance.

The path pass samples the thermally emitting scattering annulus and jets at their retarded source events. I/Q/U are separately accumulated before detector analysis. A following pass filters the diffuse sky and integrates point stars over local lensed pixel footprints; certified nonlinear footprints remain open. Photographic jitter averages linear HDR samples; missing samples retain their missing weight. It does not establish stellar-image completeness.

Geometry uses an f32 embedded tolerance of $4\times10^{-5}$ and a 2048-attempt dispatch limit. These are interactive implementation choices, not physical capture conditions or acceptance gates. Accepted vacuum steps supply cubic Hermite dense output to a separate material march. Endpoint derivatives and the middle Bernstein derivative coefficient bound the cubic's coordinate rates; material support, vertical height and radial/azimuthal structure set local sampling distances. Direction normalization is a dense-output approximation, not a certified enclosure. Empty radial intervals are culled after geometric turn clipping.

Each material interval uses two ordered midpoint cells and analytic exponential attenuation. This avoids repeating the geodesic integrator for every density sample and removes an explicit-Euler optical-depth stability restriction. The remaining radial interval is checked after every cell: once it lies beyond all material, sampling ends even if the geometric segment continues to infinity. A 512-interval limit per geometric segment reports unresolved work when exhausted. Transmission below $10^{-4}$ terminates an optically thick ray; the neglected background is a stated image approximation, not exact zero light.

Material sampling accounts for radial and angular motion, retarded emission time, and the age-weighted shear of the two source cohorts. This resolves the base lattice before filtering its fine octaves, avoiding the smooth bands caused by averaging a strongly sheared field on an unrelated radial step. The local photon and oblate-coordinate Jacobians estimate each cell's change in radius, canonical emission longitude, normalized height and emission time. Analytic differential rotation maps that span into the sheared lattice; quintic slope bounds include the added vertical warp. Fine octaves fade when their nominal lattice footprint grows from half to two cells, and absent frequencies avoid texture sampling. Each time cohort has its own footprint before temporal blending. These are line-footprint and scheduling approximations, not exact cell-average extinction or a propagated pixel beam. Nonlinear transfer convergence and lensing magnification still require separate checks. Uniform material bypasses procedural work.

In Schwarzschild, $\mathcal R=E^2r^4+K r(2-r)>0$ for $0<r<2$ and nonzero E: a ray directed inward along the backward path cannot turn there. Once it lies below every source, its singularity destination follows from this causal condition. The inspector retains the integrated interior endpoint rather than inventing intermediate positions at the singularity. This analytic termination applies only to the Schwarzschild source-free interior, not to charged or rotating extensions.

The heated annulus uses canonical Cartesian momentum with a pole-free oblate position and null-chart time. Its smooth density boundary is traversed with the Hamiltonian before conversion back to separable flow. The canonical solver owns departure across the boundary, including when the starting radius is exactly representable on it. Inside this field, material scales still constrain Hamiltonian steps and transfer receives the canonical photon directly.

### Celestial source integration

A deterministic u32 hash generates the 64³ periodic scalar lattice on the GPU. Material samples use native texture interpolation after a quintic coordinate warp. The one-time sky generation loads eight lattice neighbors and interpolates explicitly in f32, avoiding hardware filtering precision differences in the independent source comparison. The same lattice supplies material structure and the synthetic Galactic band. The sky keeps three spectral coefficients separate from RGB until frequency transport.

Each cube texel's area is the sum of two spherical-triangle solid angles. For unnormalized corners $a,b,c$, each triangle uses

$$
\Omega=2\operatorname{atan2}\!\left(|a\cdot(b\times c)|,
|a||b||c|+(a\cdot b)|c|+(b\cdot c)|a|+(c\cdot a)|b|\right).
$$

The face-grid determinant reduces to the squared texel side length. This avoids subtracting four nearly equal corner primitives in f32. Each mip integrates four children by their areas, using f32 work textures before independent f16 output rounding. A binary64 rectangular-area reference checks source coefficients and integrated flux.

A single Morton ordering partitions the stellar catalogue by common spatial prefixes; bounds merge bottom-up into a preorder tree. Keys are 10-bit ordering coordinates only. Leaves preserve normalized f32 directions, spectra and repeated positions. Escape indices provide stackless traversal without a per-pixel candidate cap. The spatial-prefix principle follows [Karras](https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees); the host implementation is a serial range partition, not that paper's parallel GPU builder.

For detector increments $x,y$ in a source tangent plane, let $p=0.65^2$ and $s=10^{-12}$. Stellar covariance is $C=p(xx^T+yy^T)+sI$. Its determinant and the Gaussian quadratic form are evaluated as nonnegative sums:

$$
\det C=p^2(x\times y)^2+sp(|x|^2+|y|^2)+s^2,
$$

$$
d^TC^{-1}d=\frac{p[(d\times x)^2+(d\times y)^2]+s|d|^2}{\det C}.
$$

Here the two-dimensional cross product is a scalar determinant. These forms retain nearly rank-one footprints without subtracting large covariance terms. The kernel normalization is $2\pi\sqrt{\det C}$; circular 3.5-sigma support discards $e^{-3.5^2/2}\approx0.22\%$ of the infinite Gaussian tail. Direct binary64 source sums check tree queries and repeated positions. This is local linear image formation, not nonlinear caustic completeness.

Neighboring endpoints must share their source domain and equatorial-crossing order, with a spherical chord at most 1/8. A central derivative requires aligned, locally consistent one-sided slopes; otherwise the smaller admissible slope is used. An isolated direction uses the unlensed detector width. The same gradients filter the diffuse cube. This suppresses false footprints across image branches while retaining a local approximation near unresolved critical structure.

WGSL arithmetic remains f32. Guard unsafe operations before evaluation; `select` is not a lazy arithmetic guard. Independent references use the same quantized physical inputs and quantity-specific tolerances. [Coverage](coverage.md) records what has actually been exercised.
