# Physical model

Nullray traces vacuum light in a stationary Kerr–Newman spacetime to an opaque equatorial thermal disk or a fixed distant sky. Geometry, source emission, and display processing are separate contracts. The implementation is an optical prototype with the limitations recorded in [Coverage](coverage.md).

## Units and conventions

Use $G=c=M=1$ and metric signature $(-,+,+,+)$. Restore physical length with $r_g=GM/c^2$ and time with $t_g=GM/c^3$. The Schwarzschild horizon is at $2r_g$.

| Symbol                 | Meaning                                                        |
| ---------------------- | -------------------------------------------------------------- |
| $a$                    | Dimensionless spin $Jc/(GM^2)$                                 |
| $q$                    | Geometrized electric charge divided by mass                    |
| $(t,r,\theta,\phi)$    | Boyer–Lindquist coordinates                                    |
| $p^\mu$, $p_\mu$       | Future-directed photon tangent and its metric-lowered covector |
| $E=-p_t$, $L_z=p_\phi$ | Conserved Killing energy and axial angular momentum            |
| $\mathcal C$           | Carter constant in the convention below; it may be negative    |
| $\gamma$               | Unscaled Mino parameter, increasing toward the future          |
| $g$                    | Observed-to-emitted frequency ratio                            |

Changing mass at fixed normalized distances leaves optical geometry unchanged. Physical time and a mass-dependent emission prescription would change. Charge is not the Carter constant, and Killing energy is not locally measured energy.

## Metric and horizons

Define

$$
\Sigma=r^2+a^2\cos^2\theta,\qquad
\Delta=r^2-2r+a^2+q^2.
$$

Then

$$
ds^2=-\frac{\Delta}{\Sigma}(dt-a\sin^2\theta\,d\phi)^2
+\frac{\sin^2\theta}{\Sigma}[(r^2+a^2)d\phi-a\,dt]^2
+\frac{\Sigma}{\Delta}dr^2+\Sigma\,d\theta^2,
$$

$$
r_\pm=1\pm\sqrt{1-a^2-q^2}.
$$

The scene domain is subextremal, $a^2+q^2<1$, with an exterior camera. UI margins are practical placement limits; they do not establish accuracy over the entire domain. The shadow is a set of ray destinations, not the projection of a Euclidean horizon sphere. These conventions agree with [Wang, Lee, and Lin, Section II](https://arxiv.org/html/2208.11906v1#S2).

## Observer and backward tracing

The implemented observer is a stationary zero-angular-momentum observer (ZAMO). Away from the axis,

$$
A=(r^2+a^2)^2-a^2\Delta\sin^2\theta,\quad
\alpha=\sqrt{\Sigma\Delta/A},\quad
\omega=\frac{a(2r-q^2)}{A},
$$

$$
u_{\rm Z}=\alpha^{-1}(\partial_t+\omega\partial_\phi),\quad
e_{(r)}=\sqrt{\Delta/\Sigma}\,\partial_r,\quad
e_{(\theta)}=\Sigma^{-1/2}\partial_\theta,\quad
e_{(\phi)}=\frac{\sqrt{\Sigma/A}}{\sin\theta}\partial_\phi.
$$

This tetrad follows by orthonormalizing the metric. The axis uses a regular limiting frame; the horizon is excluded because a stationary ZAMO becomes singular there. Camera forward/right/up rotate the local spatial basis.

Let $n^i$ point from the camera toward the apparent source. An arriving, future-directed photon has

$$
p^\mu=\omega_{\rm obs}(u_{\rm obs}^\mu-n^ie_{(i)}^\mu),\qquad
\omega_{\rm obs}=1.
$$

Keep this momentum convention and evaluate toward decreasing affine/Mino parameter. The shader uses positive backward time $\tau=\gamma_{\rm obs}-\gamma$, so all path derivatives reverse consistently. A past-directed tracing direction must not be mixed into future-directed frequency contractions.

Inside the ergosphere, positive local energy can coexist with zero or negative $E$. The solver retains unnormalized $E,L_z,\mathcal C$; it never divides all photon data by $E$. Navigation changes a sequence of stationary observer placements and orientations. It is not a physical velocity or dynamically integrated worldline; moving-observer aberration is not implemented.

## Separated geodesics

For affine parameter $\lambda$, set $d\gamma=d\lambda/\Sigma$ and $P(r)=E(r^2+a^2)-aL_z$. The Carter constant and potentials are

$$
\mathcal C=p_\theta^2-a^2E^2\cos^2\theta+L_z^2\cot^2\theta,
$$

$$
R(r)=P(r)^2-\Delta[(L_z-aE)^2+\mathcal C],\qquad
\Theta(\theta)=\mathcal C+a^2E^2\cos^2\theta-L_z^2\cot^2\theta.
$$

Away from coordinate singularities,

$$
\frac{dr}{d\gamma}=s_r\sqrt R,\quad
\frac{d\theta}{d\gamma}=s_\theta\sqrt\Theta,
$$

$$
\frac{d\phi}{d\gamma}=\frac{aP}{\Delta}+\frac{L_z}{\sin^2\theta}-aE,\qquad
\frac{dt}{d\gamma}=\frac{(r^2+a^2)P}{\Delta}+aL_z-a^2E\sin^2\theta.
$$

The signs are those of future-directed derivatives and change at simple turns. Numerical evaluation tracks phase rather than perpetuating an initial sign. Literature normalized by $E$ also changes its Mino scale; adapting those formulas requires consistent conversion, especially when $E\le0$. [Numerics](numerics.md) owns the inverse-radius and real-elliptic implementation.

Boyer–Lindquist coordinates are the formula boundary. Other coordinate systems require explicit Jacobians for vectors and inverse transformations for covectors. A spherical navigation chart or oblate display map is not automatically a Kerr–Schild transformation.

## Source boundaries

In backward tracing, capture means reaching the inner dark boundary before a source. This stationary exterior model assigns no incident radiation from the past-horizon extension. It does not simulate collapse, Hawking emission, or an observer crossing the horizon.

The sky is at asymptotic infinity in the stationary frame. Its direction is the limiting outward position direction of the backward ray, opposite the future-directed spatial photon momentum there. The source energy is $E$, so $g_{\rm sky}=1/E$ only for $E>0$. A zero/negative-energy ray cannot become a sky sample. Reaching a merely large radius is insufficient without controlling the angular tail.

## Disk visibility

Both faces of an optically thick equatorial annulus emit isotropic local specific intensity. Trace backward to the first transverse crossing in the closed interval $[r_{\rm in},r_{\rm out}]$. A crossing through the central hole is not a hit; later crossings are not added through an opaque foreground disk. Repeated apparent images arise from different paths.

Observers on the emitting surface are rejected. Exactly coplanar overlap has no transverse intersection and remains an explicit surface degeneracy. Near-coplanar rays remain ordinary cases. No arbitrary thickness or epsilon displacement resolves this physical convention.

## Neutral circular emitter

The disk follows neutral circular geodesics appropriate to the charged metric. For orbital orientation $s=\pm1$,

$$
\Omega_s=\frac{s\sqrt{r-q^2}}{r^2+sa\sqrt{r-q^2}},\qquad
u_{\rm em}=u^t(\partial_t+\Omega_s\partial_\phi),
$$

$$
u^t=[-(g_{tt}+2\Omega_sg_{t\phi}+\Omega_s^2g_{\phi\phi})]^{-1/2}.
$$

Require real orbital frequency and positive timelike normalization. These expressions follow from the radial circular-geodesic condition; see [Pugliese, Quevedo, and Ruffini](https://arxiv.org/abs/1303.6250). The product fixes $s=+1$ across signed spin. Prograde means $sa>0$; it is undefined at zero spin.

With emitter constants $\varepsilon=-u_t$ and $\ell=u_\phi$,

$$
\mathcal R_m=[\varepsilon(r^2+a^2)-a\ell]^2-\Delta[r^2+(\ell-a\varepsilon)^2].
$$

Circular orbits satisfy $\mathcal R_m=\partial_r\mathcal R_m=0$. Marginal radial stability adds $\partial_r^2\mathcal R_m=0$ at fixed $\varepsilon,\ell$. The ISCO is the boundary of the stable exterior branch connected to large radius. The CPU checks this branch and rounds the inner edge toward its stable side in f32. The outer radius is 64. There is no plunging emission or returning-radiation heating.

For the implemented $s=+1$ shader, set $u=1/r$ and $v=u\sqrt{u(1-q^2u)}$. Cancelling the common orbital denominator in the frequency contraction gives

$$
\Omega=\frac{v}{1+av},\qquad
g=\frac{\sqrt{1-3u+2q^2u^2+2av}}{E+(aE-L_z)v}.
$$

The radicand, orbital denominator, and emitted-energy denominator must be positive. This algebraic form avoids rebuilding the equatorial metric in the shader; GPU checks compare it with the CPU's metric contraction, including zero and negative Killing energy.

## Disk emission

The stationary zero-inner-torque flux follows [Page and Thorne (1974), equations 11b–12](https://articles.adsabs.harvard.edu/pdf/1974ApJ...191..499P), with the neutral charged-orbit quantities above:

$$
F(r)=-\frac{\dot M\,\Omega'(r)}{4\pi r[\varepsilon(r)-\Omega(r)\ell(r)]^2}
\int_{r_{\rm in}}^r[\varepsilon(s)-\Omega(s)\ell(s)]\ell'(s)\,ds.
$$

Using proper vertical height at the equator gives the reduced metric factor $r$. The CPU integrates on logarithmic radial nodes and uploads $[F/F_{\max}]^{1/4}$. Peak temperature, initially 7000 K and adjustable over 1000–30000 K, is an appearance normalization rather than inferred mass or accretion rate. The charged disk is a neutral test-emitter model, not a plasma equilibrium.

Structure multiplies bolometric flux by $1+cS$, with $0\le c\le1$. A bounded sum of four azimuthal harmonics has zero azimuthal mean. Smooth interpolation between deterministic radial phase nodes controls spatial coherence. Thus modulation is nonnegative and preserves azimuthally averaged bolometric flux at a fixed coordinate time; it need not preserve observer-band brightness.

Evaluate structure at $t_{\rm em}=t_{\rm obs}+\Delta t$, with advection at coordinate angular velocity $\Omega_s$. Smoothly replaced 24-unit cohorts limit indefinite winding. CPU epoch splitting keeps the local time small before GPU upload; cohort integers still have f32 range limits, and long propagation delays retain optical uncertainty. These prescribed fluctuations are inspired by correlated emission models such as [Lee and Gammie](https://arxiv.org/abs/2011.07151), but are not a GRMHD or stochastic-field simulation.

## Distant sky

The attributed [HYG 4.1 subset](../NOTICE.md#hyg-bright-star-catalogue) contains 8,920 sources with apparent visual magnitude $m_V\le6.5$. J2000 equatorial axes place +x at zero right ascension and +z at the north celestial pole. This is a fixed Earth-observed celestial backdrop, not a relocated three-dimensional Galaxy. Parallax, proper motion, extinction correction, and variability are excluded.

[Ballesteros' color-temperature approximation](https://arxiv.org/abs/1201.1809) maps B−V to a blackbody proxy. Forty missing colors use an explicit 6500 K assumption. Scene luminance is $F_Y=4\times10^{-5}10^{-0.4m_V}$: relative V flux approximates relative CIE Y, and the overall scale is artistic. Source storage retains $F_Y/[Y(T)/Y(6500)]$ and temperature; it does not rasterize a finite angular star profile.

Diffuse radiation is a separate synthetic Galactic band represented by three blackbody basis coefficients at 4500, 6500, and 12000 K. Its cube mips conserve projected solid-angle-weighted flux. A factor of 1024 before half-float storage reduces underflow and is removed at sampling. The pole follows the [J2000 Galactic orientation](https://docs.astropy.org/en/stable/_modules/astropy/coordinates/builtin_frames/galactic.html); brightness structure remains prescribed.

Sky brightness scales both source components linearly from zero to eight. It cannot change endpoint geometry or frequency. Zero sky brightness avoids unnecessary source filtering for an already resolved sky endpoint; failed geometry remains failed.

## Spectral transfer and display

Use future-directed endpoint momenta:

$$
g=\frac{-p_\mu u^\mu_{\rm obs}}{-p_\mu u^\mu_{\rm em}},\qquad
I_{\nu,\rm obs}(\nu)=g^3 I_{\nu,\rm em}(\nu/g).
$$

For the disk, $g=1/[u^t(E-\Omega_sL_z)]$. The denominator must be positive even when Killing energy is not. For a blackbody,

$$
g^3B_{\nu/g}(T)=B_\nu(gT).
$$

Evaluate the shifted temperature once; do not multiply by another $g^3$. Bolometric intensity scales as $g^4$, which is not a substitute for a frequency-resolved spectrum. Lensing changes apparent solid angle, not local surface brightness at fixed shift. Point-source amplification must emerge from the image footprint once. See [general-relativistic radiative transfer](https://arxiv.org/abs/1207.4234).

Planck radiance is integrated against all 1 nm [CIE 1931 two-degree samples](https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer) from 360 to 830 nm. The global normalization sets $Y(6500\,\mathrm K)=1$. [NOTICE](../NOTICE.md) records data attribution and checksums. A 1024-entry log-temperature table spans 100–1,000,000 K; 0–100 K resolves as negligible visible radiance, while negative or higher temperatures remain unresolved.

Radiance stays in signed linear sRGB through optical filtering. The shared presentation shader converts to linear Display P3 at D65 using [W3C conversion coefficients](https://www.w3.org/TR/css-color-4/#color-conversion-code), clips negative destination channels, and applies exposure $2^{EV}$. For exposed color $c$, the curve is

$$
c_{\rm out}=\frac{Hc}{H+\max(c)},\qquad H=1\ \text{(SDR)},\quad H=4\ \text{(HDR)}.
$$

This common scale preserves channel ratios. It is an artistic bounded display curve, not a physical luminosity or measured monitor peak. The P3 transfer function is encoded once into an `rgba16float`, `display-p3` canvas. HDR requests extended tone mapping; automatic selection follows `(dynamic-range: high)`. Actual headroom belongs to the browser, operating system, and display. [WebGPU canvas tone mapping](https://www.w3.org/TR/webgpu/#gpucanvastonemapping) defines that contract.

## Pixel formation and limits

Diffuse filtering and stellar integration use the escaped ray map. A point source is weighted by a unit-integral screen-space tent and divided by its local source solid-angle Jacobian. This already accounts for magnification. The approach follows the source distinction in [Bruneton](https://arxiv.org/abs/2010.08735), but its affine beam approximation does not resolve all nonlinear or multiple images.

Pixel integration must include coverage, branch changes, source variation, and failed sample weights. Narrow disk images and photon-ring structure arise from the physical paths; their resolved appearance depends on sampling and precision. [DNGR](https://arxiv.org/abs/1502.03808) and [AART](https://arxiv.org/abs/2211.07469) motivate beam-aware and targeted sampling, not a claim that the present implementation has achieved their coverage.

[The numerical method](numerics.md#pixel-integration) describes the implemented finite work. Complete critical-ray accuracy, caustic-resolved stellar flux, and joint-domain image convergence remain open.
