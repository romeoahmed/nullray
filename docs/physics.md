# Physical model

Nullray traces light in a stationary Kerr–Newman spacetime through a finite thermal disk and prescribed jets to a fixed distant sky. Vacuum optical imaging and a narrowband cold-plasma preview share the geometric solver. Geometry, source emission, and display processing are separate contracts. The implementation is an optical prototype with the limitations recorded in [Coverage](coverage.md).

## Scope

The background is the stationary Einstein–Maxwell Kerr–Newman solution. Its analytic extension is a mathematical model, not a simulation of gravitational collapse or a claim of traversable astrophysical interiors. The inner Cauchy horizon has a separate perturbative stability problem; the [Dafermos–Luk result](https://arxiv.org/abs/1710.01722) concerns continuous extendibility in perturbed vacuum Kerr, not a charged accretion flow.

Ingoing and outgoing charts cover parts of the extension; chart identity and spacetime-block identity must remain distinct. Negative-radius ends and further positive-radius exterior copies are different destinations. [Adamo and Newman, Sections 3.2–3.3](https://arxiv.org/html/1410.6626v2) describe the horizon and singularity structure with the opposite metric signature, which is translated below.

Matter is prescribed on this fixed geometry. The disk is a neutral circular-emitter model with a finite pressure-supported atmosphere. The jets are test outflows. Plasma heating changes a specified refractive Hamiltonian. None of these evolves the metric, fluid dynamics or magnetic launching. Vacuum polarization uses hidden symmetry; refracted paths require a separate transport law.

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

The geometry distinguishes subextremal pairs, an extremal horizon, and superextremal naked singularities. Schwarzschild has one regular horizon; its zero radial root is singular. Stationary limits satisfy $r=1\pm\sqrt{1-q^2-a^2\cos^2\theta}$ where real. The curvature singularity is $\Sigma=0$. For nonzero spin, the regular part of $r=0$ connects to negative radius. Scene preparation accepts regular placements with a valid timelike observer; the controls expose both charts, signed radius, and the available observer models. The shadow is a set of ray destinations, not the projection of a Euclidean horizon sphere. These conventions agree with [Wang, Lee, and Lin, Section II](https://arxiv.org/html/2208.11906v1#S2).

## Observer and backward tracing

The metric is evaluated in ingoing or outgoing Cartesian Kerr–Schild coordinates $(T,X,Y,Z)$. With chart sign $\epsilon=\pm1$, rotating longitude $\psi$, and unit direction $\boldsymbol n=(\sin\theta\cos\psi,\sin\theta\sin\psi,\cos\theta)$,

$$
\boldsymbol X=r\boldsymbol n+\epsilon a\boldsymbol e_z\times\boldsymbol n,\qquad
 g_{\mu\nu}=\eta_{\mu\nu}+\frac{2r-q^2}{\Sigma}k_\mu k_\nu,\qquad
 k_\mu=(1,\epsilon\boldsymbol n).
$$

The chart longitude is not Boyer–Lindquist $\phi$. Tangent transformations use explicit Jacobians. Horizon-regular geometry supplies an orthonormal reference frame; its worldline is accelerated. Local Lorentz boosts construct a physical observer. Static observers require a timelike stationary Killing vector; circular ZAMOs require a timelike zero-angular-momentum worldline. Neither is available everywhere. Custom velocity means $d(X,Y,Z)/dT$ in the selected chart and must lie inside the local future light cone.

Free fall is a neutral timelike geodesic launched with a measured velocity in the regular frame and advanced by proper time. Its current camera frame is reconstructed by a local boost at the endpoint; it is not a parallel-transported gyroscope. Camera forward/right/up rotate the prepared spatial triad.

Let $n^i$ point from the camera toward the apparent source. An arriving, future-directed photon has

$$
p^\mu=\omega_{\rm obs}(u_{\rm obs}^\mu-n^ie_{(i)}^\mu),\qquad
\omega_{\rm obs}=1.
$$

Keep this momentum convention and evaluate toward decreasing affine/Mino parameter. The integrator uses negative Mino steps for backward rays. A past-directed tracing direction must not be mixed into future-directed frequency contractions.

Inside the ergosphere, positive local energy can coexist with zero or negative $E$. The solver retains unnormalized $E,L_z,\mathcal C$; it never divides all photon data by $E$. Navigation changes placement and orientation. The explicitly selected observer model supplies velocity, aberration, and Doppler shift.

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

The signs are those of future-directed derivatives and change at simple turns. Numerical evaluation tracks phase rather than perpetuating an initial sign. Literature normalized by $E$ also changes its Mino scale; adapting those formulas requires consistent conversion, especially when $E\le0$. [Numerics](numerics.md) owns the regular chart transport, radial coordinate changes, and numerical integration. Timelike motion adds the mass terms documented there.

Boyer–Lindquist coordinates are the formula boundary. Other coordinate systems require explicit Jacobians for vectors and inverse transformations for covectors. A spherical navigation chart or oblate display map is not automatically a Kerr–Schild transformation.

## Source boundaries

Horizons are chart/block events, not opaque capture surfaces. The analytic extension distinguishes black-hole, white-hole, inner, and repeated exterior blocks. Negative radius and a further positive-radius universe are different destinations. For zero spin, the entire zero-radius surface is singular, so a directly placed negative-radius observer belongs to a disconnected negative-mass component; it is not a black-hole interior connected to the positive-radius extension. Crossing the ring terminates a ray; exhausted numerical work stays unresolved. The current atlas coverage and its event limitations are recorded in [Numerics](numerics.md#ordered-events), not inferred from a successful horizon crossing.

Source-free trapped characteristics use the zero-radiance homogeneous solution. This is an explicit boundary prescription for rays that never meet the disk, jet or asymptotic sky. A negative radial potential at two radii enclosing the observer can prove that exclusion; numerical budget exhaustion alone cannot. Such rays have a distinct source-free outcome, separate from singularity termination. This prescription does not infer that every bounded geodesic is dark.

The sky is specified at each asymptotic end. Its direction is the limiting outward spatial position direction of the backward ray. A block's time orientation determines the future stationary source observer, so the source frequency is its signed contraction with the photon, not an absolute value imposed on Killing energy. The reciprocal-radius endpoint represents infinity; a finite-radius cutoff does not establish an asymptotic direction.

The original positive-radius exterior supplies the default illumination. The other-universe luminosity control scales stellar, diffuse and jet light and disk thermal flux in the remaining source domains; its default is zero. Disk extinction remains present there. This boundary prescription leaves all geometric passages open. Naked and disconnected components have their own illuminated boundary. Domain identity remains available independently of the light in it.

## Disk visibility

At zero thickness, both faces of an optically thick equatorial annulus emit through a conservative electron-scattering atmosphere. Trace backward to the first transverse crossing in the closed interval $[r_{\rm in},r_{\rm out}]$. A crossing through the central hole is not a hit; later crossings are not added through an opaque foreground disk. Repeated apparent images arise from different paths. Finite thickness instead integrates the volume in path order, as described below.

Observers on the emitting surface are rejected. An exactly coplanar ray arriving from outside the annulus meets its first radial edge; this is the adopted opaque surface boundary convention. Near-coplanar rays remain ordinary cases. No arbitrary thickness or epsilon displacement resolves this physical convention.

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

Circular orbits satisfy $\mathcal R_m=\partial_r\mathcal R_m=0$. Marginal radial stability adds $\partial_r^2\mathcal R_m=0$ at fixed $\varepsilon,\ell$. The ISCO is the boundary of the stable exterior branch connected to large radius. For subextremal defaults, the CPU checks this branch and rounds the inner edge toward its stable side in f32. The general source default has outer radius 32; the initial photograph uses 14. Other geometries default to a truncated annulus beginning at 16; this is source boundary data, not a general naked-singularity ISCO result. Explicit annuli must have finite supported circular-emitter profiles. There is no plunging emission or returning-radiation heating.

For $s=+1$, set $u=1/r$ and $v=u\sqrt{u(1-q^2u)}$. Cancelling the common orbital denominator in the frequency contraction gives

$$
\Omega=\frac{v}{1+av},\qquad
g=\frac{\sqrt{1-3u+2q^2u^2+2av}}{E+(aE-L_z)v}.
$$

The radicand, orbital denominator, and emitted-energy denominator must be positive. GPU checks compare this algebraic form with metric contraction, including zero and negative Killing energy. The image pass currently evaluates the emitter directly in its regular chart.

## Disk emission

The stationary zero-inner-torque flux follows [Page and Thorne (1974), equations 11b–12](https://articles.adsabs.harvard.edu/pdf/1974ApJ...191..499P), with the neutral charged-orbit quantities above:

$$
F(r)=-\frac{\dot M\,\Omega'(r)}{4\pi r[\varepsilon(r)-\Omega(r)\ell(r)]^2}
\int_{r_{\rm in}}^r[\varepsilon(s)-\Omega(s)\ell(s)]\ell'(s)\,ds.
$$

Using proper vertical height at the equator gives the reduced metric factor $r$. The CPU integrates on logarithmic radial nodes and uploads $[F/F_{\max}]^{1/4}$. Peak temperature, initially 2800 K and adjustable over 1000–30000 K, is an appearance normalization rather than inferred mass or accretion rate. The charged disk is a neutral test-emitter model, not a plasma equilibrium.

The radial envelope is $e(x)=\sqrt{27x/4}(1-x)$ for $x=(r-r_{\rm in})/(r_{\rm out}-r_{\rm in})$ in $(0,1)$ and zero outside. It has unit maximum at $x=1/3$. Scale height is $H/r=h_0e(x)$. Vertical gray optical depth is $\tau_0N e(x)^2(r_{\rm in}/r)^2$ before density structure; defaults are $h_0=0.022$ and $\tau_0=1.8$. The factor $N$ normalizes its single interior peak to one. With $b=(r_{\rm out}-r_{\rm in})/r_{\rm in}$, that peak is at $x_*=2/[3+b+\sqrt{(3+b)^2+4b}]$. Inverse-square dilution concentrates the luminous column inward while the outer atmosphere becomes tenuous. This column is prescribed, not derived from fluid mass conservation. Gaussian vertical extinction is normalized by $\sqrt{2\pi}H$.

Large clouds and up to six finer scales have different roles in opacity and thermal emission. With structure strength $s$, large-cloud field $C$ and fine field $f$, density scales by $D=\exp[6s(0.4C+0.6f)]$ while temperature scales by $\exp[1.1s(0.2C+0.8f)]$. These prescribe distinct column and heating variations, rather than an equation of state. Large clouds displace the Gaussian center by $0.6sCH$ and multiply its height by $1+0.4sC$, retaining vertical normalization. Conservative support includes the displaced atmosphere through at least 4.5 local scale heights. The cached periodic lattice uses quintic interpolation; logarithmic radius and azimuth span planar coordinates while height has its own coordinate. The volume filters structure below the nominal transfer-cell footprint, as described in [Numerics](numerics.md#radiation-and-sampling). Common time cohorts blend over two inner-edge orbital periods, so shear depends on structure age rather than total playback time. Each cohort is filtered before blending to preserve continuity at renewal. This is a prescribed pressure-supported test atmosphere, not hydrodynamic equilibrium. Zero thickness samples the same thermal field at the equatorial Milne surface.

The volume uses the local circular Kerr–Newman emitter velocity where timelike. Its scattering angular law is the local plane-parallel Milne closure about the latitude normal. Along the backward ray, observer-frame emissivity $j$ and extinction $a$ in a cell of affine length $\ell$ update intensity and transmission by

$$
I\leftarrow I+\mathcal Tj\frac{1-e^{-a\ell}}a,\qquad
\mathcal T\leftarrow\mathcal T e^{-a\ell}.
$$

The zero-extinction limit is $j\ell$. Q/U use the same extinction and their own signed emissivities in the detector basis. This homogeneous-cell solution remains bounded at large optical depth, following the transfer strategy discussed by [Mościbrodzka and Gammie](https://arxiv.org/abs/1712.03057). It models polarized emission and scalar absorption; multiple scattering, dichroism, circular polarization and Faraday conversion are not solved.

The pattern is evaluated at the actual emission event in canonical ingoing longitude and null-chart time. Outgoing paths transform their endpoint coordinates before evaluating it. Free-fall preparation retains the change in chart time along the observer worldline; playback adds the launch epoch. Circular advection has the same angular velocity at fixed radius in these stationary coordinates.

## Prescribed jets

The optional source is a bipolar collimated outflow outside the outer horizon. For $s=|z|/r_{\rm out}$ its cylindrical width is

$$
w(z)=r_{\rm out}\tan\theta_{\rm out}\,s^{0.6}\left(\frac{1+16s^2}{17}\right)^{0.2}.
$$

The angle control sets the outer width. This joins an inner near-parabolic shape to approximately conical expansion, motivated by the observed transition in [M87 (Asada and Nakamura)](https://arxiv.org/abs/1110.1793), rather than fitted to that system. The local circular ZAMO is boosted along the same streamline's tangent after metric-orthogonal projection. Its logarithmic width slope is $0.6+0.4(16s^2)/(1+16s^2)$. A faster diffuse core and slower, broad luminous sheath have smooth transverse profiles, a small helical displacement and advected knots. The outer transverse taper fades to zero; the source has no hard luminous cone wall.

The density envelope follows $[w(z_{\rm base})/w(z)]^2$, the inverse cross-sectional area, with smooth launch/end tapers. This is a prescribed optical source, not an exact relativistic continuity solution or a dynamical jet-launching calculation. Its velocity remains future timelike in each illuminated exterior domain.

Electrons have $dn/d\gamma\propto\gamma^{-3}$ above a configurable $\gamma_{\min}$, and the field is isotropically tangled. For $F(x)=x\int_x^\infty K_{5/3}(z)\,dz$, finite-cutoff emissivity contains $\int_0^{\nu/\nu_c}F(x)\,dx$ instead of extending the high-frequency power law through all frequencies. This produces the $\nu^{1/3}$ low-frequency and $\nu^{-1}$ high-frequency limits. The worker evaluates the kernel, field-orientation average and CIE response, then uploads a shifted spectral table. See [Dexter, Appendix A2](https://arxiv.org/pdf/1602.03184).

With $\rho=n/n_0$ and $B/B_0=\sqrt\rho$, transfer samples that table at $\omega_{\rm em}/\sqrt\rho$ and scales it by $\rho^{3/2}/\omega_{\rm em}^2$. This recovers $\rho^2/\omega_{\rm em}^3$ on the high-frequency power-law tail. Disk opacity attenuates foreground and background jet contributions in path order. Tangling gives unpolarized jet emission. Defaults are $10^6$ solar masses, $n_0=10^6$ cm⁻³, $B_0=1000$ G and $\gamma_{\min}=600$. Self-absorption, Faraday effects and dynamical launching remain outside this prescription.

## Plasma refraction

The optional narrowband mode uses the collisionless, cold, nonmagnetized plasma Hamiltonian $H=\tfrac12(g^{\mu\nu}p_\mu p_\nu+\omega_p^2)=0$. The detector measures frequency $\nu_o$ and launches spatial momentum of magnitude $\sqrt{1-\nu_p^2/\nu_o^2}$ after normalizing its measured energy to one. A frequency at or below the detector's plasma cutoff is not a propagating camera ray. The Hamiltonian and separability condition follow [Perlick and Tsupko, Sections II–III](https://arxiv.org/pdf/1702.08768); replacing the radial metric function by Kerr–Newman $\Delta$ retains that separation.

The prescribed scalar density is

$$
n_e(r,\theta)=n_0\frac{r_0^2r^2}{\Sigma(r^2+r_0^2)},\qquad
f_r(r)=A\frac{r^2}{r^2+r_0^2},\qquad
A=\frac{\nu_{p0}^2}{\nu_o^2}r_0^2.
$$

Here $\nu_{p0}^2=n_0e^2/(4\pi^2\epsilon_0m_e)$ after converting cm⁻³ to m⁻³; constants use [CODATA 2022](https://www.physics.nist.gov/cuu/pdf/wallet_2022.pdf). This even, stationary profile is specified on both signed-radius sheets. It approaches inverse-square dilution at large radius and has $f_\theta=0$. With zero heating it is stationary; it is not a fluid equilibrium. Pressure, magnetic birefringence, collisions and absorption are excluded.

The radial potential becomes $R=P^2-\Delta(K+f_r)$, with $K=(L-aE)^2+C$. The angular system is unchanged. Evolving the derivative of this potential bends the trajectory; there is no image-space displacement. The [numerical implementation](numerics.md#radial-flow-and-integration) uses the same chart transitions, source ordering and unresolved outcomes as vacuum propagation.

Heating prescribes $T/T_0=1+cW(r)f(\boldsymbol n,v)$ above $T_0=10^5$ K in the compact annulus $r_0<r<3r_0$. Here $W=64s^3(1-s)^3$, $s=(r-r_0)/(2r_0)$, and $f$ is a bounded rotating sixfold pattern. Constant pressure gives $n_e/n_{\rm background}=T_0/T$. The annulus lies outside the ergoregion with a subluminal pattern. Its full Hamiltonian evolves both $p_t$ and $p_\phi$; a time-dependent, nonaxisymmetric refractive field does not preserve the vacuum Carter data. Heating is a prescribed thermal perturbation, not a fluid-energy solver.

Thermal occupation obeys phase-space conservation $I_\nu/(\nu^3n^2)$ along a transparent ray, giving $I_{\nu,o}=n_o^2B_\nu(gT)$. The monochrome preview displays brightness temperature divided by 6500 K before exposure. Jets use their spectrum at the same observing frequency. Vacuum Milne/WP polarization is disabled on refracted paths: the vacuum invariant does not supply plasma polarization transport. The model is not a visible-color radio photograph, and magnetized transport and collisional absorption are omitted.

## Distant sky

The rendered [HYG 4.1 subset](../NOTICE.md#hyg-star-catalogue) contains 108,071 sources with apparent visual magnitude $m_V\le10$. The 8,920 bright sources retain their catalogue records; a compact binary supplies 99,151 fainter sources. J2000 equatorial axes place +x at zero right ascension and +z at the north celestial pole. This is a fixed Earth-observed celestial backdrop, not a relocated three-dimensional Galaxy. Parallax, proper motion, extinction correction, and variability are excluded.

[Ballesteros' color-temperature approximation](https://arxiv.org/abs/1201.1809) maps B−V to a blackbody proxy. Forty bright and 319 faint sources with missing colors use an explicit 6500 K assumption. Scene luminance is $F_Y=4\times10^{-5}10^{-0.4m_V}$: relative V flux approximates relative CIE Y, and the overall scale is artistic. Source storage retains $F_Y/[Y(T)/Y(6500)]$ and temperature; it does not rasterize a finite angular star profile.

Diffuse radiation is a separate synthetic Galactic band with warped dust lanes, represented by three blackbody basis coefficients at 4500, 6500, and 12000 K. The GPU generates 256-pixel cube faces and solid-angle-weighted mips, conserving integrated coefficients up to storage and arithmetic rounding. A factor of 1024 before half-float storage reduces underflow and is removed at sampling. The pole follows the [J2000 Galactic orientation](https://docs.astropy.org/en/stable/_modules/astropy/coordinates/builtin_frames/galactic.html); brightness structure remains prescribed. Vacuum images filter the cube with local lensed direction gradients.

A separate pass integrates each stellar source over a locally linear lensed detector footprint. A Gaussian detector point-spread function has width 0.65 pixels; a prescribed source width of $10^{-6}$ rad regularizes finite-source caustics. These widths define an image-formation approximation, not measured angular sizes of the catalogue. Near source-domain discontinuities an isolated pixel uses the unlensed footprint. Nonlinear critical images are therefore approximate; the method does not certify all stellar images.

Sky brightness scales both stars and diffuse light linearly from zero to 64. It cannot change endpoint geometry or frequency. Setting source brightness to zero does not turn failed geometry into resolved darkness.

## Polarization

The principal conformal Killing–Yano tensor $h$ and its dual define the complex vacuum invariant $\kappa=h(f,p)+i\,{*h}(f,p)$. In Cartesian Kerr–Schild coordinates,

$$
h=dT\wedge(X\,dX+Y\,dY+Z\,dZ)+a\,dX\wedge dY.
$$

For a unit screen polarization vector, $|\kappa|^2=(L_z-aE)^2+\mathcal C$. The hidden symmetry persists for the charged radial metric function; see [Frolov, Krtouš, and Kubizňák, Sections 3.2 and 3.4](https://arxiv.org/html/1705.05482v3). The disk uses the conservative Milne atmosphere tabulated by [Silant’ev, Alekseeva and Ananjevskaja, Table 4, h = 0](https://doi.org/10.1093/mnras/stz123). The emitted electric vector lies parallel to the surface and perpendicular to the photon in the emitter frame. Angular intensity and polarized intensity are interpolated together, with $2\int_0^1\mu I(\mu)\,d\mu=1$ so the effective-temperature flux remains $\pi B_\nu(T)$.

Endpoint Walker–Penrose contractions transport this direction to the detector. Equatorial reflection symmetry supplies the principal-ray limit for coplanar disk emission; general degenerate screen reconstruction remains unresolved. Each detector screen uses projected camera up and its perpendicular right axis. Stokes I/Q/U are accumulated in this common convention before applying a linear analyzer:

$$
I_\alpha=\tfrac12[I+Q\cos(2\alpha)+U\sin(2\alpha)].
$$

The analyzer reuses the physical Stokes images. Vacuum transport does not create polarization from the unpolarized diffuse sky. Circular polarization is zero for the present sources.

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

Radiance stays in signed linear sRGB through optical filtering. The shared presentation shader converts to linear Display P3 at D65 using [W3C conversion coefficients](https://www.w3.org/TR/css-color-4/#color-conversion-code). Photographic white balance applies diagonal P3 gains that make a selected 2500–12000 K blackbody neutral while preserving that reference's Y luminance. This is camera calibration, independent of physical source temperature and frequency shift; diagnostic and monochrome plasma views omit it. Negative destination channels are clipped and exposure $2^{EV}$ is applied.

Vacuum photographs use a common highlight scale, retaining the color direction instead of compressing each channel separately. For exposed P3 color $c$ and content peak $H$, set $p=\max(c)/H$. Values $p\le0.7$ remain linear. Above that shoulder,

$$
p'=1-\frac{0.09}{p-0.4},\qquad w=0.08(p-p'),\qquad
c_{\rm out}=Hp'\frac{c/(Hp)+w\boldsymbol 1}{1+w}.
$$

The shoulder joins with unit slope, retains channel ordering and desaturates extreme highlights toward neutral. It adapts the peak-compression construction of [Khronos PBR Neutral](https://github.com/KhronosGroup/ToneMapping/tree/main/PBR_Neutral), with no surface-reflection offset; attribution and changes are in [NOTICE](../NOTICE.md#photographic-tone-mapping). Scientific false-color and monochrome plasma views retain the simpler per-channel curve

$$
c_{\rm out}=\frac{Hc}{H+c},\qquad H=1\ \text{(SDR)},\quad H=4\ \text{(HDR)}.
$$

Both are artistic bounded display curves, not physical luminosities or measured monitor peaks. Normalized bloom redistributes linear radiance before this transform, with equal energy per spatial octave and a fractional outer octave tied to image height. It preserves constant light and separates broad glare from the resolved disk. The P3 transfer function is encoded once into an `rgba16float`, `display-p3` canvas. HDR requests extended tone mapping; automatic selection follows `(dynamic-range: high)`. Actual headroom belongs to the browser, operating system, and display. [WebGPU canvas tone mapping](https://www.w3.org/TR/webgpu/#gpucanvastonemapping) defines that contract.

## Pixel formation and limits

The image pass traces jittered point samples and accumulates linear radiance with explicit resolved weight. Unresolved samples contribute zero radiation and keep their missing weight; a photograph reports their fraction separately. Diagnostic views mark that missing coverage in magenta. This presentation choice does not reclassify an unfinished trajectory as capture or certify its missing light. The following stellar pass uses the local footprint described above. The source distinction in [Bruneton](https://arxiv.org/abs/2010.08735) motivates treating a star's integrated flux separately from diffuse surface brightness.

Pixel integration must include coverage, branch changes, source variation, and failed sample weights. Narrow disk images and photon-ring structure arise from the physical paths; their resolved appearance depends on sampling and precision. [DNGR](https://arxiv.org/abs/1502.03808) and [AART](https://arxiv.org/abs/2211.07469) motivate beam-aware and targeted sampling, not a claim that the present implementation has achieved their coverage.

[The numerical method](numerics.md#radiation-and-sampling) describes the implemented finite work. Complete critical-ray accuracy, caustic-resolved stellar flux, and joint-domain image convergence remain open.
