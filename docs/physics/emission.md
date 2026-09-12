# Emission and source boundaries

Sources are prescribed on the [fixed spacetime](spacetime.md); [transport](transport.md) determines their observed radiation. They do not solve relativistic fluid dynamics.

Implementation: [disk preparation](../../src/physics/disk.ts), [synchrotron spectra](../../src/physics/synchrotron.ts), [material transfer](../../src/gpu/wgsl/imaging/matter.wgsl), [procedural structure](../../src/gpu/wgsl/imaging/structure.wgsl), and [sky inputs](../../src/physics/sky.ts).

## Source boundaries

Horizons do not absorb light. The [atlas](spacetime.md#analytic-extension-versus-implemented-atlas) determines connected domains; illumination is independent boundary data.

Proven source-free trapped characteristics use the zero-radiance homogeneous solution. Negative radial-potential barriers can establish exclusion from matter and infinity; exhausting the integration budget cannot. This boundary prescription does not make every bounded ray dark.

Each asymptotic end has a sky. Its direction is the limiting outward position direction, $\operatorname{sign}(r)\boldsymbol n$, and its future source frequency is $s_{\rm block}E>0$. Use the block orientation rather than $|E|$. Reciprocal-radius zero represents infinity; a finite cutoff does not establish the endpoint direction.

Universe 0, stationary side +1 is illuminated by default. Other-universe luminosity scales stellar/diffuse/jet light and disk thermal flux in other source domains, leaving extinction intact; it defaults to zero. Naked and disconnected components have their own illuminated boundary. Geometry and domain identity are unaffected.

## Disk visibility

A zero-thickness disk is an opaque equatorial annulus with two emitting faces. Trace to the first transverse crossing in the closed interval $[r_{\rm in},r_{\rm out}]$; crossings through the hole are not hits. Later images are not added through opaque foreground matter. Finite thickness instead uses ordered volume transfer.

Surface observers are rejected. An exactly coplanar incoming ray meets the first annulus radial edge; near-coplanar rays use ordinary crossings. This is a boundary convention, not an epsilon displacement.

## Neutral circular emitter

For neutral positive-radius equatorial circular motion with orbital orientation $s=\pm1$ and stationary time orientation $\eta=\pm1$,

$$
\Omega_s=\frac{s\sqrt{r-q^2}}{r^2+sa\sqrt{r-q^2}},\qquad
u_{\rm em}=\eta\widehat u^t(\partial_t+\Omega_s\partial_\phi),
$$

$$
\widehat u^t=[-(g_{tt}+2\Omega_sg_{t\phi}+\Omega_s^2g_{\phi\phi})]^{-1/2}.
$$

Require $r>q^2$, $\Delta>0$, finite coefficients and positive timelike norm. Production fixes $s=+1$; this is prograde only when $a>0$. The block selects $\eta$, which is independent of $s$. Radial force balance is $\partial_rg_{tt}+2\Omega_s\partial_rg_{t\phi}+\Omega_s^2\partial_rg_{\phi\phi}=0$; see [Pugliese, Quevedo and Ruffini](https://arxiv.org/abs/1303.6250).

With emitter constants $\varepsilon=-u_t$ and $\ell=u_\phi$,

$$
\mathcal R_m=[\varepsilon(r^2+a^2)-a\ell]^2-\Delta[r^2+(\ell-a\varepsilon)^2].
$$

At fixed $\varepsilon,\ell$, circular orbits satisfy $\mathcal R_m=\mathcal R_m^\prime=0$, radial stability has $\mathcal R_m^{\prime\prime}<0$, and marginal stability has $\mathcal R_m^{\prime\prime}=0$. The default subextremal inner edge follows the stable exterior branch connected to large radius and is rounded outward in f32. Other geometries use a truncated inner edge of 16 M, not a general naked-singularity ISCO. Explicit annuli are checked for sampled emitter/flux validity, not continuous stability.

## Disk emission

The flux per disk face follows the stationary zero-inner-torque conservation law of [Page and Thorne (1974), equations 11b–12](https://articles.adsabs.harvard.edu/pdf/1974ApJ...191..499P), using the neutral charged-orbit data on the $\eta=+1$ branch. Primes denote radial derivatives:

$$
F(r)=-\frac{\dot M\,\Omega'(r)}{4\pi r[\varepsilon(r)-\Omega(r)\ell(r)]^2}
\int_{r_{\rm in}}^r[\varepsilon(s)-\Omega(s)\ell(s)]\ell'(s)\,ds.
$$

Proper vertical height gives the reduced determinant factor $r$ in this equatorial law; the four-dimensional BL determinant remains $-\Sigma^2\sin^2\theta$. Production integrates Simpson cells between 512 log-radius nodes with unit accretion-rate normalization, then uploads $T_{\rm radial}/T_{\rm peak}=(F/F_{\max})^{1/4}$. This preserves the shape and $F(r_{\rm in})=0$, not an absolute accretion luminosity. Peak temperature is an independent 1000–30000 K control. No plunging emission or returning-radiation heating is included.

For $x=(r-r_{\rm in})/(r_{\rm out}-r_{\rm in})\in(0,1)$, set $e(x)=\sqrt{27x/4}(1-x)$, zero outside. It peaks at one at $x=1/3$. The coordinate scale height is $H=h_0re(x)$; the unperturbed vertical optical depth is $\tau_0N e(x)^2(r_{\rm in}/r)^2$. With $b=(r_{\rm out}-r_{\rm in})/r_{\rm in}$, its peak is at $x_*=2/[3+b+\sqrt{(3+b)^2+4b}]$, and $N=(1+bx_*)^2/e(x_*)^2$ normalizes it to one. For coordinate height $z=r\cos\theta$, corrugation $\delta z=0.6s_cCH$, width $H_c=(1+0.4s_cC)H$, and density factor $D$,

$$
\alpha_{\rm em}(r,z)=\frac{\tau_0N e(x)^2(r_{\rm in}/r)^2D}{\sqrt{2\pi}H_c}
 \exp\left[-\frac{(z-\delta z)^2}{2H_c^2}\right].
$$

The unstructured Gaussian ($s_c=0$) has the stated column in $dz$ before truncation. With structure, $C$ and $D$ also vary with height, so the local $1/H_c$ factor does not preserve the integrated column exactly. Neither construction guarantees a proper-height column away from the equator. $h_0=0$ selects the separate opaque surface model; it is not the finite-optical-depth limit of this Gaussian.

For strength $s_c\in[0,1]$, large-cloud field $C$ and fine field $f$, set $D=\exp[6s_c(0.5C+0.5f)]$ and multiply temperature by $\exp[0.9s_c(0.2C+0.8f)]$. Support is truncated at $|z|<(4.5+2.4s_c)H$. These are prescribed density/heating variations, not an equation of state. A periodic quintic lattice supplies clouds and up to six fine scales. Two finite-age cohorts blend over two inner-edge orbital periods; each is filtered before blending. See [material sampling](../numerics/image-formation.md#path-and-material-sampling).

At finite height, apply equatorial $\Omega(r)$ to a local circular velocity and require timelikeness. This motion is generally accelerated. The latitude normal supplies a Milne angular closure in scalar-absorption transfer; this is not a multiple-scattering solution. Zero thickness samples the equatorial surface closure.

Evaluate structure at the emission event in canonical ingoing longitude and null time. Outgoing paths transform coordinates first. Observer preparation retains its chart-time offset; playback adds the launch epoch.

## Prescribed jets

The optional source is a bipolar collimated outflow outside any outer horizon. Here $r_{\rm out}$ denotes the jet extent, independent of the disk outer edge. For $s=|z|/r_{\rm out}$ its cylindrical width is

$$
w(z)=r_{\rm out}\tan\theta_{\rm out}\,s^{0.6}\left(\frac{1+16s^2}{17}\right)^{0.2}.
$$

The angle fixes outer width. The inner near-parabolic profile approaches conical expansion, motivated by [M87 observations](https://arxiv.org/abs/1110.1793), without fitting that system. Its logarithmic width slope is $0.6+0.4(16s^2)/(1+16s^2)$. A circular ZAMO is boosted along the metric-projected tangent of this nominal streamline. A faster diffuse core, slower luminous sheath, helical displacement and knots prescribe the source structure.

Density scales as $[w(z_{\rm base})/w(z)]^2$ with smooth launch, outer and transverse tapers. This inverse-area prescription does not solve relativistic continuity or jet launching. Accepted local velocities are future timelike.

Electrons have $dn/d\gamma\propto\gamma^{-3}$ above $\gamma_{\min}$, with isotropically tangled field. Here $\gamma$ is electron Lorentz factor. For $F(x)=x\int_x^\infty K_{5/3}(z)\,dz$, with $\nu_c$ the single-electron critical frequency at $\gamma_{\min}$ for the chosen pitch angle, integrating the cutoff introduces $\int_0^{\nu/\nu_c}F(x)\,dx$ and gives $\nu^{1/3}$ / $\nu^{-1}$ low/high-frequency limits. Preparation integrates the kernel, pitch-angle average and CIE response into a shifted table; see [Dexter, Appendix A2](https://arxiv.org/pdf/1602.03184).

With $\rho=n/n_0$, $B/B_0=\sqrt\rho$ and $\varepsilon_e=-p\cdot u_e$, vacuum transfer samples that table at $\varepsilon_e/\sqrt\rho$ and multiplies by $\rho^{3/2}/\varepsilon_e^2$. The high-frequency tail therefore scales as $\rho^2/\varepsilon_e^3$. Disk opacity attenuates jet light in path order. Tangling gives zero net polarization; self-absorption and Faraday effects are absent.

## Distant sky

The [HYG 4.4 subset](../../NOTICE.md#hyg-star-catalogue) supplies 108,067 sources with $m_V\le10$. J2000 equatorial +x is zero right ascension, and +z is the north celestial pole. This fixed Earth-observed backdrop excludes parallax, proper motion, extinction correction and variability. Selection, binary layout and missing-color counts belong to NOTICE.

[Ballesteros' color-temperature approximation, equation 14](https://arxiv.org/abs/1201.1809) maps $b=B-V$ to a blackbody proxy:

$$
T(b)=4600\ \mathrm K\left(\frac1{0.92b+1.7}+\frac1{0.92b+0.62}\right).
$$

Missing colors use 6500 K. Scene luminance is $F_Y=4\times10^{-5}10^{-0.4m_V}$, approximating relative CIE Y by relative V flux with an artistic normalization. Stored amplitude is $F_Y/[Y(T)/Y(6500)]$ alongside temperature.

The synthetic Galactic band uses three blackbody coefficients at 4500, 6500 and 12000 K. The GPU generates 256-pixel cube faces and solid-angle-weighted mips. Coefficients are scaled by 1024 before f16 storage to limit underflow, then unscaled during sampling. The [J2000 Galactic orientation](https://docs.astropy.org/en/stable/_modules/astropy/coordinates/builtin_frames/galactic.html) sets its pole; brightness and dust are prescribed.

[Stellar filtering](../numerics/image-formation.md#celestial-source-integration) uses a local lensed footprint with a 0.65-pixel Gaussian detector PSF and $10^{-6}$ rad (0.206 arcsec) source width. These are regularization parameters, not measured stellar sizes. Isolated endpoint branches fall back to an unlensed footprint; nonlinear critical images remain approximate.

Sky brightness scales stars and diffuse light together. Plasma preview uses only diffuse light. A dark source never converts failed geometry into resolved darkness.
