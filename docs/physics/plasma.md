# Narrowband plasma refraction

This mode traces one observing frequency through prescribed cold, collisionless, nonmagnetized plasma. Heating changes refractive density; vacuum WP transport and catalogue stars are bypassed.

Implementation: [profile coefficients](../../src/physics/plasma.ts), [input validation](../../src/scene/plasma.ts), [canonical GPU flow](../../src/gpu/wgsl/geodesics/medium.wgsl), and [brightness preview](../../src/gpu/wgsl/imaging/transfer.wgsl).

## Normalized Hamiltonian

Normalize momentum by detector angular frequency $\omega_o=2\pi\nu_o$:

$$
\chi(x)=\frac{\omega_p^2(x)}{\omega_o^2}=\frac{\nu_p^2(x)}{\nu_o^2},\qquad
\mathcal H=\frac12(g^{\mu\nu}p_\mu p_\nu+\chi)=0.
$$

Every $p$ below is detector-normalized. Along its rescaled affine parameter,

$$
\frac{dx^\mu}{d\lambda}=g^{\mu\nu}p_\nu,\qquad
\frac{dp_\mu}{d\lambda}=-\frac12(\partial_\mu g^{\alpha\beta})p_\alpha p_\beta
 -\frac12\partial_\mu\chi.
$$

See [Perlick and Tsupko, Sections II–III](https://arxiv.org/html/1702.08768v2), translating their past-directed convention. Substituting Kerr–Newman $\Delta$ preserves the separation below.

At the detector,

$$
p_o^\mu=u_o^\mu-\sqrt{1-\chi_o}\,\ell^i e_{(i)}^\mu.
$$

Require $0\le\chi_o<1$. At an emitter, $\varepsilon_e=-p\cdot u_e>0$, $g=1/\varepsilon_e$ and $\mathfrak n_e^2=1-\chi_e/\varepsilon_e^2>0$. Nonpropagating source evaluations are rejected. For $\chi>0$, the path is not a vacuum null geodesic.

## Separable background

With density normalization $n_0$ in cm⁻³ and scale $r_0$ in $r_g$,

$$
n_e=n_0\frac{r_0^2r^2}{\Sigma(r^2+r_0^2)},\qquad
\chi=\frac{f_r(r)}{\Sigma},\qquad
f_r=A_p\frac{r^2}{r^2+r_0^2},\qquad
A_p=\frac{\nu_{p0}^2}{\nu_o^2}r_0^2.
$$

The SI conversion is

$$
\nu_{p0}^2=\frac{(10^6n_0)e^2}{4\pi^2\epsilon_0m_e}.
$$

The factor $10^6$ converts cm⁻³ to m⁻³; constants follow [CODATA 2022](https://www.physics.nist.gov/cuu/pdf/wallet_2022.pdf). At the equator, $n_e(r_0)=n_0/2$.

The profile is even in $r$, decays as $r^{-2}$ and has $f_\theta=0$. In the unheated separable region $E,L,\mathcal C$ are constant:

$$
R=P^2-\Delta(K+f_r),\qquad
\Theta=\mathcal C+a^2E^2\cos^2\theta-L^2\cot^2\theta.
$$

Radial acceleration differentiates the entire $R$, including $f_r$. Observer free fall remains a vacuum timelike geodesic.

## Heated annulus

For $r_0<r<3r_0$, use canonical ingoing longitude $\psi_+$ and null time $w_+$. Define

$$
s=\frac{r-r_0}{2r_0},\quad W(s)=64s^3(1-s)^3,\quad
\omega_h=\frac{1.5}{r_0^{3/2}},
$$

$$
f=\frac12\left[1+\sin^6\theta\cos(6\psi_+-\omega_h w_++2\pi s)\right],\qquad
\frac{T}{T_0}=1+c_hWf,\quad T_0=10^5\ \mathrm K,
$$

$$
\chi_{\rm heated}=\frac{\chi_{\rm background}}{1+c_hWf},\qquad 0\le c_h\le9.
$$

$W,W^\prime,W^{\prime\prime}$ vanish at both edges. Angular pattern speed is $\omega_h/6$. Scene validation places the annulus outside the ergoregion with a conservative timelike-pattern bound. Constant-pressure density reduction is prescribed, not a solved energy equation.

Inside the annulus, Hamiltonian evolution includes all time/spatial gradients: $p_t,p_\phi$ and Carter data are no longer conserved. Both charts evaluate the canonical ingoing pattern. Outside, the solver reconstructs separated constants from the exiting state.

## Radiation preview and limits

For the two transverse modes of this isotropic dispersion relation, phase-space occupation satisfies $I_\nu=(2h\nu^3/c^2)\mathfrak n^2f_{\rm occ}$. Collisionless transport conserves $f_{\rm occ}$, hence $I_\nu/(\nu^3\mathfrak n^2)$. This is the unmagnetized reduction of [occupation-number transfer](https://arxiv.org/pdf/astro-ph/0311360), not a vacuum-intensity invariant. With the adopted thermal equilibrium boundary $I_{\nu,e}=\mathfrak n_e^2B_\nu(T)$,

$$
I_{\nu,o}(\nu_o)=\mathfrak n_o^2B_{\nu_o}(gT).
$$

Thermal sources contribute normalized Rayleigh–Jeans brightness temperature before display mapping:

$$
\frac{T_b}{6500\ \mathrm K}
 =\frac{\mathfrak n_o^2}{6500\ \mathrm K}
 \frac{T_q}{\exp[T_q/(gT)]-1},\qquad T_q=\frac{h\nu_o}{k_B}.
$$

$T_b$ is not a Planck-inverted thermodynamic temperature. The shader uses a small-argument expansion; diffuse sky uses the same thermal basis at mip zero. The [jet table](../../src/physics/synchrotron.ts) instead divides its intensity by $B_{\nu_o}(6500\,\mathrm K)$, rather than the Rayleigh–Jeans reference $2k_B(6500\,\mathrm K)\nu_o^2/c^2$. These agree only in the low-frequency limit: mixed-source output is not a uniformly calibrated brightness-temperature image. Q/U are zero.

Broadband dispersion, magnetized birefringence, Faraday effects, collisional absorption and backreaction are absent. This source/medium combination is a narrowband preview, not a calibrated radio observation.
