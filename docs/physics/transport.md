# Radiation, polarization and display

Use the [future-momentum convention](observers.md#camera-rays). [Emission](emission.md) owns source prescriptions; [image formation](../numerics/image-formation.md) owns discretization.

Implementation: [spectral preparation](../../src/physics/radiation.ts), [vacuum invariants](../../src/gpu/wgsl/geodesics/geometry.wgsl), [polarimeter](../../src/gpu/wgsl/passes/polarimeter.wgsl), and [presentation](../../src/gpu/wgsl/passes/present.wgsl).

## Vacuum spectral transfer

Use future-directed endpoint momenta:

$$
g=\frac{-p_\mu u^\mu_{\rm obs}}{-p_\mu u^\mu_{\rm em}},\qquad
I_{\nu,\rm obs}(\nu)=g^3 I_{\nu,\rm em}(\nu/g).
$$

With unit detector frequency and [signed circular-emitter normalization](emission.md#neutral-circular-emitter), $g=1/[\eta\widehat u^t(E-\Omega_sL)]$. The denominator must be positive; do not replace $E$ with $|E|$. For blackbody radiation,

$$
g^3B_{\nu/g}(T)=B_\nu(gT).
$$

Evaluate $gT$ once without an extra $g^3$. Bolometric intensity instead scales as $g^4$. Lensing changes apparent solid angle; point-source amplification belongs in the footprint, not an additional radiance factor. See [covariant transfer](https://arxiv.org/html/1207.4234v1).

Per-frequency and per-wavelength radiance use different Jacobians:

$$
B_\nu(T)=\frac{2h\nu^3}{c^2[\exp(h\nu/k_BT)-1]},\qquad
B_\lambda(T)=\frac{2hc^2}{\lambda^5[\exp(hc/\lambda k_BT)-1]},
$$

$$
I_{\lambda,o}(\lambda)=g^5I_{\lambda,e}(g\lambda),\qquad
 g^5B_{g\lambda}(T)=B_\lambda(gT).
$$

The Planck helper returns per-wavelength radiance in W sr⁻¹ m⁻³. The $g^3$ and $g^5$ laws apply to different spectral densities.

## Spectral lookup and color

Integrate Planck radiance over all 1 nm [CIE 1931 two-degree samples](https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer), 360–830 nm: $XYZ=\int B_\lambda(T)(\bar x,\bar y,\bar z)\,d\lambda$. Trapezoidal spacing is $10^{-9}$ m; normalize $Y(6500\,\mathrm K)=1$. A 1024-entry log-temperature table covers 100–1,000,000 K. Below 100 K, nonnegative temperatures resolve as negligible visible light; negative or higher temperatures are unresolved. [NOTICE](../../NOTICE.md) owns data provenance.

Optical storage/filtering uses signed linear sRGB coefficients, not three separately traced wavelengths. Presentation converts to linear Display P3/D65 using [W3C coefficients](https://www.w3.org/TR/css-color-4/#color-conversion-code), clips negative destination channels, applies diagonal white-balance gains and exposure $2^{EV}$. White balance neutralizes a selected 2500–12000 K blackbody while preserving its reference Y; diagnostic/plasma views omit it.

## Display transform

For nonnegative exposed P3 color $c$ and content peak $H$, set $p=\max(c)/H$. Vacuum presentation is linear for $p\le0.7$; above that,

$$
p'=1-\frac{0.09}{p-0.4},\qquad w=0.08(p-p'),\qquad
c_{\rm out}=Hp'\frac{c/(Hp)+w\boldsymbol 1}{1+w}.
$$

This joins with unit slope and desaturates extreme highlights. It adapts [Khronos PBR Neutral](https://github.com/KhronosGroup/ToneMapping/tree/main/PBR_Neutral); [NOTICE](../../NOTICE.md#photographic-tone-mapping) records the changes. Diagnostic/plasma views use

$$
c_{\rm out}=\frac{Hc}{H+c}.
$$

Use $H=1$ in SDR and $H=4$ in HDR; these are display parameters, not physical luminosities. Before tone mapping, bloom mixes $I_{\rm glare}=(1-b)I+b\mathcal B[I]$ under the [finite-filter assumptions](../numerics/image-formation.md#bloom). Encode the Display P3 transfer function once into the `rgba16float` canvas. HDR requests extended tone mapping; Auto follows `(dynamic-range: high)`. Actual headroom depends on the browser/OS/display, as specified by [WebGPU canvas tone mapping](https://www.w3.org/TR/webgpu/#gpucanvastonemapping).

## Polarization

For an affinely propagated vacuum null tangent $p$ and parallel-transported unit screen vector $f$, $f\cdot p=0$. The closed conformal Killing–Yano tensor $h$ and its Hodge dual define $\kappa=h(f,p)+i\,{*h}(f,p)$. In the Cartesian KS convention,

$$
h=dT\wedge(X\,dX+Y\,dY+Z\,dZ)+a\,dX\wedge dY.
$$

Use orientation $\epsilon_{TXYZ}=+1$. Screen gauge $f\mapsto f+\zeta p$ leaves $\kappa$ unchanged, and $|\kappa|^2=K=(L-aE)^2+\mathcal C$. The hidden symmetry survives the charged radial metric function; see [Frolov, Krtouš and Kubizňák, Sections 3.2 and 3.4](https://arxiv.org/html/1705.05482v3). The conservative [Milne table, h = 0](https://doi.org/10.1093/mnras/stz123) gives angular intensity $A(\mu)$ and polarized intensity $P(\mu)$, with emission-angle cosine $\mu\in[0,1]$ and normalization $2\int_0^1\mu A(\mu)\,d\mu=1$. Thus $I_\nu(\mu)=A(\mu)B_\nu(T)$ has single-face flux $\pi B_\nu(T)$. Interpolate $A,P$ together, not polarization fraction alone; uniform resampling and f32 storage perturb the exact normalization.

The emitted electric vector is parallel to the surface and perpendicular to the photon in the emitter frame. Endpoint WP contractions recover its detector direction. At $K=0$ the invariant loses screen information: selected equatorial fallbacks exist, but a thin-surface sample may be unresolved and the volume helper may return zero Q/U. This is incomplete handling, not proof of unpolarized emission.

Project camera up into the detector screen and take right as $\boldsymbol\ell\times\boldsymbol u_{\rm screen}$. For electric-vector angle $\chi_s$ from up toward right, $Q=I_{\rm pol}\cos2\chi_s$ and $U=I_{\rm pol}\sin2\chi_s$. Diagnostics use luminance-weighted $\Pi_L=\sqrt{Q_Y^2+U_Y^2}/I_Y$ and $\chi_s=\tfrac12\operatorname{atan2}(U_Y,Q_Y)$ modulo $\pi$, only where defined. Accumulate I/Q/U in this common basis before a linear analyzer:

$$
I_\alpha=\tfrac12[I+Q\cos(2\alpha)+U\sin(2\alpha)].
$$

Diffuse sky and tangled-field jets are unpolarized; $V=0$ for all current sources. The analyzer reuses the accumulated Stokes images.

## Ordered volume transfer

Let $d\ell=-d\lambda>0$ denote backward detector-normalized affine increment. In vacuum the local rest-frame light path is $ds_e=\varepsilon_e\,d\ell$, not generally $d\ell$. With $\nu_e=\nu_o\varepsilon_e$,

$$
\widehat j_\nu=\frac{j_{\nu,e}(\nu_o\varepsilon_e)}{\varepsilon_e^2},\qquad
\widehat\alpha=\varepsilon_e\alpha_{\nu,e}(\nu_o\varepsilon_e),\qquad
\varepsilon_e=-p\cdot u_e>0.
$$

For $j_{\nu,e}=\alpha_{\nu,e}B_{\nu_e}(T)$, $\widehat j_\nu=\widehat\alpha B_{\nu_o}(gT)$. Disk emission adds its Milne closure; jets use their spectral table. This is scalar absorption, not a polarized scattering matrix. See [Younsi, Wu and Fuerst](https://arxiv.org/html/1207.4234v1).

For a homogeneous cell of affine width $\Delta\ell$ and foreground transmission $\mathcal T$,

$$
\tau=\widehat\alpha\Delta\ell,\qquad
I\leftarrow I+\mathcal T\widehat j_\nu\Delta\ell\,\varphi(\tau),\qquad
\mathcal T\leftarrow\mathcal T e^{-\tau},
$$

$$
\varphi(\tau)=\begin{cases}(1-e^{-\tau})/\tau,&\tau>0,\\1,&\tau=0.\end{cases}
$$

Q/U use the same attenuation with signed emissivities. For $\tau\le10^{-3}$, use $\varphi\simeq1-\tau/2+\tau^2/6$. Exact constant-coefficient integration does not make an inhomogeneous sampled cell exact. [ipole](https://arxiv.org/abs/1712.03057) also uses semianalytic cells but includes a more general transport matrix.

In plasma, $ds_e=\varepsilon_e\mathfrak n_e\,d\ell$: extinction gains $\mathfrak n_e$, and the occupation-normalized emissivity uses $1/\mathfrak n_e$. Apply the detector factor $\mathfrak n_o^2$ once after transfer. [Plasma](plasma.md#radiation-preview-and-limits) defines the adopted source normalization.

## Storage and missing light

Radiation validity is separate from geometric destination: a resolved endpoint can still have unsupported spectral transfer. Invalid samples keep zero contribution and missing weight under the [image estimator](../numerics/image-formation.md#sampling-and-missing-weight).

Samples and detector outputs use rgba16float; histories use rgba32float. Before f16 storage, values above magnitude 65504 are peak-scaled. The path pass scales I/Q/U together; celestial composition has a separate I clamp. This lossy safeguard is not physical attenuation and does not report missing trajectory weight. Extreme saturation and Stokes consistency across the second clamp remain unverified.
