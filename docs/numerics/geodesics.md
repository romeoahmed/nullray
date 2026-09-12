# Geodesic integration and events

These algorithms implement the [metric](../physics/spacetime.md), [observer convention](../physics/observers.md) and [plasma Hamiltonian](../physics/plasma.md). Work budgets limit computation, not the spacetime.

Implementation: [host flow](../../src/physics/geodesic.ts), [GPU flow](../../src/gpu/wgsl/geodesics/orbit.wgsl), [event tracing](../../src/gpu/wgsl/geodesics/trace.wgsl), [source exclusion](../../src/gpu/wgsl/geodesics/barriers.wgsl), and [canonical medium](../../src/gpu/wgsl/geodesics/medium.wgsl).

## Regular coordinates

With chart sign $\epsilon=\pm1$, define BL-to-null primitives

$$
I'(r)=\frac1\Delta,\qquad r_*'(r)=\frac{r^2+a^2}{\Delta},\qquad
w_\epsilon=t+\epsilon r_*,\quad \psi_\epsilon=\phi+\epsilon aI,\quad
T=w_\epsilon-\epsilon r.
$$

The Cartesian chart is

$$
\boldsymbol X=r\boldsymbol n+\epsilon a\boldsymbol e_z\times\boldsymbol n,\qquad
 k_\mu=(1,\epsilon\boldsymbol n),\qquad
 g_{\mu\nu}=\eta_{\mu\nu}+F k_\mu k_\nu,\quad F=\frac{2r-q^2}{\Sigma}.
$$

With $k^\mu=\eta^{\mu\nu}k_\nu$ and $k^2=0$, $g^{\mu\nu}=\eta^{\mu\nu}-Fk^\mu k^\nu$. This rank-one inverse applies at $\Sigma>0$; raising and lowering are distinct operations.

For $d=1-a^2-q^2$ and $z=r-1$, the implemented primitive convention is

$$
I(r)=\begin{cases}
 \dfrac{\log|z-\sqrt d|-\log|z+\sqrt d|}{2\sqrt d},&d>0,\\
 -1/z,&d=0,\\
 \dfrac{\operatorname{atan2}(z,\sqrt{-d})-\pi/2}{\sqrt{-d}},&d<0,
\end{cases}
$$

$$
r_*=r+\log|\Delta|+(2-q^2)I.
$$

Logarithms use dimensionless lengths. Additive constants fix the source-pattern convention. In regular chart overlap,

$$
w_{-\epsilon}=w_\epsilon-2\epsilon r_*,\qquad
\psi_{-\epsilon}=\psi_\epsilon-2\epsilon aI.
$$

A chart change rotates both angular vectors at the same event; a horizon crossing separately changes the block. Do not evaluate primitives at horizon poles.

Direct radius resolves regular $r=0$ crossings; $x=1/r$ resolves signed infinity, with $x^\prime=-r^\prime/r^2$. Ordinary flow switches when the active coordinate exceeds unit magnitude. Reciprocal zero is infinity, not a sheet crossing. Bifurcation patches retain direct radius.

## Pole-free angular motion

Store the angular direction $\boldsymbol n$ and canonical vector $\boldsymbol J$ rather than dividing by $\sin\theta$. For mass squared $m^2\in\{0,1\}$,

$$
\begin{aligned}
\boldsymbol n'&=\boldsymbol J\times\boldsymbol n+\varpi\boldsymbol e_z\times\boldsymbol n,\\
\boldsymbol J'&=a^2(E^2-m^2)n_z(\boldsymbol n\times\boldsymbol e_z)
 +\varpi\boldsymbol e_z\times\boldsymbol J.
\end{aligned}
$$

Primes denote future Mino derivatives. Invariants are $|\boldsymbol n|=1$, $\boldsymbol J\cdot\boldsymbol n=0$, $J_z=L$ and $J^2-a^2(E^2-m^2)n_z^2=\mathcal C+L^2$. Away from axes, $\boldsymbol J=p_\theta\boldsymbol e_\phi-(L/\sin\theta)\boldsymbol e_\theta$; its vector representation supplies the regular axis limit. It is not conserved Euclidean angular momentum.

Let $K=(L-aE)^2+\mathcal C$, $P=E(r^2+a^2)-aL$ and $v=r'$. The horizon-regular transport is

$$
D=\frac{P+\epsilon v}{\Delta}
 =\frac{K+m^2r^2}{P-\epsilon v},\qquad
\varpi=a(D-E),
$$

$$
w_\epsilon'=a\bigl[L-aE(1-n_z^2)\bigr]+(r^2+a^2)D.
$$

Chart selection keeps the rationalized denominator well conditioned in ordinary flow. The unreduced form supplies a removable principal-ray limit when regular; neither form may divide by zero. $\varpi$ is a Mino rate, not $d\phi/dt$.

## Radial flow and integration

For $b=a(aE-L)$, write the radial polynomial as

$$
R(r)=(E^2-m^2)r^4+2m^2r^3+A r^2+2Kr+B,
$$

where $A=2Eb-K-m^2(a^2+q^2)$ and $B=b^2-(a^2+q^2)K$. Evolve $r''=\tfrac12\,dR/dr$, where primes on the path mean Mino derivatives. The reciprocal first integral is

$$
(x')^2=E^2-m^2+2m^2x+Ax^2+2Kx^3+Bx^4,
$$

so the corresponding acceleration is

$$
x''=m^2+A x+3Kx^2+2Bx^3.
$$

Acceleration form passes through ordinary radial turns without manually reversing a square root. First-integral preservation is exact-arithmetic behavior; numerical drift remains.

For [separable plasma](../physics/plasma.md#separable-background), subtract $\tfrac12\partial_r(\Delta f_r)$ from direct acceleration, or half the $x$ derivative of $x^2(1-2x+(a^2+q^2)x^2)A_p/(1+r_0^2x^2)$ from reciprocal acceleration. Replace $K$ by $K+f_r$ in regular null transport. Timelike observers retain vacuum mass terms. The independent Hamiltonian reference differentiates the density and inverse metric directly.

## Discrete integration

| Solver | Scheme and budget                                                                          | Error contract                                                               |
| ------ | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| Host   | Dormand–Prince 5(4), default tolerance $10^{-10}$, 10,000 attempts                         | Finite-interval motion; proper-time evolution divides Mino rates by $\Sigma$ |
| GPU    | Bogacki–Shampine 3(2), local tolerance $4\times10^{-5}$, 2048 attempts shared with heating | Embedded estimates; scaled component errors, not a global image bound        |

Ordinary GPU endpoints are projected onto $v^2=R(r)$ or $v^2=x^4R(1/x)$ using scaled Newton corrections normal to the constraint. Corrections adjust velocity on radial legs and coordinate at simple turns; their size participates in acceptance. A degenerate uncorrectable gradient rejects the step. Factored $\Delta$ uses the same representable horizons as atlas transitions. Bifurcation patches use separate regular equations.

## Bifurcation continuation

A critical horizon has $P(r_h)=0$. A simple radial turn there can cross a bifurcation sphere rather than an ordinary stationary exterior. The implemented patch covers exactly $E=0$, $aL=0$ at nondegenerate horizons; near-zero values are not snapped to this family.

For the selected horizon $r_h$ and other radial root $r_o$, write

$$
s_h=r_h^2+a^2,\qquad \kappa_h=\frac{r_h-1}{s_h},\qquad
\Omega_h=\frac a{s_h},\qquad \mathcal B=K+m^2r^2+f_r(r).
$$

Here $\mathcal B>0$, $R=-\Delta\mathcal B$, $v=dr/d\gamma$, and $f_r=0$ in vacuum. For null-chart sign $\sigma$ and $w=w_\sigma$, replace divergent $w_\sigma$ and longitude with

$$
\beta_h=\sigma\kappa_h w-\log|v|,\qquad \widetilde\psi=\psi_\sigma-\Omega_h w.
$$

For this exact family, $P=aL=0$, $w_\sigma^\prime=\sigma(r^2+a^2)v/\Delta$ and $v^\prime=-\partial_r(\Delta\mathcal B)/2$. Differentiation cancels the selected horizon factor:

$$
\frac{d\beta_h}{d\gamma}=-v\left[\frac{1-\kappa_h(r+r_h)}{r-r_o}+\frac{d\mathcal B/dr}{2\mathcal B}\right].
$$

The angular-vector system adds axial rate $-\sigma\Omega_h(r+r_h)v/(r-r_o)$. Both rates are finite at $r_h$. The nearest root selects the patch; its overlap midpoint is a coordinate choice.

At a radial turn on that bifurcation surface, switch the null-chart sign using finite expressions:

$$
\begin{aligned}
\beta_{h,\rm new}&=-\beta_h+2\kappa_h r-\left(1+\frac{r_o^2+a^2}{s_h}\right)\log|r-r_o|-\log\mathcal B,\\
\widetilde\psi_{\rm new}&=\widetilde\psi+2\sigma\Omega_h\left(r+2\log|r-r_o|\right).
\end{aligned}
$$

Outer bifurcation joins black/white-hole blocks of the same universe; inner bifurcation joins black-hole $n$ to white-hole $n+1$. The inverse reverses the increment. CPU proper-time and GPU null flow use the same cancellation. Ordinary null coordinates can be recovered away from the sphere; direct placement on it remains unsupported.

## Ordered events

The GPU brackets equatorial crossings, coplanar annulus entry and radial endpoints using accepted states and cubic dense output. It selects the earliest event along the backward path. Horizon departure is compared with the current block so step boundaries neither erase nor duplicate crossings. Disk hits require valid emitters and positive local frequency.

Source exclusion requires a strictly negative radial-potential enclosure between the observer and the innermost emitter, plus a lower barrier at zero for rotating geometry or the zero-radius singularity for zero spin. Guarded f32 interval operations include outward rounding and subnormal-flush coverage. Positive reciprocals stay within [WGSL divisor bounds](https://www.w3.org/TR/WGSL/#floating-point-accuracy) and widen by four neighbors. A vacuum-quartic stationary point supplies only a candidate; its sign must be independently enclosed. Uncertain signs or overflow leave ordinary tracing active.

Exactly representable Schwarzschild/extremal horizon generators with zero reduced vacuum data may also resolve source-free when all emitters are outside. Inspection records their fixed radius/latitude once because reduced constants lose the affine scale. Nearby rays are not snapped to that family.

In Schwarzschild, $R=E^2r^4+Kr(2-r)>0$ for $0<r<2$, $K\ge0$ and $(E,K)\ne(0,0)$. An inward backward ray cannot turn. Below all sources, this permits analytic singularity termination; inspection keeps its actual endpoint. The shortcut does not apply to charged or rotating interiors.

At $a\ne0$, a ray can cross $r=0$ away from the ring; at $a=0$, the entire zero-radius set is singular. Integer metadata preserves source-domain identity independently of light.

Opposite endpoint radial-velocity signs trigger dense-output bisection to the far side of a turn, separating legs for event and material searches. Multiple turns with equal endpoint signs, simultaneous events and general critical families remain open. The CPU instead reintegrates partial steps to refine turns and finite endpoints.

## Result semantics

| GPU ray kind                 | Meaning                                                                     | Radiation consequence                                                            |
| ---------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| 0: unresolved                | Budget exhausted, arithmetic/transport failure or unsupported evolution     | Sample retains zero radiance and missing weight                                  |
| 1: asymptotic source         | A signed infinity endpoint in a recorded source domain                      | Requires positive source frequency and supported spectrum                        |
| 2: disk/material termination | Opaque surface hit, or volume transmission below the cutoff                 | May contain integrated disk and jet emission; not always a geometric surface hit |
| 3: singularity               | Singular endpoint or applicable analytic Schwarzschild interior termination | Foreground emission may remain visible                                           |
| 4: source-free               | Proven source exclusion with the declared homogeneous boundary solution     | Resolved darkness; distinct from singularity and unfinished work                 |

`RayPath.kind` calls kind 2 `disk` even for volume-opacity termination. Destination diagnostics can show geometry despite failed spectral evaluation; ordinary image coverage uses final radiation validity.

Inspection stores accepted states rather than exact horizon intersections. A source-free path may show only a finite prefix. The recording cap does not change the tracing budget.

## Arithmetic rules

Quantize shared inputs before CPU/GPU comparisons. WGSL optics use f32; host/reference calculations use binary64. Factoring, rearrangement and coordinate changes require quantity-specific verification.

Guard arithmetic before evaluation; `select` is not lazy. Do not convert NaNs, broad clamps or minimum steps into success. Account for [WGSL floating-point accuracy](https://www.w3.org/TR/WGSL/#floating-point-accuracy), including permitted subnormal behavior.
