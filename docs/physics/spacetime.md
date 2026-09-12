# Spacetime and conventions

Nullray uses a fixed Kerr–Newman metric with prescribed test sources. It does not evolve Einstein–Maxwell fields or source backreaction. [Coverage](../validation/coverage.md) states the implemented continuation domain.

Implementation: [metric and Jacobians](../../src/physics/geometry.ts), [constants](../../src/physics/motion.ts), [block transitions](../../src/physics/atlas.ts), and their [WGSL counterpart](../../src/gpu/wgsl/geodesics/geometry.wgsl).

## Units and notation

Use metric signature $(-,+,+,+)$ and $G=c=M=1$. Here $M$ is the geometrized mass scale. For physical mass $M_{\rm phys}$, angular momentum $J_{\rm phys}$ and SI electric charge $Q_{\rm SI}$,

$$
r_g=\frac{GM_{\rm phys}}{c^2},\qquad
 t_g=\frac{GM_{\rm phys}}{c^3},\qquad
 a=\frac{J_{\rm phys}c}{GM_{\rm phys}^2},\qquad
 q=\frac{Q_{\rm SI}}{\sqrt{4\pi\epsilon_0G}\,M_{\rm phys}}.
$$

Multiply distances and times by $r_g$ and $t_g$; the Schwarzschild radius is $2r_g$. Neutral optics depend on $q^2$. Photons and neutral emitters have no Lorentz-force term. The jet mass control changes emission normalization, not dimensionless geometry.

| Symbol                     | Contract                                                                 |
| -------------------------- | ------------------------------------------------------------------------ |
| $(t,r,\theta,\phi)$        | Boyer–Lindquist (BL) coordinates; $r$ may be signed                      |
| $(T,X,Y,Z)$                | Cartesian Kerr–Schild (KS) coordinates in the selected chart             |
| $w_\epsilon,\psi_\epsilon$ | Null-chart time and rotating longitude; distinct from $T,t,\phi$         |
| $\epsilon=+1,-1$           | Ingoing/outgoing chart sign; not a block's time orientation              |
| $p^\mu,p_\mu$              | Future-directed tangent and metric-lowered covector                      |
| $E=-p_t$, $L=p_\phi$       | Killing constants; `energy`, `angularMomentum` in code                   |
| $\mathcal C$               | Carter constant; `carter` in code; not charge and not always nonnegative |
| $K=(L-aE)^2+\mathcal C$    | Alternate separation constant                                            |
| $m^2$                      | 0 for vacuum light, 1 for unit timelike four-velocity; `massSquared`     |
| $\lambda,\gamma$           | Affine and unscaled Mino parameters, $d\lambda=\Sigma\,d\gamma$          |
| $g$                        | Observed/emitted frequency ratio; the metric always carries indices      |

Physical angles are radians. [Observers](observers.md) defines backward tracing; [plasma](plasma.md) uses separately normalized momentum.

## Metric

Set

$$
\Sigma=r^2+a^2\cos^2\theta,\qquad
\Delta=r^2-2r+a^2+q^2,\qquad
\mathcal A=(r^2+a^2)^2-a^2\Delta\sin^2\theta.
$$

The BL line element is

$$
ds^2=-\frac{\Delta}{\Sigma}(dt-a\sin^2\theta\,d\phi)^2
 +\frac{\sin^2\theta}{\Sigma}[(r^2+a^2)d\phi-a\,dt]^2
 +\frac{\Sigma}{\Delta}dr^2+\Sigma\,d\theta^2.
$$

Useful components and the determinant are

$$
g_{tt}=-1+\frac{2r-q^2}{\Sigma},\qquad
 g_{t\phi}=-\frac{a(2r-q^2)\sin^2\theta}{\Sigma},\qquad
 g_{\phi\phi}=\frac{\mathcal A\sin^2\theta}{\Sigma},\qquad
 \det g=-\Sigma^2\sin^2\theta.
$$

The cross term is $2g_{t\phi}\,dt\,d\phi$. See [Adamo and Newman, Section 3](https://arxiv.org/html/1410.6626v2), translating their opposite metric signature.

## Horizons, stationary limits and singularities

For $d=1-a^2-q^2\ge0$, roots of $\Delta$ are

$$
r_\pm=1\pm\sqrt d.
$$

For $d>0$ the roots are distinct. Schwarzschild is exceptional: $r_-=0$ is singular, leaving only the regular $r_+=2$ horizon. At $d=0$ the horizon is degenerate; at $d<0$ there is no horizon. Production evaluates $r_-=(a^2+q^2)/r_+$ to reduce cancellation.

Stationary limits solve $g_{tt}=0$:

$$
r_{\mathrm e\pm}(\theta)=1\pm\sqrt{1-q^2-a^2\cos^2\theta}
$$

These surfaces exist where the radicand is nonnegative and the metric is regular. The ergoregion has $g_{tt}>0$. At $a=0$, stationary limits coincide with horizon roots; there is no distinct exterior ergoregion.

Curvature is singular at $\Sigma=0$. For $a\ne0$ this is the ring $r=0,\theta=\pi/2$, with KS cylindrical radius $|a|$. Other points on $r=0$ are regular and can connect signed-radius sheets. For $a=0$ the whole zero-radius set is singular: directly placing an observer at negative $r$ selects a disconnected component.

The shadow follows ray destinations, source boundaries and foreground emission, rather than the coordinate horizon shape. A bright arc alone does not identify a resolved photon subring.

## Separated neutral motion

With $P=E(r^2+a^2)-aL$, define

$$
\mathcal C=p_\theta^2+\cos^2\theta
 \left[a^2(m^2-E^2)+\frac{L^2}{\sin^2\theta}\right],
$$

$$
R=P^2-\Delta(K+m^2r^2),\qquad
\Theta=\mathcal C+a^2(E^2-m^2)\cos^2\theta-L^2\cot^2\theta.
$$

Away from BL coordinate poles,

$$
\begin{aligned}
 r'&=s_r\sqrt R,&\theta'&=s_\theta\sqrt\Theta,\\
 \phi'&=\frac{aP}{\Delta}+\frac{L}{\sin^2\theta}-aE,&
 t'&=\frac{(r^2+a^2)P}{\Delta}+aL-a^2E\sin^2\theta.
\end{aligned}
$$

Primes denote future Mino derivatives; signs change at ordinary turns. Timelike motion has $d\tau/d\gamma=\Sigma$. See [Wang, Lee and Lin, Section II](https://arxiv.org/html/2208.11906v1#S2). Production uses [regular vector flow](../numerics/geodesics.md) at horizons and axes.

Retain $E=0$ and $E<0$. Literature ratios $L/E$ and $\mathcal C/E^2$ do not cover every detector launch. Rescaling momentum also rescales the affine/Mino parameter.

## Analytic extension versus implemented atlas

The atlas stores exterior, black-hole, interior and white-hole blocks, exterior-copy labels and stationary-side orientation. Naked and disconnected components have separate identities. Chart switches preserve an event; horizon crossings update block history. Reciprocal-radius zero ends at infinity.

“Antiverse” is used informally for a negative-radius end here. It is not the same destination as another positive-radius exterior and does not specify antimatter sources. Source illumination is independent [boundary data](emission.md#source-boundaries).

Ordinary continuation and selected exact bifurcation families are implemented. General horizon-critical motion, ambiguous simultaneous events and direct bifurcation-sphere placement remain open; no chronology classifier is exposed. Mathematical continuation does not establish stable astrophysical traversal. [Dafermos and Luk](https://arxiv.org/abs/1710.01722) address perturbed vacuum Kerr, not this prescribed charged flow.
