# Observers and camera rays

An observer supplies a timelike velocity and orthonormal frame; the camera supplies orientation and field of view. Navigation does not create physical aberration or Doppler shift.

Implementation: [observer frames](../../src/physics/observer.ts), [coupled preparation](../../src/scene/preparation.ts), [camera](../../src/scene/camera.ts), and [launch shader](../../src/gpu/wgsl/passes/image.wgsl).

## Regular reference frame

In the [KS chart](../numerics/geodesics.md#regular-coordinates), let $F=(2r-q^2)/\Sigma$ and $\boldsymbol n=(\sin\theta\cos\psi_\epsilon,\sin\theta\sin\psi_\epsilon,\cos\theta)$. A regular tetrad is

$$
\begin{aligned}
 u_0^\mu&=(1+F/2,-\epsilon F\boldsymbol n/2),\\
 e_r^\mu&=(\epsilon F/2,(1-F/2)\boldsymbol n),\\
 e_\theta^\mu&=(0,\cos\theta\cos\psi_\epsilon,\cos\theta\sin\psi_\epsilon,-\sin\theta),\\
 e_\phi^\mu&=(0,-\sin\psi_\epsilon,\cos\psi_\epsilon,0).
\end{aligned}
$$

The frame satisfies $u_0^2=-1$, $e_i\cdot e_j=\delta_{ij}$ and $u_0\cdot e_i=0$. It is generally accelerated. On an axis the angular basis retains an orientation choice without evaluating $1/\sin\theta$.

For measured local velocity $\boldsymbol\beta$, $|\boldsymbol\beta|<1$, set $b^\mu=\beta^i e_i^\mu$ and $\Gamma=(1-|\boldsymbol\beta|^2)^{-1/2}$:

$$
 u^\mu=\Gamma(u_0^\mu+b^\mu),\qquad
 \widetilde e_i^\mu=e_i^\mu+\beta_i
 \left(\Gamma u_0^\mu+\frac{\Gamma^2}{\Gamma+1}b^\mu\right).
$$

$\Gamma^2/(\Gamma+1)=(\Gamma-1)/|\boldsymbol\beta|^2$ for nonzero speed and tends to $1/2$ at rest, avoiding cancellation and division by zero.

## Observer choices

| Choice    | Definition and domain                                                                                                                                             |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Regular   | The accelerated tetrad above at a regular representable event                                                                                                     |
| Static    | $u=\eta\,\partial_t/\sqrt{-g_{tt}}$; $g_{tt}<0$, with $\eta=\pm1$ chosen future-directed                                                                          |
| ZAMO      | Circular zero-angular-momentum observer, $\Omega_Z=-g_{t\phi}/g_{\phi\phi}$ away from the axis; the code uses its axis-regular equivalent and checks timelikeness |
| Custom    | Coordinate velocity $d(X,Y,Z)/dT$ in the chosen KS chart                                                                                                          |
| Free fall | Neutral unit-mass geodesic launched by a local boost of the regular frame, advanced by requested proper time                                                      |

For a custom trial tangent $v^\mu=(1,\boldsymbol v_{\rm coord})$, require $g(v,v)<0$ and $-g(u_0,v)>0$. Then $u=v/\sqrt{-g(v,v)}$. A coordinate speed bound $|\boldsymbol v_{\rm coord}|<1$ is not the curved-coordinate light-cone condition.

Free fall retains block orientation and chart-time offset. The endpoint frame is reconstructed by a local boost, not gyroscope transport. Failed preparation rejects the scene update; camera-only edits reuse the endpoint. Playback advances the source epoch independently of the proper-time input.

## Camera rays

Let $\boldsymbol\ell$ be a unit local vector pointing out of the camera toward an apparent source. The arriving photon is future-directed:

$$
p^\mu=u_o^\mu-\ell^i e_{(i)}^\mu,\qquad
 -p_\mu u_o^\mu=1.
$$

Trace with negative affine/Mino steps; keep the future momentum in frequency contractions. Positive detector energy can coexist with nonpositive Killing energy.

For raster width $W$, height $H$, pixel $(i,j)$, subpixel offset $(\delta_x,\delta_y)$ and vertical field of view $\vartheta$,

$$
x=\frac{i+1/2+\delta_x-W/2}{H},\qquad
 y=\frac{j+1/2+\delta_y-H/2}{H},
$$

$$
\boldsymbol\ell=\operatorname{normalize}\left[
 \boldsymbol f+2\tan(\vartheta/2)(x\boldsymbol r_c-y\boldsymbol u_c)\right].
$$

Camera forward/right/up are expressed in the same prepared tetrad for every sample. The minus sign on $y$ maps downward raster rows to camera up. The [plasma launch](plasma.md#normalized-hamiltonian) reduces the spatial momentum magnitude according to the local cutoff while retaining unit detector energy.
