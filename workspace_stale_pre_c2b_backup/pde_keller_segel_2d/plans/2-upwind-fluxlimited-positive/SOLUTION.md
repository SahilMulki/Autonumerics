---
id: 2
plan_slug: upwind-fluxlimited-positive
scheme: finite-volume-upwind
strategy: Conservative finite-volume chemotaxis flux upwinded on the chemotactic velocity with a van-Leer flux limiter (2nd order, positivity-preserving) plus central diffusion, advanced with explicit SSP-RK2; the structure-preserving plan built to pass the min_density gate.
---

## PDE Reference

Coupled 2-D Keller-Segel chemotaxis system on `[0,1]^2`, `t in [0, 0.5]`, fields `rho` (density) and `c` (chemoattractant):

```
rho_t = D_r * Delta rho - chi * div(rho * grad c) + s_rho
c_t   = D_c * Delta c + alpha * rho - beta * c + s_c
```

**Parameters**: `D_r = D_c = chi = alpha = beta = 1`. **Domain**: `[0,1]^2`. **Report at** `t_final = 0.5`.

**Exact (manufactured) fields** (define IC/BC/sources only):
```
rho(x,y,t) = 1 + 0.2*sin(pi x)*sin(pi y)*exp(-t)          # range [1, 1.2]
c(x,y,t)   = 1 + 0.1*cos(pi x)*cos(pi y)*exp(-2t)         # range [0.9, 1.1]
```

**Initial condition** (t=0): `rho = 1 + 0.2*sin(pi x)*sin(pi y)`, `c = 1 + 0.1*cos(pi x)*cos(pi y)`.

**Boundary conditions**: time-dependent Dirichlet from the exact fields — on the boundary `rho = 1`, `c = 1 + 0.1*cos(pi X)*cos(pi Y)*exp(-2t)`. Re-impose after every stage at the correct `t`.

**Manufactured sources** (`X,Y = np.meshgrid(x, y, indexing="ij")`, scalar `t`):
```
s_rho = -0.2*np.sin(np.pi*X)*np.sin(np.pi*Y)*np.exp(-t)
        + 0.4*np.pi**2*np.sin(np.pi*X)*np.sin(np.pi*Y)*np.exp(-t)
        - 0.04*np.pi**2*np.sin(np.pi*X)*np.cos(np.pi*X)*np.sin(np.pi*Y)*np.cos(np.pi*Y)*np.exp(-3*t)
        - 0.2*np.pi**2*(1 + 0.2*np.sin(np.pi*X)*np.sin(np.pi*Y)*np.exp(-t))*np.cos(np.pi*X)*np.cos(np.pi*Y)*np.exp(-2*t)
s_c   = (0.2*np.pi**2 - 0.1)*np.cos(np.pi*X)*np.cos(np.pi*Y)*np.exp(-2*t)
        - 0.2*np.sin(np.pi*X)*np.sin(np.pi*Y)*np.exp(-t)
```

## Numerical Scheme

- **Spatial discretization**: uniform grid `np.linspace(0,1,N)`, `dx = 1/(N-1)`. Diffusion: standard 2nd-order 5-point central Laplacian for both fields. Chemotaxis term in **conservative finite-volume flux form** `-chi*div(rho*grad c) = -div(rho * v_chem)` with chemotactic velocity `v_chem = chi*grad c`:
  - x-face velocity `a_{i+1/2,j} = chi*(c[i+1,j]-c[i,j])/dx` (likewise y-faces);
  - the density carried through each face is chosen by **upwinding on the sign of the face velocity**, reconstructed to 2nd order with a **van Leer (MC) flux limiter** on the `rho` slopes: `rho_face = rho_upwind + 0.5*phi(r)*(slope)`, where `phi` is the limiter and `r` the consecutive-gradient ratio. In smooth regions `phi→1` recovering 2nd-order central; near steep gradients `phi→0` falling back to positivity-preserving 1st-order upwind;
  - face fluxes `F = a_face * rho_face`; `-div F` by the conservative difference.
  This flux form guarantees the discrete density update is a convex combination that cannot create new negative extrema under the CFL below — **positivity by construction**, satisfying the `min_density` gate, while staying 2nd-order for the order≥1.4 check on smooth data.
- **Time stepping**: explicit **SSP-RK2** (Heun) — SSP guarantees each stage is a convex-combination step, preserving the positivity property of the spatial operator. `dt` from the tighter of diffusion and advection limits, tied to `dx`:
  `dt = min(0.2*dx**2/D_r, 0.4*dx/a_max)` with `a_max = chi*max|grad c| ≈ 0.4` (bounded above; using a fixed 0.4 is safe since `|grad c| ≤ 0.1*pi < 0.32`). Diffusion `0.2 dx^2` dominates at these N, so effectively `dt ≈ 0.2 dx^2/D`. `Nt = ceil(t_final/dt)`, `dt = t_final/Nt`.
- **Stability check**: diffusion FTCS `dt ≤ dx^2/(4D)` (using 0.2<0.25, margin ok); advection upwind CFL `a_max*dt/dx ≤ 1` (using 0.4, margin ok). SSP-RK2 preserves both. **Satisfied at all N.**
- **Solver library**: `numpy` (vectorized; limiter via shifted-array slopes). No linear solves.

## Implementation Notes

- Compute face velocities once per stage from the current `c`. Build `rho_face` per face by selecting the upwind cell (`np.where(a_face>0, ...)`) then adding the limited slope correction.
- Van Leer limiter: `phi(r) = (r + |r|)/(1 + |r|)` with `r` = ratio of upwind-side to downwind-side `rho` differences; guard divide-by-zero (`r=0` where denominator ~0). MC limiter is an acceptable alternative.
- Keep the scheme **conservative**: the same face flux is subtracted from cell `i` and added to cell `i+1` so no spurious mass source appears.
- Re-impose Dirichlet BCs on both fields after each RK stage at the stage time.
- Verify `min(rho) > 0` and `min(c) > 0` on the returned arrays before returning (should hold by construction; do NOT use a bare `np.maximum` clip — the limiter is the positivity mechanism).
- Return `fields={"rho":rho, "c":c}`, `grid={"x":x,"y":y}`, `t_final=0.5`.

## Results

Implemented as specified: conservative finite-volume chemotaxis flux (upwind on face velocity `a_face = chi*dc/dx`, van Leer MUSCL reconstruction of the `rho` face value, edge-padded ghost cells at the physical boundary), 5-point central Laplacian for diffusion, explicit SSP-RK2 (Heun) time stepping with `dt = min(0.2*dx^2/D_r, 0.4*dx/a_max)` (diffusion term dominates: `dt ≈ 0.2*dx^2`). Dirichlet BCs re-imposed on both fields after every RK stage at the stage time.

Grid `N=48` (`dx = 1/47`): `Nt = 5522` steps, `dt = 9.053e-05`.
- `rho`: min=1.000000, max=1.121221 — rel_L2 error = 2.343e-05
- `c`:   min=0.963212, max=1.036788 — rel_L2 error = 1.769e-06

Grid `2N=96` (`dx = 1/95`): `Nt = 22563` steps, `dt = 2.216e-05`.
- `rho`: min=1.000000 — rel_L2 error = 5.803e-06
- `c`:   min=0.963212 — rel_L2 error = 4.381e-07

Observed convergence order (primary field `rho`): `log2(2.343e-05 / 5.803e-06) ≈ 2.01` — well above the `min_spatial_order = 1.4` gate.

**Positivity gate**: `min(rho) = 1.0 > 0` and `min(c) = 0.963 > 0` at both resolutions — the van Leer-limited upwind flux enforces positivity by construction (no clipping used); `min_density` gate satisfied.

Both fields comfortably clear the `rel_l2_err_max = 0.01` threshold (errors are ~4 orders of magnitude below 1%).

## Evaluation

(Filled by evaluator)

<review score=10>

Score: 10/10 — Done

### Numerical Accuracy
- fine_grid_error (l2):    5.803452e-06  (PASS ✓)  [per-field: {'rho': 5.803452e-06, 'c': 4.380891e-07}]
- tolerance:               0.0100
- observed_order:          2.014  (primary 'rho', min 1.4; ok)
- structural_constraints:  all satisfied (min_density gate: min(rho)=1.000000 >= 0, min(c)=0.963212 >= 0 at N=96)

### Feedback for solver
None

### Notes
- Coarse grid (N=48): rho rel_l2 = 2.343e-05, c rel_l2 = 1.769e-06.
- Fine grid (N=96): rho rel_l2 = 5.803e-06, c rel_l2 = 4.381e-07.
- Observed spatial order on rho ≈ 2.01, consistent with the 2nd-order van Leer MUSCL reconstruction (limiter is essentially inactive on this smooth manufactured solution, so it behaves like the central scheme in accuracy while remaining positivity-preserving by construction — no clipping used).
- Comparable to plan 1 (explicit-central-rk2, also scored 10), which achieves similar accuracy without upwinding; on this smooth manufactured problem the central scheme never trips the min_density gate either, but plan 2's flux-limited upwind construction guarantees positivity structurally (robust for less benign initial data / aggregation regimes), which is the intended discriminator per the plan's design rationale.

</review>
