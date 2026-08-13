---
id: 3
plan_slug: imex-adi-upwind
scheme: finite-difference-implicit
strategy: IMEX time integration — Crank-Nicolson diffusion solved dimension-by-dimension with ADI (sparse tridiagonal), coupled to an explicit positivity-preserving upwind chemotaxis/reaction step, removing the dx^2 explicit-diffusion time-step restriction.
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

**Boundary conditions**: time-dependent Dirichlet from the exact fields — on the boundary `rho = 1`, `c = 1 + 0.1*cos(pi X)*cos(pi Y)*exp(-2t)`. Re-impose at each sub-step at the correct `t`.

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

- **Spatial discretization**: uniform grid `np.linspace(0,1,N)`, `dx = 1/(N-1)`. Diffusion via the 2nd-order central Laplacian, split by direction (`Delta = D_xx + D_yy`) for ADI. Chemotaxis flux `-chi*div(rho*grad c)` in the **same conservative upwind finite-volume form as Plan 2** (face velocity `chi*grad c`, `rho` upwinded on the velocity sign — first-order upwind is sufficient here; optionally van-Leer-limited), evaluated explicitly. Reaction `alpha*rho - beta*c` and sources `s_rho, s_c` explicit.
- **Time stepping**: **IMEX** with a **Crank-Nicolson / Peaceman-Rachford ADI** treatment of the (stiff) diffusion and an explicit treatment of the non-stiff chemotaxis + reaction + source terms. Per step of size `dt`:
  1. explicit half-update of chemotaxis + reaction + sources (evaluated at `t`, positivity-preserving upwind), giving the explicit RHS `E^n`;
  2. ADI sweep for CN diffusion: implicit sweep in x (solve `N` tridiagonal systems `(I - 0.5*dt*D_r*D_xx)`), then implicit sweep in y (`(I - 0.5*dt*D_r*D_yy)`), each with the CN explicit half on the opposite operator, incorporating `E^n`.
  Diffusion is unconditionally stable, so `dt` is set by the advection CFL only: `dt = 0.4 * dx / a_max`, `a_max = chi*max|grad c| ≈ 0.4` (bounded). CN is 2nd-order in time; combined with 2nd-order space the temporal error `O(dt^2)=O(dx^2)` stays subdominant. `Nt = ceil(t_final/dt)`, `dt = t_final/Nt`. This uses **~100 steps at N=96** versus ~11k for the explicit plans.
- **Stability check**: CN/ADI diffusion is **unconditionally stable** (no `dt ≤ dx^2` restriction). Explicit chemotaxis upwind needs `a_max*dt/dx ≤ 1`; with `dt = 0.4 dx/a_max` this is 0.4 < 1. **Satisfied at all N.** Positivity: the explicit upwind chemotaxis sub-step is positivity-preserving under this CFL; the implicit CN diffusion sweep with Dirichlet data is an M-matrix solve that does not create negative undershoot for these smooth positive fields.
- **Solver library**: `scipy.sparse` + `scipy.sparse.linalg` (tridiagonal `spsolve` / banded solve for the ADI sweeps; factorize once per direction with `scipy.linalg.solve_banded` or a cached `splu`, since the operator is constant in time).

## Implementation Notes

- Build the 1-D tridiagonal CN operators `(I ± 0.5*dt*D*L1)` once (constant coefficients ⇒ same matrix every step); reuse across all rows/columns and both fields (`rho`, `c` share `D=1`). Prefer a banded/`solve_banded` solve applied to all `N` right-hand sides at once for speed.
- ADI is applied independently to `rho` and `c`; the coupling (`alpha*rho`, chemotaxis) lives entirely in the explicit `E^n` term.
- Handle Dirichlet BCs by moving known boundary values to the RHS of each tridiagonal solve (interior unknowns only) and overwriting boundary rows/cols at the new time each step.
- Chemotaxis flux computed conservatively (same face flux in/out of neighboring cells); upwind on the face-velocity sign for positivity — no bare `np.maximum` clip.
- Return `fields={"rho":rho, "c":c}`, `grid={"x":x,"y":y}`, `t_final=0.5`.

## Results

Implemented exactly as designed: Peaceman-Rachford ADI (Crank-Nicolson, banded/tridiagonal `scipy.linalg.solve_banded` sweeps, one shared operator for `rho` and `c` since `D_r=D_c=1`) for diffusion, coupled to an explicit van-Leer-limited conservative upwind chemotaxis flux + reaction + manufactured sources. To secure the observed spatial-convergence order (min 1.4) given `dt` scales linearly with `dx` (not `dx^2`, since diffusion is unconditionally stable), the explicit terms are advanced with a trapezoidal predictor-corrector pass each step (predict with `E^n`, then correct with `0.5*(E^n+E^{n+1,pred})`), giving an overall ~2nd-order-in-time IMEX scheme matching the 2nd-order CN diffusion and 2nd-order (smooth-region) MUSCL chemotaxis reconstruction.

- Grid / dt: `N=48`: `dx=1/47≈0.02128`, `dt≈0.02083`, `Nt=24`. `N=96`: `dx=1/95≈0.01053`, `dt≈0.01042`, `Nt=48`.
- `N=96` representative values at `t_final=0.5`: `min(rho)=1.000000`, `max(rho)=1.121306`, `min(c)=0.963212`, `max(c)=1.036788`.
- Relative L2 errors: `N=24`: `rho=2.749e-04`, `c=8.119e-05`. `N=48`: `rho=7.586e-05`, `c=2.170e-05`. `N=96`: `rho=2.093e-05`, `c=5.636e-06`. Well under the 1% threshold at every grid tested.
- Observed convergence order (48→96): `rho: 1.86`, `c: 1.94` — comfortably clears the `min_spatial_order=1.4` floor.
- `min_density` gate: `min(rho)=1.0` at every N tested (matches the analytic floor exactly, always ≥0) — the van Leer limited upwind chemotaxis flux and the M-matrix CN diffusion solve introduce no negative undershoot.

## Evaluation

(Filled by evaluator)

<review score=10>

Score: 10/10 — Done

### Numerical Accuracy
- fine_grid_error (l2):  rho=2.093107e-05, c=5.636317e-06  (PASS ✓)  [required max = 2.093107e-05]
- tolerance:               0.0100
- observed_order:          1.858  (primary 'rho', min 1.4; ok)
- structural_constraints:  all satisfied (min(rho)=1.000000 >= 0, min(c)=0.963212 >= 0 at N=96 — no negative undershoot)

### Feedback for solver
None.

</review>
