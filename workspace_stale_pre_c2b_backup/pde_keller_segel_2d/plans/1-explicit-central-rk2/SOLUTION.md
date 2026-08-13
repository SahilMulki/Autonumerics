---
id: 1
plan_slug: explicit-central-rk2
scheme: finite-difference-explicit
strategy: Conservative 2nd-order central-difference finite-volume for both diffusion and the chemotaxis flux, advanced with explicit SSP-RK2; the high-accuracy contrast plan that carries no positivity mechanism.
---

## PDE Reference

Coupled 2-D Keller-Segel chemotaxis system on `[0,1]^2`, `t in [0, 0.5]`, fields `rho` (density) and `c` (chemoattractant):

```
rho_t = D_r * Delta rho - chi * div(rho * grad c) + s_rho
c_t   = D_c * Delta c + alpha * rho - beta * c + s_c
```

**Parameters**: `D_r = D_c = chi = alpha = beta = 1`.

**Domain**: `x in [0,1]`, `y in [0,1]`. **Report at** `t_final = 0.5`.

**Exact (manufactured) fields** — used only to define IC/BC/sources, not inside the update:
```
rho(x,y,t) = 1 + 0.2*sin(pi x)*sin(pi y)*exp(-t)          # range [1, 1.2], strictly positive
c(x,y,t)   = 1 + 0.1*cos(pi x)*cos(pi y)*exp(-2t)         # range [0.9, 1.1], strictly positive
```

**Initial condition** (t=0): `rho = 1 + 0.2*sin(pi x)*sin(pi y)`, `c = 1 + 0.1*cos(pi x)*cos(pi y)`.

**Boundary conditions**: time-dependent Dirichlet from the exact fields. On the boundary `sin(pi x)sin(pi y)=0`, so `rho = 1`; `c = 1 + 0.1*cos(pi X)*cos(pi Y)*exp(-2t)`. Re-impose after **every** stage/step at the correct `t`.

**Manufactured sources** (with `X,Y = np.meshgrid(x, y, indexing="ij")`, scalar `t`):
```
s_rho = -0.2*np.sin(np.pi*X)*np.sin(np.pi*Y)*np.exp(-t)
        + 0.4*np.pi**2*np.sin(np.pi*X)*np.sin(np.pi*Y)*np.exp(-t)
        - 0.04*np.pi**2*np.sin(np.pi*X)*np.cos(np.pi*X)*np.sin(np.pi*Y)*np.cos(np.pi*Y)*np.exp(-3*t)
        - 0.2*np.pi**2*(1 + 0.2*np.sin(np.pi*X)*np.sin(np.pi*Y)*np.exp(-t))*np.cos(np.pi*X)*np.cos(np.pi*Y)*np.exp(-2*t)
s_c   = (0.2*np.pi**2 - 0.1)*np.cos(np.pi*X)*np.cos(np.pi*Y)*np.exp(-2*t)
        - 0.2*np.sin(np.pi*X)*np.sin(np.pi*Y)*np.exp(-t)
```

## Numerical Scheme

- **Spatial discretization**: uniform cell-centered/collocated grid `x = np.linspace(0,1,N)`, `y = np.linspace(0,1,N)`, `dx = 1/(N-1)`, `X,Y = np.meshgrid(x,y,indexing="ij")`. Both Laplacians use the standard 2nd-order 5-point central stencil. The chemotaxis term is written in **conservative flux form** `div(F)`, `F = rho * grad c`:
  - face-centered `grad c` by central differences: `(c[i+1,j]-c[i,j])/dx` at the x-face between `i` and `i+1`;
  - `rho` on that face by the **arithmetic average** `0.5*(rho[i,j]+rho[i+1,j])` (central / non-upwinded);
  - `div F ≈ (Fx_{i+1/2} - Fx_{i-1/2})/dx + (Fy_{j+1/2} - Fy_{j-1/2})/dy`.
  Fully 2nd-order in space. Reaction terms `alpha*rho - beta*c` evaluated pointwise. Sources `s_rho, s_c` added pointwise at the stage time.
- **Time stepping**: explicit **SSP-RK2** (Heun) — 2nd order in time; matches the 2nd-order spatial error. `dt` tied to the explicit-diffusion stability limit so it holds at any `N`:
  `dt = 0.2 * dx**2 / D_r` (limit is `dx^2/(4D)=0.25 dx^2`; 0.2 keeps a safety margin and also satisfies the far looser advection CFL `v*dt/dx<<1` since `|v|=|chi grad c| ~ 0.12`). `Nt = ceil(t_final/dt)`, then `dt = t_final/Nt`. Temporal error `O(dt)=O(dx^2)` stays subdominant/comparable to spatial `O(dx^2)`.
- **Stability check**: FTCS 2-D diffusion needs `D*dt/dx^2 * (2 directions) ≤ 0.5`, i.e. `dt ≤ dx^2/(4D)`. With `dt = 0.2 dx^2/D` we have margin 0.2 < 0.25. RK2 has a slightly larger stability region than forward Euler, so this is safely stable. **Satisfied at all N.**
- **Solver library**: `numpy` only (vectorized slicing; no linear solves needed).

## Implementation Notes

- Store `rho, c` as `(N,N)` arrays indexed `[i,j] ~ (x_i, y_j)`.
- Right-hand side function `rhs(rho, c, t)` returns `(drho, dc)`; call it twice per SSP-RK2 step (predictor at `t`, corrector at `t+dt`), average.
- Re-impose Dirichlet BCs on the boundary rows/cols of both fields after **each** RK stage, using the stage time (`t` for predictor state, `t+dt` for the corrector state), so BCs are time-consistent.
- Chemotaxis flux only needs `div F` at interior nodes; boundary nodes are overwritten by Dirichlet anyway.
- **Positivity**: this plan has NO upwinding/limiter — it relies on the manufactured fields staying well away from zero (`rho>=1`, `c>=0.9`). For this smooth benchmark the central scheme should not undershoot, but it is the structural-gate *contrast* plan; if `min_density` is tripped, Plan 2 (flux-limited upwind) is the positivity-safe alternative.
- Nt is large at N=96 (~11k steps) but each step is a handful of vectorized array ops over 9216 nodes — runs in a few seconds.

## Results

Implemented exactly as specified: conservative central-difference finite-volume for the 5-point Laplacians and the flux-form chemotaxis term `div(rho*grad c)` (face rho by arithmetic averaging, no upwinding), advanced with explicit SSP-RK2 (Heun), Dirichlet BCs re-imposed at each stage's stage-time.

- **Grid / dt**: `N=48` → `dx=1/47≈0.021277`, `dt=0.2*dx²/D_r≈9.053e-5`, `Nt=5524` steps to `t_final=0.5`. `N=96` → `dx=1/95≈0.010526`, `dt≈2.216e-5`, `Nt≈22563` steps. Runtime: ~2.9 s at N=48, ~40 s at N=96 (uv run python).
- **Representative values (N=48, t=0.5)**: `rho ∈ [1.000000, 1.121221]` (exact range `[1, 1+0.2·e^{-0.5}]=[1, 1.121306]`), `c ∈ [0.963212, 1.036788]` (exact range `[1-0.1·e^{-1}, 1+0.1·e^{-1}]=[0.963212, 1.036788]`) — matches the manufactured solution closely.
- **Convergence check** (rel. L2 vs. manufactured exact fields, computed independently of the harness):
  - N=48: rel_l2(rho) = 2.355e-5, rel_l2(c) = 1.770e-6
  - N=96: rel_l2(rho) = 5.818e-6, rel_l2(c) = 4.379e-7
  - Observed order on primary field rho: `log2(2.355e-5/5.818e-6) ≈ 2.02` (spec floor is 1.4) — comfortably 2nd order as designed.
- **Positivity gate**: `min(rho) = 1.000000` at both N=48 and N=96 (attained exactly on the boundary where the manufactured solution equals 1) — no negative density observed for this smooth benchmark, so the central (non-upwinded) chemotaxis discretization does not trip the `min_density` gate here, consistent with the plan's design note.

## Evaluation

(Filled by evaluator)

<review score=10>

Score: 10/10 — Done

### Numerical Accuracy
- fine_grid_error (l2):    {'rho': 5.817745e-06, 'c': 4.379343e-07}  (PASS ✓)
- tolerance:               0.0100
- observed_order:          2.017  (primary 'rho', min 1.4; ok)
- structural_constraints:  all satisfied (min(rho) = 1.00000000 at N=96, well above 0; min_density gate not tripped)

### Feedback for solver
None.

</review>
