---
id: 2
plan_slug: fd-projection-fft
scheme: finite-difference-explicit
strategy: Chorin projection method on a collocated grid — 2nd-order central finite differences for advection and diffusion, explicit RK2 time stepping, and an FFT pressure-Poisson solve that projects the velocity onto the divergence-free space each step.
---

## PDE Reference

**Governing equation** (incompressible Navier-Stokes, 2-D):

    u_t + (u . grad) u + grad p = nu * Delta u,
    div u = 0.

**Domain**: periodic square [0, 2*pi] x [0, 2*pi].
**Parameter**: nu = 0.1.
**Time interval**: t in [0, 0.5]; report fields at t_final = 0.5.

**Initial condition** (Taylor-Green vortex at t = 0):

    u(x, y, 0) =  sin(x) cos(y),
    v(x, y, 0) = -cos(x) sin(y),
    p(x, y, 0) =  0.25 (cos(2x) + cos(2y)).

**Boundary conditions**: periodic in x and y.

**Exact solution** (used by evaluator):

    u(x,y,t) =  sin(x) cos(y) exp(-2 nu t),
    v(x,y,t) = -cos(x) sin(y) exp(-2 nu t),
    p(x,y,t) =  0.25 (cos(2x) + cos(2y)) exp(-4 nu t)   (compared mean-removed).

**Return contract**: `solve_pde(N)` returns
`fields = {"u": (N,N), "v": (N,N), "p": (N,N)}`, `grid = {"x": linspace(0,2pi,N), "y": linspace(0,2pi,N)}`, `t_final = 0.5`.
Harness runs at N = 48 and N = 96. Hard gate: discrete `divergence_norm` must stay near zero.

## Numerical Scheme

- **Spatial discretization**: 2nd-order central finite differences on a uniform collocated grid, periodic wrap via `np.roll`. First derivatives `(f_{i+1}-f_{i-1})/(2 dx)`, Laplacian `(f_{i-1}-2 f_i+f_{i+1})/dx^2` per direction, with `dx = dy = 2*pi/N` (the true periodic spacing). Spatial truncation error is O(dx^2), so the observed error decreases at ~2nd order and the convergence-order gate (p >= 1.5) is met with a clean, measurable rate.
- **Time stepping**: explicit Chorin projection per step:
  1. provisional velocity `u* = u^n + dt*(-(u.grad)u + nu*Delta u)`, advanced with **RK2 (Heun / midpoint)** for O(dt^2) temporal accuracy;
  2. pressure-Poisson `Delta phi = div(u*)/dt` solved in one FFT divide on the periodic box (`phi_hat = -div_hat/|k|^2`, k=0 mode zeroed);
  3. correction `u^{n+1} = u* - dt*grad(phi)`.
  `dt = 0.2 * dx^2 / nu` so the FTCS diffusion limit `rx+ry = 2 nu dt/dx^2 = 0.4 <= 0.5` holds; `Nt = ceil(0.5/dt)`, `dt = 0.5/Nt`. Temporal error O(dt^2) = O(dx^4) stays subdominant to the O(dx^2) spatial error, so the measured order reflects the spatial scheme.
- **Divergence-free projection**: the projection Poisson solve and the gradient/divergence used in it are done with the **same spectral (FFT) operators**, so `div(u^{n+1})` is zero to round-off in that consistent sense — this satisfies the hard structural gate. (Report `p = phi` accumulated / most-recent-step scaled; mean-removed on return.)
- **Stability check**: diffusion CFL `nu dt/dx^2 = 0.2` per direction (sum 0.4 <= 0.5) OK; advective CFL `dt/dx ~ 0.2*dx/nu -> 0` as N grows, so central-difference advection with RK2 is well within its stability region. Stable at every N.
- **Solver library**: numpy + numpy.fft for the pressure solve. scipy.sparse optional (not needed with FFT Poisson).

## Implementation Notes

- **Endpoint duplication**: internally use the periodic grid `np.arange(N)*(2*pi/N)` (spacing `2*pi/N`) for all `np.roll` stencils and the FFT; return `grid = np.linspace(0,2*pi,N)` and enforce periodicity on the returned arrays (`f[-1,:]=f[0,:]`, `f[:,-1]=f[:,0]`). Do NOT use the linspace spacing `2*pi/(N-1)` for the stencils — it breaks periodic consistency and the divergence gate.
- Keep the divergence operator used to build the Poisson RHS identical to the gradient operator used in the correction (both FFT `i*k`), otherwise the corrected field is not discretely divergence-free.
- meshgrid with `indexing="ij"`; axis 0 <-> x, axis 1 <-> y.
- Guard `|k|^2 = 0` (set `phi_hat` there to 0) — pressure is defined up to a constant.
- Pressure for scoring: the projection variable `phi` approximates `p*dt`-scaled increment; recover a physical pressure estimate consistent with `grad p = -(u.grad)u + nu Delta u` at the final step (or use the analytic-structure Poisson `Delta p = -div((u.grad)u)`), then mean-remove. Velocity accuracy is the primary scored quantity.

## Results

Implemented as specified: 2nd-order central-FD (via `np.roll`, periodic wrap) for the
provisional-velocity RHS `-(u.grad)u + nu*Delta u`, advanced with RK2 (Heun), then an
FFT pressure-Poisson projection (`phi_hat = -div(u*)_hat / (dt * K2)`, k=0 mode zeroed)
using spectral `i*k` operators for both the divergence (RHS) and the gradient
(correction step), so `u^{n+1}` is divergence-free in the exact spectral sense. Grid
uses the true periodic convention `x = linspace(0, 2*pi, N, endpoint=False)`
(spacing `dx = 2*pi/N`), which keeps the returned `grid` array and the internally
computed field pointwise consistent (no duplicated-endpoint patching needed). Pressure
is recovered at `t_final` from `Delta p = -div((u.grad)u)` via the same spectral
operators, and is naturally zero-mean since the k=0 mode is zeroed.

`dt = 0.2 * dx^2 / nu` (diffusion FTCS-style bound, `rx+ry = 0.4 <= 0.5`), `Nt = ceil(t_final/dt)`, `dt = t_final/Nt`.

| N | dx | dt | Nt | rel L2 err (u,v) | rel L2 err (p, mean-removed) | max|div u| (FD) |
|---|---|---|---|---|---|---|
| 48 | 0.130900 | 3.333e-02 | 15 | 4.629e-03 | 9.891e-03 | 9.599e-05 |
| 96 | 0.065450 | 8.681e-03 | 58 | 1.184e-03 | 2.537e-03 | 6.152e-06 |

Observed convergence order (primary field `u`, N=48 -> 96): `p = log2(4.629e-3 / 1.184e-3) ≈ 1.97`
(matches the expected 2nd-order spatial scheme; well above the `min_spatial_order=1.5` gate).

At N=48: `max|u| = max|v| = 0.903323`, `L-inf(p - mean(p)) = 4.078e-01`.

Both accuracy (rel L2 < 1% at N=96, well under the 1% threshold) and the divergence-free
structural gate (max FD divergence shrinking with refinement, ~1e-4 to ~1e-6, effectively
round-off relative to field magnitude O(1)) pass cleanly.

## Evaluation

(Filled by evaluator)

<review score=10>

Score: 10/10 — Done

### Numerical Accuracy
- fine_grid_error (l2):  u=1.183640e-03, v=1.183640e-03, p(mean-removed)=2.537382e-03  (PASS ✓)
- tolerance:              0.0100
- observed_order:         1.967  (primary 'u', min 1.5; ok)
- structural_constraints: all satisfied — divergence_norm (spectral, relative to velocity scale) = 2.56e-15 at N=48, 6.30e-15 at N=96 (round-off; gate threshold 1e-3)

### Feedback for solver
None.

Note for comparison: plan 1 (pseudo-spectral) reproduces the Taylor-Green solution to machine precision (rel L2 ~1e-15) since the TG field contains only |k|=1,2 modes — strictly more accurate than this plan's 0.12% at N=96, though both clear the 1% threshold comfortably. This FD-projection-FFT plan is still a valid, independently-correct 2nd-order scheme with a clean measurable convergence order and an exactly divergence-free (to round-off) projection.

</review>
