---
id: 1
plan_slug: pseudo-spectral
scheme: spectral
strategy: Fourier pseudo-spectral method with exact (integrating-factor) viscous term, RK4 for the nonlinear advection, and a Fourier-space Leray projection that makes the velocity divergence-free to machine precision.
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

- **Spatial discretization**: Fourier pseudo-spectral. Integer wavenumbers `kx = ky = fftfreq(N, d=1.0/N)` (period 2*pi). All linear operators (grad, Laplacian, divergence) are exact diagonal multiplications in Fourier space. The nonlinear term `(u.grad)u` is evaluated pseudo-spectrally: derivatives taken in Fourier space, products formed in physical space, with the **2/3 dealiasing rule** applied to the nonlinear product. Spatial accuracy is spectral (exponential); for the Taylor-Green field only wavenumbers |k|=1 (velocity) and |k|=2 (pressure) are present, so both N=48 and N=96 resolve the field to machine precision.
- **Time stepping**: integrating-factor method — the viscous term `exp(-nu |k|^2 t)` is integrated **exactly**, and the projected nonlinear term is advanced with classical **RK4**. `dt` tied to the advective CFL: `dt = 0.25 * dx` with `dx = 2*pi/N` (U_max ~ 1), so `Nt = ceil(0.5/dt)`, `dt = 0.5/Nt`. Temporal error is O(dt^4) and strongly subdominant.
- **Divergence-free projection**: after forming the nonlinear RHS in Fourier space, apply the Leray projector `RHS_hat <- RHS_hat - k (k . RHS_hat)/|k|^2` (k=0 mode left untouched). The velocity stays divergence-free to round-off by construction — this satisfies the hard structural gate.
- **Pressure recovery**: pressure is obtained from the pressure-Poisson relation `p_hat = -(k . N_hat)/|k|^2` where `N_hat` is the (dealiased) nonlinear term in Fourier space, then mean-removed (k=0 set to 0). Returned p matches the analytic pressure up to the additive constant the evaluator removes.
- **Stability check**: viscous term is exact (unconditionally stable, no dt <= dx^2/(4nu) restriction). RK4 on the advection is stable for `dt/dx` well below 1; `dt = 0.25 dx` gives CFL ~ 0.25. Stable at any N.
- **Solver library**: numpy (numpy.fft). scipy not required.

## Implementation Notes

- **Endpoint duplication**: `np.linspace(0, 2*pi, N)` includes both 0 and 2*pi (same periodic point) with spacing `2*pi/(N-1)`, which is NOT the FFT sampling grid. Work internally on the true periodic grid `xf = np.arange(N) * (2*pi/N)` (period 2*pi, no duplicate). Because the solution is a smooth trig polynomial, evaluate the returned fields at the linspace points either (a) by trig interpolation of the spectral coefficients, or (b) simplest and exact-enough: compute on `xf`, and since fields are periodic set the returned array's last row/column equal to the first (`f[-1,:]=f[0,:]`, `f[:,-1]=f[:,0]`) while filling interior from the periodic solution resampled onto the linspace nodes. Keep the returned `grid` exactly `np.linspace(0,2*pi,N)`.
- Use `indexing="ij"` meshgrid so array axis 0 <-> x, axis 1 <-> y, consistent with the `(N,N)` field convention.
- Guard the `|k|^2 = 0` mode (set projector/pressure contribution to 0 there) to avoid division by zero.
- 2/3 dealiasing: zero Fourier modes with `|kx| >= N/3` or `|ky| >= N/3` in the nonlinear product.
- **Order-gate caveat**: because the Taylor-Green modes are fully resolved, the spatial error is ~machine precision at both N — the observed spatial-convergence order is not meaningful (both errors are far below the 1% tolerance). This plan relies on the harness's tolerance-relative super-convergence waiver (error below tol at both grids passes the order check). It is the most accurate and most robustly divergence-free plan; plans 2 and 3 provide clean measurable 2nd-order convergence as a hedge.

## Results

Solver implemented exactly as specified: Fourier pseudo-spectral spatial discretization on the periodic grid `xf = arange(N)*(2*pi/N)`, integrating-factor RK4 in time (viscous term `exp(-nu*|k|^2*dt)` exact, advection+pressure-projection RK4), 2/3-rule dealiasing, Leray projection applied to the RHS every stage, pressure recovered from `Delta p = -(u_x^2 + 2 u_y v_x + v_y^2)` (verified algebraically to reproduce the exact TG pressure), fields resampled onto the returned `linspace(0,2*pi,N)` grid via exact trigonometric interpolation of the retained Fourier coefficients (this also naturally reconciles the duplicated periodic endpoint with the FFT grid).

Ran `uv run python solver.py` at N=48 and N=96:

| N | dt used | Nt steps | max\|u\| | max\|v\| | max\|p\| | divergence_norm | rel_L2(u) | rel_L2(v) |
|---|---|---|---|---|---|---|---|---|
| 48 | 3.125e-02 | 16 | 0.904332 | 0.904332 | 0.409365 | 2.21e-16 | 7.88e-16 | 8.08e-16 |
| 96 | 1.613e-02 | 31 | 0.904714 | 0.904714 | 0.409365 | 2.20e-16 | 5.57e-16 | 5.59e-16 |

Both grids match the analytic Taylor-Green solution (`u,v` at t=0.5) to machine precision (rel L2 ~1e-15, far below the 1% tolerance), and the discrete divergence stays at round-off (~2e-16), satisfying the hard `divergence_norm` structural gate. Max|p| = 0.409365 matches the analytic `0.5*exp(-0.2) = 0.40937` exactly. As anticipated in the plan, the observed spatial order is not meaningful here (error saturates at machine precision at both N) — this relies on the harness's tolerance-relative super-convergence waiver.

## Evaluation

(Filled by evaluator)

<review score=10>

Score: 10/10 — Done

### Numerical Accuracy
- fine_grid_error (l2), per required field: u = 5.566e-16, v = 5.594e-16  (PASS ✓)
- gauge field p (mean-removed): 8.993e-16
- tolerance:               0.0100
- observed_order:          inf  (primary 'u', min 1.5; ok — both errors are already below 1e-9 at N=48 and N=96, so the harness's tolerance-relative super-convergence waiver applies, exactly as anticipated in the plan's "Order-gate caveat" note)
- structural_constraints:  all satisfied — divergence_norm = 2.207e-16 (N=48), 2.204e-16 (N=96), both at round-off, well under the 1e-6 gate threshold

### Feedback for solver
None. The Fourier pseudo-spectral discretization exactly resolves the Taylor-Green vortex (only |k|=1 velocity and |k|=2 pressure modes are present), the integrating-factor RK4 keeps temporal error negligible, and the per-stage Leray projection enforces div u = 0 to machine precision by construction — satisfying the hard structural gate. Results match the solver's own reported table exactly. For reference, plan 2 (fd-projection-fft) also scores 10/10 with a more standard finite-difference + FFT-Poisson projection; this spectral plan is the more accurate of the two (error ~1e-15 vs. finite-precision-limited FD order), though both clear the bar comfortably.

</review>
