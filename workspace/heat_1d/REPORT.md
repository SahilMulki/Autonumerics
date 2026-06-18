# Autonumerics Report — heat_1d

## Problem

- **Type**: PDE — 1D heat (diffusion) equation
- **Governing equation**: u_t = α·u_xx, α = 0.1
- **Domain**: x ∈ [0, 1], t ∈ [0, 0.5]
- **Boundary conditions**: Dirichlet, u(0,t) = u(1,t) = 0
- **Initial condition**: u(x,0) = sin(πx)
- **Analytic solution**: u(x,t) = exp(-α·π²·t)·sin(πx)
- **Pass threshold**: relative L2 error < 1%

## Plans

### 1-fd-explicit — FTCS explicit finite difference
- **Scheme**: Forward Time, Centered Space (explicit); 2nd-order central difference in space, forward Euler in time.
- **Discretization**: Nx = 100, dx ≈ 0.010101, dt ≈ 4.0813e-04 (Nt = 1225), r = α·dt/dx² = 0.4 (CFL-safe, < 0.5).
- **Final score**: 10/10
- **Iterations**: 1
- **Key metrics**: relative L2 error = 5.80e-05; max |u(T)| = 0.610386 (analytic 0.610421) at x = 0.5.

### 2-crank-nicolson — Crank-Nicolson implicit
- **Scheme**: Crank-Nicolson (implicit, 2nd order in time and space); tridiagonal solve via scipy.sparse.linalg.spsolve.
- **Discretization**: Nx = 100, dx ≈ 0.010101, dt = 0.01 (Nt = 50), r ≈ 9.80 (unconditionally stable, no CFL limit).
- **Final score**: 10/10
- **Iterations**: 1
- **Key metrics**: relative L2 error = 3.7e-05; max pointwise error = 2.28e-05; max |u(T)| = 0.610444 at x = 0.5.

## Recommendation

**Best plan: 2-crank-nicolson.**

Both plans reached the terminal score of 10 on their first cycle and are well within the 1% accuracy threshold. Crank-Nicolson is recommended because:

- It is marginally more accurate (rel L2 = 3.7e-5 vs 5.8e-5).
- It reaches that accuracy in **50 time steps** versus FTCS's **1225 steps** — a ~24× reduction in time-stepping cost — because it is unconditionally stable and not bound by the CFL constraint r ≤ 0.5.

FTCS (1-fd-explicit) remains an excellent, numpy-only alternative when simplicity is preferred and the CFL-limited step count is acceptable.

## Failures / Outstanding Errors

None. Both plans passed cleanly on the first iteration with no remaining errors.
