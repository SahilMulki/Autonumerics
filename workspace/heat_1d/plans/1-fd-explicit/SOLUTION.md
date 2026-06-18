---
id: 1
plan_slug: fd-explicit
scheme: finite-difference-explicit
strategy: FTCS finite difference with Nx=100, dt satisfying CFL dt <= dx^2/(2*alpha).
---

## PDE Reference

- **Governing equation**: u_t = alpha * u_xx,  alpha = 0.1
- **Domain**: x in [0, 1],  t in [0, 0.5]
- **Boundary conditions**: u(0, t) = 0,  u(1, t) = 0  (Dirichlet)
- **Initial condition**: u(x, 0) = sin(π x)
- **Analytic solution**: u(x, t) = exp(-alpha * π² * t) * sin(π x)

## Numerical Scheme

- **Spatial discretization**: 2nd-order central difference, Nx=100 interior + boundary points
- **Time stepping**: FTCS (Forward Time Centered Space), explicit
- **dt**: chosen so that r = alpha * dt / dx² = 0.4 (CFL-safe, < 0.5)
- **Stability check**: r = 0.4 ≤ 0.5 — stable
- **Solver library**: numpy only

## Implementation Notes

- Re-impose Dirichlet BCs (u[0]=0, u[-1]=0) after every time step.
- r = 0.4 provides CFL safety margin of 20%.

## Results

- **Grid**: Nx = 100, dx = 1/99 ≈ 0.010101
- **dt used**: 4.0813e-04 (Nt = 1225 steps), r = alpha·dt/dx² ≈ 0.4 (CFL-stable)
- **t_final**: 0.5
- **max |u(T)|**: 0.610386 (numerical), 0.610421 (analytic) — at x = 0.5
- **Self-check relative L2 error vs analytic**: 5.80e-05 (well below the 1% threshold)

## Evaluation

(Filled by evaluator)

<review score=10>

Score: 10/10 — Done

**Score: 10/10**

### Numerical Accuracy
- relative_l2_error:  0.000058  (PASS ✓)
- threshold:          0.0100

### Feedback for solver
- None

</review>
