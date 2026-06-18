---
id: 2
plan_slug: crank-nicolson
scheme: finite-difference-implicit
strategy: Crank-Nicolson implicit scheme, Nx=100, dt=0.01, unconditionally stable.
---

## PDE Reference

- **Governing equation**: u_t = alpha * u_xx,  alpha = 0.1
- **Domain**: x in [0, 1],  t in [0, 0.5]
- **Boundary conditions**: u(0, t) = 0,  u(1, t) = 0  (Dirichlet)
- **Initial condition**: u(x, 0) = sin(π x)
- **Analytic solution**: u(x, t) = exp(-alpha * π² * t) * sin(π x)

## Numerical Scheme

- **Spatial discretization**: 2nd-order central difference, Nx=100
- **Time stepping**: Crank-Nicolson (implicit), dt=0.01, Nt=50
- **Stability check**: Crank-Nicolson is unconditionally stable — no CFL constraint
- **Solver library**: scipy.sparse + scipy.sparse.linalg.spsolve

## Implementation Notes

- Assemble tridiagonal system with lower/main/upper diagonals: -r/2, 1+r, -r/2 on LHS; r/2, 1-r, r/2 on RHS.
- Dirichlet BCs encoded by zeroing the first and last rows and setting diagonal to 1, RHS to 0.
- r = alpha * dt / dx² (appears in both LHS and RHS matrices).

## Results

- **Grid**: Nx = 100, dx = 1/99 ≈ 0.010101
- **Time stepping**: dt = 0.01 (Nt = 50), r = alpha*dt/dx² ≈ 9.80 (well inside the unconditional-stability regime of Crank-Nicolson)
- **Solution at t_final = 0.5**:
  - max |u(T)| = 0.610444 (at x = 0.5)
  - u(0.5, 0.5) = 0.610444
- **Accuracy vs analytic** u(x,t) = exp(-alpha·π²·t)·sin(πx):
  - relative L2 error = 3.7e-5 (PASS, threshold 1%)
  - max pointwise error = 2.28e-5
- Clean run, no exceptions. Boundary points held at 0 (interior-only solve with implicit Dirichlet enforcement).

## Evaluation

(Filled by evaluator)

<review score=10>

Score: 10/10 — Done

**Score: 10/10**

### Numerical Accuracy
- relative_l2_error:  0.000037  (PASS ✓)
- threshold:          0.0100
- max pointwise error: 2.28e-05

### Comparison
- This Crank-Nicolson plan (rel_l2 = 3.7e-5) is the most accurate plan, slightly better than 1-fd-explicit (rel_l2 = 5.80e-5).

### Feedback for solver
- None

</review>
