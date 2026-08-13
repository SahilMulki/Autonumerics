---
id: 2
plan_slug: milstein
scheme: milstein
strategy: Tamed-drift Milstein (adds the multiplicative-noise correction on top of a tamed drift) for strong order 1.0 on the diffusion, estimating E[X(T)] and Var[X(T)] by Monte Carlo.
---

## SDE Reference

Stochastic Ginzburg-Landau (Ito), scalar, multiplicative noise:

```
dX(t) = ((sigma**2/2) * X(t) - X(t)**3) * dt + sigma * X(t) * dW(t),   X(0) = 1.0
```

Parameters:
- `X_0  = 1.0`
- `sigma = 4.0`
- Time interval: `T0 = 0.0`, `T = 1.0`

Drift  `f(X) = (sigma**2/2) * X - X**3`
Diffusion `g(X) = sigma * X`,  so `dg/dX = sigma`

Reference moments (discretization-free Monte-Carlo of the exact pathwise solution
`X_t = X_0*exp(sigma*W_t) / sqrt(1 + 2*X_0**2 * integral_0^t exp(2*sigma*W_u) du)`):
- `E[X(T)]  ~= 0.659`
- `Var[X(T)] ~= 1.117`

The exact solution stays positive for `X_0 > 0`. Mean is not near zero, so the mean check applies.

## Numerical Scheme

- **Method**: Milstein with **tamed drift** (taming still required at sigma=4)
- **dt**: 0.0005 (refined down from initial 0.005; at dt=0.005 variance error was ~345%,
  convergence testing at dt in {0.005, 0.002, 0.001, 0.0005, 0.0002} showed dt=0.0005 is
  needed to bring variance_rel_err comfortably under 10%; still well within the dt <= 0.04
  verifier ceiling)
- **num_paths**: 50000
- **T**: 1.0
- **seed**: 42
- **Milstein correction**: `0.5 * sigma**2 * X_n * (dW**2 - dt)`
  (from `g = sigma*X`, `dg/dX = sigma`: `0.5 * g * g' * (dW**2 - dt) = 0.5 * sigma**2 * X_n * (dW**2 - dt)`)

Tamed-drift Milstein update:

```
f       = (sigma**2/2) * X - X**3
f_tamed = f / (1 + dt * abs(f))          # tame the drift; do not tame diffusion
dW      = sqrt(dt) * Z,   Z ~ N(0,1)
X       = X + f_tamed * dt + sigma * X * dW + 0.5 * sigma**2 * X * (dW**2 - dt)
```

## Implementation Notes

- The Milstein correction improves the **diffusion** accuracy (strong order 1.0) but does NOT
  by itself fix the cubic-drift explosion. The drift must still be tamed:
  `f_tamed = f / (1 + dt*|f|)`. Do NOT tame the diffusion or the correction term.
- Plain (untamed) integration at sigma=4 produces Inf/NaN; taming is mandatory.
- Keep `dt <= 0.04` (the verifier re-runs at `dt <= 0.04/T = 0.04`). Starting at `dt = 0.005`;
  the solver may refine down if variance accuracy is poor.
- The exact solution is strictly positive; watch for non-finite outputs as a sign of instability.
- Evaluation targets: `variance_rel_err < 10%` AND `mean_rel_err < 5%`.

## Results

Solver run (`solve_sde(num_paths=50000, dt=0.0005, T=1.0, seed=42)`):

- `empirical_mean     = 0.661698`  (reference `0.659`, rel err ≈ 0.41%)
- `empirical_variance = 1.117239`  (reference `1.117`, rel err ≈ 0.02%)
- `dt used = 0.0005`, `num_paths = 50000`, all terminal values finite.

Convergence check (tamed-drift Milstein, same seed/paths, varying dt):

| dt      | mean   | variance |
|---------|--------|----------|
| 0.005   | 0.7819 | 4.9671   |
| 0.002   | 0.7002 | 1.4175   |
| 0.001   | 0.6854 | 1.2267   |
| 0.0005  | 0.6617 | 1.1172   |
| 0.0002  | 0.6581 | 1.1166   |

At the plan's original dt=0.005 the variance is badly biased high (drift-taming alone is not
enough to control the multiplicative-noise variance at this sigma with a coarse grid). dt=0.0005
gives both mean and variance well inside the 5%/10% thresholds while remaining fast (~1.7s for
50000 paths, 2000 steps).

## Evaluation

(Filled by evaluator)

<review score=10>

Score: 10/10 — Done

**Score: 10/10**

### Numerical Accuracy
- mean_relative_error:     0.0041  (PASS)
- variance_relative_error: 0.0002  (PASS)
- Overall: PASS ✓

### Feedback for solver
None.

Note for reference: `evaluate.py` reran `solve_sde(num_paths=50000, dt=0.0005, T=1.0, seed=42)`
directly (matching the SOLUTION.md "Results" section) and reproduced empirical_mean = 0.661698,
empirical_variance = 1.117239, all terminal values finite. Reference moments used:
exact_mean = 0.659, exact_var = 1.117 (per problem_spec.json analytic_moments). Tamed-drift
Milstein at dt=0.0005 is well within the verifier's dt <= 0.04 ceiling and comfortably clears
both thresholds. For comparison, plan 1 (Euler-Maruyama, dt=0.0002) also passes (mean rel err
~1.0%, variance rel err ~0.5%) but needs a smaller dt (2.5x more steps) to reach comparable
accuracy — consistent with Milstein's higher strong order on this multiplicative-noise problem.

</review>
