---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Tamed Euler-Maruyama to stabilize the superlinear cubic drift and estimate E[X(T)] and Var[X(T)] by Monte Carlo.
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
Diffusion `g(X) = sigma * X`

Reference moments (discretization-free Monte-Carlo of the exact pathwise solution
`X_t = X_0*exp(sigma*W_t) / sqrt(1 + 2*X_0**2 * integral_0^t exp(2*sigma*W_u) du)`):
- `E[X(T)]  ~= 0.659`
- `Var[X(T)] ~= 1.117`

The exact solution stays positive for `X_0 > 0`. Mean is not near zero, so the mean check applies.

## Numerical Scheme

- **Method**: Euler-Maruyama with **tamed drift** (required — plain EM blows up at sigma=4)
- **dt**: 0.0002 (refined down from the initial 0.005; see Results for rationale)
- **num_paths**: 50000
- **T**: 1.0
- **seed**: 42

Tamed drift update (diffusion term left unmodified):

```
f      = (sigma**2/2) * X - X**3
f_tamed = f / (1 + dt * abs(f))          # taming keeps the drift increment bounded
dW      = sqrt(dt) * Z,   Z ~ N(0,1)
X       = X + f_tamed * dt + sigma * X * dW
```

## Implementation Notes

- **Plain (untamed) EM is NOT usable at sigma=4.** The cubic `-X**3` drift causes explosive
  intermediate values that systematically bias `E[X(T)^2]` downward and produce Inf/NaN.
  Use the tamed drift `f_tamed = f / (1 + dt*|f|)`; do NOT tame the diffusion term.
- Alternative if taming is insufficient: truncate/clip X per step. Prefer taming first.
- A small time step is required for stability; the verifier re-runs at `dt <= 0.04/T = 0.04`,
  so keep `dt <= 0.04`. Starting at `dt = 0.005`; the solver may refine down if variance
  accuracy is poor.
- The exact solution is strictly positive; large downward-biased or negative empirical means
  signal drift instability — check for finite outputs.
- Evaluation targets: `variance_rel_err < 10%` AND `mean_rel_err < 5%`.

## Results

Ran tamed Euler-Maruyama (num_paths=50000, seed=42, T=1.0).

At the plan's originally specified `dt=0.005`, the scheme was far too coarse:
`empirical_mean = 0.5742` (rel err 12.9% vs 0.659), `empirical_variance = 3.7338`
(rel err 234% vs 1.117) — the tamed drift alone was insufficient at this step size
because the multiplicative diffusion `sigma*X = 4X` can still produce large excursions
within a single dt=0.005 step before the cubic drift pulls X back down.

Refined `dt` downward (all still <= 0.04 cap) and observed clean convergence:

| dt | empirical_mean | mean rel err | empirical_variance | var rel err |
|---|---|---|---|---|
| 0.005  | 0.5742 | 12.9%  | 3.7338 | 234.4% |
| 0.002  | 0.6379 | 3.2%   | 1.2958 | 16.0%  |
| 0.001  | 0.6567 | 0.35%  | 1.1942 | 6.9%   |
| 0.0005 | 0.6476 | 1.7%   | 1.1045 | 1.1%   |
| 0.0002 | 0.6524 | 1.0%   | 1.1117 | 0.5%   |
| 0.0001 | 0.6561 | 0.44%  | 1.1220 | 0.45%  |

Selected **dt = 0.0002** as the final configuration (runtime ~3.7s for 50000 paths,
5000 steps): `empirical_mean = 0.652371`, `empirical_variance = 1.111680`.
Both are comfortably within the evaluation thresholds (mean rel err < 5%, variance
rel err < 10%). Checked stability across additional seeds (1, 7, 123) at dt=0.0002 —
all passed both thresholds (worst case: seed 1, variance rel err 3.3%). All terminal
values are finite (no Inf/NaN).

## Evaluation

(Filled by evaluator)

<review score=10>

Score: 10/10 — Done

**Score: 10/10**

### Numerical Accuracy
- mean_relative_error:     0.0101  (PASS)
- variance_relative_error: 0.0048  (PASS)
- Overall: PASS ✓

### Feedback for solver
None.

Note for reference: `evaluate.py` reran `solve_sde(num_paths=50000, dt=0.0002, T=1.0, seed=42)`
directly (matching the SOLUTION.md "Results" section) and reproduced empirical_mean = 0.652371,
empirical_variance = 1.111680, all terminal values finite. Reference moments used:
exact_mean = 0.659, exact_var = 1.117 (per problem_spec.json analytic_moments). Tamed-drift
Euler-Maruyama at dt=0.0002 clears both thresholds comfortably and stays within the verifier's
dt <= 0.04 ceiling. For comparison, plan 2 (Milstein, dt=0.0005) also passes with slightly
tighter errors (mean rel err ~0.41%, variance rel err ~0.02%) while using 2.5x fewer steps —
consistent with Milstein's higher strong order on this multiplicative-noise problem. Both plans
are valid; Milstein is the more efficient choice if step count matters, but this EM plan is
independently a clean pass.

</review>
