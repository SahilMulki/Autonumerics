# Autonumerics Report — sde_black_scholes

## Problem

**Type:** SDE (Itô)
**Family:** Black-Scholes / geometric Brownian motion — scalar, linear, multiplicative noise, non-stiff.

Governing equation: `dX = r*X dt + sigma*X dW`, with `X_0 = 100.0`, `r = 0.05`, `sigma = 0.20` on `t ∈ [0, 1]`.

Analytic moments at `T = 1`:
- Mean: `X_0 * exp(r*T) ≈ 105.127`
- Variance: `X_0^2 * exp(2*r*T) * (exp(sigma^2*T) - 1) ≈ 451.03`

Evaluation via Monte Carlo comparison of empirical mean/variance against these analytic moments
(thresholds: variance rel. err < 10%, mean rel. err < 5%).

## Plans

| Plan | Scheme | Strong order | Final score | Iters | Mean rel. err | Var rel. err |
|---|---|---|---|---|---|---|
| 1-euler-maruyama | Euler-Maruyama | 0.5 | **10** | 1 | 0.0020 | 0.0187 |
| 2-milstein | Milstein (GBM correction) | 1.0 | **10** | 1 | 0.0020 | 0.0189 |

Both used `dt=0.01`, `num_paths=50000`, `T=1.0`, `seed=42`.

### 1-euler-maruyama
Baseline Euler-Maruyama Monte Carlo. Empirical mean 105.340, variance 459.465.
Both moment errors well within thresholds → score 10 on the first cycle.

### 2-milstein
Milstein scheme adding the GBM correction `0.5*sigma^2*X_n*(dW^2 - dt)` for strong order 1.0.
Empirical mean 105.338, variance 459.532. Both moment errors well within thresholds → score 10 on the first cycle.

## Best Plan

**Recommendation: `2-milstein`.**

Both plans tie on score (10) and iteration count (1), and their weak-moment accuracy is nearly
identical here (the moments being compared are weak quantities, where both schemes have comparable
weak order). Milstein is recommended as the superior scheme because it has strong order 1.0 versus
Euler-Maruyama's 0.5, giving better pathwise accuracy for the same `dt` at negligible extra cost for
this scalar multiplicative-noise problem. If minimizing per-step work is the priority and only weak
moments matter, `1-euler-maruyama` is an equally valid choice.

## Failures

None. Both plans reached score 10 on their first solver↔evaluator cycle with no errors.
