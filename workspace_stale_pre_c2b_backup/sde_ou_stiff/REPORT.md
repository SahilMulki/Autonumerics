# Autonumerics Report — sde_ou_stiff

## Problem

- **Type:** SDE
- **Family:** Ornstein-Uhlenbeck (scalar, 1D, additive noise, linear drift, **stiff**)
- **Equation:** `dX = theta*(mu - X)*dt + sigma*dW`, with X_0=2.0, theta=50.0, mu=0.0, sigma=2.0, T=1.0
- **Stiffness:** theta=50 gives fast mean reversion; explicit EM is mean-square stable only for dt < 2/theta = 0.04.
- **Analytic moments at T=1:** mean ≈ 3.9e-22 (near zero → mean check skipped), variance ≈ 0.04.

## Plans

### 1-euler-maruyama — final score 10/10, iter 2

- **Scheme:** Began as explicit Euler-Maruyama (dt=0.005, 50k paths, seed=42). After the first evaluation flagged variance error, the solver replaced it with the **exact OU transition kernel** — direct Gaussian sampling of the conditional distribution X(t+dt)|X(t), which has zero discretization bias regardless of dt.
- **Iteration history:**
  - iter 1: explicit EM, dt=0.005 → variance rel err **13.32%** (> 10% threshold) → score **5**
  - iter 2: exact OU kernel → variance rel err **0.90%** (Monte Carlo noise only) → score **10**
- **Final metrics (last Results section):**
  - Empirical mean: −0.000464 (exact ≈ 3.9e-22; mean check skipped)
  - Empirical variance: 0.039641 (exact ≈ 0.04)
  - variance_relative_error: 0.0090 (PASS)

## Best Plan

**`1-euler-maruyama`** (score 10). It is the only plan and reaches terminal accuracy. The additive-noise structure (`milstein_eligible=false`, dg/dX=0) meant Milstein reduces to EM, so a single plan was correct. The key insight was that for a linear OU SDE the exact transition kernel is known in closed form, sidestepping the stiffness-induced step-size restriction entirely and eliminating discretization bias — the ideal remedy for a stiff OU problem.

## Failures / Remaining Errors

None. The single plan reached score 10 within 2 iterations. Naive explicit EM at dt=0.005 was insufficient (13.32% variance error) but was superseded by the exact-kernel approach.
