# Autonumerics Report — sde_exponential_ou

## Problem

- **Type**: SDE
- **Family**: Exponential Ornstein-Uhlenbeck (scalar, state_dimension=1, multiplicative noise)
- **Equation**: `dX = X(-theta·log(X) + sigma²/2) dt + sigma·X dW`
- **Parameters**: X_0 = 1.0, theta = 1.0, sigma = 0.4, T = 1.0
- **Analytic moments** (X_t lognormal; log X is an OU process): with
  `v_t = sigma²/(2·theta)·(1 - e^(-2·theta·t))`, `mean = e^(v_t/2)`,
  `variance = e^(v_t)·(e^(v_t) - 1)`.
- **Thresholds**: variance_rel_err < 0.10, mean_rel_err < 0.05.

## Plans

### 1-euler-maruyama — Euler-Maruyama (strong order 0.5)
- dt=0.01, num_paths=50000, seed=42, positivity guard `np.maximum(X, 1e-12)` on the log-drift.
- **Final score: 10** | **iter: 1**
- Empirical mean 1.038344, empirical variance 0.078993.
- mean_rel_err = 0.0030 (PASS), variance_rel_err = 0.0292 (PASS).

### 2-milstein — Milstein (strong order 1.0)
- dt=0.01, num_paths=50000, seed=42, same positivity guard; Milstein correction
  `0.5·sigma²·X_n·(dW² − dt)` reusing the same dW draw.
- **Final score: 10** | **iter: 1**
- Empirical mean 1.038326, empirical variance 0.078941.
- mean_rel_err = 0.0030 (PASS), variance_rel_err = 0.0285 (PASS).

## Recommendation

**Best plan: 2-milstein.** Both plans reached score 10 on the first cycle, so
the choice comes down to a tiebreak: Milstein achieved a slightly lower variance
relative error (0.0285 vs 0.0292) and is the theoretically superior scheme
(strong order 1.0 vs 0.5), exploiting the multiplicative diffusion g(X)=sigma·X.
Euler-Maruyama is an equally acceptable, simpler alternative for this problem
since accuracy at dt=0.01 was already comfortable.

## Plans that failed score 10

None — both plans passed all thresholds on the first solver↔evaluator cycle.
No remaining errors.
