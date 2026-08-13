# Report — Ornstein-Uhlenbeck SDE

## Problem

- **Type**: SDE (Ornstein-Uhlenbeck, mean-reverting, additive noise)
- **Equation**: `dX(t) = theta*(mu - X(t))*dt + sigma*dW(t)`, `X(0) = 2.0`
- **Parameters**: theta=1.5, mu=0.0, sigma=0.5, over T ∈ [0, 1.0]
- **Analytic moments at T=1**: exact_mean ≈ 0.446260, exact_variance ≈ 0.079184

## Plans

### 1-euler-maruyama — **score 10/10**, iter 1

- **Scheme**: Euler-Maruyama Monte Carlo (strong order 0.5, weak order 1.0)
- **Discretization**: dt=0.01 (100 steps), num_paths=50000, seed=42, vectorized over paths
- **Milstein note**: not eligible — additive noise gives dg/dX = 0, so Milstein reduces to Euler-Maruyama
- **Results**:
  - Empirical mean: 0.443810 vs exact 0.446260 → **mean_relative_error = 0.0055** (PASS, threshold 0.05)
  - Empirical variance: 0.081047 vs exact 0.079184 → **variance_relative_error = 0.0235** (PASS, threshold 0.10)

## Recommendation

**Best plan: `1-euler-maruyama`** (score 10/10 in a single iteration).

Euler-Maruyama reproduces both the mean and variance of the OU process well within thresholds. Because the noise is additive (constant diffusion), higher-order schemes such as Milstein offer no accuracy benefit here, so the simple EM scheme is the correct and most efficient choice. No plans failed.
