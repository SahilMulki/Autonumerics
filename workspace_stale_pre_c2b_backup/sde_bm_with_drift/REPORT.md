# Autonumerics Report — sde_bm_with_drift

## Problem

- **Type:** SDE (Itô)
- **Family:** Brownian motion with constant drift (additive noise)
- **Equation:** `dX(t) = mu·dt + sigma·dW(t)`, X_0 = 1.0, mu = 0.5, sigma = 0.3, on [0, 1]
- **Character:** scalar (1D), additive noise (constant diffusion), linear, non-stiff
- **Analytic moments at T = 1:** mean = X_0 + mu·T = 1.5, variance = sigma²·T = 0.09

## Plans

### 1-euler-maruyama — **score 10/10**, iter 1

- **Scheme:** Monte Carlo Euler-Maruyama, dt = 0.01, num_paths = 50000, seed = 42, T = 1.0
- **Rationale:** Additive noise means dg/dX = 0, so Milstein reduces exactly to EM — only one scheme is warranted. X(T) is exactly Gaussian, so EM introduces no scheme bias; residual error is pure Monte Carlo sampling noise.
- **Final metrics (last Results):**
  - Empirical mean: 1.502579 (exact 1.5) → mean relative error **0.0017** (threshold 0.05, PASS)
  - Empirical variance: 0.091344 (exact 0.09) → variance relative error **0.0149** (threshold 0.10, PASS)
  - Overall: PASS ✓

## Recommendation

**Best plan: `1-euler-maruyama`** — the sole plan, reaching terminal score 10/10 on its first cycle. Because the process is Gaussian with constant coefficients, Euler-Maruyama is distributionally exact and no higher-order or alternative scheme can improve on it; the only knobs are path count and dt, both already comfortably within thresholds. No refinement needed.

## Failures

None. All plans reached score 10.
