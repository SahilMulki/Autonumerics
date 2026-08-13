# Autonumerics Report — sde_cir

## Problem

- **Type:** SDE
- **Family:** Cox-Ingersoll-Ross (CIR) square-root process
- **Equation:** `dX(t) = kappa*(theta - X)*dt + sigma*sqrt(X)*dW(t)`
- **Parameters:** `X_0 = 0.5, kappa = 2.0, theta = 1.0, sigma = 0.5, T = 1.0`
- **Character:** scalar (1D), multiplicative noise, linear drift, non-stiff. Feller condition `2*kappa*theta = 4.0 >= sigma^2/2 = 0.125` holds, so the process stays strictly positive.
- **Analytic moments at T=1:** mean `= theta + (X_0 - theta)*exp(-kappa) ≈ 0.9323`; variance from the closed-form CIR expression.

## Plans

Both schemes used dt=0.01, num_paths=50000, seed=42, with the mandatory positivity guard `np.maximum(X, 0.0)` before the sqrt diffusion term.

| Plan | Scheme | Final score | Iters | Mean rel err | Variance rel err |
|---|---|---|---|---|---|
| 1-euler-maruyama | Euler-Maruyama (strong order 0.5) | 10/10 | 1 | 0.0040 | 0.0276 |
| 2-milstein | Milstein (strong order 1.0), correction `(sigma^2/4)*(dW^2 - dt)` | 10/10 | 1 | 0.0040 | 0.0279 |

Both PASS on both thresholds (mean rel err < 0.05, variance rel err < 0.10) on the first iteration.

## Best Plan

**Recommendation: `1-euler-maruyama`.**

Both plans reached the terminal score of 10 in a single iteration with essentially identical accuracy. For this CIR process the Milstein correction is state-independent — `g*g' = sigma^2/2`, so the correction reduces to `(sigma^2/4)*(dW^2 - dt)` — and provides no measurable moment-accuracy advantage at dt=0.01. Euler-Maruyama is therefore the preferred choice: same accuracy, lower complexity and cost. Milstein remains the better option if tighter pathwise (strong) convergence is needed at coarser step sizes.

## Failures

None. Both plans passed on the first solver↔evaluator cycle; no errors remained.
