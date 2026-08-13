# Report — sde_oscillator_long_horizon

## Problem

- **Type:** SDE (stochastic differential equation), Monte Carlo moment estimation.
- **Family:** `stochastic_oscillator` — a 2D linear harmonic oscillator with **additive** noise entering only the Y component.
- **System (Itô):**
  ```
  dX = Y dt,                 X(0) = 1.0
  dY = -X dt + sigma dW(t),  Y(0) = 0.0
  ```
- **Parameters:** `X_0 = 1.0`, `Y_0 = 0.0`, `sigma = 0.3`, `T = 10*pi ≈ 31.4159` (five full periods).
- **Analytic moments at T:** `mean_X = 1.0`, `mean_Y = 0.0`, `variance_X ≈ 1.4137`, `variance_Y ≈ 1.4137`.

## Plans

### 1-euler-maruyama — **score 10/10**, 1 iteration

- **Scheme:** Explicit Euler-Maruyama on the coupled 2D system, `dt = 0.001` (rescaled so terminal time lands exactly on `10*pi`), `num_paths = 50000`, seed 42. Simultaneous update from old X, Y. Milstein not applicable (additive/constant diffusion → zero correction; multi-D Milstein undefined without Lévy areas), so EM is the sole appropriate scheme.
- **Key metrics (final run):**
  - `empirical_mean = [1.0138, -0.0016]` (exact `[1.0, 0.0]`)
  - `empirical_variance = [1.4336, 1.4404]` (exact `≈ [1.4137, 1.4137]`)
  - Component X: mean rel err **0.0138** (PASS < 0.05), variance rel err **0.0141** (PASS < 0.10)
  - Component Y: mean check skipped (near-zero exact mean), variance rel err **0.0189** (PASS < 0.10)

## Best Plan

**1-euler-maruyama** — the only plan and a clean pass. It reached score 10 on the first solver↔evaluator cycle: both component means and variances land within thresholds. The small step size (`dt = 0.001`) successfully controls the amplitude drift inherent to explicit EM on a harmonic oscillator (deterministic step matrix spectral radius `sqrt(1 + dt^2) > 1`) over the long five-period horizon.

## Failures

None. The single plan reached the terminal condition (score 10) with no remaining errors.
