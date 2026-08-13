# Report: sde_ginzburg_landau_s4

## Problem

**Type**: SDE (Itô), family: stochastic Ginzburg-Landau, `sigma = 4`.

```
dX(t) = ((sigma^2/2) * X(t) - X(t)^3) dt + sigma * X(t) dW(t),   X(0) = 1.0,   T = 1.0
```

Scalar state, multiplicative noise (`g(X) = sigma*X`), nonlinear cubic drift. No elementary closed-form moment formula; reference moments were obtained by discretization-free Monte Carlo of the known exact pathwise solution:

- `E[X(T)]  ≈ 0.659`
- `Var[X(T)] ≈ 1.117`

The dominant numerical challenge: at `sigma=4` the cubic `-X^3` drift is superlinear, so **plain Euler-Maruyama blows up** (Inf/NaN, or systematically biased moments). Both plans therefore used a **tamed drift**: `f_tamed = f / (1 + dt*|f|)`, with the diffusion left untamed.

## Plans

| Plan | Scheme | Final dt | Score | Iter |
|---|---|---|---|---|
| `1-euler-maruyama` | Tamed Euler-Maruyama | 0.0002 (refined from 0.005) | **10/10** | 1 |
| `2-milstein` | Tamed-drift Milstein | 0.0005 (refined from 0.005) | **10/10** | 1 |

Both plans converged in a single solver↔evaluator cycle — the plan-creator's initial `dt=0.005` was too coarse for either scheme (variance rel. errors of 234%–345%), but each solver ran its own convergence sweep and selected a smaller `dt` (still within the verifier's `dt <= 0.04` ceiling) that cleared both accuracy thresholds.

### 1-euler-maruyama

- 50,000 paths, `dt=0.0002` (5000 steps), seed 42.
- `empirical_mean = 0.652371` (rel err 1.01%, threshold 5%)
- `empirical_variance = 1.111680` (rel err 0.48%, threshold 10%)
- All terminal values finite; robustness checked across seeds 1, 7, 123 (worst-case variance rel err 3.3%).

### 2-milstein

- 50,000 paths, `dt=0.0005` (2000 steps), seed 42.
- Uses the multiplicative-noise Milstein correction `0.5*sigma^2*X*(dW^2 - dt)` on top of the same tamed drift.
- `empirical_mean = 0.661698` (rel err 0.41%, threshold 5%)
- `empirical_variance = 1.117239` (rel err 0.02%, threshold 10%)
- All terminal values finite.

## Best Plan: `2-milstein`

Both plans reach the terminal score of 10/10 in a single iteration, so the tie-break is efficiency: Milstein achieves **tighter accuracy** (mean rel err 0.41% vs 1.01%; variance rel err 0.02% vs 0.48%) using **2.5x fewer time steps** (2000 vs 5000), consistent with its higher strong convergence order (1.0) on this multiplicative-noise SDE. The Euler-Maruyama plan is an independently valid, clean pass, but Milstein is the more compute-efficient scheme for this problem.

## Outstanding Issues

None. Both plans reached score 10 within the max_iter budget; no unresolved errors.
