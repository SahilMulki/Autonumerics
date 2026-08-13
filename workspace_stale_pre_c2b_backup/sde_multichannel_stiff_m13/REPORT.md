# REPORT: sde_multichannel_stiff_m13

## Problem

**Type**: SDE — family `multichannel_linear_multiplicative`.

A 2D linear Itô system driven by m = 13 independent Brownian motions with non-commuting, multiplicative (state-dependent) noise:

    dX = F @ X dt + sum_{r=1}^{13} G_r @ X dW_r,   X(0) = (1, 1)^T

- F = [[-2, 3], [-3, -2]] (eigenvalues -2 ± 3i — spiral decay, flagged stiff)
- G_r = 0.6·M1 for 7 odd channels, G_r = 0.6·M2 for 6 even channels, with [M1, M2] ≠ 0 (non-commuting)
- T = 1.0

An exact analytic solution was derived and verified: closed-form mean via `expm(F·T)·X0`, and closed-form variance via a reduced 3×3 second-moment ODE (`expm(A·T)·(1,1,1)` on the vectorized symmetric second-moment matrix), cross-checked against direct fine-step integration of the full Lyapunov ODE.

Reference (exact) terminal moments at T=1:
- mean_X = -0.1148824, mean_Y = -0.1530794
- variance_X = 0.1715989, variance_Y = 0.1627163

Because the diffusion generators M1 and M2 do not commute, multi-dimensional Milstein would require simulating Lévy areas and was ruled ineligible; both plans use vector Euler-Maruyama.

## Plans

| Plan | Scheme | Final dt | Score | Iterations |
|---|---|---|---|---|
| `1-euler-maruyama` | Euler-Maruyama | 0.001 (refined from initial 0.005) | **10/10** | 1 |
| `2-euler-maruyama-fine` | Euler-Maruyama | 0.001 | **10/10** | 1 |

Both plans use the same channel-aggregation trick: instead of drawing 13 independent Brownian increments per step, the 7 identical odd channels and 6 identical even channels are aggregated into two correlated normals scaled by `sqrt(7)` and `sqrt(6)` respectively — exact in law and far cheaper than drawing all 13.

**Plan 1** started at the plan-creator's proposed dt=0.005, but the solver diagnosed a ~5.3% deterministic drift-discretization bias in mean_X at that step size (isolated by testing the noise-free forward-Euler drift recursion in isolation), which combined with Monte Carlo sampling noise to exceed the 5% mean-error threshold. The solver self-corrected by refining dt to 0.001 before running the final solution.

**Plan 2** was designed from the outset as the accuracy-first fine-step alternative (dt=0.001, 5× finer than plan 1's original proposal), anticipating the same O(dt) multiplicative-noise variance bias.

Both plans converged to the same dt and produced numerically identical results:
- mean_X = -0.117573 (rel. err. 2.34%, threshold 5%)
- mean_Y = -0.151942 (rel. err. 0.74%, threshold 5%)
- variance_X = 0.173102 (rel. err. 0.88%, threshold 10%)
- variance_Y = 0.173268 (rel. err. 6.48%, threshold 10%)

Y's variance had the tightest margin (6.48% vs. the 10% threshold), consistent with the expected O(dt) Euler-Maruyama variance bias for multiplicative noise — well controlled at dt=0.001.

Both runs used `num_paths=50000`, `T=1.0`, `seed=42` (1000 steps), with mean and variance rel. errors computed against the analytic reference above.

## Best Plan

**`1-euler-maruyama`** is the recommended solution. Both plans scored 10/10 in 1 iteration with numerically identical results (plan 1's solver independently converged to the same dt=0.001 as plan 2's design), so the tie is broken by lower plan index; either is equally valid to use going forward.

## Notes

No plan failed to reach score 10 — no open numerical issues remain. Implicit/semi-implicit schemes (the theoretically preferred approach for stiff SDEs) are not yet supported by this pipeline; explicit Euler-Maruyama with a sufficiently fine step (dt=0.001, giving dt·|λ| ≈ 0.0036) proved adequate for this problem's stiffness level.
