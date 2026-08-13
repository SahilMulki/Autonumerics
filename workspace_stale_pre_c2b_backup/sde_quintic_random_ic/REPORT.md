# Report: sde_quintic_random_ic

## Problem

**Type**: SDE (classified per pipeline convention), **family**: noise-free quintic decay with a random Gaussian initial condition.

```
dX = -X^5 dt,     X(0) = xi ~ N(0, sigma_bar^2),   sigma_bar = 1/3
T0 = 0, T = 1
```

There is no Wiener/diffusion term (`g = 0`); all randomness comes from the Gaussian initial condition. The drift `-X^5` is strongly superlinear (stiff), so a plain explicit Euler-Maruyama step blows up on large-`|xi|` paths. Milstein is not eligible (no diffusion term to correct). An exact per-path closed form exists because the ODE is separable:

```
X_t = xi / (1 + 4t xi^4)^(1/4)
```

giving `exact_mean = 0` (odd symmetry, skipped per the near-zero-mean rule) and `exact_variance = 0.09248275891171216` (a 1D Gaussian quadrature, cross-checked against a 2M-path Monte Carlo simulation during formulation).

## Plans

| Plan | Scheme | Score | Iter | Variance rel. err |
|---|---|---|---|---|
| 1-tamed-euler-maruyama | Tamed explicit Euler: `drift = -X^5`; `X += drift*dt / (1 + dt*|drift|)`, dt=0.001, 50000 paths | **10/10** | 1 | 0.35% |
| 2-exact-flow-integration | Direct closed-form flow map `X_T = xi/(1+4T xi^4)^(1/4)` per path, 50000 paths | **10/10** | 1 | 0.35% |

Both plans used seed 42 and skipped the mean check (`empirical_mean ≈ -0.0002`, well under the `|exact_mean| < 0.01` threshold). Both passed on the first solver↔evaluator cycle with no refinement needed.

- **Plan 1** stabilizes the quintic drift by dividing the increment by `1 + dt·|drift|`, which caps the per-step update and prevents the double-exponential divergence that plain explicit Euler suffers on large-`|xi|` paths. At `dt = 0.001` this taming bias is small enough that the variance estimate lands at 0.35% relative error.
- **Plan 2** sidesteps discretization entirely by evaluating the exact per-path solution in one step. Its only error source is Monte Carlo sampling noise at 50000 paths, and it lands at essentially the same 0.35% relative error as plan 1 — confirming plan 1's tamed-Euler bias is negligible at this `dt`.

## Best Plan

**`2-exact-flow-integration`** is the recommended solution. Both plans tie on score (10/10) and iteration count (1), but plan 2 is preferable on methodological grounds: it has zero discretization bias by construction (uses the exact closed-form flow map rather than a stabilized numerical stepper), requires no `dt`-tuning or taming heuristics, and its result served as an internal cross-check that validated plan 1's tamed-Euler scheme is accurate. For SDEs in this family where the noise is confined to the initial condition and the drift ODE is separable/solvable, integrating the exact flow map is both simpler and strictly more robust than any time-stepping scheme.

## Outstanding Issues

None. Both plans reached score 10 on the first iteration with no non-finite values, no refinement needed, and no unresolved evaluator feedback.
