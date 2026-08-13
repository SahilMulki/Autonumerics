---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Tamed Euler-Maruyama to keep the cubic drift from exploding at sigma=6, with a small time step for stability.
---

## SDE Reference

Stochastic Ginzburg-Landau equation (Ito form):

```
dX(t) = ((sigma**2/2) * X(t) - X(t)**3) * dt + sigma * X(t) * dW(t)
```

Parameters:
- `X_0 = 1.0`
- `sigma = 6.0`
- `T0 = 0.0`, `T = 1.0`

Drift:  `f(X) = (sigma**2/2) * X - X**3`
Diffusion:  `g(X) = sigma * X`

Reference moments (discretization-free Monte-Carlo of the exact pathwise solution
`X_t = X_0*exp(sigma*W_t) / sqrt(1 + 2*X_0**2 * integral_0^t exp(2*sigma*W_u) du)`):
- `E[X(T)] ~= 0.676`
- `Var[X(T)] ~= 1.88`

Mean is NOT near zero, so the mean check applies.

## Numerical Scheme

- **Method**: Euler-Maruyama with TAMED drift (plain EM is unusable at sigma=6)
- **dt**: 0.0001 (revised down from the originally proposed 0.005 — see Implementation Notes)
- **num_paths**: 50000
- **T**: 1.0
- **seed**: 42

**Tamed update** (diffusion term left unmodified):
```
f       = (sigma**2/2) * X - X**3
f_tamed = f * dt / (1 + abs(f) * dt)          # tame only the drift
dW      = sqrt(dt) * Z,   Z ~ N(0,1)
X_{n+1} = X + f_tamed + g(X) * dW
        = X + f_tamed + sigma * X * dW
```

The taming caps the per-step drift increment so the cubic `-X**3` term cannot
produce explosive intermediate values, while preserving the correct drift for
moderate `X`. Do NOT tame the diffusion term.

## Implementation Notes

- **Plain (untamed) EM will fail here**: at sigma=6 the `-X**3` drift causes
  explosive intermediate values, yielding partial Inf/NaN terminal paths and a
  severely underestimated variance. Taming the drift is mandatory.
- **dt=0.005 (the originally proposed value) is NOT small enough, even with the
  drift tamed.** Only the drift is tamed here; the diffusion term `sigma*X*dW`
  is left unmodified (linear in X), so a single large Gaussian draw while X is
  transiently elevated can still produce a large multiplicative jump. Empirically:
  - `dt=0.005` -> mean=0.165, var=90.4 (both badly wrong; finite but heavily biased)
  - `dt=0.002`  -> mean=0.616, var=77.0 (still unstable)
  - `dt=0.001`  -> mean=0.641, var=2.51 (mean converged, variance still off)
  - `dt=0.0005` -> mean=0.640, var=1.88 (variance converged, mean rel err ~5.4%, borderline)
  - `dt=0.0001` -> mean=0.667, var=1.88 (both well within tolerance)
  `dt=0.0001` (Nt=10000 steps) was adopted for a comfortable safety margin
  (mean rel err ~1.3%, variance rel err ~0.1%) against the 5%/10% thresholds.
  It remains well inside the `dt <= 0.04/T = 0.04` verifier cap.
- The exact solution stays positive for `X_0 > 0`; the tamed scheme should keep
  paths finite. Check `np.all(np.isfinite(terminal_paths))` before reporting.
- Reference moments are Monte-Carlo estimates (mean ~0.676, variance ~1.88), not
  closed-form values. Match them within the eval tolerances
  (`variance_rel_err < 10%`, `mean_rel_err < 5%`).

## Results

Ran `solve_sde(num_paths=50000, dt=0.0001, T=1.0, seed=42)` (tamed EM, Nt=10000 steps):

- Empirical mean:     0.667205
- Empirical variance: 1.881942
- All terminal paths finite: True
- dt (actual):        0.000100

Comparison to reference moments (mean ~0.676, variance ~1.88):
- Mean relative error:     ~1.30%  (threshold 5%) -> PASS
- Variance relative error: ~0.10%  (threshold 10%) -> PASS

## Evaluation

(Filled by evaluator)

<review score=0>

Awaiting solver.

</review>
