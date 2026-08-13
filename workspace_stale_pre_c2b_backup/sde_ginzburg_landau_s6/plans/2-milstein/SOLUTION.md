---
id: 2
plan_slug: milstein
scheme: milstein
strategy: Tamed drift plus the Milstein diffusion correction to reach strong order 1.0 on the multiplicative noise while keeping the cubic drift finite.
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
Diffusion:  `g(X) = sigma * X`,  `dg/dX = sigma`

Reference moments (discretization-free Monte-Carlo of the exact pathwise solution
`X_t = X_0*exp(sigma*W_t) / sqrt(1 + 2*X_0**2 * integral_0^t exp(2*sigma*W_u) du)`):
- `E[X(T)] ~= 0.676`
- `Var[X(T)] ~= 1.88`

Mean is NOT near zero, so the mean check applies.

## Numerical Scheme

- **Method**: Milstein with TAMED drift (plain drift explodes at sigma=6)
- **dt**: 0.0001 (see Implementation Notes: dt=0.005 and even dt=0.001 were
  empirically insufficient — rare paths with a large diffusion shock in one
  step overwhelm the drift taming and blow up, inflating the empirical
  variance far above the 1.88 reference)
- **num_paths**: 50000
- **T**: 1.0
- **seed**: 42

**Milstein correction** (from problem_spec.json):
```
0.5 * g(X) * dg/dX * (dW**2 - dt)
= 0.5 * (sigma*X) * sigma * (dW**2 - dt)
= 0.5 * sigma**2 * X * (dW**2 - dt)
```

**Tamed update** (tame only the drift; apply the Milstein correction on the
multiplicative diffusion):
```
f       = (sigma**2/2) * X - X**3
f_tamed = f * dt / (1 + abs(f) * dt)
dW      = sqrt(dt) * Z,   Z ~ N(0,1)
X_{n+1} = X + f_tamed + sigma * X * dW
        + 0.5 * sigma**2 * X * (dW**2 - dt)
```

The Milstein term raises the strong order from 0.5 to 1.0 on the multiplicative
noise. It improves diffusion accuracy but does NOT by itself fix the drift
explosion, so the tamed drift is still required.

## Implementation Notes

- **Taming is mandatory**: at sigma=6 the `-X**3` drift alone causes explosive
  intermediate values (partial Inf/NaN, biased variance). Tame the drift; leave
  the diffusion and the Milstein correction untamed.
- **Empirically, dt must be much smaller than the initial 0.005 guess.**
  The drift taming caps the drift step at magnitude ~1 (a Hutzenthaler-Jentzen-
  Kloeden style cap, independent of dt), but the diffusion term `sigma*X*dW`
  and the Milstein term `0.5*sigma**2*X*(dW**2-dt)` are left untamed and scale
  with X. A single large Gaussian shock can push X to a large value faster
  than the capped drift can pull it back, and with only 50000 paths a rare
  outlier path dominates the ddof=1 variance estimator. Measured with
  `num_paths=50000, seed=42`:
  - `dt=0.005`: mean≈1.63, var≈320 (fails badly — outlier explosion)
  - `dt=0.001`: mean≈0.78, var≈15.5 (still fails — one path reached X≈796)
  - `dt=0.0005`: mean≈0.696 (2.9% err), var≈2.01 (7.1% err) — passes, thin margin
  - `dt=0.0001`: mean≈0.678 (0.3% err), var≈1.90 (1.1% err) — passes with margin
  `dt=0.0001` (10000 steps) is used as the final value; runtime ≈8s for 50000 paths.
- The Milstein correction `0.5*sigma**2*X*(dW**2 - dt)` can be large when
  `sigma=6`; combined with the small `dt` above it stays controlled, and
  `np.all(np.isfinite(terminal_paths))` is verified True at dt=0.0001.
- Reference moments are Monte-Carlo estimates (mean ~0.676, variance ~1.88), not
  closed-form. Match within tolerances (`variance_rel_err < 10%`,
  `mean_rel_err < 5%`).

## Results

Run with `num_paths=50000, dt=0.0001 (Nt=10000), T=1.0, seed=42`:

- Empirical mean:     0.677748  (reference ≈0.676, rel. err ≈0.26%)
- Empirical variance: 1.901061  (reference ≈1.88, rel. err ≈1.12%)
- All terminal values finite: True

Both moments are well within the evaluation thresholds
(`variance_rel_err < 10%`, `mean_rel_err < 5%`).

## Evaluation

(Filled by evaluator)

<review score=10>

Score: 10/10 — Done

**Score: 10/10**

### Numerical Accuracy
- mean_relative_error:     0.0026  (PASS)
- variance_relative_error: 0.0112  (PASS)
- Overall: PASS ✓

### Feedback for solver
None

</review>
