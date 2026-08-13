---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Euler-Maruyama Monte Carlo on the 2D linear oscillator with additive noise entering only the velocity (Y) component.
---

## SDE Reference

Coupled Ito system (stochastic harmonic oscillator):

    dX = Y * dt,                 X(0) = 1.0
    dY = -X * dt + sigma * dW,   Y(0) = 0.0

over t in [0, 2*pi]. Noise enters only the Y (velocity) component.

**Parameters**:
- X_0 = 1.0
- Y_0 = 0.0
- sigma = 0.3
- T = 2 * pi = 6.283185307179586

**Structure**: state_dimension = 2, additive noise, linear, non-stiff.
Diffusion matrix G = [[0], [sigma]] (noise only on Y).

**Analytic moments** (t = T):
- mean_X     = X_0 * cos(t) + Y_0 * sin(t)
- variance_X = sigma**2 / 2 * (t - sin(t) * cos(t))
- mean_Y     = -X_0 * sin(t) + Y_0 * cos(t)
- variance_Y = sigma**2 / 2 * (t + sin(t) * cos(t))

At t = 2*pi: mean_X ~= 1.0, mean_Y ~= 0.0 (near zero — mean check on Y is skipped per near_zero_mean_threshold = 0.01).

## Numerical Scheme

- **Method**: Euler-Maruyama (strong order 0.5, weak order 1.0)
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 6.283185307179586
- **Milstein correction**: N/A — additive noise (dg/dstate = 0) and multi-D (would require Lévy area). Milstein == EM here, so EM is the only scheme proposed.

**Per-step update** (all paths vectorized, shape (num_paths,) per component):
```
dW = sqrt(dt) * Z,   Z ~ N(0, 1)
X_{n+1} = X_n + Y_n * dt
Y_{n+1} = Y_n - X_n * dt + sigma * dW
```
Note: use the old X_n when updating Y (compute both updates from the same time-level values, do not overwrite X before using it in the Y update).

## Implementation Notes

- Only one Brownian driver W (scalar noise) shared across the Y component; X has no stochastic term.
- Store state as two arrays X, Y of shape (num_paths,), or a (num_paths, 2) array. If using a combined array, be careful to compute the Y increment from the pre-update X.
- No positivity or log guards needed (linear system, state can be negative — that is physical for an oscillator).
- Report empirical mean and variance for BOTH X(T) and Y(T). The evaluator compares per-component moments.
- Mean of Y at T = 2*pi is near zero; the evaluator skips the relative mean check on Y and relies on the variance check. Mean of X (~1.0) is checked normally.
- Both variances are strictly positive and grow ~ sigma^2/2 * t; use ddof=1 for empirical variance.
- If variance accuracy is insufficient, the solver may refine dt (e.g. 0.005 or 0.001); weak order 1.0 in dt should make 0.01 adequate over one period.

## Results

Ran with num_paths=50000, dt=0.01, T=2*pi (Nt=629 steps).

- Empirical mean X:     1.035287  (exact: 1.000000, rel error: 3.53%)
- Empirical mean Y:     0.002206  (exact: ~0.0, near-zero — mean check skipped)
- Empirical variance X: 0.288725  (exact: 0.282743, rel error: 2.12%)
- Empirical variance Y: 0.294114  (exact: 0.282743, rel error: 4.02%)

All errors are within the thresholds (mean_rel_err < 5%, variance_rel_err < 10%).

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy
- Component X
  - mean_relative_error:     0.0353  (PASS)
  - variance_relative_error: 0.0212  (PASS)
- Component Y
  - mean_relative_error:     skipped (near-zero exact mean)
  - variance_relative_error: 0.0402  (PASS)
- Overall: PASS ✓

### Feedback for solver
- None

</review>
