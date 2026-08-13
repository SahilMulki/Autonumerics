---
id: 1
plan_slug: tamed-euler-maruyama
scheme: euler-maruyama
strategy: Stabilized (tamed) explicit Euler on the quintic drift, dividing the drift by 1 + dt*|drift| so large-|xi| paths cannot overshoot and blow up.
---

## SDE Reference

Noise-free Ito SDE with a random Gaussian initial condition:

```
dX = -X**5 * dt,     X(0) = xi ~ Normal(0, sigma_bar**2)
```

- `sigma_bar` = 1/3 = 0.3333333333333333
- `drift_exponent` = 5
- Time interval: T0 = 0.0, T = 1.0
- Drift: f(X) = -X**5
- Diffusion: g(X) = 0 (NO Wiener increment; the only randomness is in xi)

Exact per-path solution: `X_t = xi / (1 + 4*t*xi**4)**0.25`.

Analytic moments at T = 1:
- `exact_mean = 0.0` (odd symmetry; near-zero -> mean check skipped)
- `exact_variance = 0.09248275891171216` (1D Gaussian quadrature of `xi**2 / sqrt(1 + 4*T*xi**4)` against `N(0, sigma_bar**2)`)

## Numerical Scheme

- **Method**: Euler-Maruyama (tamed / stabilized explicit variant)
- **dt**: 0.001
- **num_paths**: 50000
- **T**: 1.0
- **seed**: 42
- **Milstein correction**: N/A (g == 0; noise-free)

Monte Carlo is over the initial condition ONLY. Initialize the path ensemble by
sampling `X = rng.normal(0.0, sigma_bar, size=num_paths)` — do NOT start from a
constant `X_0`. There is no `dW` term; each step is a purely deterministic
tamed drift update:

```
drift = -X**5
X = X + (drift * dt) / (1.0 + dt * np.abs(drift))
```

## Implementation Notes

- **Do NOT use plain explicit Euler** `X = X - X**5 * dt`. The quintic drift is
  strongly superlinear (stiff): paths with large |xi| overshoot in one step,
  flip sign with a larger magnitude, and diverge double-exponentially, producing
  inf/nan and a wrecked variance estimate. The taming denominator
  `1 + dt*|drift|` caps the per-step increment and keeps the scheme stable.
- No Wiener increment: the random-number generator is used exactly once, to draw
  the `num_paths` Gaussian initial conditions. Do not add `sqrt(dt)*Z` noise.
- Return `empirical_mean = mean(X)` and `empirical_variance = var(X, ddof=1)` at
  the terminal time. The exact mean is ~0, so the evaluator skips the mean check
  (|exact_mean| < near_zero_mean_threshold = 0.01); the variance target is
  `variance_rel_err < 10%` vs 0.09248275891171216.
- If accuracy is marginal, the solver may refine `dt` (e.g. 5e-4 or 1e-4);
  taming introduces an O(dt) bias that shrinks with smaller steps.
- Sanity check: the terminal variance (0.0925) must be below the initial-condition
  variance `sigma_bar**2 = 0.1111`, since the decay contracts every path toward 0.

## Results

Ran `solve_sde(num_paths=50000, dt=0.001, T=1.0, seed=42)` with the tamed
explicit Euler scheme described above.

- `empirical_mean`     = -0.000202
- `empirical_variance` = 0.092804
- `dt` used             = 0.001
- `num_paths`           = 50000

Comparison to analytic moments:
- `exact_mean` = 0.0 → |empirical_mean| = 0.0002, well under the
  `near_zero_mean_threshold` = 0.01, so the mean check is skipped as expected.
- `exact_variance` = 0.09248275891171216 → relative error
  = |0.092804 - 0.09248275891171216| / 0.09248275891171216 ≈ 0.35%,
  well within the 10% threshold.

No non-finite values were produced; the taming denominator successfully
prevented blow-up on large-|xi| paths.

## Evaluation

(Filled by evaluator)

<review score=10>

Score: 10/10 — Done

### Numerical Accuracy
- mean_relative_error:     N/A  (skipped, |exact_mean| = 0.0 < near_zero_mean_threshold = 0.01; empirical_mean = -0.000202)
- variance_relative_error: 0.0035  (PASS, threshold < 0.10)
- Overall: PASS ✓

### Feedback for solver
None.

</review>
