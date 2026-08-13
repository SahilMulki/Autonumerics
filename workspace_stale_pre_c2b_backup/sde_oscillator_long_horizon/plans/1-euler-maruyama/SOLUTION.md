---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Euler-Maruyama with a small step size to control long-horizon phase and variance error for the 2D additive-noise oscillator.
---

## SDE Reference

Coupled Ito system (2D, additive noise entering only the Y component):

```
dX = Y dt,                 X(0) = 1.0
dY = -X dt + sigma dW(t),  Y(0) = 0.0
```

Parameters:
- `X_0 = 1.0`
- `Y_0 = 0.0`
- `sigma = 0.3`

Time interval: `T0 = 0.0`, `T = 10*pi = 31.41592653589793` (five full periods).

Diffusion matrix: `G = [[0], [sigma]]` (constant — additive noise).

Analytic per-component moments (valid for any t, evaluated at t = T):
```python
mean_X     = X_0 * np.cos(t) + Y_0 * np.sin(t)
variance_X = sigma**2 / 2 * (t - np.sin(t) * np.cos(t))
mean_Y     = -X_0 * np.sin(t) + Y_0 * np.cos(t)
variance_Y = sigma**2 / 2 * (t + np.sin(t) * np.cos(t))
```
At `T = 10*pi`: `mean_X = 1.0`, `mean_Y = 0.0` (near-zero — mean check for Y is skipped).

## Numerical Scheme

- **Method**: Euler-Maruyama (additive noise + multi-D, so Milstein is neither needed nor well-defined — the diffusion is constant, giving zero Milstein correction, and multi-D Milstein requires Lévy areas).
- **dt**: 0.001
- **num_paths**: 50000
- **T**: 31.41592653589793 (10*pi)
- **Milstein correction**: 0 (additive noise, Milstein == EM)

## Implementation Notes

- State array shape `(num_paths, 2)`, columns `[X, Y]`.
- Vectorized update per step (use current-step X and Y, i.e. explicit EM):
  ```
  dW  = sqrt(dt) * Z,  Z ~ N(0,1), shape (num_paths,)
  X_next = X + Y * dt
  Y_next = Y - X * dt + sigma * dW
  ```
  Update both simultaneously from the old X, Y (do not overwrite X before computing Y's drift).
- Noise enters only the Y component; the X update is purely deterministic per step.
- **Long-horizon stability warning**: explicit EM on the harmonic oscillator has an amplitude-amplifying tendency — the deterministic step matrix has spectral radius `sqrt(1 + dt^2) > 1`, so energy grows geometrically over `Nt = T/dt` steps. Over five periods this inflates both the mean amplitude (phase/amplitude error) and the variance. A small `dt` (0.001) keeps `(1 + dt^2)^(Nt/2)` close to 1 and controls this drift. If the evaluator reports variance or mean_X error above threshold, reduce `dt` further (e.g. 0.0005 or 0.0002).
- `mean_Y` is near zero at T = 10*pi; the evaluator skips the relative mean check for Y (uses `near_zero_mean_threshold = 0.01`).
- Ensure `Nt = round(T/dt)` and rescale `dt = T/Nt` so the terminal time lands exactly on T = 10*pi.

## Results

Run with `num_paths=50000`, `dt=0.001`, `T=10*pi`:

- `empirical_mean`: `[1.0138329172525071, -0.0015946290869691333]`
- `empirical_variance`: `[1.4335943559847664, 1.4403833747213974]`

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy

Component X:
- mean_relative_error:     0.0138  (PASS)
- variance_relative_error: 0.0141  (PASS)
- Overall: PASS

Component Y:
- mean_relative_error:     skipped (near-zero exact mean at T=10*pi)
- variance_relative_error: 0.0189  (PASS)
- Overall: PASS

### Feedback for solver
- None

</review>
