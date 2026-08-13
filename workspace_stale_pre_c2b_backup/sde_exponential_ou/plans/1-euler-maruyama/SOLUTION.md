---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Baseline strong-order-0.5 Euler-Maruyama Monte Carlo with a positivity guard on the log-drift.
---

## SDE Reference

Ito SDE (exponential Ornstein-Uhlenbeck):

    dX(t) = X(t) * (-theta * log(X(t)) + sigma**2 / 2) * dt + sigma * X(t) * dW(t)

Parameters:
- X_0 = 1.0
- theta = 1.0
- sigma = 0.4
- Time interval: [0.0, 1.0], T = 1.0

Drift:      f(X) = X * (-theta * log(X) + sigma**2 / 2)
Diffusion:  g(X) = sigma * X

Analytic moments (X_t is lognormal; log X is an OU process):

    v_t = sigma**2 / (2*theta) * (1 - np.exp(-2*theta*t))
    exact_mean     = np.exp(v_t / 2)
    exact_variance = np.exp(v_t) * (np.exp(v_t) - 1)

## Numerical Scheme

- **Method**: Euler-Maruyama
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0

Scalar update:

    dW = sqrt(dt) * Z,   Z ~ N(0,1)
    X_{n+1} = X_n + f(X_n) * dt + g(X_n) * dW

## Implementation Notes

- The drift contains log(X), which requires X > 0. Apply a positivity guard
  `X_pos = np.maximum(X_n, 1e-12)` before computing log(X) in the drift term.
- With X_0 = 1.0, theta = 1.0, sigma = 0.4, X stays positive under normal
  stepping, but the guard is mandatory to avoid log of zero/negative from
  floating-point excursions.
- Monte Carlo layout: X has shape (num_paths,); step all paths simultaneously
  with NumPy broadcasting.
- EM is strong order 0.5. If variance accuracy is marginal at T=1, the solver
  may refine dt (e.g. 0.005 or 0.001).

## Results

Ran with num_paths=50000, dt=0.01, T=1.0, seed=42.

- Empirical mean:     1.038344
- Empirical variance: 0.078993

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy
- mean_relative_error:     0.0030  (PASS)
- variance_relative_error: 0.0292  (PASS)
- Overall: PASS ✓

### Feedback for solver
- None

</review>
