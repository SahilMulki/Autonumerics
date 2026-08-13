---
id: 2
plan_slug: milstein
scheme: milstein
strategy: Strong-order-1.0 Milstein Monte Carlo exploiting the multiplicative diffusion g(X)=sigma*X for a sharper variance estimate.
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
Diffusion:  g(X) = sigma * X,   dg/dX = sigma

Analytic moments (X_t is lognormal; log X is an OU process):

    v_t = sigma**2 / (2*theta) * (1 - np.exp(-2*theta*t))
    exact_mean     = np.exp(v_t / 2)
    exact_variance = np.exp(v_t) * (np.exp(v_t) - 1)

## Numerical Scheme

- **Method**: Milstein
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0
- **Milstein correction**: 0.5 * g(X) * g'(X) * (dW**2 - dt)
  = 0.5 * (sigma * X_n) * sigma * (dW**2 - dt)
  = 0.5 * sigma**2 * X_n * (dW**2 - dt)

Scalar update:

    dW = sqrt(dt) * Z,   Z ~ N(0,1)
    X_{n+1} = X_n + f(X_n) * dt + g(X_n) * dW
             + 0.5 * sigma**2 * X_n * (dW**2 - dt)

## Implementation Notes

- The drift contains log(X), which requires X > 0. Apply a positivity guard
  `X_pos = np.maximum(X_n, 1e-12)` before computing log(X) in the drift term.
- Milstein is eligible here because the noise is multiplicative scalar with
  g(X) = sigma*X and dg/dX = sigma (state_dimension == 1).
- The Milstein correction reuses the same dW draw as the EM diffusion term;
  do not draw a fresh normal for the correction.
- With X_0 = 1.0, theta = 1.0, sigma = 0.4, X stays positive under normal
  stepping, but the guard is mandatory.
- Monte Carlo layout: X has shape (num_paths,); step all paths simultaneously.
- Milstein is strong order 1.0, so it should hit the variance threshold at
  dt = 0.01; refine only if accuracy is marginal.

## Results

Empirical mean:     1.038326
Empirical variance: 0.078941

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy
- mean_relative_error:     0.0030  (PASS)
- variance_relative_error: 0.0285  (PASS)
- Overall: PASS ✓

### Feedback for solver
- None

</review>
