---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Euler-Maruyama Monte Carlo for a linear SDE with additive noise (Milstein reduces to EM here).
---

## SDE Reference

Ito SDE:

    dX(t) = (a + b * X(t)) * dt + c * dW(t),   X(0) = 0.0

Parameters:
- X_0 = 0.0
- a = 2.0
- b = -1.0
- c = 0.5

Time interval: [T0=0.0, T=1.0]

Drift: f(X) = a + b*X = 2.0 - 1.0*X  (linear, mean-reverting toward a/b = 2)
Diffusion: g(X) = c = 0.5  (constant → additive noise)

Analytic moments (t = T = 1.0):
- exact_mean     = np.exp(b*t) * (X_0 + a/b) - a/b  ≈ 2*(1 - exp(-1)) ≈ 1.2642
- exact_variance = c**2 / (2*b) * (np.exp(2*b*t) - 1) = 0.125*(1 - exp(-2)) ≈ 0.1081

## Numerical Scheme

- **Method**: Euler-Maruyama
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0
- **Milstein correction**: 0 (additive noise, g is constant so dg/dX = 0; Milstein is identical to Euler-Maruyama). No separate Milstein plan is warranted.

Scalar EM update:
```
dW_n    = sqrt(dt) * Z_n,   Z_n ~ N(0,1)
X_{n+1} = X_n + (a + b*X_n) * dt + c * dW_n
```

## Implementation Notes

- Additive noise: no positivity guard needed and no Milstein correction term.
- Mean is well above the near-zero threshold (|exact_mean| ≈ 1.264 >> 0.01), so the mean relative error check applies (target < 5%).
- Variance target: relative error < 10%.
- State is a plain scalar array of shape (num_paths,); step all paths simultaneously with NumPy broadcasting.
- Linear drift is unconditionally stable at dt=0.01 (b=-1, |1 + b*dt| < 1), no stability concern. Refine dt only if the evaluator reports accuracy issues.

## Results

Empirical mean:     1.270954
Empirical variance: 0.110484

Analytic mean ≈ 1.2642, analytic variance ≈ 0.1081.
Mean relative error ≈ 0.53% (well within 5% threshold).
Variance relative error ≈ 2.2% (well within 10% threshold).

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy
- mean_relative_error:     0.0053  (PASS)
- variance_relative_error: 0.0222  (PASS)
- Overall: PASS ✓

### Feedback for solver
- None

</review>
