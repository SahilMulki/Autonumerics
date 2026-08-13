```yaml
phase: done
problem_type: sde
problem_spec: workspace/sde_exponential_ou/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Baseline strong-order-0.5 Euler-Maruyama Monte Carlo (dt=0.01, 50000 paths, T=1.0) with a positivity guard on the log-drift.
    iter: 1
    score: 10
  2-milstein:
    one-sentence: Strong-order-1.0 Milstein Monte Carlo using the correction 0.5*sigma^2*X_n*(dW^2-dt) from g(X)=sigma*X, for sharper variance at the same dt.
    iter: 1
    score: 10
best_plan: 2-milstein
```
