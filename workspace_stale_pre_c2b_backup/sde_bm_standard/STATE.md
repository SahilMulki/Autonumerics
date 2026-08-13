---
phase: done
problem_type: sde
problem_spec: workspace/sde_bm_standard/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Euler-Maruyama Monte Carlo for standard Brownian motion (dX = dW), dt=0.01, 50k paths, seed=42, T=1.0; scored on variance_rel_err against exact_variance=1.0 (mean check skipped since exact_mean=0).
    iter: 1
    score: 10
    state: await_solver
best_plan: 1-euler-maruyama
---
