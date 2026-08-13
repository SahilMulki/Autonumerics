---
phase: done
problem_type: sde
problem_spec: workspace/sde_gbm_2d_correlated/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Euler-Maruyama MC on coupled 2D correlated GBM (dt=0.01, 50k paths, seed=42, T=1.0), correlated increments via Cholesky.
    iter: 1
    score: 10
    state: await_solver
best_plan: 1-euler-maruyama
---
