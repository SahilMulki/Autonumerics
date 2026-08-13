---
phase: done
problem_type: sde
problem_spec: workspace/sde_gbm_2d_high_corr/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Vector Euler-Maruyama Monte Carlo (dt=0.01, 50k paths, seed=42) with Cholesky-correlated Brownian increments (rho=0.95) stepping two GBM components.
    iter: 1
    score: 10
    state: await_solver
best_plan: 1-euler-maruyama
---
