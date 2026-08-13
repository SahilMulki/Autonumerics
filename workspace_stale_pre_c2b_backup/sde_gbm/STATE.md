---
phase: done
problem_type: sde
problem_spec: workspace/sde_gbm/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Vanilla Euler-Maruyama Monte Carlo (strong order 0.5), dt=0.01, 50000 paths, seed 42.
    iter: 1
    score: 10
    state: await_solver
  2-milstein:
    one-sentence: Milstein scheme with GBM correction 0.5*sigma^2*X_n*(dW^2-dt) (strong order 1.0), dt=0.01, 50000 paths, seed 42.
    iter: 1
    score: 10
best_plan: 1-euler-maruyama
---
