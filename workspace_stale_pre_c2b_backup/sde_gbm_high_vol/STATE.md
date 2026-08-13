---
phase: done
problem_type: sde
best_plan: 2-milstein
problem_spec: workspace/sde_gbm_high_vol/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Baseline Euler-Maruyama Monte Carlo (dt=0.005, 200000 paths); large path count to tame heavy-tailed variance estimator at sigma=1.0.
    iter: 2
    score: 10
    state: await_solver
  2-milstein:
    one-sentence: Milstein scheme with correction 0.5*sigma^2*X*(dW^2-dt) (dt=0.005, 200000 paths); strong order 1.0 to reduce large-sigma discretization bias.
    iter: 2
    score: 10
    state: await_solver
---
