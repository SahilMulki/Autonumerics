---
phase: done
problem_type: sde
problem_spec: workspace/sde_ornstein_uhlenbeck/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Euler-Maruyama Monte Carlo (dt=0.01, num_paths=50000, seed=42, T=1.0) for the additive-noise OU process; Milstein offers no benefit since dg/dX = 0.
    iter: 1
    score: 10
    state: await_solver
best_plan: 1-euler-maruyama
---
