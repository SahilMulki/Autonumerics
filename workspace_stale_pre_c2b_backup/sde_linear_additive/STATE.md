---
phase: done
problem_type: sde
problem_spec: workspace/sde_linear_additive/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Euler-Maruyama Monte Carlo (dt=0.01, 50000 paths, T=1.0, seed=42) for dX=(2-X)dt+0.5 dW; additive noise so Milstein reduces to EM.
    iter: 1
    score: 10
    state: await_solver
best_plan: 1-euler-maruyama
---
