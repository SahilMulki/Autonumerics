---
phase: running
problem_type: sde
problem_spec: workspace/sde_ginzburg_landau_s6/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Tamed Euler-Maruyama (drift-tamed only), dt=0.0001, 50000 paths, seed 42.
    iter: 0
    score: 0
    state: await_evaluator
  2-milstein:
    one-sentence: Tamed drift + Milstein diffusion correction, dt=0.005, 50000 paths, seed 42.
    iter: 1
    score: 10
    state: await_solver
---
