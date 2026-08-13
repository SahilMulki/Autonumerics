---
phase: done
problem_type: sde
problem_spec: workspace/sde_ginzburg_landau_s4/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Tamed Euler-Maruyama with dt=0.0002 (refined from 0.005), 50000 paths; tames the cubic drift, leaves multiplicative diffusion untamed.
    iter: 1
    score: 10
    state: await_solver
  2-milstein:
    one-sentence: Tamed-drift Milstein with dt=0.0005 (refined from 0.005), 50000 paths; adds multiplicative-noise Milstein correction for strong order 1.0.
    iter: 1
    score: 10
    state: await_solver
best_plan: 2-milstein
---
