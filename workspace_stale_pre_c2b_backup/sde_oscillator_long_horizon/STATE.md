---
phase: done
problem_type: sde
problem_spec: workspace/sde_oscillator_long_horizon/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Euler-Maruyama on the 2D additive-noise oscillator with dt=0.001, 50k paths, T=10*pi, seed 42.
    iter: 1
    score: 10
best_plan: 1-euler-maruyama
---
