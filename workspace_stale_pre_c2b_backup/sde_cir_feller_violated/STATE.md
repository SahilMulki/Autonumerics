---
phase: done
problem_type: sde
problem_spec: workspace/sde_cir_feller_violated/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Euler-Maruyama (strong order 0.5) with full-truncation positivity guard, dt=0.01, 50k paths, seed=42.
    iter: 1
    score: 10
  2-milstein:
    one-sentence: Milstein (strong order 1.0) with constant CIR correction and full-truncation guard, dt=0.01, 50k paths, seed=42.
    iter: 1
    score: 10
best_plan: 2-milstein
---
