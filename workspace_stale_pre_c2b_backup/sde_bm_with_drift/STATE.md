---
phase: done
problem_type: sde
problem_spec: workspace/sde_bm_with_drift/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Monte Carlo Euler-Maruyama (dt=0.01, num_paths=50000, seed=42, T=1.0) for additive-noise scalar SDE; EM is distributionally exact here, targeting mean=1.5, variance=0.09.
    iter: 1
    score: 10
best_plan: 1-euler-maruyama
---
