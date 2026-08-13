---
phase: done
problem_type: sde
problem_spec: workspace/sde_log_heston_feller_violated/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Full-truncation Euler-Maruyama on the 2D log-Heston system with positivity guard on Y before sqrt, shared dW1, dt=0.001.
    iter: 1
    score: 10
best_plan: 1-euler-maruyama
---
