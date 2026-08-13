---
phase: done
problem_type: sde
problem_spec: workspace/sde_cir/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Baseline Euler-Maruyama (strong order 0.5) with positivity guard before sqrt diffusion; dt=0.01, 50k paths, seed=42.
    iter: 1
    score: 10
  2-milstein:
    one-sentence: Milstein (strong order 1.0) adding (sigma^2/4)*(dW^2-dt) correction, same positivity guard; dt=0.01, 50k paths, seed=42.
    iter: 1
    score: 10
best_plan: 1-euler-maruyama
---
