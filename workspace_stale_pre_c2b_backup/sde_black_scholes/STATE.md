---
phase: done
problem_type: sde
problem_spec: workspace/sde_black_scholes/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Baseline Euler-Maruyama Monte Carlo (strong order 0.5), dt=0.01, num_paths=50000, T=1.0, seed=42.
    iter: 1
    score: 10
  2-milstein:
    one-sentence: Milstein scheme with GBM correction 0.5*sigma^2*X_n*(dW^2-dt) for strong order 1.0, dt=0.01, num_paths=50000, T=1.0, seed=42.
    iter: 1
    score: 10
best_plan: 2-milstein
---
