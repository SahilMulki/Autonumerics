---
phase: done
problem_type: sde
problem_spec: workspace/sde_stochastic_oscillator/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Euler-Maruyama MC on 2D linear stochastic harmonic oscillator; additive noise in Y, dt=0.01, num_paths=50000, seed=42, T=2*pi.
    iter: 1
    score: 10
    state: await_solver
best_plan: 1-euler-maruyama
---
