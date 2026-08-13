---
phase: done
problem_type: sde
problem_spec: workspace/sde_multichannel_stiff_m13/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Vector Euler-Maruyama; solver refined dt from 0.005 to 0.001 after finding drift discretization bias, using odd/even channel aggregation.
    iter: 1
    score: 10
    state: await_solver
  2-euler-maruyama-fine:
    one-sentence: Accuracy-first Euler-Maruyama at dt=0.001 (5x finer) to suppress O(dt) multiplicative-noise variance bias in this stiff, strong-noise regime.
    iter: 1
    score: 10
    state: await_solver
best_plan: 1-euler-maruyama
---
