---
phase: done
problem_type: sde
problem_spec: workspace/sde_quintic_drift_noise/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Tamed Euler-Maruyama (drift increment divided by 1+dt*|drift|) to prevent blow-up from superlinear -X^5 drift, dt=0.01, 50k paths, seed=42, T=1.0.
    iter: 1
    score: 10
    state: await_solver
  2-milstein:
    one-sentence: Tamed Milstein (drift taming plus 0.5*X*(dW^2-dt) diffusion correction), dt=0.01, 50k paths, seed=42, T=1.0.
    iter: 1
    score: 10
    state: await_solver
best_plan: 2-milstein
---
