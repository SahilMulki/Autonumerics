---
phase: running
problem_type: pde
problem_spec: workspace/heat_1d/problem_spec.json
plans:
  1-fd-explicit:
    one-sentence: FTCS explicit finite difference, Nx=100, dt with r=alpha*dt/dx^2=0.4 (CFL-safe).
    iter: 1
    score: 10
    state: await_solver
  2-crank-nicolson:
    one-sentence: Crank-Nicolson implicit, Nx=100, dt=0.01, unconditionally stable.
    iter: 1
    score: 10
    state: await_solver
---
