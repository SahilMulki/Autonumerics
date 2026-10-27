---
phase: running
problem_type: sde
path: B
problem_spec: workspace/sde_hhw_call_corr/problem_spec.json
single_family_pool: true
plans:
  1-euler-maruyama:
    one-sentence: Correlated 3-factor Euler-Maruyama (full-truncation V, log-Euler S, Euler R), dt=0.004, 1e6 paths, discounted-asset control variate.
    iter: 0
    score: 0
    provenance: null
    est_err: null
    state: await_solver
---
