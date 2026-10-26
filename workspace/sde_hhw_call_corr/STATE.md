phase: done
problem_type: sde
path: B
problem_spec: workspace/sde_hhw_call_corr/problem_spec.json
plans:
  1-euler-maruyama-full-truncation:
    one-sentence: Euler-Maruyama, full-truncation V, log-S step, left-point rate shared by S drift and discount.
    iter: 2
    score: 7
    certified: false
    provenance: self_convergence
    est_err: 8.9e-07
    state: stopped
  2-qe-martingale-corrected:
    one-sentence: Andersen QE for V, martingale-corrected log-S, exact R transition, trapezoid discount.
    iter: 2
    score: 7
    certified: false
    provenance: self_convergence
    est_err: 8.7e-06
    state: stopped
best_plan: 1-euler-maruyama-full-truncation
best_provenance: self_convergence
single_family_pool: true
