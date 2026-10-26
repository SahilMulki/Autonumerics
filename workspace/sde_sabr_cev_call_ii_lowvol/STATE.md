phase: done
problem_type: sde
path: B
problem_spec: workspace/sde_sabr_cev_call_ii_lowvol/problem_spec.json
plans:
  1-euler-maruyama-absorbing:
    one-sentence: Euler-Maruyama on S absorbed at 0, exact log-step for sigma, put-call-parity pricing.
    scheme_family: euler-maruyama
    iter: 2
    score: 2
    certified: false
    provenance: none
    est_err: 2.22e-14
    state: stopped
  2-qe-martingale-absorbing:
    one-sentence: Andersen QE martingale-preserving step for S with absorption, exact sigma, parity pricing.
    scheme_family: other
    iter: 2
    score: 2
    certified: false
    provenance: none
    est_err: 2.34e-14
    state: stopped
best_plan: 1-euler-maruyama-absorbing
best_provenance: none
