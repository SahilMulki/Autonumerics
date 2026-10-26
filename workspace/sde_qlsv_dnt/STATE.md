phase: done
problem_type: sde
path: B
single_family_pool: true
problem_spec: workspace/sde_qlsv_dnt/problem_spec.json
ambiguities:
  - R36: 'problem.md: "They must differ ... and they must agree to within the sampling noise the two counts imply." No numeric bar given; formulator used |p_400k - p_100k| <= 2*sqrt(se400^2+se100^2).'
plans:
  1-euler-maruyama-bessel:
    one-sentence: Full-truncation EM for V, exact Bessel-3 step for R=1/(c S), Brownian-bridge barrier weights.
    iter: 2
    score: 7
    certified: false
    provenance: self_convergence
    est_err: 1.7e-02
    state: stopped
  2-euler-maruyama-drift-implicit:
    one-sentence: Full-truncation EM for V, drift-implicit positive-root step for R=1/(c S) using supplied dW, Brownian-bridge barrier weights.
    iter: 2
    score: 7
    certified: false
    provenance: self_convergence
    est_err: 1.7e-02
    state: stopped
best_plan: 2-euler-maruyama-drift-implicit
best_provenance: self_convergence
