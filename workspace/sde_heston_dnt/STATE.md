phase: done
problem_type: sde
path: A
problem_spec: workspace/sde_heston_dnt/problem_spec.json
ambiguities:
  - R16: 'problem.md says "this problem has no terminal-moment formula, so the quantity of interest below is the whole grade"; formulator kept the no-touch functional as the external criterion but derived and validated exact moments (Tier-A) as internal supporting evidence'
plans:
  1-euler-maruyama:
    one-sentence: Euler-Maruyama, full-truncation V, log-Euler S, Brownian-bridge barrier weights, dt=0.002, 400k paths.
    iter: 2
    score: 9
    certified: false
    provenance: analytic
    est_err: see SOLUTION.md
    state: stopped
  2-euler-maruyama-cir-milstein-v:
    one-sentence: Same as plan 1 plus scalar CIR Milstein correction on V only.
    iter: 2
    score: 9
    certified: false
    provenance: analytic
    est_err: 1.422e-03
    state: stopped
best_plan: 1-euler-maruyama
best_provenance: analytic
