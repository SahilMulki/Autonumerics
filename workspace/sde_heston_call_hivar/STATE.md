phase: done
problem_type: sde
path: A
problem_spec: workspace/sde_heston_call_hivar/problem_spec.json
ambiguities:
  - R30: 'problem.md says "this problem has no terminal-moment formula, so the quantity of interest below is the whole grade"; formulator kept the three prices as the external grade but added exact terminal moments (validated) as pipeline evidence.'
plans:
  1-full-truncation-euler:
    one-sentence: Full-truncation Euler-Maruyama, log-Euler S, one path set prices all spots.
    iter: 2
    score: 9
    certified: false
    provenance: analytic
    est_err: 2.528e-03
    state: stopped  # plateaued; refine left solver.py unchanged (same hash, score 9 is Dynkin-unavailable ceiling)
  2-andersen-qe:
    one-sentence: Andersen Quadratic-Exponential scheme for V with martingale-corrected S.
    iter: 2
    score: 9
    certified: false
    provenance: analytic
    est_err: 2.4e-03
    state: stopped  # plateaued; refine left solver.py unchanged, 9 is Dynkin-unavailable ceiling
  3-truncated-milstein:
    one-sentence: Milstein on V alone with positivity truncation, log-Euler S.
    iter: 2
    score: 1
    certified: false
    provenance_new: none
    provenance: analytic
    agent_cap: "4 (kernel 8): price_S100 -0.503% vs 0.4% tol; V atom at 0 (25.8%)"
    state: stopped  # refine regressed: solver timed out >300s (score 1)
best_plan: 1-full-truncation-euler
best_provenance: analytic
