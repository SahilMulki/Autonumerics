---
phase: done
problem_type: sde
path: A
problem_spec: workspace/sde_geometric_asian/problem_spec.json
ambiguities:
  - R21: 'problem.md says "this problem has no terminal-moment formula, so the quantity
      of interest below is the whole grade"; the formulator read this as a statement about
      what the independent check grades, not a mathematical claim, and declared exact GBM
      terminal moments (validated by an exactly-closing moment ODE) alongside the two
      gated functionals.'
pool_findings: 'cli.py gate: no findings over the plan pool, before or after the run.'
spec_repairs:
  - 'Round 1 held the whole pool at 9: dynkin_ok reached no verdict because the kernel''s
     Dynkin reader found no initial state (the spec declared parameters.S_0 only).
     Formulator re-dispatched once with the finding; it added a top-level initial_condition
     block with value: ["S_0"] (binding to the existing parameter, no second source of
     truth) and extended ledger entry R2''s spec_path. No declared value moved, so the
     round-1 runs stayed valid. check-spec exits 0. In round 2 Dynkin ran and passed
     (z = 0.33 against a bar of 2.43), turning two 9s into two certified 10s.'
plans:
  1-log-exact-bridge-conditioned:
    one-sentence: Log-exact per-step stepping of ln S consuming the supplied dW, with the
      within-step Brownian-bridge contribution to the log-average integrated out in closed
      form (zero quadrature bias at any dt) plus antithetic pairing.
    iter: 2
    score: 10
    certified: true
    provenance: analytic
    est_err: 5.206e-04
    functional_rel_error: 1.537e-03
    functional_rel_tol: 0.0100
    state: winner
    note: 'scheme_family relabelled exact-transition -> other in cycle 2 (the step consumes
      the supplied dW, so the old label named the wrong family); numerics unchanged. Order
      study abstains (order_check: unresolved) because the pathwise differences sit at
      roundoff, 2.6e-13 against a 1.05e-10 floor. Ranker warns its state-variance statistic
      is 5x inside its own MC standard error, so that number is unlocated; the graded
      functional is resolved and is what the accuracy claim rests on.'
  2-milstein-trapezoid:
    one-sentence: Scalar Milstein on S itself (strong order 1.0) with O(dt^2) trapezoidal
      quadrature of the log-average, plus a terminal-price control variate added in cycle 2.
    iter: 2
    score: 10
    certified: true
    provenance: analytic
    est_err: 5.325e-03
    functional_rel_error: 1.666e-03
    functional_rel_tol: 0.0100
    state: certified
    note: 'Cycle-2 control variate on the terminal price cut functional_se_rel from 2.198e-03
      to 1.332e-03 with no added bias (the control''s mean under the Milstein step is exact).
      Carries the pool''s only resolved order measurement for a stepping scheme at design
      rate: 0.9882 +/- 0.0078 against a design 1.0.'
  3-euler-maruyama-trapezoid:
    one-sentence: Plain Euler-Maruyama (strong order 0.5) with the same trapezoidal
      quadrature; the baseline that isolates the Euler weak bias against plan 2.
    iter: 2
    score: 8
    certified: false
    provenance: analytic
    est_err: 6.130e-03
    functional_rel_error: 2.222e-03
    functional_rel_tol: 0.0100
    state: stopped
    note: 'Plateaued at iter 2 (score 8 -> 8), not stopped-unresolved: the measurement
      resolved (resolved: true) and the defect is real. Terminal-law KS fails at D = 2.08e-3
      against a 1.45e-3 critical value (1.44x critical) from the Euler dt bias, which more
      paths cannot reach; moments and both gated functionals are clean. Dynkin not_run
      because the moments cap fired first. Cycle 2 deliberately changed nothing: both named
      fixes would have made it a copy of plan 2 or plan 1, so it documents its own ceiling.'
polish_cycle: false
polish_note: 'Phase 2b did not fire -- both certified plans carry
  moment_bias_actionable: false (moment_bias_frac_tol 4.5e-13 on the winner, 0.0418 on
  plan 2, both binding on the variance), so there was no measured removable bias.'
ranker: 'decided_by: sqi -- 0.432 vs 0.423 over accuracy and structure only; order and
  consistency did not resolve across the whole pool and so were excluded from the
  comparison. A narrow margin, reported as such in REPORT.md.'
grading_violations: none
best_plan: 1-log-exact-bridge-conditioned
best_provenance: analytic
