---
phase: running
problem_type: sde
path: B
problem_spec: workspace/sde_three_halves_vol/problem_spec.json
ambiguities:
  - 'R29: problem.md mandates `def solve_sde(num_paths, dt, T, seed=42) -> dict` "exactly"; the
      formulator added the pipeline contract''s optional `dW` and `observables` keyword arguments
      (defaulting to None), since without them the CRN order ladder and the Dynkin generator check
      cannot run. Every mandated call signature stays valid and behaviour is unchanged.'
pool_findings: 'none — cli.py gate reported no pool warnings. The plan-creator deliberately shipped
  no exact-transition plan even though 1/V is exactly CIR with a samplable non-central chi-squared
  transition, because problem.md (ledger R25) forbids drawing from it; all three plans step at the
  handed-in dt. No barrier/absorbing boundary is declared and V=0 is inaccessible (Feller index of
  1/V is 5.2468), so no boundary_treatment was required. No duplicate-scheme pair.'
certification_ceiling: 'No plan on this problem can certify. There is no moment-space reference
  (moment_ode, kolmogorov and stationary_density were all declined with structural reasons), so
  D1 against the validated operator is the only circularity break available and the evidence path
  caps at 9 with provenance manufactured_partial. The loop therefore cannot exit on
  certified: true and runs to plateau or max_iter.'
unresolved_note: 'Every plan reports resolved: false and an mc_se_rel 6x-45x larger than its own
  estimated_rel_error. That statistic is the variance of V against no reference, and V has tail
  index 2.623, so Var[V] has infinite variance and is unresolvable at any path count — this is why
  the problem grades reciprocal moments instead. It is a skip, not an MC-inconclusive verdict, and
  raising num_paths cannot change it. The graded functionals are separately resolved and inside
  their gates.'
agreement_gate: 'Run at iter 2, re-run at iter 3, and re-run on all three pairs at iter 4 (a
  record is ignored once either solver hash changes). At iter 4, with the pool spanning three
  distinct families: 1 vs 3 DISAGREE (worst inv_var_mean 7.415e-03 relative = 7.95 sigma);
  2 vs 3 DISAGREE (inv_var_mean 1.233e-02 = 4.60 sigma); 1 vs 2 INCONCLUSIVE (exit 3, worst
  inv_var_variance 3.396e-02 = 3.01 sigma, a hair above the 3-sigma bar — neither a break nor a
  cap). Plan 3 is the outlier, disagreeing with both siblings, which matches it carrying about
  +0.7% mean bias (inv_var_mean 11.704) while plan 1 reads 11.6226 against the exact 11.6228 and
  plan 2 reads 11.5585. Note plan 3''s relative gap to plan 1 SHRANK this round (1.213e-02 ->
  7.415e-03) while its sigma ROSE (3.32 -> 7.95), because its new antithetic estimator cut its own
  standard error about 4x and so sharpened the disagreement. Per the disagreement rule all three
  plans carry agreement_cap: 8, so no second circularity break was earned anywhere in the pool and
  the winner is selected on quality rather than on cross-plan corroboration. The headline
  moment-space figures (~0.76) are dominated by Var[V], which is unresolvable here and is not
  evidence.'
plans:
  1-log-coordinate-tamed-milstein:
    one-sentence: Strang splitting in the reciprocal chart Y=1/V (half drift flow, full noise step, half drift flow, both flows in closed form), plus an antithetic pair estimator for the graded functionals only.
    iter: 4
    score: 9
    agreement_cap: 8
    certified: false
    provenance: manufactured_partial
    est_err: 5.360e-04
    scheme_family: strang-splitting
    state: stopped
    stop_reason: 'plateaued at the evidence-path ceiling (9 -> 9 across a full refine cycle), after
      the resolved:false retry had been acted on once. Not "stopped unresolved" in the sense the
      rule warns about: its graded functional IS resolved and measured, and the resolved: false
      flag is the kernel reporting "no reference variance to resolve against" for Var[V].'
    unresolved_retry: true
    unresolved_retry_outcome: 'functional_limited_by was sampling, so the instruction was to cut the
      sampling term at a fixed path count. The solver added an antithetic pair estimator feeding the
      functionals only (terminal_paths and path_integrals still come from the supplied dW, so the
      CRN ladder and Dynkin check see unchanged data). It worked: functional_rel_error 8.660e-03 ->
      6.994e-03, and the inv_var_mean standard error fell about 4x (0.0150 -> 0.0039). Still
      resolved: false afterwards, for the structural reason above.'
    note: 'The pool''s most accurate plan and the best-ranked. dt bias is essentially eliminated:
      functional_bias_rel 2.971e-07 against functional_se_rel 6.994e-03, so the graded
      inv_var_variance error of 6.994e-03 against a 0.10 gate is pure sampling, about 14x inside
      the tolerance. estimated_rel_error 5.360e-04. At 200k paths inv_var_mean reads 11.6226
      against the exact 11.6228. Gated reciprocal checks pass with 0.3277 decades headroom. Strong
      order 0.886 +/- 0.098 vs floor 0.5. Dynkin median z 1.84 / bar 2.43 — thin, and the pooled z
      of 2.60 disagrees with the median verdict. Its weak-order-2 claim is UNEVIDENCED (all three
      weak tests sat at the noise floor) and was softened to "expected" in the notes. The directory
      slug still names the retired log-chart scheme; it is a label only.'
  2-balanced-implicit-v-coordinate:
    one-sentence: Drift-implicit Euler on x=ln V with the stiff restoring term solved by Newton; strictly positive with no clipping. The round-3 Milstein correction was added, regressed the plan, and was reverted.
    iter: 4
    score: 9
    agreement_cap: 8
    certified: false
    provenance: manufactured_partial
    est_err: 1.844e-03
    scheme_family: implicit
    state: await_solver
    note: 'Recovered 3 -> 9 by reverting exactly to the round-2 file (hash 38359e8d... restored
      byte-for-byte from solver.r2.py). The round-3 Milstein correction had raised its strong order
      0.458 -> 0.865 while enlarging the dt bias on the gated reciprocal moments enough to fail
      both martingale checks (drift 4.09 vs bar 2.84) — the order check is not gated on this
      problem and those invariants are, so the order gain cost 6 points and bought nothing. Now
      passes all gates with the pool''s widest headroom (0.3681 decades, drift 1.26 vs bar 2.95).
      functional_rel_error 0.0133 (bias 4.463e-03 + sampling 8.846e-03) — the pool''s largest, and
      the only plan that did not add an antithetic estimator. Strong order 0.458 +/- 0.029 vs floor
      0.5, passes. Dynkin median z 1.96 / bar 2.43, pooled z 2.70 disagrees.'
    unresolved_retry: true
    unresolved_retry_outcome: 'functional_limited_by both; the solver spent the cycle on the revert
      and declined variance reduction, arguing antithetic pairing of the supplied dW would break
      the shared-path pairing the order study relies on and that a control variate would need the
      exact reference values it must not use. Plans 1 and 3 did add antithetic estimators without
      that problem, by feeding the mirror run to the functionals only.'
  3-log-coordinate-tamed-euler:
    one-sentence: Log-chart Milstein with second-order taming on both drift and noise, plus the correction 0.25*eps^2*V_n*(dW^2-dt), and an antithetic pair estimator for the graded functionals.
    iter: 4
    score: 9
    agreement_cap: 8
    certified: false
    provenance: manufactured_partial
    est_err: 1.658e-03
    scheme_family: tamed-milstein
    state: stopped
    stop_reason: 'plateaued at the evidence-path ceiling (9 -> 9 across a full refine cycle) after
      the resolved:false retry had been acted on once. Its block now reports resolved: unavailable
      — the honest skip vocabulary for "no reference variance to resolve against" — rather than
      false.'
    unresolved_retry: true
    unresolved_retry_outcome: 'Given both halves. The bias half FAILED and was reverted: a
      weak-order-2 drift term 0.5*La*dt^2 blew up E[V] on rare large-V paths and moved the graded
      variance the wrong way, so the drift sub-step stays at second-order taming and
      moment_bias_frac_tol stays 0.3146 (still actionable, binding moment: mean). The sampling half
      succeeded: an antithetic pair estimator cut functional_se_rel 8.602e-03 -> 6.968e-03 and the
      inv_var_mean standard error about 4x.'
    note: 'The pool''s outlier. It carries about +0.7% bias on inv_var_mean (11.704 at 200k against
      the exact 11.6228) and disagrees with both siblings. Its gated checks nonetheless pass with
      the pool''s widest headroom on paper (0.4074 decades) because terminal_paths — what the
      invariant check reads — still come from the plain non-antithetic run with its larger standard
      error. functional_rel_error 0.0101 (bias 3.179e-03 + sampling 6.968e-03). order_check is
      UNRESOLVED: strong order 1.13 +/- 0.20, whose 2-sigma interval is wider than the band it
      would be judged against — a property of the estimator that costs the plan nothing. Dynkin
      median z 1.94 / bar 2.43.'
---
