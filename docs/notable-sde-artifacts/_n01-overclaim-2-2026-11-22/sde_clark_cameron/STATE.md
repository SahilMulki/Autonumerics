---
phase: done
problem_type: sde
path: A
problem_spec: workspace/sde_clark_cameron/problem_spec.json
ambiguities:
  - 'R12: problem.md never states which index of the last axis is W1 vs W2 ("a numpy
      array of shape (1000, 25, 2) holding the Brownian increments, already scaled by
      sqrt(dt), whose last axis indexes the 2 independent noise sources"); the
      formulator encoded channel 0 = W1 (drives U), channel 1 = W2 (drives S).'
  - 'R21: the pathwise strong-error gate ("the harness gates the relative strong error
      sqrt(mean((X_num - X_exact)^2)) / sqrt(mean(X_exact^2)) over paths at < 2%") has
      no SDE-side key in the spec vocabulary; carried instead as the two strong-order
      floors (0.5 dW-only / 1.0 with Levy area) plus the 2% figure verbatim in
      implementation_notes.'
spec_corrections:
  - 'iter 1 capped all three evaluated plans at 9 / uncertified: dynkin_ok returned
      unavailable because verification.initial_condition was nested where nothing reads it
      (the kernel calls spec.get("initial_condition") at the top level). Re-dispatched the
      formulator once with the finding; it moved the field to the top level as explicit
      floats [1.0, 0.0], mathematically identical. check-spec still exits 0, no plan
      directory was modified, and the Dynkin check now reaches a verdict and passes
      (dynkin_z 2.23 vs bar 2.53, replicated over 3 seeds; binding test function S*U, the
      probe for the R12 channel-ordering ambiguity). Plans re-evaluated on unchanged solvers.'
pool_findings:
  - 'gate warning [verification.generator]: state component 1 (U) is conditionally
      tractable and no plan integrates it out (none declares exact_sampling_components: 1).
      Re-dispatched the plan-creator once; it added 4-conditional-mc-projection, which
      Rao-Blackwellises U as far as the pathwise contract permits (replaces U''s step
      increment by its exact conditional expectation given the resolved S/W2 sub-path),
      but deliberately did NOT declare exact_sampling_components: full integration-out of
      component 1 is incompatible with this contract -- E[U(T)|sigma(W2)] = 0, so returning
      it pathwise gives ~100% strong error against the supplied-path reference, and drawing
      U(T) from its conditional law needs fresh randomness, violating R13. The warning
      therefore persists: the pool ceiling is that only U''s within-step W1 fine structure
      can be integrated out, not the component.'
plans:
  1-levy-area-milstein-trapezoid:
    one-sentence: Order-1 Levy-area Milstein -- exact step for S, U advanced with S_n*dW1_n plus the iterated integral I_{1,2} assembled exactly from dW_sub, plus a trapezoidal mu*(dt/2)*dW1_n surrogate for I_{0,1}.
    iter: 2
    score: 10
    provenance: analytic
    est_err: 1.14e-02
    state: winner
    certified: true
    dynkin: pass (z 2.20-2.23 vs bar 2.531, replicated over 3 seeds; binding test function X0*X1)
    note: winner (certified); not bias-actionable (metrics block carries no moment_bias_* field)
  2-levy-area-milstein:
    one-sentence: Canonical Levy-area Milstein exactly as the spec's milstein_correction states it -- I_{1,2} only, no further correction.
    iter: 2
    score: 10
    provenance: analytic
    est_err: 9.80e-03
    state: winner
    certified: true
    dynkin: pass (z 2.20-2.23 vs bar 2.531, replicated over 3 seeds; binding test function X0*X1)
    note: 'winner (certified); moment_bias_actionable: true, binding variance at 0.2131 of tolerance'
  3-euler-maruyama-control:
    one-sentence: Euler-Maruyama on the step increments only, dW_sub accepted and ignored; the order-0.5 baseline that measures the Clark-Cameron lower bound.
    iter: 2
    score: 10
    provenance: analytic
    est_err: 1.62e-02
    state: winner
    certified: true
    dynkin: pass (z 2.20-2.23 vs bar 2.531, replicated over 3 seeds; binding test function X0*X1)
    note: 'winner (certified); ranker pick; moment_bias_actionable: true, binding variance at 0.3874 of tolerance'
    polish_cycle: true
    polish_pre_score: 10
    polish_pre_moment_bias_binding: variance
    polish_pre_moment_bias_frac_tol: 0.3874
    polish_reverted: true
    polish_revert_reason: 'moment_bias_frac_tol not strictly below the pre-cycle 0.3874 --
      the solver deliberately changed nothing and named U''s increment as the floor for the
      dW-only scheme class (a higher-order U update needs the iterated integral, which lives
      only in dW_sub; consuming it would duplicate plan 2, destroy the sqrt(dt) baseline and
      risk R22 exclusion). solver.py is byte-identical to the certified artifact (sha256
      9ecba458...), so no restore was needed.'

  4-conditional-mc-projection:
    one-sentence: Rao-Blackwellised dW1-only scheme -- U's increment replaced by its exact conditional expectation given the resolved S/W2 sub-path; L2-optimal in its class, so it measures the Clark-Cameron bound.
    iter: 0
    score: 0
    provenance: null
    est_err: null
    state: unfinished (never solved -- three plans certified before it was dispatched)
best_plan: 3-euler-maruyama-control
best_provenance: analytic
best_plan_note: 'the ranker returns this plan with decided_by: sqi, and all three plans tie at
  score 10 / certified / analytic. The index that separated them excluded accuracy -- it did not
  resolve on any plan (every plan''s moment error sits below its own MC error bar), so
  comparable_keys restricted the comparison to order, structure and consistency, and order
  fidelity is scored against each plan''s own design rate (0.5 for this control, 1.0 for the two
  Milstein plans). The control therefore won the deciding component while converging at half its
  rivals'' rate. R21''s pathwise 2% gate -- the quantity the problem is actually scored on -- has
  no spec key and was never measured; on the plans'' own derivations this plan is expected to miss
  it and plans 1 and 2 to clear it. REPORT.md recommends 2-levy-area-milstein or
  1-levy-area-milstein-trapezoid for any use outside the ranking machinery.'
---
