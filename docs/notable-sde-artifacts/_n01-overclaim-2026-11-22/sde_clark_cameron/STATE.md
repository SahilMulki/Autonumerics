---
phase: done
problem_type: sde
path: A
problem_spec: workspace/sde_clark_cameron/problem_spec.json
ambiguities:
  - 'R7: "This problem is graded **path by path**, not in distribution: the harness
      compares your terminal states to a reference solution driven by the very same
      Brownian path." -- no evaluation_thresholds key carries a pathwise strong-error
      tolerance against an external fine-grid reference; the formulator carried it as
      rel_l2_err_max = 0.02 with the pathwise definition stated in implementation_notes,
      relying on the CRN strong-order ladder to discriminate the same two scheme families.'
  - 'R18: "the harness gates the relative strong error
      `sqrt(mean((X_num - X_exact)^2)) / sqrt(mean(X_exact^2))` over paths at < 2%" --
      same gap; 0.02 recorded verbatim at the right magnitude under a metric the kernel
      can evaluate (rel_l2_err_max), discriminated by R9''s declared strong-order floors.'
pool_findings:
  - 'gate: state component 1 (U) is conditionally tractable and no plan integrates it
      out (none declares exact_sampling_components: 1). Rao-Blackwellising U would
      remove that leg''s sampling variance entirely. Re-dispatched the plan-creator
      once for a plan that conditions on the driving components; it declined, with reason:
      conditional MC on U is structurally available but the gate here is pathwise against a
      shared Brownian path, so a conditionally-exact draw of U(T) misses that path''s own
      realisation by ~141% relative; and declaring exact_sampling_components: 1 would abstain
      the strong-order ladder on the only component carrying the Levy area, which is the
      pool''s sole discriminator. Recorded as the pool''s verification ceiling.'
  - 'All three plans scored 9 / uncertified on iter 1 with dynkin_ok: unavailable -- the kernel
      could not form E[phi(X_0)] because verification.initial_condition.value carries the
      strings ["S_0","U_0"] rather than numbers. Spec defect, not a solver defect; formulator
      re-dispatched. Its first repair put the numeric value under
      verification.initial_condition; verified against verifylib/kernel/sde/dynkin.py:203
      (initial_moments reads spec.get("initial_condition") at the TOP level), the key was
      moved to the top level on a second dispatch. All three plans then certified at 10.
      No solver changed across any of these re-evaluations (hashes identical throughout),
      so the re-evaluations are not refine cycles and were not counted as iters.'
  - 'rank note: accuracy did not resolve on plan 1 -- its estimated_rel_error 5.667e-03 sits
      below its own mc_se_rel 8.323e-03 -- so ranking.comparable_keys dropped the accuracy
      component from the pool comparison, leaving order/structure/consistency. The restricted
      index then picked 3-euler-maruyama-control (sqi 0.918) over 1-levy-area-milstein-midpoint-drift
      (0.892) on the residual-slope term. Per-plan unrestricted indices run the other way:
      0.796 / 0.892 / 0.837. See REPORT.md -- this is the sde_sciml_stiffquad_mild failure
      mode, and order fidelity is measured against each plan''s own design order, so a scheme
      designed at order 0.5 scores full marks for being exactly as slow as designed.'
plans:
  1-levy-area-milstein-midpoint-drift:
    one-sentence: Levy-area Milstein (sub-increment I_21) with exact S leg and midpoint drift in the U noise term; expected strong order 1.0.
    iter: 2
    score: 10
    certified: true
    provenance: analytic
    est_err: 5.667e-03
    state: winner
  2-levy-area-milstein:
    one-sentence: Levy-area Milstein implementing the spec's milstein_correction literally (frozen integrand plus I_21); expected strong order 1.0.
    iter: 2
    score: 10
    certified: true
    provenance: analytic
    est_err: 0.0261
    state: winner
  3-euler-maruyama-control:
    one-sentence: Plain multi-dimensional Euler-Maruyama on dW alone; strong order 0.5, expected to miss the 2% pathwise gate (control plan).
    iter: 2
    score: 10
    certified: true
    provenance: analytic
    est_err: 0.0337
    state: winner
    polish_cycle: true
    polish_pre_score: 10
    polish_moment_bias_binding: variance
    polish_moment_bias_frac_tol: 0.2880
    polish_reverted: true
    polish_revert_reason: 'moment_bias_frac_tol not strictly below the pre-cycle 0.2880 --
      the solver declined to change the sub-step and solver.py is byte-identical
      (e79300b8...), so no evaluator pass was spent re-measuring an unchanged file and
      nothing needed restoring. Reason given: the bias is the U update''s left-endpoint
      quadrature, not the omitted Levy area; the removable share is exactly plan 1''s
      midpoint drift term, and adopting it would stop this plan being the plain
      Euler-Maruyama control the pool is built around.'
best_plan: 3-euler-maruyama-control
best_provenance: analytic
best_plan_note: 'ranker verdict, decided_by: sqi, over a comparison restricted to
  order/structure/consistency -- accuracy was dropped because it did not resolve on plan 1
  (est_rel_err 5.667e-03 < mc_se_rel 8.323e-03). Unrestricted per-plan indices run the other
  way: 0.796 (plan 3) / 0.892 (plan 1) / 0.837 (plan 2). On the evidence measured,
  1-levy-area-milstein-midpoint-drift is the plan to ship: lowest moment error, strong order
  0.9932 +/- 0.0110 against the Levy-area floor of 1.0, and the lowest design-estimated
  pathwise error (~0.7%) against the problem''s stated 2% gate, which plan 3 is expected to
  fail at ~8.4%. See REPORT.md "Best plan".'
---
