---
phase: done
problem_type: sde
path: A
problem_spec: workspace/sde_clark_cameron/problem_spec.json
ambiguities:
  - R5: 'problem.md states only that "the last axis indexes the 2 independent noise sources" and never says which column of dW is W1 and which is W2; formulator encoded dW[:, :, 0] = W1, dW[:, :, 1] = W2 (naming order in the equations). A swap is invisible to every moment/invariant/Dynkin check (W1, W2 are i.i.d.) but gives ~100% pathwise strong error against the graded reference. All three solvers report the self-check S_T == S_0 + mu*T + sum(dW[:,:,1]) passing to ~2e-15.'
  - R26: 'problem.md states only the 2% pathwise strong-error gate and no moment tolerances; formulator supplied the template defaults (10% variance, 5% mean) against the exact moments, and set num_paths: 8000 rather than the graded 1000 (1000 paths cannot resolve a 10% variance tolerance on U; bar ~5511). graded_num_paths: 1000 carries the contract count.'
spec_defects:
  - 'dynkin unavailable on all three plans: verification.initial_condition.value is the strings ["S_0", "U_0"], which the kernel reader cannot turn into numbers, so E[phi(X_0)] cannot be formed. Under the SDE analytic rubric a dynkin_ok with no verdict caps at 9 and leaves the plan uncertified, so no plan in this pool can certify until the spec carries numeric initial values. Formulator re-dispatched once with this finding.'
  - 'strong_rel_err is null on all three plans: the spec declares evaluation_thresholds.strong_rel_err_max = 0.02 (the problem''s only stated accuracy criterion, R22) and the kernel measured no pathwise strong error against a sub-grid reference, so the declared gate is unverified and caps the score at 9. Reported by all three evaluators independently.'
ceiling: 9
ceiling_reason: 'No plan on this problem can certify, and that is the designed outcome rather than a pool failure. The problem.md accuracy criterion (R22/R7) is a 2% pathwise relative strong error, carried correctly by evaluation_thresholds.strong_rel_err_max = 0.02 -- verified by verifylib/schema.py:check_strong_gate_carried, which hard-errors if that quote is not carried. But nothing in verifylib/kernel/ ever writes summary["strong_rel_err"]: it has exactly one reader (score.py:820) and one pass-through (sde/driver.py:826) and no producer, because the pathwise reference belongs to the independent external grader, which this pipeline must not read. So strong_rel_err is None unconditionally and score.py applies strong_gate_not_measured, capping every plan at 9. The kernel comments say this is deliberate: "the plan has not been shown wrong, it has not been shown right."'
pool_findings:
  - 'gate warning [verification.generator]: state component 1 (U) is conditionally tractable and no plan integrates it out (none declares exact_sampling_components: 1). Plan-creator re-dispatched once; it declined to add a plan and recorded why. Carried to REPORT.md as a verification gap: conditional Monte Carlo on U is unavailable here because requirement R23 (with R22/R7) grades terminal_paths[:, 1] path by path against a reference on the same W1 increments. E[U_T | W2] = 0 for every path, so a Rao-Blackwellised U leg scores exactly 100% relative strong error; conditioning on only the W2 sub-path plus the supplied W1 step increments discards the sub-increment information that resolves the Levy area and falls back to O(sqrt(dt)) (Clark-Cameron lower bound, ~4.5x over the 2% gate); conditioning on the full supplied information reproduces the sub-grid reference to rounding and is dt-invariant, which R23 excludes rather than scores. No admissible middle. The variance the technique would remove is sampling variance on the moment tolerances (the R26-flagged secondary criterion, already resolved at 8000 paths), not the pathwise error the problem is graded on.'
plans:
  1-levy-area-milstein-drift-corrected:
    one-sentence: Exact S update plus U from the left-point Ito sub-increment Levy-area sum I_21 and the increment-measurable conditional mean of the drift-noise integral 0.5*dt*dW1; scheme_family milstein, levy_area_treatment sub-increments, expected strong order 1.0.
    iter: 3
    score: 9
    certified: false
    provenance: analytic
    est_err: 1.21e-02
    observed_order: 0.9343
    observed_order_se: 0.0351
    design_order: 1.0
    scheme_family: milstein
    resolved: true
    mc_se_rel: 0.0208
    solve_time_s: 1.383
    dynkin_ok: true
    dynkin_z: 2.0342
    dynkin_bar: 2.4284
    dynkin_median_z: 1.8957
    dynkin_seeds: 3
    dynkin_replicated: true
    state: stopped
    stopped_reason: plateau
    solver_sha256: 7a10d2583d0bd5c10c6d312731f7640f98b34496325a3e6609e5159d368d34d0
    history: '5 (iter 1, pinned by the zero-power martingale invariant) -> 9 (iter 2, after the invariant was removed from the spec). solver.py unchanged across both cycles: the solver judged the scheme correct and recorded why rather than making a speculative edit.'
    note: 'invariants_ok now true (estimator_stability 3.192e-03 against a 1e-2 bar). Order 0.9343 +/- 0.0351 read against its own design rate of 1.0, ~1.9 se under, inside the band. No moment_bias_actionable. Uncertified only by the two ceiling caps, neither of which is a defect in this plan.'
  2-levy-area-milstein:
    one-sentence: The spec's canonical order-1 recipe U_{k+1} = U_k + S_k dW1_k + I_21 with I_21 from the 256 sub-increments; independent cross-check on plan 1 at the same declared order.
    iter: 3
    score: 9
    certified: false
    provenance: analytic
    est_err: 1.21e-02
    observed_order: 0.9827
    observed_order_se: 0.0307
    design_order: 1.0
    scheme_family: milstein
    resolved: true
    mc_se_rel: 0.0209
    solve_time_s: 1.461
    dynkin_ok: true
    dynkin_z: 2.0342
    dynkin_bar: 2.4284
    moment_bias_binding: variance
    moment_bias_frac_tol: 0.2131
    bias_retry: true
    state: stopped
    stopped_reason: plateau
    solver_sha256: 88edd7c60c28cce05fede7e3994fb9d3d5d7fd34c55e71f786b76062f34e792c
    history: '5 (iter 1, pinned by the zero-power martingale invariant) -> 9 (iter 2, after the invariant was removed). solver.py unchanged across both cycles.'
    note: 'Given the bias instruction on the variance of U (0.2131 of tolerance). The solver answered "the current sub-step is the floor" and the evaluator independently re-derived and confirmed it: the bias is the left-point weight on S_k dW1_k (the left Riemann sum of E[S^2] undershoots by 0.08, the dt^2/2 term adds back 0.02, net 0.06 = 2.1% of 17/6, matching the ladder), NOT the dt^3/3 residual the notes had blamed. The only bias-removing terms are the conditional-mean projection 0.5*mu*dt*dW1 -- which is plan 1s scheme, so this plan would become a copy of it and the pool would lose its independent cross-check -- or the full sub-increment assembly, which R23 excludes for dt-invariance. No third option found by either agent.'
  3-euler-maruyama-control:
    one-sentence: Increment-only Euler-Maruyama, expected strong order 0.5, flagged by the plan-creator as expected to miss the 2% pathwise gate (spec-measured 8.9e-02) while its moments stay clean -- the Clark-Cameron lower bound as a measurement.
    iter: 2
    score: 5
    certified: false
    provenance: analytic
    est_err: 2.00e-02
    observed_order: 0.4965
    observed_order_se: 0.0348
    design_order: 0.5
    scheme_family: euler-maruyama
    resolved: true
    mc_se_rel: 0.0207
    solve_time_s: 1.048
    moment_bias_binding: variance
    moment_bias_frac_tol: 0.3874
    bias_retry: true
    kernel_score: 9
    agent_cap: 5
    agent_cap_reason: 'Pool control, declared EM at strong order 0.5. The problem states a 2% pathwise strong-error gate; the kernel does not measure it, but its own CRN ladder shows the dt=0.04 vs dt=0.02 pathwise difference in U is 0.0788 against terminal scale 1.246 (6.3%, 3x the gate) and the formulator measured 8.9e-2 (4.5x). Order 0.4965 clearing its own floor of 0.5 is conformance to a low design rate, not accuracy. Moment-only 9 would overclaim.'
    state: stopped
    stopped_reason: plateau
    solver_sha256: 78fc58a63d9546682ef70d4fbc9af71b8ca09ab29ba70decac36c230cd4960d1
    history: '5 (iter 1, martingale) -> 5 (iter 2, kernel 9 capped to 5 by the evaluator). iter >= 2 with no improvement, so stopped by the plateau rule. Answered the bias instruction with "the increment-only U sub-step is the floor": any reduction needs the Levy-area term, which would turn the control into a duplicate of plan 2.'
leakage_redaction: 'The Stop-gate guardrail flagged error-severity leakage in plans 2 and 3 SOLUTION.md: both named the independent external grader by path in evaluator prose. The citation originated in the conductor dispatch prompts, not in the agents own analysis. One sentence in each file was redacted to remove the path while preserving every claim; no measurement, score or metrics value was touched. Re-checked clean. Later dispatches carried an explicit instruction not to name those paths.'
compare_top_two: 'INCONCLUSIVE (exit 3), recorded in agreements.json: 2-levy-area-milstein vs 1-levy-area-milstein-drift-corrected differ by 1.142e-02 on the shared seed against a worst tolerance of 0.05, and by 2.000e+00 in moment space on independent seeds, at num_paths=8000, dt=0.04, seeds 42 / 3444837047, families milstein / milstein. The Monte-Carlo interval straddles the tolerance, so this is neither a circularity break nor a cap. Run for information only -- this is Path A, so the agreement gate is not required, and both plans are the same scheme family in any case.'
ranker_decided_by: sqi
ranker_note: 'The restricted index was computed over order and structure only; accuracy and consistency did not resolve on every plan in the pool, and ranking.comparable_keys drops a component no pool-wide comparison was made on. The two 9s were therefore separated on order fidelity alone, 0.983 against 0.934 -- a gap of 0.048 between estimates whose own standard errors are 0.031 and 0.035 (combined ~0.047), i.e. about one standard error. This is not a resolved separation and the report says so.'
best_plan: 2-levy-area-milstein
best_provenance: analytic
best_score: 9
best_certified: false
---
