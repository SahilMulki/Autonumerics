---
phase: done
problem_type: sde
path: A
problem_spec: workspace/sde_kubo_oscillator/problem_spec.json
ambiguities:
  - R18: 'problem.md gates a pathwise relative strong error < 20%, but evaluation_thresholds
      has no pathwise-strong-error key (the SDE accuracy keys are moment-space and
      functionals takes only solver-reported scalars). Formulator carried it as the exact
      pathwise reference (exact_pathwise_solution), the strong-order floors 0.5/1.0 measured
      by the CRN ladder, and the moment tolerances; the 20% number is recorded but not
      enforced as its own numeric gate.'
plans:
  1-stratonovich-implicit-midpoint:
    one-sentence: Stratonovich implicit midpoint (Cayley transform of the step angle theta_n = a*dt + sigma*dW_n); strong order 1, exactly amplitude-preserving.
    iter: 2
    score: 3
    certified: false
    provenance: analytic
    est_err: 1.07e-02
    bias_retry: true
    bias_moment: mean (moment_bias_frac_tol 0.2587 at iter 1; 2.84e-03 after)
    bias_retry_outcome: 'accuracy improved (worst moment error 2.34e-02 -> 1.07e-02, mean bias
      0.2587 -> 0.0028 of tolerance) but the score REGRESSED 9 -> 3: with the initial condition
      now readable the Dynkin check reached a verdict and FAILED (seed-42 z 2.31 vs bar 2.53,
      3-seed median 3.20, pooled 5.72). The failing test function is X1^2+X2^2, whose
      extrapolated generator residual is ~1e-11 +/- 3.5e-12 -- roundoff scale, against the
      order study own roundoff floor of 6.6e-13. The three signal-carrying functions sit
      inside noise (z -1.52, +0.55, -1.65). Evaluator judged it a probable kernel artifact
      and applied no cap; the 3 is what the kernel computed and is recorded as such.'
    state: stopped
  2-stratonovich-heun:
    one-sentence: Explicit derivative-free Stratonovich Heun predictor-corrector; strong order 1, independent error structure from plan 1 (amplitude drifts up ~0.75*dt per unit time).
    iter: 2
    score: 9
    certified: false
    provenance: analytic
    est_err: 1.02e-02
    bias_retry: true
    bias_moment: mean (moment_bias_frac_tol 0.3768 at iter 1; 1.84e-03 after)
    bias_retry_outcome: 'accepted -- Heun corrector replaced by a 4-stage explicit
      derivative-free Stratonovich RK; worst moment error 6.29e-02 -> 1.02e-02, max||X|-1|
      0.155 -> 0.0077. Score held at 9: the ceiling is the rubric 9 row (dynkin_ok never
      reached a verdict), not accuracy.'
    stale_family_label: 'scheme_family: milstein and the scheme: title line both name Heun/Milstein
      while the code is a 4-stage derivative-free Stratonovich RK. Documentation finding only --
      milstein_eligible is true in the spec, so no milstein_claim cap and no score cost.'
    state: stopped
  3-euler-maruyama:
    one-sentence: Euler-Maruyama on the Ito form including the -0.5*sigma^2*X Stratonovich-to-Ito correction; strong order 0.5 baseline.
    iter: 2
    score: 3
    certified: false
    provenance: analytic
    est_err: 6.04e-02
    note: 'gated amplitude_conservation violated (certification ceiling 3): 7.016% of paths
      breach ||X|-1| > 0.25 at 50k paths, 6.35-7.05% over four seeds at the graded 2000 paths.
      Solver deliberately declined the Milstein rescue (documented Implementation Note 9):
      with one noise channel, constant Dg and J^2 = -I it reduces to I + theta*J + 0.5*theta^2*J^2,
      a near-duplicate of plan 2, and would remove the pool only order-0.5 anchor. solver.py
      unchanged across both cycles (sha af56c6c3). No clipping/renormalisation/projection.'
    state: stopped
pool_findings: none (cli.py gate reported no pool warnings)
spec_repair: 'Round 1 left dynkin_ok without a verdict on every plan -- verifylib/kernel/sde/dynkin.py:203
  reads a TOP-LEVEL initial_condition and params X_0/Y_0/Z_0, while the spec carried
  parameters.X1_0/X2_0 and verification.initial_condition, so the kernel set
  random_initial_condition: true and the analytic rubric 9 row bound the pool. Formulator
  re-dispatched once (its single allowed re-dispatch); it added a top-level
  initial_condition {law: deterministic, value: [1.0, 0.0]} as a restatement, amended R3
  spec_path, and verified initial_moments() now returns x0 [1.0, 0.0] with
  random_initial_condition false. check-spec still exits 0; ledger still 18 mapped /
  1 ambiguous / 0 dropped.'
best_plan: 2-stratonovich-heun
best_provenance: analytic
best_score: 9
best_certified: false
decided_by: score
final_gate: 'ok: no findings (exit 0)'
phase_2b: 'did not run -- fires only on a plan that left the loop certified: true, and none did'
report: workspace/sde_kubo_oscillator/REPORT.md
---
