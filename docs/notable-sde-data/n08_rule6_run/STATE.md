---
phase: done
problem_type: sde
path: B
problem_spec: workspace/sde_three_halves_vol/problem_spec.json
best_plan: 1-lamperti-strang-split
best_provenance: surrogate
decided_by: sqi
ambiguities:
  - R14: 'problem.md requires "exactly" `def solve_sde(num_paths, dt, T, seed=42) -> dict`;
      the formulator kept those four parameters with exact names/semantics and added the
      project contract''s two optional kwargs (`dW=None`, `observables=None`) that the kernel
      needs for the CRN strong-order study and the Dynkin check. A call with only the stated
      arguments reproduces the stated behaviour exactly.'
spec_gate: 'check-spec exit 0; final gate exit 0. Two warnings, both folded into REPORT.md:
  reference check unavailable (no analytic moments claimed, expected on Path B), and a state_clip
  note on plan 1''s declared reflecting boundary X = abs(X).'
pool_findings: 'cli.py gate on the plan pool emitted no pool warnings; three distinct scheme families.
  terminal_law is null by deliberate formulator choice (the exact law is a reciprocal scaled
  noncentral chi-square, outside the registry), so the KS shape test never ran; both problem.md and
  the spec forbid drawing from the exact transition, so no exact-transition plan was requested.'
agreement_gate: 'Final (re-run on the shipped solvers): all three pairs INCONCLUSIVE (exit 3) in
  moment space on V, where the MC interval straddles the tolerance. Graded-functional separations:
  0.81 sigma (1 vs 2), 3.42 sigma (1 vs 3, above the 3-sigma bar so no break earned), 2.38 sigma
  (2 vs 3). No cap in force. An earlier DISAGREE (exit 2) on the round-1 solvers capped plans 1 and 3
  at 8; plan 3''s Heun drift broke that tie and the cap was lifted. No second circularity break
  earned -- an inconclusive comparison is not an agreement.'
single_family_pool: false
certification_ceiling: 'spec-level. verification.stationary_density is declared at T_stat = 6 but the
  graded horizon is T = 0.1, about half the reciprocal process''s relaxation time of 0.2, so the
  kernel''s moment comparison for V cannot decide: moments_ok / variance_ok unavailable and
  resolved: false on every plan, which holds the whole pool at 9 uncertified. Not a solver defect;
  reported independently by all three evaluators. The graded reciprocal functionals are clean against
  a twice-validated exact reciprocal-CIR reference.'
plans:
  1-lamperti-strang-split:
    one-sentence: Lamperti transform X = V^-1/2 (additive noise) advanced by a Strang composition with both sub-flows exact, reflection at X=0, antithetic pairs plus a Brownian-aggregate control variate.
    scheme_family: other
    iter: 3
    score: 9
    certified: false
    provenance: surrogate
    est_err: 0.7516
    functional_rel_error: 2.478e-03
    functional_rel_tol: 0.1000
    resolved: false
    functional_limited_by: sampling
    unresolved_retry: true
    contract_finding: 'Round 2 pooled the functionals over 8 antithetic blocks, computing them from 8x
      the declared path budget rather than from the returned terminal_paths. Flagged by the evaluator
      and reverted in round 3, replaced by a no-extra-path control variate the evaluator verified
      unbiased over 200 seeds. The shipped solver honours the contract.'
    state: stopped
    stop_reason: stopped unresolved
  2-split-step-milstein-v:
    one-sentence: Strang splitting in V with exact logistic drift half-flows plus the exact additive-noise flow in Z = V^-1/2; was a Milstein noise sub-step scoring 3 in round 1.
    scheme_family: strang-splitting
    scheme_family_was: milstein
    iter: 3
    score: 9
    certified: false
    provenance: surrogate
    est_err: 0.6990
    functional_rel_error: 8.660e-03
    functional_rel_tol: 0.1000
    resolved: false
    functional_limited_by: sampling
    bias_retry: true
    bias_retry_moment: mean
    bias_retry_outcome: 'accepted: both gated martingale constraints had failed in round 1 (score 3);
      replacing the Milstein noise sub-step with its exact flow repaired both and took the score to 9.
      Thinnest martingale headroom in the pool (worst drift 1.360 against a bar of 1.930, 0.15 decades).'
    unresolved_retry: true
    state: stopped
    stop_reason: stopped unresolved
  3-truncated-milstein-log:
    one-sentence: Explicit Mao-truncated Milstein in L = log V with a Heun drift sub-step (was explicit Euler) and antithetic pairing; the independent-coordinate cross-check.
    scheme_family: tamed-milstein
    iter: 3
    score: 9
    certified: false
    provenance: surrogate
    est_err: 2.2648
    functional_rel_error: 1.21e-02
    functional_rel_tol: 0.1000
    resolved: false
    functional_limited_by: sampling
    unresolved_retry: true
    observed_order: 0.7999
    design_order: 1.0000
    note: 'The only plan reading below its design rate (0.80 +/- 0.13 against 1.0) and the only one
      whose bias term is a material share of its graded error (2.282e-03 of 1.21e-02). Its evaluator
      measured, over 300 seeds at two path counts, a real +0.364% dt bias on E[1/V] and -0.90% on
      Var[1/V]. Still carries moment_bias_actionable: true (mean, 0.2616 of tolerance) -- a live
      unspent repair on the pool''s third-ranked plan.'
    state: stopped
    stop_reason: stopped unresolved
---
