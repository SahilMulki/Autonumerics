---
phase: done
problem_type: sde
path: A
problem_spec: workspace/sde_merton_jump_call/problem_spec.json
ambiguities:
  - 'R18: problem.md states "this problem has no terminal-moment formula, so the
      quantity of interest below is the whole grade"; the formulator found the moment
      hierarchy does close exactly for the compensated jump diffusion and set
      has_analytic_solution: true, deriving E[X_T] = X_0*exp(r*T) and
      Var[X_T] = X_0^2*(exp((2*(r-lam*kappa)+sigma^2+lam*eta2)*T) - exp(2*r*T)).
      The substance of the sentence is preserved: call_price remains the gated
      functional at 1%.'
pool_findings: 'cli.py gate reported no pool findings on the three plans.'
verification_gaps:
  - 'dynkin_test_functions is null by construction: the kernel builds a diffusion-only
      generator, which has no home for the jump term lam*E[phi(xY)-phi(x)]. On phi = x
      the missing term is lam*kappa*x = -0.055x against a declared drift coefficient
      of +0.105, so a diffusion-only Dynkin residual would fire on every correct
      solver. Declaring no test functions turns a false failure into a skip, which
      caps this problem at score 9 under the project manual 9 row. No plan can lift
      that ceiling.'
plans:
  1-jump-adapted-milstein-control-variate:
    one-sentence: Milstein on the continuous leg with jump times inserted as sub-step boundaries (strong order 1.0), plus the jump-free GBM call as an analytic control variate on the graded call_price.
    iter: 1
    score: 9
    certified: false
    provenance: analytic
    est_err: 2.289e-03
    state: stopped
  2-jump-adapted-exponential-euler:
    one-sentence: Exact exponential log-coordinate flow per sub-step with jumps at their exact times, antithetic Brownian pairs for sampling error.
    iter: 1
    score: 3
    certified: false
    provenance: analytic
    est_err: 0.0124
    state: stopped
  3-euler-maruyama-grid-jumps:
    one-sentence: Plain Euler-Maruyama with jumps applied at grid points, no variance reduction; the discretised baseline with a cleanly measurable strong order of 0.5.
    iter: 1
    score: 3
    certified: false
    provenance: analytic
    est_err: 1.762e-03
    state: stopped
dynkin_finding: 'All three plans report dynkin_z ~ 72.07 against a bar of 2.43 with a
  dt-independent residual of -1.369 on phi = X, which matches the missing jump term
  lam*kappa*E[X]*T = -1.39 to within 2%. verifylib/kernel/sde/dynkin.py:66
  test_functions() falls through to DEFAULT_TEST_FUNCTIONS when
  verification.dynkin_test_functions is null, so the spec cannot express the skip its
  dynkin_note assumes, and verification.generator (_vector_payload) carries only drift
  and diffusion_matrix, so there is no spec route to a jump-aware generator either.
  The check therefore fires identically on every correct solver for this problem:
  plans 2 and 3 are scored 3 by the rubric Dynkin-fails row and plan 1 is scored 9 on
  dynkin unresolved. The conductor did not edit the spec or cap anything; this is
  recorded as a verification gap bounding the whole run.'
stopped_reason: 'Round 2 dispatched a solver for all three plans; all three returned
  the file unchanged (solver.py hash identical to the solver_sha256 in its own metrics
  block) on the ground that the evaluator reviews named no solver defect and the
  binding finding is the diffusion-only Dynkin generator, which no solver change can
  reach without falsifying the compensated drift a = r - lam*kappa. Re-grading
  identical bytes against an unchanged spec cannot produce new evidence, so each plan
  is terminal at iter 1 rather than cycled to max_iter. No plan certified, so Phase 2b
  (the winner bias-removal cycle) did not fire.'
best_plan: 1-jump-adapted-milstein-control-variate
best_provenance: analytic
gate_findings: 'cli.py gate exit 0 with one warning: state_clip at plan 1 solver.py:88.
  Examined: the clipped variable u is the within-step fractional jump time used by the
  Brownian-bridge split, not the state; X is never clipped. Recorded in REPORT.md as a
  false positive rather than dismissed.'
---
