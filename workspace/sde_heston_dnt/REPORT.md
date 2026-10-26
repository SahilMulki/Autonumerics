# Report: sde_heston_dnt (SDE, Heston, double no-touch probability)

## Ambiguities
- **R16** — problem.md: "this problem has no terminal-moment formula, so the quantity of interest below is the whole grade". The formulator kept the no-touch functional (S0=75/100/125, rel_tol 0.6%) as the external criterion, but derived and validated exact terminal moments (E[S]=100, E[V]=0.048380, Var S=606.655, Var V=0.0102127) and used them as internal Tier-A evidence. The internal pipeline therefore scores something problem.md says does not exist.

## Provenance
Winner scored against `analytic` (validated closed-form moments, path A). Note that the kernel scores moments, not the no-touch probability that is the actual external grade.

## Plans
| Plan | Scheme | Score | Cert | Provenance | est. rel err (measured) | Strong order (expected 0.3) | Iters | Stop |
|---|---|---|---|---|---|---|---|---|
| 1-euler-maruyama | EM, full-truncation V, log-Euler S, Brownian-bridge barrier weights, dt=0.002, 400k paths | 9 | no | analytic | 2.45e-3 | 0.282 | 2 | plateaued |
| 2-euler-maruyama-cir-milstein-v | Plan 1 + scalar CIR Milstein correction on V | 9 | no | analytic | 1.42e-3 | 0.243 | 2 | plateaued |

No-touch probabilities vs the formulator's 2-D Kolmogorov reference (0.83454 / 0.89983 / 0.66840, tol 0.6%; checked by the solver/evaluator, not graded by the kernel):
- Plan 1 (400k): 0.83510 / 0.89962 / 0.66715 (errors 0.07% / 0.02% / 0.19%)
- Plan 2 (400k): 0.83513 / 0.89998 / 0.66688 (errors 0.07% / 0.02% / 0.23%)
- 100k second runs agree within MC noise for both.

## Recommendation
**1-euler-maruyama.** Both plans tie on score (9), provenance and accuracy headroom; the ranker (`sqi` 0.717 vs 0.559) separates them on cost/structure. Plan 2's V-only Milstein correction adds nothing measurable.

## Why nothing reached 10 (not certified)
The Dynkin generator check came back `unavailable` for both plans: its finest ladder level needs a 12.8 GB increment array (400000 paths x 2000 steps x 2 channels) against the kernel's 12 GB budget. This follows from the spec's path count, dt and noise dimension, not from the solvers; the rubric caps an unresolved Dynkin at 9. Both plans stopped after the second cycle with unchanged solvers and identical scores (plateau).

## Verification gaps
- Dynkin check not run (above); nothing independently establishes the paths come from this SDE beyond moment agreement.
- The no-touch functional (the external grade) is not scored by the SDE kernel; only the evaluators' own comparison to the formulator's PDE reference supports it. The PDE reference was computed by the formulator, not independently validated.
- `variance_nonnegative` listed in `invariants_not_reported` (non-gating) for both plans.
- Weak order indeterminate (noise floor) for both plans.
- Strong order measured 0.24-0.28 vs 0.3, inside the +/-0.4 band; reduced rate is attributed to the Feller violation.
- No terminal_law declared; cross-plan agreement not available (Path A, gate not applicable; both plans are Euler-family).
- `verifylib gate`: ok, no findings.
