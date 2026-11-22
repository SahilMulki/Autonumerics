# REPORT — sde_kubo_oscillator

**Problem type:** SDE (Kubo oscillator — Stratonovich rotation with multiplicative noise; amplitude |X| = 1 is conserved). Path A (closed-form moments declared).

**Outcome:** no plan certified. All three plans finished at score **6**, `certified: false`, and were **stopped unresolved** after the mandatory unresolved-measurement retry. The default best plan is `1-cayley-midpoint`.

## Provenance of the scoring

All three plans were scored against the spec's **`analytic`** moments (`reference_outcome: validated` — the closed form was checked against the spec's own moment ODE). This is the strongest tier; the 6 is not a provenance limit but the rubric's MC-inconclusive row.

## Ambiguities recorded by the formulator

- **R7** — "The harness checks that the sample you return has conserved it; a scheme that reads the coefficients as Ito coefficients solves a different equation, whose amplitude grows." Tolerance unstated; formulator encoded a gated constraint: mean amplitude deviation over paths < 0.25.
- **R8** — "This problem is graded **path by path**, not in distribution." No `evaluation_thresholds` key carries a path-by-path reference; carried via exact moments, a CRN strong-order study, and the recorded `exact_pathwise_solution`.
- **R9** — gates the relative strong error `sqrt(mean((X_num - X_exact)^2)) / sqrt(mean(X_exact^2))` over paths at < 20%. No kernel key exists for a path-by-path error tolerance, so it is not a kernel threshold.

## Why nothing certified: the measurement is unresolved at the graded path count

Every plan's `<metrics>` block records `resolved: false`. The ledger (R12, R14) fixes the graded call at `solve_sde(num_paths=2000, dt=0.04, T=1, seed=42)` with harness-supplied `dW` of shape (2000, 25). At 2000 paths the variance confidence-interval relative width (2 × SE) is ~0.05–0.07 per component against the kernel's 0.05 resolution bar, so the kernel cannot resolve its own variance statistic. The solvers were dispatched once with the "raise `num_paths`" instruction (no functional is declared, so there is no bias/sampling split); all three reported, correctly, that the path count is fixed from outside and that antithetic/control-variate pairing would break the pathwise grading. The re-evaluation returned `resolved: false` again, so each plan is **stopped unresolved** — not plateaued. The solvers' own scans put the bar at roughly ≥ 4000–4500 paths; this is a spec-level mismatch between the declared sample size and the resolution bar, not a solver defect.

**Never read the error estimates below as measurements**: on every plan `mc_se_rel` (~0.032–0.034) is comparable to or larger than the recorded error.

## Plans

| plan | scheme | family | score | cert | provenance | est. rel. error (variance, worst comp.) | mean rel. err | mc_se_rel | observed strong order (floor) | iter | state |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-cayley-midpoint | Stochastic implicit midpoint (Cayley) on the Stratonovich form; explicit rotation by 2·arctan((a·dt+σ·dW)/2); preserves \|X\|=1 to round-off | implicit | 6 | no | analytic | 0.0340 (unresolved) | 0.0262 | 0.0343 | 0.959 ± 0.034 (0.5) | 2 | stopped unresolved |
| 2-stochastic-heun | Stochastic Heun predictor-corrector on the Stratonovich form, no Ito correction | other | 6 | no | analytic | 0.0934 (unresolved) | 0.0368 | 0.0340 | 1.001 ± 0.033 (0.5) | 2 | stopped unresolved |
| 3-milstein | Milstein on the Ito form with drift −½σ²X and correction ½(−σ²X)(dW²−dt) | milstein | 6 | no | analytic | 0.1502 (unresolved; fails the 10% bar, 2-SE interval ≈ 7.7%–22%) | 0.0170 | 0.0318 | 0.968 ± 0.020 (1.0) | 2 | stopped unresolved |

Key metrics common to all three: `invariants_ok: true`, `constraints_ok: true`, amplitude drift 0.0 (conserved to round-off), `order_check: true`, weak order ≈ 1.0–1.2, `functionals_ok: unavailable` (no gated functional declared), `cross_check_ok: unavailable`, Dynkin `not_run`. No `agent_cap` was applied on any plan. Solver hashes were verified unchanged across every evaluation.

Evaluator notes worth carrying: plan 3's weak-order study attributes ~8% of the exact Y variance to dt-bias at dt = 0.04, so even with zero sampling noise the Milstein plan would sit only 1–2 points inside the 10% bar; the Stratonovich schemes (plans 1 and 2) carry far less bias (variance errors 0.019/0.034 and 0.077/0.093).

## Best plan

`1-cayley-midpoint`, as a **default rather than a measured ranking**. The ranker printed:

```
best: 1-cayley-midpoint   decided_by: tie
note: the index above is over order, structure only. accuracy, consistency did not resolve on every plan in this pool, and a component no comparison was made on does not decide one -- see ranking.comparable_keys. Each plan's own index, over whatever it measured, is 1-cayley-midpoint=1.000, 2-stochastic-heun=0.640, 3-milstein=0.553.
warning: the winner was not chosen on quality -- score, provenance and the index are all tied and `tie` broke it. Spend one measurement on the pool rather than reading this as a ranking.
```

Score, provenance and the comparable index are tied across the pool. Plan 1's accuracy component returned no value because its error (0.0340) sits inside its own error bar (0.0343). As the extra measurement the ranker asked for, `cli.py compare` was run on all three pairs at the graded grid (2000 paths, dt 0.04): every pair came back **INCONCLUSIVE** (exit 3), differing by 5.7e-2 / 7.4e-2 / 7.0e-2 in moment space on independent seeds against a worst tolerance of 0.05 — the Monte-Carlo interval straddles the tolerance, so this is neither an agreement break nor a cap. Results are in `agreements.json`. The pool is indistinguishable on everything the kernel can measure at the spec's path count. The reasons to prefer plan 1 are qualitative: smallest variance errors, exact amplitude conservation by construction, and the Stratonovich-native scheme; none of this is a certified claim.

## Verification gaps

- **Unresolved measurement on every plan** (`resolved: false`). Accuracy at 2000 paths was not established for any plan; the scores are the rubric's MC-inconclusive 6, not a quality grade.
- **Spec/path-count mismatch**: `graded_num_paths = 2000` with `ci_mult = 2` and a 0.05 CI bar cannot be resolved on component Y by any correct scheme (needs ~4000+ paths). Only a spec change can lift the ceiling.
- **Pairwise agreement gate inconclusive** on all three pairs (see above); no circularity break earned from cross-plan agreement, and no disagreement cap applied.
- **Path-by-path grading (R8/R9)** is not a kernel threshold; the pipeline evidence is in moment space plus the CRN strong-order study. The strong error against `exact_pathwise_solution` was not gated by the kernel.
- **Dynkin identity not run** on any plan (kernel default). All three evaluators flagged that the kernel's printed skip reason ("a declared constraint gate failed") is wrong for these runs, since no gate failed — a kernel message bug, not a finding about the plans.
- **Final gate (`cli.py gate`, exit 0)** — one warning: plan 1's review's Numerical Accuracy section states `variance_rel_err` 0.0194 (component 0) where the `<metrics>` block records the worst component, 0.0340. Transcription drift between prose and block, not a disagreement about the result.
- **Evaluator concurrency**: two evaluators reported that a sibling plan's concurrent kernel run clobbered a shared `/tmp` JSON output on first attempt; both discarded and re-ran until the solver hash matched. Final `evaluation.json` files carry the correct hashes.
- `invariants_not_reported`: none listed in any metrics block.
