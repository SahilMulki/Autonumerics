# The SDE Half Is Under-Verified — Audit and Implementation Plan

**Status:** **rev 5, 2026-10-15 — implemented, then independently evaluated and corrected.** All
eleven steps of [§7](#7-revised-shipping-order-rev-2) landed; the log is
[§12](#12-implementation-log--2026-10-15). A separate session then graded the shipped work against
[the companion](expectations-sde-verification-gaps.md) and found every §F guard holding and **nine
places the work fell short** — three of them mis-priced grades, two too high and one too low. [§13](#13-second-round--from-an-independent-evaluation-2026-10-15) is that round:
[REV 58]–[REV 71], **+47 tests**, suite **690 passed / 0 failed** of 697 collected, and **one live
score moved** — the one plan sitting on a repriced row. The field-level before/after is
[§L of the companion](expectations-sde-verification-gaps.md#l-measured-after-implementation), whose
own field-move count §L reported as 35 and which re-derives to **61**; round 2's deltas are in
[§M](expectations-sde-verification-gaps.md#m-the-second-round-measured).

~~**rev 4, 2026-10-15 — implemented.** All eleven steps of
[§7](#7-revised-shipping-order-rev-2) landed; the log is [§12](#12-implementation-log--2026-10-15)
and the field-level before/after is
[§L of the companion](expectations-sde-verification-gaps.md#l-measured-after-implementation). Suite
**643 passed / 0 failed** of 650 collected; **one live score moved and 35 (plan, field) pairs moved
across 9 of 12 plans**. Seven measured revisions, [REV 51]–[REV 57], six correcting this plan and one
([REV 56]) a bug step 8 introduced and closed.~~ ~~**rev 3, 2026-10-14 — ready to implement, step 0 respecified.**~~ Rev 1 audited against the code; all 14 findings upheld, a 15th found, fixes and ordering revised, baseline measured, and all six open design decisions resolved in [§10](#10-decisions--resolved-2026-10-04). Rev 3 re-read the whole plan against the code, found **step 0 under-specified in a way that would false-*fail* correct solvers**, added ten revisions, and resolved six further decisions in [§11](#11-decisions--resolved-2026-10-14). Nothing removed.
**Scope:** what the kernel does and does not measure about an SDE solver, on both paths
**Related:** [plan-no-closed-form.md](plan-no-closed-form.md),
[plan-hallucination-guardrails.md](plan-hallucination-guardrails.md),
[no-closed-form-findings.md](no-closed-form-findings.md)
**Companion, added 2026-10-14:** [expectations-sde-verification-gaps.md](expectations-sde-verification-gaps.md)
— every claim below re-anchored to the **field value** that carries it, with today's reading from the
12 live `_eval_metrics.json` files, the regression surface, and three corrections to this document
(§K there), one of which changes a row in Phase 1a. See [REV 49] and [REV 50].

Provenance labels: **[M]** = reproduced by running it, with the output quoted; **[R]** = read
from the code and traced by hand, not run; **[D]** = a divergence between a manual and the code
that implements it, where the manual is the thing that is wrong.

**Rev 2, 2026-10-03 — audited against the code, nothing removed.** Every claim in §2 was
re-checked by reading the implementation, and every `[M]` claim was re-run. The audit holds:
all fourteen findings reproduce, and the line citations are accurate. The revisions are almost
entirely in §3 onward — the *fixes* and their *ordering* — plus four factual corrections in §2.

Revision labels, in addition to the provenance labels above:
**[REV n]** = a change to make before implementing, with the reasoning;
**[V]** = re-verified as written in Rev 1, no change. Original text is left standing in every
case so the two readings can be compared.

| # | Touches | Change | Severity |
|---|---|---|---|
| [REV 1](#0-the-one-sentence-version) | §0 | mechanism 3 is narrower than stated; mechanism 1 is not load-bearing | framing |
| [REV 2](#1-why-this-exists) | §1 | 21-of-23 is evidence about the oracle, not about formulated specs | framing |
| [REV 3](#1-why-this-exists) | §1 | Path-A grading *is* tested; what is missing is a defective solver on Path A | framing |
| [REV 4](#2a-finding-1--the-path-a-rubric-certifies-a-solver-that-never-integrates-the-sde-m) | §2a | "every Euler–Maruyama spec declares 0.5" is false | **fact** |
| [REV 5](#2a-finding-1--the-path-a-rubric-certifies-a-solver-that-never-integrates-the-sde-m) | §2a | the cap under 1b comes from `ceiling()`, not the rubric | **blocking** |
| [REV 6](#2b-finding-2--the-crn-order-ladder-crashes-on-every-multi-noise-sde-m) | §2b | `noise_dimension` already exists in the spec surface | **fact** |
| [REV 7](#2b-finding-2--the-crn-order-ladder-crashes-on-every-multi-noise-sde-m) | §2b | four multi-noise problems, not six | **fact** |
| [REV 8](#2b-finding-2--the-crn-order-ladder-crashes-on-every-multi-noise-sde-m) | §2b | `observables_payload` admits prose and crashes rather than skipping | **fact** |
| [REV 9](#2i-finding-9--stability_check-is-unimplemented-in-the-kernel-r) | §2i | the structured `stability_check` lives in the oracle; `:310` contradicts two live specs | **fact** |
| [REV 10](#2j-finding-10--the-clip-check-is-promised-in-two-places-and-implemented-in-none-d) | §2j | §21 is a reviewer instruction; only `solver-sde.md:99` is a false promise | framing |
| [REV 11](#2k-finding-11--the-order-study-reduces-every-system-to-component-0-r) | §2k | a fourth `[:, 0]` site in `dynkin._phi_values` | scope |
| [REV 12](#2n-finding-14--nothing-verifies-the-milstein-eligibility-claim-r) | §2n | `milstein_eligible` means two different things on this set | scope |
| [REV 13](#3-fixes) | §3 | the phase order is inverted; see [§7](#7-revised-shipping-order-rev-2) | **blocking** |
| [REV 14](#phase-1--close-the-path-a-hole-findings-1-6) | 1a | the 7 row should be 8 (Path-B parity); row order unspecified | **blocking** |
| [REV 15](#phase-1--close-the-path-a-hole-findings-1-6) | 1b | it is the entire fix; reason text, cost and sequencing all need changing | **blocking** |
| [REV 16](#phase-1--close-the-path-a-hole-findings-1-6) | 1c | false-fails correct solvers on ~6 problems, measured | **blocking** |
| [REV 17](#phase-1--close-the-path-a-hole-findings-1-6) | 1d | needs the same fallback in `reference.py` or it caps at 9 | **blocking** |
| [REV 18](#phase-1--close-the-path-a-hole-findings-1-6) | 1e | `verification_manual.md` §24 needs the same [D] treatment | correctness |
| [REV 19](#phase-2--make-multi-dimensional-sdes-verifiable-at-all-findings-2-11-14) | 2a | the proposed default is wrong on m13 and on both oscillators | **blocking** |
| [REV 20](#phase-2--make-multi-dimensional-sdes-verifiable-at-all-findings-2-11-14) | 2b | collides with `schema.py`'s leakage model | correctness |
| [REV 21](#phase-2--make-multi-dimensional-sdes-verifiable-at-all-findings-2-11-14) | 2d | read `milstein_eligible` instead of re-deriving it | simplification |
| [REV 22](#phase-3--add-the-two-missing-classes-of-evidence-findings-3-4) | 3a | self-contradictory, and the gate flip is a no-op | **blocking** |
| [REV 23](#phase-3--add-the-two-missing-classes-of-evidence-findings-3-4) | 3b | the KS gate rejects correct solvers, measured | **blocking** |
| [REV 24](#phase-4--honesty-about-monte-carlo-noise-findings-7-8) | Phase 4 | must precede Phase 1b, not follow it | **blocking** |
| [REV 25](#phase-5--the-promised-checks-that-do-not-exist-findings-9-10-12) | 5a | wrong semantics, and the structured data does not exist yet | correctness |
| [REV 26](#phase-5--the-promised-checks-that-do-not-exist-findings-9-10-12) | 5b | false-positives on CIR as specified | correctness |
| [REV 27](#phase-6--spec-coverage-finding-13) | Phase 6 | a prerequisite of 1c, not a follow-up | **blocking** |
| [REV 28](#4a-canaries-to-add) | §4a | three rows in the canary table need restating | correctness |
| [REV 29](#4b-two-structural-additions-worth-more-than-any-single-canary) | §4b | add a spec-field-coverage test; it catches six of the fourteen findings | scope |
| [REV 30](#4d-cost) | §4d | 1b and 1c are the wrong way round on cost | **fact** |
| [REV 31](#5-what-this-does-not-fix) | §5 | two more limitations belong here | scope |
| [REV 32](#6-acceptance-criteria) | §6 | five of the nine criteria are unmet or unmeetable as worded | **blocking** |
| [REV 33](#phase-1--close-the-path-a-hole-findings-1-6) | 1c | amends [REV 16]: 10/12 live plans are `inferred`, so (c) is the item that bites today | **blocking** |
| [REV 34](#phase-1--close-the-path-a-hole-findings-1-6) | 1c, `ranking.py` | the same two-sided penalty exists a third time, as dead code | **blocking** |
| [REV 35](#phase-4--honesty-about-monte-carlo-noise-findings-7-8) | Phase 4a | Finding 8 is live: two plans at z = 3.24 / 3.14, scoring 3 | **blocking** |
| [REV 35a](#phase-4--honesty-about-monte-carlo-noise-findings-7-8) | Phase 4a | **withdraws [REV 35]'s claim**: scaling says real bias, not seed noise | **fact** |
| [REV 36](#phase-4--honesty-about-monte-carlo-noise-findings-7-8) | 1c | the strong order spans 6.8x on one plan and `orders_ok` is `True` every time | **blocking** |
| [REV 35b](#phase-4--honesty-about-monte-carlo-noise-findings-7-8) | Phase 4a | **closes the Dynkin question**: 11 runs at full M, 8/8 fail, 3/3 pass, no crossing | **fact** |
| [REV 36a](#phase-4--honesty-about-monte-carlo-noise-findings-7-8) | 1c | confirmed: 12.4x order spread at M=50000; give `strong_order` an error bar | **blocking** |
| Finding 15 resolution | §2o | **kernel bug in `_widen`**, measured; one spec affected | **blocking** |
| Finding 15 | §2o *(new)* | the Kolmogorov surrogate no longer passes its trust guard | **blocking** |
| §9b | §9 *(new)* | the baseline is measured; **6 of 12 recorded scores are wrong** | **blocking** |
| [§10 D1–D6](#10-decisions--resolved-2026-10-04) | §10 *(new)* | **all six design decisions resolved**; each binding on its phase | — |
| [§7 step 0](#7-revised-shipping-order-rev-2) | §7 | two measured bugs now precede everything: `_widen` and `strong_order`'s error bar | **blocking** |

| [REV 37](#phase-4--honesty-about-monte-carlo-noise-findings-7-8) | §7 step 0, `crn.guards` | **step 0 as specified false-*fails* correct solvers**: a `None` strong order is `orders_ok: False`, not `unavailable` | **blocking** |
| [REV 38](#phase-1--close-the-path-a-hole-findings-1-6) | 1a, D1 | `failed(orders_ok)` is **two** rows in `rubric_sde`, at 8 and at 3; D1 does not say which | **blocking** |
| [REV 39](#2o-finding-15--the-kolmogorov-surrogate-no-longer-passes-its-own-trust-guard-m-rev-2) | §2o, §7 step 0 | the `_widen` fix is offered as two rules and neither is picked | **blocking** |
| [REV 40](#9b-the-6-baseline--measured-2026-10-04) | §9b, §9c, criterion 7 | §24's motivating case closes with a **one-line spec edit**; the per-family floor already works | **fact** |
| [REV 41](#phase-3--add-the-two-missing-classes-of-evidence-findings-3-4) | 3b, D4 | the "free" cheap tier has **no reference to grade against** | **blocking** |
| [REV 42](#phase-4--honesty-about-monte-carlo-noise-findings-7-8) | 4a | no combination rule, and the cost is never multiplied through 1b | **blocking** |
| [REV 43](#phase-1--close-the-path-a-hole-findings-1-6) | 1c, `ranking.py` | [REV 34]'s own ask was never satisfied — no criterion, no step | correctness |
| [REV 44](#phase-1--close-the-path-a-hole-findings-1-6) | 1a | no Path-A row for "the order study could not be run" | scope |
| [REV 45](#9d-two-measurements-this-audit-did-not-take) | §9b, §9d | nothing writes the corrected scores back, and the baseline predates step 0 | correctness |
| [REV 46](#4b-two-structural-additions-worth-more-than-any-single-canary) | §4b | `weak_order_out_of_band` is a seventh declared-and-never-read instance | scope |
| [§11 D7–D12](#11-decisions--resolved-2026-10-14) | §11 *(new)* | **six further decisions resolved**, two of them inside step 0 | — |

| [REV 47](#9b-the-6-baseline--measured-2026-10-04) | §9b, §2o, §11 | **the dating question is closed**: 6 of 7 `STATE.md` files predate both kernel changes | **fact** |

| [REV 48](#9d-two-measurements-this-audit-did-not-take) | §9d, §4d, criterion 9 | **the suite baseline is measured**: 580 passed, 0 failed, 6m08s; 11 tests are 75% of the wall time | **fact** |

| [REV 49](#6-acceptance-criteria) | §6, all criteria | **9 of the 23 criteria are worded as scores where an objective field exists**; a score is a composite and 6 of 12 drifted with no code change | framing |
| [REV 50](#phase-1--close-the-path-a-hole-findings-1-6) | 1a, §7 step 0, 2c | **`_martingale` ignores a declared `expr`**: two live Path-A plans fail `invariants_ok` on artifacts and score 10; 1a's invariants row is **not** a transcription | **blocking** |

**Rev 4, 2026-10-15 — measured during implementation.** Seven revisions whose evidence is a run of
the code as shipped, not a reading of it. Six correct the plan; [REV 56] is a bug the plan's own
step 8 introduced and closed.

| # | Touches | Change | Severity |
|---|---|---|---|
| [REV 51](#4a-canaries-to-add) | §4a | the `sde_milstein_post_step` row is wrong: the post-step correction is `O(dt)`, so the rate is unchanged (0.998 vs 0.998) | **fact** |
| [REV 52](#4a-canaries-to-add) | §4a, 3b | the `sde_uniform_increments` row is wrong on additive noise: the CLT closes the window, measured over five `dt` levels | **fact** |
| [REV 53](#5-what-this-does-not-fix) | §5 *(new)* | **both ladders supply their own increments**, so the increment law is structurally invisible to everything but the graded run | scope |
| [REV 54](#d11--the-cheap-tier-ships-report-only-until-a-law-is-declared) | D11, §6 | D11's third bullet is withdrawn: at `ci_mult·se` tier 1 rejects the honest control at 2.2–2.7 σ, and no looser bar catches anything | **fact** |
| [REV 55](#7-revised-shipping-order-rev-2) | §7 steps 2 + 8 | the two steps multiply: m13's Dynkin increments reach **41.6 GB**; guarded at 12 GB rather than silently reducing `num_paths` | **blocking** |
| [REV 56](#7-revised-shipping-order-rev-2) | §7 step 8, §20 | **step 8 capped a correct solver at 3**: Richardson at `assumed_1` cannot resolve an order-0.2 residual; abstain where the rate is below the band and the residual shrinks | **blocking** |
| [REV 57](#what-rev-3-leaves-open) | D8, §11 | **closes Rev 3's last open measurement**: the median order is biased low where the mean works and no more credible where it does not | **fact** |

**Rev 5, 2026-10-15 — from an independent evaluation of the implementation.** A separate session
graded the shipped work against [the companion](expectations-sde-verification-gaps.md) and found
every §F guard holding, two places the work went past the plan, and **nine places it fell short**.
**Fourteen revisions**, [REV 58]–[REV 71]: thirteen answer the evaluation and [REV 71] is a second
instance of [REV 66]'s defect class, found while measuring [REV 64]. Three are holes the
implementation itself opened or left open — a 10 reachable with no generator evidence, a crashed
ladder collecting the credit the plan reserves for an honest abstention, and a dependency floor that
let the suite be red on a numpy the project claims to support. Nothing below is a reinterpretation;
each row is a measurement.

| # | Touches | Change | Severity |
|---|---|---|---|
| [REV 58](#phase-1--close-the-path-a-hole-findings-1-6) | 1a, Finding 6 | the `*_value_at_T` fallback keyed on the expression being **absent**; a present-but-unevaluable one (`scipy.integrate.quad`) raised through to `"unavailable"` and the 9 the branch exists to prevent | **blocking** |
| [REV 59](#d8--abstain-and-add-a-second-statistic-abstention-costs-nothing) | D8b, §4c | **a crashed CRN ladder was collecting D8b's abstention credit**: one `"unavailable"` token meant both "the estimator could not read it" and "a level crashed", and `ceiling` credited all seven skips | **blocking** |
| [REV 60](#phase-1--close-the-path-a-hole-findings-1-6) | 1b, §4c, 2c | both SDE rubrics read a *failing* Dynkin and ignored one that never reached a verdict, so a 10 was reachable on moment evidence alone — Finding 1's hole through the budget, the crash and the abstention doors | **blocking** |
| [REV 61](#4c-contract-round-trip) | §4c, 2b | the vector generator's validation was done **by hand and not in the suite**; five assertions, including a step-size guard one decade wide | gap |
| [REV 62](#phase-2--make-multi-dimensional-sdes-verifiable-at-all-findings-2-11-14) | 2d, D5 | `_milstein_claim` and `stability_ok` both shipped correct, unexercised and **unasserted**; the inverse false positive is the half that matters | gap |
| [REV 63](#6-acceptance-criteria) | criterion 7 | the per-family half of criterion 7 was **prose inside an error message**: a spec admitting a Milstein plan and declaring no `_milstein` floor passed the check | gap |
| [REV 64](expectations-sde-verification-gaps.md#a1-the-fields-that-carry-the-plans-claims) | §A1, D2 | `_headroom` divided by the declared tolerance where `_martingale` judges against `ci_mult·se + tol`, so a **passing** invariant reported −6.75 decades on a ranking anchor | **fact** |
| [REV 65](#4b-two-structural-additions-worth-more-than-any-single-canary) | §4b criterion 11 | "has a consumer" was a text search over file contents, so **a docstring mention counted**; five live fields had nothing else | gap |
| [REV 66](#phase-5--the-promised-checks-that-do-not-exist-findings-9-10-12) | 5a, Finding 9 | Finding 9's shape one field over: prose in an invariant `expr` (`X ~ Normal(...)`) is a permanent `not_reported` with a `SyntaxError`, refused at the chokepoint now | gap |
| [REV 67](expectations-sde-verification-gaps.md#j1-the-suite--2026-10-14-tree-d5bb447) | §J1 | **the declared numpy floor was wrong**: `>=1.26.0` against 15 `np.trapezoid` call sites, so a supported numpy gives 5 failures | **blocking** |
| [REV 68](#phase-4--honesty-about-monte-carlo-noise-findings-7-8) | §19 | `relative_shift` could not be read — 8.5% is a real bias on one plan and noise on another — so the shift is reported in its own paired standard errors; `m13`'s is **33.2 σ**, signal | **fact** |
| [REV 69](expectations-sde-verification-gaps.md#l-measured-after-implementation) | companion §L1, L3, L5 | three after-column readings corrected: the abstention's recorded cause (the near-zero-`s1` rule, not `se_order`), `noise_dimension`'s recorded value (`null`, not `1`), and the field-move count — **61, not 35**, re-derived from §B's own tables | **fact** |
| [REV 70](#12c-deliberate-scope-limits) | §12c | the `_generator_nd` round-trip claim was true of a hand measurement and false of the suite | **fact** |
| [REV 71](#phase-5--the-promised-checks-that-do-not-exist-findings-9-10-12) | 5a, Finding 9 | found while measuring [REV 64]: **three live specs declare a `tol` for a check nothing can evaluate**, including a Cholesky-bug detector on the one 2-noise plan | gap |

---

## 0. The one-sentence version

Path B — no closed form — is the better-verified half of the SDE kernel, and **almost every SDE
problem is Path A**: 21 of the 23 carry ground-truth moments in `benchmark/problems.py`, and on
that path the only thing measured is whether two terminal moments land inside their tolerances. A
solver that never integrates the SDE at all scores **10, certified, `provenance: analytic`**.

That is not a slip in one function. It is the consequence of three decisions that were each
locally reasonable — `project_manual.md`'s analytic SDE rubric lists only the moment thresholds,
`verification_manual.md` §24 skips Dynkin on Path A as redundant, and §24's `run_crn_ladder` gate
requires `expected_strong_order >= 1.0` — and which together leave the Path-A scheme unexamined.

> **[REV 1] — two of the three mechanisms are narrower than this says.**
> *Mechanism 3* does not leave "the Path-A scheme unexamined" in general: `run_crn_ladder` is
> `path == "B" or not passed(moments_ok) or (would_certify and expected_strong >= 1.0)`
> ([`sde/driver.py:153`](../verifylib/kernel/sde/driver.py#L153)), so the ladder already runs on
> any Path-A plan whose moments *miss*, and on the three live specs that declare `1.0`. What it
> exempts is a **certifying** plan on a spec declaring `0.5` — see [REV 4].
> *Mechanism 1* is real but not load-bearing: enabling Dynkin on Path A caps the fabrication
> canary at 3 through `ceiling()`, before `rubric_sde_analytic` is consulted at all — see
> [REV 5]. The honest one-sentence version is **mechanism 2 alone**: Dynkin never runs on Path A,
> and Dynkin is the only check that establishes the paths came from the SDE.

## 1. Why this exists

The no-closed-form work (§0 of [plan-no-closed-form.md](plan-no-closed-form.md)) was explicitly a
Path-B project, and it did its job: CRN increments rebuilt deterministically in the sandbox child,
the measured-rather-than-assumed Richardson order in Dynkin, Bonferroni across test functions, the
three-surrogate ladder with per-route trust guards, and §14's three-way pass/fail/**inconclusive**
outcome threaded through every comparison. Path B is better verified than most research code.

The cost of that focus is that Path A never got the same treatment, and Path A is where the
benchmark actually lives. Of the 23 SDE problems in `benchmark/problems.py`, 21 carry ground-truth
moments and only `sde_quintic_drift_noise` and `sde_fene_blowup` do not; of the seven problems
formulated into `workspace/` so far, two declare `has_analytic_solution: false`. Every finding
below sits in a configuration the existing test suite never enters.

> **[REV 2] — right number, wrong artifact.** Verified: 23 SDE problems, and the two without
> moments are exactly `sde_quintic_drift_noise` and `sde_fene_blowup`. But `benchmark/problems.py`
> carries moments as *callables* (the oracle) and 15 of the 23 have `ground_truth_kind: None`, so
> that count does not establish which **path** a formulated spec takes — `analytic_moments.has_analytic_solution`
> does. The live evidence is **5 of 7 Path A**, with `sde_ginzburg_landau_s6` and
> `sde_double_well_langevin` correctly Path B (both have reference-quality, not closed-form,
> moments). The conclusion holds directionally; cite the live specs for it, because Phase 6 will
> decide what the real ratio becomes.

**The test suite is green.** `verifylib/tests/test_kernel_sde_path_b.py` plus
`test_reference_sde.py`: 48 passed in 5m35s, exit 0, measured 2026-10-02. That is the point rather
than a mitigation. All four SDE canaries (`sde_honest`, `sde_ignores_crn`, `sde_negative_paths`,
`sde_wrong_drift`) are **scalar additive-noise Ornstein–Uhlenbeck**, and every one of them is
exercised through `sde_spec_no_closed_form.json`. Nothing tests Path-A grading, multiplicative
noise, more than one state component, more than one noise channel, or a random initial condition.

> **[REV 3] — Path-A grading is tested; defective solvers on Path A are not.**
> [`test_kernel_canaries.py:118-133`](../verifylib/tests/test_kernel_canaries.py#L118-L133) runs
> `sde_honest.py` through `sde_spec.json` twice: at 60 paths (asserts 6, "not a solver bug") and at
> full paths (asserts 10). So the Path-A *fixture* exists and §4b is an extension of it rather than
> a new thing. The accurate claim is that **no defective solver is ever run on Path A**, which is
> what let Finding 1 survive. The second of those two tests is also the regression guard Phases 1b
> and 1c must keep green — it does stay green with both enabled (measured: order 0.986 against a
> declared 1.0, Dynkin z = 1.06 against a bar of 2.28, score 10).

---

## 2. Findings

Ordered by what they cost, not by how hard they are to fix. The two columns that matter are what
the kernel reports today and what it should report.

### 2a. Finding 1 — the Path-A rubric certifies a solver that never integrates the SDE **[M]**

`rubric_sde_analytic` ([`verifylib/kernel/score.py:263`](../verifylib/kernel/score.py#L263)) reads
exactly five fields: `crashed`, `gate_violation`, `resolved`, `moments_ok`, `variance_ok`. It does
not read `orders_ok`, `dynkin_ok`, or `invariants_ok`.

Measured. A "solver" that draws terminal states straight from the OU law and never touches `dt`,
`dW` or the drift:

```python
def solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None):
    m = mu + (X_0 - mu) * np.exp(-theta * T)
    v = sigma**2 / (2*theta) * (1 - np.exp(-2*theta*T))
    X = m + np.sqrt(v) * np.random.default_rng(seed).standard_normal(num_paths)
    ...
```

against `canaries/kernel/sde_spec.json` with `expected_strong_order` set to `0.5`, which is what
every Euler–Maruyama spec in the set declares:

```
score 10  provenance analytic  certified True
orders_ok: not_run | dynkin_ok: not_run | moments_ok: True | resolved: True
```

Leave `expected_strong_order` at `1.0` and the order study *does* run, returns **`orders_ok:
False`** — and the score is **still 10, certified**:

```
strong_order=1.0 -> 10 analytic True | orders: False | dynkin: not_run
```

> **[REV 4] — the premise is false, and it matters for 1c.** Of the 7 live specs, **three declare
> `1.0`** — `sde_ornstein_uhlenbeck`, `sde_bm_with_drift`, `sde_double_well_langevin` — and all
> three are additive-noise, where Euler–Maruyama genuinely has strong order 1. The four that
> declare `0.5` (`sde_cir_feller_violated`, `sde_gbm_2d_high_corr`, `sde_ginzburg_landau_s6`,
> `sde_multichannel_stiff_m13`) are all genuinely multiplicative, where 0.5 is the right floor.
> `canaries/kernel/sde_spec.json` itself declares `1.0` for the same reason. So the `>= 1.0`
> condition exempts the four multiplicative specs, not "every Euler–Maruyama plan in the set".
> The measurement above is still valid — it was taken with the canary's floor *overridden* to 0.5,
> which is what the 16 stale specs will default to. Say that, because [REV 16] turns on the
> mirror-image failure.

Three separate things have to be true for this to happen, and all three are:

| # | Mechanism | Where |
|---|---|---|
| 1 | The rubric never reads `orders_ok` or `dynkin_ok` | [`score.py:263-277`](../verifylib/kernel/score.py#L263) |
| 2 | `run_dynkin = (path == "B")`, so Dynkin never runs on Path A | [`sde/driver.py:157`](../verifylib/kernel/sde/driver.py#L157) |
| 3 | `run_crn_ladder` needs `expected_strong_order >= 1.0` at certification, so an Euler–Maruyama plan's order is never measured | [`sde/driver.py:153`](../verifylib/kernel/sde/driver.py#L153) |

> **[V]** Reproduced verbatim, both configurations:
> `eso=0.5 -> score=10 prov=analytic certified=True, orders_ok=not_run, dynkin_ok=not_run`;
> `eso=1.0 -> score=10 prov=analytic certified=True, orders_ok=False`.
>
> **[REV 5] — mechanism 2 alone closes this, and the cap lands somewhere the plan does not
> expect.** Measured with `run_dynkin` forced `True` on Path A and nothing else changed:
>
> ```
> fabricated: score=3  certified=False  dynkin_ok=False  dynkin_z=792   bar=2.28
> honest:     score=10 certified=True   dynkin_ok=True   dynkin_z=1.06  order=0.986
> ```
>
> The 3 comes from [`ceiling()`](../verifylib/kernel/score.py#L133) via `gates_clean`, not from a
> rubric row: a Path-A run with Dynkin enabled sets `operator_validated: True` and
> `d1_outcome: "stalled"` in the evidence dict ([`sde/driver.py:218-227`](../verifylib/kernel/sde/driver.py#L218)),
> and `gates_clean` returns `False` for exactly that pair. Two consequences the plan must carry:
> 1. **Phase 1a's `failed(dynkin_ok) -> 3` row is cosmetic.** Keep it for the rubric's
>    self-consistency, but do not describe it as the fix.
> 2. **The binding reason is PDE vocabulary** — *"D1 stalled with a validated operator: the
>    residual does not fall under refinement and the operator it is measured against is known to be
>    right"*. It names neither Dynkin nor the order study, so acceptance criterion 1 fails as
>    worded. 1b needs a matching `ceiling()` reason for `kind == "sde"` evidence.
> 3. Dynkin on Path A is therefore a **hard cap**, not a rubric row. That is a larger behavioural
>    change than "mirroring `rubric_pde_analytic`'s shape" implies, and it is what makes
>    [REV 24]'s sequencing non-optional.

The asymmetry with the rest of the kernel is sharp in both directions:

* `rubric_pde_analytic` ([`score.py:226`](../verifylib/kernel/score.py#L226)) requires `order_ok`
  **and** `invariants_ok` for a 10, and has an explicit row for "accurate on one grid, not
  converging" at 7.
* `rubric_sde` — the Path-**B** table — caps at **8** for precisely the defect Path A certifies:
  *"the CRN order study reads a strong order near zero: the solver does not consume the supplied
  Brownian increments (manual §18), so nothing about its discretization was measured and it cannot
  be certified"* ([`score.py:363`](../verifylib/kernel/score.py#L363)).

So the kernel already contains the correct judgement, written out in prose, on the path where it is
least needed.

**The root cause is a manual, not the code. [D]** `project_manual.md:122-128` gives the analytic
SDE rubric as variance threshold, mean threshold, MC-resolved — and nothing else. `score.py` is
faithful to it. Fixing the code without fixing §122 re-opens the gap the next time someone checks
the implementation against the spec.

### 2b. Finding 2 — the CRN order ladder crashes on every multi-noise SDE **[M]**

`_crn_increments` honours a noise dimension: `shape = (paths, fine) if not crn.get("m") else
(paths, fine, int(crn["m"]))` ([`sandbox.py:182`](../verifylib/kernel/sandbox.py#L182)). But
`ladder_spec`'s `m` parameter ([`crn.py:30`](../verifylib/kernel/sde/crn.py#L30)) is never passed
by either caller ([`sde/driver.py:402`](../verifylib/kernel/sde/driver.py#L402),
[`:455`](../verifylib/kernel/sde/driver.py#L455)), and **no spec field declares the noise
dimension at all**. The plumbing exists on both ends and nothing connects it.

> **[REV 6] — the field already exists; it is unread, not undeclared.**
> [`sde_multichannel_stiff_m13/problem_spec.json`](../workspace/sde_multichannel_stiff_m13/problem_spec.json)
> declares `"noise_dimension": 13`, and
> [`sde_gbm_2d_high_corr/problem_spec.json`](../workspace/sde_gbm_2d_high_corr/problem_spec.json)
> declares `"noise_dimension": null` — the only two occurrences in the repo, and
> `grep -rn noise_dimension verifylib/ benchmark/ references/ agents/` returns nothing. So the
> name is already chosen and the `null` is itself a formulator gap (that spec's solver indexes
> `dW[:, n, 0]` and `dW[:, n, 1]`, so it is 2). Restate the finding as a *third* unread field,
> alongside `cross_moment` and `*_value_at_T`, which is the pattern [REV 29] generalises.
> Phase 2a becomes "read and validate the field that exists", not "add one".

Measured, against the real plan in `workspace/sde_gbm_2d_high_corr/plans/1-euler-maruyama/`:

```
{'seed': 42, 'num_paths': 2000, 'Nt_fine': 200, 'T': 1.0, 'aggregate': 4, 'm': None, 'dt': 0.02}
crashed exception IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed
```

The solver is correct and is following its own contract — `solver-sde.md` specifies
`(num_paths, Nt, m)` for `m` noise sources, and the solver indexes `dW[:, n, 0]` and `dW[:, n, 1]`.

Two consequences compound:

* `orders_ok` is `"unavailable"` for all six multi-dimensional problems (`sde_gbm_2d_correlated`,
  `sde_gbm_2d_high_corr`, `sde_stochastic_oscillator`, `sde_oscillator_long_horizon`,
  `sde_log_heston_feller_violated`, `sde_multichannel_stiff_m13`).

> **[REV 7] — four, not six: this conflates state dimension with noise dimension.** Checked
> against each `problem.md`:
>
> | Problem | State | Noise | Affected by Finding 2 |
> |---|---|---|---|
> | `sde_gbm_2d_correlated` | 2 | 2 | yes |
> | `sde_gbm_2d_high_corr` | 2 | 2 | yes |
> | `sde_log_heston_feller_violated` | 2 | 2 | yes |
> | `sde_multichannel_stiff_m13` | 2 | **13** | yes |
> | `sde_stochastic_oscillator` | 2 | **1** | **no** |
> | `sde_oscillator_long_horizon` | 2 | **1** | **no** |
>
> Both oscillators are `dX = Y dt; dY = -X dt + 0.3 dW` — one noise channel, so `m=None` is
> already the correct payload and nothing crashes there. Separately, four of the six have **no
> live `problem_spec.json` at all**, so "`orders_ok` is `unavailable` for all six" is measured on
> `sde_gbm_2d_high_corr` and inferred for the rest. This is what makes acceptance criterion 3
> unmeetable until Phase 6 lands — see [REV 32].
* Dynkin is *also* structurally unavailable on those problems, because `observables_payload`
  returns `None` unless `drift_expression` and `diffusion_expression` are both `str`
  ([`dynkin.py:73-82`](../verifylib/kernel/sde/dynkin.py#L73)), and every multi-D spec declares
  them as **lists**.

> **[REV 8] — the guard is weaker than this, and it crashes instead of skipping.**
> `observables_payload` tests `isinstance(drift, str)`, which admits **prose**.
> `sde_multichannel_stiff_m13`'s `drift_expression` is
> `"X @ F.T  (row-major paths, X shape (num_paths, 2), F = [[F11, F12], [F21, F22]])"` — a `str`,
> so a payload *is* built, and the observables run then fails inside the child:
>
> ```
> status: crashed  bad_schema  operands could not be broadcast together with shapes (500,) (500,2)
> ```
>
> `_dynkin` turns that into `"unavailable"` with the reason *"the observables run crashed"*, so the
> net outcome matches the finding — but via a burned sandbox run per level and a reason that again
> blames the solver. The cheap fix, worth doing even if 2b is deferred: have
> `observables_payload` test that the expression **evaluates**, not that it is a string. Note also
> that m13's spec already declares its test functions over the full state
> (`"X[:, 0]**2 + X[:, 1]**2"`), so the generator surface is half-prepared for multi-D already.

So multi-dimensional SDEs get moment matching and nothing else — which is Finding 1 again, with no
escape hatch even in principle.

The reported reason also blames the wrong party: *"the CRN level at dt=0.02 crashed: exception:
IndexError…"* reads, in a review, as a solver defect.

### 2c. Finding 3 — correlation is never graded, and the spec says so in as many words **[R]**

`workspace/sde_gbm_2d_high_corr/problem_spec.json`, in `verification.cross_moment.note`:

> This is the ONLY place rho appears in any terminal statistic. A solver that drops the correlation
> entirely (independent increments) still passes both per-component moment checks, so this check is
> the only thing that detects […]

The spec carries `expression`, `value_at_T`, `correlation_of_terminal`, `correlation_value_at_T`,
`covariance_value_at_T` and a `tol`. The kernel never reads any of them. The only consumer of
`cross_moment` anywhere is [`reference.py:531`](../verifylib/reference.py#L531), which uses it to
check the *spec against itself* — the moment-ODE consistency ledger — not to grade a solver. The
entry also carries `gate: false`.

ρ = 0.95 was chosen to make the correlation matrix near-singular (condition number 39, Cholesky
second diagonal 0.31225). That is the entire difficulty of the problem, and a solver that generates
two independent increment streams scores 10 on it.

> **[V]** Verified. `cross_moment`'s only consumers are
> [`reference.py:531`](../verifylib/reference.py#L531) (the spec-vs-spec ledger) and
> `schema.py`'s `ANSWER_KEY_PATHS`. The near-singularity numbers check out: eigenvalues
> 1.95 / 0.05 give condition number 39, and `sqrt(1 - 0.95**2) = 0.31225`. See [REV 22] — the
> `gate: false` key this finding points at is read by **no** code path, which changes 3a.

### 2d. Finding 4 — there is no distributional evidence of any kind **[R]**

`moment_stats` ([`sde/constraints.py:27`](../verifylib/kernel/sde/constraints.py#L27)) returns mean,
variance and their standard errors. `_compare` ([`sde/driver.py:351`](../verifylib/kernel/sde/driver.py#L351))
compares those two. Nothing anywhere computes a third or fourth moment, a quantile, an empirical
CDF or a two-sample statistic: `grep -rni 'kurtosis|skewness|kolmogorov_smirnov|empirical_cdf'`
over `verifylib/` returns zero hits in the kernel.

> **[V]** Verified, zero hits. And `moment_stats` does already compute `mu4`
> ([`sde/constraints.py:27`](../verifylib/kernel/sde/constraints.py#L27)), so the cheap tier of 3b
> is as free as claimed. The *richer* tier is not — see [REV 23].

OU, GBM, Black–Scholes and CIR all have **fully known terminal laws**, not just known moments. The
class of defect this misses is large and cheap to produce:

* increments drawn uniform (or Rademacher) and rescaled to unit variance — exact first two moments,
  wrong law, and for an additive-noise linear SDE the terminal mean and variance are *exactly*
  right;
* a Milstein correction evaluated at `X_{n+1}` instead of `X_n` — the error shows in the third
  moment long before it shows in the second;
* a correlated or short-period RNG.

This is the largest missing *class* of evidence rather than a missing check. It is also the one
with the best cost ratio: the third and fourth standardized central moments come free from the
terminal paths already in memory.

### 2e. Finding 5 — Dynkin assumes a deterministic initial state **[R]**

`x0 = (spec.get("parameters") or {}).get("X_0", 0.0)`
([`sde/driver.py:464`](../verifylib/kernel/sde/driver.py#L464)), and `per_path_residual` forms
`phi(X_T) - phi(x0) - integral(L phi)` ([`dynkin.py:113`](../verifylib/kernel/sde/dynkin.py#L113)).

`sde_quintic_random_ic` is `dX = -X**5 dt` with `X(0) = xi ~ Normal(0, sigma_bar**2)` and **no
`X_0` key in `parameters` at all**. So `x0` silently defaults to `0.0`, and for `phi = X**2` the
identity is formed with `phi(0) = 0` where it needs `E[phi(X_0)] = sigma_bar**2 = 0.1111`. The
residual is off by a fixed −0.111 at every level, which survives the Richardson extrapolation as an
offset and lands on `rubric_sde`'s row:

> 3, "Dynkin fails: the paths do not satisfy the generator built from the spec's own drift and
> diffusion. The usual cause is the wrong equation integrated correctly — a sign or a coefficient"

— for a correct solver, with feedback that names the wrong cause. The kernel has no concept of a
random initial condition anywhere: `SdeContext.x0`
([`invariants.py:694`](../verifylib/kernel/invariants.py#L694)) and the `martingale` evaluator carry
the same assumption.

> **[V]** Verified. `sde_quintic_random_ic`'s `parameters` are `{sigma_bar, drift_exponent}` with
> no `X_0`, `sigma_bar = 1/3`, and `sigma_bar**2 = 0.1111`. One caveat for sequencing: the spec
> this is read from is the **stale** one, which has no `verification` block and so no
> `dynkin_test_functions` — the problem cannot reach Dynkin today at all. The finding is a real
> structural assumption, but reproducing it end to end needs Phase 6 first, which is another
> reason [REV 27] moves Phase 6 forward.

### 2f. Finding 6 — one non-evaluable moment discards the whole reference, and `*_value_at_T` is never read **[M]**

`_claimed` ([`sde/driver.py:298`](../verifylib/kernel/sde/driver.py#L298)) reads
`{kind}_expression`, then the per-component `{kind}_X` spellings. It never falls back to
`{kind}_value_at_T`. `_reference` ([`:318`](../verifylib/kernel/sde/driver.py#L318)) then evaluates
mean and variance in one `try`, and **returns `None` for both** if either raises.

Measured on `sde_quintic_random_ic`, whose variance is stated as a quadrature expression:

```
claimed var: ['scipy.integrate.quad(lambda xi: (xi**2 / np.sqrt(1.0 + 4.0*t*xi**4)) * ... )[0]']
_reference → (None, {'reason': "the claimed variance is not evaluable: name 'scipy' is not defined"}, 'unavailable')
```

The mean is `"0.0"` and evaluates fine. `variance_value_at_T: 0.09248275891171216` is sitting in
the same object. Both are thrown away.

The downstream reporting is then wrong twice over. `moments_ok`, `resolved` and `variance_ok` are
all `"unavailable"`, which `passed()` and `failed()` both correctly refuse
([`metrics.py:66-78`](../verifylib/kernel/metrics.py#L66)) — so the run falls through every row of
`rubric_sde_analytic` to the default:

> 4, "the variance check fails, though the code ran cleanly"

No variance check ran. There is no "no reference was evaluable" row in the analytic SDE rubric, and
the PDE side has one (`rubric_pde_analytic`: `2, "no error could be measured against the closed
form"`).

> **[V]** Reproduced end to end on a grafted spec: `score=4`, reason *"the variance check fails,
> though the code ran cleanly"*, with `moments_ok / variance_ok / resolved` all `"unavailable"`.
> Provenance comes out `analytic_unvalidated` at tier `A_unavailable`, because `check_reference`
> independently reports `unavailable` for the same spec — which is exactly the coupling [REV 17]
> turns into a requirement on 1d.

### 2g. Finding 7 — cross-plan agreement runs on a shared seed, against its own manual **[D]**

`verification_manual.md` §22, lines 1072-1084:

> Same rules as §11, with one addition specific to Monte Carlo: sibling plans must be run with
> **independent seeds**, and agreement is tested against the *combined* standard error […] Two
> plans agreeing on a shared seed proves nothing — they saw the same noise.

`_compare_sde` ([`cli.py:360`](../verifylib/cli.py#L360)) runs both plans at the graded
configuration **including the same `seed`**, and its docstring argues the opposite case explicitly
(two schemes that consume increments the same way share a Brownian path, so the difference is
nearly noise-free).

Both readings are defensible — CRN is the right tool for *detecting* a disagreement, and an
independent seed is the right tool for *earning* agreement as evidence — but they cannot both be
the rule, and agreement is priced as a circularity break worth a 10
([`score.py:103`](../verifylib/kernel/score.py#L103)). The code does implement §22's conservative
combined-SE formula; it is only the seed that diverges.

> **[V]** Verified. `_compare_sde` uses `np.hypot(se_a, se_b)` — §22's conservative combined SE —
> and the same `seed` for both plans. This is the lowest-risk finding in the document: no
> false-failure surface either way, which is why [§7](#7-revised-shipping-order-rev-2) leaves it
> late without concern.

### 2h. Finding 8 — one seed, everywhere **[R]**

`seed` is read once from `evaluation_thresholds` (default 42) and used for the graded solve, the
CRN ladder and the Dynkin ladder. No replication anywhere.

`dynkin.py:40-48` concedes the consequence in its own docstring:

> Measured on a correct Euler-Maruyama OU solver at 40 000 paths across six seeds, the extrapolated
> residual lands between 0.2 and 3.2 standard errors of zero. The dominant term is the *sample mean
> of the martingale part*, which is a property of the seed rather than of the scheme, so […] a
> residual false-failure rate survives this correction.

That is the right thing to write down and the wrong place to stop. A 3–5 seed replication converts
a known-and-tolerated false-failure rate into a measured one, and costs less than the dt/4 Dynkin
level already being paid for.

> **[V]** Verified, and this is the finding that reorders the plan. `dynkin.py:40-48` is quoted
> accurately, and under Phase 1b that tolerated false-failure rate stops costing a Path-B plan
> two points and starts costing a Path-A plan a hard cap at 3 (see [REV 5]). At the docstring's
> own worst observed z of 3.2, against a K=2 Bonferroni bar of ~2.24, a **correct** solver caps at
> 3. See [REV 24].

### 2i. Finding 9 — `stability_check` is unimplemented in the kernel **[R]**

`project_manual.md:304` makes it the scoring criterion for the blow-up and domain problems, and
says: *"A spec with `has_analytic_solution: false` and `stability_check: null` has no scoreable
criterion at all and is a formulator error."*

The only implementation is [`benchmark/verify.py:422`](../benchmark/verify.py#L422) — the benchmark
oracle, which does not ship with the plugin. `grep -rn stability_check verifylib/` returns one hit,
in a test fixture. `sde_fene_blowup` declares `{"type": "domain", "abs_bound": 1.0}` and
`sde_quintic_drift_noise` declares `{"type": "finite"}`; the kernel grades both off surrogates and
Dynkin and never applies the declared criterion. The `finite` invariant partially covers the
`finite` type; nothing covers `domain`.

> **[REV 9] — the structured forms live in the oracle, not in any formulated spec, and `:310` is
> the line that is wrong.**
> `{"type": "finite"}` and `{"type": "domain", "abs_bound": 1.0}` are at
> [`benchmark/problems.py:3224`](../benchmark/problems.py#L3224) and
> [`:3261`](../benchmark/problems.py#L3261) — the oracle's problem definitions. Neither
> `sde_fene_blowup` nor `sde_quintic_drift_noise` has a live `problem_spec.json`, and the one
> stale instance degraded to **prose**:
> `"stability_check": "all terminal states must satisfy |X(T)| < 1 and be finite (no NaN/Inf)"`.
> So 5a needs a schema rule and a formulator fix before it can grade anything — see [REV 25].
>
> And do not enforce the quoted sentence from `project_manual.md:310`.
> `sde_ginzburg_landau_s6` is deliberately `has_analytic_solution: false` with
> `stability_check: null`, and its own `implementation_notes` explain why: setting it non-null
> would tell the evaluator to drop the moment tolerances `problem.md` actually scores on, so
> finiteness is carried as a gated `verification.constraints` entry instead. `sde_double_well_langevin`
> is in the same state. Enforcing `:310` hard-fails **2 of the 7 live specs**; the Path-B surrogate
> tier has made that sentence obsolete and it should be corrected alongside `project_manual.md:122`.

### 2j. Finding 10 — the clip check is promised in two places and implemented in none **[D]**

`verification_manual.md` §21: *"A bare `np.maximum(X, 0)` clip hides the defect without fixing the
scheme — flag it if you see it in `solver.py`."* And `agents/solver-sde.md`, told to the solver as
a fact about the evaluator: *"a bare `np.maximum(X, 0)` clip at the end hides the defect rather
than fixing it, and the evaluator checks `solver.py` for exactly that."*

`leakage.py`'s two scans look for answer-key citations and prohibited imports
([`leakage.py:29-47`](../verifylib/leakage.py#L29)). There is no clip check, in the kernel or in
the gate. The claim in `solver-sde.md` is load-bearing — it is the deterrent — and it is false.

> **[REV 10] — one promise, not two.** §21's line reads *"flag it if you see it in `solver.py`"* —
> an instruction to the **reviewer**, which is defensible as written and is not a claim about the
> kernel. The false factual claim is only
> [`agents/solver-sde.md:99`](../agents/solver-sde.md#L99): *"the evaluator checks `solver.py` for
> exactly that."* Verified: no clip scan exists in `leakage.py`, `gate.py`, `review.py` or
> `audit.py`. Keep the finding; narrow the `[D]` to the one line that is actually false, because
> the fix differs — §21 needs nothing, `solver-sde.md` needs either 5b or a deletion.

### 2k. Finding 11 — the order study reduces every system to component 0 **[R]**

`strong_order` ([`crn.py:49`](../verifylib/kernel/sde/crn.py#L49)), `weak_order`
([`:66`](../verifylib/kernel/sde/crn.py#L66)) and `richardson_moment`
([`:177`](../verifylib/kernel/sde/crn.py#L177)) each open with
`reshape(len(x), -1)[:, 0]`. Component 1 of a two-component system appears in no order estimate and
in no Richardson error estimate. `_compare` already does the right thing here — every declared
moment of every component, worst outcome wins — so the pattern to follow is in the same file.

> **[REV 11] — there is a fourth site, in a different file.**
> `dynkin._phi_values` ([`dynkin.py:87`](../verifylib/kernel/sde/dynkin.py#L87)) also opens with
> `reshape(len(X), -1)[:, 0]`, so **Dynkin** is component-0-only too. That one is worse than the
> three in `crn.py`: `sde_multichannel_stiff_m13` declares
> `"X[:, 0]**2 + X[:, 1]**2"` as a test function, so the reshape silently evaluates half of a
> declared test function and the residual it forms is not the identity the spec asked for. Fold it
> into 2c.

### 2l. Finding 12 — `empirical_mean` / `empirical_variance` are transported and never read **[R]**

`_dump_sde` marshals them ([`sandbox.py:264`](../verifylib/kernel/sandbox.py#L264)) and
`_reconstruct` rebuilds them ([`:473`](../verifylib/kernel/sandbox.py#L473)). No consumer exists.
The kernel recomputes both from `terminal_paths`, which is the right call for *grading* — but it
means the numbers a solver prints into `SOLUTION.md`'s Results section have no kernel counterpart,
and a solver reporting `np.var(X)` where it claims `ddof=1` is invisible. One line of comparison
turns a free field into a reported-vs-measured check.

> **[V]** Verified. `grep -rn empirical_mean verifylib/` outside the tests returns only
> `sandbox.py:264` and `:473` — marshalling and reconstruction, no consumer.

### 2m. Finding 13 — spec coverage **[M]**

Only **7** SDE problems have a live `workspace/<slug>/problem_spec.json` with a `verification`
block. The other 16 sit in `workspace_stale_pre_c2b_backup/` with no `verification` key at all —
so `expected_strong_order` is absent, defaults to `0.5`
([`crn.py:117`](../verifylib/kernel/sde/crn.py#L117)), and by Finding 1 no scheme check runs on any
of them. No invariants, no `dynkin_test_functions`, no `moment_ode`.

| Live, with a `verification` block | Stale |
|---|---|
| `sde_bm_with_drift`, `sde_cir_feller_violated`, `sde_double_well_langevin`, `sde_gbm_2d_high_corr`, `sde_ginzburg_landau_s6`, `sde_multichannel_stiff_m13`, `sde_ornstein_uhlenbeck` | the remaining 16 |

> **[V]** Verified exactly: 23 `workspace/sde_*` directories, 7 with a `problem_spec.json`
> carrying a `verification` block, 22 in `workspace_stale_pre_c2b_backup/` of which **every one
> has no `verification` key at all**, and `sde_double_well_langevin` live-only. 23 − 7 = the 16.
> Of the 7 live, 5 are Path A and 2 are Path B.

### 2n. Finding 14 — nothing verifies the Milstein eligibility claim **[R]**

`milstein_eligible: false` and the prose in `diffusion_derivative` ("multi-dimensional Milstein
requires the Levy area of (W1, W2), which is not computable in closed form") are advice to the
plan-creator. Nothing checks that a plan on a multi-noise problem has not declared
`scheme_family: milstein` and implemented the scalar correction anyway. With Finding 2 in place the
strong-order study cannot catch it either, because it never runs there.

> **[REV 12] — `milstein_eligible` means two different things on this set, and nothing reads it.**
> It appears in four specs, always `false`, for two unrelated reasons: on
> `sde_multichannel_stiff_m13` it is the Lévy-area reason this finding describes, but on
> `sde_bm_with_drift`, `sde_ornstein_uhlenbeck` and `sde_double_well_langevin` it is
> `dg/dX = 0` — Milstein is *bit-identical* to Euler–Maruyama, which is not a defect and must not
> be capped. `grep -rn milstein_eligible verifylib/` returns nothing, so this is a fourth unread
> spec field. See [REV 21]: read the field rather than re-deriving a weaker version of it from
> `noise_dimension`.

### 2o. Finding 15 — the Kolmogorov surrogate no longer passes its own trust guard **[M]** *(rev 2)*

Found while recording the baseline ([§9b](#9b-the-6-baseline--measured-2026-10-04)), and it is not
one of the fourteen above. All three `sde_ginzburg_landau_s6` plans now report:

```
surrogate_ok: unavailable
  kolmogorov: "the truncation guard failed: widening the domain moved the answer by 1.39e-02"
  moment_ode: "no moment ODE, or it does not close exactly"
  stationary_density: "the spec does not declare the process ergodic"
```

So the §15-17 ladder produces **no reference at all** on that problem, and the three plans drop
from `provenance: surrogate` to `self_convergence` / `manufactured_partial`. `STATE.md` records
all three at **10, surrogate**; they now measure **3, 3, 9**.

The guard is `max(shifts) >= 0.1 * trust_tol`
([`surrogates.py:189-194`](../verifylib/kernel/sde/surrogates.py#L189)) with
`trust_tol = variance_rel_err_max = 0.1`, so the bar is **0.01 — one tenth of the tolerance** —
against a measured truncation sensitivity of 0.0139. It fails by 39%.

Whether that is a regression or a correct tightening is genuinely open and worth settling before
anything else:
* *Correct* is plausible. The spec's own notes say `Var[X(1)] ~ 1.88` is "tail-dominated" and `X(1)`
  spans ~12 decades, so a far-field boundary that moves the answer 1.4% under a 1.5x widening may
  simply be a route that should not be trusted.
* *Regression* is also plausible, because the provenance recorded in `STATE.md` requires this route
  to have passed at some point, and the `0.1 *` factor is the kind of change that postdates a
  recorded score.

Either way it is a **live contradiction between a spec and the kernel**: that spec's
`implementation_notes` state in as many words that "the scoreable moment criterion comes from
`verification.kolmogorov` (Tier A', provenance 'surrogate')", and today it does not. Settle this
before step 1 of [§7](#7-revised-shipping-order-rev-2): if the guard is right, the spec needs a
different scoring route declared; if it is wrong, three plans are under-scored by 7, 7 and 1 and
the baseline in §9b is not a baseline.

This finding belongs to no phase in §3. It is in scope only because the plan cannot measure what
it moves without a trustworthy starting point.

#### Finding 15 — resolved: a kernel bug in `_widen`, not a correct tightening **[M]**

Three independent lines of evidence, and they agree.

**1. The guard is original, so nothing regressed in it.**
`git log -S "0.1 * float(trust_tol)" -- verifylib/kernel/sde/surrogates.py` returns exactly one
commit — `3447862`, 2026-09-07, the commit that *created* the file. The `0.1 x trust_tol` factor has
never been anything else, so "the bar got stricter after the score was recorded" is ruled out.
(`workspace/` is untracked, so `STATE.md` has no history to date the recorded 10 against.)

**2. The spec documents the exact check, and the kernel performs a different one.**
`verification.kolmogorov.note` says, in as many words:

> Truncation guard checked: **[0,24] vs [0,36]** at Nx=64001 moves the mean by **0.36%** and the
> variance by 0.11%, inside `0.1*variance_rel_err_max`.

The formulator measured the far-field widening — lower bound held at 0, upper bound 24 → 36 — got
0.36%, and knew the bar was 0.01. But `_widen`
([`surrogates.py:99-102`](../verifylib/kernel/sde/surrogates.py#L99)) widens **symmetrically about
the midpoint**:

```python
mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * factor
return [mid - half, mid + half]        # [0, 24] at 1.5x  ->  [-6.0, 30.0]
```

**3. `[-6, 30]` is not a wider version of this problem; it is a different problem.**
Ginzburg–Landau at σ=6 is strictly positive — the spec carries a **gated** constraint
`{"name": "positivity", "expr": "X > 0", "gate": true}` and its note derives positivity from the
exact solution. The diffusion `g(x) = sigma*x` vanishes at `x = 0`, which the same spec calls out as
the reason the scheme "degenerates to ~first order in dx". So the widened solve imposes an
artificial boundary at `x = -6`, across a degenerate point, over a region the process never visits.
The 1.39% shift is the cost of that, not a measure of far-field truncation.

**Measured, by swapping only `_widen` and changing nothing else** (same spec, same Nx=64001,
Nt=16000):

```
symmetric about midpoint (current)   widened [-6.0, 30.0]
   truncation_shift [mean, E[X^2]] = [0.00109, 0.01385]   max 0.01385  vs bar 0.01  -> FAILS
   mean=0.68003  variance=1.88808

far field only (what the spec checked)   widened [0.0, 36.0]
   truncation_shift [mean, E[X^2]] = [0.00361, 0.00052]   max 0.00361  vs bar 0.01  -> PASSES
   mean=0.68003  variance=1.88808
```

Three things to read off it:

* **0.361% reproduces the spec's recorded 0.36% to three figures.** The formulator's note was
  accurate; the kernel was measuring something else.
* **The reference itself is identical either way** — `mean=0.68003`, `variance=1.88808` in both
  runs. Only the *verdict* changes, which is the signature of a bad guard rather than a bad
  reference.
* **The reference is good.** Against the spec's independent discretization-free Monte Carlo
  (`mean 0.6745`, `Var 1.879`) the Kolmogorov solve is within **0.8%** on the mean and **0.5%** on
  the variance — comfortably inside the 5%/10% tolerances it would be used to police. So the
  rejection is discarding trustworthy Tier-A' evidence, not protecting against bad evidence.

Note which moment moves: symmetric widening leaves the mean almost untouched (0.00109) and
corrupts `E[X^2]` (0.01385), which is what adding a spurious negative-`x` region with an artificial
boundary would do to a second moment. Far-field widening does the reverse and both are small.

**Blast radius: exactly one live spec, and it is the one where it matters most.** Of the four specs
declaring `kolmogorov.bounds`, three are two-sided and symmetric widening is harmless:

| Spec | bounds | symmetric widen | one-sided? |
|---|---|---|---|
| `sde_ornstein_uhlenbeck` | [-6, 6] | [-9, 9] | no — fine |
| `sde_double_well_langevin` | [-3, 3] | [-4.5, 4.5] | no — fine |
| `sde_bm_with_drift` | [-4, 7] | [-6.75, 9.75] | no — fine |
| `sde_ginzburg_landau_s6` | **[0, 24]** | **[-6, 30]** | **yes — broken** |

And `ginzburg_landau_s6` is the only one of the four whose spec says Kolmogorov is "the ONLY Tier-A'
route for this problem", so the bug costs three plans their entire reference tier.

**The fix.** A truncation guard must widen the *artificial* boundaries; a boundary that coincides
with the state space's own edge is not artificial. The support is already declarable — the gated
`positivity` constraint is right there in the same spec — so `_widen` should hold any boundary
pinned by a declared support and extend the rest. The minimal version is: hold `lo` when the spec
declares a gated positivity constraint and `lo == 0`, extend `hi` by the factor. A spec-independent
fallback that also works here: widen away from `x0` only.

> **[REV 39] — the fix is offered as two rules, neither is chosen, and they differ on the three
> specs that work today.** "Hold `lo` when the spec declares a gated positivity constraint and
> `lo == 0`" and "widen away from `x0` only" are not the same rule. On `sde_bm_with_drift`
> (`[-4, 7]`, `X_0 = 0`) the first changes nothing and the second moves the widened domain, so the
> second silently re-measures three specs that currently pass their guard.
> [§7](#7-revised-shipping-order-rev-2)'s step 0 calls both of its items "bugs with measured causes,
> not design calls" — the *bug* is measured; the *fix* is a choice. See
> [§11 D7](#d7--pin-boundaries-a-declared-support-already-fixes).

**Consequences for this plan.** The §9b baseline for those three plans is *not* a baseline — they
should be at or near their recorded 10/surrogate, and `provenance: self_convergence` is wrong. Fix
this before step 1 of [§7](#7-revised-shipping-order-rev-2) and re-measure. It also partly reframes
[REV 35a]: plans 1 and 2 reach Dynkin as their only evidence *because* the surrogate wrongly
dropped out, so once the surrogate is restored, the Dynkin verdict stops being the sole thing
standing between them and a 10.

*Correction, after [REV 35b]:* the last sentence above is **wrong** and is left standing only so the
error is visible. `rubric_sde` places `failed(dynkin_ok) -> 3` **before** the surrogate row
([`score.py:347`](../verifylib/kernel/score.py#L347)), so restoring the surrogate does not lift
plans 1 and 2 off 3. Verified directly against the rubric:

```
surrogate restored, Dynkin still fails  -> (3, "Dynkin fails: ...")
surrogate restored, Dynkin passes       -> (10, "a surrogate reference passed its trust guard ...")
surrogate unavailable, Dynkin fails     -> (3, "Dynkin fails: ...")
```

So the real effect of fixing `_widen` on §9b is narrower than claimed: **plan 3 goes 9 -> 10**
(matching its recorded score) and **plans 1 and 2 stay at 3 with `provenance` corrected from
`self_convergence` to `surrogate`** — a better-labelled 3, not a higher score. Which raises a
question this plan does not answer: `STATE.md` records **10** for all three, and under the current
rubric plans 1 and 2 could not have scored 10 with Dynkin failing, so either the recorded scores
predate findings §7.6's row ordering, or predate the downward-refinement change in `dynkin.py`, or
the plans themselves changed. Worth one `git log` before trusting any recorded SDE score.

*Closed by [REV 47], 2026-10-14:* it is the first two hypotheses and not the third — the three
`ginzburg_landau_s6` scores were recorded 2026-08-23, a month before `rubric_sde`'s
`failed(dynkin_ok) -> 3` row was moved ahead of the surrogate row in `a507038`.

---

## 3. Fixes

Grouped so that each phase is independently shippable and independently testable. Phase 1 is the
one that changes benchmark numbers.

> **[REV 13] — the order is close to exactly backwards; see
> [§7](#7-revised-shipping-order-rev-2).** The phases are each independently shippable, as
> claimed, but not in this sequence. Every item in Phase 1 either (a) depends on spec coverage
> that Phase 6 schedules last, or (b) converts a silent pass into a **hard cap** through a check
> whose false-failure rate is inherited rather than measured. Phase 1 is indeed the one that
> changes benchmark numbers — including, as written, on correct solvers. §7 gives the order that
> lands the same work without that window.

### Phase 1 — close the Path-A hole (Findings 1, 6)

**1a. Extend `rubric_sde_analytic` to read the scheme checks.**
[`score.py:263`](../verifylib/kernel/score.py#L263), mirroring `rubric_pde_analytic`'s shape:

| Condition | Score | Reason |
|---|---|---|
| `failed(dynkin_ok)` | 3 | the generator built from the spec's own drift and diffusion rejects the paths — as `rubric_sde` already says |
| moments pass, `failed(orders_ok)` | 7 | the moments match the closed form but the observed strong order does not clear its floor: accurate at this `dt`, not converging at the rate the scheme claims |
| moments pass, `failed(invariants_ok)` (non-gate) | 5 | a declared invariant drifts materially |
| reference was never evaluable | 2 | no moment could be measured against the closed form |

The 7 and 2 rows are transcriptions of rows that already exist on the PDE side; the 3 row is a
transcription of a row that already exists on the SDE Path-B side. Nothing here is a new judgement.

> **[REV 44] — there is no Path-A row for "the order study could not be run", and step 0 makes that
> state common.** The table adds a row for *"reference was never evaluable"* and none for
> `orders_ok: "unavailable"`, which is in `SKIPS` — so `passed()` and `failed()` both refuse it and
> every one of these four rows falls through. `sde_multichannel_stiff_m13` is in exactly that state
> today (score 6, `orders_ok: unavailable`). After step 0's error bar ([REV 37]) abstention becomes
> the *expected* outcome on heavy-tailed multiplicative noise, so the hole widens at the moment the
> plan starts relying on the check — the fails-open shape Finding 9 describes, one level up. The row
> is needed whichever way [§11 D8](#d8--abstain-and-add-a-second-statistic-abstention-costs-nothing)
> had gone; what D8 settles is that it **reports** rather than caps.

> **[REV 14] — the 7 row contradicts Path B and acceptance criterion 2; and the row order is
> unspecified.**
> 1. **7 should be 8.** [`rubric_sde`](../verifylib/kernel/score.py#L333) already scores this exact
>    state — moments pass, `orders_ok` fails — at **8**, with a reason naming the §18 `dW`
>    contract. Transcribing the **PDE** 7 row instead gives the same defect 7 on Path A and 8 on
>    Path B: a *new* cross-path divergence in a plan whose stated purpose is removing one. It also
>    makes criterion 2 ("capped on Path A **exactly** as it is on Path B (≤ 8)") self-inconsistent
>    — "exactly" and "≤ 8" are different claims and this table meets only the weaker one. Reuse
>    `rubric_sde`'s row and its wording.
> 2. **Split the reason on the measured order.** `rubric_sde`'s 8 row *asserts* "a strong order
>    near zero" without reading `strong_order`. After 1c that row fires on both "ignores `dW`"
>    (order ≈ 0) and "converging slower than the scheme claims" (order below floor), which need
>    different feedback. Read `m["summary"]["observed_order"]` and branch.
> 3. **State the row order explicitly**, as `rubric_sde` does (findings §7.6: gate, then a Dynkin
>    or order *failure*, then inconclusive). As drafted it is ambiguous where these four rows sit
>    relative to the existing `failed(resolved) -> 6` and `variance_ok -> 8` rows, and that choice
>    decides what a sign-flipped drift scores.
> 4. The `failed(dynkin_ok) -> 3` row is **cosmetic** once 1b lands — see [REV 5]. Keep it, but
>    the fix is 1b.

> **[REV 38] — `rubric_sde` has *two* `failed(orders_ok)` rows, at 8 and at 3, and neither
> [REV 14] nor D1 says which is being transcribed. [R]**
> [`score.py:354`](../verifylib/kernel/score.py#L354) returns **8** — but only inside
> `if surrogate and moments_ok and resolved and constraints_ok:`.
> [`score.py:368`](../verifylib/kernel/score.py#L368) returns **3**, unguarded, and it is what
> fires whenever there is no surrogate. Path A has no surrogate at all, so "reuse `rubric_sde`'s
> row" does not pick a score — it picks two. [REV 14]'s argument ("`rubric_sde` already scores this
> exact state at 8") reads the first row and the second is the one Path A would structurally land
> on.
>
> This matters beyond the wording: it **couples step 0's two items**, which
> [§7](#7-revised-shipping-order-rev-2) treats as independent. Fixing `_widen` restores
> `ginzburg_landau_s6`'s surrogate, which moves a step-0-induced `orders_ok: False` from the 3 row
> to the 8 row on the same plan. See
> [§11 D10](#d10--what-path-a-substitutes-for-the-surrogate-conjunct).

> **[DECIDED — D1]** Scores **8**, reusing `rubric_sde`'s row and wording, with the reason branching on the measured order. Row order follows `rubric_sde`'s per findings §7.6. See [§10 D1](#d1--the-orders-fail-row-scores-8-not-7).

> **[REV 50] — the invariants row is the one row here that is *not* a transcription, and shipping it
> today drops two live, plausibly-correct Path-A plans by five points on a kernel artifact. [M]**
> 1a closes with *"Nothing here is a new judgement"*, which holds for three of its four rows. It does
> not hold for `moments pass, failed(invariants_ok) (non-gate) -> 5`: `grep` over
> [`score.py`](../verifylib/kernel/score.py) confirms **neither `rubric_sde_analytic` nor
> `rubric_sde` reads `invariants_ok` at all**, so there is no Path-B row to mirror and
> [D1](#d1--the-orders-fail-row-scores-8-not-7)'s parity argument does not reach it.
>
> **Two live Path-A plans already carry `checks.invariants_ok: False` and score 10**, read from the
> §9b sweep's metrics files:
>
> | Plan | failing invariant | `max_drift` | `worst_headroom` | gate? |
> |---|---|---|---|---|
> | `bm_with_drift/1-euler-maruyama` | `martingale` | 0.50258 | −1.400 | no |
> | `gbm_2d_high_corr/1-euler-maruyama` | `martingale` | 0.10475 | −9.020 | no |
>
> **Both failures are artifacts of `_martingale`, not solver defects.**
> [`invariants.py:557`](../verifylib/kernel/invariants.py#L557) computes
> `|mean(X[:, 0]) - (reference_value_at_T or x0)|`, and its registered keys are
> `("ci_mult", "reference_value_at_T")` — **`expr` is not among them and is silently ignored:**
> * `sde_gbm_2d_high_corr` declares `expr: "np.exp(-mu1*T) * X[..., 0]"`. Ignored, so the kernel
>   computes `|E[X_T] - X_0| = e^(mu1*T) - 1 = 0.10517`; measured **0.10475**, **0.27 SE** away. The
>   discount the spec asked for never happens.
> * `sde_bm_with_drift` declares the martingale in **prose only** — no `expr`, no
>   `reference_value_at_T`, just a note reading *"M_t = X_t - mu*t is a martingale, so
>   E[X_T - mu*T] = X_0"*. The kernel computes `|E[X_T] - X_0| = mu*T = 0.5`; measured **0.50258**,
>   **1.92 SE** away. That solver is pathwise exact.
>
> Three consequences:
> 1. **`_martingale` belongs in [§7](#7-revised-shipping-order-rev-2)'s step 0**, with `_widen` and
>    `strong_order`'s error bar. Same shape as both: a measured kernel bug, masked today, unmasked by
>    a later step. Either honour `expr`, or make a `martingale` entry carrying an unusable `expr` a
>    schema error — silently ignoring it is the fails-open shape of
>    [Finding 9](#2i-finding-9--stability_check-is-unimplemented-in-the-kernel-r), and a seventh
>    instance of [REV 29]'s declared-and-never-read pattern, in a new direction: declared, *read*,
>    and discarded by the evaluator that claims to use it.
> 2. **`sde_bm_with_drift` needs `reference_value_at_T: 1.5`** (or an `expr`) — a one-line spec edit
>    in [REV 40]'s class, belonging in
>    [§9c](#9c-spec-and-artifact-work-that-gates-specific-acceptance-criteria).
> 3. **The invariants row must not ship before both.** As drafted it is the
>    false-hard-fail-on-correct-work this document keeps warning about, on two plans, from a bug in
>    the kernel's own evaluator.
>
> Also: `_martingale` carries the same `reshape(...)[:, 0]` reduction as
> [Finding 11](#2k-finding-11--the-order-study-reduces-every-system-to-component-0-r) — a **fifth**
> site after the three in `crn.py` and `dynkin._phi_values`. On `gbm_2d_high_corr` that means the
> spec's *"Same for Y with mu2"* is never tested. Fold it into 2c.

**1b. Run Dynkin on Path A, once, at certification.**
[`sde/driver.py:157`](../verifylib/kernel/sde/driver.py#L157): `run_dynkin = (path == "B") or
would_certify`. §24 argues Dynkin "adds nothing on Path A that exact moments do not already do
better", and the fabrication canary is the counterexample: exact *terminal moments* do not
establish that the paths came from the SDE, and Dynkin is the only check in the manual that does.
§24's own reasoning already concedes half of this one paragraph earlier — "Euler–Maruyama and
Milstein have the same weak order, so the moment comparison cannot tell them apart" — and then
stops at the strong order.

> **[REV 15] — this is the entire fix for Finding 1, and it needs three things the draft does not
> mention.**
> Measured: forcing `run_dynkin = True` on Path A caps the fabrication canary at 3 and leaves the
> honest solver at 10 (see [REV 5]). The argument against §24 is correct and the measurement
> supports it. What is missing:
> 1. **A `ceiling()` reason for SDE evidence.** The cap surfaces as *"D1 stalled with a validated
>    operator…"* — PDE vocabulary that names neither Dynkin nor the order study, so criterion 1
>    fails. Either branch the reason on `evidence["kind"]` or re-word it to cover both.
> 2. **Sequencing.** 1b routes a check with a documented residual false-failure rate
>    (`dynkin.py:40-48`: z between 0.2 and 3.2 across six seeds) into `gates_clean`, i.e. a hard 3
>    on 5 of 7 live Path-A specs and 21 of 23 after Phase 6. Ship Finding 5's random-IC fix and
>    4a's replication **first**, or ship 1b in two steps: `dynkin_z` reported on Path A as a
>    diagnostic, promoted to a cap once the false-failure rate is measured rather than inherited.
>    See [REV 24].
> 3. **Cost — this is the expensive phase, not 1c.** The Dynkin ladder refines *downward* from the
>    graded `dt` (`dt, dt/2, dt/4`) at full `num_paths`, so ~7× a graded solve plus the observables
>    accumulation; `dynkin.py` records ~30 s per plan. §4d prices 1c and not this. See [REV 30].

**1c. Drop the `expected_strong_order >= 1.0` condition on `run_crn_ladder.`**
[`sde/driver.py:153`](../verifylib/kernel/sde/driver.py#L153). Measuring a floor of 0.5 and getting
0.0 is exactly how a solver that ignores `dW` is caught; the condition as written exempts every
Euler–Maruyama plan in the set from the only check that would catch it. Cost is one extra ladder
per certifying plan, which §24 Rule 2 already budgets for.

> **[REV 16] — as written this false-fails correct solvers on about six of the 23 problems.
> Measured.**
> Honest Euler–Maruyama on the OU canary, floor at the kernel default 0.5,
> `scheme_family: euler-maruyama` declared in `SOLUTION.md`, ladder forced on:
>
> ```
> declared=True:  orders_ok=False  observed=0.9863  band=two-sided  floor=0.5
> declared=False: orders_ok=True   observed=0.9863  band=one-sided  floor=0.5
> ```
>
> Euler–Maruyama has strong order **1**, not 0.5, whenever the noise is additive, and six of the
> 16 stale specs are additive or noise-free — `sde_bm_standard`, `sde_linear_additive`,
> `sde_ou_stiff`, `sde_stochastic_oscillator`, `sde_oscillator_long_horizon`,
> `sde_quintic_random_ic` — all defaulting to 0.5 via
> [`crn.py:117`](../verifylib/kernel/sde/crn.py#L117). Under 1c plus 1a's new row, a correct
> solver on each scores 7 instead of 10.
>
> `sde_bm_standard` is worse than probabilistic. `dX = dW` makes Euler–Maruyama *exact*, so the
> CRN pathwise differences are floating-point roundoff and the estimate is junk:
>
> ```
> strong_order -> (-0.1226, {'s0': 1.74e-16, 's1': 1.90e-16})   orders_ok: False
> ```
>
> which fails **both** bands, deterministically, on a perfect solver.
>
> Three changes:
> * **(a)** Make Phase 6 — or at minimum `expected_strong_order` on every spec — a prerequisite.
>   See [REV 27].
> * **(b)** When the floor came from `kernel_default` rather than `spec`, use the **one-sided**
>   band. That is `expected_orders`' own stated rule — *"an inference must never make a test
>   stricter"* — and `cfg` already records the source, so the condition is available for free.
>   One-sided still catches order ≈ 0, which is the defect 1c exists for.
> * **(c)** Give `strong_order` the noise-floor discipline `weak_order` already has. It guards only
>   `s0 <= 0.0` (exact zero) where it needs "both differences are at the roundoff scale of the
>   terminal values → **indeterminate**, which is not a failure" — `weak_order`'s
>   `w1 < floor_mult * se` test, applied to the strong order.
>
> The cost claim is fine and understated in the other direction: `dt0` defaults to `4*dt` and the
> ladder runs at `dt0, dt0/2, dt0/4` at `conv_num_paths` (20 000), well under one graded solve. The
> risk here is calibration, not cost. §24 Rule 2 also budgets the ladder only when
> `expected_strong_order >= 1.0`, so that clause is not support for dropping the condition.
> **[REV 33] — amends [REV 16]: the two-sided failure is latent, not current, and today the same
> miscalibration produces a false *pass*.**
> `plan_meta.read` over all 12 live SDE plans: **10 carry `scheme_family_source: inferred`**, only
> `sde_double_well_langevin`'s three declare it. Inferred means `guards` uses the **one-sided**
> band, so 1c would not fail those 10 today — [REV 16]'s measurement was taken with
> `scheme_family` *declared*, which is what `agents/` instructs plan-creators to do and what
> `double_well` already does. So the exposure is: latent on the current pool, live the moment the
> field is declared or Phase 6 brings new plans in.
>
> What bites **today** is the roundoff floor, [REV 16](c), and it fails open. `sde_bm_with_drift`
> is `dX = 0.5 dt + 0.3 dW`, where Euler–Maruyama is pathwise exact — its own `SOLUTION.md` says
> so. The ladder already runs there (`expected_strong_order: 1.0`), and measured:
>
> ```
> score=10 orders_ok=True observed_order=2.2685  floor=1.0
> strong_detail={'s0': 9.41e-15, 's1': 1.95e-15}
> ```
>
> An order of 2.27 from differences at 1e-15 is noise, it clears the one-sided bar of 0.6 by
> accident, and `orders_ok: True` is reported as evidence. Declare `scheme_family` on that plan and
> the two-sided band fails it at `|2.27 - 1.0| = 1.27`, and under 1a it drops from 10 to 7 (or 8
> per [REV 14]) — a correct, exactly-integrating solver. **So (c) is not a hardening of (b); it is
> the prerequisite of both**, and it is the one item in 1c that should land before anything else in
> Phase 1. Order it first within 1c, and give it its own canary per [REV 28].
>
> **[REV 34] — the same two-sided miscalibration exists a third time, in `ranking.py`, as
> unreachable code.**
> `ranking.order_margin` ([`ranking.py:251-259`](../verifylib/ranking.py#L251)) has an SDE branch
> that scores `1 - |observed - order_floor| / max(order_floor, 0.5)` — two-sided distance from the
> floor, the same shape as `guards`. For a correct additive-noise Euler–Maruyama at observed 0.986
> against a floor of 0.5 that is **0.028**: last in its pool, on a right answer.
>
> It never runs. The branch is guarded by `if observed is None or design is None`, and
> `design_order` is set only in [`kernel/driver.py:103`](../verifylib/kernel/driver.py#L103) from
> `plan["spatial_order"]` — the **PDE** driver. The SDE driver never sets it, and no SDE plan
> declares `spatial_order`. So the SDE order component of the quality index is dead code today,
> and it will activate with this penalty already in it the moment anyone wires `design_order` on
> the SDE side (which the open SDE weak-error work would do).
>
> Consequence for Phase 1: the band fix belongs in **three** places, not one — `crn.guards`
> (`orders_ok`), 1a's new rubric row ([REV 14]), and `ranking.order_margin`. Fixing one or two
> leaves the defect reachable from the others, and the ranking one is invisible to every acceptance
> criterion in §6 because it moves ranks, not scores. Add a criterion for it; see [REV 32].

> **[REV 43] — [REV 34]'s own ask was never satisfied.** It closes with "Add a criterion for it;
> see [REV 32]", and `ranking.order_margin` appears in this document's index, in [REV 34], in
> [REV 36]'s cross-reference and in [§8](#8-rev-2-verification-log)'s log — and in **no acceptance
> criterion and no step of [§7](#7-revised-shipping-order-rev-2)**. Step 7 reads "1c, per D2" and
> names `crn.guards` only. Because the branch moves ranks rather than scores, nothing in
> [§6](#6-acceptance-criteria) would catch it either way, which is the exact blind spot [REV 34]
> described. It is unreachable today, so both options are nearly free. See
> [§11 D12](#d12--fix-the-ranking-branch-rather-than-delete-it).

> **[DECIDED — D2]** (a) and (b) both adopted: the band is **one-sided whenever the floor's source is `kernel_default`**, and Phase 6 declares the floors so that path stays a fallback. (c) is **superseded** by [REV 36a]'s error bar on `strong_order`, which subsumes the roundoff case — implement one mechanism, not two. See [§10 D2](#d2--one-sided-band-on-a-defaulted-floor-and-phase-6-declares-the-floors).

**1d. Make the two moments independent, and read `*_value_at_T`.**
`_claimed` gains a `{kind}_value_at_T` fallback; `_reference` evaluates mean and variance in
separate `try` blocks and returns whichever succeeded, recording the other's reason. A reference
with one usable moment is weaker evidence than one with two and much stronger than none.

> **[REV 17] — there is a second half in `reference.py`, or this caps at 9.**
> `reference_outcome` is computed independently of `_reference`, by
> `check_reference` → `check_moment_ode_consistency` →
> [`_moment_observables`](../verifylib/reference.py#L500), which reads
> `am[f"{kind}_expression"]` and has **no `*_value_at_T` fallback either**. Add the fallback only
> to `_claimed`/`_reference` and the moments become gradeable while `reference_outcome` stays
> `"unavailable"` → `tier_of` returns `A_unavailable` → `ceiling()` returns **9** with provenance
> `analytic_unvalidated`. Acceptance criterion 6 would read as satisfied while the plan can never
> reach 10 on that spec.
>
> The change is small: `_moment_observables` already accepts a numeric `claimed` and already has an
> `at_T=True` mode (it uses it for `cross_moment.covariance_value_at_T`), so it is
> `am.get(f"{kind}_expression") or am.get(f"{kind}_value_at_T")` with `at_T=True` on the fallback.
> Both halves in one commit, and a test that asserts the *provenance*, not just the score.

**1e. Update `project_manual.md:122`** so the rubric table and the code say the same thing. **[D]**
This is the fix; 1a is its implementation.

> **[REV 18] — `verification_manual.md` §24 needs the same treatment, in the same commit. [D]**
> §24 Rule 2 states `run_crn_ladder` and `run_dynkin` as literal pseudocode, including both
> `expected_strong_order >= 1.0` and `run_dynkin = analytic is None`, and its prose argues
> *"Dynkin adds nothing on Path A that exact moments do not already do better."* Phases 1b and 1c
> change both. By this document's own `[D]` discipline — the manual is the specification and the
> code is faithful to it — leaving §24 standing re-opens exactly the gap Finding 1 describes the
> next time someone checks the implementation against the spec. 1e should name three edits:
> `project_manual.md:122-128` (the rubric), `project_manual.md:310` (see [REV 9]), and
> `verification_manual.md` §24 Rule 2.

### Phase 2 — make multi-dimensional SDEs verifiable at all (Findings 2, 11, 14)

**2a. Declare and thread the noise dimension.** A `noise_dimension` field in the spec schema,
defaulting to `state_dimension` when `drift_expression` is a list and to 1 otherwise; passed as
`m=` from both `ladder_spec` call sites. The sandbox side already honours it — this is a
three-line change plus a schema entry, and it is what unblocks the strong-order gate on six
problems.

> **[REV 19] — the field exists (see [REV 6]) and the proposed default is wrong on the problems it
> matters most for.**
> `noise_dimension = state_dimension if isinstance(drift_expression, list) else 1` fails in both
> directions on this very set:
> * `sde_multichannel_stiff_m13`: `state_dimension: 2`, drift is a **string**, 13 channels → the
>   default gives **1** and it still raises `IndexError`. Its spec already declares `13`.
> * `sde_stochastic_oscillator` / `sde_oscillator_long_horizon`: 2 states, **1** noise → the
>   default gives **2** and would hand a correct solver an array of the wrong shape, converting a
>   working plan into a crash.
>
> Noise dimension is not derivable from state dimension. Do this instead: read the declared field;
> make it **schema-required** for any SDE whose solver consumes more than one channel; treat
> absent-or-`null` on such a spec as a formulator error rather than inferring; and fix the two
> specs that have it wrong (`sde_gbm_2d_high_corr` → 2, and whatever Phase 6 formulates for
> `sde_gbm_2d_correlated` → 2 and `sde_log_heston_feller_violated` → 2).
>
> Verified that the rest of 2a is as cheap as claimed: `ladder_spec(..., m=2)` against the real
> `sde_gbm_2d_high_corr/plans/1-euler-maruyama` runs clean and recovers
> `corr(X_T, Y_T) = 0.945` against the spec's 0.9478. Per [REV 7] this unblocks **four** problems,
> not six.

**2b. Let the generator take a vector drift and diffusion.** `observables_payload` currently
refuses a list ([`dynkin.py:73`](../verifylib/kernel/sde/dynkin.py#L73)). The multi-dimensional
generator is `L phi = sum_i f_i d_i phi + 0.5 sum_ij (G G^T)_ij d_ij phi`, which needs
`diffusion_matrix_expression` (already in every multi-D spec, today as prose) in a machine-readable
form. This is the largest single item in the plan and the one to defer if anything is deferred —
2a alone restores the strong-order gate, which is most of the value.

> **[REV 20] — this collides with the leakage model, and should route around it.**
> [`schema.py:80`](../verifylib/schema.py#L80) classifies `drift_expression`,
> `diffusion_expression` and `diffusion_matrix_expression` as **`GUIDANCE_PATHS`**: *"Read by an
> agent, not evaluated by the pipeline. A program here is still a leak vector; prose here is normal
> and correct."* `check_expressions` enforces that — a program in one of those fields is an error
> whose message names the archived Heston leak. Phase 2b proposes making
> `diffusion_matrix_expression` machine-readable, i.e. promoting a guidance field into an evaluated
> one, which is the shape `schema.py` exists to refuse. (`dynkin.observables_payload` already
> evaluates the first two, so the contradiction predates this plan — it is just not load-bearing
> until 2b.)
>
> Put the machine-readable generator under an **evaluated** path instead —
> `verification.generator.drift` / `verification.generator.diffusion_matrix`, which `ANSWER_KEY_PATHS`
> and `check_expressions` can police like `verification.residual_operator` — and leave the prose
> fields as guidance. Or amend `schema.py` with the reasoning, explicitly. Either way the decision
> belongs in the plan, because it is the difference between a schema entry and a leak surface.
> The [REV 8] evaluability guard should land regardless, since it is cheap and 2b is the deferral
> candidate.

> **[DECIDED — D3]** New evaluated path: `verification.generator.drift` and `verification.generator.diffusion_matrix`, added to `ANSWER_KEY_PATHS`. The prose fields stay in `GUIDANCE_PATHS` untouched, and `observables_payload` falls back to them only when they parse — which also delivers [REV 8]. See [§10 D3](#d3--verificationgenerator-a-new-evaluated-path).

**2c. Vectorize the order study over components.** Worst component decides, as `_compare` already
does. Removes the `[:, 0]` in three places.

> **[REV 21a]** Four places — `dynkin._phi_values` is the fourth, and the one with a declared
> multi-component test function behind it. See [REV 11].

**2d. Check the Milstein claim against the noise dimension.** A plan declaring
`scheme_family: milstein` on a problem with `noise_dimension > 1` and no declared Lévy-area
treatment is a cap, not a pass. Cheap once 2a exists.

> **[REV 21] — read `milstein_eligible` rather than re-deriving a weaker version of it.**
> Per [REV 12] the spec already states the answer in four places and nothing reads it, so the
> check is "a plan declares `scheme_family: milstein` on a spec with `milstein_eligible: false`"
> — one field lookup, no inference, and it covers the Lévy-area case *and* any future reason a
> formulator records. Deriving it from `noise_dimension > 1` instead misses nothing on this set but
> adds a second source of truth for a fact the spec already carries.
>
> Guard against the inverse false positive: `milstein_eligible: false` on
> `sde_bm_with_drift` / `sde_ornstein_uhlenbeck` / `sde_double_well_langevin` means `dg/dX = 0`, so
> a "Milstein" plan there is bit-identical to Euler–Maruyama and is **correct**. The cap belongs on
> the Lévy-area reason only, which means the spec needs to say which reason applies — a small
> addition to Phase 6's schema work.

### Phase 3 — add the two missing classes of evidence (Findings 3, 4)

**3a. Grade `cross_moment`.** A per-pair covariance and correlation check with its own standard
error, under the same three-way outcome as every other SDE comparison, wired into `_compare` and
reported in `m["summary"]`. The spec surface already exists and already carries a `tol`. Flip
`gate` to `true` on `sde_gbm_2d_high_corr`, where the spec's own note says it is the only detector.

> **[REV 22] — the two sentences fix it two different ways, and the second one is a no-op.**
> The first sentence is right and **sufficient**: wired into `_compare`, a correlation miss makes
> `moments_ok` `False`, `rubric_sde_analytic` falls through to the `variance_ok` row at **8**, and
> the honest plan keeps its 10 — which is acceptance criterion 4, exactly.
>
> The second sentence routes the same failure through `constraints_ok`/`gate_failed` instead, which
> caps at 4 by rubric and 3 by ceiling, and double-counts a check the first sentence already
> grades. Worse, it changes nothing as written:
> [`run_declared`](../verifylib/kernel/invariants.py#L731) reads only
> `verification.invariants` and `verification.constraints`, and evaluates entries through an
> `expr` key. `cross_moment` is in neither list and has no `expr`, so `cross_moment.gate` is read
> by **no code path at all** — flipping it is inert until something reads it.
>
> Drop the gate flip. Keep `gate: false` and let the three-way moment outcome carry it; a cross
> moment outside tolerance is an accuracy miss, not a structural violation, and the gate vocabulary
> is reserved for negative paths and support breaches.

**3b. Add distributional evidence, in two tiers.**

* *Cheap, and do this first:* the third and fourth standardized central moments against the
  reference or surrogate, with standard errors, same three-way outcome. Free — the terminal paths
  are already in memory, and the fourth central moment is already computed for `se_variance`.
* *Richer, where the law is known:* a `terminal_law` spec field (`normal`, `lognormal`,
  `noncentral_chi2`, with parameters as expressions) and a one-sample Kolmogorov–Smirnov statistic
  against it. This is the check that separates "right moments" from "right process", and it applies
  to OU, GBM, Black–Scholes, BM-with-drift, linear-additive and CIR — 11 of the 23 problems.

> **[REV 23] — the KS tier as specified rejects correct solvers. Measured. And the count is low.**
> One-sample KS against the exact terminal law, correct Euler–Maruyama, n = 50 000:
>
> | Process | dt | KS D | crit (α = 0.05) | Verdict |
> |---|---|---|---|---|
> | OU | 1e-3 | 0.00477 | 0.00607 | pass, by a hair |
> | OU | **1e-2** | 0.00663 | 0.00607 | **reject** |
> | GBM | 1e-3 | 0.00580 | 0.00607 | pass, by a hair |
> | GBM | **1e-2** | 0.00678 | 0.00607 | **reject** |
>
> At 50 000 paths the bar is 0.006 in sup-norm while a weak-order-1 scheme's CDF bias is `O(dt)`,
> so the test resolves the discretization it is not supposed to be judging — and the two passes
> above are inside one seed's worth of noise of the bar. The live specs run at `dt = 0.01` and
> `0.005` with 50 000 paths, i.e. exactly this regime; `sde_cir_feller_violated` at `dt = 0.005`
> against a noncentral χ², where a full-truncation Euler has real mass near zero, is the worst
> case. A goodness-of-fit test at this `n` is a *resolution* instrument, not a correctness one.
>
> Pick one of: **(a)** standardize the sample before the test so it judges shape only — which is
> what Finding 4's three defect classes actually corrupt, and it kills the `O(dt)` moment bias
> that drives the rejection; **(b)** require `D < max(crit, C * dt**p)` with `C` calibrated on an
> honest solver per family, the same way the order floors were tuned; or **(c)** ship it as a
> reported diagnostic with no verdict. The cheap tier has none of this exposure and should ship on
> its own, ahead of this.
>
> The **count is ~16, not 11**: add both oscillators (linear + additive noise → Gaussian),
> `sde_exponential_ou` (exactly lognormal), both `sde_gbm_2d_*` (lognormal marginals), and
> `sde_quintic_random_ic` (an exact monotone map of a Gaussian, so the terminal law is closed-form
> even though the *variance* needs quadrature). That strengthens the phase's case.

> **[REV 41] — the cheap tier is free to *compute* and has nothing to grade against. [M]** "Free —
> the terminal paths are already in memory" is true of the sample side and silent about the
> reference side. Censused across all seven live specs, `analytic_moments` carries exactly:
>
> ```
> ['derivation', 'has_analytic_solution', 'mean_X', 'mean_Y', 'mean_expression',
>  'note', 'state_dimension', 'variance_X', 'variance_Y', 'variance_expression']
> ```
>
> No third or fourth moment anywhere — and the §15-17 surrogate ladder returns a mean and a variance
> only, so on Path B there is no reference even in principle. The one formulator who thought about it
> wrote the target as **prose**, in `sde_ornstein_uhlenbeck`'s note: *"for additive-noise OU the
> exact terminal law is Gaussian; sample skewness ~ 0 and excess kurtosis ~ 0 within MC error is a
> free extra check the closed-form moments do not cover."*
>
> Two corrections follow. [REV 23]'s "the cheap tier has none of this exposure and should ship on
> its own, ahead of this" is right about false rejections and wrong that it can *grade*: until a
> `terminal_law` exists — which is the KS tier's own prerequisite — tier 1 can only report. And
> **D4 leans on it**: "the first two moments keep their own check — that is 3a and the cheap tier of
> 3b" is right for moments 1 and 2, but the cheap tier's own moments 3 and 4 have no declared
> target. See [§11 D11](#d11--the-cheap-tier-ships-report-only-until-a-law-is-declared).

> **[DECIDED — D4]** Standardize the sample (and the reference law) before testing, so the KS statistic measures **shape** only. The first two moments keep their own checks in 3a and 3b's cheap tier. See [§10 D4](#d4--the-ks-tier-standardizes-before-testing).

### Phase 4 — honesty about Monte Carlo noise (Findings 7, 8)

**4a. Seed replication.** `seed_replicates` in `evaluation_thresholds` (default 1, so nothing
changes until a spec opts in), applied to the Dynkin check and the moment comparison, with the
across-seed spread reported in the metrics. This is the fix `dynkin.py`'s own docstring asks for.

> **[REV 42] — 4a has no combination rule, and 4a's cost is never multiplied through 1b.**
> 1. **No rule.** "Applied to the Dynkin check and the moment comparison, with the across-seed
>    spread reported" produces *k* z values and no statement of how they become one verdict. Against
>    the sd ≈ 0.33–0.37 that [REV 35b] measured and a bar at 2.43, the choice is not cosmetic:
>    all-must-pass is **stricter** than a single seed, the median is roughly neutral, and pooling the
>    seeds is ~`sqrt(k)` more powerful and would fail plans that pass today. Since
>    [§7](#7-revised-shipping-order-rev-2)'s step 5 exists only to make step 8's 1b safe, this rule
>    *is* 1b's false-failure rate. See
>    [§11 D9](#d9--the-median-verdict-and-replicate-only-near-the-bar).
> 2. **No cost.** [REV 30] prices 1b at ~7x a graded solve; [REV 24] and [REV 35b] then ask for
>    `seed_replicates >= 3` wherever Dynkin gates, which after 1b is everywhere. Unconditionally
>    that is **~21x a graded solve per certifying Path-A plan**, against §9b's measured
>    `ginzburg_landau` solves of ~10 minutes each — the 33.5-minute 12-plan baseline becomes hours,
>    and §7's table has a risk column and no cost column. D9's second half is what keeps this
>    affordable.

**4b. Resolve the §22 shared-seed divergence.** **[D]** Either change `_compare_sde` to independent
seeds and keep §22, or amend §22 to state the CRN rationale and keep the code. Recommend the
latter for *detecting* disagreement and the former for *earning* the circularity break — i.e. run
both, and let only the independent-seed comparison grant `agreement_is_break`.

> **[REV 24] — Phase 4a belongs before Phase 1b, and 4b should price its extra solves.**
> 4a is written as an opt-in that changes nothing until a spec asks (`seed_replicates` default 1).
> That is the right default *after* Phase 1b, and the wrong one before it: 1b takes a check whose
> own docstring records a surviving false-failure rate and makes it a hard cap at 3 on 5 of 7 live
> Path-A specs (21 of 23 after Phase 6). Either default `seed_replicates` to 3 on Path A when
> Dynkin is the thing being gated on, or keep 1b diagnostic-only until the replication exists. The
> `sde_random_ic_honest` canary in §4a is the same argument in a single case.
>
> 4b's recommendation (run both comparisons) is sound but doubles `compare`'s cost from 2 solves to
> 4 — worth one sentence, since `compare` is already the slowest CLI path on an SDE.
> **[REV 35] — Finding 8 is not hypothetical: it is costing two live plans 6 points each, right
> now, on Path B.** Measured in [§9b](#9b-the-6-baseline--measured-2026-10-04):
>
> ```
> ginzburg_landau_s6/1-tamed-euler-maruyama  dynkin_ok=False  z=3.242  bar=2.428  score=3
> ginzburg_landau_s6/2-tamed-milstein        dynkin_ok=False  z=3.136  bar=2.428  score=3
> ginzburg_landau_s6/3-log-transform-euler   dynkin_ok=True   z=1.099  bar=2.428  score=9
> ```
>
> Those z values sit exactly at the top of the range `dynkin.py:40-48` reports **for a correct
> solver**: "between 0.2 and 3.2 standard errors of zero… a property of the seed rather than of
> the scheme". All three integrate the same SDE, all three measure strong orders consistent with
> their own schemes (0.452, 0.644, 1.019), and the one that passes is the log transform — where
> the noise becomes additive and the martingale term is best behaved. `STATE.md` records all three
> at 10.
>
> This does not *prove* a false failure — plans 1 and 2 could carry a real bias — and the plan
> should not assert either reading without the test. Two cheap discriminators, both from the
> docstring's own mechanism: **seed spread at fixed path count** (if z crosses the bar across seeds
> for the *same* plan, the verdict is a property of the seed) and **path scaling at fixed seed**
> (a real bias gives `z ~ sqrt(M)`, martingale noise gives `z ~ O(1)`). That is precisely what 4a
> buys, and it is why 4a's default of `seed_replicates: 1` is the wrong default to ship: the
> replication is not an optional refinement for specs that opt in, it is the thing that tells a
> failure from a seed. Make it ≥ 3 wherever Dynkin is load-bearing, which after 1b is everywhere.
> **[REV 35a] — the test above was run, and it does not support [REV 35]'s reading. Corrected.**
> [REV 35] proposed two discriminators; both were run on 2026-10-04 and the conclusion goes the
> other way, so the claim "costing two live plans 6 points each" is **withdrawn** as unsupported.
> Leaving it above because the *sequencing* argument it makes survives on other grounds, and
> because the correction is the more useful record.
>
> **Path scaling, plan 1, seed 42** — a real bias predicts `z ~ sqrt(M)`, martingale noise
> predicts `z ~ O(1)`:
>
> ```
> M=12500  z=1.406      M=25000  z=1.686      M=50000  z=3.242
> growth over 4x paths: 2.306   (bias predicts 2.000, pure noise predicts 1.000)
> ```
>
> That is close to `sqrt(M)` and far from flat, so the residual on plans 1 and 2 looks like a
> **real, small generator inconsistency** that only clears the bar once the standard error has
> shrunk enough to resolve it — not a seed artifact. Dynkin is probably doing its job there, and on
> a problem whose surrogate has died (Finding 15) it is the only check that could.
>
> Two honest limits on that conclusion. The middle point undershoots its own prediction (1.686
> against 1.988), so with one seed per path count the trend carries an O(1) seed term and three
> points do not pin an exponent. And the **seed spread was run at M=12500, where all nine runs
> passed** (z = 0.42–1.89, no crossing) — so it tested nothing: the bias is below the bar at that
> path count by construction. The decisive experiment is the seed spread **at M=50000**, ~80 minutes
> for 3 seeds x 2 plans, and it has not been run.
>
> What survives for Phase 4a: the replication is still the instrument that would have answered this
> in one step instead of two, and `dynkin.py`'s documented 0.2–3.2 range still means a bar at 2.43
> cannot distinguish a 3.24 from noise *without* it. The argument for defaulting
> `seed_replicates >= 3` wherever Dynkin gates is unchanged. What is withdrawn is the claim that
> these two specific plans are being wronged.
>
> **[REV 36] — the strong-order estimate on this problem is not a measurement, and `orders_ok` says
> `True` anyway. [M]** The same 10 runs, reading `observed_order`:
>
> ```
> plan 1 (tamed EM):   0.452 (M=50k)  0.547 (M=25k)  1.382 / 1.308 / 3.069 (M=12.5k, seeds 42/7/101)
> plan 2 (tamed Mil):  0.644 (M=50k)                 1.208 / 1.317 / 2.856 (same seeds)
> plan 3 (log transf): 1.019 (M=50k)                 1.052 / 1.108 / 1.156 (same seeds)   <- tight
> ```
>
> Plan 1's strong order spans **0.452 to 3.069, a factor of 6.8, on one plan against one spec**, and
> `orders_ok` came back `True` in **all ten runs** — including at 3.069 against a declared floor of
> 0.5 — because the family is `inferred`, so `guards` applies the one-sided band and accepts
> anything above `min(candidates) - 0.4 = 0.1`. Plan 3, where the log transform makes the noise
> additive, is tight across all four configurations. So the estimator is well-behaved on the easy
> plan and junk on the two hard ones, and the kernel reports a pass either way.
>
> This generalises [REV 16](c). A roundoff floor is not enough: the strong order needs a
> **stability guard across the ladder**, the direct analogue of `richardson_stable` for the moments
> (`rubric_sde` already reads that for the Richardson moment and has no counterpart for the order).
> Without one, 1c gates on a number that moves by 6.8x with the seed, and [REV 34]'s ranking
> component would score plans on it. Add to Phase 1c, ahead of (a) and (b):
> *an order estimate that is not stable across the ladder is `unavailable`, not a pass.*
> **[REV 35b] — closed. The seed spread was re-run at the full path count and the failure is real.**
> 11 runs at `M = 50000`, the configuration these plans are graded at
> ([`benchmark/results/sde_dynkin_seed_study_2026-10-04.json`](../benchmark/results/sde_dynkin_seed_study_2026-10-04.json)):
>
> | Plan | seeds 42 / 7 / 101 / 2024 | mean | sd | failed |
> |---|---|---|---|---|
> | 1 tamed EM | 3.242 / 3.365 / 3.487 / 2.727 | 3.205 | 0.334 | **4 / 4** |
> | 2 tamed Milstein | 3.136 / 3.223 / 3.769 / 2.897 | 3.256 | 0.368 | **4 / 4** |
> | 3 log transform | 1.099 / 1.603 / 1.253 / — | 1.318 | 0.258 | 0 / 3 |
>
> **No crossing of the bar in either direction, in any run.** The two groups separate cleanly —
> worst passing z is 1.603, best failing z is 2.727, and the bar at 2.428 sits in the gap with no
> overlap. The verdict is a property of the plan, not of the seed, so:
>
> * **[REV 35]'s claim is definitively withdrawn.** Plans 1 and 2 are not being wronged; Dynkin is
>   detecting a real generator inconsistency in the tamed schemes that the log transform does not
>   have. [REV 35a]'s provisional reading was right and is now properly powered.
> * **The sequencing argument for 4a is weakened, not refuted.** sd ≈ 0.33–0.37 against a bar at
>   2.43 means a *marginal* plan — one whose true z sits near 2.4 — would still flip with the seed,
>   which is the case 4a protects. These two plans simply are not marginal. So keep 4a before 1b,
>   but on the honest grounds that the check's resolution is unmeasured, not on a claim that it is
>   currently misfiring.
> * **Finding 15's fix does not change these two scores** — see the correction appended to that
>   finding.
>
> **[REV 36a] — confirmed, and worse at full path count than at reduced.** Same 11 runs,
> `observed_order`:
>
> ```
> plan 1 (tamed EM):    0.452 / 1.357 / 2.726 / 1.78    spread 6.0x
> plan 2 (tamed Mil):   0.644 / 1.337 / 0.220 / 2.73    spread 12.4x
> plan 3 (log transf):  1.019 / 1.136 / 1.145 /  —      spread 1.1x
> ```
>
> `orders_ok` is `True` in **all 11 runs**. Plan 2's estimate spans **0.220 to 2.730 — a factor of
> 12.4 — on one plan against one spec at 50 000 paths**, and both ends pass, because the family is
> `inferred` so the one-sided bar is `min(candidates) - 0.4 = 0.1`. An estimator that returns 0.22
> and 2.73 for the same scheme is not measuring the strong order, and the kernel calls it a pass
> either way. So [REV 36] understated it: this is not a reduced-path artifact.
>
> Note the pattern, because it points at the fix. Plan 3 — the log transform, where the change of
> variable makes the noise **additive** — is tight to 1.1x across every configuration. Plans 1 and 2
> keep the multiplicative `sigma*X dW` at `sigma = 6`, which makes the pathwise differences
> `|X^dt - X^(dt/2)|` violently heavy-tailed, so `s0` and `s1` are sample means with enormous
> standard errors and `log2(s0/s1)` inherits all of it. `strong_order`
> ([`crn.py:46-53`](../verifylib/kernel/sde/crn.py#L46)) returns the ratio with **no error bar at
> all**, while `weak_order` right below it does compute `se` and discards anything at the noise
> floor. **The fix is to give `strong_order` the error bar `weak_order` already has**: bootstrap or
> jackknife `se(s0)` and `se(s1)`, propagate to the order, and return `unavailable` when the
> implied uncertainty is comparable to the band width. That subsumes [REV 16](c)'s roundoff floor —
> an exactly-integrating scheme is the limiting case where both means are at machine epsilon and the
> uncertainty is total — so implement one mechanism, not two.

> **[REV 37] — step 0's error bar, implemented as [REV 36a] specifies it, converts a false *pass*
> into a false *failure*. [R]** This is the one place where Rev 2's own recommendation is unsafe as
> written, and [§7](#7-revised-shipping-order-rev-2) prices it at *"Risk to correct work: none"*.
> [`crn.guards`](../verifylib/kernel/sde/crn.py#L149) reads:
>
> ```python
> finite = strong is not None and np.isfinite(strong)
> s_ok   = finite and shrinking and abs(strong - float(expected_strong)) < 0.4
> w_ok   = weak is None or (...)        # <- the weak order's None is forgiven
> return {..., "orders_ok": bool(s_ok)}
> ```
>
> So a `None` out of `strong_order` lands as **`orders_ok: False` — a verdict of failure**, not an
> abstention. The forgiveness one line below is the *weak* order's, and nothing mirrors it for the
> strong order. [REV 36a] prescribes "return `unavailable` when the implied uncertainty is
> comparable to the band width" as a change to `strong_order` alone; implemented that way it
> hard-fails exactly the heavy-tailed plans it was written for, and under **D1** that is an 8 on
> Path A and `rubric_sde`'s row on Path B — or a **3**, per [REV 38].
>
> **The same bug is live today**, one step worse than [REV 16](c) says. `strong_order`'s existing
> guard returns `None` on `s0 <= 0.0` ([`crn.py:52`](../verifylib/kernel/sde/crn.py#L52)), so an
> exactly-integrating scheme is *failed* rather than excused. [REV 16](c) read that as a missing
> guard; it is a guard that reports the wrong outcome.
>
> **Step 0 therefore needs three touches, not one:** `strong_order` returns its error bar, `guards`
> gives `orders_ok` a third value, and `_orders` propagates it. The plumbing is already half-built —
> [`_orders`](../verifylib/kernel/sde/driver.py#L406) already returns the literal `"unavailable"`
> for a crashed ladder level and for fewer than three levels, and `SKIPS` already contains it, so
> `passed()` and `failed()` both refuse it. `guards` is the only thing that cannot reach it.
> See [§11 D8](#d8--abstain-and-add-a-second-statistic-abstention-costs-nothing).

> **[DECIDED — D6]** `compare` runs **both** seedings; only the independent-seed comparison may set `cross_plan_agreement`. Cost is four solves per compare instead of two, accepted. §22 is amended in the same commit. See [§10 D6](#d6--run-both-seedings-only-the-independent-one-earns-the-break).

### Phase 5 — the promised checks that do not exist (Findings 9, 10, 12)

**5a. Implement `stability_check` in the SDE driver** as a gated structural check: `finite` over
`terminal_paths`, and `domain` as `max |X| < abs_bound`. Fail is a gate violation, which already
caps at 4.

> **[REV 25] — the semantics contradict the manual this cites, and the structured input does not
> exist yet.**
> 1. **Semantics.** `project_manual.md:304` says the moment tolerances *do not apply* and
>    `stability_check` **is** the criterion; `benchmark/verify.py:422` implements it that way. 5a
>    proposes it as a gate layered *on top* of moment scoring. Those are different designs and the
>    plan has to choose; `sde_ginzburg_landau_s6`'s spec note shows a formulator already reasoning
>    about the difference and routing around it.
> 2. **Input.** Per [REV 9] no live spec carries a structured `stability_check`, and the one stale
>    instance is prose. 5a without a schema rule and a formulator fix grades nothing. Add: a
>    non-`null`, non-`dict` `stability_check` is a **schema error**, not a silently ignored field —
>    otherwise the prose form fails open, which is the exact shape of Finding 9.
> 3. Do **not** encode `project_manual.md:310` as a validity rule; it hard-fails two live specs.
>    See [REV 9].

> **[DECIDED — D5]** `stability_check` **replaces** moment scoring when non-null, matching `project_manual.md:304` and the oracle. A non-`dict` value is a schema error, and `project_manual.md:310` is corrected in the same commit. See [§10 D5](#d5--stability_check-replaces-moment-scoring).

**5b. Add the clip scan** to `leakage.py` or `gate.py`: an AST check for `np.maximum(X, 0)` /
`np.clip` applied to the state variable at the tail of the step loop. Keep it narrow, per
`leakage.py`'s own rule about the 124-of-137 false-positive proposal. If it is not implemented,
delete the claim from `solver-sde.md` — a deterrent that does not exist is worse than no deterrent,
because the agents are told it does.

> **[REV 26] — as specified this false-positives on CIR, which is one of the seven live problems.**
> [`agents/solver-sde.md:80`](../agents/solver-sde.md#L80) *instructs* the solver to apply
> implementation notes "e.g. `np.maximum(X, 0.0)` before `sqrt` for CIR" — and
> `sde_cir_feller_violated` is live, with two plans. A scan for "`np.maximum(X, 0)` / `np.clip`
> applied to the state variable" matches the endorsed `np.sqrt(np.maximum(X, 0))` inside the
> diffusion term as readily as the defect.
>
> Narrow it to **assignment to the state name** — `X = np.maximum(X, 0)` / `X[:] = ...` /
> `np.clip(X, 0, None, out=X)` as a statement in the step loop's body — and never match a clip used
> as a sub-expression of another call. That distinction is the whole check: full-truncation Euler
> clips *inside* the coefficient and keeps the scheme honest; clipping the *state* after the update
> is what hides the defect. Per [REV 10], if 5b is dropped the deletion is from
> `solver-sde.md:99` only; §21 needs no change.

**5c. Compare the solver's reported moments against the kernel's.** One comparison in the driver,
reported as a diagnostic, never a gate. Catches a `ddof` error and a mis-transcribed Results
section.

### Phase 6 — spec coverage (Finding 13)

Bring the 16 stale SDE specs up to the current schema: `verification.expected_strong_order` at
minimum, then `moment_ode` where it closes, `dynkin_test_functions`, and the invariants each
family warrants. Without this, Phases 1–3 improve the kernel and change nothing about 16 of the 23
problems, because the checks they add are all spec-driven.

> **[REV 27] — this is a prerequisite of 1c, not a follow-up to it.** The last sentence is right
> and its consequence is stronger than "changes nothing": with the 16 stale specs defaulting to
> `expected_strong_order: 0.5`, 1c does not merely fail to help them, it **actively false-fails
> six of them** (see [REV 16]). Promote `verification.expected_strong_order` on every spec to its
> own early step — it is a one-line-per-spec change and it is the gate on Phase 1c — and leave
> `moment_ode`, `dynkin_test_functions` and the invariants as the larger follow-on. Finding 5 and
> Finding 6's measurements are also both taken on stale specs (`sde_quintic_random_ic`), so their
> canaries cannot be reproduced end to end until this lands.

---

## 4. Testing

The canary set is the weakest part of the SDE story, and its weakness has the same shape as the
findings: four solvers, all scalar additive-noise OU, all run through Path B. The untested
configurations are exactly where Findings 1–5 live.

### 4a. Canaries to add

Each one is pinned to a finding and to the check that must catch it.

| Canary | Defect | Must be caught by | Today |
|---|---|---|---|
| `sde_fabricated_sampler.py` | draws from the exact law; never integrates | Dynkin on Path A (1b) | **10, certified** |
| `sde_two_component.py` | honest, 2 state components, 2 noise channels | the CRN ladder must *run* | **IndexError** |
| `sde_drops_correlation.py` | independent increments on a ρ=0.95 spec | the `cross_moment` check (3a) | **10** |
| `sde_uniform_increments.py` | Rademacher increments rescaled to σ²=1 | the distributional check (3b) | nothing sees it |
| `sde_milstein_post_step.py` | correction evaluated at `X_{n+1}` | the strong-order gate (1c) | only if the spec declares ≥ 1.0 |
| `sde_clips_positivity.py` | bare `np.maximum(X, 0)` after the step | the clip scan (5b) | nothing sees it |
| `sde_random_ic_honest.py` | correct solver, random `X(0)` | Dynkin must **not** fire (Finding 5) | false-fails at 3 |

The last row is the one to write first. A canary that asserts a correct solver is *not* failed is
the kind this repo has needed repeatedly — I9, I10 and I15 in
[plan-no-closed-form.md](plan-no-closed-form.md) are all false-failures on correct work, and the
reason given there holds here: *a false hard-fail on correct work is invisible as a false
positive.*

> **[REV 28] — three rows need restating, and one row is missing.**
> * `sde_fabricated_sampler` — "Must be caught by: Dynkin on Path A (1b)" is right, but the cap
>   arrives through `ceiling()`/`gates_clean`, not through 1a's rubric row, and the reason text
>   names D1 (see [REV 5]). Assert the **score and the provenance**, and assert the reason only
>   after 1b fixes the wording — otherwise the canary encodes the bug.
> * `sde_two_component` — "Today: IndexError" verified, and `m=2` already fixes it, so this canary
>   should also assert the recovered correlation (0.945 measured against the spec's 0.9478), not
>   only that the ladder runs. It is the cheapest regression guard for 2a.
> * `sde_drops_correlation` — "≤ 8 … the `cross_moment` check (3a)" is achievable **without** the
>   gate flip; see [REV 22]. Pin the expected score to 8 and the honest plan to 10.
> * **Missing, and it is the counterpart of the last row:** a canary for a correct
>   additive-noise Euler–Maruyama on a spec whose `expected_strong_order` is the kernel default
>   0.5, asserting that 1c does **not** fail it ([REV 16]). `sde_bm_standard` is the extreme case —
>   an exact scheme whose CRN differences are pure roundoff — and it is the one that must not
>   regress.

> **[REV 51] — the `sde_milstein_post_step` row is wrong, measured.** "Must be caught by: the
> strong-order gate (1c)" assumes evaluating the Milstein correction at `X_{n+1}` instead of `X_n`
> costs the scheme its order. It does not. For GBM the correction is `0.5·σ²·X·(dW² − dt)`, so the
> post-step version differs from the correct one by `0.5·σ²·(X_{n+1} − X_n)·(dW² − dt) ≈
> 0.5·σ³·X·dW·(dW² − dt)`, which is `O(dt^{3/2})` per step **with zero mean**; summing `T/dt` of
> them in L² gives `O(dt)`, the same order as the scheme itself. The rate is unchanged and only the
> error constant moves, roughly doubling. Measured against the exact GBM solution, M = 40 000, four
> `dt` levels:
>
> | canary | observed strong order |
> |---|---|
> | `sde_milstein_honest` | 0.998 |
> | **`sde_milstein_post_step`** | **0.998** |
> | `sde_milstein_no_correction` | 0.504 |
> | wrong-sign correction | 0.505 |
>
> So the canary is kept and **inverted**: it asserts the order study does *not* fail it
> (`test_the_post_step_correction_is_not_failed_by_the_order_study`). That is worth more than the
> row as written, because it is what stops a future tightening of the band from costing the correct
> Milstein plan beside it. The two rows that *do* separate at 1c are the missing correction and the
> wrong sign, and both are canaries now.

> **[REV 52] — the `sde_uniform_increments` row is wrong on additive noise, measured.** "Must be
> caught by: the distributional check (3b)" is true only where the noise is multiplicative. On a
> linear additive-noise SDE the terminal state is a **sum** of the increments, so the CLT makes its
> law Gaussian to within the test's resolution however the increments were drawn. Measured on the OU
> canary spec, M = 40 000, Rademacher increments rescaled to σ² = 1 against a Gaussian-increment
> control, bar = 1.3581/√M = 0.00679:
>
> | `Nt` | `dt` | KS, Rademacher | KS, Gaussian control | mean err | variance err |
> |---|---|---|---|---|---|
> | 1000 | 0.001 | 0.00373 | 0.00295 | 0.06% | 0.89% |
> | 100 | 0.01 | 0.00304 | 0.00305 | 1.12% | 1.74% |
> | 25 | 0.04 | 0.00528 | 0.00288 | 4.84% | 3.85% |
> | 10 | 0.1 | **0.01264** | 0.00288 | 11.98% | 9.96% |
> | 4 | 0.25 | **0.09579** | 0.00294 | 31.31% | 27.31% |
>
> At the live specs' `dt` it is **indistinguishable from the control**. It clears the bar only from
> `dt ≥ 0.1`, where the mean error is already 12% against a 5% threshold — so there is no `dt` at
> which the KS tier is the check that catches it. A check that rejected the top two rows would be
> wrong, and the canary asserts the pass (`terminal_law_ok is True`).
>
> The defect is real; the terminal law of an additive-noise SDE is not where it shows.
> **Multiplicative noise is what keeps that window open**, because the terminal state is then a
> product whose factors carry `dW²` explicitly. The same Rademacher increments on GBM:
>
> | check | correct Milstein | Rademacher | bar | verdict |
> |---|---|---|---|---|
> | first two moments | pass | pass | 5% / 10% | blind |
> | `observed_order` | 0.980 | 0.980 | 1.0 ± 0.4 | blind |
> | **standardized KS** | **0.0045** | **0.0197** | **0.0068** | **caught** |
>
> Both canaries ship, on the two noise structures, and the pair is the assertion. See [REV 53] for
> why the order study's blindness there is structural rather than incidental.

### 4b. Two structural additions worth more than any single canary

**A Path-A fixture suite.** Every existing SDE canary runs through
`sde_spec_no_closed_form.json`. Run all of them through `sde_spec.json` as well and assert the
score. That alone would have caught Finding 1 on the day `rubric_sde_analytic` was written.

**A rubric-parity test.** Assert that for each evidence field, the PDE and SDE analytic rubrics
either both read it or both carry a recorded reason why not. The divergence in Finding 1 was silent
for the life of the file, and a table-driven test is the only thing that keeps the two rubrics from
drifting again.

> **[REV 29] — add a spec-field-coverage test; it catches six of the fourteen findings at once.**
> The findings share a shape the document never names: **a spec field that is declared and never
> read.** `cross_moment` (F3), `noise_dimension` (F2, per [REV 6]), `*_value_at_T` (F6),
> `stability_check` (F9), `empirical_mean`/`empirical_variance` (F12) and `milstein_eligible`
> (F14) are all instances — six of fourteen. A table-driven test that every key appearing in any
> `workspace/*/problem_spec.json` has either a kernel consumer or a recorded reason it has none
> catches all six and the next one. That is a stronger structural addition than rubric parity,
> which only covers Finding 1; keep both.
>
> One correction to the Path-A fixture item: per [REV 3] it already exists in part
> (`test_kernel_canaries.py:118-133`), so this is "extend the fixture to the defective canaries",
> and the full-path honest test there is a guard Phases 1b and 1c must not break.
>
> Also worth asserting in the parity test: that the two rubrics agree on which **evidence fields**
> they read, not on score values — see [REV 32] on criterion 8.

> **[REV 46] — a seventh instance of the pattern, in the other direction. [R]**
> [REV 29] names six declared-and-never-read **spec** fields. The same shape exists on the metrics
> side: `guards` computes `weak_order_out_of_band`
> ([`crn.py:165`](../verifylib/kernel/sde/crn.py#L165)) and **no code reads it** — `grep` over
> `verifylib/` returns the definition and nothing else. Its own docstring says an out-of-band weak
> order is *"flagged rather than fatal"*; it is flagged to nobody, so the weak order is entirely
> decorative today and `orders_ok` gates on `s_ok` alone
> ([`crn.py:172`](../verifylib/kernel/sde/crn.py#L172)). Add the second direction to the coverage
> test — a metrics key with no consumer and no recorded reason — and know it before the open SDE
> weak-error work builds on it.

### 4c. Contract round-trip

`test_kernel_sandbox.py:126` only ever passes `{"m": None}`. Add a property test that
`ladder_spec(..., m=k)` round-trips through the sandbox at the shape `solver-sde.md` promises, for
`k` in {1, 2, 3}. That is the test whose absence let Finding 2 survive.

### 4d. Cost

The Path-B suite is 48 tests in 5m35s because the canaries run real CRN and Dynkin ladders. A
Path-A suite of the same canaries roughly doubles it unless the new canaries run at reduced
`num_paths` — which they can: the fabrication, correlation and uniform-increment canaries all fail
by orders of magnitude rather than marginally, so a few thousand paths separates them. Reserve the
50 000-path configuration for the honest controls, where the margin is the thing being measured.

> **[REV 30] — 1b and 1c are the wrong way round, and the pipeline cost is missing.**
> *1c is cheap*: `dt0` defaults to `max(dt, 4*dt)` and the ladder runs at `dt0, dt0/2, dt0/4` at
> `conv_num_paths` (20 000 by default, against a graded 50 000), so well under one graded solve.
> *1b is the expensive one*: the Dynkin ladder refines **downward** from the graded `dt` at full
> `num_paths`, so ~7× a graded solve plus the observables accumulation, and `dynkin.py` records
> ~30 s per plan. §3's 1c note ("one extra ladder per certifying plan, which §24 Rule 2 already
> budgets for") prices the cheap phase and not the expensive one — and §24 Rule 2 budgets the
> ladder only when `expected_strong_order >= 1.0`, which is the condition 1c removes.
>
> The *suite* estimate here is sound. What is missing is the **pipeline** estimate: 1b adds a
> ~7× solve to every certifying Path-A plan, across 5 live specs now and 21 problems after Phase 6.

---

## 5. What this does not fix

* **A mis-transcribed equation still passes everything.** Dynkin builds its generator from the
  spec's own `drift_expression`, so a drift wrong in `problem_spec.json` fools the moments, the
  surrogate and Dynkin alike. This is the same limitation D1, MMS and the degenerate limit carry on
  the PDE side, and `agreement.py` already says so; the requirements ledger is the only thing
  reaching outside the spec. Nothing in this plan changes that, and `REPORT.md` must not imply
  otherwise.
* **Phase 3b's KS test needs a declared law.** Where the terminal law is genuinely unknown — every
  Path-B problem — the moment-based tier (3a's first bullet) is all that is available.
* **Multi-dimensional Dynkin (2b) is real work.** It needs a machine-readable diffusion matrix
  across six specs, and the central-difference Hessian in the sandbox child grows from 3 evaluations
  to `O(d²)`. 2a is what delivers most of Phase 2's value.

> **[REV 31] — two more belong on this list, and one entry needs a correction.**
> * **The KS tier needs calibration, not just a declared law.** The second bullet above frames the
>   limit as "where the terminal law is unknown". Per [REV 23] the sharper limit is that where the
>   law *is* known, a KS test at 50 000 paths resolves the scheme's own `O(dt)` bias and rejects
>   correct solvers. A declared `terminal_law` is necessary and not sufficient.
> * **Dynkin's false-failure rate is reduced, not removed.** 4a converts a tolerated rate into a
>   measured one; it does not make the check exact. With 1b routing Dynkin into `gates_clean`, the
>   residual rate now shows up as a hard 3 rather than a two-point cap, and `REPORT.md` should say
>   so wherever it quotes `dynkin_z`.
> * **"across six specs" is four** — see [REV 7]. And the Hessian growth is `O(d²)` in *state*
>   dimension (2 for every one of them), not in the noise dimension, so the sandbox cost is 4
>   evaluations per test function rather than 3, not the blow-up the phrasing implies. m13's 13
>   noise channels enter only through `G G^T`, which is a 2×2 matrix. That makes 2b meaningfully
>   cheaper than this bullet suggests — the real cost is the machine-readable diffusion matrix and
>   [REV 20]'s schema question, not the arithmetic.

> **[REV 53] — the increment law is structurally invisible to every check but one.** This belongs on
> §5's list and is the sharpest limitation the implementation surfaced. **Neither ladder can ever
> see the solver's own increments**, because both *supply* them: that is what §18's common random
> numbers are — the sandbox child rebuilds `dW` from a seed and passes it in through the solver's
> `dW=` hook. So the CRN order study measures the plan's *scheme* driven by Gaussian noise, and
> reads the same `observed_order` for a Rademacher solver as for the correct one (0.980 against
> 0.980, [REV 52]). Dynkin is blind for a second and independent reason: the generator identity
> holds in expectation for **any** increment law with the right first two moments, which is exactly
> what a σ²-rescaled Rademacher draw has.
>
> The **graded run is the only one that uses the solver's own draws**, so its terminal distribution
> is the only window onto them — which is precisely the evidence class Finding 4 says was missing,
> and an argument for 3b the plan did not make. [REV 52] is the limit of that window: it is open on
> multiplicative noise and closed by the CLT on additive noise.
>
> **The consequence to state in `REPORT.md`:** on an additive-noise problem a wrong increment law is
> not detectable by anything in this plan. Not by the moments (they are right by construction once
> σ² is matched), not by the order study or Dynkin (structurally), and not by the KS tier (the CLT).
> Nothing here fixes that, and §4a's row implies otherwise.

---

## 6. Acceptance criteria

1. `sde_fabricated_sampler` scores ≤ 3 on Path A, with a reason naming Dynkin or the order study.
2. A solver that ignores `dW` is capped on Path A exactly as it is on Path B (≤ 8).
3. The CRN ladder completes on all six multi-noise problems; `orders_ok` is a verdict, not
   `"unavailable"`, on each.
4. `sde_drops_correlation` scores ≤ 8 on `sde_gbm_2d_high_corr`; the honest plan still scores 10.
5. `sde_random_ic_honest` is not failed by Dynkin — either it passes, or Dynkin reports
   `unavailable` with the random IC as its stated reason.
6. A spec whose variance expression is non-evaluable but whose `variance_value_at_T` is present
   grades against the latter, and one with neither reports "no reference was evaluable" rather than
   "the variance check fails".
7. Every SDE problem's spec declares `expected_strong_order`.
8. The rubric-parity test passes, and `references/project_manual.md`'s analytic SDE rubric matches
   `rubric_sde_analytic` field for field.
9. The full suite is green, and no currently-passing canary regresses.

> **[REV 32] — five of the nine are unmet or unmeetable as worded. Revised set:**
>
> | # | Rev 1 | Status | Rev 2 wording |
> |---|---|---|---|
> | 1 | reason names Dynkin or the order study | **fails** | *unchanged, and it requires the `ceiling()` reason fix in [REV 15]* — today the binding reason is "D1 stalled with a validated operator" |
> | 2 | "capped exactly as on Path B (≤ 8)" | **inconsistent** | "scores **8**, through the same row and reason `rubric_sde` uses, with the reason naming the measured strong order" ([REV 14]) |
> | 3 | ladder completes on all six | **unmeetable** | "completes on all **four** multi-noise problems, of which only `sde_gbm_2d_high_corr` and `sde_multichannel_stiff_m13` can be measured before Phase 6" ([REV 7]) |
> | 4 | `sde_drops_correlation` ≤ 8 | holds | "scores exactly 8 via `moments_ok`, with `cross_moment.gate` still `false`" ([REV 22]) |
> | 5 | `sde_random_ic_honest` not failed | holds | *unchanged; and it must be asserted before 1b ships, not after* ([REV 24]) |
> | 6 | grades against `variance_value_at_T` | **half-met** | add: "and reports `provenance: analytic` at tier `A`, not `analytic_unvalidated` at `A_unavailable`" ([REV 17]) |
> | 7 | every spec declares `expected_strong_order` | holds | *unchanged, and it is a **precondition** of 1c, not an outcome* ([REV 27]) |
> | 8 | rubrics match "field for field" | **unmeetable** | `project_manual.md` gives bands (7–9, 4–5) and the code gives single values widened by `BANDS`; re-word as "the two analytic rubrics read the same **evidence fields**, or carry a recorded reason why not" ([REV 29]) |
> | 9 | suite green, no canary regresses | holds | add explicitly: "`test_the_same_sde_solver_at_full_path_count_scores_ten` still scores 10 with 1b and 1c both enabled" — measured, it does ([REV 3]) |
>
> Two to add:
> 10. A correct additive-noise Euler–Maruyama plan on a spec with no declared
>     `expected_strong_order` is **not** failed by the order study ([REV 16]).
> 11. No spec field is both declared in `workspace/` and unread by the kernel without a recorded
>     reason ([REV 29]).
> 12. **D1:** the orders-fail row returns 8 with `rubric_sde`'s reason, and the reason names which
>     of the two defects it saw — near-zero order, or below floor.
> 13. **D2:** a plan whose `expected_strong_order` came from `kernel_default` is judged one-sided,
>     selected off `cfg`'s recorded `source`.
> 14. **Step 0 / Finding 15:** `_widen` holds a boundary pinned by a declared support, and
>     `sde_ginzburg_landau_s6/3-log-transform-euler` scores **10, surrogate** again.
> 15. **Step 0 / [REV 36a]:** `strong_order` returns `unavailable` when its own error bar admits
>     more than the band width; re-running the 11-run seed study no longer reports
>     `orders_ok: True` at both 0.220 and 2.730.
> 16. **D4:** a correct Euler–Maruyama at dt = 0.01 and 50 000 paths passes the standardized KS
>     check on OU and GBM — the configuration that rejects it today ([REV 23]).
> 17. **D6:** a shared-seed agreement never sets `cross_plan_agreement`; an independent-seed one
>     does, and `compare` records both.

>
> Six more from Rev 3's reading, resolved in [§11](#11-decisions--resolved-2026-10-14):
> 18. **[REV 37] / D8:** `orders_ok` has a third value. A `strong_order` that abstains reports
>     `"unavailable"` and is **not** `failed` — asserted on an exactly-integrating scheme
>     (`dX = dW`), which today returns `None` and is scored a failure.
> 19. **D8b:** the abstention costs nothing. A plan with `orders_ok: "unavailable"` and every other
>     check clean keeps the score it has today.
> 20. **[REV 40]:** `sde_cir_feller_violated/2-milstein` runs the CRN ladder and reports an order
>     verdict — reachable by the one-line spec edit alone, before 1c.
> 21. **[REV 38] / D10:** a Path-A plan with `moments_ok`, `resolved` and `constraints_ok` passing
>     and `failed(orders_ok)` scores **8**, never 3, through the same conjunction `rubric_sde`'s 8
>     row uses. Asserted as a row-*order* test, not only a score test.
> 22. **[REV 43] / D12:** `ranking.order_margin` scores a correct additive-noise Euler–Maruyama at
>     observed 0.986 against a floor of 0.5 **above 0.9**, not 0.028 — asserted by calling it
>     directly, since no score-based criterion can see it.
> 23. **[REV 45]:** every live `STATE.md` score matches the re-measured baseline, and the baseline
>     file is dated after step 0.

> **[REV 49] — nine of these 23 are worded as a score where an objective field is available, and a
> score is the wrong instrument for a before/after. [M]** A score is a composite:
> `rubric_sde_analytic` or `rubric_sde` proposes one, `ceiling()`, `gates_clean()` and `tier_of()`
> can each lower it, `BANDS` widens the manual's bands into single values, and several inputs are
> **formulator tolerance choices** rather than measurements. Four facts from this document's own
> baseline:
> 1. **Scores drift with no code change.** [§9b](#9b-the-6-baseline--measured-2026-10-04) re-ran all
>    12 live plans and **6 of 12** diverged from `STATE.md` — purely from kernel changes between the
>    recording and the re-run ([REV 47] dates every one). A score-anchored before-column is a moving
>    baseline.
> 2. **One live score turns on a third decimal.** `cir_feller_violated/2-milstein` is a 10 rather
>    than an 8 on a **4.96%** mean error against a **5.00%** tolerance; §9b's own note says "treat
>    the 10 as fragile".
> 3. **Step 0's main fix produces, by design, zero score movement.** Under
>    [D8b](#d8--abstain-and-add-a-second-statistic-abstention-costs-nothing) an abstaining order
>    study costs nothing, so the correctness fix the whole step exists for is invisible to every
>    score.
> 4. **Where the score does not move, three fields do.** On `ginzburg_landau_s6` plans 1 and 2 the
>    score stays 3 across step 0 while `certification.tier` goes `C -> A_prime`,
>    `detail.surrogate_ok.route` goes `null -> "kolmogorov"` and `certification.provenance` goes
>    `self_convergence -> surrogate`. Scored, step 0's `_widen` fix reads as a one-point change on
>    one plan; measured, it restores an entire evidence tier on three.
>
> The criteria stand as written. What changes is what each is *checked* by: the substitute field for
> every one of the 23 is tabulated in
> [expectations-sde-verification-gaps.md §D](expectations-sde-verification-gaps.md#d-the-23-acceptance-criteria-with-an-objective-anchor),
> and the per-plan before-column is its §B. Three criteria — 2, 12 and 21 — cannot be de-scored,
> because they are claims *about* the rubric; assert the row reached and the reason string by calling
> the rubric directly with synthetic evidence, and treat the score as that row's label.
>
> Two criteria are also **unexercisable as worded**, which only a field-level reading shows:
> criterion 13 (a `kernel_default` floor judged one-sided) has no live instance —
> `config.expected_strong_order.source` is `"spec"` on **all 12** plans — so it needs a `guards`
> unit test or a stale spec at step 6; and criterion 20's verdict cannot catch its own motivating
> case, see §K2 of the companion.

Scores that are *expected* to move, and why — to be filled in as Phase 1 lands, in the spirit of
[plan-no-closed-form.md](plan-no-closed-form.md) §0's table rather than silently folded in.
**Per [REV 49], record the field-level diff beside it; on today's pool one score moves and nineteen
field values do.**

> **Measured, 2026-10-15.** One score moved — correctly predicted — and **61** `(plan, field)` pairs
> moved across **9 of 12** plans, over 19 of the 25 fields the companion's §B records. The plan count
> was exact; the field count was low by more than three times, for the reason [REV 49] gives in the
> first place: the fields a plan can *name in advance* are not the fields that move. Re-derived in
> [REV 69] from §B's own tables after a hand count said 35.

> **Four criteria re-read after implementation, 2026-10-15.** An independent evaluation checked all
> 23 against the metrics; these four were met in a narrower sense than their wording claims. See
> [§13](#13-second-round--from-an-independent-evaluation-2026-10-15).
>
> | # | as worded | what was actually true | revision |
> |---|---|---|---|
> | 3 | the ladder completes on all four multi-noise problems | **two of four.** `sde_gbm_2d_correlated` and `sde_log_heston_feller_violated` still have no live spec, exactly as [REV 7] foresaw and §E's *"Not predictable yet"* repeated. Open; see [§13d](#13d-what-this-round-still-leaves-open) | — |
> | 6 | a non-evaluable variance expression grades against `variance_value_at_T` | **half-met, in the way §C7 predicted.** The fallback keyed on the expression being *absent*; a present-but-unevaluable one still raised through to `"unavailable"` and the 9 | [REV 58] |
> | 7 | every SDE problem's spec declares `expected_strong_order` | met for the base field and **not enforced per family**: `expected_strong_order_milstein` existed only as prose inside the error message for the base field | [REV 63] |
> | 11 | no spec field is declared and unread without a recorded reason | met against a **text search over file contents**, so a docstring mention counted as a consumer. Five live fields had nothing else | [REV 65] |
>
> Criterion 19 is met and its wording is now stale: the abstention's token is `"unresolved"`, not
> `"unavailable"`, and the two are priced differently on purpose ([REV 59]). The criterion's claim —
> that an abstention costs nothing — is asserted at the one place it is priced.

---

## 7. Revised shipping order **(Rev 2)**

§3's grouping is kept exactly as written; this is the order to *land* it in. The reason for
inverting it is one sentence: **every item in Phase 1 either depends on spec coverage that Phase 6
schedules last, or converts a silent pass into a hard cap through a check whose false-failure rate
is inherited rather than measured.**

| Step | Items | Why here | Risk to correct work |
|---|---|---|---|
| **0** | **`_widen`** (Finding 15), **`strong_order`'s error bar** ([REV 36a]) and **`_martingale`** ([REV 50]) | ~~two~~ **three** measured bugs, not design calls: one spec loses its only reference tier, the order study reports a 12.4x spread as a pass, and the martingale evaluator discards a declared `expr` | ~~none — both *remove* false verdicts~~ — **wrong as specified**: none *only with* [REV 37]'s `guards` change. See the Rev 3 amendment below. |
| 1 | **1d** + its `reference.py` half ([REV 17]) | pure win; a reference that was being thrown away starts counting | none |
| 2 | **2a** (read the declared `noise_dimension`, schema-require it, fix the specs) + **2c** (four sites, [REV 11]) + [REV 8]'s evaluability guard | unblocks four problems, changes no score | none |
| 3 | **3b tier 1** (3rd/4th standardized moments) and **3a** via `_compare` only ([REV 22]); 3b's KS tier per **D4** | the new evidence classes, reported before they gate | low — new checks start as diagnostics |
| 4 | **5c**, **5b** narrowed ([REV 26]), **[REV 29]**'s spec-field-coverage test | cheap, and the coverage test is what keeps this class from recurring | low |
| 5 | Finding 5's random-IC fix, then **4a** seed replication | both are preconditions of 1b | none; both *remove* false failures |
| 6 | **Phase 6**, `expected_strong_order` on every spec first ([REV 27]) | the precondition of 1c | none |
| 7 | **1c**, per **D2** — one-sided band on a defaulted floor; (c) superseded by step 0 | the order study starts gating | **was** high; D2 and step 0 are what remove it |
| 8 | **1b**, with [REV 15]'s `ceiling()` reason fix | the fabrication hole closes | moderate — mitigated by steps 5 and 6 |
| 9 | **1a** per **D1** (the 8 row), **1e** + §24 ([REV 18]) + the **D1**/**D5** manual edits | by now a transcription of what already holds | none |
| 10 | **2b** per **D3**, **2d**/[REV 21], **4b** per **D6**, **5a** per **D5** | the deferrable tail | low |

Three things this order buys that §3's does not:

* **No window in which a correct solver is capped.** In §3's order, Phase 1 ships before Phase 4
  and Phase 6, so between those commits a correct additive-noise plan scores 7 ([REV 16]) and a
  correct plan with a 3σ Dynkin seed scores 3 ([REV 24]). In this order the false-failure sources
  are closed before the checks that would expose them start gating.
* **The benchmark-number change happens once.** Steps 7–9 are the only ones that move a score, and
  they land together, so the "scores that moved, and why" table §6 asks for is written once rather
  than amended three times.
* **The deferral candidate is still 2b**, as §3 intended — but [REV 31] shows it is cheaper than
  §5 claims, so the thing to actually defer is 5a, which needs formulator work before it grades
  anything.

> **[Rev 3 amendment to §7] — one row respecified, four items placed.** The table above stands;
> these are the changes Rev 3's reading makes to it.
> * **Step 0 is three touches, not two.** `strong_order`'s error bar is unsafe without the `guards`
>   change that gives `orders_ok` a third value ([REV 37]), and `_widen`'s rule is a choice, now made
>   in [§11 D7](#d7--pin-boundaries-a-declared-support-already-fixes).
> * **Step 0 gains a third bug, [REV 50]'s `_martingale`.** It ignores a declared `expr`, so two live
>   Path-A plans fail `invariants_ok` on artifacts today; 1a's invariants row must not ship before it
>   is fixed and before `sde_bm_with_drift`'s spec declares `reference_value_at_T`.
> * **Step 0 gains two more items.** `ranking.order_margin`
>   ([§11 D12](#d12--fix-the-ranking-branch-rather-than-delete-it)), because it is dead code today
>   and fixing it later means remembering to; and [REV 40]'s one-line spec edit
>   (`expected_strong_order_milstein: 1.0` on `sde_cir_feller_violated`), which closes §24's own
>   motivating case with no kernel change at all. [REV 45]'s re-measurement and `STATE.md`
>   write-back close step 0.
> * **Step 3 ships 3b tier 1 report-only**
>   ([§11 D11](#d11--the-cheap-tier-ships-report-only-until-a-law-is-declared)), and its reference
>   field — a declared `terminal_law`, shared with D4's KS tier — joins Phase 6's scope in step 6.
> * **Step 5's replication is the median rule, fired only near the bar**
>   ([§11 D9](#d9--the-median-verdict-and-replicate-only-near-the-bar)). That is what keeps step 8
>   affordable; see [REV 42](2).

> **[REV 55] — steps 2 and 8 multiply, and on one problem the product does not fit in memory.**
> Neither step's cost line accounts for the other. Step 2 makes the Dynkin increment array
> `num_paths × Nt_fine × m` rather than `num_paths × Nt_fine`, and step 8 runs that array on Path A
> plans that never built it before. Measured with `_dynkin_footprint` at each live spec's own
> configuration, float64, three levels:
>
> | problem | `num_paths` | `dt` | `m` | `Nt_fine` | increments |
> |---|---|---|---|---|---|
> | `bm_with_drift` | 50 000 | 0.01 | 1 | 400 | 0.16 GB |
> | `ornstein_uhlenbeck` | 50 000 | 0.01 | 1 | 400 | 0.16 GB |
> | `cir_feller_violated` | 50 000 | 0.005 | 1 | 800 | 0.32 GB |
> | `double_well_langevin` | 200 000 | 0.01 | 1 | 400 | 0.64 GB |
> | `gbm_2d_high_corr` | 50 000 | 0.005 | **2** | 800 | 0.64 GB |
> | `ginzburg_landau_s6` | 50 000 | 0.0002 | 1 | 20 000 | **8.00 GB** |
> | `multichannel_stiff_m13` | 50 000 | 0.0005 | **13** | 8 000 | **41.60 GB** |
>
> Ginzburg's 8 GB is already live and is why those three plans take ~10 minutes each — 29 of the
> baseline's 33.5. m13's 41.6 GB is what step 2 would newly allocate.
>
> Guarded at `DYNKIN_MAX_BYTES = 12 GB`, refusing with the footprint and the three levers in the
> stated reason. **Silently reducing `num_paths` was rejected**: it is the one knob that would make
> m13 fit, and [REV 35a] measured what it costs — ginzburg's z falls from 3.24 to 1.41 between
> 50 000 and 12 500 paths, so an automatic reduction converts a true positive into a pass. A refusal
> a reviewer can see is better than a verdict computed at a path count nobody chose.

> **[REV 56] — step 8's "moderate risk" was real, and it landed on a correct solver.** Shipping 1b
> took `sde_cir_feller_violated/2-milstein` from **10 to 3** on a Dynkin failure at z = 5.1, and the
> solver is correct: it is the standard full-truncation Milstein, and it converges to the exact
> noncentral-chi-squared transition law. The cause is in Dynkin's extrapolation, not in the plan.
>
> `2·kappa·theta = 1.0` against `sigma² = 4` violates the Feller condition, so zero is attainable
> and `sqrt(X)` is non-Lipschitz exactly where the paths spend their time. Every truncated scheme
> converges slowly there. Measured against the exact law, `E[tanh(X_T)]`:
>
> | `Nt` | 200 | 800 | 3 200 | 12 800 | exact |
> |---|---|---|---|---|---|
> | `E[tanh(X_T)]` | 0.2961 | 0.2903 | 0.2890 | 0.2861 | 0.2800 |
> | error | 0.0161 | 0.0103 | 0.0090 | 0.0061 | — |
>
> The error falls 2.6× over a **64×** refinement — order ≈ 0.2. The Dynkin residual behaves the same
> way: 0.0316 → 0.0263 → 0.0207 across the ladder, shrinking monotonically and 11 s.e. clear of its
> own noise, from which three levels spanning a 4× `dt` range read a rate of **−0.05**. That is
> below `dynkin.ORDER_BAND`'s lower edge, so the check fell back to `assumed_1`, under-corrected,
> and reported the leftover — the scheme's own bias — as a generator violation.
>
> **The fix is an abstention, and it is the mirror image of a fix this file already made.** §20's
> own docstring records the `sde_double_well_langevin` regression where a hard-coded `p = 1`
> *over*-shot through zero on a weak-order-2 scheme and capped two correct plans at 3; this is the
> same bug from the other side. Where `p` is below the band **and the residual is shrinking**,
> `dynkin.check` now returns `"unresolved"` — a skip, with the sequence and the rate in the reason —
> and records the `z` anyway, per D8: an abstention is not silence.
>
> **Why this cannot reopen Finding 1's hole**, which is the only question that matters here. The
> three failure modes land in three *different branches* of `residual_order`, so the separation is
> structural rather than a tuned threshold:
>
> | solver | residual vs `dt` | branch | outcome |
> |---|---|---|---|
> | `sde_fabricated_sampler` | `dt`-independent | differences inside their own noise | **`False`**, score 3 |
> | `ginzburg_landau_s6` 1 and 2 | clean rate 1.20 / 1.36 | measured, limit outside the CI | **`False`**, score 3 |
> | this CIR plan | shrinking, rate −0.05 | **below the band** | **`"unresolved"`**, score 10 |
> | a diverging scheme | growing under refinement | below the band, not shrinking | **`False`** |
>
> A solver that never integrates the SDE has a residual that does not move with `dt` at all, so its
> differences between levels are Monte-Carlo noise and the *first* guard in `residual_order` catches
> it before any rate is read. That is asserted end-to-end (`sde_fabricated_sampler` still scores 3)
> and as a unit (`test_a_residual_that_does_not_move_with_dt_still_fails`), and the growing case has
> its own test too, because admitting it would have been the easy mistake.
>
> Two further things this found, both recorded as tests rather than fixed:
> * **Extrapolating at the measured low `p` is not the alternative.** At `p = 0.07`, `2^p − 1 = 0.05`
>   multiplies the correction and its noise by ~22, so as `p → 0` everything passes. The lower band
>   edge exists to stop that.
> * **The same scheme fails the KS tier, correctly, at 48× the bar.** Full truncation leaves **12%
>   of the paths exactly at zero**, where the exact law has no atom, while the mean sits 4.96% out
>   against a 5% tolerance and the variance 1.8% against 10%. That is Finding 4's third defect class
>   — "a clipped state that moves mass to a point and leaves the mean inside tolerance" — caught on
>   a real scheme instead of on a canary written to be caught. It is invisible on the live plan
>   because that spec declares no `terminal_law`; declaring one is formulator work, so this ships as
>   a canary assertion and a flagged finding, not a live spec edit.

## 8. Rev 2 verification log

What was re-run, so the next reader does not have to repeat it. Everything below was measured on
2026-10-03 against the working tree at `d5bb447`.

| Claim | Method | Result |
|---|---|---|
| §2a: fabrication scores 10, certified, on Path A | `run()` on `canaries/kernel/sde_spec.json` with a law-sampling solver, floors 0.5 and 1.0 | reproduced verbatim, both floors |
| [REV 5]: 1b alone caps it, and where | same, `run_dynkin` forced `True` | 3 / not certified via `ceiling()`; honest solver 10, z = 1.06 |
| [REV 16]: 1c false-fails correct additive-noise EM | honest canary, floor 0.5, `scheme_family` declared | `orders_ok: False` at observed 0.9863 |
| [REV 16]: the exact-scheme case | `strong_order` on CRN increments for `dX = dW` | `-0.1226` from `s0 = 1.7e-16`, `s1 = 1.9e-16` |
| §2b: the IndexError | `sandbox.run` on the real `sde_gbm_2d_high_corr` plan, `m=None` | reproduced verbatim |
| [REV 19]: `m=2` is sufficient | same, `m=2` | clean; `corr(X_T, Y_T) = 0.945` vs spec 0.9478 |
| [REV 8]: m13's Dynkin failure mode | `observables_payload` + `sandbox.run` with `m=13` | payload built from prose; `bad_schema` broadcast error |
| §2f: the discarded reference | `_claimed` / `_reference` on the stale quintic spec, then a grafted canary run | reproduced; score 4, "the variance check fails" |
| [REV 17]: the provenance coupling | same run, inspecting `certification` | tier `A_unavailable`, provenance `analytic_unvalidated`, ceiling 9 |
| [REV 23]: the KS gate | one-sample KS, correct EM, n = 50 000, OU and GBM at dt ∈ {1e-3, 1e-2} | rejects at 1e-2 on both; marginal at 1e-3 |
| §2m: spec coverage | directory and key census over `workspace/` and the stale backup | 23 / 7 live / 16 stale-only, exactly as stated |
| §1: suite size | `pytest --collect-only` on the two SDE files | 48 collected |
| [REV 33]: declaration status | `plan_meta.read` over all 12 live SDE plans | 10 `inferred`, 3 `declared` (all `double_well`) |
| [REV 33]: the live roundoff case | `run()` on the real `sde_bm_with_drift/plans/1-euler-maruyama` | 10, `orders_ok: True`, order 2.27 from s0/s1 ~ 1e-15 |
| [REV 34]: the ranking branch | `grep` + guard trace through `ranking.order_margin` | unreachable; `design_order` is PDE-only |
| §6 baseline | recorded `score` / `provenance` in all 7 live `STATE.md` | 10, 8, 3/10/3, 10, 10/10/10, 6, 10 |
| [REV 35a]: path scaling | plan 1, seed 42, M = 12500 / 25000 / 50000 | z = 1.406 / 1.686 / 3.242; 2.31x over 4x paths |
| [REV 35a]: seed spread | 3 plans x 3 seeds at M = 12500 | all 9 pass, z = 0.42–1.89 — **underpowered, tested nothing** |
| [REV 36]: order scatter | same 10 runs, `observed_order` | plan 1 spans 0.452–3.069; `orders_ok: True` in all 10 |
| Finding 15: the guard's age | `git log -S "0.1 * float(trust_tol)"` | one commit — the file's own creation; not a regression |
| Finding 15: the cause | `kolmogorov()` with `_widen` swapped, same spec | symmetric → 1.385% FAILS; far-field → 0.361% PASSES; reference identical |
| [REV 35b]: seed spread at full M | 3 plans x 4/4/3 seeds at M = 50000 | plan 1 4/4 fail, plan 2 4/4 fail, plan 3 0/3; gap 1.603 → 2.727 |
| [REV 36a]: order scatter at full M | same 11 runs | plan 2 spans 0.220–2.730 (12.4x); `orders_ok: True` in all 11 |
| Finding 15: rubric consequence | `rubric_sde` called directly with synthetic evidence | restoring the surrogate leaves plans 1 and 2 at **3** |

Not re-run, and flagged as such: the 5m35s suite wall time (§1), and the four stale multi-noise
problems' `orders_ok` verdicts ([REV 7]), which have no live spec to run against.

### 8a. Rev 3 verification log

Read from the implementation on 2026-10-14, against the same tree. Rev 3 ran **no new pipeline
measurements** — the three it would have wanted ([REV 45]'s re-measurement, the `git log` on
`score.py`/`dynkin.py`, and D8's median-vs-mean comparison) are listed as outstanding in
[§11](#11-decisions--resolved-2026-10-14) rather than claimed.

| Claim | Method | Result |
|---|---|---|
| [REV 37]: a `None` strong order is a *failure*, not a skip | read `crn.guards` 149-172 | `finite = strong is not None …`; `s_ok = finite and …`; `"orders_ok": bool(s_ok)` |
| [REV 37]: the third value is already reachable downstream | read `_orders` | returns the literal `"unavailable"` for a crashed level and for `< 3` levels |
| [REV 38]: two `failed(orders_ok)` rows | read `rubric_sde` 336-380 | `:354` → **8** under `surrogate and moments_ok and resolved and constraints_ok`; `:368` → **3**, unguarded |
| an order failure is a rubric cap, not a ceiling | read `gates_clean` | reads `gate_violation`, `ic_consistent`, `d1_outcome`/`operator_validated` only |
| [REV 40]: the per-family floor fires on an inferred family | read `expected_orders` + `FAMILY_ORDER_KEY` | the family key is selected *before* the `declared` branch; `candidates` collects every `expected_strong_order*` |
| [REV 40]: which specs declare it | `grep -rn expected_strong_order workspace/sde_*/problem_spec.json` | `ginzburg_landau_s6` and `double_well_langevin` declare `_milstein`; `cir_feller_violated` does not |
| [REV 41]: no higher-moment reference exists | key census over `analytic_moments` in all 7 live specs | mean/variance and per-component spellings only; OU states the target in a prose `note` |
| [REV 43]: `order_margin`'s coverage in this plan | `grep -n order_margin` over the plan | index, [REV 34], [REV 36], §8 — no criterion, no §7 step |
| [REV 46]: `weak_order_out_of_band` has no consumer | `grep -rn` over `verifylib/` | one hit: its own definition at `crn.py:165` |
| §2g / D6: `compare` reaches the SDE path at all | read `cli._compare`'s dispatch | `equation_type == "SDE"` → `_compare_sde` at [`cli.py:272`](../verifylib/cli.py#L272); D6 is implementable as written |

| D2 / D12: the floor's source really is recorded | read `sde/driver.py` 50-55 | `cfg.set("expected_strong_order", …, "spec" if verification.get(…) is not None else "kernel_default")` — [REV 16](b) was right, and `CONFIG_SOURCES` carries both |
| D8: a third value for `orders_ok` needs no change to `metrics` | read `metrics.record` | accepts `True`, `False` or any member of `SKIPS`, and raises on anything else |
| [REV 47]: when the recorded scores were written | `stat` on all 7 live `STATE.md` + `git log -S` on `score.py` / `dynkin.py` | 6 of 7 dated 2026-08-19..23; the row ordering is `a507038` (09-20) and the downward ladder `ea48da7` (09-26) |
| §9d: the suite's real size | `pytest --collect-only` over the repo | **583** collected, 587 total — `pyproject.toml:43` deselects 4 `slow` tests, all `pde_*` truth routes, none on the SDE path |

| [REV 48]: the suite baseline | `pytest -q --durations=15 -rf`, full repo | **580 passed, 3 skipped, 4 deselected, exit 0, 368.92s**; 11 SDE Path-B tests are 276.8s of it |

---

## 9. Pending before implementation **(Rev 2)**

*Rev 2, 2026-10-04: this section has been worked through. **9a is resolved** — see
[§10](#10-decisions--resolved-2026-10-04). **9b is measured** — the baseline is in
`benchmark/results/`, and 6 of 12 recorded scores were wrong. The two items it turned up,
Finding 15 and [REV 36a], are now [§7](#7-revised-shipping-order-rev-2)'s step 0. What remains
pending is 9c (spec work), 9d (the suite baseline) and 9e (one regression surface) — none of
which is a decision.*

*Rev 3, 2026-10-14: still true, with two amendments. [§9c](#9c-spec-and-artifact-work-that-gates-specific-acceptance-criteria)
gains two rows ([REV 40], [REV 41]) and one of them — a per-family order floor on
`sde_cir_feller_violated` — is a one-line edit that closes §24's own motivating case before any code
change. [§9d](#9d-two-measurements-this-audit-did-not-take) gains [REV 45]: the §9b baseline
predates step 0, and nothing writes the corrected scores back to `STATE.md`.*

### 9a. Six decisions the plan still forks on — **RESOLVED 2026-10-04**

*Rev 2 note: all six were taken on 2026-10-04, each in favour of the recommendation recorded
below. The forks are kept as written so the alternative that was rejected stays visible; the
resolutions, with the exact code change each implies, are in
[§10](#10-decisions--resolved-2026-10-04). Nothing in this subsection blocks implementation any
more.*

Each was a design call, not a lookup, and each changed what gets written:

1. **[REV 14] — 8 or 7** for moments-pass/orders-fail on Path A. Recommended 8, matching
   `rubric_sde`. Moves benchmark numbers either way.
2. **[REV 16](b) — the band rule.** One-sided whenever the floor came from `kernel_default`, or
   make a declared `expected_strong_order` a hard precondition and refuse to run the ladder
   without it. The second is stricter and needs Phase 6 complete first.
3. **[REV 20] — where the machine-readable generator lives.** A new evaluated path
   (`verification.generator.*`) or an amendment to `schema.py`'s `GUIDANCE_PATHS`. This is the
   leakage model, so it is the one decision worth writing down even if 2b is deferred.
4. **[REV 23] — which KS mitigation**: standardize before testing, a `dt`-scaled bar, or
   diagnostic-only. Recommended the first; it is the only one that needs no per-family calibration.
5. **[REV 25] — `stability_check` semantics.** Replaces moment scoring (`project_manual.md:304`,
   `benchmark/verify.py`) or gates on top of it. The manual and 5a currently disagree.
6. **Phase 4b / §22 — amend the manual or change the code.** The plan recommends running both
   comparisons; that doubles `compare`'s solves and should be stated.

### 9b. The §6 baseline — **measured 2026-10-04**

*Rev 2 note: this subsection originally read "has no baseline and cannot be filled in yet". It has
now been measured. The original concern was right and understated: the recorded values were not
merely absent from disk, **6 of the 12 are wrong**.*

All 12 live plans re-evaluated at their declared `evaluation_thresholds`, 33.5 minutes wall,
exit 0. Full metrics written to each `<plan>/_eval_metrics.json`; consolidated summary in
[`benchmark/results/sde_baseline_2026-10-04.json`](../benchmark/results/sde_baseline_2026-10-04.json). `Rec.` is what `STATE.md` records;
⚠ marks a divergence.

| Plan | P | Rec. | **Now** | Prov. | `orders_ok` | order / floor | `dynkin_ok` (z/bar) | s |
|---|---|---|---|---|---|---|---|---|
| `bm_with_drift` / `1-euler-maruyama` | A | 10 anal | **10** | analytic | True | 2.238 / 1.000 | not_run — | 1s |
| `cir_feller_violated` / `1-euler-maruyama` | A | 8 anal | **8** | analytic | True | 0.384 / 0.500 | not_run — | 1s |
| `cir_feller_violated` / `2-milstein` | A | 8 anal | **10** ⚠ | analytic | not_run | — | not_run — | 1s |
| `double_well_langevin` / `1-tamed-heun-drift-trapezoid` | B | 3 surr | **10** ⚠ | surrogate | True | 1.046 / 1.000 | True 1.69/2.43 | 41s |
| `double_well_langevin` / `2-drift-implicit-trapezoid` | B | 10 surr | **10** | surrogate | True | 1.026 / 1.000 | True 1.74/2.43 | 36s |
| `double_well_langevin` / `3-strang-splitting-exact-flows` | B | 3 surr | **10** ⚠ | surrogate | True | 1.020 / 1.000 | True 1.76/2.43 | 32s |
| `gbm_2d_high_corr` / `1-euler-maruyama` | A | 10 anal | **10** | analytic | not_run | — | not_run — | 1s |
| `ginzburg_landau_s6` / `1-tamed-euler-maruyama` | B | 10 surr | **3** ⚠ | self_convergence | True | 0.452 / 0.500 | False 3.24/2.43 | 639s |
| `ginzburg_landau_s6` / `2-tamed-milstein` | B | 10 surr | **3** ⚠ | self_convergence | True | 0.644 / 1.000 | False 3.14/2.43 | 628s |
| `ginzburg_landau_s6` / `3-log-transform-euler` | B | 10 surr | **9** ⚠ | manufactured_partial | True | 1.019 / 0.500 | True 1.10/2.43 | 622s |
| `multichannel_stiff_m13` / `1-euler-maruyama` | A | 6 anal | **6** | analytic | unavailable | — | not_run — | 3s |
| `ornstein_uhlenbeck` / `1-euler-maruyama` | A | 10 anal | **10** | analytic | True | 1.008 / 1.000 | not_run — | 2s |

**Six of twelve diverge from `STATE.md`, in both directions.** The recorded scores predate kernel
changes and cannot be used as the before-column:

* **`double_well` 1 and 3: 3 → 10.** Expected and already documented — `dynkin.py`'s docstring
  describes the ladder being re-anchored to refine *downward* from the graded `dt`, and says these
  exact plans "stayed capped" before that fix. They now measure z = 1.69–1.76, clean.
* **`ginzburg_landau_s6` 1, 2, 3: 10, 10, 10 → 3, 3, 9.** Two causes, both worth settling before
  step 1: Finding 15 (the Kolmogorov surrogate no longer passes its trust guard, so all three lose
  `provenance: surrogate`) and a Dynkin failure on plans 1 and 2 at z = 3.24 and 3.14 against a bar
  of 2.43 — see [REV 35].
* **`cir_feller_violated` 2: 8 → 10.** Its mean error is 4.96% against a 5% tolerance, so this plan
  sits on the boundary and will move with the kernel or the seed. Treat the 10 as fragile.

> **[REV 47] — the dating question is closed, and it explains every divergence. [M]** §9b asserts
> above that "the recorded scores predate kernel changes" and Finding 15's correction asks for
> *"one `git log` before trusting any recorded SDE score"*. Run, 2026-10-14. `workspace/` is
> gitignored ([`.gitignore:35`](../.gitignore#L35)) so `STATE.md` has no history, but its **mtime**
> dates each recording, and the two kernel changes are in the tracked history:
>
> | | Recorded | Relative to the two kernel changes |
> |---|---|---|
> | `sde_ornstein_uhlenbeck` | 2026-08-19 18:39 | before both |
> | `sde_bm_with_drift` | 2026-08-19 19:43 | before both |
> | `sde_gbm_2d_high_corr` | 2026-08-19 20:23 | before both |
> | `sde_cir_feller_violated` | 2026-08-19 06:48 | before both |
> | `sde_multichannel_stiff_m13` | 2026-08-20 01:07 | before both |
> | **`sde_ginzburg_landau_s6`** | **2026-08-23 17:49** | **before both** |
> | `sde_double_well_langevin` | 2026-09-26 00:51 | same day as the second |
>
> * `rubric_sde`'s row ordering (findings §7.6, the `failed(dynkin_ok) -> 3` row placed **ahead** of
>   the surrogate row) landed in **`a507038`, 2026-09-20**.
> * `dynkin.py`'s downward-refining ladder landed in **`ea48da7`, 2026-09-26** — the only change to
>   that file after its creation in `3447862`.
>
> **This answers the question Finding 15 left open.** That correction offered three hypotheses —
> the recorded scores predate the row ordering, or predate the downward-refinement change, or the
> plans themselves changed. It is the **first two**, and not the third. `ginzburg_landau_s6`'s
> 10/10/10 was recorded on 2026-08-23, a month before `failed(dynkin_ok) -> 3` was placed ahead of
> the surrogate row — so at the time those 10s were written, a Dynkin failure did **not** cap a
> plan whose surrogate passed. There is no need to suspect the plans; the rubric changed underneath
> them.
>
> And `double_well_langevin`'s 3/10/3 was recorded at 00:51 on the same day as `ea48da7`, which is
> exactly the "stayed capped" state that commit's own docstring describes fixing — confirming §9b's
> reading of those two rows as "expected and already documented".
>
> **So all six divergences are now accounted for**, with no residual mystery: three
> `ginzburg_landau_s6` plans against a pre-2026-09-20 rubric, two `double_well` plans against a
> pre-`ea48da7` Dynkin ladder, and `cir_feller_violated/2-milstein` sitting at 4.96% against a 5%
> tolerance. The recorded values are not wrong so much as **stale**, which is a weaker claim than
> §9b made and a better-supported one. [REV 45]'s write-back is still needed.

**Two rows are the live instances of what Phase 1 exists to catch**, and neither needs Phase 6:

* **`bm_with_drift`: 10, `orders_ok: True`, observed order 2.238** from `s0 = 9.4e-15`,
  `s1 = 2.0e-15`. Euler–Maruyama is pathwise exact on `dX = 0.5 dt + 0.3 dW` and its own
  `SOLUTION.md` says so, so the order study is reading roundoff and reporting it as a pass
  ([REV 33]). It clears the one-sided bar of 0.6 only by accident of the plan not declaring
  `scheme_family`.
* **`cir_feller_violated` 2 (Milstein): 10, `orders_ok: not_run`, `run_crn_ladder: False`.** This is
  §24's own motivating case, uncaught. §24 justifies the `>= 1.0` condition with exactly one
  scenario — *"a Milstein plan with a broken correction term will pass the thresholds and still be
  mislabelled in `REPORT.md`; the strong-order check is the only thing that catches it"* — and here
  is a live Milstein plan certified at 10 with no order evidence at all, because the spec declares
  only `expected_strong_order: 0.5` and no `expected_strong_order_milstein`, so `expected_orders`
  returns 0.5 and the condition is false. **This strengthens 1c's rationale and complicates its
  fix**: dropping the condition is not sufficient, the spec has to declare the per-family order —
  Phase 6 again, per [REV 27].

> **[REV 40] — §24's own motivating case closes with a one-line *spec* edit, before any code
> change. [R]** The bullet above — "dropping the condition is not sufficient, the spec has to
> declare the per-family order — Phase 6 again" — understates how close it already is. The mechanism
> exists and works:
>
> ```python
> FAMILY_ORDER_KEY = {"milstein": "expected_strong_order_milstein",
>                     "tamed-milstein": "expected_strong_order_milstein"}
> ```
>
> ([`crn.py:85`](../verifylib/kernel/sde/crn.py#L85)), and the family key is selected **even on an
> inferred family** — [`crn.py:120-125`](../verifylib/kernel/sde/crn.py#L120), whose comment says
> why: *"selecting among the spec's declared values is safe even on an inference: every candidate
> came from the formulator"*. Two live specs already use it: `sde_ginzburg_landau_s6` and
> `sde_double_well_langevin` both declare `expected_strong_order_milstein: 1.0`.
> **`sde_cir_feller_violated` declares only `expected_strong_order: 0.5`** — and that single
> omission is the entire reason its Milstein plan certifies at 10 with `run_crn_ladder: False`.
>
> So adding `expected_strong_order_milstein: 1.0` to that one spec makes the ladder run there
> **today**, with no kernel change and ahead of 1c. Two consequences:
> * Add it to [§9c](#9c-spec-and-artifact-work-that-gates-specific-acceptance-criteria) as a row and
>   to step 0 of §7. It is the cheapest score-relevant fix in this document.
> * **Acceptance criterion 7 is insufficient for its own motivating case.** "Every SDE problem's
>   spec declares `expected_strong_order`" is *already true* of `sde_cir_feller_violated` while the
>   hole stays open. Re-word it to require a floor for every scheme family the spec's plans use.

Also measured, as context for [REV 7] and [REV 19]: `gbm_2d_high_corr` scores **10** today with
`orders_ok: not_run` *and* the correlation ungraded, so 2a and 3a will move it in opposite
directions — 2a supplies the order verdict it currently lacks, 3a supplies the ρ check it currently
evades. It is the single best end-to-end test of Phase 2 + Phase 3 together.

### 9c. Spec and artifact work that gates specific acceptance criteria

| Needed | Gates | Today |
|---|---|---|
| `expected_strong_order` on the 16 stale specs | 1c, criterion 7 | absent; defaults to 0.5 ([REV 16]) |
| `noise_dimension` on `sde_gbm_2d_high_corr` | criterion 3 | declared `null` ([REV 6]) |
| Live specs for `sde_gbm_2d_correlated`, `sde_log_heston_feller_violated` | criterion 3 | none ([REV 7]) |
| An evaluable `drift_expression` on `sde_multichannel_stiff_m13` | Dynkin there | prose ([REV 8]) |
| A live spec for `sde_quintic_random_ic` | Findings 5 and 6's canaries | none; measurements were taken on the stale spec |
| Structured `stability_check` on `sde_fene_blowup`, `sde_quintic_drift_noise` | 5a | oracle-only; the stale one is prose ([REV 9]) |
| Which reason `milstein_eligible: false` carries | 2d / [REV 21] | unrecorded; the field means two things |

| A per-family floor (`expected_strong_order_milstein`) wherever a Milstein plan exists | 1c, criterion 7, §24's motivating case | absent on `sde_cir_feller_violated`; present on the other two ([REV 40]) |
| A declared `terminal_law` (serves the KS tier **and** 3b tier 1's reference) | 3b both tiers, criterion 16 | no spec declares it; OU states the target in prose ([REV 41]) |
| `reference_value_at_T` or an `expr` the evaluator uses on `sde_bm_with_drift`'s `martingale` | 1a's invariants row | prose-only, so the kernel measures `E[X_T] - X_0 = mu*T` instead ([REV 50]) |

### 9d. Two measurements this audit did not take

* **The full-suite baseline.** §1's "48 passed in 5m35s" was re-verified as 48 *collected*, not
  re-run; and the wider suite (580 tests per the scoring/ranking work) has no measured baseline in
  this audit. Criterion 9 needs one before step 1. **— measured 2026-10-14, see [REV 48].**
* **`orders_ok` on the four stale multi-noise problems** ([REV 7]) — unmeasurable until they have
  live specs, so criterion 3 should be scoped to the two that do.

> **[REV 48] — the suite baseline, measured. [M]** The gate on criterion 9, and on step 1 of
> [§7](#7-revised-shipping-order-rev-2). Run 2026-10-14 at tree `d5bb447` plus this untracked
> document, `python3 -m pytest -q --durations=15 -rf`:
>
> ```
> 580 passed, 3 skipped, 4 deselected in 368.92s (0:06:08)     exit 0
> ```
>
> * **583 collected, 580 passed, 0 failed.** The 4 deselected come from `pyproject.toml:43`'s
>   `addopts = "-m 'not slow'"` and are all four `pde_*` truth-route tests; the 3 skips are
>   `"not staged in workspace/"` guards in `test_kernel_end_to_end.py` and `conftest.py`, also all
>   PDE. **Nothing on the SDE path is skipped or deselected**, so criterion 9's "full suite" means
>   these 583 and the number to hold green is **580 passed / 0 failed**.
> * **§1's figure is confirmed as a measurement.** The two SDE files collect exactly **48**, and the
>   whole 583-test suite now runs in 6m08s against §1's 5m35s for those two files alone —
>   consistent, and no longer only a collection count.
> * **The cost concentration is the finding worth carrying forward.** The eleven slowest tests are
>   *all* `test_kernel_sde_path_b.py`, at 21.05–44.37s each, and they total **276.8s of 368.92s —
>   75% of the suite's wall time, from 11 of 583 tests.** They are the canaries that run real CRN
>   and Dynkin ladders.
>
>   That sharpens two estimates the plan makes loosely. [§4d](#4d-cost)'s "a Path-A suite of the
>   same canaries roughly doubles it" is now checkable: ~277s of SDE ladder work becomes ~554s and
>   takes the suite past 10 minutes — *before* 1b adds a Dynkin ladder to every Path-A canary and
>   before [REV 42](2)'s replication multiplies it again. §4d's advice to run the new defective
>   canaries at reduced `num_paths` is not an optimisation; on these numbers it is the difference
>   between a suite someone runs and one they skip.
> * Full detail, including all 15 durations:
>   `benchmark/results/sde_suite_baseline_2026-10-14.json`. That path is gitignored
>   ([`.gitignore:23`](../.gitignore#L23)) and the gitignore is deliberately left alone, so **the
>   numbers above are the durable record** — read them here, not from the file.

> **[REV 45] — nothing in the plan writes the corrected scores back, and the baseline itself
> predates step 0.** Three items, all cheap, none of them a decision:
> * **Write-back.** [§9b](#9b-the-6-baseline--measured-2026-10-04) establishes that 6 of 12
>   `STATE.md` scores are wrong and concludes they "cannot be used as the before-column". No step of
>   [§7](#7-revised-shipping-order-rev-2) corrects the seven `STATE.md` files, no criterion asserts
>   they match a re-measured baseline, and [§6](#6-acceptance-criteria)'s "scores that are
>   *expected* to move" table names no before-column. Left as is, the next reader of `STATE.md` gets
>   the same wrong numbers this audit just disproved.
> * **The baseline is pre-step-0.** Finding 15 says to fix `_widen` "before step 1 of §7 and
>   re-measure", but §7's step 0 does not carry the re-measurement and this subsection does not list
>   it. After step 0 the only recorded baseline is stale for precisely the three plans step 0 moves.
>   Re-run those three: `ginzburg_landau_s6` was ~31 of the baseline's 33.5 minutes, so this is one
>   half-hour job rather than a repeat of the sweep.
> * **One command the plan asked for and never ran.** Finding 15's correction ends *"Worth one
>   `git log` before trusting any recorded SDE score"*, and §9b asserts "the recorded scores predate
>   kernel changes" without it. `workspace/` is untracked so `STATE.md` has no history, but
>   `dynkin.py` and `score.py` do, and the dates of the row-ordering and downward-refinement commits
>   settle it in one command.

### 9e. One regression surface to watch

[REV 17]'s `reference.py` change touches `_moment_observables`, which feeds
`check_moment_ode_consistency` — the spec-vs-spec ledger that `test_reference_sde.py` covers.
Adding a `*_value_at_T` fallback there changes which pairs get compared, so that file's tests are
the ones most likely to move for a non-obvious reason. Run it first and in isolation.

---

## 10. Decisions — resolved 2026-10-04

All six forks in [§9a](#9a-six-decisions-the-plan-still-forks-on--resolved-2026-10-04) taken in
favour of the recommendation. Each row below is binding on the phase it names; where a decision
contradicts a manual, the manual edit is part of the same commit, per this document's own `[D]`
discipline.

| # | Decision | Binding on | Manual edit required |
|---|---|---|---|
| D1 | moments pass + `failed(orders_ok)` scores **8**, via `rubric_sde`'s row and wording | 1a | `project_manual.md:122-128` |
| D2 | **one-sided band when the floor's source is `kernel_default`**, *and* Phase 6 declares the floors | 1c, Phase 6 | — |
| D3 | the machine-readable generator goes under a **new evaluated path**, `verification.generator.*` | 2b | `schema.py` comment only |
| D4 | the KS tier **standardizes the sample first**, then tests shape | 3b | — |
| D5 | `stability_check` **replaces** moment scoring when non-null, matching the oracle | 5a | `project_manual.md:310` |
| D6 | `compare` runs **both** seedings; only the independent-seed one grants the break | 4b | `verification_manual.md` §22 |

### D1 — the orders-fail row scores 8, not 7

Chosen because the alternative creates the divergence this plan exists to remove: `rubric_sde`
already scores *moments pass, `orders_ok` fails* at 8 ([`score.py:356-363`](../verifylib/kernel/score.py#L356)),
and transcribing the PDE side's 7 would make the same defect worth 7 on Path A and 8 on Path B.

Implementation: reuse `rubric_sde`'s row verbatim, including its reason text, and add the
[REV 14](3) branch so the reason distinguishes the two defects it now covers:

* `observed_order` within noise of zero → *"does not consume the supplied Brownian increments
  (manual §18)"*, which is the existing wording.
* `observed_order` finite but below the floor → *"converging slower than the scheme claims"*.

Row order on Path A follows `rubric_sde`'s, per findings §7.6: `crashed`, `gate_violation`,
`failed(dynkin_ok)`, reference-unavailable, `failed(resolved)`, then the passing rows.
**Acceptance criterion 2 is restated accordingly** — see [§6](#6-acceptance-criteria)'s [REV 32]
table, row 2.

### D2 — one-sided band on a defaulted floor, and Phase 6 declares the floors

Both, because they are complementary rather than alternatives. The band rule is the guard that
makes 1c safe to ship at all; Phase 6 is what makes the guard rarely needed.

* **The rule:** `guards` takes the band side from the floor's *provenance*, not only from whether
  the plan declared its scheme family. When `cfg`'s `expected_strong_order` entry has
  `source == "kernel_default"`, the band is one-sided regardless of `declared`. The kernel's own
  rule already states the principle — *"an inference must never make a test stricter"* — and a
  floor nobody declared is an inference.
* **Why not the stricter option** (refuse to run the ladder without a declared floor): it converts
  a missing spec field into a *missing measurement*, which is how Finding 1 happened. A one-sided
  measurement is weaker evidence than a two-sided one and much stronger than none.
* **Phase 6 remains a precondition of 1c** ([REV 27]), so the one-sided path is a fallback for
  specs outside this benchmark rather than the normal case here.

Superseded note: [REV 16](c)'s roundoff floor is **not** implemented separately — [REV 36a]'s error
bar on `strong_order` subsumes it. One mechanism, not two.

### D3 — `verification.generator.*`, a new evaluated path

Chosen over amending `GUIDANCE_PATHS` because the existing classification is correct and worth
keeping: `drift_expression` and `diffusion_expression` are read by agents, prose belongs in them,
and `schema.py` says a program there is the shape the archived Heston leak took. Promoting them to
evaluated fields would delete a real guard to save a schema entry.

Implementation:

* Add `verification.generator.drift` and `verification.generator.diffusion_matrix` to
  `ANSWER_KEY_PATHS`, so `check_expressions` polices them the way it polices
  `verification.residual_operator`.
* `dynkin.observables_payload` reads `verification.generator.*` first and falls back to the prose
  fields only when they parse as expressions — which also delivers [REV 8]'s evaluability guard,
  the thing that stops m13's prose drift from crashing the child.
* The prose fields stay in `GUIDANCE_PATHS` untouched. `schema.py` gains a comment recording why
  the two live side by side, because the next reader will otherwise merge them.

### D4 — the KS tier standardizes before testing

Chosen because it is the only one of the three that needs no per-family calibration and the only
one that still tests something after the fix. Subtracting the sample mean and dividing by the
sample standard deviation removes exactly the `O(dt)` bias in the first two moments that
[REV 23]'s measurement showed driving the rejection, and leaves the **shape** — which is what
Finding 4's three defect classes actually corrupt (Rademacher increments, a post-step Milstein
correction, a short-period RNG all distort shape while matching moments).

* The reference law is standardized the same way, analytically where it is available and by its
  own first two moments otherwise.
* The first two moments keep their own check — that is 3a and the cheap tier of 3b, which already
  compare them against the reference with confidence intervals. The KS test is not asked to
  re-police them.
* A `dt`-scaled bar was rejected because `C` would need calibrating per family on honest solvers,
  i.e. the same work as tuning the order floors, for a weaker guarantee. Diagnostic-only was
  rejected because an ungated check does not satisfy acceptance criterion 4's shape.

### D5 — `stability_check` replaces moment scoring

Follow `project_manual.md:304` and `benchmark/verify.py:422`: when `stability_check` is non-null the
moment tolerances do not apply and the declared criterion is the score. The oracle is what the
benchmark is graded against, so a kernel that layered a gate *on top* would diverge from the thing
it is meant to reproduce.

Implementation, in the SDE driver:

* Non-null `stability_check` → evaluate it (`finite` over `terminal_paths`; `domain` as
  `max |X| < abs_bound`), record it as the scoring criterion, and skip the moment comparison rather
  than reporting it `unavailable`.
* A non-null, non-`dict` value is a **schema error** ([REV 25](2)), not a silently ignored field.
  That is what the one stale `sde_fene_blowup` instance would hit today.
* **`project_manual.md:310` is corrected in the same commit** ([REV 9]): a spec with
  `has_analytic_solution: false` and `stability_check: null` is *not* a formulator error, because
  the Path-B surrogate tier now scores it. Two live specs are deliberately in that state.

### D6 — run both seedings; only the independent one earns the break

Adopt Phase 4b's own recommendation. `_compare_sde` runs the pair twice: once at the shared seed,
whose near-noise-free difference is the right instrument for *detecting* disagreement, and once at
independent seeds, which is the only one that may set `cross_plan_agreement: True` and feed
`agreement_is_break`. A shared-seed agreement is recorded as a diagnostic and never as evidence.

* **Cost, stated as [REV 24] asked:** four solves per `compare` instead of two. On the SDE side
  `compare` is already the slowest CLI path, and §9b's timings put a `ginzburg_landau` graded solve
  in the minutes — so this is the one decision with a real wall-clock price. Accepted because the
  alternative prices a circularity break at a 10 on evidence §22 calls worthless.
* **`verification_manual.md` §22 is amended in the same commit** to record the two-seeding rule and
  the CRN rationale for the detection half, rather than being left contradicting the code.

### What these decisions do not settle

Both of today's measured defects outrank all six and are not design calls — they are bugs with
measured causes, and they belong before step 1:

1. **`_widen`** (Finding 15's resolution): one spec loses its only reference tier to a guard that
   widens a one-sided domain across the origin.
2. **`strong_order`'s missing error bar** ([REV 36a]): a 12.4x spread reported as a pass in all 11
   runs, which also subsumes D2's superseded roundoff floor.

[§7](#7-revised-shipping-order-rev-2)'s step 0 carries both.

*Rev 3 note, 2026-10-14: the sentence above — "not design calls" — was right about the bugs and
wrong about the fixes. Both step-0 items turned out to carry an unmade decision of their own, and a
third defect in how the error bar reaches `orders_ok`. See
[§11 D7 and D8](#11-decisions--resolved-2026-10-14), and [REV 37] for the defect.*

---

## 11. Decisions — resolved 2026-10-14

Rev 3's reading turned up six more forks, **two of them inside [§7](#7-revised-shipping-order-rev-2)'s
step 0** — the step §10 promoted above all of D1–D6 on the grounds that it held "bugs with measured
causes, not design calls". The bugs are measured; their fixes were not chosen. Same convention as
§10: each row is binding on the item it names, and any manual edit a decision implies ships in the
same commit.

| # | Decision | Binding on | Decided by |
|---|---|---|---|
| D7 | `_widen` **pins** a boundary that a declared gated support already fixes, and widens the rest as today | step 0 | recommendation |
| D8 | `strong_order` **abstains** when its error bar admits more than the band; `guards` gains a third value; a second scale statistic ships as a diagnostic | step 0, step 7 | recommendation |
| D8b | an abstaining order study **costs nothing** | 1a, 1c | **user, 2026-10-14** |
| D9 | the **median** z decides, and replication fires **only when one seed lands near the bar** | step 5, step 8 | recommendation |
| D10 | Path A substitutes `moments_ok and resolved and constraints_ok` for `rubric_sde`'s `surrogate` conjunct, and never falls to the 3 row | step 9 | recommendation |
| D11 | 3b tier 1 ships **report-only** until a standardized reference is declared | step 3, Phase 6 | recommendation |
| D12 | **fix** `ranking.order_margin` with the same band rule rather than delete it | step 0 | recommendation |

### D7 — pin boundaries a declared support already fixes

Hold any boundary that coincides with a support the spec declares through a **gated** constraint,
and widen the rest by the factor exactly as today.

* On `sde_ginzburg_landau_s6` that is `lo = 0` against
  `{"name": "positivity", "expr": "X > 0", "gate": true}`, so `[0, 24]` widens to `[0, 36]` — the
  far-field widening the formulator actually measured, reproducing their recorded 0.36% to three
  figures ([§2o](#2o-finding-15--the-kolmogorov-surrogate-no-longer-passes-its-own-trust-guard-m-rev-2)).
* On the other three specs declaring `kolmogorov.bounds` nothing is pinned, so symmetric widening
  survives unchanged and their guards are not re-measured.

**Why not "widen away from `x0` only".** It is spec-independent, which is its appeal, and it changes
the widened domain on `sde_bm_with_drift` (`[-4, 7]`, `X_0 = 0`) where the current rule works. A fix
for one broken spec should not silently re-measure three working ones, and the information needed to
do it properly is already in the specs. If a spec ever needs a support that no gated constraint
declares, add the declaration rather than falling back to a heuristic — that is [REV 29]'s pattern
applied, not an exception to it.

**The guard's own semantics settle it.** A truncation guard asks whether the *artificial* boundary
moves the answer. A boundary coinciding with the state space's edge is not artificial, and widening
across it — across the point where `g(x) = sigma*x` degenerates, into a region the process never
visits — measures the cost of a different problem. That is why the reference came out identical
(`mean = 0.68003`, `variance = 1.88808`) either way and only the verdict moved.

### D8 — abstain, and add a second statistic; abstention costs nothing

Three parts. The first two are the correctness fix; the third is what stops 1c gating on nothing.

**1. The abstention has to reach `orders_ok`.** Per [REV 37], returning `None` from `strong_order`
alone produces `orders_ok: False`, a verdict of failure. So:

* `strong_order` returns `se_s0`, `se_s1` and a propagated `se_order` in its detail — jackknife over
  the paths, one pass, no resampling loop.
* `guards` gains the third outcome: when `se_order` admits more than the band width, or either `s_i`
  is within `floor_mult * se` of zero, `orders_ok` is the literal `"unavailable"` rather than
  `False`. That is `weak_order`'s own `floor_mult` test and `weak_ok`'s own `is None` forgiveness,
  applied one line up.
* `_orders` propagates it, which it already does for a crashed level.

This also fixes the **live** `s0 <= 0.0` case, where an exactly-integrating scheme is failed today
rather than excused. [REV 16](c) read that as a missing guard; it is a guard returning the wrong
outcome.

**2. An abstention costs nothing — decided by the user, 2026-10-14.** `"unavailable"` is in `SKIPS`,
so `passed()` and `failed()` both refuse it, every rubric row falls through, and a plan with
otherwise clean evidence keeps the score it has today.

The alternative considered was a ceiling at 9, by analogy with `A_unavailable` for an unevaluable
reference. It was rejected because the two cases record *whose* limitation they are: an unevaluable
reference is a property of the spec, while an unmeasurable strong order on heavy-tailed
multiplicative noise is a property of the kernel's estimator. Charging the second to the solver is
the false hard-fail on correct work this repo keeps rediscovering —
[plan-no-closed-form.md](plan-no-closed-form.md)'s I9, I10 and I15, and
[§4a](#4a-canaries-to-add)'s own rule that *a false hard-fail on correct work is invisible as a
false positive*. [REV 44]'s new 1a row therefore **reports** the abstention and does not cap on it.

**3. A second statistic, because an error bar makes a bad estimator abstain rather than measure.**
[REV 36a] diagnosed the cause correctly: at `sigma = 6` the pathwise differences
`|X^dt - X^(dt/2)|` are violently heavy-tailed, so `s0` and `s1` are sample means with enormous
standard errors and `log2(s0/s1)` inherits all of it. An error bar is the honest response, and on
its own it leaves 1c with no verdict on exactly the problems where the order study is the only
scheme evidence there is.

So report a second scale statistic beside the mean — the **median** absolute pathwise difference,
which has bounded variance under the same tails and yields the same exponent whenever the tail index
does not drift across the ladder:

* Ship it as a **diagnostic** in step 0, next to the mean-based estimate and its bar.
* Promote it to the gate only on evidence: it must track the mean-based estimate on
  `3-log-transform-euler`, where the log transform makes the noise additive and the mean-based
  estimate is already tight to 1.1x across 11 runs. That plan is the control; plans 1 and 2 are the
  cases.
* The measurement is one re-run of the study already on disk as
  [`sde_dynkin_seed_study_2026-10-04.json`](../benchmark/results/sde_dynkin_seed_study_2026-10-04.json).

**What this does not settle, and `REPORT.md` should say so:** where the order study abstains and no
surrogate exists, Dynkin is the only remaining evidence that the paths came from the SDE. That is
`ginzburg_landau_s6` plans 1 and 2 exactly, which is why [REV 35b]'s result — the Dynkin failure
there is real and seed-independent — carries more weight than it looks.

### D9 — the median verdict, and replicate only near the bar

Two halves, answering [REV 42]'s two complaints.

**The rule: the median z decides.** With `k` seeds, the plan fails Dynkin when the median z exceeds
the Bonferroni bar.

* **Why not all-must-pass.** It is *stricter* than one seed — any seed can fail the plan — which
  inverts 4a's purpose. `dynkin.py:40-48` asks for replication to stop a seed-driven z from capping
  a correct solver, not to hand it `k` chances to cap it.
* **Why not pooling.** Averaging across independent seeds genuinely reduces the martingale-mean term
  and is the better *estimator* — which is both the argument for it and, here, against it: it makes
  the test ~`sqrt(k)` more powerful and would fail plans that pass today, with no measurement of
  what it fails. This repo has a stated preference for the other error ([REV 28]), so the median
  decides and the pooled z is **recorded alongside** as a diagnostic. If the two ever disagree on a
  live plan, that is the signal to revisit, and recording it costs nothing.

**The trigger: replicate only when one seed lands near the bar.** [REV 42](2)'s ~21x is the price of
replicating unconditionally. Run one seed; replicate to `k = 3` only when `|z - bar|` falls inside a
margin of about `2 * sd`, i.e. ~0.7 at the sd ≈ 0.35 [REV 35b] measured.

* **Same protection where it matters.** [REV 35b] is what licenses this: the two failing plans sit
  at z ≈ 3.2 and the passing one at z ≈ 1.3, so no group is near the bar and none would trigger a
  replicate. The plans 4a exists to protect are the marginal ones by definition, and those are
  exactly the ones inside the margin.
* **Cost returns to ~1x on the current pool** and rises to 3x only on genuinely undecided plans,
  which is where the compute belongs.
* **Record the margin decision in the metrics**, so a plan graded on one seed says so. "This plan
  was not marginal" is evidence; silence is not.

`seed_replicates` keeps its meaning as the replicate count. What changes is that the margin drives
it rather than a spec opting in — which is what [REV 24] and [REV 35b] asked for on different
grounds.

### D10 — what Path A substitutes for the surrogate conjunct

Per [REV 38], `rubric_sde`'s 8 row is guarded by
`surrogate and moments_ok and resolved and constraints_ok` and its 3 row is not guarded at all. Path
A has no surrogate, so the transcription must say what replaces it:
**`moments_ok and resolved and constraints_ok`** — the same conjunction with the Path-A reference in
the surrogate's place, which is what `rubric_sde_analytic`'s own 10 row already requires.

* A Path-A plan whose moments match, whose MC is resolved and whose constraints hold, with
  `failed(orders_ok)`, scores **8**. That is D1 as intended.
* A Path-A plan failing one of those conjuncts is already handled by an existing row
  (`failed(resolved) -> 6`, `gate_violation -> 4`, the `variance_ok` row at 8) and must **not** fall
  through to a transcribed 3. Path B's unguarded 3 row exists because a Path-B plan with no
  surrogate and no order evidence has nothing left; a Path-A plan always has the closed form.
* Assert it as a row-**order** test in the parity suite
  ([§4b](#4b-two-structural-additions-worth-more-than-any-single-canary)), not only as a score test,
  because the defect is an ordering one.

**The two step-0 items are coupled through this.** Fixing `_widen` restores
`ginzburg_landau_s6`'s surrogate, which moves a step-0-induced `orders_ok: False` on those plans
from the unguarded 3 row to the guarded 8 row. Under D8b they abstain rather than fail, so nothing
moves on today's pool — but the coupling is real, and §7's step 0 should stop describing its two
items as independent.

### D11 — the cheap tier ships report-only until a law is declared

Per [REV 41] no live spec declares a third or fourth moment and the surrogate ladder returns two, so
tier 1 can compute a sample statistic and has nothing to compare it against.

* **Step 3 ships tier 1 as a reported diagnostic**: the third and fourth standardized central
  moments with their standard errors, in `m["summary"]`, no verdict. Still worth shipping first — it
  is free, it is what a reviewer reads, and it is the data that would calibrate a bar later.
* **The reference is a Phase 6 field, shared with the KS tier.** Where `terminal_law` is declared the
  standardized targets follow from it analytically; for every Gaussian-law problem they are simply
  0 and 0, which is what `sde_ornstein_uhlenbeck`'s note already says in prose. Declaring the law
  once serves both D4's KS test and tier 1. Separate `skewness_value_at_T` / `kurtosis_value_at_T`
  fields were rejected as a second source of truth for the same fact — [REV 21]'s objection, reused.
* **Tier 1 gates only once the law exists**, at which point it is the same three-way outcome as
  every other moment comparison and needs no new machinery.

This narrows [REV 23]'s recommendation rather than contradicting it: the cheap tier still ships ahead
of the KS tier and still carries no false-rejection risk. What it does not do is grade anything
before Phase 6, so [§6](#6-acceptance-criteria) must not credit it with catching Finding 4's defect
classes until then.

> **[REV 54] — the third bullet is withdrawn: tier 1 does not gate once the law exists, measured.**
> "Tier 1 gates only once the law exists, at which point it is the same three-way outcome as every
> other moment comparison" assumes the standardized third and fourth moments behave like the first
> two. They do not, at any affordable `dt`, and both candidate bars fail on measurement:
>
> * **At `ci_mult · se` the bar rejects the honest control.** On the 2-D GBM canary, correct
>   Euler–Maruyama: skewness **0.892** against an exact 0.950 at `se` 0.026 — **2.2 σ**; excess
>   kurtosis **1.319** against 1.645 at `se` 0.120 — **2.7 σ**. The kurtosis is low by ~20% on
>   *both* components, which is the signature of scheme bias rather than a defect. The same thing on
>   the Feller-violated CIR canary: excess kurtosis 14.61 against an exact 17.06, 14% low.
> * **A relative tolerance loose enough to pass that catches nothing.** It has to admit a 20%
>   kurtosis error on correct work, and no defect class in §4a misses by more.
>
> So tier 1 ships as **D11's own default** — `outcome: None`, `graded: False`, and a `ci_mult_sigma`
> diagnostic beside each statistic — and stays there after Phase 6 rather than graduating. The
> reason is the same one D4 gives for standardizing before the KS test, read in the other direction:
> **the third and fourth raw standardized moments of a weak-order-1 scheme carry its `O(dt)` bias at
> several σ**, and standardizing is what removes it. That is why the KS tier grades and this one
> does not — it separates the Rademacher defect at 2.9× its bar and the truncation atom at 48×,
> while the honest controls sit at 0.47–0.57×.
>
> This does not cost the tier its purpose. It is still free, it is still what a reviewer reads, and
> it is still the data that would calibrate a bar later; what changes is that D11's "gates only
> once" should not be read as a scheduled follow-up.

### D12 — fix the ranking branch rather than delete it

`ranking.order_margin`'s SDE branch ([`ranking.py:251-259`](../verifylib/ranking.py#L251)) is
unreachable today — `design_order` is set only by the PDE driver — and carries the same two-sided
penalty D2 removes from `crn.guards`. Both options are therefore nearly free. Fix it, in step 0, with
the same band rule:

* One-sided when the floor's `source` is `kernel_default`, two-sided when the plan declared its
  scheme family — the same condition D2 puts in `guards`, so there is **one rule and three call
  sites**, which is what [REV 34] asked for.
* **Why not delete.** The open SDE weak-error work is what would wire `design_order` on the SDE side,
  and a deleted branch gets rewritten from scratch by whoever does it — probably with the same
  two-sided distance, since that is the obvious thing to write. A correct branch that does not run
  yet is a better artifact than an absent one.
* **It needs its own acceptance criterion** (criterion 22), asserted by calling `order_margin`
  directly, because the branch moves ranks rather than scores and no score-based criterion can see
  it. That is [REV 43]'s point and the reason this decision is worth recording at all.

### What Rev 3 leaves open

Not decisions — two measurements and one write-back, all cheap, listed so they are not mistaken for
done:

1. **Re-measure the three `ginzburg_landau_s6` plans after step 0**, and correct the seven `STATE.md`
   files ([REV 45]). Those three plans were ~31 of the baseline's 33.5 minutes.
2. ~~**`git log` on `score.py` and `dynkin.py`**, to date the row-ordering and downward-refinement
   commits against the recorded scores. Finding 15's own ask, still unrun.~~ **Done 2026-10-14 —
   see [REV 47].** Six of the seven `STATE.md` files predate both kernel changes; the dating
   question is closed and no plan needs re-examining. [REV 45]'s write-back is still open.
3. ~~**The median-vs-mean scale statistic on `3-log-transform-euler`**
   ([D8](#d8--abstain-and-add-a-second-statistic-abstention-costs-nothing)) — one re-run of the
   existing 11-run study, and it decides whether the order study can gate on multiplicative noise at
   all.~~ **Measured 2026-10-15, and the answer is no — see [REV 57].**

> **[REV 57] — the median does not rescue the order study on multiplicative noise. [M]** D8 added a
> median scale statistic alongside the mean and left open whether it is the better one. Read off
> `summary.observed_order_median` on all 11 live plans that produce an order:
>
> | plan | mean | median | `order_check` |
> |---|---|---|---|
> | `ginzburg_landau_s6/3-log-transform-euler` *(the control)* | 1.019 | **0.921** | true |
> | `ornstein_uhlenbeck/1` | 1.008 | 1.013 | true |
> | `double_well_langevin/1` | 1.046 | 1.024 | true |
> | `double_well_langevin/2` | 1.026 | 1.008 | true |
> | `double_well_langevin/3` | 1.020 | 1.013 | true |
> | `gbm_2d_high_corr/1` | 0.513 | 0.504 | true |
> | `cir_feller_violated/1` | 0.384 | 0.443 | true |
> | `cir_feller_violated/2-milstein` | 0.326 | **0.631** | true |
> | `ginzburg_landau_s6/1` | 0.452 | **0.156** | unavailable |
> | `ginzburg_landau_s6/2` | 0.644 | **0.778** | unavailable |
> | `bm_with_drift/1` | 2.238 | **2.939** | unavailable |
>
> Two patterns, and neither favours the median:
>
> * **Where the mean works, the median agrees and is biased slightly low.** On the five
>   well-behaved plans it lands within 0.04 of the mean and is *lower* in four of five — including
>   on the control, where 0.921 is further from the true 1.0 than the mean's 1.019. Switching to it
>   would tighten a one-sided band against correct solvers for no gain.
> * **Where the mean is unusable, the median is not more credible — it just disagrees.** On the two
>   abstaining `ginzburg` plans it differs by 0.30 and 0.13 with no reason to prefer either (the
>   mean's own `se` is 0.52 and 0.42, which is why they abstain), and on `bm_with_drift` — the pure
>   roundoff case — both numbers are nonsense at 2.24 and 2.94. `cir_feller_violated/2-milstein`
>   has them a factor of two apart, 0.326 against 0.631.
>
> So D8's **abstention** is what makes the order study honest on heavy-tailed multiplicative noise,
> not the second statistic. The median keeps its place as a recorded diagnostic — which is how D8
> ships it, and the disagreement is itself the signal worth reading — but it must not become the
> verdict. [REV 36a]'s conclusion stands unchanged.

---

## 12. Implementation log — 2026-10-15

All eleven steps of [§7](#7-revised-shipping-order-rev-2) landed, in the order §7 gives. This
section records what shipped, what the measurement said, and the seven things the plan got wrong
that only running it could show. The companion's
[§L](expectations-sde-verification-gaps.md#l-measured-after-implementation) carries the field-by-field
before/after: **one score moved and ~~35~~ 61 `(plan, field)` pairs moved across 9 of the 12 live
plans** — the 35 was counted by hand against the fields §E could name and is corrected in
[REV 69], which re-derives it from §B's own three tables. The plan count was right both times.

### 12a. What shipped

| | |
|---|---|
| Kernel, non-test | **15 files, +2 173 / −178 lines**; `sde/driver.py` +810, `sde/dynkin.py` +364, `sde/crn.py` +226, `score.py` +181, `schema.py` +143, `cli.py` +138 |
| New kernel module | `verifylib/kernel/sde/laws.py` (151 lines) — the declared terminal law and the standardized KS test |
| Tests | **+63**, 587 → 650 collected; two new files (`test_kernel_sde_path_a.py`, 19 tests; `test_spec_field_coverage.py`, 4) and +25 / +4 / +3 / +2 / +2 / +2 in `test_kernel_sde_path_b.py`, `test_kernel_invariants.py`, `test_schema.py`, `test_kernel_sandbox.py`, `test_ranking.py`, `test_kernel_score.py` |
| New canaries | 12 solvers and 5 specs, including the four Milstein variants §4a asks for and `sde_cir_truncated.py` + `sde_spec_cir.json` from [REV 56] |
| Manuals and agents | 5 files, +326 / −24; `verification_manual.md` §19–§22 and §24, `project_manual.md`'s analytic SDE rubric, three agent prompts |
| Suite | **643 passed, 3 skipped, 4 deselected, 0 failed**, 9m52s (563.9s for `verifylib/tests` + 29.0s for the rest) against [REV 48]'s baseline of 580 / 3 / 4 at 368.9s |

The suite's wall time grew 2.6×, and 11 SDE tests are 300s of the 564s. That is the cost §4d
predicted for 1b and the ladders, paid once per run.

### 12b. Seven things the plan had wrong, found by running it

Each is a `[M]` revision above, with its measurement. Six correct the plan; one is a bug the plan's
own step 8 introduced.

| | What the plan said | What the measurement said |
|---|---|---|
| [REV 51] | the post-step Milstein correction is caught by the order study | the rate is unchanged — **0.998 against 0.998** — because the perturbation is `O(dt^{3/2})` per step with zero mean |
| [REV 52] | Rademacher increments are caught by the distributional check | **only on multiplicative noise**; on additive noise the CLT closes the window and the canary must *pass* |
| [REV 53] | — | **both ladders supply their own increments**, so the increment law is structurally invisible to the order study and to Dynkin alike |
| [REV 54] | tier 1 gates once a law is declared | at `ci_mult·se` it **rejects the honest control at 2.2–2.7 σ**, and no looser bar catches anything; report-only is permanent |
| [REV 55] | steps 2 and 8 are independent | they multiply: m13's Dynkin increments reach **41.6 GB**, now refused at a 12 GB guard |
| [REV 56] | step 8's risk is "moderate — mitigated by steps 5 and 6" | it **capped a correct solver at 3**; neither mitigation covered an order-0.2 residual, and the fix is an abstention |
| [REV 57] | D8 left the median-vs-mean question open | the median is **biased low where the mean works** and no more credible where it does not |

[REV 56] is the one worth reading twice, because [§F2](expectations-sde-verification-gaps.md) called
it in advance — *"`cir_feller_violated` is the one to watch: full-truncation Euler on a
Feller-violated CIR has real mass near zero"* — and the prescribed mitigation did not cover it. F2
expected a **seed-driven** false failure and specified D9's median-over-3-seeds as the remedy. The
failure was not seed-driven: at `|z − bar| = 2.71` the replication correctly declined to fire, and
the cause was the ladder's extrapolation assumption rather than the seed. The prediction was right
and the remedy was aimed at the wrong mechanism.

### 12c. Deliberate scope limits

Three things inside the plan's phases were not done, and are not defects in what shipped:

* **Phase 6 covers the kernel requirement and three live specs, not the 22 stale ones.**
  `schema.check_strong_order_declared` makes a missing `expected_strong_order` a schema finding, and
  `terminal_law` was declared on the three live specs that already state the law in prose. Writing
  `verification` blocks for the specs in `workspace_stale_pre_c2b_backup/` is formulator work; it
  would have meant inventing references.
* **`sde_cir_feller_violated` gets no `terminal_law`,** even though [REV 56] measured that declaring
  one would catch a real defect in the scheme at 48× the bar — 12% of the paths sit exactly at zero
  where the exact law has no atom. The same reason: the exact noncentral-chi-squared law is standard
  CIR theory but that spec does not state it, and adding it would move a live plan's grade on a
  reference the formulator never declared. It ships as a canary assertion and a flagged finding.
* **2b's vector generator ships as `verification.generator.*` per D3 and is exercised only by
  `sde_gbm_2d_high_corr`,** the one live spec that declares it.

  ~~The sandbox implementation is validated to 1.6e-9 against the exact generator of `phi = XY` on
  2-D GBM, which is the round-trip §4c asks for.~~ **[REV 70]: that validation was done by hand and
  was not in the suite**, which made it the newest and most delicate numerics in the change — a
  central-difference Hessian at `h = 1e-4·(1 + |X|)`, chosen at `eps^(1/4)` — guarded by one live
  plan's Dynkin `z`. A step-size regression would have surfaced as a shifted `dynkin_z` on
  `sde_gbm_2d_high_corr` and been read as a property of the solver. [REV 61] puts five assertions
  behind it, each against a generator differentiated by hand:

  | what it pins | how |
  |---|---|
  | the mixed partial | `phi = XY` on 2-D GBM, where `d_ii phi = 0` for both components so the **whole** second-order term is the off-diagonal one. Max rel err **8.3e-9** |
  | the diagonal | `phi = X**2`, which a cross-term-only implementation would miss. **5.1e-9** |
  | every pair, not just `(0,1)` | `phi = XYZ` at `d = 3` with a dense non-square `G` (`m = 2`): each of the three pairs carries exactly one term, so a loop covering one pair is off by two |
  | the step size | `phi = exp(X+Y)`, every derivative equal to itself. Measured by substituting the constant: `h0` of 1e-2/1e-3/**1e-4**/1e-5/1e-6 gives rel err 1.6e-4 / 1.6e-6 / **1.8e-8** / 1.2e-6 / 1.0e-4. The shipped value is the balance point and the test's 1e-7 bar sits inside its nearest neighbours, so **one decade either way fails** |
  | `m`-independence | two `G` matrices with the same `G Gᵀ` at widths **2 and 26** agree *exactly* (max difference 0.0) and both match the hand answer to 7.2e-9, which is why `m13`'s 13 channels cost what two cost. The assertion keeps a 1e-9 tolerance rather than bit-equality, since a refactor that reorders the `G Gᵀ` sum is not a defect |


---

## 13. Second round — from an independent evaluation, 2026-10-15

A separate session graded the shipped work against
[the companion](expectations-sde-verification-gaps.md), reading the metrics rather than the scores.
Its verdict on what [§12](#12-implementation-log--2026-10-15) claimed: **all eight
[§F](expectations-sde-verification-gaps.md#f-what-must-not-change) must-not-change assertions hold**,
every [§I](expectations-sde-verification-gaps.md#i-if-the-expected-result-does-not-appear) failure
mode was avoided, [§C8](expectations-sde-verification-gaps.md#c8-the-kolmogorov-surrogate-works-again-on-a-one-sided-domain)'s
`truncation_shift` prediction of 0.00361 is exact to three figures, and two deliverables went past
the plan — 2b runs on a live plan, and the KS tier grades rather than reporting.

It also found **nine places the work fell short**. Three are mis-prices — a grade the evidence does
not support, in one case *below* what it does support; six are gaps, where a shipped mechanism was
correct and unguarded. [REV 58]–[REV 70] close all nine bar one, which is spec work this round still
declines to do ([§13d](#13d-what-this-round-still-leaves-open)). [REV 71] is not on the evaluation's
list at all: it is a **second instance of [REV 66]'s defect class**, found while measuring [REV 64],
and it is recorded rather than fixed for the reason §13d gives.

### 13a. The three mis-priced grades

Two are grades too high and one is a grade too low, which is worth separating because only one
direction is a hole. **One of the three is live** — [REV 60]'s third door, which is why one score
moves — and the other two are not reachable by any plan in today's pool, which is exactly why the
sweep that signed off on round 1 found neither. A hole no current artifact walks into is still a
hole, and it is cheaper to close now than when the plan that reaches it arrives and its grade has to
be explained.

**[REV 58] — a reference the kernel threw away (the grade too *low*).** [§C7](expectations-sde-verification-gaps.md#c7-a-half-usable-reference-stops-being-thrown-away)
named this in advance and the fix landed on half of it. `_moment_observables` fell back to
`{kind}_value_at_T` when the matching expression was **absent**, and the shape Finding 6 is actually
about is an expression that is *present and unevaluable*: `sde_quintic_random_ic` declares
`variance_expression: "scipy.integrate.quad(...)"` beside `variance_value_at_T: 0.09248275891171216`.
`evaluate` refuses `scipy` by design, so the comparison loop raised `NameError`,
`driver._reference_outcome`'s bare except turned it into `"unavailable"`, and the plan landed at
`tier_of → A_unavailable → ceiling() → 9, analytic_unvalidated` — the exact outcome [REV 17]'s
branch exists to prevent, on a spec that does commit to a number. Keyed on the evaluator now, not on
the field's presence. Measured by grafting the shape onto the live OU spec:

| | before | after |
|---|---|---|
| `reference_outcome` | `unavailable` | **`validated`**, `max_rel_mismatch` 1.4e-14 |
| the same shape with the literal wrong by 25% | `unavailable` | **`failed`** at 0.20 |
| the same shape with **no** literal | `unavailable`, no reason | `unavailable`, reason naming the observable and the time |

The third row is the one that says the fallback grades rather than rubber-stamps.

**[REV 59] — a crashed ladder collecting an honest abstention's credit.** `ceiling` priced D8b as
`passed(order_ok) or is_skip(order_ok)`, which credits **all seven** members of `SKIPS`. `_orders`
returns `"unavailable"` both for the measured abstention *and* for a CRN level that crashed —
`m13`'s `IndexError` was recorded under that exact token, as [§B1](expectations-sde-verification-gaps.md#b1-the-order-study)
shows. So a plan whose ladder produced no number at all could buy the circularity-break 9 that D8b
reserves for a ladder that produced numbers the estimator could not read. `ceiling` already draws
this distinction thirty lines above, for the solver: *"a crash is a property of the code"*.

The abstention now has its own token, `"unresolved"` — the word the rest of the kernel already uses
for "the check, not the solver, is the limit" — and the credit is an allow-list:

| `order_ok` | before | after |
|---|---|---|
| `True` | 9 | 9 |
| `"unresolved"` (measured abstention) | 9 | 9 — D8b, intact |
| `"not_run"` (§24's cost skip) | 9 | 9 |
| `"unavailable"` (**a level crashed**) | **9** | **8** |
| `faulty_probe`, `not_reported`, `unshadowable`, `waived_chaotic` | **9** | **8** |
| `False` | 8 | 8 |

`ranking.order` reads the vocabulary rather than the one literal now, for the same reason: an
exact-string test there would have gone on ranking plans on the estimate `guards` had just declined
to read, and that moves ranks rather than scores, so nothing would have said so.

**[REV 60] — a 10 with no generator evidence.** This is the largest of the three, and it is
Finding 1's own hole reached through three doors the implementation left open. Both SDE rubrics read
a *failing* Dynkin — `failed(dynkin_ok)` caps at 3 — and read one that **never reached a verdict** as
nothing at all, so the 10 rows fell back to moment evidence alone. Step 1b's entire argument is that
exact terminal moments cannot establish that the paths came from this SDE, and `_dynkin`'s own
`DYNKIN_MAX_BYTES` docstring says the quiet part: *"it does not certify today, so the ladder is not
run there; the guard is here because 'it does not certify' is luck rather than design."* The same
sentence applied to the missing cap.

Three doors: a plan whose increment array exceeds the 12 GB budget (the refusal is right — running
it at fewer paths could turn a real inconsistency into a pass); a plan whose generator cannot be
built or whose observables run crashed; and a plan where the extrapolation ladder abstained
([REV 56]). A 10 now requires `dynkin_ok is True`; anything else that is not a failure is **9**,
provenance unchanged, with a reason that names which door.

The third door is the one with a live consequence, and it is worth stating plainly because it
revises [REV 56]'s pricing without revising its verdict. `sde_cir_feller_violated/2-milstein`
abstains for a well-argued reason and the solver is correct — so it is not a 3, which is what a
Dynkin *failure* earns. But nothing in that run establishes that its paths came from this SDE, and
§12c had already measured that declaring the exact terminal law there would catch a real defect in
the scheme at **48× the bar**. A certified 10 on the one live plan whose scheme is known to be
wrong was not a defensible resting place. D8b's "an abstention costs nothing" is a decision about
**the order study**; the plan never extended it to the one check that cannot be substituted, and
this round does not either.

Worth saying what that plan's route back to a 10 now is, because it is not a declaration. Declaring
the exact terminal law would make the KS tier *fail* it — that is the 48× measurement — so the spec
cannot buy the grade back. Neither can more paths: the residual is already 11 s.e. clear of its own
noise, and the thing that cannot be resolved is the **rate**, not the level. What would work is a
scheme whose residual converges at a rate three levels can locate — on this problem an exact
transition-law sampler is the one case that is certain, since it has no discretization error at all
and its residual is then the trapezoid path integral's own `O(dt)` bias, which the ladder resolves.
Whether a drift-implicit square-root scheme also clears the band here is a literature claim this
round did not measure. **The direction is the correct signal to send**, and a 10 was sending the
opposite one.

### 13b. The six unguarded mechanisms

| | what was unguarded | what now holds it |
|---|---|---|
| [REV 61] | `_generator_nd`, the newest numerics in the change, had **no test in the suite** — a central-difference Hessian at `h = 1e-4·(1+|X|)` guarded by one live plan's `z` | 5 assertions against hand-differentiated generators, incl. a step-size guard **one decade wide** in each direction |
| [REV 62] | `_milstein_claim` and `stability_ok` shipped correct, unexercised and unasserted | the cap asserted through `certify` (10 → 8, `bound_by: milstein_claim`), and 7 parametrized cases for the **inverse** false positive, which is the half that matters: capping the three `zero_diffusion_derivative` specs would be a false hard-fail on live plans |
| [REV 63] | criterion 7's per-family half was **prose inside an error message**: a spec with a Milstein plan and no `_milstein` floor passed the check, which is how `cir_feller_violated`'s floor came to be fixed by hand | the floor is required wherever the spec does not say a Milstein plan is impossible, plus a parity assertion against `crn.FAMILY_ORDER_KEY` so a new scheme family cannot reopen it |
| [REV 64] | `_headroom` divided by the declared tolerance where `_martingale` judges against `ci_mult·se + tol` | the bar the check actually applied. `gbm_2d_high_corr` declares no `tol`, so `effective_tol` floored at `ROUNDOFF` and its **passing** martingale reported **−6.75 decades** on a field [§A1](expectations-sde-verification-gaps.md#a1-the-fields-that-carry-the-plans-claims) lists as a ranking anchor. Measured live after the fix: bar `1e-10` → **2.744e-3**, headroom **−6.747 → +0.692**, and `ranking.structure` on that plan **0.0 → 0.231** |
| [REV 65] | "has a consumer" was a text search over file contents, so **a docstring mention counted** | `ast.unparse` with docstrings stripped. Five live fields turned out to have nothing but prose and are recorded; `required_components` is the instructive one — `_compare`'s docstring names it *while saying the kernel checks every declared moment regardless* |
| [REV 66] | prose in an invariant `expr` (`"X ~ Normal(X_0 + mu*T, sigma**2*T)"`) is a permanent `not_reported` with a `SyntaxError` | refused at the chokepoint. `operator.is_program`'s docstring already used **this very string** as its example of text that parses as neither an expression nor a program |
| [REV 71] | **not on the evaluation's list** — found while measuring [REV 64]. Three live specs declare a `tol` on an entry that is neither a registry name nor carries an `expr`, so the tolerance is never applied and the entry is `not_reported` for ever | a **warning** naming each: `sde_gbm_2d_high_corr`'s `component_symmetry` (0.05), `pde_kdv_zabusky_kruskal`'s `kdv_third_invariant` (0.001), `pde_stefan_1d_similarity`'s `enthalpy_balance` (0.02). Not an error, and not written for them — see below |

### 13c. Two things the evaluation got right that were not defects

**The suite was red in its environment, and the floor was wrong.** 638 passed / 5 failed at numpy
1.26.4, all `np.trapezoid`, all PDE-side, all reproducing with the entire diff stashed. The failures
are not this change's — but `pyproject.toml` declared `numpy>=1.26.0` against 15 `np.trapezoid`
call sites, so **a numpy the project claims to support cannot run the suite**. [REV 67] raises the
floor and adds a guard that holds the declared floor to the APIs actually called; it fails at the
old floor, naming the five files.

**`m13`'s new `richardson_stable: False` is signal, not noise.** No prediction covered it and it
costs nothing today (`m13` is bound by `failed(resolved)` at 6). The question it raised is whether a
`relative_shift` of 8.5% against a 5% bar is a bias or Monte Carlo scatter, and the recorded detail
could not answer it. The two levels are a coarsening of **one** Brownian path, so the difference is
paired and its standard error is far below either mean's own. Measured:

| component | shift | relative | `se_shift` | in standard errors |
|---|---|---|---|---|
| 0 | 1.07e-2 | 9.07% | 3.21e-4 | **33.2 σ** |
| 1 | 2.30e-3 | 1.55% | 2.39e-4 | 9.6 σ |

So the `False` is correct: an explicit Euler–Maruyama on a stiff 13-channel system at `dt0 = 0.04`
really does carry that bias. [REV 68] reports the number so the next reader does not have to
re-derive it, and deliberately leaves the **verdict** on the relative shift alone — at 50 000 paths
almost any real shift clears the noise, so gating on standard errors would fire on correct plans.

### 13d. What this round still leaves open

* **Criterion 3 stays scoped to two problems, and the gap is in the artifacts rather than the
  mechanism.** `sde_gbm_2d_correlated` and `sde_log_heston_feller_violated` carry a `problem.md`
  in `workspace/` and nothing else — no spec, no plan, no solver — so "all four multi-noise
  problems" is unmet as worded, which
  [§E](expectations-sde-verification-gaps.md#not-predictable-yet)'s *"Not predictable yet"*
  anticipated. Worth separating the two halves, because they are not equally open:

  * *What the criterion is for* — that the CRN ladder completes at `m > 1` rather than raising
    Finding 2's `IndexError` — is covered. Live at `m = 2` on `sde_gbm_2d_high_corr` and at
    `m = 13` on `sde_multichannel_stiff_m13`, and as a unit round-trip at `m ∈ {1, 2, 3}`. Both
    missing problems are 2-D with one correlation parameter, so both are `m = 2`: **the value
    already covered twice.**
  * *What is missing* is two live artifacts. Producing them means running the formulator and the
    plan-creator on two problems and inventing the references they would declare, which is the same
    limit [§12c](#12c-deliberate-scope-limits) records for the 22 stale specs — not something to do
    inside a correction round.
* **[REV 71]'s three predicates are not written, deliberately.** The warning names them and stops
  there, for the same reason §12c gives for the others: writing one is formulator work with a grade
  attached. `component_symmetry` is the instructive case. Its note describes a real defect — *"a
  mismatch indicates the correlated-increment construction was applied asymmetrically (a common
  Cholesky bug)"* — on the one live plan with two noise channels, so it is exactly the check that
  problem wants. It is also **ungated**, which means an `expr` that failed would take
  `gbm_2d_high_corr/1` from 10 to the invariants row's 5. Choosing the predicate, and the paired
  rather than independent standard error it needs (the two columns are correlated at `rho = 0.9`,
  so an independent-SE bar is far too loose), decides a live grade. That is a decision to make
  explicitly, not inside a correction round.
* **`sde_cir_feller_violated` still gets no `terminal_law`.** The reasoning in §12c is unchanged.
  What *has* changed is the consequence: [REV 60] means the plan no longer certifies on moment
  evidence, so the withheld reference is no longer the only thing standing between a known-defective
  scheme and a 10.

### 13e. Measured

| | round 1 | round 2 |
|---|---|---|
| suite | 643 passed / 0 failed of 650 | **690 passed / 3 skipped / 0 failed of 697**, 614.5 s, exit 0 |
| tests added | +63 over the 587 baseline | **+47** |
| 12-plan sweep | 2 240 s, exit 0 | **2 379 s, exit 0** |
| live scores moved | 1 (`ginzburg_landau_s6/3`, 9 → 10) | **1** (`cir_feller_violated/2`, 10 → 9) |
| live `(plan, field)` pairs moved | 61 against §B *(§L said 35; re-derived in [REV 69])* | 7 across 4 plans |
| `STATE.md` vs the metrics | 12/12 | **12/12**, write-back corrected 1 value and is idempotent |

The round-2 footprint is deliberately small. Nothing in it changes what the kernel *measures* —
every measurement [§12](#12-implementation-log--2026-10-15) added is still there and still produces
the same numbers. What changed is how three kinds of non-verdict are **priced**, and only one live
plan sits on a repriced row. Four of the seven moved pairs are the `"unavailable"` → `"unresolved"`
rename landing on plans where the abstention was always the honest one; the fifth is that plan's
score; the remaining two are diagnostics ([REV 64]'s headroom and [REV 68]'s `shift_in_se`).

The field-level after-column for both rounds is
[§L](expectations-sde-verification-gaps.md#l-measured-after-implementation) and
[§M](expectations-sde-verification-gaps.md#m-the-second-round-measured) of the companion.
