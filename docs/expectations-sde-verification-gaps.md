# What the SDE Verification Plan Is Expected to Change — Measurement Ledger

**Companion to:** [plan-sde-verification-gaps.md](plan-sde-verification-gaps.md) (rev 5, 2026-10-15)
**Purpose:** the before-column, anchored to **objective field values** rather than to scores. Every
claim the plan makes, the measured state of the field that carries it today, and the value that
field must take after.
**Status:** ~~nothing in the plan is implemented.~~ ~~**Implemented 2026-10-15 — all eleven steps of
the plan's §7. The after-column is [§L](#l-measured-after-implementation).**~~ **Implemented and
then independently evaluated, 2026-10-15. The after-column is
[§L](#l-measured-after-implementation); the second round's deltas and the three §L readings it
corrected are [§M](#m-the-second-round-measured).** Every "today" value
below was read on 2026-10-14 from the 12 live `_eval_metrics.json` files or measured directly
against tree `d5bb447`, and is left exactly as it was written: it is the before-column, and
re-reading it against the implementation is the whole point of the file. §L records what the same
fields read afterwards, including the two places where this document's own predictions were wrong.
The environment every suite number was measured on is in [§J1](#j1-the-suite--2026-10-14-tree-d5bb447),
added after an evaluation ran the same tree on a different numpy and got five failures.

---

## 0. Why this file is anchored to measurements, not scores

A score is a composite: `rubric_sde_analytic` or `rubric_sde` proposes one, `ceiling()` and
`gates_clean()` and `tier_of()` can each lower it, `BANDS` widens the manual's bands into single
values, and several of the inputs are **formulator tolerance choices** rather than measurements.
Anchoring a before/after on it measures all of that at once. Four facts from the plan's own baseline
make the case better than an argument:

1. **Scores drift with no code change at all.** §9b re-ran all 12 live plans and **6 of 12**
   diverged from what `STATE.md` recorded — purely from kernel changes between the recording and the
   re-run. A score-anchored before-column is a moving baseline.
2. **One live score turns on a formulator's third decimal.**
   `cir_feller_violated/2-milstein` is a 10 rather than an 8 because its mean error is **4.96%**
   against a **5.00%** tolerance. The plan's own note: *"Treat the 10 as fragile."*
3. **The plan's most important correctness fix produces, by design, zero score movement.**
   D8b — your decision — is that an abstaining order study costs nothing. So step 0's main fix is
   invisible to every score. Scored, step 0 reads as a no-op.
4. **Where a score does not move, three objective fields do.** On `ginzburg_landau_s6` plans 1 and
   2 the score stays **3** across step 0, while `certification.tier` goes `C → A_prime`,
   `detail.surrogate_ok.route` goes `null → "kolmogorov"`, and `certification.provenance` goes
   `self_convergence → surrogate`. The improvement is real, checkable and completely invisible in
   the score.

So: **each claim below names a field, a path, a value today and a value expected.** Scores appear
only as a derived consequence, in a separately-labelled row, with a note where the value depends on
a tolerance rather than on a measurement. Where the plan words an acceptance criterion as a score,
[§D](#d-the-23-acceptance-criteria-with-an-objective-anchor) supplies the substitute measurement.

---

## A. The observable set, and how to capture it

Every claim in this file reads from one of five places. Four are files on disk; one is a direct
function call.

| Source | Path | What it carries |
|---|---|---|
| Per-plan metrics | `workspace/<slug>/plans/<plan>/_eval_metrics.json` | `checks.*`, `summary.*`, `detail.*`, `config.*`, `certification.*`, `evidence.*` |
| Spec | `workspace/<slug>/problem_spec.json` | declared fields, `verification.*`, tolerances |
| Source | `verifylib/` | grep-level facts: a field with no consumer, a check that does not exist |
| Suite | `pytest -q --durations=15` | pass/fail counts and wall time |
| Direct call | `ranking.order_margin`, `crn.guards`, `crn.strong_order`, `rubric_sde*` | the things no score-level run can see |

### A1. The fields that carry the plan's claims

| Field (path in `_eval_metrics.json`) | Why it is the anchor |
|---|---|
| `checks.orders_ok` | tri-valued today (`True` / `False` / a `SKIPS` member). Step 0's whole purpose is to make a fourth state — an abstention — reachable |
| `detail.orders_ok.strong_detail.s0`, `.s1` | the raw mean absolute pathwise differences. **Roundoff is visible directly here** and needs no interpretation |
| `detail.orders_ok.strong_band` | the literal string `"one-sided (…)"` or `"two-sided"`. Criterion 13's anchor |
| `detail.orders_ok.expected_strong_candidates` | what the one-sided bar is computed from (`min(candidates) - 0.4`) |
| `summary.observed_order`, `summary.order_floor` | the estimate and the floor it is judged against |
| `config.expected_strong_order.{value,source}` | `source` is one of `CONFIG_SOURCES`; D2 keys the band side off it |
| `config.scheme_family.{value,source}` | `"plan"` vs `"inferred"` — decides the band today |
| `config.run_crn_ladder.value`, `config.run_dynkin.value` | whether the check ran at all. Steps 7 and 8 flip these |
| `checks.dynkin_ok`, `summary.dynkin_z`, `summary.dynkin_bar` | the verdict, the statistic and the Bonferroni bar, separately |
| `detail.dynkin_ok.per_function_ok` | **which** test function failed. Far more diagnostic than the verdict |
| `detail.dynkin_ok.family_wise_multiplier` | the bar, derived |
| `detail.surrogate_ok.route` | `null` or the route name. Step 0's `_widen` fix is visible here |
| `detail.surrogate_ok.attempted.kolmogorov` | carries the truncation number in its reason string |
| `summary.reference_outcome` | `"validated"` / `"unavailable"`. [REV 17]'s second half is visible only here |
| `certification.tier` | `A` / `A_prime` / `A_unavailable` / `C` … — the evidence class, independent of the rubric |
| `certification.provenance` | `analytic` / `surrogate` / `self_convergence` / `manufactured_partial` / … |
| `certification.bound_by` | `"rubric"` or `"ceiling"` — **what** limited the score |
| `certification.caps` | every cap with its `source` and `reason`, so the binding constraint is explicit |
| `certification.gates_clean` | the hard-cap path step 8 routes Dynkin into |
| `evidence.d1_outcome`, `evidence.operator_validated` | the pair that produces the fabrication cap |
| `checks.invariants_ok`, `detail.constraints_ok.{outcomes,drifts,headroom}` | per-invariant verdicts. See [§K3](#k3-1as-invariants-row-is-not-a-transcription-and-two-live-plans-fail-on-artifacts) |
| `checks.richardson_stable`, `detail.orders_ok.richardson.stable` | the moment-side stability analogue the order study lacks |
| `detail.orders_ok.weak_ok` | `False` on 4 of the 9 plans that have an estimate — and **nothing reads it** ([REV 46]) |

### A2. Capturing a snapshot

The whole before/after is a diff of these fields. One script, no scoring involved:

```python
# dump.py — run from the repo root, before and after, then diff the two JSON files
import json, glob, sys
def g(d, *p):
    for k in p:
        if not isinstance(d, dict): return None
        d = d.get(k)
    return d
rows = []
for f in sorted(glob.glob('workspace/sde_*/plans/*/_eval_metrics.json')):
    d = json.load(open(f))
    ch, su, de, cf, ce, ev = (d.get(k) or {} for k in
                              ('checks','summary','detail','config','certification','evidence'))
    o = de.get('orders_ok') or {}
    rows.append(dict(
        plan='/'.join((f.split('/')[1][4:], f.split('/')[3])), path=d.get('path'),
        orders_ok=ch.get('orders_ok'), dynkin_ok=ch.get('dynkin_ok'),
        moments_ok=ch.get('moments_ok'), variance_ok=ch.get('variance_ok'),
        resolved=ch.get('resolved'), invariants_ok=ch.get('invariants_ok'),
        surrogate_ok=ch.get('surrogate_ok'), richardson_stable=ch.get('richardson_stable'),
        observed_order=su.get('observed_order'), order_floor=su.get('order_floor'),
        s0=g(o,'strong_detail','s0'), s1=g(o,'strong_detail','s1'),
        strong_band=o.get('strong_band'), candidates=o.get('expected_strong_candidates'),
        weak_ok=o.get('weak_ok'), richardson_stable_detail=g(o,'richardson','stable'),
        dynkin_z=su.get('dynkin_z'), dynkin_bar=su.get('dynkin_bar'),
        per_function_ok=g(de,'dynkin_ok','per_function_ok'),
        surrogate_route=g(de,'surrogate_ok','route'),
        kolmogorov=g(de,'surrogate_ok','attempted','kolmogorov'),
        run_crn_ladder=g(cf,'run_crn_ladder','value'), run_dynkin=g(cf,'run_dynkin','value'),
        eso=g(cf,'expected_strong_order','value'), eso_source=g(cf,'expected_strong_order','source'),
        family=g(cf,'scheme_family','value'), family_source=g(cf,'scheme_family','source'),
        reference_outcome=su.get('reference_outcome'), tier=ce.get('tier'),
        provenance=ce.get('provenance'), bound_by=ce.get('bound_by'),
        gates_clean=ce.get('gates_clean'), d1_outcome=ev.get('d1_outcome'),
        operator_validated=ev.get('operator_validated'),
        score=d.get('score'), certified=d.get('certified')))
json.dump(rows, open(sys.argv[1], 'w'), indent=1, default=str)
```

`score` and `certified` are captured last and deliberately: useful context, not the measure.

---

## B. The objective baseline — 12 live plans, field by field

Read 2026-10-14 from the `_eval_metrics.json` files written by the 2026-10-04 sweep (33.5 min wall,
exit 0, all 12 plans at their declared `evaluation_thresholds`).

### B1. The order study

| Plan | P | `orders_ok` | `observed_order` | `order_floor` | `s0` | `s1` | `strong_band` | `candidates` | `run_crn_ladder` |
|---|---|---|---|---|---|---|---|---|---|
| `bm_with_drift/1-euler-maruyama` | A | `True` | 2.238 | 1.0 | **9.38e-15** | **1.99e-15** | one-sided | [1.0] | `True` |
| `cir_feller_violated/1-euler-maruyama` | A | `True` | 0.384 | 0.5 | 0.0880 | 0.0674 | one-sided | [0.5] | `True` |
| `cir_feller_violated/2-milstein` | A | `not_run` | — | 0.5 | — | — | — | — | **`False`** |
| `double_well_langevin/1-tamed-heun-drift-trapezoid` | B | `True` | 1.046 | 1.0 | 0.0300 | 0.0145 | **two-sided** | [1.0, 1.0] | `True` |
| `double_well_langevin/2-drift-implicit-trapezoid` | B | `True` | 1.026 | 1.0 | 0.0294 | 0.0145 | **two-sided** | [1.0, 1.0] | `True` |
| `double_well_langevin/3-strang-splitting-exact-flows` | B | `True` | 1.020 | 1.0 | 0.0293 | 0.0144 | **two-sided** | [1.0, 1.0] | `True` |
| `gbm_2d_high_corr/1-euler-maruyama` | A | `not_run` | — | 0.5 | — | — | — | — | **`False`** |
| `ginzburg_landau_s6/1-tamed-euler-maruyama` | B | `True` | 0.452 | 0.5 | 0.723 | 0.529 | one-sided | [0.5, 1.0] | `True` |
| `ginzburg_landau_s6/2-tamed-milstein` | B | `True` | **0.644** | **1.0** | 0.450 | 0.288 | one-sided | [0.5, 1.0] | `True` |
| `ginzburg_landau_s6/3-log-transform-euler` | B | `True` | 1.019 | 0.5 | 0.0211 | 0.0104 | one-sided | [0.5, 1.0] | `True` |
| `multichannel_stiff_m13/1-euler-maruyama` | A | **`unavailable`** | — | 0.5 | — | — | — | — | `True` |
| `ornstein_uhlenbeck/1-euler-maruyama` | A | `True` | 1.008 | 1.0 | 0.00512 | 0.00255 | one-sided | [1.0] | `True` |

Five facts to read straight off it, none of which needs a score:

* **`bm_with_drift` is measuring roundoff and calling it a pass.** `s0 = 9.38e-15` and
  `s1 = 1.99e-15` are at machine epsilon for terminal values of order 1.5 — Euler–Maruyama is
  pathwise exact on `dX = 0.5 dt + 0.3 dW` and its own `SOLUTION.md` says so. The reported order of
  2.238 against a floor of 1.0 is noise that clears the one-sided bar of 0.6 by accident.
* **`ginzburg_landau_s6/2-tamed-milstein` passes a floor it misses by a third.** `order_floor` is
  **1.0** (correctly selected from `expected_strong_order_milstein` via `FAMILY_ORDER_KEY`, on an
  *inferred* family — exactly as [REV 40] describes), the measured order is **0.644**, and
  `orders_ok` is `True`, because the one-sided bar is `min([0.5, 1.0]) - 0.4 = 0.1`. This is the
  cleanest single demonstration that the one-sided band is near-vacuous.
* **`m13`'s `"unavailable"` is Finding 2, live, on disk.** `detail.orders_ok.reason` reads:
  *"the CRN level at dt=0.04 crashed: exception: IndexError: too many indices for array: array is
  2-dimensional, but 3 were indexed"*. The plan reproduces this on `gbm_2d_high_corr` with a side
  experiment; it does not need to — a live plan carries it.
* **`gbm_2d_high_corr`'s crash is latent, not live.** Its `run_crn_ladder` is `False`, with
  `detail.orders_ok.reason`: *"the moments match an exact reference and the scheme's strong order is
  below 1, so the order study cannot distinguish anything the moment comparison has not already
  settled"* — §24's exemption verbatim. **Step 7 is what makes the crash manifest there**, which is
  why step 2 must precede it. §7 orders them correctly; this is the measurement that says why.
* **`strong_band` is `two-sided` on exactly the three `double_well` plans** and one-sided on the
  other nine, confirming the 9/3 split independently of `plan_meta` (see
  [§K1](#k1-rev-33s-inferred-count-is-9-not-10)).

### B2. Dynkin, the reference and the certification path

| Plan | `dynkin_ok` | `dynkin_z` | `dynkin_bar` | `per_function_ok` | `surrogate_ok.route` | `reference_outcome` | `tier` | `provenance` | `bound_by` | `gates_clean` |
|---|---|---|---|---|---|---|---|---|---|---|
| `bm_with_drift/1` | `not_run` | — | — | — | `analytic_moments` | `validated` | `A` | `analytic` | rubric | `True` |
| `cir_feller_violated/1` | `not_run` | — | — | — | `analytic_moments` | `validated` | `A` | `analytic` | rubric | `True` |
| `cir_feller_violated/2` | `not_run` | — | — | — | `analytic_moments` | `validated` | `A` | `analytic` | rubric | `True` |
| `double_well_langevin/1` | `True` | 1.690 | 2.428 | `[T, T, T]` | `kolmogorov` | — | `A_prime` | `surrogate` | rubric | `True` |
| `double_well_langevin/2` | `True` | 1.741 | 2.428 | `[T, T, T]` | `kolmogorov` | — | `A_prime` | `surrogate` | rubric | `True` |
| `double_well_langevin/3` | `True` | 1.765 | 2.428 | `[T, T, T]` | `kolmogorov` | — | `A_prime` | `surrogate` | rubric | `True` |
| `gbm_2d_high_corr/1` | `not_run` | — | — | — | `analytic_moments` | `validated` | `A` | `analytic` | rubric | `True` |
| `ginzburg_landau_s6/1` | **`False`** | 3.242 | 2.428 | **`[T, F, T]`** | **`null`** | — | **`C`** | `self_convergence` | rubric | **`False`** |
| `ginzburg_landau_s6/2` | **`False`** | 3.136 | 2.428 | **`[T, F, T]`** | **`null`** | — | **`C`** | `self_convergence` | rubric | **`False`** |
| `ginzburg_landau_s6/3` | `True` | 1.099 | 2.428 | `[T, T, T]` | **`null`** | — | **`C`** | `manufactured_partial` | rubric | `True` |
| `multichannel_stiff_m13/1` | `not_run` | — | — | — | `analytic_moments` | `validated` | `A` | `analytic` | rubric | `True` |
| `ornstein_uhlenbeck/1` | `not_run` | — | — | — | `analytic_moments` | `validated` | `A` | `analytic` | rubric | `True` |

* **`run_dynkin` is `False` on all six live Path-A plans.** That is Finding 1's mechanism 2, stated
  as a config value. Step 8 flips it, and those six Dynkin verdicts are the plan's largest
  unmeasured risk ([§F2](#f2-no-live-path-a-plan-may-acquire-dynkin_ok-false)).
* **`ginzburg_landau_s6` plans 1 and 2 fail on test function 2 only**, the same one in both
  (`per_function_ok: [True, False, True]`). The third test function's residual is `-0.0056` against
  `-0.00017` at the fine level — a different scale entirely from function 2's `13.91 / 6.62`.
* **All three `ginzburg_landau_s6` plans sit at `tier: C` with `surrogate_ok.route: null`.** Step 0
  must move all three to `A_prime` with `route: "kolmogorov"`. **That is the check for step 0** — not
  the score, which moves on only one of the three.
* **`bound_by` is `"rubric"` on all 12.** No live plan is ceiling-bound today, so step 8's
  fabrication cap should be the first `bound_by: "ceiling"` in the set.

### B3. Configuration provenance

| Plan | `expected_strong_order` | `.source` | `scheme_family` | `.source` | `invariants_ok` | `weak_ok` | `richardson_stable` |
|---|---|---|---|---|---|---|---|
| `bm_with_drift/1` | 1.0 | spec | euler-maruyama | inferred | **`False`** | `False` | `True` |
| `cir_feller_violated/1` | 0.5 | spec | euler-maruyama | inferred | `True` | `False` | `True` |
| `cir_feller_violated/2` | 0.5 | spec | milstein | inferred | `True` | — | `unavailable` |
| `double_well_langevin/1` | 1.0 | spec | tamed-euler-maruyama | **plan** | `True` | `True` | `True` |
| `double_well_langevin/2` | 1.0 | spec | implicit | **plan** | `True` | `True` | `True` |
| `double_well_langevin/3` | 1.0 | spec | other | **plan** | `True` | `False` | `True` |
| `gbm_2d_high_corr/1` | 0.5 | spec | euler-maruyama | inferred | **`False`** | — | `unavailable` |
| `ginzburg_landau_s6/1` | 0.5 | spec | euler-maruyama | inferred | `True` | `False` | **`False`** |
| `ginzburg_landau_s6/2` | **1.0** | spec | milstein | inferred | `True` | `True` | **`False`** |
| `ginzburg_landau_s6/3` | 0.5 | spec | euler-maruyama | inferred | `True` | `True` | `True` |
| `multichannel_stiff_m13/1` | 0.5 | spec | euler-maruyama | inferred | `True` | — | `unavailable` |
| `ornstein_uhlenbeck/1` | 1.0 | spec | euler-maruyama | inferred | `True` | — | `unavailable` |

Three consequences the plan does not state:

* **`expected_strong_order.source` is `"spec"` on all 12.** So D2's rule — one-sided whenever the
  source is `kernel_default` — is **unexercised on today's pool** and criterion 13 cannot be checked
  against a live plan. It needs a unit test on `guards`, or a stale spec brought in at step 6.
* **`weak_ok` is `False` on 4 of the 9 plans that produce an estimate**, so
  `weak_order_out_of_band` is `True` four times and is read by nothing ([REV 46]). The weak order is
  decorative today, measurably.
* **`invariants_ok` is `False` on two live Path-A plans that both score 10.** Both failures are
  artifacts of the kernel's own `martingale` evaluator. This is a finding the plan does not carry
  and it changes one of 1a's four rows — see
  [§K3](#k3-1as-invariants-row-is-not-a-transcription-and-two-live-plans-fail-on-artifacts).

---

## C. The claims, each with its primary observable

Eleven claims, by problem class. Each gives the field that carries it. The **score** row is
secondary and is marked where its value depends on a tolerance rather than a measurement.

### C1. A solver that never integrates the SDE stops certifying

| | |
|---|---|
| **Class** | every Path-A SDE: 21 of 23 problems, 5 of 7 live specs, 6 of 12 live plans |
| **Observable** | `config.run_dynkin.value`, then `checks.dynkin_ok` + `summary.dynkin_z` + `certification.gates_clean` + `certification.bound_by` |
| **Today** | `run_dynkin: False` on all six live Path-A plans — Dynkin never runs. On the fabrication canary: `dynkin_ok: not_run`, `orders_ok: not_run`, `gates_clean: True` |
| **Expected** | `run_dynkin: True`; on the fabricated solver `dynkin_ok: False` with `dynkin_z` ≈ **792** against a bar of 2.28, `gates_clean: False`, `bound_by: "ceiling"`, and `evidence.d1_outcome: "stalled"` with `operator_validated: True` |
| **Also** | `certification.caps` must contain a cap whose `reason` names Dynkin or the order study. Today the binding reason is PDE vocabulary — *"D1 stalled with a validated operator"* — which is why criterion 1 fails on wording even once the cap is right |
| **Score (secondary)** | 10, certified → ≤ 3, not certified. Measured with `run_dynkin` forced `True`: fabricated 3, honest 10 at `dynkin_z` 1.06 |
| **Step** | 8 |

`bound_by` is the sharpest anchor here: all 12 live plans are `"rubric"`-bound today, so the first
`"ceiling"` in the set is unambiguous evidence the cap landed where [REV 5] says it does — through
`gates_clean`, not through a rubric row. 1a's `failed(dynkin_ok) -> 3` row is cosmetic.

### C2. Multi-noise SDEs become verifiable at all

| | |
|---|---|
| **Class** | the four genuinely multi-noise problems: `sde_gbm_2d_correlated`, `sde_gbm_2d_high_corr`, `sde_log_heston_feller_violated`, `sde_multichannel_stiff_m13`. **Not** the two oscillators — 2 states, 1 noise channel, nothing is broken there |
| **Observable** | `checks.orders_ok` and `detail.orders_ok.reason` |
| **Today** | `m13`: `orders_ok: "unavailable"`, reason *"the CRN level at dt=0.04 crashed: exception: IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed"*. `gbm_2d_high_corr`: `run_crn_ladder: False`, so the crash is latent |
| **Expected** | `m13`: `orders_ok` is a verdict and `detail.orders_ok.reason` carries no exception. `gbm_2d_high_corr`: the ladder runs after step 7 **without** acquiring the same crash |
| **Corroborating number** | `ladder_spec(..., m=2)` on the real `gbm_2d_high_corr` plan recovers `corr(X_T, Y_T) = 0.945` against the spec's 0.9478 — assert that, not just that the ladder ran |
| **Score (secondary)** | `m13` stays **6**: its cap is `failed(resolved)` (`resolved: False`), which step 2 does not touch. **Expect no score change on either plan.** |
| **Step** | 2, before 7 |

The field already exists — `m13` declares `"noise_dimension": 13`, `gbm_2d_high_corr` declares
`"noise_dimension": null` (itself wrong; it is 2). Expect the kernel to start **reading** a field,
not to gain one. The inferred default the plan originally proposed (`state_dimension` when the drift
is a list) is wrong in both directions on this set and must not appear: it gives 1 for `m13` and 2
for the single-noise oscillators.

### C3. Correlation stops being free

| | |
|---|---|
| **Class** | multi-component SDEs declaring `cross_moment` — today `sde_gbm_2d_high_corr`, where ρ = 0.95 is the entire difficulty (condition number 39, Cholesky second diagonal 0.31225) |
| **Observable** | a new entry in `summary` / `detail.moments_ok` for the cross moment, and `checks.moments_ok` on a correlation-dropping solver |
| **Today** | the spec carries `expression`, `value_at_T`, `correlation_of_terminal`, `correlation_value_at_T`, `covariance_value_at_T` and a `tol`; `grep` finds the only consumer is `reference.py:531`, the spec-vs-spec ledger. `checks.moments_ok: True` for a solver with independent increments |
| **Expected** | `checks.moments_ok: False` on `sde_drops_correlation`, with the cross moment named in the detail; `cross_moment.gate` stays `false`; the honest plan keeps `moments_ok: True` |
| **Score (secondary)** | canary 10 → 8, honest plan stays 10. **The 8 is tolerance-dependent** — it is `rubric_sde_analytic`'s `variance_ok` row |
| **Step** | 3 |

**Do not expect the gate flip.** It is inert as written: `run_declared` reads only
`verification.invariants` and `verification.constraints` and evaluates through an `expr` key;
`cross_moment` is in neither and has no `expr`.

### C4. Distributional evidence exists for the first time

| | |
|---|---|
| **Class** | ~16 of 23 problems have a fully known terminal law |
| **Observable** | new `summary` keys for the third and fourth standardized central moments with their standard errors; later, a KS statistic and its critical value |
| **Today** | `grep -rni 'kurtosis\|skewness\|kolmogorov_smirnov\|empirical_cdf'` over `verifylib/` returns **zero** hits in the kernel. `moment_stats` already computes `mu4` for `se_variance`, so the sample side is free |
| **Expected, step 3** | the two moments present in `summary`, **report-only**, no verdict and no `checks` entry |
| **Expected, after step 6** | a declared `terminal_law`; the KS statistic computed on the **standardized** sample against the standardized law |
| **Check** | at `dt = 0.01`, 50 000 paths, the standardized statistic must **not** reject a correct Euler–Maruyama on OU or GBM. Measured un-standardized: OU `D = 0.00663`, GBM `D = 0.00678`, both against `crit = 0.00607` — **both reject correct solvers** |
| **Score (secondary)** | none intended. Tier 1 cannot grade: censused across all 7 live specs, `analytic_moments` carries only `mean_*`/`variance_*` spellings, and the surrogate ladder returns two moments, so there is no third-moment reference even in principle |
| **Step** | 3, then 6 |

### C5. The order study stops reporting noise as a measurement

| | |
|---|---|
| **Class** | **(i)** exactly-integrating schemes — `sde_bm_standard` (`dX = dW`), `sde_bm_with_drift`. **(ii)** heavy-tailed multiplicative noise — `sde_ginzburg_landau_s6` at σ = 6 |
| **Observable** | `detail.orders_ok.strong_detail.s0`/`.s1`, a **new** `se_s0`/`se_s1`/`se_order`, and `checks.orders_ok` |
| **Today (i)** | `bm_with_drift`: `s0 = 9.38e-15`, `s1 = 1.99e-15`, `observed_order = 2.238`, `orders_ok: True` against a floor of 1.0. On `dX = dW`: `strong_order` returns `-0.1226` from `s0 = 1.74e-16`, `s1 = 1.90e-16`, and `orders_ok` is **`False`** — a deterministic failure on an exact scheme |
| **Today (ii)** | across 11 runs at 50 000 paths `observed_order` spans **0.452–2.726** (plan 1, 6.0x) and **0.220–2.730** (plan 2, 12.4x), with `orders_ok: True` in **all 11**. The control, `3-log-transform-euler`, is tight to **1.1x** |
| **Expected** | `strong_order` returns `se_s0`, `se_s1`, `se_order` (jackknife, one pass). `guards` returns `orders_ok: "unavailable"` — the literal string, not `False` — when `se_order` admits more than the band width or either `s_i` is within `floor_mult * se` of zero. `_orders` propagates it, as it already does for a crashed level |
| **Check** | on `dX = dW`: `orders_ok` must be `"unavailable"`, **not** `False` and not `True`. Re-run the 11-run study: `orders_ok: True` must no longer appear at both 0.220 and 2.730 |
| **Score (secondary)** | **none, by construction.** `"unavailable"` is in `SKIPS`, so `passed()` and `failed()` both refuse it and every rubric row falls through — D8b. `m13` keeps 6, `bm_with_drift` keeps 10 |
| **Step** | 0 |

**The most likely implementation error in the plan, and it is a false-failure.** Implemented in
`strong_order` alone — which is how rev 2 specified it — a `None` lands in `guards` as
`finite = strong is not None` → `s_ok = False` → `orders_ok: False`, a verdict of failure. The
`is None` forgiveness one line below belongs to the *weak* order. **Step 0 is three touches, not
one.**

### C6. A random initial condition stops being a Dynkin failure

| | |
|---|---|
| **Class** | `sde_quintic_random_ic` (`dX = -X**5 dt`, `X(0) = xi ~ Normal(0, sigma_bar**2)`, **no `X_0` key** in `parameters`), and any future random-IC spec |
| **Observable** | `checks.dynkin_ok` and `detail.dynkin_ok.residuals` |
| **Today** | `x0` defaults to `0.0` at `sde/driver.py:464`, so for `phi = X**2` the identity is built with `phi(0) = 0` where it needs `E[phi(X_0)] = sigma_bar**2 = 0.1111`. The residual carries a fixed **−0.111** offset at every level, survives Richardson, and lands `dynkin_ok: False` |
| **Expected** | `dynkin_ok` is `True`, or the literal `"unavailable"` with the random IC as the stated reason. **Never `False`** |
| **Score (secondary)** | the canary stops being capped at 3 |
| **Step** | 5, and asserted **before** step 8 ships |

Not reproducible end to end until `sde_quintic_random_ic` has a live spec — the measurement was
taken on the stale one, which has no `verification` block and cannot reach Dynkin at all.

### C7. A half-usable reference stops being thrown away

| | |
|---|---|
| **Class** | specs where one moment is a quadrature expression and the other a literal — today `sde_quintic_random_ic`, whose `variance_value_at_T: 0.09248275891171216` sits beside a `scipy.integrate.quad(...)` expression |
| **Observable** | `checks.moments_ok` / `.variance_ok` / `.resolved`, **plus `summary.reference_outcome` and `certification.tier`** |
| **Today** | `_reference` evaluates both moments in one `try` and returns `None` for both if either raises. Measured on a grafted spec: all three checks `"unavailable"`, `reference_outcome: "unavailable"`, `tier: A_unavailable`, `provenance: analytic_unvalidated`, ceiling **9** |
| **Expected** | the checks become verdicts, **and** `reference_outcome: "validated"` with `tier: A` and `provenance: analytic`. A spec with neither spelling reports a distinct "no reference was evaluable" reason |
| **Score (secondary)** | 4 → a real verdict. Today's 4 carries the reason *"the variance check fails, though the code ran cleanly"* when **no variance check ran** — the reason is the defect, more than the number |
| **Step** | 1 |

**`tier` is the anchor that catches the half-fix.** `reference_outcome` is computed independently by
`check_reference` → `check_moment_ode_consistency` → `_moment_observables`, which reads
`am[f"{kind}_expression"]` and has no `*_value_at_T` fallback either. Fix only `_claimed`/`_reference`
and the checks become verdicts while `tier` stays `A_unavailable` and the ceiling stays 9 — a
score-level check would read as satisfied.

### C8. The Kolmogorov surrogate works again on a one-sided domain

| | |
|---|---|
| **Class** | specs whose `kolmogorov.bounds` are one-sided. Of the four declaring bounds, exactly one: `sde_ginzburg_landau_s6`, `[0, 24]` |
| **Observable** | `detail.surrogate_ok.route`, `certification.tier`, `certification.provenance`, and the number inside `detail.surrogate_ok.attempted.kolmogorov` |
| **Today, all three plans** | `route: null`; `attempted.kolmogorov: "the truncation guard failed: widening the domain moved the answer by 1.39e-02"`; `tier: C`; `provenance: self_convergence` / `self_convergence` / `manufactured_partial` |
| **Expected, all three plans** | `route: "kolmogorov"`; `tier: A_prime`; `provenance: surrogate`; `truncation_shift` max **0.00361** against the bar of 0.01 |
| **Score (secondary)** | plan 3 only: 9 → 10. **Plans 1 and 2 stay at 3** — see [§F6](#f6-ginzburg_landau_s6-plans-1-and-2-must-stay-at-dynkin_ok-false) |
| **Step** | 0 |

**This is the clearest case for field-level checking.** Three objective fields move on all three
plans; the score moves on one. A score-anchored ledger would record step 0's `_widen` fix as a
one-point change on one plan, when it actually restores an entire evidence tier on three.

Two signatures that confirm the diagnosis rather than the verdict: the reference is **identical**
either way (`mean = 0.68003`, `variance = 1.88808`), and it is within 0.8%/0.5% of the spec's
independent discretization-free Monte Carlo. Symmetric widening leaves the mean almost untouched
(0.00109) and corrupts `E[X^2]` (0.01385) — what a spurious negative-`x` region does to a second
moment. The far-field widening the formulator actually measured reproduces their recorded 0.36% to
three figures.

### C9. Cross-plan agreement stops being earned on a shared seed

| | |
|---|---|
| **Class** | every SDE `compare` run. Reachable today: `cli.py:272` dispatches `equation_type == "SDE"` to `_compare_sde` |
| **Observable** | `checks.cross_plan_agreement`, `certification.agreement_is_break`, and the seeds recorded per comparison |
| **Today** | `_compare_sde` runs both plans at the same `seed`, against `verification_manual.md` §22's explicit *"Two plans agreeing on a shared seed proves nothing — they saw the same noise."* The combined-SE formula is already §22's conservative `np.hypot(se_a, se_b)`; only the seed diverges. Agreement is priced as a circularity break worth a 10 |
| **Expected** | two comparisons recorded. The shared-seed one is a diagnostic and may never set `cross_plan_agreement`; only the independent-seed one may, and only it feeds `agreement_is_break` |
| **Cost** | four solves per `compare` instead of two |
| **Step** | 10 |

### C10. The promised checks that do not exist start existing

| Promise | Observable | Today | Expected |
|---|---|---|---|
| `stability_check` is the scoring criterion for blow-up and domain problems (`project_manual.md:304`) | a `checks` entry for it; the moment comparison **absent** when it is non-null | `grep -rn stability_check verifylib/` → **one** hit, a test fixture. No live spec carries a structured one; the stale `sde_fene_blowup` instance is prose | non-null → evaluated (`finite` over `terminal_paths`; `domain` as `max \|X\| < abs_bound`) and **replaces** the moment comparison, matching `benchmark/verify.py:422`. A non-`dict` value is a **schema error** |
| "the evaluator checks `solver.py` for exactly that" — a bare clip (`agents/solver-sde.md:99`) | a leakage/gate finding on the canary | **false.** No clip scan in `leakage.py`, `gate.py`, `review.py` or `audit.py` | an AST check for **assignment to the state name** in the step loop; never a clip used as a sub-expression. CIR's endorsed `np.sqrt(np.maximum(X, 0))` must not match |
| the solver's reported `empirical_mean`/`empirical_variance` mean something | a reported-vs-measured diagnostic in `summary` | marshalled at `sandbox.py:264`, rebuilt at `:473`, **no consumer**. A solver reporting `np.var(X)` while claiming `ddof=1` is invisible | one comparison, diagnostic, never a gate |
| `milstein_eligible` constrains anything | a cap on a Milstein plan over a Lévy-area-ineligible spec | four specs declare it, always `false`; `grep -rn milstein_eligible verifylib/` → **nothing** | a cap when the reason is Lévy-area. **Not** when the reason is `dg/dX = 0` — on `bm_with_drift`, `ornstein_uhlenbeck` and `double_well_langevin` Milstein is bit-identical to Euler–Maruyama and correct |

### C11. The declared-and-never-read pattern stops recurring

| | |
|---|---|
| **Observable** | a table-driven test over every key in every `workspace/*/problem_spec.json`, and over the metrics keys |
| **Today** | **six of the fourteen findings are this one shape**: `cross_moment` (F3), `noise_dimension` (F2), `*_value_at_T` (F6), `stability_check` (F9), `empirical_mean`/`empirical_variance` (F12), `milstein_eligible` (F14). Plus `weak_order_out_of_band`, computed at `crn.py:165` and read by nothing — measurably live, since `detail.orders_ok.weak_ok` is `False` on 4 of the 9 plans with an estimate |
| **Expected** | every spec key has a kernel consumer or a recorded reason; every metrics key has a consumer or a recorded reason |
| **Step** | 4 |

The highest-leverage test in the plan: it catches six of the fourteen findings at once, and the
next one. [§K3](#k3-1as-invariants-row-is-not-a-transcription-and-two-live-plans-fail-on-artifacts)
is a seventh instance this file found — a field declared, *read*, and silently ignored by the
evaluator that claims to use it — so consider extending the test to assert that a declared `expr`
is actually evaluated.

---

## D. The 23 acceptance criteria, with an objective anchor

The plan's [§6](plan-sde-verification-gaps.md#6-acceptance-criteria) as amended by [REV 32] and §11.
**Nine of the 23 are worded as scores.** The right-hand column gives the field to check instead, or
in addition. Where a criterion is already field-anchored it says so.

| # | Criterion (plan's wording) | Objective anchor to check | Today |
|---|---|---|---|
| 1 | `sde_fabricated_sampler` scores ≤ 3, reason naming Dynkin or the order study | `certification.bound_by == "ceiling"`, `gates_clean == False`, `dynkin_ok == False`, and a `caps` entry whose `reason` names Dynkin or the order study | `run_dynkin: False`; no cap; `bound_by: rubric` on all 12 |
| 2 | a `dW`-ignoring solver scores **8** on Path A | `checks.orders_ok == False` **and** the rubric reason naming the measured order; the 8 itself is a rubric row, not a measurement | not capped at all |
| 3 | the ladder completes on all four multi-noise problems; `orders_ok` a verdict | `detail.orders_ok.reason` carries no exception; `checks.orders_ok` not in `SKIPS` | `m13`: `IndexError` in the reason. `gbm_2d_high_corr`: ladder exempted |
| 4 | `sde_drops_correlation` scores ≤ 8; honest plan 10 | `checks.moments_ok == False` on the canary and `True` on the honest plan, with the cross moment named in the detail | `moments_ok: True` for a correlation-dropping solver |
| 5 | `sde_random_ic_honest` not failed by Dynkin | `checks.dynkin_ok` is `True` or `"unavailable"`, **never `False`** | `False`, from the `x0 = 0.0` default |
| 6 | grades against `variance_value_at_T`, provenance `analytic` at tier `A` | `summary.reference_outcome == "validated"` **and** `certification.tier == "A"` | both `"unavailable"` / `A_unavailable` |
| 7 | every spec declares a floor for every scheme family its plans use | spec census: for each plan, `config.expected_strong_order.value` matches the family's declared floor, not the 0.5 default | `cir_feller_violated` declares no `_milstein`; 16 stale specs declare nothing |
| 8 | the two analytic rubrics read the same evidence fields | a parity test over the field *names* each rubric reads | `rubric_sde_analytic` reads 5; neither SDE rubric reads `invariants_ok` |
| 9 | the full suite is green, no canary regresses | `580 passed / 0 failed` of 583 collected | measured, exit 0, 368.92s |
| 10 | a correct additive-noise EM on an undeclared floor is not failed | `checks.orders_ok != False` with `config.expected_strong_order.source == "kernel_default"` | fails at observed 0.9863 when the family is declared |
| 11 | no spec field declared and unread without a recorded reason | the coverage test's own output | six such fields, plus `weak_order_out_of_band` |
| 12 | the orders-fail row returns 8, reason naming which defect | call `rubric_sde_analytic` with synthetic evidence; assert the **reason string**, which is the part that distinguishes near-zero from below-floor | row does not exist |
| 13 | a `kernel_default` floor is judged one-sided | `detail.orders_ok.strong_band` starts `"one-sided"` whenever `config.expected_strong_order.source == "kernel_default"` | **unexercised**: all 12 plans have `source: "spec"`. Needs a `guards` unit test |
| 14 | `ginzburg_landau_s6/3` scores 10, surrogate | **all three plans**: `detail.surrogate_ok.route == "kolmogorov"`, `tier == "A_prime"`, `provenance == "surrogate"` | `route: null`, `tier: C` on all three |
| 15 | `strong_order` abstains when its bar admits more than the band | `orders_ok == "unavailable"` on the heavy-tailed plans; `se_order` present in `detail.orders_ok` | `orders_ok: True` in all 11 runs; no `se_*` keys exist |
| 16 | a correct EM at `dt = 0.01`, 50 000 paths passes the standardized KS check | the KS `D` and its critical value, both recorded | un-standardized: `D` 0.00663 / 0.00678 vs `crit` 0.00607 — rejects |
| 17 | shared-seed agreement never sets `cross_plan_agreement` | two recorded comparisons; `checks.cross_plan_agreement` set only by the independent-seed one; `certification.agreement_is_break` follows it | one comparison, shared seed |
| 18 | `orders_ok` has a third value, asserted on `dX = dW` | `orders_ok == "unavailable"` where today it is `False` | `False`, from `strong_order` returning `None` on `s0 <= 0` |
| 19 | the abstention costs nothing | `m13` keeps `score == 6` and `bm_with_drift` keeps `score == 10` with `orders_ok == "unavailable"` | n/a |
| 20 | `cir_feller_violated/2-milstein` runs the ladder and reports a verdict | `config.run_crn_ladder.value == True` and `checks.orders_ok` not `not_run`. **See [§K2](#k2-criterion-20-measures-plumbing-not-detection) — the verdict it produces cannot catch the defect §24 named** | `run_crn_ladder: False` |
| 21 | a Path-A plan with clean conjuncts and `failed(orders_ok)` scores 8, never 3 | call the rubric directly with the four conjuncts set; assert the **row reached**, not only the number | `rubric_sde` has two such rows (8 guarded, 3 unguarded) and Path A lands on the 3 |
| 22 | `ranking.order_margin` scores a correct additive EM above 0.9 | the returned float, by direct call | **0.028**. Unreachable today (`design_order` is PDE-only) |
| 23 | every live `STATE.md` matches a re-measured baseline | diff the dump in [§A2](#a2-capturing-a-snapshot) against `STATE.md` | 6 of 12 diverge; all explained, all stale |

**Criteria 2, 12 and 21 cannot be fully de-scored**, because they are claims *about* the rubric. For
those three, assert the row reached and the reason string by calling the rubric directly with
synthetic evidence — that is objective — and treat the score as the row's label rather than the
measurement.

---

## E. What changes on the 12 live plans

Field-level, so the diff is checkable. The score column is last and annotated.

| Plan | Field that moves | Today → Expected | Score |
|---|---|---|---|
| `bm_with_drift/1` | `checks.orders_ok` | `True` (from `s0` 9.4e-15) → `"unavailable"` | 10 → 10 |
| | `checks.dynkin_ok` | `not_run` → a verdict, **must not be `False`** | |
| | `checks.invariants_ok` | `False` → `True` once `_martingale` honours the declaration ([§K3](#k3-1as-invariants-row-is-not-a-transcription-and-two-live-plans-fail-on-artifacts)) | at risk of 10 → 5 if 1a ships unchanged |
| `cir_feller_violated/1` | `checks.dynkin_ok` | `not_run` → a verdict, **must not be `False`** | 8 → 8 *(tolerance-bound: `moments_ok: False`, `variance_ok: True`)* |
| `cir_feller_violated/2` | `config.run_crn_ladder.value` | `False` → `True`, by the one-line spec edit alone | 10 → 10 *(fragile: 4.96% vs 5.00%)* |
| | `checks.orders_ok` | `not_run` → a verdict — but at a one-sided bar of 0.1, see [§K2](#k2-criterion-20-measures-plumbing-not-detection) | |
| `double_well_langevin/1,2,3` | — | nothing. Already `tier: A_prime`, `route: kolmogorov`, `per_function_ok: [T,T,T]` | 10 → 10; `STATE.md` write-back on plans 1 and 3 |
| `gbm_2d_high_corr/1` | `config.run_crn_ladder.value` | `False` → `True` (step 7) | 10 → 10 |
| | `checks.orders_ok` | `not_run` → a verdict **without** acquiring `m13`'s `IndexError` — this is the step-2-before-step-7 check | |
| | cross-moment entry | absent → present, ρ graded against 0.9478 | |
| | `checks.invariants_ok` | `False` → `True` once `_martingale` honours `expr` | at risk of 10 → 5 if 1a ships unchanged |
| `ginzburg_landau_s6/1` | `detail.surrogate_ok.route` | `null` → `"kolmogorov"` | 3 → 3 |
| | `certification.tier` | `C` → `A_prime` | |
| | `certification.provenance` | `self_convergence` → `surrogate` | |
| | `checks.dynkin_ok` | `False` → **stays `False`** (real, seed-independent) | |
| `ginzburg_landau_s6/2` | same three fields as plan 1 | `null → kolmogorov`, `C → A_prime`, `self_convergence → surrogate` | 3 → 3 |
| | `checks.orders_ok` | `True` at observed 0.644 against floor 1.0 → `"unavailable"` or a real verdict | |
| `ginzburg_landau_s6/3` | same three fields | `null → kolmogorov`, `C → A_prime`, `manufactured_partial → surrogate` | **9 → 10** |
| `multichannel_stiff_m13/1` | `checks.orders_ok` | `"unavailable"` (IndexError) → a verdict | 6 → 6 *(capped by `resolved: False`)* |
| | `detail.orders_ok.reason` | carries an `IndexError` → carries none | |
| `ornstein_uhlenbeck/1` | `checks.dynkin_ok` | `not_run` → a verdict, **must not be `False`** (canary measures z = 1.06) | 10 → 10 |

**Summary of the diff.** Twelve plans; **one score moves** (plan 3 of `ginzburg_landau_s6`, +1), two
more are *at risk* of a 5-point drop on artifacts if 1a's invariants row ships as drafted, and
**nineteen field values move** across nine plans. The field-level diff is roughly twenty times as
informative as the score diff, and it is the one that tells you whether each step did its job.

### Not predictable yet

* **The 16 stale specs** have no `verification` block, so none of the spec-driven checks reach them.
  Step 6 brings them in; expect their fields to *appear*, not to move.
* **The four stale multi-noise problems' `orders_ok`** — unmeasurable until they have live specs,
  which is why criterion 3 is scoped to the two that do.
* **Five of the six live Path-A Dynkin verdicts.** Only OU's equivalent has ever been measured.

---

## F. What must not change

Six of the plan's twelve blocking revisions concern false-*failures* on correct solvers. These are
the assertions that make that class visible, since *a false hard-fail on correct work is invisible
as a false positive*.

### F1. The existing honest-solver guard

`test_the_same_sde_solver_at_full_path_count_scores_ten`
([`test_kernel_canaries.py:128`](../verifylib/tests/test_kernel_canaries.py#L128)) asserts
`score == 10 and checks["resolved"] is True`. Measured with steps 7 and 8 both forced on: order
0.986 against a declared 1.0, `dynkin_z` 1.06 against a bar of 2.28, still 10.

### F2. No live Path-A plan may acquire `dynkin_ok: False`

**The largest unmeasured risk in the plan.** `config.run_dynkin.value` is `False` on all six live
Path-A plans, and step 8 flips it into `gates_clean` — a hard cap. `dynkin.py:40-48` records a
surviving false-failure rate: *"between 0.2 and 3.2 standard errors of zero… a property of the seed
rather than of the scheme"*. At its own worst observed z of 3.2 against a K=2 bar of ~2.24, a
**correct** solver is capped.

Check `checks.dynkin_ok` and `summary.dynkin_z` on all six, not the scores. If any comes back
`False`, read `detail.dynkin_ok.per_function_ok` first — it names which test function — then run
D9's replication: **median z over 3 seeds**, against the sd ≈ 0.33–0.37 measured on
`ginzburg_landau_s6`. `cir_feller_violated` is the one to watch: full-truncation Euler on a
Feller-violated CIR has real mass near zero.

### F3. A correct solver on a defaulted floor must not be failed

`checks.orders_ok` must not be `False` when `config.expected_strong_order.source` is
`"kernel_default"`. Six of the 16 stale specs are additive or noise-free and all default to 0.5 —
`sde_bm_standard`, `sde_linear_additive`, `sde_ou_stiff`, `sde_stochastic_oscillator`,
`sde_oscillator_long_horizon`, `sde_quintic_random_ic` — while Euler–Maruyama has strong order **1**
on additive noise. Measured with the family declared: `orders_ok: False` at observed 0.9863.

`sde_bm_standard` is the extreme case: `dX = dW` makes Euler–Maruyama exact, `s0 = 1.74e-16`,
`s1 = 1.90e-16`, `strong_order` returns `-0.1226`, and it fails **both** bands deterministically.

### F4. `strong_band` must stay `one-sided` on the nine inferred plans

`detail.orders_ok.strong_band` is the literal `"two-sided"` on exactly the three `double_well` plans
(`config.scheme_family.source == "plan"`) and `"one-sided (scheme family undeclared in the plan)"` on
the other nine. That is what keeps step 7 from failing them.

The exposure is **latent**: it goes live when `scheme_family` is declared — which `agents/`
instructs plan-creators to do — or when step 6 brings new plans in. So the band rule lands in
**three** places: `crn.guards`, 1a's new rubric row, and `ranking.order_margin`. The three
`double_well` plans are the only two-sided ones today and therefore the only ones a band change can
hit; all three are at `orders_ok: True` with observed 1.020–1.046 against a floor of 1.0.

### F5. Three working truncation guards must not be re-measured

| Spec | bounds | symmetric widen | pinned by D7? |
|---|---|---|---|
| `sde_ornstein_uhlenbeck` | [-6, 6] | [-9, 9] | no — must not change |
| `sde_double_well_langevin` | [-3, 3] | [-4.5, 4.5] | no — must not change |
| `sde_bm_with_drift` | [-4, 7] | [-6.75, 9.75] | no — must not change |
| `sde_ginzburg_landau_s6` | **[0, 24]** | **[-6, 30]** | **yes — the fix** |

Only one of the first group actually exercises the route today: `double_well_langevin`'s three plans
show `detail.surrogate_ok.route: "kolmogorov"`, while `ornstein_uhlenbeck` and `bm_with_drift` are
Path A and resolve to `analytic_moments` first. So `double_well`'s three plans are the live
regression guard — their `truncation_shift` must not change — and the other two specs' guards have
to be checked by calling `kolmogorov()` directly. If any of the three changes, the rejected "widen
away from `x0` only" rule was implemented instead of D7's pin-a-declared-support rule.

### F6. `ginzburg_landau_s6` plans 1 and 2 must stay at `dynkin_ok: False`

Counter-intuitive, and verified against the rubric directly with synthetic evidence:

```
surrogate restored, Dynkin still fails  -> (3, "Dynkin fails: ...")
surrogate restored, Dynkin passes       -> (10, "a surrogate reference passed its trust guard ...")
surrogate unavailable, Dynkin fails     -> (3, "Dynkin fails: ...")
```

`failed(dynkin_ok) -> 3` sits **ahead** of the surrogate row, so restoring the surrogate does not
lift them. And the failure is real: 4/4 seeds fail on each plan at `M = 50000`, z = 2.727–3.487
(plan 1) and 2.897–3.769 (plan 2), with no crossing of the 2.428 bar in either direction in any of
the 11 runs. If `dynkin_ok` becomes `True` on either, the row ordering has been broken — a worse
regression than the one step 0 fixes, because it re-opens what `a507038` closed.

### F7. An abstention must cost nothing

`checks.orders_ok == "unavailable"` with every other check clean must leave `score` unchanged:
`m13` at **6**, `bm_with_drift` at **10**. `metrics.record` accepts `True`, `False` or any `SKIPS`
member and raises otherwise, so a crash here means a value outside `SKIPS`.

The rejected alternative was a ceiling at 9, by analogy with `A_unavailable`. The two cases record
*whose* limitation they are: an unevaluable reference is a property of the spec; an unmeasurable
strong order on heavy-tailed multiplicative noise is a property of the **kernel's estimator**.

Today there is **no Path-A row for `orders_ok: "unavailable"` at all**, so all four of 1a's rows fall
through — and after step 0, abstention becomes the expected outcome on heavy-tailed multiplicative
noise. [REV 44]'s new row must **report** it, not cap on it.

### F8. The suite number, and one file to run first

**580 passed / 0 failed** of 583 collected. The 4 deselected are `pyproject.toml:43`'s
`-m 'not slow'`, all `pde_*` truth routes; the 3 skips are `"not staged in workspace/"` guards, also
all PDE. Nothing on the SDE path is skipped or deselected.

Step 1 touches `_moment_observables`, which feeds `check_moment_ode_consistency` — the spec-vs-spec
ledger `test_reference_sde.py` covers. The `*_value_at_T` fallback changes which pairs get compared,
so **run that file first and in isolation**.

---

## G. Cost expectations

Wall time is already objective, and the plan has a risk column and no cost column.

| | Measured today | Expected after | Why |
|---|---|---|---|
| Full suite | **368.92s** (6m08s), 583 collected | under ~12 min | step 8 adds a Dynkin ladder to every Path-A canary |
| The 11 `test_kernel_sde_path_b.py` tests | **276.8s — 75% of the suite's wall time, from 11 of 583 tests** | ~554s if the new canaries run at full paths | these are the canaries running real CRN and Dynkin ladders |
| A 12-plan re-sweep | **33.5 min**, of which `ginzburg_landau_s6` is ~31 | within a factor of ~2 | if not, D9's margin trigger is not firing |

**The 75% concentration makes §4d's advice load-bearing rather than optional.** Run the new
*defective* canaries at reduced `num_paths` — fabrication, correlation and uniform-increment
canaries all fail by orders of magnitude, so a few thousand paths separates them — and reserve
50 000 for the honest controls where the margin is the thing being measured.

**The plan's §3 priced the two expensive steps backwards; §7 corrects it.** Expect:

* **Step 7 (the order study) is cheap.** `config.dt0` is `4*dt` and the ladder runs at
  `dt0, dt0/2, dt0/4` at `conv_num_paths` — **20 000** per `config`, against a graded 50 000. Well
  under one graded solve. The risk there is calibration, not cost.
* **Step 8 (Dynkin on Path A) is the expensive one.** The ladder refines **downward** from the
  graded `dt` at full `num_paths`: ~7× a graded solve plus observables; `dynkin.py` records ~30 s
  per plan.
* **Unconditional replication would be ~21× a graded solve per certifying Path-A plan.** D9's
  trigger prevents it: replicate to `k = 3` only when `|dynkin_z - dynkin_bar|` is inside ~`2 * sd`
  ≈ 0.7. On today's pool nothing is marginal — the failing plans are at z ≈ 3.2, the passing one at
  z ≈ 1.3 — so cost returns to **~1×**. Record the margin decision in the metrics, so a plan graded
  on one seed says so.
* **`compare` doubles**, 2 solves → 4.
* **2b is cheaper than §5 claims.** The Hessian grows `O(d²)` in **state** dimension — 2 for every
  one of these problems — so 4 evaluations per test function rather than 3. m13's 13 noise channels
  enter only through `G G^T`, a 2×2 matrix.

---

## H. What this plan does not claim

1. **A mis-transcribed equation still passes everything.** Dynkin builds its generator from the
   spec's own `drift_expression`, so a drift wrong in `problem_spec.json` fools the moments, the
   surrogate and Dynkin alike — the same limitation D1, MMS and the degenerate limit carry on the
   PDE side. The requirements ledger is the only thing reaching outside the spec. **`REPORT.md` must
   not imply otherwise.**
2. **Path-B problems get no distributional evidence.** Where the terminal law is unknown, the
   moment-based tier is all there is.
3. **The KS tier needs calibration, not just a declared law.**
4. **Dynkin's false-failure rate is reduced, not removed.** With step 8 routing it into
   `gates_clean`, the residual rate shows up as a hard 3 rather than a two-point cap.
5. **Multi-dimensional Dynkin (2b) may not ship.** It is the deferral candidate; step 2 delivers
   most of Phase 2's value.
6. **The cheap distributional tier grades nothing before step 6.** No `analytic_moments` block in
   any live spec carries a third or fourth moment.
7. **Step 6's first pass is `expected_strong_order` only.**
8. **Criterion 20 being met does not close §24's motivating case** — see
   [§K2](#k2-criterion-20-measures-plumbing-not-detection).
9. **Criterion 13 cannot be checked against a live plan.** All 12 have
   `expected_strong_order.source == "spec"`.

---

## I. If the expected result does not appear

| Symptom | Field to read | Most likely cause |
|---|---|---|
| A heavy-tailed plan shows `orders_ok: False` after step 0 | `detail.orders_ok.se_order` — is it even present? | the error bar went into `strong_order` alone. `guards` reads `finite = strong is not None` and lands `None` as **failure** |
| `ginzburg_landau_s6` still shows `tier: C` after step 0 | `detail.surrogate_ok.attempted.kolmogorov` — still 1.39e-02? | `_widen` still symmetric, or the gated-constraint lookup is not finding `positivity` |
| `double_well`'s `truncation_shift` changed | `detail.surrogate_ok` on all three plans | the rejected "widen away from `x0`" rule was implemented |
| `ginzburg_landau_s6` plan 1 or 2 shows `dynkin_ok: True` | `detail.dynkin_ok.per_function_ok` | the `failed(dynkin_ok) -> 3` row moved behind the surrogate row |
| A correct additive-noise plan shows `orders_ok: False` after step 7 | `detail.orders_ok.strong_band` and `config.expected_strong_order.source` | the band is two-sided where the source says it should be one-sided |
| A live Path-A plan shows `dynkin_ok: False` after step 8 | `detail.dynkin_ok.per_function_ok`, then `summary.dynkin_z` vs `.dynkin_bar` | inherited false-failure rate **or** a real inconsistency. Do not assume either — run the median over 3 seeds |
| The fabrication canary caps but the reason is PDE vocabulary | `certification.caps[*].reason` and `.source` | `ceiling()`'s reason was not branched on `evidence["kind"]`. The cap is right; criterion 1 is not met |
| A Path-A plan with failing orders landed on the 3 row | the rubric reason string | the unguarded `rubric_sde:368` row was transcribed instead of the guarded `:354`. Path A substitutes `moments_ok and resolved and constraints_ok` for the `surrogate` conjunct |
| `orders_ok: "unavailable"` cost points | `certification.caps` — which cap bound it? | the new Path-A row capped instead of reporting |
| Criterion 6 looks met but the plan cannot reach 10 | `summary.reference_outcome` and `certification.tier` | only `_claimed`/`_reference` were fixed; `reference.py:500` still has no fallback |
| The multi-noise ladder still crashes | `detail.orders_ok.reason` | `noise_dimension` not threaded to **both** `ladder_spec` call sites (`sde/driver.py:402` and `:455`), or inferred rather than read |
| m13's Dynkin crashes with `bad_schema` | `detail.dynkin_ok.reason` | `observables_payload` still tests `isinstance(drift, str)`, which admits prose |
| Two Path-A plans dropped 5 points on invariants | `detail.constraints_ok.outcomes` and `.drifts` | 1a's invariants row shipped before `_martingale` was fixed — [§K3](#k3-1as-invariants-row-is-not-a-transcription-and-two-live-plans-fail-on-artifacts) |
| Ranks moved and no criterion caught it | the float returned by `ranking.order_margin` | expected — it is invisible to every score-based criterion |
| The suite is past 20 minutes | `--durations=15` | the new defective canaries are running at full `num_paths` |

---

## J. Reference baseline, self-contained

`benchmark/results/` and `workspace/` are gitignored, so these files will not survive a clean
checkout. **These numbers are the durable record.**

### J1. The suite — 2026-10-14, tree `d5bb447`

```
580 passed, 3 skipped, 4 deselected in 368.92s (0:06:08)     exit 0
real 369.78   user 309.90   sys 51.85
```

**The environment, added 2026-10-15 ([REV 67]).** This section gave a suite number and not the
interpreter it was measured on, which made the number look like a property of the tree. It is not.
An independent evaluation ran the same tree and got **638 passed / 5 failed**, all five
`np.trapezoid` and all five PDE-side, because it had numpy 1.26.4 — and `pyproject.toml` declared
`numpy>=1.26.0` against 15 `np.trapezoid` call sites, so that numpy satisfied the project's own
stated requirement and could not run its suite. The floor is `numpy>=2.0.0` now, with
`verifylib/tests/test_project_metadata.py` holding it to the APIs actually called. Every suite
number in this file and in the plan was measured at:

```
python 3.14.7   numpy 2.4.2   scipy 1.17.1   macOS-15.7.3-arm64
```

Slowest 15, seconds:

```
44.37  test_kernel_sde_path_b.py::test_paths_outside_a_declared_support_are_a_gate_violation
26.74  test_kernel_sde_path_b.py::test_a_solver_that_ignores_the_crn_increments_is_detected
25.63  test_kernel_sde_path_b.py::test_a_solver_that_ignores_the_crn_increments_does_not_certify
24.94  test_kernel_sde_path_b.py::test_path_b_runs_the_order_study_and_dynkin_unconditionally
24.67  test_kernel_sde_path_b.py::test_an_honest_sde_with_a_surrogate_scores_ten_surrogate
23.17  test_kernel_sde_path_b.py::test_an_honest_sde_with_no_surrogate_still_measures
22.16  test_kernel_sde_path_b.py::test_an_sde_plan_can_earn_its_second_break_from_agreement
21.50  test_kernel_sde_path_b.py::test_a_dynkin_failure_is_never_reported_as_mc_inconclusive
21.47  test_kernel_sde_path_b.py::test_a_sign_flipped_drift_is_caught_by_dynkin
21.10  test_kernel_sde_path_b.py::test_an_honest_sde_with_no_surrogate_reaches_nine_through_dynkin
21.05  test_kernel_sde_path_b.py::test_a_moment_ode_that_does_not_close_is_not_a_reference
 8.25  benchmark/tests/test_truth_routes.py::test_the_eigen_expansion_is_converged
 6.01  benchmark/tests/test_oneshot_salvage.py::test_an_empty_error_with_no_code_anywhere_still_retries
 4.16  verifylib/tests/test_kernel_horizon.py::test_the_frozen_ks_plans_grade_as_the_harness_does[1-spectral-etdrk4-9]
 2.89  verifylib/tests/test_kernel_end_to_end.py::test_the_two_level_path_is_taken_only_when_every_precondition_holds
```

### J2. The canaries that do not exist yet

The "today" column is the state of each defect, which is its canary's before-value.

| Canary to add | Defect | Observable that must catch it | Today |
|---|---|---|---|
| `sde_fabricated_sampler.py` | draws from the exact law; never integrates | `dynkin_ok: False`, `gates_clean: False`, `bound_by: "ceiling"` | **`dynkin_ok: not_run`**, score 10, certified |
| `sde_two_component.py` | honest, 2 states, 2 noise channels | `orders_ok` a verdict; recovered ρ ≈ 0.945 | **`IndexError`** in `detail.orders_ok.reason` |
| `sde_drops_correlation.py` | independent increments on ρ = 0.95 | `moments_ok: False` with the cross moment named | `moments_ok: True` |
| `sde_uniform_increments.py` | Rademacher increments rescaled to σ² = 1 | the KS statistic | nothing computes one |
| `sde_milstein_post_step.py` | correction at `X_{n+1}` | `orders_ok: False` against the per-family floor | the ladder is exempted unless the spec declares ≥ 1.0 |
| `sde_clips_positivity.py` | bare `np.maximum(X, 0)` after the step | a leakage/gate finding | no clip scan exists |
| `sde_random_ic_honest.py` | **correct** solver, random `X(0)` | `dynkin_ok` is `True` or `"unavailable"` | **`False`** |
| *(missing from the plan's table)* correct additive-noise EM on a `kernel_default` 0.5 floor | none — it is correct | `orders_ok != False` | `False` at observed 0.9863 |

Write the last two first.

### J3. The 11-run seed study at `M = 50000` (2026-10-04)

`dynkin_z`, bar 2.428:

| Plan | seeds 42 / 7 / 101 / 2024 | mean | sd | failed |
|---|---|---|---|---|
| 1 tamed EM | 3.242 / 3.365 / 3.487 / 2.727 | 3.205 | 0.334 | **4 / 4** |
| 2 tamed Milstein | 3.136 / 3.223 / 3.769 / 2.897 | 3.256 | 0.368 | **4 / 4** |
| 3 log transform | 1.099 / 1.603 / 1.253 / — | 1.318 | 0.258 | 0 / 3 |

No crossing in either direction: worst passing z 1.603, best failing z 2.727, bar in the gap.

`observed_order` from the same 11 runs, with `orders_ok: True` in **all 11**:

```
plan 1 (tamed EM):    0.452 / 1.357 / 2.726 / 1.78    spread  6.0x
plan 2 (tamed Mil):   0.644 / 1.337 / 0.220 / 2.73    spread 12.4x
plan 3 (log transf):  1.019 / 1.136 / 1.145 /  —      spread  1.1x
```

**The outstanding measurement.** D8 ships a **median** absolute pathwise difference beside the mean
as a diagnostic, promoted to the gate only if it tracks the mean-based estimate on
`3-log-transform-euler` — the control, tight to 1.1x. One re-run of the study already on disk, and
it decides whether the order study can gate on multiplicative noise at all. Until then
[§C5](#c5-the-order-study-stops-reporting-noise-as-a-measurement) delivers an honest abstention and
no new verdict on plans 1 and 2 — and where the order study abstains and no surrogate exists,
**Dynkin is the only remaining evidence that the paths came from the SDE.**

### J4. Spec coverage

| | Count |
|---|---|
| `workspace/sde_*` directories | 23 |
| with a live `problem_spec.json` carrying a `verification` block | **7** |
| of those 7, Path A / Path B | 5 / 2 |
| in `workspace_stale_pre_c2b_backup/`, every one with **no** `verification` key | 22 |
| live plans | 12 — 9 `scheme_family.source: inferred`, 3 `plan` |
| live plans with `expected_strong_order.source: "spec"` | **12 of 12** |
| genuinely multi-noise problems | 4 |
| problems with a known terminal law | ~16 |
| problems carrying ground-truth moments | 21 of 23 |

Live specs: `sde_bm_with_drift`, `sde_cir_feller_violated`, `sde_double_well_langevin`,
`sde_gbm_2d_high_corr`, `sde_ginzburg_landau_s6`, `sde_multichannel_stiff_m13`,
`sde_ornstein_uhlenbeck`.

### J5. Spec and artifact work that gates specific criteria

| Needed | Gates | Today |
|---|---|---|
| `expected_strong_order` on the 16 stale specs | step 7, criterion 7 | absent; defaults to 0.5 |
| `expected_strong_order_milstein` wherever a Milstein plan exists | step 0, criteria 7 and 20 | **absent on `sde_cir_feller_violated`**; present on the other two |
| `noise_dimension` on `sde_gbm_2d_high_corr` | criterion 3 | declared `null`; it is 2 |
| Live specs for `sde_gbm_2d_correlated`, `sde_log_heston_feller_violated` | criterion 3 | none |
| An evaluable `drift_expression` on `sde_multichannel_stiff_m13` | Dynkin there | prose |
| A live spec for `sde_quintic_random_ic` | Findings 5 and 6's canaries | none |
| Structured `stability_check` on `sde_fene_blowup`, `sde_quintic_drift_noise` | 5a | oracle-only; the stale one is prose |
| Which reason `milstein_eligible: false` carries | 2d | unrecorded; the field means two things |
| A declared `terminal_law` | both distributional tiers, criterion 16 | none declares it; OU states the target in prose |
| **An `expr` (or `reference_value_at_T`) on `sde_bm_with_drift`'s `martingale` invariant** | [§K3](#k3-1as-invariants-row-is-not-a-transcription-and-two-live-plans-fail-on-artifacts) | prose-only, so the kernel measures the wrong quantity |

---

## K. Corrections this file makes to the plan

Three, all measured 2026-10-14. The third changes one of 1a's four rows.

### K1. [REV 33]'s inferred count is 9, not 10

[REV 33] reads *"10 carry `scheme_family_source: inferred`, only `sde_double_well_langevin`'s three
declare it"* — which sums to 13 across 12 plans. Measured two independent ways, both giving 9/3:
`plan_meta.read` over all 12, and `config.scheme_family.source` in the metrics (`"inferred"` ×9,
`"plan"` ×3). `detail.orders_ok.strong_band` agrees — `"two-sided"` on exactly the three
`double_well` plans.

The conclusion is unaffected. Two details worth having:
`ginzburg_landau_s6/1-tamed-euler-maruyama` infers `euler-maruyama`, not `tamed-euler-maruyama`, and
`/2-tamed-milstein` infers `milstein` — so `FAMILY_ORDER_KEY` does select
`expected_strong_order_milstein` there, giving it `order_floor: 1.0` exactly as [REV 40] describes.

### K2. Criterion 20 measures plumbing, not detection

[REV 40]'s prediction is confirmed exactly. Adding `expected_strong_order_milstein: 1.0` to
`sde_cir_feller_violated/problem_spec.json` flips the gate with no kernel change:

```
today                  expected_strong=0.5  candidates=[0.5]       run_crn_ladder(>=1.0)=False
with _milstein: 1.0    expected_strong=1.0  candidates=[0.5, 1.0]  run_crn_ladder(>=1.0)=True
```

So criterion 20 — *"runs the CRN ladder and reports an order verdict"* — is met by the one-line spec
edit, as the plan says. **But the verdict cannot catch the defect §24 named.** That plan's
`config.scheme_family.source` is `inferred`, so `guards` applies the one-sided band, whose bar is
`min(candidates) - 0.4` = `min(0.5, 1.0) - 0.4` = **0.1**. §24's motivating case is *"a Milstein
plan with a broken correction term"*, which measures around 0.5 — Euler-like — and passes a 0.1 bar
comfortably. D2 does not change this: it makes the band one-sided when the source is
`kernel_default`, and after the edit the source is `spec` with `declared` still false, so the
existing one-sided logic applies either way.

**`ginzburg_landau_s6/2-tamed-milstein` is the live proof**: `order_floor: 1.0`,
`observed_order: 0.644`, `orders_ok: True`. A third below its declared floor, and a pass.

Catching it needs the plan to declare `scheme_family: milstein`, making the band two-sided at
`|0.644 - 1.0| = 0.356` — which still passes the 0.4 band, so even that is not sufficient on this
plan. Record criterion 20 as asserting the ladder runs, and add a separate assertion that a
*deliberately broken* Milstein correction is caught, which is what `sde_milstein_post_step.py` is
for.

### K3. 1a's invariants row is not a transcription, and two live plans fail on artifacts

**Three measured facts.**

**(a) Two live Path-A plans have `checks.invariants_ok: False` and score 10.**

| Plan | failing invariant | `max_drift` | `worst_headroom` | gate? |
|---|---|---|---|---|
| `bm_with_drift/1-euler-maruyama` | `martingale` | 0.50258 | −1.400 | no |
| `gbm_2d_high_corr/1-euler-maruyama` | `martingale` | 0.10475 | −9.020 | no |

They score 10 because **neither** `rubric_sde_analytic` **nor** `rubric_sde` reads `invariants_ok`.

**(b) Both failures are artifacts of `_martingale`, not solver defects.**
[`invariants.py:557`](../verifylib/kernel/invariants.py#L557) computes
`|mean(X[:, 0]) - reference_value_at_T or x0|`. Its recognised keys are
`("ci_mult", "reference_value_at_T")` — **`expr` is not among them and is silently ignored.** So:

* `sde_gbm_2d_high_corr` declares `expr: "np.exp(-mu1*T) * X[..., 0]"`. Ignored, so the kernel
  computes `|E[X_T] - X_0| = e^(mu1*T) - 1 = 0.10517`. Measured **0.10475** — **0.27 standard
  errors** away. The discount the spec asked for never happens.
* `sde_bm_with_drift` declares the martingale in **prose only** — no `expr`, no
  `reference_value_at_T`, just a note reading *"M_t = X_t - mu*t is a martingale, so
  E[X_T - mu*T] = X_0"*. The kernel computes `|E[X_T] - X_0| = mu*T = 0.5`. Measured **0.50258** —
  **1.92 standard errors** away.

Both are the kernel measuring a quantity the spec did not ask for. `sde_bm_with_drift`'s solver is
pathwise exact; neither plan is shown to be wrong by this.

**(c) The row is a new judgement, not a transcription.** 1a says *"Nothing here is a new
judgement"*, and that is true of three of its four rows. It is not true of
`failed(invariants_ok) (non-gate) -> 5`: `grep` confirms **no SDE rubric reads `invariants_ok` at
all**, so there is no Path-B row to mirror and D1's parity argument does not apply to it.

**What follows.**
1. **`_martingale` belongs in step 0** alongside `_widen` and `strong_order`'s error bar. It is the
   same shape as both: a measured kernel bug, currently masked, whose unmasking is scheduled by a
   later step. Either honour `expr`, or reject a `martingale` entry carrying an `expr` the evaluator
   cannot use — silently ignoring it is the fails-open shape of Finding 9.
2. **`sde_bm_with_drift`'s spec needs `reference_value_at_T: 1.5`** (or an `expr`), which is a
   one-line edit in the same class as [REV 40]'s. Added to [§J5](#j5-spec-and-artifact-work-that-gates-specific-criteria).
3. **1a's invariants row must not ship before both.** As drafted it drops two live, plausibly
   correct Path-A plans from 10 to 5 on mis-measured invariants — the exact false-hard-fail-on-correct-work
   class the plan exists to avoid, and the reason this file checks `detail.constraints_ok.outcomes`
   rather than the score.
4. `_martingale` also carries the `reshape(...)[:, 0]` component-0 reduction — a **fifth** site
   after the three in `crn.py` and `dynkin._phi_values`. On `gbm_2d_high_corr` that means the
   spec's *"Same for Y with mu2"* is never tested at all. Fold it into 2c.


---

## L. Measured after implementation

Read 2026-10-15 from the `_eval_metrics.json` files written by a fresh 12-plan sweep (**2 240 s
wall, exit 0**, all 12 plans at their declared `evaluation_thresholds`, kernel at tree `d5bb447` +
the implementation). Same fields, same order as [§B](#b-the-objective-baseline--12-live-plans-field-by-field).
Nothing above was edited: §B is the before-column.

**The headline, stated in the terms §0 asks for.** One score moved — `ginzburg_landau_s6/3`, 9 → 10,
exactly as [§E](#e-what-changes-on-the-12-live-plans) predicted. ~~Of the **fourteen fields §B
records**, **35 (plan, field) pairs moved, across 9 of the 12 plans**~~ — **[REV 69]: 61 pairs
moved, across 9 of the 12 plans, over 19 of the 25 fields §B's three tables record.**

~~§E predicted *"nineteen field values move across nine plans"*. **The plan count is exactly right**
and the field count is **35**, nearly double, because §E listed the fields it could name and four
more moved with them: `expected_strong_candidates` (4 plans), `order_band` (3), `order_floor` (1)
and `expected_strong_order` (1) — nine pairs §E had no row for. Two of §B's fourteen fields did not
move on any plan: `bound_by` and `gates_clean`. The claim §E rests on ("the field-level diff is
roughly twenty times as informative as the score diff") measures out at **35 to 1**, which is the
reason this file is written the way it is and the one number a score-anchored before-column could
not have produced.~~

**[REV 69] — the count above was taken by hand against the fields §E could name, and it was low.**
An independent evaluation counted ~60 over the same data; re-derived by parsing §B's own three
tables out of this file and comparing every cell against the live metrics, the number is **61**. §E
predicted *"nineteen field values move across nine plans"*: **the plan count is exactly right** and
the field count is more than three times it. Two rules decide the number, and both of them flatter
the result if left loose:

* **Significant figures.** §B transcribed `double_well_langevin/2`'s `s1` as `0.0145`; the live
  value is `0.014453898`. That is not a move, it is three significant figures — and no fixed
  relative tolerance expresses it, since the same table's `dynkin_bar` agrees to 1e-5 while this
  cell differs by 3.2e-3. The rule is "round the live value to the precision §B wrote". **61 cells**
  are unchanged under it and would have counted as moves without it.
* **A dash is not a value.** `—` in §B means either "the check did not run" or "not applicable", so
  moves *from* a dash are counted separately: **30 of the 61 moved from a value §B recorded, and 31
  from a dash** — the second number being, almost exactly, the checks that did not run before.

| | |
|---|---|
| cells compared | 300 |
| moved | **61**, across **9 of 12** plans, over **19 of 25** fields |
| …from a recorded value | 30 |
| …from a `—` | 31 |
| unchanged | 239, of which 61 round to §B's own figures |
| fields that moved on **no** plan | `bound_by`, `gates_clean`, `reference_outcome`, `scheme_family`, `scheme_family.source`, `expected_strong_order.source` |

So the claim §E rests on — "the field-level diff is roughly twenty times as informative as the score
diff" — measures out at **61 to 1**, three times §E's own estimate. The methodology matters more
than the number: a count produced by hand can be wrong in the direction that flatters it, and twice
now it has been. The script is in the session scratchpad and the two rules above are the whole of
it.

The three plans with **nothing** moved are `double_well_langevin/1, 2, 3` — unchanged under the
re-derivation above, and the one thing both counts agree on — which is
[§F5](#f5-three-working-truncation-guards-must-not-be-re-measured) satisfied exactly, not a
coincidence: they were the only three plans already at `tier: A_prime` with `route: kolmogorov`,
a declared `scheme_family`, a two-sided band and a clean Dynkin before any of this shipped.

### L1. The order study — after

| Plan | P | `orders_ok` | `observed_order` | `se_order` | `order_floor` | `s0` | `s1` | `strong_band` | `candidates` | `run_crn_ladder` |
|---|---|---|---|---|---|---|---|---|---|---|
| `bm_with_drift/1-euler-maruyama` | A | `unavailable` | 2.238 | 0.0137 | 1 | 9.38e-15 | 1.99e-15 | one-sided |  [1.0] | `True` |
| `cir_feller_violated/1-euler-maruyama` | A | `True` | 0.384 | 0.0152 | 0.5 | 0.088 | 0.0674 | one-sided |  [0.5, 1.0] | `True` |
| `cir_feller_violated/2-milstein` | A | `True` | 0.326 | 0.019 | 1 | 0.0726 | 0.0579 | one-sided |  [0.5, 1.0] | `True` |
| `double_well_langevin/1-tamed-heun-drift-trapezoid` | B | `True` | 1.046 | 0.0363 | 1 | 0.03 | 0.0145 | two-sided |  [1.0, 1.0] | `True` |
| `double_well_langevin/2-drift-implicit-trapezoid` | B | `True` | 1.026 | 0.0364 | 1 | 0.0294 | 0.0145 | two-sided |  [1.0, 1.0] | `True` |
| `double_well_langevin/3-strang-splitting-exact-flows` | B | `True` | 1.020 | 0.0363 | 1 | 0.0293 | 0.0144 | two-sided |  [1.0, 1.0] | `True` |
| `gbm_2d_high_corr/1-euler-maruyama` | A | `True` | 0.513 | 0.0113 | 0.5 | 0.00282 | 0.00198 | one-sided |  [0.5] | `True` |
| `ginzburg_landau_s6/1-tamed-euler-maruyama` | B | `unavailable` | 0.452 | 0.524 | 0.5 | 0.723 | 0.529 | one-sided |  [0.5, 1.0] | `True` |
| `ginzburg_landau_s6/2-tamed-milstein` | B | `unavailable` | 0.644 | 0.415 | 1 | 0.45 | 0.288 | one-sided |  [0.5, 1.0] | `True` |
| `ginzburg_landau_s6/3-log-transform-euler` | B | `True` | 1.019 | 0.0356 | 0.5 | 0.0211 | 0.0104 | one-sided |  [0.5, 1.0] | `True` |
| `multichannel_stiff_m13/1-euler-maruyama` | A | `True` | 0.783 | 0.0247 | 0.5 | 0.091 | 0.0529 | one-sided |  [0.5] | `True` |
| `ornstein_uhlenbeck/1-euler-maruyama` | A | `True` | 1.008 | 0.00588 | 1 | 0.00512 | 0.00255 | one-sided |  [1.0] | `True` |

Five things to read straight off it, against [§B1](#b1-the-order-study):

* **`run_crn_ladder` is now `True` on all 12.** It was `False` on two. `gbm_2d_high_corr/1` produces
  a verdict at order 0.513 and **does not carry `m13`'s `IndexError`** — which is the
  step-2-before-step-7 check §B1 asked for, passed.
* **`m13`'s `"unavailable"` is gone**: `orders_ok: True` at order **0.783**, no exception in the
  reason. Finding 2 is closed on the live plan that carried it.
* **`se_order` exists on all 12 and spans 0.0059 to 0.524** — a 90× range. The two plans whose `se`
  exceeds the 0.4 band (`ginzburg/1` at 0.524, `ginzburg/2` at 0.415) are exactly the two that now
  abstain. [REV 36a]'s 12.4× order spread was this, unmeasured.
  **[REV 69] — which rule fired is not the one this bullet first named.** Both conditions hold on
  both plans (`2·se_order` = 1.049 and 0.830, each above the 0.4 band), but `crn.abstention` tests
  the **near-zero** rule first, and that is what `detail.orders_ok.abstained` records:
  *"s1 = 0.529 is within 2 standard errors of zero (0.284), so the ratio is noise"* on plan 1 and
  the same sentence at `s1 = 0.288` against `0.209` on plan 2. The outcome is right either way and
  the cause stated here was not; the distinction matters because the two rules say different
  things — one is "the rate's interval is wider than the band", the other is "the finer pathwise
  difference is indistinguishable from zero, so there is no ratio to take".
* **`ginzburg_landau_s6/2` no longer passes a floor it misses by a third.** §B1 called this "the
  cleanest single demonstration that the one-sided band is near-vacuous": order 0.644 against floor
  1.0, `orders_ok: True`. It is now `"unavailable"` — recorded reason `s1 = 0.288` within `2·se_s1`
  of zero, with `se_order = 0.415` also above the band ([REV 69]).
* **`cir_feller_violated/2`'s floor moved 0.5 → 1.0** on the one-line spec edit, and the plan still
  passes at order 0.326 — because the band is one-sided off `min([0.5, 1.0]) - 0.4 = 0.1`.
  [§K2](#k2-criterion-20-measures-plumbing-not-detection)'s objection stands after implementation:
  criterion 20 measures plumbing. The order *is* 0.326 and that is the honest reading of a truncated
  scheme on a Feller-violated CIR — see [§L6](#l6-four-places-this-documents-own-predictions-were-wrong) item 2.

### L2. Dynkin, the reference and the certification path — after

| Plan | `dynkin_ok` | `dynkin_z` | `dynkin_bar` | `per_function_ok` | `surrogate_ok.route` | `reference_outcome` | `tier` | `provenance` | `bound_by` | `gates_clean` |
|---|---|---|---|---|---|---|---|---|---|---|
| `bm_with_drift/1` | `True` | 0.685 | 2.428 | `[T, T, T]` | `analytic_moments` | `validated` | `A` | `analytic` | rubric | `True` |
| `cir_feller_violated/1` | `not_run` | — | — | — | `analytic_moments` | `validated` | `A` | `analytic` | rubric | `True` |
| `cir_feller_violated/2` | `unresolved` | 5.137 | 2.428 | `[T, T, F]` | `analytic_moments` | `validated` | `A` | `analytic` | rubric | `True` |
| `double_well_langevin/1` | `True` | 1.690 | 2.428 | `[T, T, T]` | `kolmogorov` | — | `A_prime` | `surrogate` | rubric | `True` |
| `double_well_langevin/2` | `True` | 1.741 | 2.428 | `[T, T, T]` | `kolmogorov` | — | `A_prime` | `surrogate` | rubric | `True` |
| `double_well_langevin/3` | `True` | 1.765 | 2.428 | `[T, T, T]` | `kolmogorov` | — | `A_prime` | `surrogate` | rubric | `True` |
| `gbm_2d_high_corr/1` | `True` | 2.157 | 2.531 | `[T, T, T, T]` | `analytic_moments` | `validated` | `A` | `analytic` | rubric | `True` |
| `ginzburg_landau_s6/1` | `False` | 3.242 | 2.428 | `[T, F, T]` | `kolmogorov` | — | `A_prime` | `surrogate` | rubric | `False` |
| `ginzburg_landau_s6/2` | `False` | 3.136 | 2.428 | `[T, F, T]` | `kolmogorov` | — | `A_prime` | `surrogate` | rubric | `False` |
| `ginzburg_landau_s6/3` | `True` | 1.099 | 2.428 | `[T, T, T]` | `kolmogorov` | — | `A_prime` | `surrogate` | rubric | `True` |
| `multichannel_stiff_m13/1` | `not_run` | — | — | — | `analytic_moments` | `validated` | `A` | `analytic` | rubric | `True` |
| `ornstein_uhlenbeck/1` | `True` | 1.571 | 2.428 | `[T, T, T]` | `analytic_moments` | `validated` | `A` | `analytic` | rubric | `True` |

Four things, against [§B2](#b2-dynkin-the-reference-and-the-certification-path):

* **All three `ginzburg_landau_s6` plans moved to `tier: A_prime` with `route: "kolmogorov"`.** §B2:
  *"That is the check for step 0 — not the score, which moves on only one of the three."* Passed on
  all three; the score moved on one.
* **Four of the six Path-A plans now carry a Dynkin verdict**, at z = 0.69 (`bm_with_drift`), 2.16
  (`gbm_2d`), 1.57 (`ornstein_uhlenbeck`) against a 2.43–2.53 bar — all three clean. The fourth,
  `cir_feller_violated/2`, is `"unresolved"` at z = 5.14 and is [REV 56]'s whole story.
* **`gbm_2d_high_corr/1` runs 4 test functions against a bar of 2.531**, not 3 against 2.428: the
  declared `verification.generator` sends it down `generator_nd` and the family-wise multiplier
  widens for the fourth function. That is 2b and D3 working end to end on a live plan.
* **`per_function_ok` on `ginzburg/1` and `/2` is `[T, F, T]`, identical to §B2.** Same function,
  same z to three decimals. Nothing about the surrogate restoration touched the Dynkin verdict,
  which is what [§F6](#f6-ginzburg_landau_s6-plans-1-and-2-must-stay-at-dynkin_ok-false) required.

### L3. Configuration provenance and the new checks — after

| Plan | `expected_strong_order` | `.source` | `noise_dimension` | `invariants_ok` | `cross_moment_ok` | `terminal_law_ok` | `stability_ok` | `reported_moments_agree` |
|---|---|---|---|---|---|---|---|---|
| `bm_with_drift/1` | 1 | spec | `null` *(kernel_default)* | `True` | — | `True` | — | `True` |
| `cir_feller_violated/1` | 0.5 | spec | `null` *(kernel_default)* | `True` | — | — | — | `True` |
| `cir_feller_violated/2` | 1 | spec | `null` *(kernel_default)* | `True` | — | — | — | `True` |
| `double_well_langevin/1` | 1 | spec | `null` *(kernel_default)* | `True` | — | — | — | `True` |
| `double_well_langevin/2` | 1 | spec | `null` *(kernel_default)* | `True` | — | — | — | `True` |
| `double_well_langevin/3` | 1 | spec | `null` *(kernel_default)* | `True` | — | — | — | `True` |
| `gbm_2d_high_corr/1` | 0.5 | spec | 2 | `True` | `True` | `True` | — | `True` |
| `ginzburg_landau_s6/1` | 0.5 | spec | `null` *(kernel_default)* | `True` | — | — | — | `True` |
| `ginzburg_landau_s6/2` | 1 | spec | `null` *(kernel_default)* | `True` | — | — | — | `True` |
| `ginzburg_landau_s6/3` | 0.5 | spec | `null` *(kernel_default)* | `True` | — | — | — | `True` |
| `multichannel_stiff_m13/1` | 0.5 | spec | 13 | `True` | — | — | — | `True` |
| `ornstein_uhlenbeck/1` | 1 | spec | `null` *(kernel_default)* | `True` | — | `True` | — | `True` |


Against [§B3](#b3-configuration-provenance):

* **`invariants_ok` is `True` on all 12.** It was `False` on `bm_with_drift/1` and `gbm_2d_high_corr/1`,
  both scoring 10 on artifacts of the kernel's own martingale evaluator — [§K3](#k3-1as-invariants-row-is-not-a-transcription-and-two-live-plans-fail-on-artifacts)'s
  finding, now closed. The `expr` the specs declare is read as the martingale *process*, over every
  component.
* **`expected_strong_order.source` is still `spec` on all 12**, so D2's `kernel_default` branch
  remains unexercised on this pool, exactly as §B3 predicted. It is covered by
  `sde_spec_defaulted_floor.json` instead, and criterion 13 is asserted there.
* **`stability_ok` is blank on all 12** — see [§L6](#l6-four-places-this-documents-own-predictions-were-wrong)
  item 3. [REV 62] gives it and `_milstein_claim` the assertions they shipped without.
* **[REV 69]: the `noise_dimension` column above read `1 (default)` and the recorded value is
  `null`**, with `source: kernel_default`. Not a cosmetic difference: `1` would be the kernel
  *asserting* a scalar noise dimension, and `null` is it declining to — `crn.ladder_spec` builds a
  rank-2 increment array when `m` is absent and a rank-3 one when it is given, so the two spellings
  select different code. The ten scalar plans record the absence, and only `gbm_2d_high_corr` (2)
  and `m13` (13) record a number, both `source: spec`.
* **`reported_moments_agree` is `True` on all 12.** The check that reads `empirical_mean` and
  `empirical_variance` back against the kernel's own found **no disagreement anywhere**, which is
  the honest outcome to report for Finding 12: the fields were unread, not wrong.

### L4. Scores, for context — and the one that moved

| Plan | before (§E) | after | moved? |
|---|---|---|---|
| `bm_with_drift/1` | 10 | **10** | — |
| `cir_feller_violated/1` | 8 | **8** | — |
| `cir_feller_violated/2` | 10 | **10** | — *(but see [REV 56]: it went to 3 and back)* |
| `double_well_langevin/1,2,3` | 10, 10, 10 | **10, 10, 10** | — |
| `gbm_2d_high_corr/1` | 10 | **10** | — |
| `ginzburg_landau_s6/1` | 3 | **3** | — |
| `ginzburg_landau_s6/2` | 3 | **3** | — |
| `ginzburg_landau_s6/3` | 9 | **10** | **+1** |
| `multichannel_stiff_m13/1` | 6 | **6** | — |
| `ornstein_uhlenbeck/1` | 10 | **10** | — |

`certified` is `True` on 8 of 12 and `False` on `cir_feller_violated/1`, both failing `ginzburg`
plans and `m13` — the same four, for the same stated reasons, as before.

### L5. What moved, step by step

Each row is a step of the plan's §7 and the field that shows it worked. **None of these is a score.**

| Step | Field | Before (§B) | After |
|---|---|---|---|
| 0, `_widen` | `detail.surrogate_ok.route` on all 3 `ginzburg` | `null` | **`kolmogorov`** |
| 0, `_widen` | `certification.tier` on all 3 `ginzburg` | `C` | **`A_prime`** |
| 0, `_widen` | `certification.provenance` on all 3 `ginzburg` | `self_convergence` ×2, `manufactured_partial` | **`surrogate`** ×3 |
| 0, `strong_order` | `summary.observed_order_se` | *did not exist* | **0.0059 – 0.524** on all 12 |
| 0, `strong_order` | `checks.orders_ok` on `bm_with_drift/1` | `True` *(from `s0` = 9.4e-15)* | **`unavailable`** |
| 0, `strong_order` | `checks.orders_ok` on `ginzburg/1,2` | `True` at 0.452 / 0.644 | **`unavailable`** *(recorded cause: `s1` within `2·se_s1` of zero — 0.529 vs 0.284 and 0.288 vs 0.209. `se_order` 0.524 / 0.415 is also above the 0.4 band, but `abstention` tests the near-zero rule first; [REV 69])* |
| 0, `_martingale` | `checks.invariants_ok` on `bm_with_drift/1`, `gbm_2d/1` | **`False`** | **`True`** |
| 0, spec edit | `summary.order_floor` on `cir_feller_violated/2` | 0.5 | **1.0** *(the `_milstein` key)* |
| 2, `noise_dimension` | `config.noise_dimension.value` | *did not exist* | **2** on `gbm_2d`, **13** on `m13`, default 1 elsewhere |
| 2, `noise_dimension` | `checks.orders_ok` on `m13/1` | **`unavailable`** *(an `IndexError` in the reason)* | **`True`** at order 0.783 |
| 3a, cross moment | `checks.cross_moment_ok` on `gbm_2d/1` | *did not exist* | **`True`**, ρ graded against 0.9478 |
| 3b, KS tier | `checks.terminal_law_ok` | *did not exist* | **`True`** on the 3 specs that now declare a law |
| 3b, tier 1 | `skewness` / `excess_kurtosis` with `ci_mult_sigma` | *did not exist* | recorded, `graded: false` ([REV 54]) |
| 4, `_reported_moments` | `checks.reported_moments_agree` | *did not exist* | **`True`** on all 12 |
| 7, 1c | `summary.order_band` | *did not exist* | `one-sided` ×9, `two-sided` ×3 |
| 8, 1b | `config.run_dynkin.value` on the 6 Path-A plans | **`False`** ×6 | **`True`** ×4, `False` ×2 *(see below)* |
| 8, 1b | `checks.dynkin_ok` on `bm_with_drift/1`, `gbm_2d/1`, `ornstein_uhlenbeck/1` | `not_run` | **`True`** at z = 0.69 / 2.16 / 1.57 |
| 10, 2b | `detail.dynkin_ok.test_functions` on `gbm_2d/1` | *3, via the scalar path* | **4, via `generator_nd`**, bar 2.531 |

### L6. Four places this document's own predictions were wrong

The point of writing predictions down.

1. **`cir_feller_violated/1` and `m13/1` keep `dynkin_ok: not_run`.**
   [§E](#e-what-changes-on-the-12-live-plans) says *"`not_run` → a verdict, **must not be `False`**"*
   for `cir_feller_violated/1`, and §B2 says `run_dynkin` is `False` on **all six** live Path-A
   plans and *"step 8 flips it"*. It flips four. §24's cost rule keys on `would_certify`, and these
   two plans cannot reach certification for reasons that have nothing to do with Dynkin —
   `moments_ok: False` on the first, `resolved: False` on the second — so running the generator
   check could not change either outcome and it correctly does not run. The prediction read the gate
   as `path == "A"` when it is `path == "B" or would_certify`.
2. **`cir_feller_violated/2` did come back `False`, and §F2 called it.** *"`cir_feller_violated` is
   the one to watch: full-truncation Euler on a Feller-violated CIR has real mass near zero"* —
   right, and it cost a correct solver 7 points until the extrapolation was fixed. What §F2 got
   wrong is the **mechanism**: it expected a seed-driven false failure and prescribed D9's
   median-over-3-seeds, but at `|z − bar| = 2.71` the replication correctly declined to fire. The
   cause was the ladder's `assumed_1` extrapolation against an order-0.2 residual. Right call, wrong
   remedy — see [plan §12b](plan-sde-verification-gaps.md) and [REV 56].
3. **`stability_ok` is `—` on all 12.** §C10 and D5 treat `stability_check` as a promised check that
   starts existing. It exists, and **no live spec declares a structured one** — `cir_feller_violated`
   and `multichannel_stiff_m13` both carry `stability_check: null`. So D5 ships correct and
   unexercised on this pool, covered by a canary only. [§J5](#j5-spec-and-artifact-work-that-gates-specific-criteria)'s
   formulator work is what would exercise it, and §E should have listed this under "not predictable
   yet".
4. **`bound_by` stayed `"rubric"` on all 12.** §B2 expected step 8's fabrication cap to produce *"the
   first `bound_by: ceiling` in the set"*. It does — on the canary. No live plan fabricates its
   paths, so there was never going to be a live `ceiling` binding, and the prediction confused "the
   mechanism now exists" with "the pool exercises it".

### L7. What held, and is worth saying so

Every one of §F's eight must-not-change assertions holds.

| | Assertion | Measured |
|---|---|---|
| F1 | the honest-solver guard still scores 10 | `test_the_same_sde_solver_at_full_path_count_scores_ten` passes |
| F2 | no live Path-A plan acquires `dynkin_ok: False` | holds on 4 of 4 — after [REV 56]; one of them had to be fixed to get there |
| F3 | a correct solver on a defaulted floor is not failed | `sde_spec_defaulted_floor.json`, `orders_ok: True` at 1.0, band `one-sided` |
| F4 | `strong_band` stays `one-sided` on the nine inferred plans | 9 one-sided, 3 two-sided, unchanged |
| F5 | the three `double_well` truncation guards are not re-measured | scores, `z` and `per_function_ok` all identical |
| F6 | `ginzburg/1,2` stay at `dynkin_ok: False` | `False`, z = 3.242 and 3.136, `per_function_ok: [T, F, T]` — identical to §B2 |
| F7 | an abstention costs nothing | `m13` 6 → 6 and `bm_with_drift` 10 → 10, both now abstaining |
| F8 | the suite is green | **643 passed, 3 skipped, 4 deselected, 0 failed** of 650 collected (was 580 / 3 / 4 of 587) |

F6 is the one that mattered most and is the easiest to break: the two `ginzburg` plans gained a
working surrogate at step 0 and still score 3, because `failed(dynkin_ok) → 3` sits ahead of the
surrogate row. The row ordering `a507038` established is intact.

### L8. Cost, measured

| | Before | After |
|---|---|---|
| Suite | 368.9 s | **592.9 s** (563.9 for `verifylib/tests`, 29.0 for the rest) |
| 12-plan sweep | 2 010 s (§B's 33.5 min) | **2 240 s** |
| Slowest single plan | `ginzburg/1` at 588 s *(post-step-0 sweep)* | `ginzburg/1` at 533 s |
| Slowest plan that used to be free | `gbm_2d/1` at **0 s** | **397 s** — it now runs both ladders at `m = 2` |
| Slowest single test | — *([REV 48] recorded only that 11 SDE tests were 276.8 s of 368.9 s)* | 46.3 s, `test_paths_outside_a_declared_support_are_a_gate_violation` |

The sweep grew 11% and the suite 61%. `gbm_2d_high_corr/1` going from 0 s to 397 s is the whole cost
story in one line: it was free because neither ladder ran on it, which is what Finding 2 and
Finding 1 *were*.

---

## M. The second round, measured

[§L](#l-measured-after-implementation) is the after-column for the plan's eleven steps. This section
is the after-column for the **correction round** that followed it: a separate session graded the
shipped work against this file, and the plan's
[§13](plan-sde-verification-gaps.md#13-second-round--from-an-independent-evaluation-2026-10-15)
records what it found and what [REV 58]–[REV 70] did about it. Three readings in §L itself were
wrong and are corrected in place, each marked `[REV 69]`.

### M1. What the evaluation confirmed

Worth recording, because it is the part a correction round makes invisible. Checked against the
metrics rather than the scores:

| | |
|---|---|
| [§F](#f-what-must-not-change) | **all eight** must-not-change assertions hold |
| [§I](#i-if-the-expected-result-does-not-appear) | every listed failure mode avoided |
| [§C8](#c8-the-kolmogorov-surrogate-works-again-on-a-one-sided-domain) | `truncation_shift` max **0.0036081606888128783** against §C8's predicted **0.00361** — exact to three figures |
| [§K3](#k3-1as-invariants-row-is-not-a-transcription-and-two-live-plans-fail-on-artifacts) | closed, and past what it asked: `invariants_ok` is `True` on all 12 with real measured drifts, not `not_reported` |
| [§D](#d-the-23-acceptance-criteria-with-an-objective-anchor) criterion 23 | `STATE.md` agrees with the metrics on score and provenance **12/12** (§B had 6 of 12 divergent) |
| past the plan | 2b runs on a live plan (`gbm_2d_high_corr/1`, 4 test functions via `generator_nd`, validated to 8.3e-9 against an exact generator), and the KS tier **grades** rather than reporting |

### M2. The nine shortfalls, and where each landed

| the evaluation's finding | severity | closed by |
|---|---|---|
| the `*_value_at_T` fallback keyed on the expression being absent, not unevaluable — §C7's sentence verbatim | a 9 where a 10 was available | [REV 58] |
| a crashed CRN ladder collected D8b's abstention credit | a 9 on no evidence | [REV 59] |
| the Dynkin budget refusal did not cap — Finding 1's hole through the budget door | a 10 on no generator evidence | [REV 60] |
| `_generator_nd` had no test | unguarded | [REV 61] |
| `_milstein_claim` and `stability_ok` were unexercised and unasserted | unguarded | [REV 62] |
| criterion 7's per-family half was prose in an error message | unenforced | [REV 63] |
| `_headroom` ignored the `ci_mult·se` half of a statistical bar | a ranking anchor reading −6.75 on a pass | [REV 64] |
| "has a consumer" was satisfied by a docstring mention | weak guard | [REV 65] |
| `bm_with_drift`'s `gaussianity` was permanently `not_reported` with a `SyntaxError` | spec defect, unrefused | [REV 66] |

One further item is open by choice and one was not a defect:

* **Criterion 3 is still two of four multi-noise problems.** The mechanism is covered — live at
  `m = 2` and `m = 13`, unit at `m ∈ {1,2,3}`, and both missing problems are `m = 2` — and the two
  live artifacts are formulator work. Plan
  [§13d](plan-sde-verification-gaps.md#13d-what-this-round-still-leaves-open).
* **`m13`'s new `richardson_stable: False` is signal.** No prediction covered it, and the question
  it raised — bias or Monte Carlo scatter? — was unanswerable from the recorded detail. Measured in
  the paired standard error the CRN ladder makes available: **33.2 σ** on component 0, 9.6 σ on
  component 1. [REV 68] reports the number and leaves the verdict where it was.

### M3. The suite, and the environment it was measured on

The evaluation ran the same tree and got **638 passed / 5 failed**, all `np.trapezoid`, all PDE-side,
all reproducing with the entire diff stashed — on numpy **1.26.4**, which satisfied
`pyproject.toml`'s declared `numpy>=1.26.0`. The declared floor was wrong, not the evaluation.

| | §J1 baseline | after round 1 | after round 2 |
|---|---|---|---|
| suite | 580 / 3 / 4 of 587 | 643 / 3 / 4 of 650 | **690 / 3 / 4 of 697** |
| wall | 368.9 s | 592.9 s | **614.5 s** |
| declared numpy floor | `>=1.26.0` | `>=1.26.0` | **`>=2.0.0`**, held by `test_project_metadata.py` |

Every suite number in this file and in the plan was measured on
`python 3.14.7 / numpy 2.4.2 / scipy 1.17.1 / macOS-15.7.3-arm64`, now recorded in
[§J1](#j1-the-suite--2026-10-14-tree-d5bb447).

### M4. The 12 live plans — what round 2 moved

Round 1's sweep and round 2's, same 12 plans, same declared thresholds. **One score moved and
7 `(plan, field)` pairs moved across 4 plans** — small by design: round 2 corrected how evidence is
*priced*, and only one live plan sits on a repriced row.

| plan | round 1 | round 2 | what moved |
|---|---|---|---|
| `bm_with_drift/1` | 10 | 10 | `orders_ok` / `order_check` `unavailable` → **`unresolved`** — the token rename. It is the roundoff abstention, so it still earns D8b's credit and the score is unchanged |
| `cir_feller_violated/1` | 8 | 8 | — |
| **`cir_feller_violated/2`** | **10** | **9** | `dynkin_ok: "unresolved"`, so [REV 60] withholds certification. `bound_by: rubric`, provenance still `analytic`, `moments_ok` still `True` |
| `double_well_langevin/1,2,3` | 10, 10, 10 | 10, 10, 10 | — |
| `gbm_2d_high_corr/1` | 10 | 10 | — (its `invariant_worst_headroom` moved, −6.747 → **+0.692**, which is [REV 64]; that field ranks and does not score) |
| `ginzburg_landau_s6/1` | 3 | 3 | `orders_ok` → `"unresolved"`. Still `dynkin_ok: False` at z = 3.242 — [§F6](#f6-ginzburg_landau_s6-plans-1-and-2-must-stay-at-dynkin_ok-false) |
| `ginzburg_landau_s6/2` | 3 | 3 | `orders_ok` → `"unresolved"`. Still `dynkin_ok: False` at z = 3.136 |
| `ginzburg_landau_s6/3` | 10 | 10 | — |
| `multichannel_stiff_m13/1` | 6 | 6 | — (`richardson_stable` gains `shift_in_se: 33.2`, report-only) |
| `ornstein_uhlenbeck/1` | 10 | 10 | — |

| | round 1 | round 2 |
|---|---|---|
| sweep wall | 2 240 s | **2 379 s**, exit 0 |
| slowest plan | `ginzburg/1`, 533 s | `ginzburg/3`, 590 s |
| `STATE.md` vs the metrics | 12/12 agree | **12/12 agree**; the write-back corrected 1 value and is idempotent |

The four `"unavailable"` → `"unresolved"` moves are the whole visible footprint of [REV 59], and
that is the point of it: the token the *honest* abstention carries changed, and on these three plans
it was always the honest abstention. What changed is that a crashed ladder can no longer borrow its
credit — and no live plan has a crashed ladder, which is why nothing else moved.

### M5. §L's own field-move count, re-derived

§L reported **35** moved `(plan, field)` pairs across 9 of 12 plans. The plan count is right; the
field count was low, because it was taken against a hand-written list of the fields §E could name
rather than against §B's own three tables.

Re-derived by parsing those tables out of this file and comparing cell by cell against the live
metrics, with §B's three-significant-figure transcription absorbed as a tolerance rather than
counted as a move:

| | |
|---|---|
| cells compared | 300 |
| moved | **61**, across **9 of 12** plans, over **19 of 25** fields |
| …from a value §B recorded | 30 |
| …from a `—` (a check that did not run) | 31 |
| unchanged | 239, of which **61** round to §B's own significant figures |
| fields that moved on **no** plan | `bound_by`, `gates_clean`, `reference_outcome`, `scheme_family`, `scheme_family.source`, `expected_strong_order.source` |

So §E's *"nineteen field values move across nine plans"* had **the plan count exactly right** and
the field count low by more than three times, and §L's 35 was low by nearly two. The evaluation
counted ~60 over the same data; the one-cell gap is a dash-versus-value convention, stated above.
The three plans with nothing moved are `double_well_langevin/1, 2, 3` under every count — which is
[§F5](#f5-three-working-truncation-guards-must-not-be-re-measured), satisfied exactly.

The methodology matters more than the number: a before/after whose count is produced by hand is a
count that can be wrong in the direction that flatters it, which is what happened. The script is in
the session scratchpad and the rule it applies is stated above.

---

## Change log

| Date | Change |
|---|---|
| 2026-10-14 | Created from plan rev 3, then re-anchored from scores to field-level observables per review. Baselines read from the 12 live `_eval_metrics.json` files and measured against tree `d5bb447`. Three corrections to the plan in §K, one of which (§K3) changes a row in Phase 1a. |
| 2026-10-15 | Plan implemented, all eleven §7 steps. §L added as the after-column; nothing above it edited except the Status line. **One score moved; 35 (plan, field) pairs moved across 9 of 12 plans.** All eight §F assertions hold. Four of this file's own predictions were wrong and are recorded in §L6 — the largest being that step 8 flips `run_dynkin` on all six Path-A plans when §24's cost rule flips four. §F2's warning about `cir_feller_violated` was right and its prescribed remedy was aimed at the wrong mechanism; see plan [REV 56]. |
| 2026-10-15 (later) | An **independent evaluation** of the implementation, graded against §F and §D. Every §F guard holds and §C8's prediction is exact to three figures; **nine shortfalls** found, answered by plan [REV 58]–[REV 71] and recorded in §M. Three readings in §L were wrong and are corrected in place as [REV 69]: the abstention's recorded cause (the near-zero-`s1` rule, not `se_order`), `noise_dimension`'s recorded value (`null` with `source: kernel_default`, not `1`), and the field-move count — **61, not 35**, re-derived by parsing §B's own tables rather than counting by hand. §J1 now records the interpreter, after the evaluation ran the same tree on numpy 1.26.4 and got five failures against a declared floor that permitted it. |
