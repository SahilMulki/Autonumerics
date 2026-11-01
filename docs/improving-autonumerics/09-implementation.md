# 09 — What was implemented

**Status:** implemented 2026-10-29, on this machine, against the plan in
[03](03-fix-selection.md)–[06](06-retest-protocol.md) as corrected by
[07](07-review.md) and [08](08-free-measurements.md). Every fix below is in the working tree with
an acceptance test; nothing has been re-run through the pipeline yet, and §5 says what that costs.

[06](06-retest-protocol.md) §8 gives the order of work. Steps 0–5 were free and are done; this
document covers steps 0 and 6–16. Steps 17–21 are pipeline runs and are not started.

---

## §1 The fixes, and where they live

| # | What changed | Files |
|---|---|---|
| **F0** | `functionals` crosses the SDE sandbox boundary. `_dump_sde` captured four keys and this was not one of them; `_reconstruct`'s `sde` branch returned before the line that would have restored it. Both dumps now share one `_carry_functionals` | [`verifylib/kernel/sandbox.py`](../../verifylib/kernel/sandbox.py) |
| **F12** | `functional_se` crosses with it, and is declared in the generic contract rather than per problem. A name present in `functional_se` but not in `functionals` is dropped — an error bar on a value that did not cross is not about anything | [`sandbox.py`](../../verifylib/kernel/sandbox.py), [`agents/solver-sde.md`](../../agents/solver-sde.md), [`references/sde_manual.md`](../../references/sde_manual.md) |
| **F2** | `functionals.run_declared` is called from `evaluate_sde`, on the CRN dt ladder. A **gated** functional that has not converged makes the run not converged, as on the PDE path | [`verifylib/kernel/sde/driver.py`](../../verifylib/kernel/sde/driver.py) |
| **F3** | `ranking.accuracy` reads the declared functional where one exists: a dt-bias term from the ladder plus the solver's reported sampling term, against the functional's own `rel_tol` | [`verifylib/ranking.py`](../../verifylib/ranking.py), `sde/driver.py` (`_graded_functionals`) |
| **F16** | **`accuracy` returns `None` where the error is inside its own error bar**, instead of 1.0. See §2 | [`ranking.py`](../../verifylib/ranking.py) (`noise_limited`) |
| **F17** | The SDE driver publishes `design_order`, from a **declared** `scheme_family` or a spec `expected_strong_order` — never from the kernel's 0.5 default, which is a floor the kernel chose rather than a design rate | `sde/driver.py` |
| **F18** | A plan may declare `exact_sampling_components:` in its frontmatter, and the CRN order study then **abstains** on those components instead of failing them | [`plan_meta.py`](../../verifylib/kernel/plan_meta.py), [`sde/crn.py`](../../verifylib/kernel/sde/crn.py), `sde/driver.py`, `agents/plan-creator-sde.md` |
| **F19** | `resolved: false` on the graded quantity blocks `state: stopped` for one cycle, and the retry's single instruction is to raise `num_paths` | [`commands/conductor.md`](../../commands/conductor.md) |
| **F15b** | A per-component order study no longer publishes the **abstaining** component's number as the scalar `observed_order`. It publishes a measured component's, or `None` | `sde/driver.py` |
| **F4** | Cost is dropped — from the weighted component **and** from `rank_key`'s fifth key — wherever the accuracy component was not measurable for every plan in the pool. `solve_time_s` is now **CPU time** where the sandbox could measure it, with `cost_source` saying which | [`sandbox.py`](../../verifylib/kernel/sandbox.py), [`ranking.py`](../../verifylib/ranking.py), both drivers |
| **F1** | `compare` on an SDE compares the declared functionals at the **combined sampling error**, `z = \|a−b\| / hypot(se_a, se_b)`. Above 3σ it withdraws the circularity break while leaving the gate verdict standing | [`verifylib/cli.py`](../../verifylib/cli.py) |
| **F7** | `ranking.decided_by` names the key that actually separated the winner from the runner-up, and `cli rank` prints it with a warning when it is `solve_time_s`, `iter` or `tie` | [`ranking.py`](../../verifylib/ranking.py), [`cli.py`](../../verifylib/cli.py) |
| **F13** | `cli rank` warns, per plan, where `mc_se_rel` exceeds `estimated_rel_error` — the gap the snap already acted on silently | [`cli.py`](../../verifylib/cli.py) |
| **F5** | `sde_manual.md` gains **rule 0**: where the transition law is known in closed form, sample it, with the five families that covers. `plan-creator-sde.md` keys step 1 on spec fields rather than on recall, and adds a variance-reduction step | `references/sde_manual.md`, `agents/plan-creator-sde.md` |
| **F6** | `scheme_family` gains `exact-transition` and `quadratic-exponential`, with inference prefixes. Deliberately **absent** from `FAMILY_ORDER_KEY` — see §3 | `plan_meta.py`, `ranking.py` |
| **F8** | `benchmark/runner.py` registers the solver in `sys.modules` before `exec_module` and **leaves it there**; a failed import pops it | [`benchmark/runner.py`](../../benchmark/runner.py) |
| **F10** | `benchmark.report.harness_error` — one gate-relative number for every grading shape. The per-plan table had read `rel_l2_err`, which does not exist on a functional-graded problem, so the whole suite printed `-` | [`benchmark/report.py`](../../benchmark/report.py) |
| **F9** | Withdrawn in [07](07-review.md); `max_iter` is unchanged at 5 | — |
| **F14** | Not addressed. No fix here touches the evaluator, and that is still true | — |

---

## §2 The one design decision worth arguing with

**F16 could have been written three ways and two of them are wrong.**

The defect ([08](08-free-measurements.md) §2): `headroom_decades` snaps the error up to twice its
own Monte-Carlo noise — correct, two plans inside the same noise differ by the seed — and
`achievable_decades` then divides by **the same half-width**, so the tie lands at exactly 1.0. All
eleven B01–B04 plans snap, all eleven score 1.000, and 0.35 of the index orders nothing.

- **Wrong: lower the ceiling to a flat constant.** `achievable_decades`' docstring already
  disposes of this. Against a flat 6 decades an SDE's accuracy cannot exceed ~0.2, because Monte
  Carlo at the declared path count cannot get further inside the tolerance than its own sampling
  noise, and three correct plans on `sde_double_well_langevin` scored 0.177 / 0.178 / 0.177.
- **Wrong: make the ceiling pool-relative.** `sqi`'s docstring forbids it in as many words —
  *"substituting a pool median makes one plan's index depend on its siblings', which nothing a
  certification reads may do"* — and `accuracy` is in `CERT_COMPONENTS`.
- **Taken: report it as unmeasured.** `None` is already this module's word for "not measurable
  here", `sqi` renormalises over it, and the module's own rule is that `None` is not zero and must
  never be. A plan whose error sits inside its own bar has not been shown to be accurate; it has
  been shown not to have been measured.

**This does not, by itself, pick better plans.** It makes the index honest, which is a
precondition and not a result. What orders a pool is **F3**, and F3 works only because F12's
`functional_se` means the sampling term is *inside* the estimate rather than a floor underneath
it — so the functional error is never noise-limited the way the moment error always is.

And F16 is what makes **F4** necessary rather than optional: with accuracy reporting `None`,
renormalisation hands its 0.35 to whatever remains, and cost goes from a tenth of the index to a
third of it. A component promoted by the absence of the one it is subordinate to is not breaking a
tie. Hence the guard: cost contributes only where every plan in the pool had its accuracy measured,
on both code paths.

---

## §3 What was deliberately not done

- **No order requirement for the new scheme families.** [08](08-free-measurements.md) §4 measured
  six plans across three families reading **0.23–0.31** against declared floors of 0.5, passing
  only because the band is ±0.4 wide. The order study is not in a state to have more requirements
  hung off it. F6 adds the *vocabulary*, which the agreement gate needs; it adds no bar.
- **No second graded-scale solve.** [03](03-fix-selection.md) F3 specifies a dt-Richardson pair at
  the graded path count under common random numbers, costing one extra half-step solve per plan.
  What is implemented reads the bias off the **CRN ladder the evaluation already runs**, at
  `conv_paths` rather than the graded count. That is the cheaper and weaker version, and §5 says
  how to tell whether it is good enough.
- **No replication gate.** The specs declare `num_paths_replicate`, `num_paths_secondary`,
  `second_call_num_paths` and `replication_check`, nothing reads them, and
  [08](08-free-measurements.md) §3's 9–97% per-plan spread is the argument that something should.
  All four are now recorded in `test_spec_field_coverage.UNREAD` **with that argument written
  out**, rather than left to fail the test anonymously. It is a budget decision: a replication gate
  is a second graded-scale solve per plan.
- **`F14`, the evaluator.** No plan's score improved on cycle 2 anywhere on the suite, and nothing
  here changes that.

---

## §4 Test state

Baseline, measured 2026-10-28 and recorded in [06](06-retest-protocol.md) §7a: **1030 passed,
4 failed**. Three of the four are now green:

| was failing | now | why |
|---|---|---|
| `test_spec_field_coverage::test_no_sde_spec_field_is_declared_and_unread_without_a_reason` | **green** | F0/F2 gave the nine BENCHOP functionals a consumer. `evaluation_thresholds.functionals.` joins `parameters.` and `requirements.` as a value subtree — its keys are the problem's own quantity names and the kernel reads the whole map generically. The six remaining orphans are recorded in `UNREAD` with reasons |
| `test_certified::test_frozen_records_fall_back_to_the_old_test_unchanged` | **green** | The corpus moved into subdirectories of `results/` in c21077e and the test listed only the top level; the walk is recursive, and a record carrying `certified` in its `kernel` block is skipped as post-split rather than asserted against the old rule |
| `test_work_precision::test_an_sde_sweep_moves_both_axes_and_orders_by_paths_times_steps` | **green** | The timing flake. Wall time on sub-100 ms solves carries more jitter than the gap between two adjacent levels, so the cost axis is asserted to a tolerance. Fixed first, per §8 step 0, because F4 changes the very field it measures |
| `test_review::test_no_false_positives_on_archived_reviews` | **still red, by design** | A true positive against B02's **archived** metrics block, which still carries the abstaining component's `-1.0772`. F15b fixes the kernel that wrote it; the artefact on disk is unchanged and should stay unchanged until the plan is re-evaluated. Do not "fix" this by editing the block |

New: [`verifylib/tests/test_benchop_fixes.py`](../../verifylib/tests/test_benchop_fixes.py), 26
tests, one per fix plus the four-pool regression that fails if cost ever decides a BENCHOP pool
again.

**Measured after the changes: `uv run pytest verifylib/tests benchmark/tests` gives 1058 passed,
1 failed, 3 skipped** — the one failure being `test_review`, above. Against the 1030/4 baseline
that is 28 new tests passing and three repairs. One pre-existing test needed updating rather than
repairing: `test_kernel_core::test_an_undeclared_plan_falls_back_rather_than_guessing` compares
`plan_meta.read`'s whole dict, so F18's new field had to be acknowledged there. The whole-dict
comparison was kept rather than loosened, which is the point of writing it that way.

---

## §5 What has not been measured, and what it costs

**Nothing here has been through the pipeline.** The ranking changes can be seen immediately,
because `cli rank` reads the metrics blocks on disk:

```
$ uv run python -m verifylib.cli rank workspace/sde_heston_dnt
plan                         score  cert  provenance     sqi    acc    ord    str    res   cost  solve_s
1-euler-maruyama               9.0    no  analytic     1.000      -      -  1.000      -      -   57.888
2-drift-implicit-variance-leg  9.0    no  analytic     1.000      -      -  1.000      -      -   58.915
3-qe-exact-variance            9.0    no  analytic     1.000      -      -  1.000      -      -   64.866

best: 1-euler-maruyama   decided_by: tie
warning: the winner was not chosen on quality [...]
```

Cost no longer decides, and the pool reports itself as a tie rather than ranking on seven seconds
out of sixty-five. On `sde_sabr_cev_call_ii` the index now picks `3-aks-decorrelated-em`, which is
the pool's best plan at all three seeds ([08](08-free-measurements.md) §3) and is not the one that
shipped. On `sde_sabr_cev_call_i` the shipped plan still wins on `score`, 9 against 8 — and **F18
is exactly what that 8 was**, so a re-evaluation should lift the exact-BESQ(0) plan to 9, after
which structure separates them 0.392 to 0.387 in its favour. That is the single clearest prediction
this work makes and it is cheap to check.

Three things are still unmeasured and the order matters:

1. **Re-evaluate B01–B04 with the new kernel** (compute, no credits; ~4 plan-pools). This is the
   first run in which `accuracy` reads a price, `design_order` exists on an SDE, and F18's
   abstention applies. It tests the prediction above, and it tests whether F3's ladder-bias term is
   strong enough to order a pool — §3's deliberate weakening is the thing most likely to be wrong.
2. **Re-run the harness report** to see F10's gate-relative column on a suite that printed `-`.
3. Then [06](06-retest-protocol.md) §6's pre-registered targets, against the restated
   **1.69×** and **8/30 → ≥ 23/30**.

The honest statement of where this leaves things: the measurement layer is fixed and the selection
layer is unblocked, and whether selection actually improves is a question these changes make
answerable rather than one they answer.
