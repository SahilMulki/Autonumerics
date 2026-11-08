# 12 — Implementing 11: what each fix became, and what it corrected along the way

**Status:** written 2026-11-05. The record of implementing
[11-sde-accuracy-fixes.md](11-sde-accuracy-fixes.md) in full — Fix 0 through Fix 6, the three homes
of the `<metrics>` block, the agent obligations, and the test maintenance §9.2 asked for.

Read §1 first if you only read one section. **The acceptance test changed two of the fixes**, one
of them a decision §9.1 had recorded as settled, and both corrections are in the direction the plan
could not have anticipated from the records it was written against.

§7 then put the generation fixes through a live pipeline run, which worked — and produced a third
correction, this time to §6.3 of this document rather than to the plan: the two rank keys ship
different plans on that pool and each is optimal under a different error metric, so what is actually
unsettled is which metric selection optimises for. §7.3 and §7.5.

---

## §1 What the plan got wrong, and how it found out

Two things, and the second is the more interesting.

**1. `cost_source` was a sixth unreachable ranker input, and nobody had counted it.**
[11](11-sde-accuracy-fixes.md) §1 names four: `functional_rel_error`, `functional_rel_tol`,
`order_band` and `kind`. The structural test Fix 1 asked for — *"a test that fails when `ranking.py`
reads a summary key `BLOCK_KEYS` does not carry"* — found a fifth on its first run.
`ranking.cost` reads `cost_source` off every pool member to decide whether the pool mixes CPU and
wall time, and returns `None` for a mixed pool, because a forking solver's wall time is divided by
its worker count. Both drivers write the field. It was not on the block, so the set `cost` reads was
`{None}` on every pool ever ranked and **F4's own mixed-cost guard has never fired.**

That is the test earning its place on the day it landed, and it is worth being precise about why
grep did not find it: §1's audit searched for `summary.get("…")`, and this read is
`s.get("cost_source")` over the pool. A regex looks for the shape you expect; an AST walk looks for
the shape the language has.

**2. Fix 2b's scope was wrong, and the acceptance test said so.**
[11](11-sde-accuracy-fixes.md) §9.1 decision 3 settled it: *abstain on the bias term whenever any
state component is exactly sampled*, the conservative cut, matching `crn.py`. Implemented as
written, it **cost B01 the selection** — and B01 was one of the two pools the whole fix list was
measured on.

The mechanism. `sde_sabr_cev_call_i` plan 1 (`1-exact-besq-time-change`) draws the forward exactly
— component 0 of 2 — and steps the volatility with the supplied increments. Its ladder:

| functional | `d10` | `d21` | ratio | observed order |
|---|---|---|---|---|
| `price_K1` | 2.372e-03 | 4.863e-04 | 4.88 | 2.29 |
| `price_K2` | 2.574e-03 | 5.862e-04 | 4.39 | 2.13 |
| `price_K3` | 2.576e-03 | 6.878e-04 | 3.75 | 1.90 |

That is not noise. Three functionals, factor-of-four difference ratios, observed orders clustered
near 2: a clean second-order convergence worth 5.7e-04 to 1.5e-03, which the conservative cut threw
away. The plan's charged error halved, its `accuracy` went 0.109 → 0.319, it moved from third to
first in its pool, and the pool shipped 0.1108 where plan 2 sits at 0.0473.

**Why the cut is right for `crn.py` and wrong here.** An order *study* measures the whole state's
pathwise differences, and a single leg that does not consume the increments decorrelates the CRN
levels — the ratio then measures auxiliary randomness and there is no order to read, whichever leg
it was. A *functional* is a number that depends on the whole state: one discretised leg biases it
however exactly the others are drawn. So the two layers need different cuts, and the shipped rule is

> **abstain only where *every* state component is exactly sampled.**

B07's plan 1 declares all three of three, so the ~50× spurious bias [11](11-sde-accuracy-fixes.md)
§2 exists to remove is removed. B01's plan 1 and B10's plan 2 declare one of two and keep their
bias. And one claim in §2 falls with it: B10's plan 2 was **not** *"charged a ladder bias on a leg
that cannot have one"* — the bias enters through the leg that was discretised.

Abstention destroys evidence, so the bar for it is that there is provably no bias to measure, not
that part of the state has none.

---

## §2 Fix by fix

| fix | where | what shipped |
|---|---|---|
| **0** persist the kernel's JSON | `cli.py` (`_persist_evaluation`) | `evaluate` writes `evaluation.json` into the plan directory on every run and prints the path. Best-effort: a read-only directory is a reason to have no record, never a reason to fail an evaluation that already ran |
| **1** ranker inputs on the block | `kernel/metrics.py`, `references/project_manual.md`, `agents/evaluator-sde.md`, `commands/conductor.md` | nine keys added, not eight: the eight §1 lists plus `cost_source` (§1 above). `BLOCK_KEYS` went 38 → 50 across the whole change |
| **1** the structural test | `tests/test_spec_field_coverage.py` | `test_ranking_inputs_are_all_on_the_metrics_block` walks `ranking.py`'s AST for every `summary` key it reads and asserts each is on `BLOCK_KEYS`. One hand-maintained escape hatch, `RANKING_INDIRECT_READS`, for the two keys read through a loop variable — with its own test that an entry has not become direct or dead |
| **2a** the Richardson floor | `kernel/functionals.py` | `estimate_error` takes `design_order` and requires the inferred order to clear `ORDER_FLOOR_FRACTION × design_order` (half) before dividing by `2**p − 1`; below it, the two-level bound `d21 / scale`. `DEFAULT_ORDER_FLOOR = 0.5` where nothing is declared, named as the kernel's in `detail["order_floor_source"]` |
| **2b** exact-sampling abstention | `kernel/functionals.py`, `kernel/sde/driver.py` | `run_declared` takes `exact_sampling` and `state_dimension` and abstains on the bias where the declaration covers every component. The withheld estimate stays in `detail["ladder_rel_err"]` so a reader can disagree |
| **3** one-sided order | `ranking.py`, `project_manual.md`, `conductor.md`, `evaluator-sde.md` | `order` charges `max(0, target − observed)` on both branches. The `order_band` read is retired; the field stays on the block because it says which band the order *check* applied |
| **6** the retry branch | `kernel/sde/driver.py` (`_limited_by`), `conductor.md`, `evaluator-sde.md` | `functional_limited_by: sampling / bias / both / unavailable`, at a 3× dominance threshold, computed once by the kernel rather than derived at each reader. The conductor's F19 retry now branches on it instead of saying "raise `num_paths`" unconditionally |
| **4** the pool obligations | **new** `verifylib/pool.py`, `gate.py`, `kernel/plan_meta.py`, `references/sde_manual.md`, `agents/plan-creator-sde.md`, `conductor.md` | four pool-level checks, run from `gate.check_workspace` and read by the conductor at the end of Phase 1, before any solver burns a cycle. All warnings |
| **5** pool diversity | `verifylib/pool.py` | the same module's fourth check |
| — test maintenance | `tests/test_spec_field_coverage.py` | the four dead `UNREAD` exemptions retired, with their arguments kept as comments |

### Fix 4's four checks, and the signal each one reads

The hard part of Fix 4 was not the obligations. It was finding a signal for each that a plan
*declares* rather than one a reader infers from prose — the whole diagnosis in
[11](11-sde-accuracy-fixes.md) §4 is that the structured field was there and the prose decided.

**A declared `terminal_law` no plan draws from.** The obvious signal — the plan declares
`exact_sampling_components` for that component — is the wrong one, and wrong in the direction that
clears the pool this check exists to report. `exact_sampling_components` tests **non-consumption,
not exactness**: Andersen QE declares it correctly, because it draws its own normals each step and
cannot consume the supplied increments, while being a *moment-matched approximation* to the
non-central χ² transition rather than a draw from it. `crn.EXACT_SAMPLING_FAMILIES` documents that
distinction in as many words. Keyed on the component declaration alone, the check clears B03 — whose
QE plan declares `exact_sampling_components: 0, 1` and whose pool contains no exact draw at all —
and so the shipped rule requires **both** the component and `scheme_family: exact-transition`. The
finding names the QE plan and says why it does not count, rather than leaving a reader to wonder why
a declaration in the pool did not satisfy the check.

Run against the staged workspace, that cut separates exactly the comparison §4 is built on: B03 is
reported, B08 — same problem class, same declared law, same code — is not, because its pool carries
`1-exact-chi2-broadie-kaya-cv`.

**A conditionally tractable component no plan integrates out.** Detected from
`verification.generator`, whose `drift` and `diffusion_matrix` are declared as expressions in
`X[:, i]`. Component *i* qualifies when its own drift and diffusion row are of **degree at most one
in itself** — so with the other paths fixed its equation integrates in closed form — and **no other
component's coefficients read it**, so those paths can be generated first. Together that is exactly
Rao–Blackwellisation's triggering condition.

The degree analysis is syntactic, over `ast`, and returns `None` rather than 0 wherever a call wraps
the variable. That is the false positive that matters: `np.sqrt(np.maximum(X[:, 1], 0.0))` has no
polynomial degree in `X[:, 1]`, and treating a call as degree 0 would claim a CIR variance process
has a conditional closed form — asking a plan for something that does not exist. On the real Heston
generator it returns `(0,)`: the asset leg, whose diffusion is `sqrt(max(V,0))·S`, and not the
variance leg. Sympy would do this better and is not a dependency; a sound syntactic answer beats a
symbolic one that needs one.

**A declared barrier or absorbing boundary no plan treats.** This one needed a declaration that did
not exist, so plan frontmatter gained `boundary_treatment:` — `brownian-bridge | absorption |
reflection | none`. §9.1 decision 4 refused a new *spec* field, and this is not the same thing:
that decision turned on `functional_se_rel` already *measuring* what a `variance_reduction:`
declaration would have claimed, and declared-never-inferred being for facts the kernel must trust
rather than quantities it can compute. A crossing treatment is a fact about the scheme, like
`scheme_family` and `exact_sampling_components` beside it, and nothing measures it. `none` is a
legitimate declaration where the plan has the argument that the payoff is insensitive; silence is
what the gate reports.

**Three plans that are two schemes.** [11](11-sde-accuracy-fixes.md) §4's signature — *identical
`observed_order` and `estimated_rel_error` with differing `functional_se_rel`* — turned out to be
one field too strong, and the measurement is instructive. On B02 and B10 the pair's
`estimated_rel_error` is bit-identical (1.600e-15, 8.473e-16); on B06 it is 3.650e-06 against
3.648e-06 and on B07 it differs by 5%. That field is a moment error computed from the terminal
sample, and an antithetic pairing **does** change which paths are in that sample. `observed_order`
does not, because the CRN ladder takes *pathwise* differences and a wrapper does not touch the
dynamics those differences are of. So the shipped signature is **identical `observed_order` within
one declared `scheme_family`**, with the errors reported in the finding rather than required to
match. It fires on all four pools §4 lists and on no others.

---

## §3 The `<metrics>` block had drifted further than §9.2 knew

§9.2 said adding keys touches three homes besides `BLOCK_KEYS`, because
`verification_manual.md` §2 specifies the block *once*, in `project_manual.md`, and *"a schema with
two homes is a schema with none."*

Measured while doing it: **the manual carried 24 of the then-38 keys.** Missing were `graded_N`,
`estimated_rel_error_graded`, `error_horizon`, `estimated_rel_error_T`, `mean_rel_err`,
`variance_rel_err`, `invariant_worst_headroom`, `invariants_not_reported`, `analytic_numerical`,
`error_fs`, `dynkin_bar`, `ci_mult`, `solver_sha256` and `cost_source`. So the single-home rule had
already failed, silently, and saying it in prose was what it had instead of a test.

The manual's block now carries all 50 keys in `BLOCK_KEYS` order, and
`test_the_manual_and_block_keys_name_the_same_fields_in_the_same_order` holds them together. Order
too: the manual is what a reader learns the block from, and a field in a different place reads as a
different field.

That is the same shape as Fix 1 one layer out. A rule about one list needs a test, like any other.

### And a fourth home, which was never broken

`benchmark/baseline_select.py` also calls `ranking.rank_key`, and its own docstring says *"the
conductor's key and this one are the same"*. The key is — the **input** was not. It passes the
kernel's full `summary` dict straight from the JSON, so it has been ranking C0 baseline attempts
with a working `accuracy` component the whole time the pipeline was ranking pool plans without one.

This did not contaminate any published comparison: the arm tables in [10](10-sonnet-retest.md) and
[11](11-sde-accuracy-fixes.md) are computed by `pool_min` on the harness error — the genuinely best
draw, not the ranked pick — so nothing in them went through either selector. It is recorded because
it is the same defect in a third place, and because it is the reason this was invisible from inside
`ranking.py`: the module is correct, exercised, and tested, and one of its two callers was handing it
a dict the other caller could not produce.

`solution_metrics` in `benchmark/run.py`, the projection the benchmark report reads, now carries the
six functional fields too. It quoted `estimated_rel_error` alone, which on a priced problem is a
state-moment proxy and not the answer's error.

---

## §4 §9.3's risk, re-checked against the live corpus

§9.3 checked that adding keys to `BLOCK_KEYS` would not fire
`test_review::test_no_false_positives_on_archived_reviews`, which counts transcription-drift
findings over every archived `SOLUTION.md` and would reclassify any review citing a newly-carried
key from "gap in the block" to "drift".

Re-checked over all **134** archived plans: zero keyed citations of any of the nine new fields
inside a bound `<review>` section. The test is still green, and it stayed green after
`evaluator-sde.md`'s template started asking for `functional_rel_error` and `review.BOUND_KEYS`
started binding it.

---

## §5 What is deliberately not done

- **`resolved` was not redefined.** §9.2 asked what it tracks now that accuracy is the functional's.
  It stays a moment-space claim — did the variance estimate separate from its reference by more than
  the sampling noise — because `score.rubric_sde` reads it and that is a precise, score-bearing
  statement about a different quantity. What changed is the *retry gate*: it now branches on
  `functional_limited_by`, which is the graded quantity's own verdict. Both are on the block and
  `project_manual.md` says which reads which. Neither replaces the other.
- **`cross_check_ok` is reported, not scored.** `verification.expected_cross_check` was the third
  orphaned spec field, and unlike the two boundary facts it is directly checkable: the declaration
  *"price_S75 < price_S100 < price_S125"* is an ordering, and an ordering is a stronger claim about a
  multi-functional solver than any one tolerance — three prices from one set of paths can each sit
  inside their own gate while being out of order, which is a mis-wired strike and which no
  per-functional error can see. `functionals.cross_check` reads the relations and never the prose,
  and the outcome is recorded like `reported_moments_agree`: a diagnostic whose absence says nothing
  and whose failure is specific enough to act on. Wiring it into the score is a separate decision.
- **The accuracy-first rank key was not revisited** beyond what §6 records. §9.1 decision 1 deferred
  it pending a re-measurement of B07 after Fix 2b.
- **No pipeline run.** Fixes 4 and 5 change what gets *generated* and cannot be evaluated offline.
  Everything else is measured below.

---

## §6 Measured

Two measurements, because the fixes split cleanly into ones that re-rank artifacts already on disk
and ones that need the kernel re-run.

### §6.1 Fix 3 alone, on the archived blocks — all ten pools

Fix 3 needs no re-evaluation: it changes how `ranking.order` reads a number the archived
`<metrics>` blocks already carry. So it can be measured over the whole suite, by ranking each pool's
archived blocks twice — once with `order` as it stood at `341d7a7`, once as it stands now — and
reading the shipped plan's independently-graded harness error.

| pool | before ships | err | after ships | err |
|---|---|---|---|---|
| B06 `sde_hhw_call_corr` | `2-qe-variance-gaussian-rate-cv` | 0.7445 | **`1-euler-full-truncation-antithetic`** | **0.0474** |
| the other nine | unchanged | | unchanged | |
| **shipped geomean** | | **0.1693** | | **0.1285** |

**1.32× on the ten-pool shipped geomean, from one pool flipping.** The 0.1693 reproduces
[10](10-sonnet-retest.md) §8's baseline exactly, so the two numbers are on the same footing.

And it corrects an attribution. [11](11-sde-accuracy-fixes.md) §1 credits B06's recovery to Fix 1,
on the strength of the functional error ordering that pool perfectly. It does — but Fix 3 gets there
first and without re-running anything, because once over-performance stops being charged, B06's
three Euler schemes all score `order: 1.000`, the pool separates on `structure` instead, and plan 1
wins. Both fixes reach the pool best; only one of them is free.

### §6.2 The whole change set, on the five re-evaluated pools

`cli.py evaluate --json` re-run on all fifteen plans of the five pools
[11](11-sde-accuracy-fixes.md) §0 names, then ranked through the production seam —
`render_block` → `parse_metrics` → `rank_key` — so the ranker sees exactly what it would see in a
live run and nothing more.

| pool | before ships | err | after ships | err | pool best | noise bar |
|---|---|---|---|---|---|---|
| B06 | `2-qe-variance-gaussian-rate-cv` | 0.7445 | **`1-euler-full-truncation-antithetic`** | **0.0474** | 0.0474 ✓ | 0.124 |
| B01 | `1-exact-besq-time-change` | 0.1108 | **`2-euler-maruyama-absorbing-antithetic`** | **0.0473** | 0.0473 ✓ | 0.030 |
| B10 | `3-em-antithetic-martingale-cv` | 0.1783 | unchanged | 0.1783 | 0.1612 | 0.032 |
| B07 | `1-exact-chi2-gaussian-rate` | 0.0780 | unchanged | 0.0780 | 0.0388 | 0.189 |
| B03 | `1-euler-maruyama-full-truncation` | 0.4084 | unchanged | 0.4084 | 0.4084 ✓ | 0.154 |
| **shipped geomean** | | **0.2159** | | **0.1049** | 0.0894 | |

**2.06× on these five pools, and three of the five now ship the pool best.** The remaining gap to a
cherry-picking oracle is **1.17×**, which is what selection has left to win here; the rest is
generation, which is Fix 4 and 5 and cannot be measured without a run.

Reading the three unchanged rows honestly:

- **B03 was never a selection miss.** Its pool best *is* what it ships — 0.4084 against 0.4217 and
  0.5327. Selection is already optimal and the pool's ceiling is 23× behind a single unaided draw.
  That is [11](11-sde-accuracy-fixes.md) §4 confirmed from the other side: nothing in the rank key
  can reach it.
- **B10's 1.11× gap is inside its own noise bar**, as §2 said when it retired the figure. Fix 2a did
  do its job there: plan 3's charged error went from 0.0270 — last in the pool — to 8.769e-03, first
  in the pool, which is the 16× denominator inflation removed. The shipped plan does not change
  because plan 3 was already shipping.
- **B07's 2.0× gap is architectural**, and §6.3 is about why.

### §6.3 §9.1 decision 1, re-opened and now resolved

[11](11-sde-accuracy-fixes.md) §9.1 decision 1 left the score-versus-accuracy ordering **deferred,
not affirmed**, on the explicit condition that it be re-opened once Fix 2b made B07's accuracy
component work: *"if the conflict survives a working component, the design question is real."*

Fix 2b did its job on B07. Plan 1 declares all three of three components exactly sampled, and its
charged functional error went from **0.0130 to 1.593e-03** — the ~50× spurious bias removed — so its
`accuracy` went from **0.000 to 0.137**. The component is working on exactly the plan in question.

The conflict does not survive, and it does not survive in the direction of accuracy-first:

| pool | scores | score-first ships | accuracy-first ships | sqi-first ships | pool best |
|---|---|---|---|---|---|
| B06 | 3, 3, 3 | 0.0474 | 0.0474 | 0.0474 | 0.0474 |
| B01 | 9, 9, 8 | **0.0473** | 0.0592 | 0.0592 | 0.0473 |
| B10 | 2, 2, 2 | 0.1783 | 0.1783 | 0.1783 | 0.1612 |
| B07 | 6, 3, 3 | **0.0780** | 0.1760 | 0.1760 | 0.0388 |
| B03 | 8, 8, 3 | **0.4084** | 0.4217 | 0.4084 | 0.4084 |
| **geomean** | | **0.1049** | **0.1300** | **0.1292** | 0.0894 |

**Accuracy-first is 1.24× worse** — on a working component, against 1.30× on the broken one. Putting
the quality index above score is 1.23× worse. Score keeps its primacy, and this time **affirmed on
measurement rather than deferred.**

The reason is worth stating, because it is not "score is a better signal than accuracy". On B07
the accuracy component ranks the genuinely best plan (plan 3, true error 0.0388) **second** — 0.186
against plan 2's 0.199, while plan 2's true error is 0.1760. The three plans' functional errors are
1.201e-03, 1.271e-03 and 1.593e-03 against a 0.003 tolerance: a 1.3× spread on an estimate whose
own sampling term is most of it, used to separate plans whose true errors span 4.5×. **The component
is working and its resolving power is finer than its accuracy** — which is the caveat
[11](11-sde-accuracy-fixes.md) §1 raised about the functional error and which is now the binding
constraint rather than a footnote. Promoting a noisy signal above a coarse but correct one does not
help, and B07's own noise bar (0.189) is wider than every error in its pool.

So B07 stays open as a *generation* problem, not a selection one. Its pool best, 0.0388, is held by
`3-em-control-variate-antithetic` — one of the duplicate pairs Fix 5 reports.

### §6.4 Test state

Measured on a clean full run after the change set:

```
uv run pytest verifylib/tests benchmark/tests
```

- **1124 passed, 3 skipped, 15 deselected, 0 failed** (the 15 deselected are
  `pyproject.toml`'s `-m 'not slow'`, as intended). The baseline §9.2 recorded was *1070 passed with
  three known failures*.
- One failure was introduced on the way and is worth recording, because the test that caught it was
  built for exactly this. `test_kernel_core::test_an_undeclared_plan_falls_back_rather_than_guessing`
  compares `plan_meta.read`'s whole dict, and its own comment says why: *"the whole-dict comparison
  is kept rather than loosened so a new field has to be acknowledged here."* Adding
  `boundary_treatment` turned it red, and the fix was to acknowledge it. A guard that makes you
  notice is not the same as a guard that makes you work.
- The three failures [11](11-sde-accuracy-fixes.md) §9.2 recorded as the baseline are gone: both
  `test_spec_field_coverage` failures are fixed by the change set itself — the three orphaned spec
  fields now have consumers (`barrier` and `absorbing_boundary` via `pool.check_boundary_facts`,
  `expected_cross_check` via `functionals.cross_check`) and the four dead `UNREAD` exemptions are
  retired.
- `test_work_precision::test_an_sde_sweep_moves_both_axes_...`, the load-dependent flake §9.2
  diagnosed, is **made load-independent** — and the reasoning rather than a red-to-green transition
  is what justifies it, because it would not reproduce here. It passed in every run taken after the
  change set, including one under fifteen concurrent kernel evaluations.

  The argument stands without the repro. The test's real claim is the *level ordering*, asserted
  exactly a few lines above. The cost axis was asserted between **adjacent** levels, which at these
  sizes differ by less than the wall-clock jitter on a sub-100 ms solve — §9.2 recorded it failing on
  `[0.0136, 0.1079, 0.0457, 0.1439]`, a 2.4× drop that is CPU contention and not work — and a
  tolerance wide enough to absorb that is wide enough to absorb a genuine inversion. The four levels
  span 1e5 to 2e7 path-steps, so the check is now end-to-end over that 200× span, which no scheduler
  can invert. Narrower in what it asserts, and sound where it was not.
- 24 new tests: 22 in `test_pool.py` and the rest across `test_kernel_functionals.py`,
  `test_spec_field_coverage.py`, `test_benchop_fixes.py`, `test_ranking.py` and
  `test_kernel_end_to_end.py`. Three existing `test_ranking.py` tests were **rewritten**, because
  they froze the two-sided `order` rule Fix 3 reverses; each now records the measurement that
  reversed it rather than just the new expectation.
- `ruff check` adds no new findings over the pre-change tree: 12 findings, every one of them in a
  file this change set did not create, and the three in `test_spec_field_coverage.py` are the same
  three it carried at `341d7a7`.

**This is the state before §7's pipeline run.** That run changed `workspace/`, which three of these
tests read, and broke all three — see §7.5. The final state after fixing them is the same count:
1124 passed, 3 skipped, 0 failed.

---

## §7 B03 re-run through the pipeline at Sonnet — the first live test

Everything in §6 is an offline re-ranking of solvers already on disk, which can test Fixes 0–3 and 6
and cannot test Fix 4 or Fix 5 at all: those change what gets *generated*. So B03
(`sde_heston_call_feller`) — the pool [11](11-sde-accuracy-fixes.md) §4 is built on, and the one
whose ceiling was 23× behind a single unaided draw — was re-run end to end:

```bash
uv run python benchmark/run.py --suite benchop --only B03 --model sonnet --fresh --timeout 7200 \
  --results-json benchmark/results/pipeline/benchop_b03_sonnet_run4.json
```

19.4 minutes of pipeline, 22.9 wall. `--fresh`, so the formulator and both plan-creators ran from
`problem.md`; the prior workspace is kept at the path named in §7.4.

### §7.1 Fix 4 worked, and it worked at generation rather than at the gate

The pool `--fresh` produced:

| plan | `scheme_family` | `exact_sampling_components` |
|---|---|---|
| `1-exact-v-conditional-mc` | **`exact-transition`** | 0, 1 |
| `2-andersen-qe` | `quadratic-exponential` | 0, 1 |
| `3-full-truncation-euler` | `euler-maruyama` | — |

Plan 1's own one-sentence summary, from STATE.md: *"Exact non-central chi-squared variance transition
per step, with S integrated out analytically (Black-Scholes averaged over variance paths)."*

That is Fix 4's obligation 1 **and** obligation 3 in one plan — the exact draw from the declared
`terminal_law`, and conditional Monte Carlo on the asset leg. Conditional MC appeared in **zero of
the thirty** plans on the original suite, and in C0's best B03 draw, which is how
[11](11-sde-accuracy-fixes.md) §4 identified it. The old pool was Euler full-truncation, Andersen QE
and Euler antithetic: no exact draw, no conditioning.

And the mechanism matters: STATE.md records

```
pool_findings: none (gate emitted no pool warnings)
```

So the conductor ran the Phase 1 pool check and it came back **clean** — the plan-creator had
already proposed the technique, from `sde_manual.md`'s strengthened rule 0 and new rule 0b. The
gate's corrective path (re-dispatch for a replacement plan) was never needed and is therefore still
untested. `pool.check_problem` on the new pool returns zero findings, against one on the old.

**The conditioning's benefit shows up exactly where theory says it should.** Plan 1's harness noise
bar is **0.0150** against plan 3's **0.2973** — 20× tighter. The bar is driven by the sampling
standard error between the graded and coarse path counts, and integrating the asset leg out removes
that leg's variance entirely rather than reducing it. This is the one number in this document that
Rao–Blackwellisation predicts in advance.

### §7.2 The result, raw and noise-floored

| | before (run3) | after (run4) | |
|---|---|---|---|
| shipped, raw | 0.4084 | **0.0391** | **10.4×** |
| shipped, noise-floored | 0.4084 | 0.2973 | 1.37× |
| pool best, raw | 0.4084 | **0.0391** | **10.4×** |
| pool best, noise-floored | 0.4084 | **0.0779** | **5.24×** |

Both framings belong here and the gap between them is the story. Raw, the shipped plan improved
**10.4×**. Noise-floored — a plan cannot claim an error below its own error bar — it improved
**1.37×**, because the plan that ships has a bar of 0.2973 and its 0.0391 is therefore unresolved.
The *pool* improved 5.24× floored, because plan 1's 0.0779 sits a factor of 5 inside its own bar and
is the only error in either pool that is genuinely resolved.

Against the arms, on B03, both ways:

| arm | best raw | best noise-floored |
|---|---|---|
| C0@Opus, 10 draws | 0.0178 | 0.0392 |
| C0@Sonnet, 3 draws | 0.0579 | 0.3765 |
| AN pre-change | 0.4478 | — |
| AN run3 (Sonnet) | 0.4084 | 0.4084 |
| **AN run4 (Sonnet)** | **0.0391** | **0.0779** |

**Model-matched, AutoNumerics now beats a naked Sonnet draw on B03 on both framings** — 1.48× raw
and 4.83× noise-floored — where it was 7.1× behind raw before. Against ten Opus draws it went from
23× behind to **2.2× raw / 2.0× floored**. B03 was the worst pool in the suite and the one §4 said
only Fix 4 could reach; it is now the pool where the pipeline's margin over its own model is largest.

### §7.3 A new selection question, and it qualifies §6.3

§6.3 resolved §9.1 decision 1 against accuracy-first on five pools, by raw error. **This pool splits
the two keys by metric, and that is worth stating plainly rather than leaving §6.3 to stand
unqualified:**

| key | ships | raw | noise-floored |
|---|---|---|---|
| score / sqi-first (shipped) | `3-full-truncation-euler` | **0.0391** ← pool best raw | 0.2973 |
| accuracy-first | `1-exact-v-conditional-mc` | 0.0779 | **0.0779** ← pool best floored |

Each key is optimal under one metric and loses under the other. `accuracy` reads 0.451 for plan 1
against 0.048 for plan 3 — it strongly prefers the conditioned plan, and under the floored metric it
is right, by 3.8×.

So §6.3's conclusion needs its scope stated: **accuracy-first is 1.24× worse by raw error, and this
pool is evidence that the ordering may reverse under the noise-floored one.** The two metrics
disagree here because the quantity accuracy is reading — a functional error whose bias term is
abstained and whose remainder is a sampling term — is closer to *what the harness can resolve* than
to the harness's point estimate. That is not a reason to reorder the key on one pool; it is a reason
to settle which metric selection is being optimised for, which neither
[11](11-sde-accuracy-fixes.md) nor this document has done. Carried as an open item.

There is a second-order observation underneath it. Plans 1 and 2 both declare all components exactly
sampled, so Fix 2b abstains on their bias and their charged error is their sampling term alone —
2.503e-04 and 7.212e-03. That inflates `accuracy` for a fully-exact plan, because the abstained bias
is replaced by *nothing* rather than by a bound. It is the mirror image of the B01 defect in §1: one
end of Fix 2b over-charged a partially-exact plan, and this end under-charges a fully-exact one. The
abstention is still correct in principle — there is no dt bias to charge — but "no bias term" and
"bias term of zero" are not the same claim, and `accuracy` currently reads them identically.

### §7.4 What this run does not establish

- **n = 1.** One Sonnet run of one problem. [11](11-sde-accuracy-fixes.md) §0 records B03's previous
  kernel numbers as *not reproducing*, and §8 item 1 is still open, so this problem in particular has
  a history of run-to-run movement. The 10.4× raw figure is one draw and the 1.37× floored figure is
  the conservative reading of the same draw.
- **The other nine pools are unmeasured since the change.** Fixes 0–3 and 6 were measured offline on
  five; Fixes 4 and 5 have now been measured on one.
- **The gate's corrective path is untested.** Generation complied, so the conductor never had to
  re-dispatch on a pool warning. The check that fires and the instruction that follows it are
  exercised only by `test_pool.py`.
- **Fix 5 was not exercised.** The new pool has three distinct `scheme_family` values, so no
  duplicate pair arose to report.
- **All nine new `<metrics>` fields reached a live run for the first time** and are present in all
  three blocks: `kind`, `order_band`, `cost_source`, `cross_check_ok`, `functional_name`,
  `functional_rel_tol`, `functional_rel_error`, `functional_bias_rel`/`functional_se_rel` and
  `functional_limited_by`. `cross_check_ok` reads `unavailable` because B03's spec declares no
  `expected_cross_check`; it is exercised on `sde_hhw_call_zerocorr`, where all three plans' declared
  price orderings hold.

### §7.5 The run broke three tests, and all three were right to break

Worth recording, because one of them was mine and it failed in the way this folder keeps warning
about.

**`test_pool::test_the_two_heston_pools_come_out_differently`** asserted the comparison §4 is built
on against the live workspace: `sde_heston_call_feller` reported for leaving its declared transition
unused, `sde_heston_call_hivar` not. It went red **within the hour of being written**, because the
re-run gave the first problem a pool that *does* contain the exact draw. The fix working made the
test fail, and the test called it a regression.

That is [11](11-sde-accuracy-fixes.md) §8 item 8 happening to a test instead of a document, so the
rewrite is the lesson rather than a patch: the assertion is now the *property*, read off whatever is
on disk — a declared `terminal_law` is reported for a component exactly when no plan declares both
that component and `scheme_family: exact-transition`. True of any corpus. The discrimination itself
is covered by fixtures, which is where a claim about specific pools belonged.

**`test_spec_field_coverage::test_the_unread_table_names_only_fields_some_live_spec_declares`** went
red on `evaluate_at`, `functional_names` and `Ny` — the 2-D Kolmogorov block's fields. Those three
exemptions existed because the Nov-01 formulator proposed a route `surrogates.kolmogorov` refuses
outright; the Nov-05 formulator proposed none, so they aged out. The `UNREAD` note had predicted
exactly this (*"this exemption churns with every re-run while the underlying waste stays put"*) and
had been tracking whichever spelling the workspace carried.

Retired, with the datum that makes them not worth tracking: across three runs of one problem the
formulator proposed the unreachable route **twice and skipped it once**, under two different field
spellings. So it is not a stable property of the problem, there is nothing to exempt, and the fix is
still a signal to the formulator that the scalar route refuses any spec with
`state_dimension != 1` — not a table chasing field names.

**`test_benchop_fixes::test_solve_time_cannot_change_the_ranking_of_a_benchop_pool[sde_heston_call_feller]`**
is the informative one, and it is not corpus drift at all. Its second half asserted that *within a
band tied on score and provenance, the most accurate plan must win*. Its own docstring explains why
that had never fired:

> measured 2026-11-01, the accuracy component is flat across all four BENCHOP pools (`None` on
> `sde_heston_dnt`, `0.0` on the other three), so any assertion phrased in terms of an accuracy
> *gap* skips everywhere and guards nothing

and then attributes the flatness to the kernel's diagnostic path count making the graded quantity
unresolvable. **That attribution was wrong, and this is the fourth independent place the whitelist
defect shows up in the record.** The flatness was `BLOCK_KEYS` not carrying
`functional_rel_error`/`functional_rel_tol`; with them carried, the component reads 0.451 / 0.159 /
0.048 across this pool's three plans.

So the assertion fired for the first time, and failed: the winner's accuracy is 0.048 against a band
maximum of 0.451. It is accuracy-first restricted to a band, and §6.3 measured accuracy-first as
1.24× worse by raw error — on this very pool it would ship 0.0779 where the current key ships
0.0391. But §7.3 is the reason not to simply invert it either: by noise-floored error the band
maximum *is* the better plan.

**The ordering claim is withdrawn rather than inverted**, and replaced with what the change set does
establish and what was false on all four pools when that docstring was written: on a pool graded by
declared functionals, accuracy must be **measurable**, and must not be saturated at 0.0 or 1.0 across
the pool. The endpoints rather than distinctness, because two plans can legitimately tie at an
interior value and four of ten pools here hold a near-identical pair. It is live on one pool today —
the only one whose blocks post-date the fix — and arms itself as pools are re-run.

A test that cannot fire is not a passing test, and three of this change set's guards found real
things the moment a component started working: the ranker-vs-block walk found `cost_source`, the
whole-dict `plan_meta` comparison found `boundary_treatment`, and this one found the question in
§7.3.

The pre-change workspace is preserved at
`/private/tmp/claude-502/-Users-sahilmulki-Autonumerics/acf236b5-504c-4b87-955f-a814f873ce5c/scratchpad/b03_before/sde_heston_call_feller`,
since `--fresh` deletes a gitignored directory and three documents in this folder have already been
invalidated by exactly that.
