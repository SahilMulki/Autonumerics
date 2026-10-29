# Improving AutoNumerics — what BENCHOP-SLV measured, and what to change

**Status:** **proposal, written 2026-10-28, revised the same day against
[07-review.md](07-review.md), and revised again 2026-10-29 against the free measurements in
[08-free-measurements.md](08-free-measurements.md).** Nothing here is implemented.

> **[M, 2026-10-29]** The offline measurements [06](06-retest-protocol.md) §1 said to run before
> implementing anything have been run, and **one of them disconfirmed the plan's central
> assumption**. `accuracy` — 0.35 of the quality index — reads **1.000 on all eleven** B01–B04
> plans, because it snaps the error up to twice its own Monte-Carlo noise and then uses the same
> half-width as the ceiling. `order` (0.25) returns `None` on **all 88** metrics blocks in
> `workspace/`. `consistency` (0.10) is `None` throughout. The live index is **⅔ structure + ⅓
> cost**, and in **four pools of four** the most accurate plan is the slowest and the index ranks
> it last. So **[F0](03-fix-selection.md#f0)/[F2](03-fix-selection.md#f2) are necessary and not
> sufficient**: wiring the price into a saturated component changes nothing. Read
> [08](08-free-measurements.md) §2 before [03](03-fix-selection.md). The evidence is complete: all ten
BENCHOP-SLV problems have been run through the pipeline (C2) and through the Opus one-shot
baseline (C0, 100 attempts), and every number quoted is read from the graded records now in
`benchmark/results/benchop-step10/`.

**The review changed the plan materially, and the corrections are marked inline throughout.** In
summary: the quantitative case was inflated by a statistic that selects on Monte-Carlo luck; F1 and
F3 as first written could not work; a prerequisite ([F12](03-fix-selection.md#f12)) was missing from
the fix list; F9 was withdrawn; F6 lost its measured evidence; and most of the "free" offline work
in [06](06-retest-protocol.md) §1 needs artefacts that are not on disk. One review finding is
**rejected** — see [01](01-what-was-measured.md) §1a. **Read
[06-retest-protocol.md](06-retest-protocol.md) §8 before changing any code**, and
[07-review.md](07-review.md) for the reasoning behind each correction.

## The finding, in one paragraph

On the BENCHOP-SLV suite the pipeline is **less accurate than a single Opus completion with no
tools**, and almost none of that gap is the pipeline failing to *produce* a good solver. The
pipeline's own plan pools already contain a plan that ties or beats C0's best attempt on **six of
ten** problems; it ships that plan on **none** of them. Shipped answers are a geometric mean
**8.4×** further from the reference than C0's best of ten attempts, and **2.6×** further than
C0's *median single attempt* — the comparison that matters, since both arms get one draw. The
pipeline's own best plan is only 3.9× off C0's best, so **selection costs something on top of
everything else: between 1.3× and 2.2×**, depending on whether each plan's error is floored at its
own Monte-Carlo error bar. ~~2.2×, and 104× on B10.~~ **[C]** B10's apparent 104× is a real
selection failure — its two plans differ on the graded price at **3.4–4.5σ** — but the honest size
of it is **≈7×**; the rest was a lucky draw. [01](01-what-was-measured.md) §3 has both figures and
the reason neither alone is the answer. The pipeline is a competent solver generator bolted to a
ranker that cannot see the quantity it is being graded on — and on the SDE path that quantity never
even crosses the sandbox boundary.

Two things follow, and they set the order of work:

1. **Fix ranking before anything else** — but fix the *measurement* first. It is the cheapest
   change and it needs no model upgrade; on this evidence it converts a 7–0–3 loss into a 4–0–6
   near-draw. ~~How much of that is real rather than Monte-Carlo luck is not yet known.~~
   **[M]** It is now measured: [08](08-free-measurements.md) §3 re-solved all eleven B01–B04 plans
   at two further seeds, and the selection loss on three-seed means is **1.69×** — above
   [07](07-review.md)'s noise-floored 1.33× and below the raw 2.18×. It is present at every seed
   (2.47× / 3.79× / 1.89×) and the best plan is seed-stable in two of the four pools.
2. **Do not read this as "Opus is a better SDE solver than the pipeline."** C0 won by *picking
   better schemes* — exact non-central-χ² / Bessel transitions or Andersen QE on nine of ten
   problems — and the pipeline's planner is told in writing to propose Euler–Maruyama only on
   every multi-dimensional problem, which is all ten. The pipeline improvised past that rule in
   nine of ten pools and the improvisations were usually the better plan. It then ranked them last.

## The fix list, by leverage

Effort is a rough engineering estimate; "evidence" names the measurement in
[01-what-was-measured.md](01-what-was-measured.md) that justifies the fix.

| # | Fix | File | Effort | Evidence | Expected effect |
|---|---|---|---|---|---|
| **F0** | Carry `functionals` across the **SDE** sandbox boundary — `_dump_sde` never captures it | [`verifylib/kernel/sandbox.py:305`](../../verifylib/kernel/sandbox.py#L305), [`:521`](../../verifylib/kernel/sandbox.py#L521) | **S** | the harness fixed the same bug on its own boundary | **Blocks everything else.** The price the solver computes is discarded before the kernel sees it |
| **F12** | Return a per-functional standard error from `solve_sde`, and declare functionals in the generic contract | [`agents/solver-sde.md`](../../agents/solver-sde.md), [`references/sde_manual.md`](../../references/sde_manual.md) | **M** | `grep -c functional` = 0 in all five manuals; `mc_se_rel` is a *moment* SE | **Prerequisite for F3.** Without it F3 has no error bar and F1 no interval |
| **F3** | Make the ranker's accuracy component read the functional where one is declared | [`verifylib/ranking.py:99`](../../verifylib/ranking.py#L99), [`:463`](../../verifylib/ranking.py#L463) | **M** | moment error orders plans right in 8/15 within-pool pairs (95% CI 0.30–0.75) | The core selection fix |
| **F16** | **Unsaturate `accuracy`** — the 2×-noise snap declares a tie *at the component's ceiling*, so accuracy never orders anything | [`verifylib/ranking.py:155`](../../verifylib/ranking.py#L155), [`:169`](../../verifylib/ranking.py#L169) | **M** | **`accuracy` = 1.000 on 11/11 plans** ([08](08-free-measurements.md) §2) | **Now the core selection fix.** Without it F0/F2/F3 change nothing |
| **F17** | **Give the SDE path a `design_order`, or drop `order`'s weight honestly** | [`verifylib/kernel/sde/driver.py`](../../verifylib/kernel/sde/driver.py), [`verifylib/ranking.py:268`](../../verifylib/ranking.py#L268) | **M** | **`ranking.order()` returns `None` on all 88 blocks on disk** ([08](08-free-measurements.md) §2) | A quarter of the index starts existing — or stops being advertised |
| **F18** | **Exempt an exactly-*sampled* step from the CRN order check**, as exact *integration* already is | [`verifylib/kernel/sde/crn.py`](../../verifylib/kernel/sde/crn.py) (`abstention`) | **M** | B01's BESQ(0) plan: order −0.0251, hard-failed; B02's lognormal leg: roundoff, free ([08](08-free-measurements.md) §4) | Stops docking every exact-transition scheme a point |
| **F19** | **`resolved: false` on the graded quantity must block `state: stopped`** | [`commands/conductor.md`](../../commands/conductor.md) | **S** | B02: `mc_se_rel` 0.18–0.28, all three plans `resolved: false`, all plateaued at iter 2 ([08](08-free-measurements.md) §5) | The pipeline stops shipping answers it recorded as unmeasured |
| **F13** | ~~Refuse an error estimate below the plan's own sampling noise~~ — **already implemented in `ranking`**; re-scoped to the reporting layer | [`verifylib/ranking.py:155`](../../verifylib/ranking.py#L155) | **S** | the snap exists and **is** the saturation; B02's 25–38× gap is still unreported | Flag it where a human reads it, not where it silently ties |
| **F4** | Stop cost deciding ties — **on both paths**, the `sqi` component *and* `rank_key`'s 5th lexicographic key | [`verifylib/ranking.py:369`](../../verifylib/ranking.py#L369), [`:437`](../../verifylib/ranking.py#L437) | **M** | **[rev]** cost is not a tiebreaker: with three of five components dead it is **⅓ of the live index**, and on B04 it is the *whole* decision ([08](08-free-measurements.md) §2) | Promoted. Four of four pools rank the most accurate plan last, on time alone |
| **F1** | Compare the functionals across plans **at the sampling error, not at the gate** | [`verifylib/cli.py:360`](../../verifylib/cli.py#L360) | **M** | B10's plans agree on both moments to 14 digits and differ on price at 3.4–4.5σ | A disagreement detector below the gate. **It cannot order a pool** |
| **F2** | Run the functional self-convergence check on the SDE path | [`verifylib/kernel/sde/driver.py`](../../verifylib/kernel/sde/driver.py) | **S** | `functionals.py` exists and is PDE-only | A gated functional that has not converged stops certifying |
| **F5** | Widen the SDE scheme menu past Euler–Maruyama | [`agents/plan-creator-sde.md:21`](../../agents/plan-creator-sde.md#L21), [`references/sde_manual.md:166`](../../references/sde_manual.md#L166) | **M** | C0 chose exact/QE on 9/10 | Puts the winning schemes on the menu. **Needed before the Sonnet switch** |
| **F6** | Give `scheme_family` real values for QE and exact transitions | [`agents/plan-creator-sde.md:62`](../../agents/plan-creator-sde.md#L62), [`verifylib/kernel/sde/crn.py:221`](../../verifylib/kernel/sde/crn.py#L221) | **M** | code reading only — **this suite does not measure its cost** | Stops judging an exact scheme against Euler's 0.5 floor |
| **F8** | Register the solver module in `sys.modules` — and **leave it registered** | [`benchmark/runner.py:157`](../../benchmark/runner.py#L157) | **S** | `PicklingError` on `2-qe-log-euler` | Forking solvers stop crashing. Blocked on the process-model decision in F4 |
| **F10** | Report the per-plan harness error, `decided_by` and the margin | [`benchmark/report.py:552`](../../benchmark/report.py#L552) | **S** | reads `rel_l2_err`, which does not exist on a functional-graded problem | Makes the next run's failure analysis free |
| ~~**F11**~~ | ~~Investigate B01~~ — **answered** ([08](08-free-measurements.md) §4): the CRN order study reads −0.0251 on an exact BESQ(0) step and `score.rubric_sde` docks it. Superseded by **F18** | — | — | the plan predicted its own penalty in SOLUTION.md before implementation | Closed |
| **F7** | ~~Break the plan-ordering bias~~ → **report the tie** rather than break it invisibly | [`commands/conductor.md:123`](../../commands/conductor.md#L123) | **S** | ~~7/10 winners were plan 1~~ position, family and cost are collinear here | An audit trail, not a fix for a measured bias |
| **F14** | Decide what to do about the evaluator, which no other fix touches | [`agents/evaluator-sde.md`](../../agents/evaluator-sde.md) | **?** | no plan's score improved on cycle 2, anywhere | Currently a hoped-for side effect of F1–F3 |
| ~~**F15**~~ | ~~Diagnose B02~~ — **answered** ([08](08-free-measurements.md) §5): it **fails the external gate at 1 of 3 seeds**, every plan recorded `resolved: false`, and nothing acts on it. Superseded by **F19** | — | — | shipped plan 1.1380 at seed 202; `mc_se_rel` 25–38× the reported estimate | Closed |
| ~~**F9**~~ | ~~Cut `max_iter` 5 → 2~~ — **withdrawn**, it saves nothing | — | — | every plan already stopped at `iter: 2` | **Zero.** The plan-count increase is a cost *rise*, proposed separately |

Detail, rationale and acceptance tests:

**Baseline test state, measured 2026-10-28:** `uv run pytest verifylib/tests benchmark/tests` gives
**1030 passed, 4 failed**. The suite is not red from a regression —
[06](06-retest-protocol.md) §7a characterises all four, and one of them,
`test_spec_field_coverage::test_no_sde_spec_field_is_declared_and_unread_without_a_reason`, **is
M1 already named by the test suite**: the nine BENCHOP functionals are declared in live specs and
read by nothing in `verifylib/`. It is F0/F2's free acceptance test.

| Document | Covers |
|---|---|
| [01-what-was-measured.md](01-what-was-measured.md) | The complete graded record — both arms, every plan, every number this folder cites, and where each came from |
| [02-root-causes.md](02-root-causes.md) | The six mechanisms, each traced to the line of code that implements it and the measurement that proves it fires |
| [03-fix-selection.md](03-fix-selection.md) | F1–F4 and F7: the ranking and selection fixes. The highest-leverage work |
| [04-fix-planning.md](04-fix-planning.md) | F5–F6: the scheme menu and the `scheme_family` vocabulary |
| [05-fix-loop-and-harness.md](05-fix-loop-and-harness.md) | F8–F11: the refine loop's iteration budget and the four harness defects the runs exposed |
| [06-retest-protocol.md](06-retest-protocol.md) | The all-Sonnet retest: what to hold fixed, what counts as success, what the budget buys, and the artefact question that has to be settled first |
| [07-review.md](07-review.md) | **An independent review of everything above**, recomputed from the records: three wrong numbers in §2 of 01, the half of the selection loss that is Monte-Carlo luck, why F1 and F3 cannot work as specified, why F9 saves nothing, what is no longer on disk, and what will not generalize past BENCHOP. Every correction above traces to a numbered section of it; one of its findings is rejected in [01](01-what-was-measured.md) §1a |
| [08-free-measurements.md](08-free-measurements.md) | **The four free measurements, run.** The quality index is anti-correlated with accuracy (8 of 30 within-pool pairs, p = 0.008); seed replication puts the selection loss at **1.69×**; F11 and F15 answered; F6's 104× unsupported |

## What this folder does not claim

- **It does not claim the pipeline failed.** Every plan the harness could grade passed the external
  gate — all **23 graded** plans have `harness_error < 1.0` (the 24th crashed in verification) —
  **though [08](08-free-measurements.md) §3 qualifies this: B02's shipped plan fails the gate at
  1 of 3 seeds and a sibling at 2 of 3, so on that problem the pass is the seed.** And
  every problem-level record carries the verdict `UNDERCLAIM`: the harness passes, the pipeline's
  own kernel scores 2–9 and certifies nothing. The pipeline is accurate enough and knows it least.
- **It does not claim these fixes close the gap.** On the measured record, perfect selection takes
  the pipeline from losing 7 problems to losing 4. The remaining four (B02, B03, B07, B08) are a
  crashed solver and three cases where the pool simply contained nothing as good as what C0 wrote.
  Those need F5/F6 and better solvers, which is a longer road.
- **It does not assume Opus.** None of F1–F11 is a model change, and
  [02-root-causes.md](02-root-causes.md) §7 gives the reason a model upgrade cannot reach most of
  them: the planner and formulator already run Opus, and the defects are in version-controlled
  Python and in the rules the manuals state.
