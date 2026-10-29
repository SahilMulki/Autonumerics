# 06 — The all-Sonnet retest

**Status:** proposal, 2026-10-28; **revised the same day against [07-review.md](07-review.md)**,
which found §1's central factual claim wrong (most of the "free" offline work needs artefacts that
are not on disk) and §6's primary target written against a statistic that has since been
deflated. §2's warning survived the review intact. This document exists because the plan is to implement fixes and
retest with **every agent at Sonnet, against a Sonnet C0 baseline, with fewer runs** — in order to
establish that AutoNumerics is at least on par with a single LLM completion before spending Opus
budget on it. That is the right experiment. This is how to run it so the result means something.

---

## §1 Read this first: the free measurement

> **[DONE, 2026-10-29]** All four measurements this section asks for have been run.
> **Results: [08-free-measurements.md](08-free-measurements.md).** The estimator validation in
> step 3 passed — and uncovered that the index built on the estimator is anti-correlated with
> accuracy, which changes the fix order. The seed replication is in [08](08-free-measurements.md)
> §3; [F11](05-fix-loop-and-harness.md#f11) and [F15](05-fix-loop-and-harness.md#f15) are
> answered in §4 and §5. The rest of this section is kept as written, because the artefact
> constraint it establishes still governs B05–B10.

**Do not spend a single pipeline call until the offline re-rank has been run — but first settle
what is actually on disk, because it is less than this section originally claimed.**

> **[C, against [07](07-review.md) §9]** ~~The ten pools' `<metrics>` blocks are on disk in
> `workspace/`.~~ **They are not.** `workspace/` holds plan directories for **B01–B04 only**:
>
> | problem | slug | plan dirs |
> |---|---|---|
> | B01 | `sde_sabr_cev_call_i` | 2 |
> | B02 | `sde_sabr_cev_call_ii` | 3 |
> | B03 | `sde_heston_call_feller` | 3 |
> | B04 | `sde_heston_dnt` | 3 |
> | **B05–B10** | `sde_qlsv_dnt`, `sde_hhw_call_corr`, `sde_hhw_call_zerocorr`, `sde_heston_call_hivar`, `sde_heston_dnt_hivar`, `sde_sabr_cev_call_ii_lowvol` | **0 — `problem.md` only** |
>
> Those six ran in cloud sessions and the artefacts never came back. Consequences:
>
> - **[F1](03-fix-selection.md#f1)'s acceptance tests cannot be run on B09/B10.** Use B01–B04.
> - **[F8](05-fix-loop-and-harness.md#f8)'s "free data point" does not exist.** B07's
>   `2-qe-log-euler` has a recorded `solver_sha256` and no file.
> - **The offline fixture is faithful for four problems, not ten.** The graded JSONs carry a
>   *reduced* per-plan `kernel` block — score, provenance, `estimated_rel_error`, `d1_outcome`,
>   `reference_outcome`, `invariant_max_drift`, `dynkin_z`, `solve_time_s` — with no order or
>   structure fields and no `config`. `sqi` renormalises over whatever is measurable, so an index
>   computed from those blocks is **not** the index the live run computed, and "today's ranked
>   order" recorded from it would not be today's ranked order.
> - **`--skip-run` is free of credits, not of compute.** Verification re-solves at 0.4–4M paths
>   plus the two-level check: minutes per plan.
> - **[F11](05-fix-loop-and-harness.md#f11) is genuinely free.** B01's two plan directories survive.
>
> **Decide before anything else whether B05–B10 are re-run to recover their artefacts, or whether
> every claim about them stays frozen at what the JSON records.** It is step 2 of §8.

What *is* available offline:

- B01–B04's eight plan directories, with `SOLUTION.md`, `solver.py` and full metrics.
- Every plan's harness error and margin, for all ten problems, in the graded JSONs
  ([01](01-what-was-measured.md) §0).
- [`verifylib/cli.py rank`](../../verifylib/cli.py) as a pure function of metrics blocks, and
  [`benchmark/run.py --skip-run`](../../benchmark/run.py#L875), which re-grades *without* invoking
  the conductor.

So, scoped to the four problems that still have artefacts:

1. Build a fixture from the ten pools: `{problem: [(plan, metrics_block, harness_error)]}`.
2. Record today's ranked order and today's selection loss **in both forms**: geometric mean
   shipped ÷ best-in-pool = **2.18× raw / 1.33× noise-floored**; winner-is-best on **1 of 9 raw,
   2 of 9 floored** (pools with two or more gradeable plans). ~~2.18×, winner-is-best on 3 of 10.~~
3. **Validate the error estimator before wiring it in** ([F3](03-fix-selection.md#f3)): run the
   dt-Richardson-plus-standard-error estimate over B01–B04's eight plans and check whether its
   predicted ordering matches the known harness errors. If it cannot separate B04's
   `1-euler-maruyama` (0.1741) from `3-qe-exact-variance` (0.0603), nothing built on it will
   improve selection.
4. Then implement F3, re-rank the fixture, and recompute both numbers.

That loop costs no credits and answers the central question of the whole retest before any are
spent. If F3 does not move the fixture, it will not move a live run either.

**Also free, and arguably the highest-value measurement available: seed replication.**
**[rev, from [07](07-review.md) §11.1]** Every conclusion in this folder rests on **one seed (42)
per plan**, and [01](01-what-was-measured.md) §3 shows what that costs — two of the three headline
selection failures are substantially Monte-Carlo luck. Re-solving the *shipped* and *best-in-pool*
plans at two further independent seeds costs minutes of compute, no credits and no pipeline call,
and it converts "the pool contained a plan 104× better" into a confidence interval. B01–B04 have
the solvers. **Do this before implementing anything**, because it is what decides whether the
selection program is worth 1.3× or 2.2×.

---

## §2 The warning that could invalidate the whole retest

> **Moving the planner from Opus to Sonnet without [F5](04-fix-planning.md#f5) first will probably
> make the pipeline worse, and the fixes will take the blame.**

The planner and formulator currently run `model: opus`
([`agents/plan-creator-sde.md:5`](../../agents/plan-creator-sde.md#L5)); only the solvers and
evaluators are Sonnet. An all-Sonnet run **downgrades the planner**.

And the measured record shows the planner's Opus-ness is doing real work. The scheme rule at
[`plan-creator-sde.md:21`](../../agents/plan-creator-sde.md#L21) admits Euler–Maruyama alone on
every multi-dimensional problem, which is all ten — and the Opus planner **disobeyed it in nine of
ten pools**, writing QE, exact-transition, AKS and Milstein-variant plans anyway. Nine of the 24
plans are non-Euler, and they are usually the more accurate plan.

A Sonnet planner is more likely to follow a written rule literally. If it does, the pools become
Euler-only, the best plan in the pool gets worse, and the run regresses for a reason that has
nothing to do with the fixes under test.

**Therefore: F5 and F6 are not optional for an all-Sonnet retest.** They are what makes the
instruction say what the Opus planner was already inferring. Land them before the model change, and
verify with the planner-only acceptance test in [04](04-fix-planning.md#f5) — one plan-creator call
per problem, no solver time, and it directly measures whether Sonnet proposes a non-Euler plan.

---

## §3 What to hold fixed

A comparison is only worth running if the two arms differ in the one thing under test.

| Hold fixed | Why |
|---|---|
| **The problems and their gates** | Do not re-cut a tolerance after seeing a result. `plan-benchop-slv.md` step 8 is explicit that moving a line after seeing which scheme lands on it is how a benchmark stops meaning anything |
| **The error metric** | `harness_error` = worst functional error ÷ **its own** per-value gate (`functional_tol_used`, not the declared scalar — see [01](01-what-was-measured.md) §1a). Same number as [01](01-what-was-measured.md) |
| **`num_paths`, `dt`, `seed`** | From the manifest, as now. A pipeline that wins by spending more paths has not won |
| **The suite boundary** | BENCHOP stays out of the 49-problem main denominator. The runner already refuses to merge them; keep it that way |
| **The Opus numbers from this study** | Keep them as a second, fixed yardstick. A Sonnet-only comparison can improve because the baseline got worse, and the Opus column is the only thing that will show that |

**[rev] One caveat on comparing across problems.** `harness_error` normalises by a gate this
project authored, and the suite's per-value gates span **0.002 to 0.015** — a factor of 7.5. A plan
on B06 is judged seven times harder than one on B03's `price_S125`. *Within* a pool the
normalisation is harmless and all the selection statistics are within-pool. *Across* problems every
geometric mean in this folder is partly a function of those authoring choices, and the headline
ratios are sensitive to gates nobody would defend to two significant figures. State it when
quoting them; do not re-cut the gates to fix it.

And one thing that must **change**: `solve_time_s` is the cost input and it is machine-dependent.
[`ranking.cost`](../../verifylib/ranking.py#L369) is pool-relative and log-scaled, so a uniform CPU
slowdown cancels — which is why the cloud shards were measurement-safe. **This only holds if all
plans in one pool run on one machine.** Do not split a pool across local and cloud.

---

## §4 The baseline

**Re-measure C0 at Sonnet. Do not reuse the Opus baseline and do not skimp on reps.**

The instinct to cut runs is right, but it should cut the *pipeline* runs, not the baseline. C0 cost
**$1.69 per problem for ten attempts** at Opus — $0.17 and 48 s per attempt. At Sonnet it is a
fraction of that. The baseline is the cheapest part of the whole study and it is the thing
everything else is measured against.

Report three cuts, which [`benchmark/baseline_select.py`](../../benchmark/baseline_select.py)
already produces and documents:

| cut | what it is | role |
|---|---|---|
| `first` | rep 1 only | **the headline.** One completion, one pipeline run — the honest single-draw comparison |
| `kernel` | the attempt `verifylib.ranking.rank_key` selects | matched attempts, matched selection rule, no oracle |
| `pass_at_k` | any attempt passes | the oracle-selection upper bound, which brackets how much of the gap is selection |

Using the *same* `rank_key` for both arms is deliberate and load-bearing — a selector with its own
rule would measure the rule rather than the baseline. Note the consequence: **F3 changes
`rank_key`, so it changes the `kernel` cut of the baseline too.** That is correct and it must be
stated when the numbers are reported, or the two arms will look like they moved for different
reasons.

**[rev, from [07](07-review.md) §11.2] Add a fourth cut: the matched-spend baseline.** The three
cuts above compare one pipeline run against one completion, or against ten. Neither is the
comparison a user faces. C0 costs $0.17 and 48 s; the pipeline costs 25–79 min and, at Opus, $4.88
for one problem. **At the pipeline's budget a user could run 50–100 C0 attempts and select among
them with the same `rank_key`** — which is precisely the "matched attempts, matched selection rule,
no oracle" arm the `kernel` cut defines, just at the right N. Report it at matched spend. It is the
bar that decides whether the pipeline is worth running at all, and on this evidence it is a much
harder bar than parity-with-one-completion.

Also carry forward the degeneracy check. On the Opus baseline, B09 produced **one** distinct answer
in ten attempts and B04 **two**; on those problems "best of ten" is meaningless. Count distinct
answers per problem and print it beside the baseline.

---

## §5 Which problems to run

With the pipeline at **24.6–78.7 min** per problem, a full ten-problem retest is the expensive
option. A five-problem subset buys most of the signal, chosen so that each problem tests a
different mechanism:

| | why it is in the subset | what a fix should do |
|---|---|---|
| **B10** | the extreme selection failure: 0.0041 vs 0.4250, both kernel 2, decided on cost | the winner must change. If F1–F4 do not fix B10 they have not worked |
| **B04** | pool contained a QE plan at 0.0603 = C0's best, to four decimals; shipped 0.1741 | the winner must change, and the shipped error must reach ≈0.06 |
| **B05** | a pure cost tie: identical `est_err`, decided on 61.4 s vs 77.4 s | the winner must change |
| **B08** | cost at weight 0.10 beat accuracy at weight 0.35 | the winner must change |
| **B01** | the kernel score itself inverted, 9 vs 8 against accuracy ([F11](05-fix-loop-and-harness.md#f11)) | diagnose before running |

**[rev] B02 belongs in the subset too** (tracked as [F15](#f15)). At 0.9594 it is within 4% of failing the external gate
and it is the only plan on the suite anywhere near it — the closest thing here to an actual
failure. ~~It is dismissed as an implementation-quality gap and left out.~~ It is not a selection
problem and F1–F4 will not move it, which is exactly why it needs diagnosing separately rather than
deferring: *what is wrong with that solver* is an open question and nothing in this folder answers
it.

Left out, with reasons: **B03, B06** are implementation-quality gaps — the pool's best plan is
still 25× and 3.4× behind C0 — so nothing in F1–F9 will move them and they are expensive to re-run.
**B09** is already a tie and C0 produced one distinct answer in ten attempts. **B07** ~~needs no run
at all~~ **must be re-run if it is wanted at all**: [F8](05-fix-loop-and-harness.md#f8)'s offline
re-grade is impossible because the plan directory is gone (§1).

Run the full ten only once the subset shows the fixes landing.

---

## §6 Pre-registered success criteria

Write these down before running, so the result cannot be re-interpreted afterwards.

**Primary — selection (what F1–F4 are for).** **[rev, against [07](07-review.md) §2]** The
original targets were written against the raw `best-in-pool` statistic, which is a minimum over
2–3 noisy draws: it is biased downward by construction, it gets *more* biased as the pool grows —
which [F9](05-fix-loop-and-harness.md#f9) proposes to do — and chasing it rewards a ranker that
picks lucky seeds. **Grade on the noise-floored version**, and report the raw one beside it so the
inflation stays visible.

**[M, 2026-10-29]** Both figures below are estimates of a quantity that has since been
**measured directly**. [08](08-free-measurements.md) §3 re-solved all eleven B01–B04 plans at two
further seeds; on three-seed means the loss is **1.69×**, between the raw 2.18× and the
noise-floored 1.33×. Grade on that where it is available (B01–B04) and keep the floored statistic
for B05–B10, whose artefacts are gone.

| | today | target |
|---|---|---|
| **geometric mean, shipped ÷ best-in-pool, three-seed means (B01–B04)** | **1.69×** | **< 1.15×** |
| geometric mean, shipped ÷ best-in-pool, **noise-floored** (all ten) | **1.33×** | **< 1.10×** |
| *same, raw (reported, not graded)* | *2.18×* | *— falls out* |
| winner is the pool's best, floored, of pools with ≥2 gradeable plans | **2 / 9** | **≥ 7 / 9** |
| **within-pool pairs where the higher SQI is the more accurate plan** | **8 / 30** ([08](08-free-measurements.md) §2) | **≥ 23 / 30** |
| B10's winner | `1-euler-maruyama-absorbing` | `2-qe-martingale-absorbing` |

The SQI row is new and is the one to watch: at 8 of 30 the index is *anti*-correlated with
accuracy (binomial p = 0.008), so a fix that merely randomised it would score 15/30 and look like
progress. The target is set where the index is actually informative.

~~geometric mean < 1.2×; winner-is-best ≥ 8/10 from 3/10.~~ The old 1.2× target was already nearly
met (1.34×) on the honest statistic the day it was written, which is the clearest sign it was
measuring the wrong thing.

B10 stays the single must-pass case, but for the right reason: its two plans differ on the graded
price by **3.4σ to 4.5σ**, so the better plan is better as a matter of discretization bias and not
of luck. A ranker that still ships the worse one has not been fixed.

**Secondary — the headline the retest exists to answer:**

| | today (vs Opus C0) | target (vs Sonnet C0) |
|---|---|---|
| problems where the pipeline beats or ties C0's median single attempt | 6 / 10 tie, **0 wins** | **≥ 3 wins**, 0 losses |
| geometric mean, shipped ÷ C0 median | **2.55×** | **≤ 1.0×** |

"On par" should mean *wins sometimes*. A system that proposes several plans, implements them,
measures them and iterates, and still never beats one completion on any problem, has not reached
parity — it has reached a tie by being consistently slightly worse everywhere.

**Diagnostic, to record whether or not it is a target:**

- The UNDERCLAIM rate. It is 15/15 today. If it is still 10/10 after the fixes, the certification
  ladder needs its own study ([05](05-fix-loop-and-harness.md) §5).
- Non-Euler plans proposed per problem. Zero would mean the Sonnet planner followed the rule
  literally and §2's warning came true.
- Which ranking key decided each winner — add it to the record
  ([F10](05-fix-loop-and-harness.md#f10)); it is one field and it would have saved this study a day.

---

## §7 Budget

| | per problem | 5 problems | notes |
|---|---|---|---|
| C0 at Sonnet, 10 reps | well under $1.69 | ~$5–8 | the Opus figure is the ceiling |
| C2 at Sonnet | **25–80 min**, rising with plan count | **~3–6 h** | ~~halved by F9~~ **[C]** [F9](05-fix-loop-and-harness.md#f9) is withdrawn: `max_iter` was never reached, so lowering it saves nothing. A third plan *raises* this |
| Offline re-ranks and re-grades | — | **free** | `--skip-run`, no pipeline call |

The offline work in §1 should come first precisely because it is free, and because it can
*disconfirm* the fixes before any credits are spent.

---

## §7a Baseline test state, measured 2026-10-28

`uv run pytest verifylib/tests benchmark/tests` — **1030 passed, 4 failed, 3 skipped, 11m32s.**
Record this before touching code; four red tests is a baseline you can diff against, and not
knowing which four is how a regression hides.

| failing test | what it is | bearing on this plan |
|---|---|---|
| `test_spec_field_coverage::test_no_sde_spec_field_is_declared_and_unread_without_a_reason` | the nine BENCHOP functionals are declared and read by nothing in `verifylib/` | **This is M1, already named by the suite.** [F0](03-fix-selection.md)/[F2](03-fix-selection.md) are its fix and it is their acceptance test |
| `test_review::test_no_false_positives_on_archived_reviews` | 1 of 115: B02's `2-exact-vol-em-absorbed` has `observed_order 0.235` in its review prose disagreeing with its `<metrics>` | **Caused by the step-10 runs**, not a code regression — a BENCHOP plan tripped the transcription-drift check. On B02, the problem closest to failing the gate ([F15](#f15)) |
| `test_certified::test_frozen_records_fall_back_to_the_old_test_unchanged` | `only 28 frozen pipeline records checked`, against `assert seen > 50` | A **coverage threshold**, not a correctness failure: the frozen corpus shrank below the bar, most likely when `results/` was reorganised in c21077e. This is the decision outstanding from an earlier step — fix the corpus path or lower the bar |
| `test_work_precision::test_an_sde_sweep_moves_both_axes_and_orders_by_paths_times_steps` | asserts `solve_s` rises monotonically across the sweep; got 0.0633 before 0.0538 on sub-100 ms solves | **A timing flake**, and pointedly the same class of defect [F4](03-fix-selection.md#f4) identifies: wall-clock used as a signal. Either seed it with a synthetic clock or compare with a tolerance |

So the suite is **not** red from a regression, and three of the four are informative rather than
obstructive. But **fix the flake before starting**: a test that fails intermittently on timing will
produce false alarms throughout the work, and F4 will be changing the very field it measures.

---

## §8 Order of work

**[rev]** Reordered against [07](07-review.md) §12. Steps 1–5 cost no credits, and two of them can
disconfirm the plan before anything is built.

0. **Record the baseline test state** (§7a: 1030 pass / 4 fail) and **fix the timing flake in
   `test_work_precision`** so later runs are diffable.
1. **Back up `benchmark/results/benchop-step10/` outside the repo.** It has been copied there out
   of the session scratchpad, but that path is gitignored and the credits that produced it are
   spent. Nothing else matters if it is lost.
2. **Decide the B05–B10 artefact question** (§1). Six of ten pools have no plan directories.
   Everything downstream depends on whether they can be touched again, and several acceptance
   tests are written assuming they can.
3. **Re-state the selection loss floored at each plan's own error bar, and replicate B01–B04's
   shipped and best plans at two further seeds** (§1, [01](01-what-was-measured.md) §3). If the
   deflated loss is 1.33× and stays there under replication, the selection program is worth much
   less than first claimed and §6's targets need rewriting again before code is written.
4. **[F11](05-fix-loop-and-harness.md#f11)** — diff B01's two metrics blocks. Free, B01's
   directories survive, and it is the one place the primary ranking key fired backwards. Do it
   early: if the kernel's own score mis-orders plans, F3's index — which sorts *below* `score` —
   can never outvote it.
5. **Build the offline re-rank fixture** on B01–B04 and record today's numbers, noting that the
   reduced `kernel` blocks in the JSON do not reproduce the live index (§1).
6. **[F0](03-fix-selection.md)** — carry `functionals` across the SDE sandbox boundary.
7. **[F12](03-fix-selection.md#f12)** — the solver-contract extension: per-functional standard
   errors returned by `solve_sde` and declared in `agents/solver-sde.md` and
   `references/sde_manual.md`, not only in each problem's `output_contract`. **Without this F3 has
   no error bar to snap to and F1 has no interval.** It was missing from the original fix list.
8. **The functional error estimator** — dt-Richardson at the graded path count under common random
   numbers, plus the reported standard error. **Validate it on B01–B04 against the known harness
   errors before wiring it into `accuracy`** ([F3](03-fix-selection.md#f3)). That is the real test
   of whether any oracle-free signal can order these plans.
9. **[F13](05-fix-loop-and-harness.md#f13)** — refuse an error estimate below the plan's own
   sampling noise. Cheap, general, and it makes B10's tie honest.
10. **[F3](03-fix-selection.md#f3)**, scoped as written, **with the consensus clause dropped**.
11. **[F1](03-fix-selection.md#f1)** as a disagreement detector at the sampling error — not at the
    gate, where it cannot fire.
12. **[F4](03-fix-selection.md#f4) on both cost paths** — the `sqi` component *and* `rank_key`'s
    fifth lexicographic key — plus a decision on CPU-time-vs-wall-time and the process model.
13. **[F10](05-fix-loop-and-harness.md#f10)**, extended to carry `decided_by` and each plan's
    `harness_error_margin`. One line for the main defect
    ([`report.py:552`](../../benchmark/report.py#L552) reads `rel_l2_err`, which does not exist
    here), and it is what made the review expensive.
14. **[F8](05-fix-loop-and-harness.md#f8)**, with the process-model decision from step 12 settled
    first, and **leaving the `sys.modules` registration in place** — the originally quoted
    `finally` would have reinstated the bug.
15. **[F5](04-fix-planning.md#f5) and [F6](04-fix-planning.md#f6)**, with F6 argued from the code
    rather than from B10's 104×, and F5's Step 1b keyed on a spec field rather than on the
    planner's recall of which SDEs have closed transition laws. **Verify at Sonnet, with the
    planner-only test, before the model change** — §2.
16. **[F7](03-fix-selection.md#f7)** as a reported tie, not a hidden randomization.
17. **[F14](05-fix-loop-and-harness.md#f14)** — decide what to do about the evaluator, which no
    other fix touches, and start recording per-cycle score deltas.
18. **Leave `max_iter` at 5** ([F9](05-fix-loop-and-harness.md#f9), withdrawn). If the plan count
    goes up, say so and expect wall time to rise with it.
19. **Run the Sonnet C0 baseline** on all ten problems — cheap, and it is the yardstick — reporting
    all four cuts including matched spend (§4).
20. **Run the C2 subset at Sonnet** (B10, B04, B05, B08, B01, plus B02 for diagnosis) and compare
    against §6.
21. Only then consider Opus, more reps, or the full ten.
