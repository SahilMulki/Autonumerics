# 01 — What was measured

**Status:** complete record, 2026-10-28; **revised the same day against
[07-review.md](07-review.md)**. Every number in this folder is quoted from here. Corrections the
review produced are marked inline; one of its findings is **rejected** and the rebuttal is in §1a,
because applying it would have made the two arms incommensurable.

## §0 Provenance of the numbers

Nothing below is re-derived or remembered. The sources:

| Arm | What | Records |
|---|---|---|
| **C0** | Opus one-shot, no tools. 10 attempts per problem (5 `naked` + 5 `manual`), 100 units, all PASS, all independently verified. `$16.86`, mean wall 48 s per attempt | four shard JSONs `s1.json`–`s4.json` |
| **C2** | The full pipeline, `claude --plugin-dir . -p "/conductor workspace/{slug}/problem.md"`. B01–B03 local at `--timeout 2400`; B04–B10 in cloud sessions at 5400 | `p_regrade123.json`, `p_b04.json`, `p_regrade4.json`, `p_b0506.json`, `p_b0708.json`, `p_b0910.json` |

The records have been copied out of the session scratchpad into
`benchmark/results/benchop-step10/`, together with the analysis scripts (`best.py`, `noise.py`,
`assemble.py`, `cpuprobe.py`). **That directory is gitignored** — `.gitignore:27` excludes
`benchmark/results/*`, which is the repo's standing convention that run results are regenerable and
not committed. These are not regenerable: the credits that produced them are spent. **Back them up
somewhere outside the repo** before the next run overwrites anything.

Two caveats that were found during analysis and must be carried forward:

- **`wrong_plan_won` on `p_local.json` is stale.** It pre-dates the step-10 margin fix and reports
  `margin: 0.0`, which calls every ranking wrong. B01–B03 were regraded into `p_regrade123.json`
  and B04 into `p_regrade4.json`; **those are the authoritative local records**, and after regrading
  B01 and B02 flip to `wrong_plan_won: False`. Do not quote `p_local.json`.
- **C0's ten attempts are not ten samples on every problem.** Distinct answers per problem:
  B01 6, B02 10, B03 9, **B04 2**, B05 6, B06 10, B07 10, B08 8, **B09 1**, B10 10. On B09 all ten
  attempts produced one answer and on B04 two, so "best of ten" means nothing there and the
  best/median/worst columns collapse.

## §1 The error metric

Every error in this folder is `harness_error` = **max over the problem's declared functionals of
(relative error ÷ that functional's own gate)**. It is dimensionless and 1.0 sits exactly on the
pass line, so 0.25 means "a quarter of the way to failing" and numbers from different problems are
comparable. It is computed by [`_functional_errs`](../../benchmark/verify.py#L1078):

```python
abs_errs[name] = abs(val - float(ref))
errs[name]     = abs_errs[name] / (abs(float(ref)) + 1e-14)
```

against independently authored truth in `benchmark/problems.py`, never against anything the
formulator wrote.

**The noise margin.** `hypot(reference_abs_err, sampling_se) / ref / tol`, where
`_functional_sampling_se = |delta| / sqrt(n_hi/n_lo − 1)` comes from the two-level anti-hardcoding
re-solve ([`benchmark/run.py:555`](../../benchmark/run.py#L555)). Its own docstring says it takes
the larger of the two available divisors so that the comparison *errs towards not calling a ranking
wrong*, and it is one realization — a noise estimate with a single degree of freedom. **It is
therefore generous**, and `wrong_plan_won: False` should be read as "not provably wrong", not as
"right". B04-local is the clean example: the winner is 2.9× less accurate than a sibling and the
margin still calls it a tie.

### §1a Rejected correction — the denominator is the per-value gate, not the declared scalar

[07](07-review.md) §1 recomputes the C0 column as `worst_functional_rel_err / functional_tol` and
reports B03's best as 0.0072 (not 0.0178), B03's median as 0.0709 (not 0.1265) and B08's median as
0.0392 (not 0.1430), concluding both headline geometric means should move against the pipeline
(8.40 → 8.69, 2.55 → 2.90). **That recomputation uses the wrong denominator and the original
numbers stand.**

`functional_tol` is a single declared scalar. The gate each value is actually held to is
`functional_tol_used`, computed per value, and on B03 and B08 the per-value gates are **tighter**
than the declared scalar, not merely widened:

| | declared `functional_tol` | `functional_tol_used` |
|---|---|---|
| B03, B08 | 0.015 | `price_S75: 0.015`, `price_S100: 0.004`, `price_S125: 0.002` |

Those are the only two problems where the two differ, which is exactly the two the review flags —
the other eight reproduce identically under either convention, which is what makes the cause
legible.

The decisive argument is cross-arm consistency. The C2 records compute their own `harness_error`
against `functional_tol_used`: B10's winner carries `worst_functional_rel_err: 0.004250`,
`functional_tol_used: 0.01` and `harness_error: 0.4250`. Scoring C0 against the declared scalar
while C2 is scored against the per-value gate would compare two different quantities and call the
difference a result. §1's definition — *max over functionals of (relative error ÷ **its own**
gate)* — is the one both arms use, and it is the one that matches the pass/fail decision.

The suite's per-value gates span **0.002 to 0.015**, a factor of 7.5. See §8 for what that means
for comparing numbers across problems.

## §2 Headline: C2 as shipped vs C0

Gate units; lower is better; 1.0 is the pass line. `verdict` applies the benchmark's own margin
rule to the pair.

| | C0 best | C0 median | C0 worst | C2 shipped | ship/C0-best | ship/C0-median | vs C0 best | vs C0 median |
|---|---|---|---|---|---|---|---|---|
| B01 | 0.0180 | 0.0737 | 0.2539 | 0.2584 | 14.4× | 3.51× | tie | tie |
| B02 | 0.0116 | 0.1658 | 0.4163 | 0.9594 | 82.6× | 5.79× | **C0** | **C0** |
| B03 | 0.0178 | 0.1265 | 0.6549 | 0.4606 | 25.9× | 3.64× | **C0** | **C0** |
| B04 | 0.0603 | 0.0603 | 0.3537 | 0.3121 | 5.2× | 5.17× | **C0** | **C0** |
| B05 | 0.0391 | 0.2025 | 0.3625 | 0.1936 | 5.0× | 0.96× | **C0** | tie |
| B06 | 0.1445 | 0.3259 | 0.8696 | 0.5177 | 3.6× | 1.59× | tie | tie |
| B07 | 0.0666 | 0.1232 | 0.4903 | 0.2952 | 4.4× | 2.40× | **C0** | tie |
| B08 | 0.0292 | 0.1430 | 0.2788 | 0.3836 | 13.1× | 2.68× | **C0** | tie |
| B09 | 0.2243 | 0.2243 | 0.2243 | 0.3017 | 1.3× | 1.35× | tie | tie |
| B10 | 0.0528 | 0.1825 | 0.6320 | 0.4250 | 8.1× | 2.33× | **C0** | **C0** |

- **Against C0's best of ten:** C0 wins 7, ties 3, loses 0. Geometric mean ratio **8.41×**.
- **Against C0's median single attempt** — the apples-to-apples single-draw comparison, since the
  pipeline also gets one shot: C0 wins 4, ties 6, loses 0. Geometric mean ratio **2.55×**.

The honest headline is the second one. The pipeline is not catastrophically behind a single Opus
completion; it is consistently, modestly behind it and **never ahead of it on any problem**. A
system that can propose multiple plans, implement them, measure them and iterate should beat a
single completion sometimes. It does not do so once in ten.

## §3 The pipeline's own pool already contains better

Same metric. "best in pool" is the most accurate plan the pipeline wrote and the harness graded,
whether or not the conductor shipped it.

| | C2 shipped | C2 best in pool | ship / best | best vs C0-best |
|---|---|---|---|---|
| B01 | 0.2584 | **0.0259** | 9.98× | tie |
| B02 | 0.9594 | 0.7592 | 1.26× | C0 |
| B03 | 0.4606 | 0.4478 | 1.03× | C0 |
| B04 ☁ | 0.3121 | 0.3121 | 1.00× | C0 |
| B04 💻 | 0.1741 | **0.0603** | 2.89× | tie |
| B05 | 0.1936 | 0.1315 | 1.47× | tie |
| B06 | 0.5177 | 0.4897 | 1.06× | tie |
| B07 | 0.2952 | *(QE plan crashed in verification)* | — | C0 |
| B08 | 0.3836 | 0.3350 | 1.15× | C0 |
| B09 | 0.3017 | 0.2968 | 1.02× | tie |
| B10 | 0.4250 | **0.0041** | **103.7×** † | tie |

† raw. See the noise-floored column below: B10's honest figure is **7.4×**, and it is the one
problem on the suite where the selection failure is statistically resolvable.

~~**Geometric mean shipped / best-in-pool = 2.18×.** That factor is pure selection loss.~~
**[rev, against [07](07-review.md) §2]** 2.18× is reproducible but is an **upper bound**, and it is
convention-dependent. Both corrections below are material.

**The convention, stated.** 2.18× is one row per problem, B04's cloud replicate, and B07 counted as
ratio 1.0 because only one of its two plans was gradeable. Dropping B07 gives 2.38×; using B04's
local replicate instead gives 2.42×; both B04 rows and no B07 gives 2.43×. The spread is the
convention, not the data, and 2.18× is the least unfavourable of them — which is the reason to
state it rather than quote it bare.

**The winner's curse.** `best-in-pool` is a minimum over 2–3 noisy draws, so it is biased downward
by construction and gets more biased as the pool grows. The repo's own rule applies here and was
not applied: [`ranking.accuracy`](../../verifylib/ranking.py#L99) snaps an error up to its own
uncertainty, because *"a plan cannot claim accuracy finer than its own error bar."* The records
carry that bar per plan as `harness_error_margin`. Flooring each plan's error at its own margin and
then taking the pool minimum:

| | shipped | best in pool (raw) | best, floored at its own bar | which plan |
|---|---|---|---|---|
| B01 | 0.2584 | 0.0259 | **0.2584** | *the shipped plan* |
| B02 | 0.9594 | 0.7592 | 0.7592 | 3-aks-decorrelated-em |
| B03 | 0.4606 | 0.4478 | 0.4478 | 3-qe-exact-variance |
| B04 ☁ | 0.3121 | 0.3121 | 0.3121 | 1-euler-maruyama |
| B04 💻 | 0.1741 | 0.0603 | **0.1668** | 3-qe-exact-variance |
| B05 | 0.1936 | 0.1315 | 0.1315 | 1-euler-maruyama-bessel |
| B06 | 0.5177 | 0.4897 | 0.4897 | 2-qe-martingale-corrected |
| B07 | 0.2952 | — | 0.2952 | (sibling crashed) |
| B08 | 0.3836 | 0.3350 | 0.3350 | 2-andersen-qe |
| B09 | 0.3017 | 0.2968 | 0.2968 | 1-euler-maruyama |
| B10 | 0.4250 | **0.0041** | **0.0573** | 2-qe-martingale-absorbing |

> **Geometric mean shipped ÷ best-in-pool: raw 2.18×, noise-floored 1.33×.**

Two individual claims collapse and must not be repeated:

- **B01's 9.98× is not a selection failure.** `2-exact-conditional-besq0` measured 0.0259 against
  a noise bar of **0.5293** gate units — its measured error is one twentieth of its own standard
  error, so the pool contains no evidence it is better than the plan that shipped. A correct
  implementation of F3's own snap-up rule would not change B01's winner, and should not.
- **B10's 103.7× is really ≈7×.** The two plans' prices differ by 8.7e-5 to 1.5e-4 absolute against
  combined sampling standard errors of 1.9e-5 to 4.4e-5 — **3.4σ to 4.5σ**, so plan 2 *is*
  genuinely the better scheme and this *is* a real selection failure. But its measured 0.0041 is
  0.07 of its own error bar: a lucky draw on top of a real bias. The defensible statement is
  **"perfect selection takes B10 from 0.425 to ≈0.057"**, a factor of 7.4.

**Neither bound is the answer.** The raw statistic selects on Monte-Carlo luck; the floored one uses
`harness_error_margin`, which §1 above establishes is deliberately conservative — the larger of two
divisors, one degree of freedom, built to *avoid* calling a ranking wrong. So it over-deflates. The
honest reading is **selection costs somewhere between ~1.3× and ~2.2×, and a single-seed design
cannot narrow it.** Resolving it is cheap and is now step 4 of
[06](06-retest-protocol.md) §8: re-solve the shipped and best plans at two further seeds on
B01–B04, the four problems whose solvers still exist.

**How often was the shipped plan the pool's best?** Raw: **1 of the 9 pools that had two or more
gradeable plans** (B04 cloud). Noise-floored: **2 of 9** (B01, B04 cloud). ~~3 of 10.~~

With perfect selection the suite result moves from **7–0–3** to **4–0–6**. B04's local replicate is
the sharpest single demonstration that survives the noise floor only partly — its pool contained a
QE plan at 0.0603, which is C0's best answer *to four decimal places*, and it shipped an Euler plan
at 0.1741; floored, that 2.89× becomes 1.04×.

## §4 The full per-plan record

This is the table the rest of the folder argues from. `harn` is the harness error, `kscore` the
kernel's score, `est_err` the `estimated_rel_error` the ranker's accuracy component actually reads,
`solve_s` the solver time the cost component reads, `*` the plan the conductor shipped.

| id | plan | harn | kscore | provenance | est_err | solve_s | |
|---|---|---|---|---|---|---|---|
| B01 | 1-euler-maruyama-absorbing | 0.2584 | 9 | analytic | 3.68e-03 | 28.7 | **\*** |
| B01 | 2-exact-conditional-besq0 | **0.0259** | **8** | analytic | **7.07e-04** | 93.5 | |
| B02 | 1-euler-maruyama-absorbed | 0.9594 | 7 | surrogate | 7.52e-03 | 43.8 | **\*** |
| B02 | 2-exact-vol-em-absorbed | 0.9098 | 7 | surrogate | 7.91e-03 | 55.8 | |
| B02 | 3-aks-decorrelated-em | **0.7592** | 7 | surrogate | 7.37e-03 | 74.7 | |
| B03 | 1-full-truncation-log-euler | 0.5803 | 9 | analytic | 2.55e-03 | 20.9 | |
| B03 | 2-drift-implicit-variance | 0.4606 | 9 | analytic | **1.32e-03** | 20.1 | **\*** |
| B03 | 3-qe-exact-variance | **0.4478** | 9 | analytic | 3.26e-03 | 82.7 | |
| B04 | 1-euler-maruyama | 0.1741 | 9 | analytic | 7.53e-03 | 57.9 | **\*** |
| B04 | 2-drift-implicit-variance-leg | 0.2246 | 9 | analytic | 9.82e-03 | 58.9 | |
| B04 | 3-qe-exact-variance | **0.0603** | 9 | analytic | **4.95e-03** | 64.9 | |
| B05 | 1-euler-maruyama-bessel | **0.1315** | 7 | self_convergence | 1.70e-02 | 77.4 | |
| B05 | 2-euler-maruyama-drift-implicit | 0.1936 | 7 | self_convergence | 1.70e-02 | **61.4** | **\*** |
| B06 | 1-euler-maruyama-full-truncation | 0.5177 | 7 | self_convergence | 8.88e-07 | 93.5 | **\*** |
| B06 | 2-qe-martingale-corrected | **0.4897** | 7 | self_convergence | 8.69e-06 | 96.1 | |
| B07 | 1-euler-maruyama-full-truncation | 0.2952 | 3 | surrogate | 2.26e-03 | 245.7 | **\*** |
| B07 | 2-qe-log-euler | *crash* | 3 | surrogate | **4.54e-04** | **111.1** | |
| B08 | 1-full-truncation-euler | 0.3836 | 9 | analytic | 2.53e-03 | **80.5** | **\*** |
| B08 | 2-andersen-qe | **0.3350** | 9 | analytic | **2.36e-03** | 217.5 | |
| B08 | 3-truncated-milstein | 0.3395 | 1 | none | — | — | |
| B09 | 1-euler-maruyama | **0.2968** | 9 | analytic | 8.53e-03 | 77.3 | |
| B09 | 2-euler-maruyama-cir-milstein | 0.3017 | 9 | analytic | **1.80e-03** | **51.0** | **\*** |
| B10 | 1-euler-maruyama-absorbing | 0.4250 | 2 | none | 2.22e-14 | **171.3** | **\*** |
| B10 | 2-qe-martingale-absorbing | **0.0041** | 2 | none | 2.34e-14 | 198.8 | |

Read the columns against each other and the mechanism falls out:

- **The kernel score is tied inside the pool in 8 of 9 pools, and provenance in 9 of 9.** The first
  two lexicographic ranking keys — the two that are *supposed* to decide — never separate anything
  on this suite. Every decision fell through to the quality index and to cost.
- **`est_err` is the wrong quantity.** It is the worst of the mean and variance errors
  ([`verifylib/kernel/metrics.py:210`](../../verifylib/kernel/metrics.py#L210)), and its ordering
  disagrees with the harness error in B03, B06 and B09, ties at round-off in B05 and B10, and is
  absent in B08's third plan.
- **Where everything ties, `solve_s` decides**, and the more accurate plan is slower in 5 of the 6
  pools that contain both an Euler and an exact/QE plan (93.5 vs 28.7; 82.7 vs 20.9; 64.9 vs 57.9;
  217.5 vs 80.5; 198.8 vs 171.3). B07 is the one exception and its better plan crashed.

## §5 What C0 actually did differently

C0's best attempt per problem, by scheme family, classified from the completion text:

| scheme family C0 used | problems |
|---|---|
| exact transition (non-central χ² / BESQ / Bessel) | B01, B02, B03, B05, B10 |
| Andersen QE | B04, B07, B09 |
| conditional Black–Scholes + control variate | B08 |
| Euler, full truncation | B06 |

**Nine of ten.** C0 reached for an exact transition law or a QE scheme on nine of the ten problems.
The pipeline's planner is instructed to propose Euler–Maruyama alone on every multi-dimensional
problem, and all ten of these are multi-dimensional. See
[04-fix-planning.md](04-fix-planning.md) §1.

There is no variance reduction anywhere in the pipeline's 24 plans. C0 used a control variate on
B08 and the BENCHOP authors' own SDE entries (MCA, MLMC, SGBM, mSABR) are built on variance
reduction and multilevel sampling. This is a whole axis of method choice the pipeline never visits.

## §6 Cost

| | per problem | notes |
|---|---|---|
| C0, ten attempts | **$1.69**, ~8 min total | $0.17 and 48 s per attempt |
| C2, Sonnet solvers | **24.6–78.7 min wall** | per problem, from each record's own `wall_seconds` |
| C2, `--model opus` | **$4.88** for one problem (B06), truncated at `iter: 0` | plus ~$5–6 for a killed first attempt |

~~The pipeline costs roughly 30–60× the wall-clock of ten C0 attempts.~~ **[C, against
[07](07-review.md) §1]** The earlier 43–87 min range was derived by halving two-problem session
totals; the records carry a per-problem `wall_seconds` and the real figures are B04 24.6, B09 28.2,
B05 33.0, B06 37.8, B10 52.7, B07 58.9, B08 78.7 minutes. The pipeline costs roughly **20–60× the
wall-clock of ten C0 attempts** and currently returns a less accurate answer. Note what those
numbers track: the three longest runs are B07, B08 and B10, and **wall time scales with plan
count**, not with iterations — B08 at 3 plans took 78.7 min against B09 at 2 plans in 28.2. Every run spent exactly **two** solver↔evaluator cycles per plan — every one of the 24 plan
records carries `iter: 2`, and `total_iters == 2 × n_plans` in all 15 — because the plateau
early-stop fired universally: no plan's score improved after its second cycle anywhere on the
suite. **Cycles 3–5 of the `max_iter = 5` budget were therefore never reached, which means
`max_iter = 5` cost exactly what `max_iter = 2` would have cost.** See
[05-fix-loop-and-harness.md](05-fix-loop-and-harness.md#f9), where the original claim that lowering
it saves money is withdrawn.

## §7 Verdict distribution

Every one of the 15 **problem-level** records carries the verdict **`UNDERCLAIM`**: the independent harness
passes, and the pipeline's own kernel scores 2–9 and certifies nothing. Not one plan on the suite
reached `certified: true`, so the conductor's winner early-exit never fired and every pool was
decided by the Phase-3 ranker after all plans had plateaued.

**[C]** Per *plan* the picture is one step less clean: 23 plans were graded and all 23 clear the
gate, while B07's `2-qe-log-euler` crashed in verification and carries `FAIL`. "All 24 graded plans
pass" is wrong by one; 24 is the plan count, 23 is the graded count.

That is a double failure of calibration worth stating plainly: on this suite the kernel is
**too harsh** — it certifies nothing that in fact passes the external gate — and simultaneously
**too blind** — it cannot distinguish a price at 0.0041 of its gate from one at 0.4250, and scores
both 2.
