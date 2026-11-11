# C0 @ Opus, three naked draws — and the first head-to-head against round 3's pipeline

**Measured 2026-11-09.** A re-run of the Opus one-shot baseline on BENCHOP-SLV at **3 draws**,
compared against the **most recent AutoNumerics pipeline run at Sonnet**. Error is in
**gate units** throughout ([10](10-sonnet-retest.md) §9: worst scored functional's relative error
divided by that functional's own tolerance, so **1.0 is the pass line** on every problem).

## The run

```bash
uv run python benchmark/oneshot.py --suite benchop --condition c0 --manuals naked --reps 3 \
  --model opus --c0-max-turns 8 --agent-timeout 1800 --timeout 1800 --mem-mb 4096 \
  --results-json benchmark/results/baselines/oneshot_benchop_c0_opus_naked3.json
```

30 units, **30 PASS / 0 FAIL / 0 CRASHED**, 30/30 independently verified, 54.7 min, **$5.76**.
29 of 30 units answered in a single turn.

**The artefacts are committed in [`data/`](data/)**, because `benchmark/results/` is gitignored and
git cannot un-ignore a file inside an ignored subdirectory (`.gitignore:27` says so explicitly).
This is [11](11-sde-accuracy-fixes.md) §8 item 8's failure mode; the copy is the fix.

| file | what it is |
|---|---|
| [`data/oneshot_benchop_c0_opus_naked3.json`](data/oneshot_benchop_c0_opus_naked3.json) | the graded record, all 30 units |
| [`data/oneshot_benchop_c0_opus_naked3.log`](data/oneshot_benchop_c0_opus_naked3.log) | the driver's per-unit console log |
| [`data/c0_opus_naked3_compare.py`](data/c0_opus_naked3_compare.py) | every number in the table below |
| [`data/c0_opus_naked3_bars.py`](data/c0_opus_naked3_bars.py) | the noise bars in § The bars |

### This is the `naked` arm, and that limits what it compares to

The 2026-10-25 ten-draw Opus baseline pooled **5 `naked` + 5 `manual`**
([01](01-what-was-measured.md) §1), so this is **not** a three-draw subset of it — it is the
no-reference-material condition on its own. It is likewise **not** a clean model swap against
[10](10-sonnet-retest.md) §7's Sonnet C0 control, which ran `--manuals manual` to match the
formulator's knowledge. Against the pipeline the knowledge asymmetry runs **in the pipeline's
favour**: the pipeline's formulator reads `references/sde_manual.md` and this arm reads nothing.

## The pipeline side: which run this is

"Most recent at Sonnet" is two files, because no single run covers all ten problems:

- **B01, B02, B04–B10** — `benchmark/results/pipeline/benchop_sonnet_other9_run2.json`,
  generated 2026-11-07, problems started 2026-11-05 through 2026-11-07.
- **B03** — `benchmark/results/pipeline/benchop_b03_sonnet_run4.json`, 2026-11-05: the round-3 live
  test written up in [12](12-implementation.md) §7.

Both are post-round-3, so this is the first head-to-head in which the pipeline's accuracy component
and order band actually reach the ranker. Pools are 3 plans on all ten problems.

## The table

`AN ship` is the plan the pipeline selected; `AN pool` is the pool's best plan by harness error —
the gap between them is the selection loss. Lower is better everywhere; **bold** marks the winner
of each comparison.

| id | problem | C0 draw 1 | draw 2 | draw 3 | C0 median | C0 best-of-3 | AN ship | AN pool best | median ÷ ship | best3 ÷ pool |
|---|---|---|---|---|---|---|---|---|---|---|
| B01 | `sde_sabr_cev_call_i` | 0.0317 | 0.0180 | 0.0623 | 0.0317 | 0.0180 | **0.0189** | 0.0189 | 1.68 | **0.95** |
| B02 | `sde_sabr_cev_call_ii` | 0.0913 | 0.1106 | 0.1890 | 0.1106 | 0.0913 | **0.0015** | 0.0015 | 75.37 | 62.24 |
| B03 | `sde_heston_call_feller` | 0.0391 | 0.0143 | 0.1539 | 0.0391 | 0.0143 | 0.0391 | 0.0391 | 1.00 | **0.37** |
| B04 | `sde_heston_dnt` | 0.0603 | 0.0603 | 0.0603 | 0.0603 | 0.0603 | **0.0255** | 0.0255 | 2.37 | 2.37 |
| B05 | `sde_qlsv_dnt` | 0.0672 | 0.2496 | 0.2025 | **0.2025** | 0.0672 | 0.2299 | 0.0441 | **0.88** | 1.52 |
| B06 | `sde_hhw_call_corr` | 0.4847 | 0.2221 | 0.1508 | 0.2221 | 0.1508 | **0.0898** | 0.0898 | 2.47 | 1.68 |
| B07 | `sde_hhw_call_zerocorr` | 0.1097 | 0.0250 | 0.2301 | 0.1097 | 0.0250 | **0.0085** | 0.0085 | 12.84 | 2.92 |
| B08 | `sde_heston_call_hivar` | 0.0126 | 0.0305 | 0.0292 | 0.0292 | 0.0126 | 0.0274 | 0.0274 | **1.07** | **0.46** |
| B09 | `sde_heston_dnt_hivar` | 0.2243 | 0.2243 | 0.2243 | **0.2243** | 0.2243 | 0.2774 | 0.2774 | **0.81** | **0.81** |
| B10 | `sde_sabr_cev_call_ii_lowvol` | 0.1863 | 0.4721 | 0.0692 | **0.1863** | 0.0692 | 0.2403 | 0.0291 | **0.78** | 2.38 |

| geometric mean, ten problems | gate units |
|---|---|
| C0 @ Opus, median draw | 0.0932 |
| C0 @ Opus, best-of-3 | 0.0484 |
| **AN @ Sonnet, shipped** | **0.0393** |
| AN @ Sonnet, pool best | 0.0270 |

**Head to head.** On the comparison that matters — one draw each, median against shipped — the
pipeline at Sonnet is **2.37× more accurate than Opus with no tools**, winning **7 of 10**. On
generation quality alone (best-of-3 against pool best) the pipeline leads **1.79×**, winning 6 of
10. Both directions reverse [01](01-what-was-measured.md)'s headline finding, which had the
pipeline 2.6× *behind* C0's median single attempt and losing 7–0–3.

**Selection is now cheap.** Shipped ÷ pool best is **1.46×** on the geomean, and `wrong_plan_won`
is true on only **2 of 10** problems (B05, B10) — against four of ten in
[10](10-sonnet-retest.md) and the 1.69× of [08](08-free-measurements.md) §3. Those two carry
essentially the whole loss: B05 ships 5.2× its pool best and B10 ships 8.3× its.

## Four things the table does not say

1. **Most of this is inside the noise.** The bars below, computed per
   [10](10-sonnet-retest.md) §9, exceed the measured error on **many** units in both arms — B10's
   shipped plan reads 0.2403 against a bar of 0.0022, but B03's shipped 0.0391 sits under a bar of
   0.2973, and C0's best B03 draw (0.0143) under a bar of 0.0052. Per-problem winners are not
   individually significant; the 7–3 count and the geomean are the claim.
2. **B06's pipeline number comes from a `RUN_ERROR`.** Its run status is `timeout` — the problem
   exceeded the 7200 s budget — and the 0.0898 is read from plan harness blocks that were graded
   before the conductor was killed. It is a plan's error, but it is not a completed pipeline run,
   and B06 is one of the pipeline's wins.
3. **B04's three Opus draws, and two of B09's, agree to 12 significant figures** — not
   bit-identical, as a first pass of this document claimed, and the difference matters: they are
   *different programs* evaluating *the same estimator on the same sample*. Best-of-3 is
   effectively best-of-1 there. § The duplicate-answer mechanism explains why.
4. **Best-of-N is a winner's curse** ([10](10-sonnet-retest.md) §8). The best-of-3 column is biased
   low by construction and the 1.79× generation figure should not be quoted without that caveat.

Every problem-level verdict on the pipeline side is still **UNDERCLAIM** (or `RUN_ERROR` on B06):
the harness passes, the kernel certifies nothing. That has not changed since round 1.

## The bars

Noise bar in gate units for the functional that decided each score, per
[10](10-sonnet-retest.md) §9. Ground truth is `exact` on all ten problems, so the reference term is
zero and the bar is the sampling SE alone, read off the anti-hardcoding check's two path levels.

| id | C0 bar, draw 1 | draw 2 | draw 3 | AN ship gate units | AN ship bar |
|---|---|---|---|---|---|
| B01 | 0.2206 | 0.4369 | 0.0213 | 0.0189 | 0.0116 |
| B02 | 0.4143 | 0.0910 | 0.0968 | 0.0015 | 0.0185 |
| B03 | 0.0236 | 0.0052 | 0.2693 | 0.0391 | 0.2973 |
| B04 | 0.1668 | 0.1668 | 0.1668 | 0.0255 | 0.0576 |
| B05 | 0.0768 | 0.1815 | 0.2734 | 0.2299 | 0.0731 |
| B06 | 0.0900 | 0.1215 | 0.2420 | 0.0898 | 0.1487 |
| B07 | 0.0646 | 0.0213 | 0.0734 | 0.0085 | 0.0203 |
| B08 | 0.0021 | 0.0272 | 0.0244 | 0.0274 | 0.0081 |
| B09 | 0.1674 | 0.1674 | 0.1674 | 0.2774 | 0.1381 |
| B10 | 0.2169 | 0.5286 | 0.2594 | 0.2403 | 0.0022 |

## Six arms, best of the pool

**Added 2026-11-09.** Every arm measured on BENCHOP-SLV to date, on one axis: **the best result
available to that arm** — best-of-N over draws for the one-shot arms, best plan in the pool for the
pipeline arms. This is a *generation-quality* comparison and deliberately excludes the shipped plan
and the median draw, so **nothing here reflects selection**. Gate units, lower is better, 1.0 is the
pass line. Produced by [`data/six_arm_pool_best.py`](data/six_arm_pool_best.py).

| id | problem | Opus one-shot, best of 10 draws | Sonnet one-shot, best of 3 draws | Opus one-shot, best of 3 draws | AutoNumerics run 1, best plan in pool | AutoNumerics run 2 (Sonnet), best plan in pool | AutoNumerics run 3 (Sonnet), best plan in pool |
|---|---|---|---|---|---|---|---|
| B01 | `sde_sabr_cev_call_i` | **0.0180** | **0.0180** | **0.0180** | 0.0259 | 0.0473 | 0.0189 |
| B02 | `sde_sabr_cev_call_ii` | 0.0116 | 0.4596 | 0.0913 | 0.7592 | 0.1674 | **0.0015** |
| B03 | `sde_heston_call_feller` | 0.0178 | 0.0579 | **0.0143** | 0.4478 | 0.4084 | 0.0391 |
| B04 | `sde_heston_dnt` | 0.0603 | 0.0603 | 0.0603 | 0.0603 | 0.1353 | **0.0255** |
| B05 | `sde_qlsv_dnt` | **0.0391** | **0.0391** | 0.0672 | 0.1315 | 0.1006 | 0.0441 |
| B06 | `sde_hhw_call_corr` | 0.1445 | 0.5199 | 0.1508 | 0.4897 | **0.0474** | 0.0898 |
| B07 | `sde_hhw_call_zerocorr` | 0.0666 | 0.0659 | 0.0250 | 0.2952 | 0.0388 | **0.0085** |
| B08 | `sde_heston_call_hivar` | 0.0292 | 0.2417 | **0.0126** | 0.3350 | 0.1543 | 0.0274 |
| B09 | `sde_heston_dnt_hivar` | 0.2243 | **0.0782** | 0.2243 | 0.2968 | 0.1175 | 0.2774 |
| B10 | `sde_sabr_cev_call_ii_lowvol` | 0.0528 | 0.1106 | 0.0692 | **0.0041** | 0.1612 | 0.0291 |
| | **geometric mean** | 0.0443 | 0.0979 | 0.0484 | 0.1451 | 0.1090 | **0.0270** |

**Corrected 2026-11-09:** a first version of this table left run 2's B03 cell empty and averaged
that column over nine problems. B03 *was* run in that arm — separately, as
`benchop_b03_sonnet_run3.json`, which [10](10-sonnet-retest.md) §0 names as part of the same arm —
so the cell is 0.4084 and the geomean is over ten problems like every other column. The earlier
figure was 0.0941, biased low by dropping that arm's worst problem.

### What each column is

| column | record | model | draws or plans |
|---|---|---|---|
| Opus one-shot, best of 10 | `benchop-step10/s1–s4.json`, 2026-10-25 | opus | 10 per problem (5 `naked` + 5 `manual`) |
| Sonnet one-shot, best of 3 | `baselines/oneshot_benchop_c0_sonnet.json`, 2026-11-02 | sonnet | 3 per problem, `manual` |
| Opus one-shot, best of 3 | `data/oneshot_benchop_c0_opus_naked3.json`, 2026-11-09 | opus | 3 per problem, `naked` |
| AutoNumerics run 1 | `benchop-step10/p_regrade123`, `p_regrade4`, `p_b0506`, `p_b0708`, `p_b0910`, 2026-10-26 | default (Opus); the config does not record `--model` | pool of 1–3: B07 **1**; B01, B05, B06, B09, B10 **2**; rest 3 |
| AutoNumerics run 2 | `pipeline/benchop_sonnet.json` 2026-11-02 (nine) + `benchop_b03_sonnet_run3.json` 2026-11-01 (B03) — [10](10-sonnet-retest.md) §0's "Now" arm | sonnet | pool of 3, except B04 **2** |
| AutoNumerics run 3 | `pipeline/benchop_sonnet_other9_run2.json` 2026-11-07 + `benchop_b03_sonnet_run4.json` 2026-11-05 | sonnet | pool of 3 on all ten |

**Pool and draw counts are not equal across columns**, so this table is not a controlled comparison.
Run 1's B07 pool held one plan and the Opus column had ten draws to minimise over: the former is
handicapped and the latter flattered, both by construction
([10](10-sonnet-retest.md) §8 — best-of-N is a winner's curse, and it gets worse with N).

### What it shows

- **The pipeline's generation improved 5.4× across three runs**, 0.1451 → 0.1090 → 0.0270, while
  moving *down* a model tier at run 2. Run 3 has the best geomean on the table, but it wins only
  **3 rows outright** (B02, B04, B07) against Opus×3's 2 and one each for Sonnet×3, run 1 and run 2.
  **Its lead is breadth, not row wins**: it is top-two on **6 of 10** problems, more than any other
  arm, and its worst problem is 0.2774 — where the two arms that beat it on individual rows blow up
  far worse (Sonnet×3 to 0.5199 on B06, run 1 to 0.7592 on B02). The Opus one-shot arms are the
  genuine competition: both cap at 0.2243, better than run 3's worst.
- **Run 3 beats the ten-draw Opus baseline by 1.64×** on the geomean, with one tenth the draws and
  the cheaper model.
- **Run 1 was the worst arm**, 3.3× behind the Opus one-shot it was being compared against. That is
  the state [01](01-what-was-measured.md) measured.
- **Three draws of Opus ≈ ten draws of Opus** (0.0484 vs 0.0443, 1.09×). Seven more draws bought 9%,
  which is what the duplicate-answer mechanism below predicts.
- **Sonnet one-shot is the weakest non-pipeline arm**, 2.0× behind Opus one-shot at the same draw
  count — a real model effect, and the reason run 2 and run 3's Sonnet numbers are the interesting
  ones.

## The duplicate-answer mechanism: why independent runs return the same number

Several cells repeat across rows and across columns. B04 reads **0.0603 in four of the six arms** —
Opus×10, Sonnet×3, Opus×3 and AutoNumerics run 1 — and B05 reads **0.0391** for both Opus×10 and
Sonnet×3. These are not coincidences, and they are not evidence that the arms are equally good.

### They are not exactly equal — they agree to 8–12 significant figures

Carried to full precision ([`data/duplicate_answers_exact.py`](data/duplicate_answers_exact.py)),
**no two cells in the table are bitwise equal**:

| problem | arm | gate units, full precision | agreement |
|---|---|---|---|
| B04 | Opus ×10 | 0.060328675637088817 | |
| B04 | Sonnet ×3 | 0.06032867563731055 | 12 sig figs |
| B04 | Opus ×3 | 0.060328675637133164 | 12 sig figs |
| B04 | AutoNumerics run 1 | 0.060328675640813775 | 10 sig figs |
| B05 | Opus ×10 | 0.039109992708609438 | |
| B05 | Sonnet ×3 | 0.03910999211121563 | 8 sig figs |

A 12-digit agreement between two separately written programs is not luck. Nothing in Monte Carlo
produces it by chance: two *independent* samples of 400 000 paths would differ in the third or
fourth digit, which is exactly what the noise bars in § The bars say. Agreement at the twelfth digit
means the two programs computed **the same estimator on the same sample**, and differ only in
floating-point rounding.

### Why that happens: the grading is deterministic and the problems have canonical answers

Four facts compose:

1. **Nothing about the evaluation is random.** The solver contract fixes `num_paths`, `dt` *and*
   `seed` — on B04, 400 000 paths, `dt = 0.002`, `seed = 42`, identical in every arm. The harness
   does not reseed per draw or per arm. So the "Monte Carlo" in this benchmark is **common random
   numbers**: every solver is handed the same stream.
2. **A solver's answer is a pure function of its RNG consumption pattern.** Both of the B04 solvers
   below call `np.random.default_rng(seed)` and then, inside the time loop, draw exactly two normal
   vectors of length `n` in the same order — the variance innovation, then the price innovation. Same
   seed plus same consumption order means **the same pseudo-random numbers land in the same places**.
   The paths are not merely statistically similar; they are the same paths.
3. **These problems have a dominant canonical scheme.** B04 and B09 are Heston double-no-touch:
   Andersen QE (or an exact non-central-χ² transition) for the variance, a log-price step, and a
   Brownian-bridge continuous-monitoring correction. That is the textbook answer, and a competent
   model writes it — whether it is Opus with no manual, Sonnet with the manual, or the pipeline's
   planner. The models are not converging on a number; they are converging on an *algorithm*, and
   the algorithm plus the fixed stream determines the number.
4. **The last digits differ because the floating-point order does.** The programs are not the same
   program. For B04, Sonnet's `c0-manual/rep3` and Opus's `c0-naked/rep1` share no variable names
   (`kappa, eta, sig, rho` against `kappa`/`theta`/`sigma`/`rho`), spell the barriers differently,
   and group the QE constants differently. Summing 400 000 doubles in a different association order
   perturbs the mean in the last two or three ULPs — hence 12 digits of agreement and not 16.

The clearest instance is **within** the new Opus×3 arm, on B09: reps 2 and 3 are visibly different
programs — rep 2 uses a trapezoidal integrated-variance log-S step and computes survival as
`clip(1 - p_up - p_down)`; rep 3 uses Andersen's `K0..K4` form and computes `(1 - p_up)(1 - p_down)`.
Both forms are algebraically the same thing at `g1 = g2 = 0.5`, and `p_up · p_down` underflows for
essentially every path, so the two programs agree to 14 significant figures:

```
rep 2  no_touch_S75 = 0.7913035174062937   (0x1.9525bc10f0896p-1)
rep 3  no_touch_S75 = 0.7913035174062923   (0x1.9525bc10f088ap-1)
```

### Why it matters for every number in this folder

- **Repeated draws are not replications.** [01](01-what-was-measured.md) §0 already counted the
  distinct answers in the Opus×10 arm: **B09 1 of 10, B04 2 of 10**, B01 6, B05 6, B08 8. On B09
  "best of ten" is best of *one*. The new Opus×3 arm repeats the pattern — B04 1 of 3 distinct,
  B09 2 of 3.
- **A cross-arm tie on such a problem measures nothing about the models.** B04's four-way 0.0603 says
  that four arms each found the canonical scheme. It does not say they are equally capable, and
  averaging it into a geomean as if it were four independent measurements overstates the agreement.
- **It is also why seed replication is the right instrument.**
  [08](08-free-measurements.md) §3 re-solved at further seeds for exactly this reason: under common
  random numbers, a single-seed difference between two solvers can be pure sampling luck, while a
  single-seed *agreement* to 12 digits is near-proof that they are the same scheme.
- **The effect is strongest where the answer is most constrained.** The problems with many distinct
  answers (B02, B06, B07, B10 — 10 of 10 in the Opus arm) are the ones where the scheme is genuinely
  open, and those are where the arms actually separate. Read the table's spread there, not on B04.

## Reproducibility: are the winning solvers still on disk?

**Audited 2026-11-09**, [`data/solver_availability_audit.py`](data/solver_availability_audit.py).
For each arm, whether the `solver.py` behind that arm's *best* result on each problem still exists.

| arm | best-result solvers recoverable | where | why |
|---|---|---|---|
| Opus one-shot, best of 10 | **0 of 10** | — | `workspace_oneshot/opus/` was overwritten by the 2026-11-09 run |
| Sonnet one-shot, best of 3 | **10 of 10** | `workspace_oneshot/sonnet/<slug>/c0-manual/rep<n>/` | untouched since 2026-11-02 |
| Opus one-shot, best of 3 | **10 of 10** | `workspace_oneshot/opus/<slug>/c0-naked/rep<n>/` | written 2026-11-09 |
| AutoNumerics run 1 | **0 of 10** | — | no workspace snapshot was ever archived |
| AutoNumerics run 2 | **9 of 9** | `results/pipeline/archive_sonnet_other9_2026-11-05/workspace_snapshot/<slug>/plans/<plan>/` | snapshotted before the round-3 run |
| AutoNumerics run 3 | **10 of 10** | `workspace/<slug>/plans/<plan>/` | live, and each carries Fix 0's `evaluation.json` |

**The 2026-11-09 Opus run destroyed the 2026-10-25 Opus baseline's solvers.** `oneshot.py` writes to
`workspace_oneshot/<model>/<slug>/c0-<manuals>/rep<n>`, a path that depends on model, condition and
rep but **not on the run**, so three naked Opus draws landed exactly on top of the old arm's
`c0-naked/rep1–3` and the old `c0-manual/rep1–5` directories are gone as well. The audit script
reports 3 of 10 "present" for that arm; all three are this collision, and two of them hold a
visibly different solver (B06 0.1445 recorded against 0.4847 for the file now at that path, B07
0.0666 against 0.1097). The Oct-25 arm now exists **only** as the graded numbers in `s1–s4.json`.
B09 is the exception that proves the duplicate-answer point: the file at that path is today's, yet
it scores 0.224267076, the same as the Oct-25 draw, because both wrote the canonical scheme.

### The surviving ones re-run bitwise

Two spot-checks through the same sandboxed path that graded them
([`data/reverify_spotcheck.py`](data/reverify_spotcheck.py)):

```
uv run python benchmark/verify.py --slug sde_heston_dnt \
  --plan-dir workspace/sde_heston_dnt/plans/1-exact-chi2-bridge --no-work-precision
```

| unit | recorded gate units | recomputed 2026-11-09 | equal |
|---|---|---|---|
| AutoNumerics run 3, B04 best plan `1-exact-chi2-bridge` | 0.025491186346961462 | 0.025491186346961462 | **bitwise** |
| Sonnet one-shot, B04 rep 3 | 0.06032867563731055 | 0.06032867563731055 | **bitwise** |

Bitwise agreement is the expected result, not a lucky one, and for the same reason the arms collide
on B04: fixed `num_paths`, `dt` and `seed` make the whole grading path deterministic. It does
confirm that the recorded numbers are reproducible and that re-verification is cheap — ~20–38 s per
solve.

The B04 plan was then re-verified a **third** time, by hand, on the same machine. Every graded
quantity repeated to the printed precision — all three functional errors, the two-level delta, both
work-precision errors — and **only the two wall-clock figures moved**: 35.66 s → 37.99 s of solve
time and 7.795 s → 7.963 s of sweep. That is the line to hold in mind when reading any cost figure
in this folder: the accuracy numbers are deterministic, the timings are machine load and drift ~6%
between runs on an idle box. `cost` is out of `ranking.WEIGHTS` and out of `rank_key`'s tiebreak as
of round 3, but anything still reading `solve_s` is reading that noise.

**One caveat on method.** A `verify.py --plan-dir` run does **not** rewrite the plan's
`verify.json`; the one in a one-shot rep directory is the original from its own run. Reading that
file back after a re-run looks like a confirmation and is not one. The spot-check above captures
`verify_problem`'s return value directly.

### What to do before the next run

`workspace/` and `workspace_oneshot/` are both gitignored and both are clobbered by the next run
that selects the same problem — `--fresh` for the pipeline, the model/condition/rep path for
one-shot. The two arms that are currently live and unarchived are **AutoNumerics run 3** and
**Opus one-shot ×3**. Copy them somewhere outside the repo, or into an archive directory under
`results/pipeline/`, before running either arm again.
