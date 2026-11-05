# 10 — The Sonnet retest

**Status:** complete record, 2026-11-02. Ten BENCHOP-SLV problems, full pipeline, all seven agents
pinned to `--model sonnet`. Supersedes nothing in [01-what-was-measured.md](01-what-was-measured.md);
that file stays the record of the C0 and pre-change C2 arms and every baseline number below is quoted
from it.

## §0 Provenance of the numbers

| Arm | What | Records |
|---|---|---|
| **C0** | Opus one-shot, no tools, 10 draws per problem. `$16.86`, mean wall 48 s per draw | `benchmark/results/benchop-step10/s1.json`–`s4.json` |
| **C2 pre-change** | Full pipeline before this folder's fixes, formulator at Opus | `p_regrade123.json`, `p_b04.json`, `p_regrade4.json`, `p_b0506.json`, `p_b0708.json`, `p_b0910.json` |
| **Now** | Current `main` + the uncommitted formulator trim, `--model sonnet` | `benchmark/results/pipeline/benchop_sonnet.json` (nine) + `benchop_b03_sonnet_run3.json` (B03) |

Per-plan kernel metrics are read from the `<metrics>` blocks in
`workspace/{slug}/plans/{plan}/SOLUTION.md`, **not** from the `kernel` dicts in the result JSONs.
The latter are summaries with fields dropped; `rank_key` computed on them gives a different and wrong
answer. Anyone re-deriving §3 must use the `<metrics>` blocks.

The analysis scripts are beside the records in `benchmark/results/benchop-step10/`:
`c0draws.py` (per-draw gate units), `bo3.py` (the best-of-three subsets in §2.2), `rankcheck.py`
(the §3 rank recomputation). **That directory and `workspace/` are both gitignored** and the credits
that produced the records are spent — back them up outside the repo.

### The run is not homogeneous

Two inhomogeneities, both real, neither large enough to invalidate the arm:

- **B01, B02 and B03 ran with the pre-trim formulator.** B03 at 12:23 and B01/B02 at 18:24–18:57 on
  Nov 1; the `agents/formulator.md` trim landed at 19:16 and B04–B10 ran after it. B01 and B02 were
  carried into the final results file by `--resume`, not re-run. The recorded config hash
  `d859234297741d97` is the sha256 of the **trimmed** file as pinned to `model: sonnet`, confirming
  the trim was live for B04–B10.
- **The arm changes the model and the code together.** Every comparison against C2 pre-change is
  Opus→Sonnet *and* the fix set, and nothing here can separate them.

## §1 Results

All ten `completed`, all ten `UNDERCLAIM`, all ten harness-passed, **0 of 10 certified**.
Total wall **237.7 min ≈ 4.0 h**.

| id | slug | shipped | pool best | kernel score | provenance | tier | wall (min) |
|---|---|---|---|---|---|---|---|
| B01 | `sde_sabr_cev_call_i` | 0.1108 | 0.0473 | 9 | `surrogate` | A′ | 9.8 |
| B02 | `sde_sabr_cev_call_ii` | 0.1674 | 0.1674 | 2 | `none` | C | 19.3 |
| B03 | `sde_heston_call_feller` | 0.4084 | 0.4084 | 6 | `analytic_unvalidated` | A\_unavailable | 53.0 |
| B04 | `sde_heston_dnt` | 0.1353 | 0.1353 | 9 | `surrogate` | A′ | 34.7 |
| B05 | `sde_qlsv_dnt` | 0.1006 | 0.1006 | 7 | `self_convergence` | C | 40.9 |
| B06 | `sde_hhw_call_corr` | 0.7445 | **0.0474** | 3 | `self_convergence` | C | 19.0 |
| B07 | `sde_hhw_call_zerocorr` | 0.0780 | 0.0388 | 6 | `surrogate` | A′ | 23.3 |
| B08 | `sde_heston_call_hivar` | 0.1543 | 0.1543 | 7 | `self_convergence` | C | 9.4 |
| B09 | `sde_heston_dnt_hivar` | 0.1175 | 0.1175 | 7 | `self_convergence` | C | 11.2 |
| B10 | `sde_sabr_cev_call_ii_lowvol` | 0.1783 | 0.1612 | 2 | `none` | C | 17.1 |

Error is in **gate units**: max over the problem's scored functionals of
(relative error ÷ that functional's own gate). Lower is better, 1.0 is the pass line. The gates are
not uniform within a problem — B03's three strikes are gated at 0.015, 0.004 and 0.002 — so a
per-functional breakdown is sometimes needed and §4 gives it.

## §2 Comparison

### §2.1 Against the recorded baselines

| id | C0 best/10 | C0 median | C2 pre-change | now | vs C2 | vs C0 median |
|---|---|---|---|---|---|---|
| B01 | 0.0180 | 0.0737 | 0.2584 | 0.1108 | 2.33× | 1.50× |
| B02 | 0.0116 | 0.1658 | 0.9594 | 0.1674 | 5.73× | 1.01× |
| B03 | 0.0178 | 0.1265 | 0.4606 | 0.4084 | 1.13× | 3.23× |
| B04 | 0.0603 | 0.0603 | 0.3121 | 0.1353 | 2.31× | 2.24× |
| B05 | 0.0391 | 0.2025 | 0.1936 | 0.1006 | 1.93× | 0.50× |
| B06 | 0.1445 | 0.3259 | 0.5177 | 0.7445 | 0.70× | 2.28× |
| B07 | 0.0666 | 0.1232 | 0.2952 | 0.0780 | 3.78× | 0.63× |
| B08 | 0.0292 | 0.1430 | 0.3836 | 0.1543 | 2.49× | 1.08× |
| B09 | 0.2243 | 0.2243 | 0.3017 | 0.1175 | 2.57× | 0.52× |
| B10 | 0.0528 | 0.1825 | 0.4250 | 0.1783 | 2.38× | 0.98× |

Nine of ten improved against C2 pre-change; geometric mean **2.20×**, on the cheaper model.

### §2.2 The matched comparison: C0 best-of-three

The pipeline builds and grades **exactly three plans** on every one of the ten problems, then ships
one. The honest matched control is therefore **three Opus draws with the best taken**, not one draw.
Computed exactly over all 120 three-subsets of each problem's ten draws:

| id | C0 best-of-3, median | now shipped | now pool best | P(shipped ≤ bo3) |
|---|---|---|---|---|
| B01 | 0.0200 | 0.1108 | 0.0473 | 0.03 |
| B02 | 0.0682 | 0.1674 | 0.1674 | 0.08 |
| B03 | 0.0392 | 0.4084 | 0.4084 | 0.00 |
| B04 | 0.0603 | 0.1353 | 0.1353 | 0.00 |
| B05 | 0.0391 | 0.1006 | 0.1006 | 0.29 |
| B06 | 0.2271 | 0.7445 | 0.0474 | 0.00 |
| B07 | 0.0799 | 0.0780 | 0.0388 | 0.70 |
| B08 | 0.0402 | 0.1543 | 0.1543 | 0.03 |
| B09 | 0.2243 | 0.1175 | 0.1175 | 1.00 |
| B10 | 0.0692 | 0.1783 | 0.1612 | 0.08 |

| geometric mean, ten problems | gate units |
|---|---|
| C0 best-of-10 (oracle) | 0.0443 |
| **C0 best-of-3 (oracle), median subset** | **0.0654** |
| C0 best-of-3 (oracle), mean subset | 0.0910 |
| C0 single draw, median | 0.1461 |
| **now, pool best (oracle)** | **0.1090** |
| **now, shipped (blind)** | **0.1693** |

The two comparisons that are actually like-for-like:

- **Oracle against oracle — the pools.** 0.1090 vs 0.0654: **three Opus draws contain a better
  solver than the pipeline's three plans, by 1.67×.** The pipeline beats the matched pool on 3 of 10.
  **Superseded by [§7](#7-the-model-matched-control-c0-at-sonnet-three-draws):** this arm's control
  runs on Opus while the pipeline runs on Sonnet. At matched model the gap is 1.11× over all ten and
  1.75× over the eight where naked draws are reliable — so the 1.67× is part model, part generation.
- **Blind against blind — no selector either side.** 0.1693 vs 0.1461: **1.16×, a tie.** Shipping
  the pipeline's choice is worth about the same as taking one Opus draw and not looking at it.

Cost, to put the control in proportion: three C0 draws cost **$5.06 and 24 min** of agent wall across
the ten problems. The pipeline cost 237.7 min, roughly 10×.

## §3 The four selection misses, with mechanisms

`verifylib.ranking.rank_key` recomputed on the `<metrics>` blocks reproduces **the plan that actually
shipped on 10 of 10**, so the selection is fully explained by that function. It picks the pool's most
accurate plan on 6 of 10. The four it misses:

| id | miss | provenance across pool | first key that separated | cause |
|---|---|---|---|---|
| B06 | **15.7×** | uniform `self_convergence` | `sqi` | `order`, two-sided against a defaulted floor |
| B01 | 2.3× | uniform `surrogate` | `sqi` | `accuracy` component inverted |
| B07 | 2.0× | uniform `surrogate` | `score` | score 6 beats score 3; the 3 is 2× more accurate |
| B10 | 1.1× | uniform `none` | `sqi` | `accuracy` component inverted |

**Provenance separated no pool in this run.** On all ten problems it was either uniform across the
three plans or perfectly co-linear with `score`, and in all four misses it was uniform. It caused
none of them.

### §3.1 B06 — the 15.7× miss, in full

All three plans are scored identically by the first two rank keys: `score: 3`, provenance
`self_convergence`, `certified: false`, all three capped by "a gate:true structural violation". The
ranking therefore falls entirely to the quality index, and the index is almost entirely
**unmeasurable** here:

| component | weight | plan 1 (EM-FT antithetic) | plan 2 (QE + CV) | plan 3 (EM-FT + CV) |
|---|---|---|---|---|
| `accuracy` | 0.35 | **None** | **None** | **None** |
| `order` | 0.25 | 0.0000 | **0.0152** | 0.0000 |
| `structure` | 0.20 | None | 0.0 | 0.0 |
| `consistency` | 0.10 | None | None | None |
| **sqi** | | 0.0000 | **0.0084** | 0.0000 |
| harness error | | **0.0474** | **0.7445** ← shipped | 0.5017 |

`accuracy` returns `None` on all three through `noise_limited`: `estimated_rel_error` is ≈3e-06 while
`mc_se_rel` is ≈0.03, so the reported error sits four decades inside its own error bar and the
component correctly declines to read it. That removes the only component correlated with the true
error. `structure` and `consistency` abstain or tie at zero.

**So the entire ranking of B06 was decided by the `order` component**, and inside it by
`observed_order` 0.9924 against 1.0078 — a 1.5% difference between two noisy Monte-Carlo convergence
slopes. The plan with the *lower* observed order won, because `order` is **two-sided** against
`order_floor`, and the floor is **0.5** while all three schemes have true strong order **1**:

```
plan 1: 1 - |1.0078 - 0.5| / 0.5 = -0.0156  ->  clipped to 0.0000
plan 2: 1 - |0.9924 - 0.5| / 0.5 =  0.0152
plan 3: 1 - |1.0078 - 0.5| / 0.5 = -0.0156  ->  clipped to 0.0000
```

Converging *better* than the floor is scored as a defect exactly as hard as falling short of it. The
code already knows this: `ranking.order`'s one-sided branch carries the comment *"a correct
additive-noise Euler-Maruyama has strong order 1 and a defaulted floor of 0.5, so the two-sided
distance scored it 0.028 — last in its pool, on a right answer."* That branch is gated on
`order_band: one-sided`, and **`order_band` is declared on 0 of the 40+ plans in `workspace/`.** The
fix exists and has never once executed.

This is the first measurement of **F16** and **F17** ([09](09-implementation.md) §21–22) in
action, and it shows the pair is incomplete. F16 made `accuracy` return `None` rather than a
saturated 1.0 where the error is inside its own error bar — correct, and it is what emptied 0.35 of
the index here, leaving `order` alone to decide. F17 made the SDE driver publish a `design_order`
from a declared `scheme_family` instead of the kernel's 0.5 default — but `ranking.order` computes
its SDE target as `order_floor or design`, so it reads the **floor** first and F17's number is never
consulted. On B06 both are 0.5 in any case: Euler–Maruyama's strong order under multiplicative noise
genuinely *is* 0.5, and the schemes measured ~1.0. So F17 cannot rescue this even once the ranker
honours it. The defect is the two-sidedness itself, exactly as `ranking.order`'s own comment says.

Two further notes on B06. `invariant_max_drift` orders the three plans **perfectly** against the true
error — 0.786 / 0.8365 / 0.8549 against 0.0474 / 0.5017 / 0.7445 — and contributed nothing, because
`structure` abstained on plan 1 and tied at 0.0 on the other two. And the winning plan is the one
with the control variate: a smoothly-converging biased scheme looks better under self-convergence
than a noisier unbiased one, which is the failure mode self-convergence has by construction.

### §3.2 B01 and B10 — the accuracy component inverted

Both pools tie on score and provenance, both fall to the index, and in both the index is the
`accuracy` component alone (every other component `None`). In both it ranks backwards:

| id | shipped | sqi | error | pool best | sqi | error |
|---|---|---|---|---|---|---|
| B01 | `1-exact-besq-time-change` | 0.2264 | 0.1108 | `2-em-absorbing-antithetic` | 0.1408 | 0.0473 |
| B10 | `3-em-antithetic-martingale-cv` | 0.1477 | 0.1783 | `2-poisson-gamma-cev-step` | 0.1053 | 0.1612 |

B01 also shows the score/index tension inside one pool: plan 3 has by far the highest index (0.4690)
and lost on `score` 8 < 9.

### §3.3 B07 — score, working as designed, costing accuracy

`1-exact-chi2-gaussian-rate` scores 6; `3-em-control-variate-antithetic` scores 3 and is **2.0× more
accurate**. Score is the first key after scorability, so the 6 ships. Nothing is miscalibrated here:
`score` measures how much was *established* about a plan, and an exact-transition scheme establishes
more than an EM-plus-control-variate one. This miss is architectural, not a bug.

## §4 B03 — why the pipeline is worse than C0 here

B03 is the only problem where the pipeline is clearly behind C0's *median single draw* (3.23×), and
the only one where its pool best equals its shipped value — selection was correct and the pool was
simply bad. The cause is not resolution and not selection.

**Both arms ran the identical budget.** C0's best draw and the pipeline's winner were both graded at
`num_paths: 2000000, dt: 0.002, T: 1.0, seed: 42`. Same grid, same paths, same seed. The difference
is entirely scheme quality.

Per-functional gate units, all three plans:

| plan | `price_S75` (gate .015) | `price_S100` (.004) | `price_S125` (.002) | max = scored |
|---|---|---|---|---|
| 1 EM full-truncation ← shipped | 0.2679 | **0.4084** | 0.3792 | **0.4084** |
| 2 Andersen QE | **0.4217** | 0.0881 | 0.0224 | 0.4217 |
| 3 EM antithetic | **0.5327** | 0.4685 | 0.3713 | 0.5327 |
| C0 best draw | 0.0072 | 0.0073 | 0.0180 | **0.0178** |

C0's best draw is uniformly good; every pipeline plan is bad on at least one strike. Its solver
combines three techniques **no plan in the pool used**:

1. **Exact noncentral-χ² CIR transition.** `rng.noncentral_chisquare` sampled on the grid, with zero
   discretisation bias in V and valid at and through V = 0 — which is precisely what this problem's
   violated Feller condition demands. The pool used full-truncation Euler (biased), Andersen QE (an
   *approximation* to that same noncentral χ²), and EM antithetic.
2. **Conditional Monte Carlo on the asset leg.** Having sampled the variance path it does not
   simulate S at all; it integrates S out in closed form with a Black–Scholes call on the conditional
   forward, `_bs_call(F, K, sv)`. This removes the entire S-leg sampling variance, which is the
   dominant error on the thin out-of-the-money payoff. C0's `price_S75` error is 1.08e-04 against the
   pool's best 4.02e-03 — **37×**.
3. **A martingale control variate** regressed per spot on `F − E[F]`, exploiting `E[F] = S_0`.

The pipeline's own work-precision table confirms the diagnosis: going from 500k to 2M paths made the
error **worse**, 0.003647 → 0.004019. The estimator is bias-dominated, so paths cannot help — and
that is exactly the regime conditional Monte Carlo removes.

Two things follow. **This is a plan-generation failure, not a selection or a resolution failure** —
no ranker can recover a technique absent from all three plans, and the pool's ceiling even with
per-functional cherry-picking is 0.2679, still 15× behind C0's best draw. And **the metric being a
max over functionals makes diversity cheap to exploit**: plan 1's S75 with plan 2's S100 and S125
would score 0.2679, a 1.52× gain from solvers already built and already graded.

## §5 Findings

**F1 — Accuracy improved 2.20× against the pre-change pipeline, on the cheaper model.** Nine of ten
problems improved. This is the one unambiguous win in the run.

**F2 — Against a matched control the pipeline is behind, and the previous report's "level with C0"
was too generous.** *(Revised by [§7](#7-the-model-matched-control-c0-at-sonnet-three-draws): the
control here is Opus against a Sonnet pipeline. Model-matched, the pool comparison is a tie at 1.11×
and the blind comparison reverses to a 2.4× pipeline win. What survives is the 1.75× pool gap on the
eight problems where naked Sonnet draws do not blow up.)* Comparing a three-plan system to a single draw flattered it. Matched
oracle-to-oracle, three Opus draws contain a better solver than three pipeline plans by **1.67×**
(§2.2). Blind-to-blind the pipeline ties a single unexamined draw at 1.16×, for ~10× the wall clock.
Part of the 1.67× is Opus-vs-Sonnet and this run cannot size that part.

**F3 — Self-assessment moved the wrong way while accuracy improved.** Mean kernel score **7.1 → 5.8**;
`analytic` provenance **5/10 → 0/10** (B01, B03, B04, B08, B09 all lost validated closed-form
references). The trim is exonerated: analytic was lost on *both* sides of the trim boundary. The
remaining difference is the model. Confirming that is the cheapest high-value experiment available —
re-run two or three of the five regressed problems with the formulator alone pinned to Opus.

**F4 — The whole selector rests on one miscalibrated component.** §3: `rank_key` reproduces the
shipped plan 10/10 and misses the best 4/10. `provenance` separated no pool. `certified` is not a
rank key at all. Of the four misses, one is `score` working as designed (B07) and **three are the
quality index**, in which `accuracy` — the only component correlated with truth — abstained on B06
and inverted on B01 and B10.

**F5 — `order`'s two-sided penalty is the single largest recoverable defect, and its fix is dead
code.** It cost 15.7× on B06 by rewarding the worse-converging plan. `order_band: one-sided` is
declared on **0 of 40+ plans**, so the branch written to prevent exactly this has never run.

**F6 — `invariant_max_drift` ranked B06 perfectly and was ignored.** 0.786 / 0.8365 / 0.8549 against
0.0474 / 0.5017 / 0.7445, monotone. `structure` abstained on the plan that reported no
`invariant_worst_headroom` and tied the other two at 0.0.

**F7 — Plan-generation diversity, not ranking, is B03's problem, and probably the suite's ceiling.**
*(Downgraded by [§7](#7-the-model-matched-control-c0-at-sonnet-three-draws): "the suite's ceiling" is
not supported once the model is matched — selection accounts for essentially the whole remaining gap.
The B03 half stands and strengthens: one naked Sonnet draw beats the entire pipeline pool there by
7.1×.)*
No exact-transition-plus-conditional-MC plan was proposed on B03; the pool's ceiling is 15× behind
one Opus draw (§4). The matched-pool result in F2 says the same thing suite-wide: the pipeline's
weakness is what it *builds*, not which one it picks.

**F8 — The kernel certifies on the wrong quantity.** `resolved: false` on **8 of 10**, `mc_se_rel`
up to 0.9923. Certification reads *state moments*, which are badly conditioned on this suite, while
the suite *scores* option functionals, which are well conditioned — B01's prices carry SEs ≈6e-5 and
the kernel called them clean at 5.7e-04 against a 4e-03 tolerance. B01 shows both inside one plan.

**F9 — 0/10 certified, 10/10 harness-passed, all ten UNDERCLAIM.** Unchanged from pre-change. Six of
ten sit at tier C or `A_unavailable`, where 10 is unreachable by construction.

**F10 — `graded_num_paths` is now declared on 8 of 10** (B04/B09 legitimately equal to `num_paths`),
so the formulator learned that rule. The kernel still grades diagnostics at the reduced count while
the harness runs at 400k–4M.

**F11 — The order study looks biased where it does run.** `observed_order` 0.2688, 0.2700, 0.2708,
0.279, 0.321, 0.3226 against a 0.5 floor, all on harness-passing solvers. Six correct solvers missing
their rate by the same ~40% is an estimator artefact.

**F12 — B01's generator-check failure collapses three causes into one string** — ladder memory
budget, generator not buildable from the declared drift and diffusion, or the observables run
crashing — so the reason it stalled at 9 is not diagnosable from the record.

## §6 Suggested fixes, in measured-leverage order

Counterfactuals, geometric mean over the ten, recomputed from the pools:

| if selection were fixed on | shipped geomean | gain | vs C0 single draw |
|---|---|---|---|
| nothing (as shipped) | 0.1693 | — | 1.16× behind |
| **B06 only** | **0.1285** | **1.32×** | **0.88× — ahead** |
| B06 + B01 | 0.1181 | 1.43× | 0.81× |
| B06 + B01 + B10 | 0.1169 | 1.45× | 0.80× |
| all four (perfect selection) | 0.1090 | 1.55× | 0.75× |

### Fix 1 — make `order` one-sided whenever the floor was defaulted. *(1.32× alone)*

The cheapest change in this document and the largest single gain. Either default `order_band` to
`one-sided` when `order_floor` was defaulted rather than declared by the plan, or require the
formulator to emit `order_band` and treat its absence as `order: None`. Both are small; the first
needs no agent change and fixes the 40+ plans already on disk. Recovers B06 outright and flips the
suite to beating a single Opus draw.

### Fix 2 — stop the quality index ranking on a single abstaining component.

Where only one component is measurable, the index is that component and renormalisation hides it.
B06 ranked on `order` alone; B01 and B10 on `accuracy` alone, inverted; B10's raw summary saturates
`accuracy` at 1.000 for all three plans. Two concrete moves:

- **Require a minimum measured weight.** If measurable components carry less than, say, half the
  total weight, the index is not a ranking — return `None` and let `decided_by` report `tie` rather
  than ship a coin flip read off Monte-Carlo noise.
- **Admit `invariant_max_drift` into `structure` as a bare value**, not only via
  `invariant_worst_headroom`. It ranked B06 perfectly (F6) and abstained.

### Fix 3 — grade accuracy on the declared functional, not on state moments. *(F8)*

`_functional_fraction` already prefers `functional_rel_error` where the kernel writes it. It is not
being written on these problems, so `accuracy` falls through to the moment error and then abstains
via `noise_limited` — which is how B06's 0.35 of weight vanished. Writing `functional_rel_error` and
`functional_rel_tol` whenever the spec declares a gated functional fixes the accuracy component and
the certification ceiling in the same change, and it is the one fix that serves both F4 and F8.

### Fix 4 — invest in plan diversity. *(F7, the suite's ceiling)*

The matched-pool result (F2) and B03 (§4) both say generation is the binding constraint. Two
tractable asks of the plan creators, both drawn from what C0's winning draws actually did:

- **Require one variance-reduction-by-conditioning plan where the payoff admits one.** Conditional
  Monte Carlo / mixing closed forms were absent from all 30 plans and present in C0's best B03 draw.
- **Require one exact-transition plan where the state has a known transition law.** B03's pool had
  none despite a CIR variance factor; B08's did.

### Fix 5 — exploit the max-over-functionals metric. *(1.52× on B03 alone, free)*

Nothing in the harness requires one solver to answer every functional. Per-functional best-of-pool
would take B03 from 0.4084 to 0.2679 from solvers already built and graded. This is a scoring and
assembly change, not a solver change. Worth deciding deliberately: it may be outside the spirit of
"ship one plan", and if so it should be ruled out in writing rather than left unnoticed.

### Fix 6 — reporting and hygiene, no accuracy effect.

- Split B01's three collapsed generator-failure causes into distinct messages (F12).
- Close `reference.py`'s `_moment_observables` pairing gap: the variance half is unambiguously
  derivable, the mean half needs a decision on whether to require `mean_X_from` spellings.
- Harden `override_agent_models` to checksum the original before restoring; it restored agent files
  at 05:33 on Nov 2 and nearly lost the formulator trim.
- Investigate the `observed_order` bias in F11 before trusting any order-based cap.

### What will *not* improve accuracy

Asked directly, and worth recording because both look like accuracy work and are not:

- **Certification.** `certified` does not appear in `rank_key`. Fixing it cannot change which plan
  ships. It changes the number reported next to the answer. Worth doing for honest reporting; it has
  exactly zero accuracy leverage.
- **Provenance.** It is a rank key, but it separated no pool in this run and caused none of the four
  misses. Its indirect route to accuracy is through `score`, whose ceiling it sets — and `score`
  chose wrong on exactly one problem (B07), where the better-evidenced plan was already the one with
  the higher score. Raising provenance would move scores in an uncontrolled direction. The reason to
  fix provenance is F3 — the pipeline not knowing how good its own answers are — which is a real
  problem and a different problem.

## §7 The model-matched control: C0 at Sonnet, three draws

Added 2026-11-02, after §1–§6. §2.2 compared a Sonnet pipeline against an **Opus** one-shot control
and could not separate generation quality from the model. This arm removes that confound: same model,
same three attempts, same knowledge setting.

**Run:** `oneshot.py --suite benchop --condition c0 --manuals manual --reps 3 --model sonnet
--c0-max-turns 8 --agent-timeout 1800 --timeout 1800 --mem-mb 4096`, results in
`benchmark/results/baselines/oneshot_benchop_c0_sonnet.json`, analysis in `c0sonnet.py` beside it.
30 units, **26 PASS / 4 FAIL / 0 CRASHED**, $2.99, 57.7 min + a 16.6 min recovery pass.

Two notes on comparability. `--manuals manual` matches the pipeline's knowledge, since the formulator
reads `references/sde_manual.md`; the Opus C0 baseline pooled 5 `naked` + 5 `manual`, so this arm
matches the pipeline but is not a clean model-swap against §2.2's numbers. And six units were lost to
a transient `ENOTFOUND` DNS outage on the first pass — recovered with `--resume --redo CRASHED`, all
six then passing or failing on their merits. They are not a Sonnet reliability signal; the four
`FAIL`s are.

| id | three draws (gate units) | best-of-3 | pipeline pool | pipeline ship | pool winner |
|---|---|---|---|---|---|
| B01 | 0.0180 / 0.0695 / 0.0434 | **0.0180** | 0.0473 | 0.1108 | C0 2.6× |
| B02 | **39.3 FAIL** / 0.4838 / 0.4596 | 0.4596 | 0.1674 | 0.1674 | pipeline 2.7× |
| B03 | 0.0579 / 0.6010 / 0.6368 | **0.0579** | 0.4084 | 0.4084 | C0 7.1× |
| B04 | 0.3412 / 0.4145 / 0.0603 | **0.0603** | 0.1353 | 0.1353 | C0 2.2× |
| B05 | 0.1823 / 0.0391 / 0.0554 | **0.0391** | 0.1006 | 0.1006 | C0 2.6× |
| B06 | **129.6 FAIL** / 0.5199 / **213.1 FAIL** | 0.5199 | 0.0474 | 0.7445 | pipeline 11.0× |
| B07 | 0.5166 / 0.0659 / **9.88 FAIL** | 0.0659 | 0.0388 | 0.0780 | pipeline 1.7× |
| B08 | 0.6510 / 0.2537 / 0.2417 | 0.2417 | 0.1543 | 0.1543 | pipeline 1.6× |
| B09 | 0.0782 / 0.2236 / 0.2550 | **0.0782** | 0.1175 | 0.1175 | C0 1.5× |
| B10 | 0.1106 / 0.2913 / 0.1399 | **0.1106** | 0.1612 | 0.1783 | C0 1.5× |

| geometric mean | all ten | excl. B06 | excl. B06, B02 |
|---|---|---|---|
| C0@Sonnet best-of-3 (oracle) | 0.0979 | 0.0813 | 0.0655 |
| C0@Sonnet median single draw | 0.4077 | 0.2149 | 0.1942 |
| pipeline pool best (oracle) | 0.1090 | 0.1196 | 0.1146 |
| pipeline shipped (blind) | 0.1693 | 0.1436 | 0.1409 |
| **pool ÷ C0 best-of-3** | **1.11×** | **1.47×** | **1.75×** |
| **shipped ÷ C0 median draw** | **0.42×** | 0.67× | 0.73× |
| C0@Sonnet bo3 ÷ C0@Opus bo3 | 1.50× | 1.43× | 1.18× |

### What this changes

**§2's headline claim is wrong as stated, and the correction cuts both ways.** §2.2 concluded that
"three Opus draws contain a better solver than the pipeline's three plans, by 1.67×" and F7 read that
as plan generation being the binding constraint. At matched model the pool gap collapses to **1.11×**,
a tie, and the pipeline wins the pool comparison on 4 of 10. So most of that 1.67× was the model, not
generation quality.

**But the tie is bought by robustness, not by better solvers.** The 1.11× is carried almost entirely
by B06, where 2 of 3 naked draws blow up (129.6 and 213.1 gate units) and the pipeline's pool is 11×
better. Drop B06 and the pool gap returns to 1.47×; drop B02 as well — the other problem where a
naked draw fails outright — and it is **1.75×**, statistically indistinguishable from the 1.67× the
Opus arm showed. The honest statement is therefore two-part:

- **On the eight problems a naked Sonnet draw handles reliably, C0's three draws still contain a
  better solver than the pipeline's three plans, by ~1.75×.** F7 stands on those.
- **On the hard problems the pipeline does not catastrophically fail, and naked draws do.** 4 of 30
  draws exceed their gate by 10×–200× (B02 ×1, B06 ×2, B07 ×1) against **0 of 30 pipeline plans** —
  every plan on every problem passed the harness. That is a real and previously unmeasured pipeline
  advantage, and it is the first measured thing the verification machinery buys.

**Blind against blind, the pipeline now wins clearly.** Against a single unexamined Sonnet draw it is
**0.42×** — 2.4× better — and still 0.73× ahead with both failure problems removed. The §2.2 finding
that it merely ties an unexamined draw was an artefact of comparing against Opus.

**Selection is now the whole of the remaining gap.** Shipped ÷ C0 best-of-3 is 1.73× while pool ÷ C0
best-of-3 is 1.11×; the 1.55× selection loss of §6 accounts for essentially all of it. Everything in
§6's Fix 1 and Fix 2 is therefore better-targeted than §2.2 implied, and Fix 4 (plan diversity) is
worth less than F7 claimed — though not nothing, since it is what §4's B03 needs and B03 is the one
problem where a single Sonnet draw beats the entire pipeline pool by 7.1×.

**The model effect, isolated:** C0@Sonnet best-of-3 is 1.50× worse than C0@Opus best-of-3 across the
ten, but only 1.18× worse once B06 and B02 are removed. Sonnet's deficit against Opus is concentrated
in catastrophic failures on the hardest problems rather than spread as a uniform accuracy penalty —
which is the same shape as the pipeline-vs-C0 result above, and is consistent with F3's hypothesis
that the formulator at Sonnet writes weaker verification plans without writing worse solvers.

## §8 Four arms, pool best, with the noise bar

Every figure here is the **best plan or draw the arm produced**, whether or not it was selected, so
this table measures what each arm can *build* and says nothing about selection. Scripts:
`fourarm.py` and `finaltab.py` beside the records. Errors and bars are recomputed by calling the
harness's own `run.plan_harness_error` on each record, so all four arms use one code path.

`*` marks a figure **below its own noise bar** — a number the records contain no evidence is better
than the bar itself. "Separable?" asks whether the spread across the four arms exceeds the largest
bar on that problem.

| id | gate(s) | noise bar | C0@Opus best/10 | C0@Sonnet best/3 | AN pre-change | AN now | separable? |
|---|---|---|---|---|---|---|---|
| B01 | 0.004 | 0.437 | 0.0180\* | 0.0180\* | 0.0259\* | 0.0473 | **no** |
| B02 | 0.010 | 0.450 | 0.0116\* | 0.4596 | 0.7592 | 0.1674\* | yes |
| B03 | .015/.004/.002 | 0.376 | 0.0178\* | 0.0579\* | 0.4478 | 0.4084 | yes |
| B04 | 0.006 | 0.205 | 0.0603\* | 0.0603\* | 0.0603\* | 0.1353\* | **no** |
| B05 | 0.008 | 0.178 | 0.0391 | 0.0391 | 0.1315 | 0.1006\* | **no** |
| B06 | 0.003 | 0.161 | 0.1445\* | 0.5199 | 0.4897 | 0.0474\* | yes |
| B07 | 0.003 | 0.173 | 0.0666\* | 0.0659\* | 0.2952 | 0.0388\* | yes |
| B08 | .015/.004/.002 | 0.248 | 0.0292 | 0.2417 | 0.3350 | 0.1543\* | yes |
| B09 | 0.006 | 0.167 | 0.2243 | 0.0782 | 0.2968 | 0.1175 | yes |
| B10 | 0.010 | 0.170 | 0.0528\* | 0.1106 | 0.0041\* | 0.1612\* | **no** |

| geometric mean, ten problems | C0@Opus | C0@Sonnet | AN pre-change | AN now |
|---|---|---|---|---|
| raw pool best | 0.0443 | 0.0979 | 0.1450 | 0.1090 |
| **floored at each plan's own bar** | **0.0914** | **0.1572** | **0.2631** | **0.1558** |

### What the bar does to the conclusions

- **Four of ten problems — B01, B04, B05, B10 — cannot distinguish any of the four arms.** The whole
  spread sits inside one noise bar. Any ranking of the arms on those problems is reading seeds.
- **Noise-floored, the pipeline's pool and three naked Sonnet draws are a dead tie: 0.1558 vs
  0.1572, 0.99×.** §7's raw 1.11× and the 1.75×-excluding-failures both shrink to nothing once each
  figure is held to its own error bar. This is the most defensible form of the model-matched result.
- **The improvement over the pre-change pipeline survives: 0.2631 → 0.1558, 1.69×**, essentially the
  raw 2.20× of §2.1 after flooring. That remains the run's one solid gain.
- **Opus's advantage also survives: 1.70× over both Sonnet arms.** It is a genuine model effect at
  the pool level, not a failure artefact — the pipeline and naked Sonnet draws land in the same
  place and Opus is ahead of both.
- **7 of 10 "AN now" pool bests, and 7 of 10 C0@Opus best-of-ten, are below their own bar.** A pool
  minimum over several noisy draws is biased downward by construction, and the more draws the worse
  it gets. This is why §2.2's best-of-three figures and §7's raw numbers should not be quoted without
  the bar: **best-of-N is a winner's curse**, and C0@Opus with ten draws suffers it most.

## §9 How a gate unit is calculated

One number per solver, so that problems with different gates and different functionals compare.

**Step 1 — relative error per functional.** For each scored functional (BENCHOP's are option prices
or no-touch probabilities), `verify.py` compares the solver's value against independently-authored
truth in `benchmark/problems.py`:

```
rel_err[name] = |v_solver - v_truth| / |v_truth|
```

**Step 2 — divide by that functional's own gate.** Each functional carries its own tolerance,
`functional_tol_used[name]`, falling back to the problem's `functional_tol`. The gates are *not*
uniform: B03 and B08 gate their three strikes at 0.015, 0.004 and 0.002 — a factor of 7.5 — because
a deep out-of-the-money price is intrinsically harder than an at-the-money one.

```
gate_units[name] = rel_err[name] / tol[name]
```

**Step 3 — take the worst.** The solver is scored on its weakest functional:

```
gate units = max over name of gate_units[name]
```

So **1.0 is exactly the pass line** on every problem, lower is better, and 0.25 means "the worst
functional used a quarter of its allowance". The implementation is `run.plan_harness_error`.

Worked example, B06 plan 1 (`1-euler-full-truncation-antithetic`), from the record:

| functional | abs err | truth | rel err | gate | gate units |
|---|---|---|---|---|---|
| `price_S75` | 1.0213e-03 | 37.52 | 2.722e-05 | 0.003 | 0.0091 |
| `price_S100` | 2.8715e-03 | 57.41 | 5.002e-05 | 0.003 | 0.0167 |
| `price_S125` | 1.1173e-02 | 78.58 | 1.4219e-04 | 0.003 | **0.0474** ← worst |

The plan's recorded `harness_error` is 0.04740.

### The noise bar

Because a price is a Monte-Carlo estimate, the same solver re-run with another seed lands somewhere
else. The bar is that scatter, expressed in the same gate units, computed for the functional that
*decided* the score:

```
sampling_SE = |v_graded - v_coarse| / sqrt(r - 1),   r = paths_graded / paths_coarse
noise bar   = hypot(reference_abs_err, sampling_SE) / |v_truth| / tol
```

The two levels come free from the anti-hardcoding check, which already re-solves at a coarser path
count. Under nested sampling (same seed, same `dt`, `r`× the paths)
`Var(graded − coarse) = sigma^2 (r−1)/n_graded`, hence the `sqrt(r−1)` divisor; if the two levels are
in fact independent the divisor would be `sqrt(r+1)`, so this is the **larger** of the two and errs
toward calling differences unresolvable. `reference_abs_err` adds the truth's own uncertainty where
the reference is itself an estimate. It is one realization, so it has a single degree of freedom —
enough to stop a difference far below the noise from reading as a finding, not enough to settle a
knife-edge call.

Same example, B06 plan 1:

```
price_S125:  graded 78.5688 (1,000,000 paths)   coarse 78.5191 (250,000)   delta 0.049755   r = 4
sampling_SE = 0.049755 / sqrt(3)           = 0.028726
reference_abs_err                          = 0.005
noise bar   = hypot(0.028726, 0.005) / 78.58 / 0.003 = 0.1237
```

Recorded `harness_error_margin`: **0.1237**. So that plan's error of 0.0474 is **a third of its own
standard error** — the record contains no evidence it is better than any other plan within ~0.12 gate
units of it, which is most of its pool. Two figures should be treated as tied whenever they differ by
less than the larger of their bars.
