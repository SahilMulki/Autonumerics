# SciMLBenchmarks (SDE) — External Benchmark Adoption Plan

**Status:** **researched 2026-11-09; rev 1 the same day; rev 2–5 2026-11-10; rev 6–7 2026-11-11;
rev 8–10 2026-11-12; rev 11–12 2026-11-13 — steps 1–10 of §7 done on the scheme-matched arm. The suite exists, is
calibrated, is smoke-tested end to end, has its comparator numbers, has its C0 baseline in both
arms, and has **all nine of its entries through the pipeline** — 5 `VERIFIED_PASS`, 4
`UNDERCLAIM`, **zero overclaims**, 139.3 min of wall time. Rev 12 then answers what the suite was
built to ask (10f): on an equal-budget comparison the pipeline **as shipped is 1.412× worse than
Opus answering cold**, while **the best plan in each of its own pools is 2.65× better** — so the
deficit is selection, not generation. Measuring the three candidate ranker fixes retires two as
inert and **withdraws the resolution floor as harmful**; the one fix that crosses the line is in
the *scorer*. What remains is that fix, the unrestricted arm, and the optional Phase 2 (step 11).** Rev 6–7 `[2026-11-11]` diagnose the M01 probe and
run the other two. **`wrong_plan_won: true` on all three problems, by three different
mechanisms** (§10a): on M01 the scores separated and did so backwards, decided by a terminal-law
KS statistic that a *pathwise-exact* solver also fails on the run's own increments; on M06 and M09
no *score* separated anything (10/10/10 and 8/8/8) and the winner was chosen below
`score`: on M06 by a quality index renormalised over different components for different
plans, so the best plan lost **for being too accurate to resolve**, and on M09 by
`run.py`'s fallback in declaration order, because the run timed out before the ranker
ever ran. Both shipped plans 5.1× and 3.3× worse than the best available. Rev 8 fixes
both (10b): a component may decide a comparison only where every plan in the pool
measured it, and the record now says whether the winner was the conductor's choice or
the grader's. So the inversion is a property of the
SDE evaluator, not of the pathwise route, and it does not need a failing gate — a tie is enough.
Three more harness defects came out of these runs (§7a defects 8–10): a dead workspace graded as a
live run, a `score_cap` that is recorded and never applied, and the formulator-written
`num_paths` that the uncapped violation was about. Rev 9 `[2026-11-12]` runs **M02**, the
first entry under the fix (10c): `UNDERCLAIM` in 13.0 min, and it answers 10b's open
question — the ranker's winner was *not* the first plan and the conductor shipped it, so
**the conductor obeys its ranker**. The fix was inert there, correctly, because all three
plans were missing the same components. What M02 finds instead is the next layer: with
every accuracy component in the pool unresolved, the winner was chosen on
`invariant_worst_headroom` **2.2379 against 2.2330** — the index is being read to a
precision it does not have, and unlike every other comparison in the repo it has no
notion of its own resolution. Defects 9 and 10 also reproduced there on a different wrong
path count. **Rev 10** `[2026-11-12]` runs the remaining five (10d): all five completed at
7200 s, none timed out, **80.0 min** for five problems — 2 `VERIFIED_PASS`, 3 `UNDERCLAIM`,
and across the eight completed entries **no overclaim anywhere**. The fix flips no winner
on the five and widens M07's margin **8.8×**. `wrong_plan_won: true` on M05 by **622×**,
44.6× outside the margin, because that problem's order check never ran and the selection
fell to `('structure',)` alone — 10c's resolution defect with a measured cost, and the
case for a floor that abstains. The vocabulary gap is now an **incentive** rather than an
omission: M05's order-1.5 plan withholds `scheme_family` deliberately, with the correct
reason written above its own metrics block — no kernel family key maps to order 1.5, so a
correct scheme fails every band it could declare, and silence scores better than the
truth. Defects 9 and 10 now stand at **seven of eight** completed entries, so no sciml
score on record has been capped when seven should have been. Rev 5
`[2026-11-10]` closes the comparator track and the smoke test: §3e (the pinned environment — both
upstream manifests instantiate verbatim, and the three published `BoundsError` cells §1a.2 names now run), §3f
(`comparators.json`, 279 rows over 19 schemes), §5a (the EM anchor, three numbers rather than one)
and §7a (the smoke test and the four harness defects it found). Four of rev 5's findings change how
the result must be read rather than confirming the plan: **the `evals` axis over-charges
multi-stage schemes by up to 6× against measured seconds**, so it flatters the pipeline (§3f);
**§10's prediction 1 is confirmed under one reading of "matched work" and falsified on four of nine
entries under the other**, and the two readings differ by two orders of magnitude (§3f); **the
scalar EM anchor's 1.03× parity is an artifact of a latency-bound dependency chain**, with 2.71× the
honest ceiling (§5a); and **nothing in this repository had ever rendered a work-precision series**,
which on this suite is the primary axis (§7a). Rev 5 then ran step 9 (§9a — 18/18 PASS, and the
two arms turn out to measure completely different things: **0 of 8 transcriptions in one, 8 of 8 in
the other**) and probed step 10 on one problem (§10a — which came back `wrong_plan_won: True`, a
field that was `null` until §7a's first defect was fixed). Two further defects were found by
*running* rather than by testing, and both would have invalidated the runs that found them: the
`unrestricted` arm staged a byte-identical brief and excluded transcriptions, so it was a label on
the scheme-matched experiment; and the two arms shared their artifact paths and unit keys, so the
second deleted the first's solver sources and would have deduped the first's records. Rev 1 narrows the scope to **SDE only, confirmed by the project
owner**, and adds the two things the first draft left as one-line gestures: the **environment setup** needed to produce comparator numbers (§3) and
the **harness work list** needed to run the problems at all (§4, items H1–H10 with files, line
numbers and acceptance tests). Rev 1 also records three findings from reading the harness that the
first draft had wrong or missing, all in §4a: the graded step is forced _finer_ than every published
point on the weak-error problem, two Phase-2 problems exceed the solver budget by orders of
magnitude, and the benchmark problem entries have no way to declare a noise dimension.

**Rev 2 `[2026-11-10]` — step 2 is done and it changed the problem set.** Every formula in §2 was
re-read against the raw `SDEProblemLibrary.jl` at the pinned version and against the raw `.jmd`
sources of all five SDE benchmark notebooks; §2a is the diff, and four of its findings overturn the
plan rather than confirming it. **S06–S08's terminal state is the constant −1 to machine precision**
at SciML's own `T = 3`, so three of the nine Phase-1 entries are vacuous as this harness grades them
(§2a.1). **SciML's `:l2` axis is a timeseries error, absolute rather than relative**, so H1 as
written does not measure the published quantity (§2a.2). **Three of §2's nine entries are not in the
library at all** and two more are in the library but in no notebook, so their `[L]`/`[W]` labels and
their comparator status were both wrong (§2a.3). And **§1a.3 is withdrawn** — the page prose does not
disagree with the library; the summarising agent conflated two of its sections (§2a.4). Step 2 also
found a fifth notebook that grades on the terminal state, which is the axis this harness already
has, and names the cheapest honest entry in the adoption (§2a.5).

**Rev 3 `[2026-11-10]` — step 5 is done: H1, H2, H3, H4, H8, H9 and H10 are shipped** with 28
acceptance tests, and all 66 existing problem statements and the whole manifest hash identically
before and after. §4b records it. Implementing it mostly confirmed the plan; the four places it did
not are worth more than the code. **H1 was bigger than one route**, because the `dW` the plan said
"the contract already accepts" is the _kernel's_ contract and the benchmark's own SDE contract had
no such argument on any problem — so the route needed a contract extension and increment plumbing
through the sandbox child as well. **The gate had to be measured before it could be set, and the
first value was wrong**: 5% failed a _correct_ Euler–Maruyama at 5.36%, which is the one thing this
repo's calibration rule forbids and the kind of false hard-fail that is invisible as a false
positive; at 10% a correct EM clears by 1.9x and a one-decision-wrong variant fails by 4.1x. **H2
needed the distinction §2a.1 found rather than the single verdict H2 specified**, so a constant and
a transcription are both excluded but named apart — and `validate_ground_truth.py` now refuses a
vacuous entry at _authoring_ time, which is what stops the stiff-quadratic problems being declared
at `T = 3` in step 6 and discovered in step 9. **H9's deny-rule half was already done**; the hole
was that `harvest.jl` is a `.jl` file under `benchmark/` and every interpreter in the deny list was
a Python one.

**Rev 4 `[2026-11-10]` — steps 6 and 7 are done: the nine entries are authored and every gate is
calibrated both ways.** §2b records it with the measured tables. **Every gate separates** — a
correct Euler–Maruyama clears by 2.0–7.3x, a one-decision-wrong variant fails by 2.2–238x, the
narrowest separation is **4.4x** against BENCHOP's 6.8x — and no gate was moved after seeing where a
scheme landed. Step 7 is folded into step 6 because it could not be deferred: three entries were
*vacuous* until calibration found them. The findings: §2a.2's alternative to moving the stiff
horizon was **measured and rejected** (at `T = 3` a constant solver's relative timeseries error is
3.5e-13 at 10 saves and still inside any gate at 1000, because the transient is 0.3% of the
interval); **three entries were vacuous rather than one**, since `additive` and `wave` are
answerable by a constant at a default gate, which establishes that a strong-error gate's scale is a
property of each problem's terminal spread and not a universal number; **two of §7's one-decision
controls fail** — a dropped Milstein correction is Euler–Maruyama, a correct scheme and not a wrong
one, and the β²-drop is unobservable where `β² ≪ α`; **§10's prediction 4 is falsified** with its
mechanism inverted, since the Monte-Carlo floor does bind on the weak observable but by making a
correct scheme's error *unmeasurable* rather than by forcing the gate wider; and M09's published
value is **derived rather than transcribed**, which is how the first row of its moment system turned
out to show that SciML's own graded observable cannot see the channel structure at all. Step 6 also
caught **two defects in step 5's own work**, the worse being that only the pipeline verdict knew
about the transcription exclusion — so a transcribing C0 baseline, the case §10 predicts is most
likely, would have been recorded as a clean pass.

**Verdict unchanged:** feasible; the cheap half of the suite is trivially winnable for a reason with
no BENCHOP analogue (§0b); the comparison is cross-runtime (§0c); and the outcome on the non-stiff
set restates a documented gap in `sde_manual.md` (§0d). Rev 2 does not move the verdict, but it moves
the _cheapest honest path to it_: three Phase-1 entries turn out to have a published **weak** axis
that the repo's existing moment route already grades (§2a.5), so a comparator table no longer waits
on H1, and the one Phase-2 problem graded on a terminal state is affordable today (§2a.5).

**Scope:** the SDE sections of SciMLBenchmarks — 4 non-stiff and 5 stiff notebooks, comparing 6–12
published schemes as implemented in `StochasticDiffEq.jl` — as a fourth suite alongside `main`,
`no-closed-form` and `benchop`. **No PDE entries**, as in the BENCHOP-SLV adoption. Note that
SciML's stochastic heat problem is method-of-lines and is therefore graded under the _SDE_ contract
as an N-state system, not as a PDE.

**Related:** [plan-benchop-slv.md](plan-benchop-slv.md) (the adoption this one copies its shape
from), [improving-autonumerics/](improving-autonumerics/) (what the BENCHOP runs measured about the
pipeline), [plan-sde-verification-gaps.md](plan-sde-verification-gaps.md) (the CRN ladder this plan
reuses), [plan-no-closed-form.md](plan-no-closed-form.md) (the reference route Phase 2 needs).

**Provenance of every fact below.** `[W]` read off a SciMLBenchmarks page on 2026-11-09, `[L]` read
off `SDEProblemLibrary.jl` master on 2026-11-09, `[R]` measured or read in this repo, `[N]` new in
this plan. **One caveat applies to every `[L]` and `[W]` item:** both were fetched through a
summarising agent, so the formulas below are _close to_ verbatim and **must be re-read against the
raw file at the pinned version before any of them is transcribed into `problems.py`** (step 2).

---

## 0. Verdict

**Feasible.** The three things that made BENCHOP-SLV expensive are all cheaper here, and the
machinery the comparison needs mostly exists:

- **Truth is free.** `[L]` Every non-stiff SciML test problem ships an `analytic(u0, p, t, W)` —
  an exact solution that is a closed-form function of the **terminal Brownian value alone**:
  `u0*exp(0.63155t + 0.87W)` for the linear problem, `atan(0.1W + tan(u0))` for the wave problem,
  `(2t + 1 + W/3)^3` for the nonlinear test, and `((1+u0)e^{W} + u0 - 1)/((1+u0)e^{W} + 1 - u0)` for
  the cubic. BENCHOP step 6 cost a re-derivation of 24 published numbers by two independent method
  families; here there is nothing to re-derive. The _weak_ truth is a 1-D Gaussian integral of a
  known transform, which this repo already computes to machine precision by Gauss–Hermite — see
  `quintic_ic_moments(n_gh=160)` at `benchmark/problems.py` `[R]`.
- **The comparators are rerunnable.** MIT-licensed Julia, 6–12 schemes per notebook, on your own
  hardware. BENCHOP-SLV gave plots and encrypted p-code; SciML gives source.
- **The mechanisms exist.** The supplied-increment contract (`dW`, `verification_manual.md` §18,
  `verifylib/kernel/sde/crn.py`) is exactly the common-random-number protocol a strong-error
  measurement needs; `work_precision_levels` already re-solves an SDE problem at a list of
  `(num_paths, dt)` pairs and emits error-vs-time; `runner.py:353` already separates the solve clock
  from the process clock; a fourth suite is one tuple at `benchmark/setup.py:434` `[R]`.

**And it is largely pointless in the form that is cheap.** That is §0b, and it is the finding this
plan exists to record.

### 0b. The cheap half is trivially winnable, and winning it measures nothing

`[N]` On BENCHOP the answer key was a _number_, so the anti-memorisation question was "did the
solver recall the price?" and the defence was a shifted-parameter control. Here the answer key is a
**closed-form pathwise solution**, and the solver is handed the Brownian increments. So a solver can
return

```python
X_T = u0 * np.exp(0.63155 * T + 0.87 * dW.sum(axis=1))
```

which is **exact at every `dt`**, consumes the supplied increments (so it is not the `dW`-ignoring
defect `verifylib/kernel/score.py` catches), varies correctly with the path count (so the
two-path-count anti-hardcoding check passes), and beats every scheme in `StochasticDiffEq.jl` by an
unbounded margin on every axis SciML plots. It also simulates nothing.

This is not hypothetical and it is not a leakage failure: deriving the analytic solution is what the
pipeline's formulator is _for_, and `sde_manual.md` rule 0 already instructs plans to sample an
exact transition law where one exists. The point is narrower and worse: **SciML's problems are
yardsticks, not tasks.** Their exact solutions exist so that a _scheme_ can be measured against
them. Adopting them as graded problems scores the pipeline against a ruler it can print.

Three consequences shape the whole plan:

1. The comparison is only meaningful in a **scheme-matched arm** (§6) that forbids the exact path
   and requires a discretised stepper — and that arm has to be _enforced_ rather than requested.
2. `[R]` Enforcement is nearly free, because the CRN ladder already detects it: a pathwise-exact
   solver returns **bit-identical** terminal states at every level of the ladder, so the pathwise
   difference `s0` collapses to rounding. That is the same reading the repo already has a precedent
   for — `sde_bm_with_drift`, where Euler–Maruyama is pathwise exact and `s0 = 9.4e-15` with the
   strong order abstaining (`verifylib/kernel/sde/crn.py`) `[R]`. What is missing is not the
   detector but the _verdict_: on this suite a degenerate ladder means "not a scheme", and nothing
   today says so. That is **H2**.
3. The half of SciML that is **immune** — the problems with no closed form (Brusselator, Lorenz,
   oscillating reaction network, Oval2, stochastic heat) and the high-order weak problem, whose
   truth is a moment rather than a path — is exactly the half that needs a harness-owned reference
   and a state dimension above 3. **The honest half is the expensive half.** Any plan that runs the
   cheap half alone will produce a table of wins that means nothing, and a reviewer who reads
   `SDEProblemLibrary.jl` will say so in one sentence.

### 0c. The second problem: the comparison is cross-runtime

`[N]` SciML's output axis is wall-clock at matched accuracy. `StochasticDiffEq.jl` compiles a
scalar per-step loop and ensembles over threads (the published runs used 128 threads on an EPYC 7502
`[W]`); an AutoNumerics solver is numpy, restricted to "standard library and numpy only" by the
solver contract `[R]`, and gets its speed by vectorising **across paths**. On a scalar 1000-path
strong-error test numpy vectorisation is competitive or better; on Oval2 (19 states, few paths,
stiff) it loses by a large factor that has nothing to do with the scheme. Quoting raw seconds side
by side is a language benchmark wearing a numerics benchmark's clothes. §5 is the fix, and **H4** is
the code it needs.

### 0d. The third problem: the outcome on the non-stiff set is already known

`[R]` `references/sde_manual.md` offers EM, Milstein (scalar multiplicative only, rule 2), exact
transition sampling (rule 0), conditional Monte Carlo (rule 0b), and for stiffness rule 3 —
_"implicit or semi-implicit scheme required. Not yet supported; document the limitation."_ There is
**no Rößler-family scheme in the manual at all** — no SRA, SRI, SOSRA, SOSRI, DRI1, PL1WM, RDI\*WM.
`[W]` SciML's non-stiff headline is that those methods are _"orders of magnitude more efficient"_
and _"should be used whenever applicable"_, and its high-order weak page concludes _"DRI1 in
particular achieves the smallest errors for fixed timesteps."_

So in the scheme-matched arm the pipeline enters a contest between EM/Milstein and nine schemes
designed to beat them, on problems chosen by SciML _because_ the Rößler methods apply (additive and
diagonal noise `[W]`). It will lose, and the measurement will be a restatement of rule 3 and of the
manual's missing scheme table. That is a legitimate thing to measure **once**, as the cost of a
documented gap — the BENCHOP runs were valuable for the same reason — but it is worth one
measurement, not a suite. The suite earns its keep only if the manual gains an SRK section and the
run is repeated; §12 makes that the plan's own success criterion rather than a hope.

---

## 1. What SciMLBenchmarks actually measures `[W]`

~~Three axes, and the repo currently grades on none of them:~~ **Rev 2: three axes across _five_
notebooks, and the repo already grades on one of them** — §2a.5 adds the two pages this table
missed, one of which grades the terminal moment and one the terminal state. Note also that
`error_estimate = :l2` is a **timeseries** error over every saved step and is **absolute**, not the
terminal relative error this plan assumed (§2a.2); `:final` is the terminal one.

| Page                                 | Axis                                           | Settings as published                                                                                                                                                  |
| ------------------------------------ | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Basic SDE Work-Precision (non-stiff) | **Strong** error, `error_estimate = :l2`       | `N = 1000` runs; `tspan (0,1)`; `reltols = abstols = 10^-(1:5)`; fixed-step `dt = 5^-(2:6)` (1/25 … 1/15625); `maxiters = 1e7`                                         |
| High-Order Weak Work-Precision       | **Weak** error, `error_estimate = :weak_final` | `numtraj = 1e5`; `seed = 100`; `dts = 1, 1/2, 1/4, 1/8`; observable `h(z) = z²` on component 1; `expected_value = exp(-3.0)`; `adaptive = false`; `test_dt = 1//10000` |
| Quadratic Stiffness                  | Strong, `:l2`, plus `@time`                    | `numruns = 10`; `reltols = abstols = 10^-(3:5)`; EM/SRIW1-fixed `dt = 8^-(2:4)` and `8^-(3:5)`                                                                         |

**Schemes compared, in the published order** `[W]`: SRIW1 (adaptive and fixed), EM, RKMil, SRA1,
SRA2, SRA3, SOSRA, SOSRA2 (additive-noise set); SRI, SRIW1, SRIW2, SOSRI, SOSRI2 (diagonal set);
DRI1, PL1WM, RDI1WM, RDI2WM, EM, SimplifiedEM (weak set); SRIW1, EM, ImplicitRKMil (stiff set).
That clears the "three or more comparators" bar many times over, **but all of them are one library's
implementations of other authors' published schemes.** The honest framing of any result is
"AutoNumerics vs `StochasticDiffEq.jl`'s algorithm set", not "vs nine independent groups".

### 1a. The published numbers cannot be used as they stand

Four findings, each of which forces a local rerun rather than a citation `[W]`:

1. **Almost no numbers are published.** Pages 1 and 2 are plots only — no timings, no error values,
   no tables. The only printed numbers anywhere are page 3's `@time` lines, and most of those are
   compilation-dominated (_"99.89% compilation time"_, _"99.98% compilation time"_), so they are
   first-run artefacts, not timings.
2. **Half of page 1 does not run.** Four cells report `BoundsError: attempt to access 4-element
Vector{Float64} at index [5]` and `… 0-element Vector{Matrix{Float64}} at index [0]` — the 2-D
   linear and scalar linear first cells, and the wave first cell. The published non-stiff
   comparison is _incomplete_. Regenerating it is a small contribution in its own right.
3. ~~**The page prose disagrees with the library.** Page 1 describes the linear problem as
   `dX = αX dt + βX dW` with `α = 1/10, β = 1/20`; the library has `f = 1.01u`, `g = 0.87u`,
   `analytic = u0*exp(0.63155t + 0.87W)` `[L]` (and `1.01 - 0.87²/2 = 0.63155` checks out, so the
   source is self-consistent and the page is stale — those α, β values belong to the _additive_
   problem, `p = (0.1, 0.05)`). The page also writes the solution exponent with α and β transposed.~~
   **Withdrawn in rev 2 — this finding was an artefact of the summarising agent (§2a.4).** The page
   describes the _additive_ problem with `α = 1/10, β = 1/20`, which is exactly its
   `p = (0.1, 0.05)`, and separately describes the linear problem as `α = 1.01, β = 0.87` with
   solution `X₀e^{(α − β²/2)t + βW_t}` — correct, and not transposed. The standing instruction
   survives the withdrawal for a different reason: **problem definitions are read from the library at
   the pinned version because the pages are rendered from a drifting `master` (§2a.7), not because
   their prose is wrong.**
4. **Versions and hardware drift across the three pages.** StochasticDiffEq 6.87.0 / 7.1.5 / 7.1.4;
   SDEProblemLibrary 1.1.0 / 1.2.4 / 1.2.3; Julia 1.10.10 / 1.11.9 / 1.11.9 with a 1.12.7 docs
   footer; all three on a 128-thread EPYC 7502. This machine is 8 cores / 17 GB `[R]`. Nothing is
   comparable without a local rerun at pinned versions.

---

## 2. The problem set

~~`[L]` Everything below is from the library source, with the §0 caveat: re-read the raw file at the
pinned version before transcribing.~~ **Rev 2: the re-read is done (§2a), and the labels were wrong.**
S01–S05 are `[L]`, confirmed verbatim against v1.2.4 and confirmed numerically to solve their own
equations. **S06–S08's parameter sets and the whole of S09 are `[W]`, not `[L]`** — the library's
`prob_sde_stiffquadito` carries `p = (1.0, 1.0)`, and no problem resembling S09 exists in the
library at all (§2a.3). **S06–S08 are additionally vacuous at the `tspan` below** and their graded
horizon has to move into the transient (§2a.1); the `0.5, (0,3)` in their row is SciML's, not this
suite's.

### Phase 1 — closed-form, pathwise-exact truth, `d ≤ 2`

| #       | Candidate slug                         | SDE                                                                                                                                                             | `u0`, `tspan`  | Exact solution                                                     | dims (state, noise)    |
| ------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------------------------------------------------------------------ | ---------------------- |
| S01     | `sde_sciml_linear_ito`                 | `f = 1.01u`, `g = 0.87u`                                                                                                                                        | 0.5, (0,1)     | `u0·exp(0.63155t + 0.87W)`                                         | 1, 1                   |
| S02     | `sde_sciml_additive`                   | `f = p₂/√(1+t) − u/(2(1+t))`, `g = p₁p₂/√(1+t)`, `p = (0.1, 0.05)`                                                                                              | 1.0, (0,1)     | `u0/√(1+t) + p₂(t + p₁W)/√(1+t)`                                   | 1, 1                   |
| S03     | `sde_sciml_cubic`                      | `f = −0.25u(1−u²)`, `g = 0.5(1−u²)`                                                                                                                             | 0.5, (0,1)     | `((1+u0)e^W + u0 − 1)/((1+u0)e^W + 1 − u0)`                        | 1, 1                   |
| S04     | `sde_sciml_wave`                       | `f = −0.01·sin(u)cos³(u)`, `g = 0.1cos²(u)`                                                                                                                     | 1.0, (0,1)     | `atan(0.1W + tan u0)`                                              | 1, 1                   |
| S05     | `sde_sciml_nltest`                     | `f = ⅓u^{1/3} + 6u^{2/3}`, `g = u^{2/3}`                                                                                                                        | 1.0, (0,10)    | `(2t + 1 + W/3)³`                                                  | 1, 1                   |
| S06–S08 | `sde_sciml_stiffquad_{mild,det,mixed}` | `f = −(α + β²u)(1−u²)`, `g = β(1−u²)`, `(α,β) = (50,1), (500,1), (5000,70)`                                                                                     | 0.5, (0,3)     | `((1+u0)e^{−2αt+2βW} + u0 − 1)/((1+u0)e^{−2αt+2βW} − u0 + 1)`      | 1, 1                   |
| S09     | `sde_sciml_weak_noncommutative`        | `f₁ = −(273/512)u₁`; `f₂ = −(1/160)u₁ − (−785/512 + √2/8)u₂`; `g` the 2×2 matrix with `g₁₁ = u₁/4`, `g₁₂ = u₁/16`, `g₂₁ = ((1−2√2)/4)u₁`, `g₂₂ = u₁/10 + u₂/16` | `[1,1]`, (0,3) | **no pathwise form**; `E[X₁(t)²] = e^{−t}`, so `e^{−3}` at `T = 3` | 2, 2 (non-commutative) |

Notes that decide the entries:

- **Every Phase-1 exact solution is a function of `W_T` alone**, so the harness needs no reference
  solver: it sums the supplied increments and evaluates a formula. This is what makes Phase 1 cheap
  — and, per §0b, what makes it cheatable.
- **S06–S08 are the suite's hardest problems and they have a closed form** `[L]`. SciML's own
  conclusion on the mixed case is _"Noise stiffness is tough"_, _"the adaptive algorithm is
  unstable"_, and _"the best solution is to run an explicit integrator with a low enough dt"_ `[W]`.
  Against `references/sde_manual.md` rule 3 ("not yet supported") this is the single most
  informative entry in the plan, and the only one where nobody — library or pipeline — does well.
  Note the implementation constraint: **scipy is blocked in SDE solvers** `[R]`, so an implicit
  scheme here has to hand-roll its Newton iteration in numpy. That is a page of code, not a
  blocker, but it is the kind of thing a plan omits and a run discovers.
- **S09 is the one Phase-1 entry immune to §0b**, because its truth is a moment and no pathwise
  formula exists. It is also a natural fit for the existing `functional_truth` machinery: one
  scalar, exactly known, graded relatively, with the two-path-count check as the anti-constant
  defence `[R]`. Its noise is non-commutative 2×2, which no problem in the three existing suites
  has — which is **H3**, and which exercises `crn.ladder_spec(..., m=2)` on a shape it has not seen.
- **`prob_sde_2Dlinear` is excluded**: `u0 = ones(4,2)/2` is a 4×2 _array_ state, which is a
  Julia-ism rather than a distinct problem, and it is one of the cells that errors `[W]`.
- **`prob_sde_additivesystem` (`d = 4`) is deferred to Phase 2** solely on the dimension ceiling
  (**H5**). It is the cheapest forcing case for that change: same equation as S02, four components,
  `p = ([0.1]*4, [0.5, 0.25, 0.125, 0.1115])`.
- **Stratonovich variants are excluded.** `[L]` They share `f` and `g` with their Itô counterparts
  and differ _only_ in the analytic function; the problem object carries no interpretation flag. A
  problem statement that cannot be distinguished from another by its own text is not a benchmark
  entry, it is a trap.

### Phase 2 — no closed form; needs a reference route, `d > 3`, and a budget

| Candidate                             | Shape `[L]`                                                                                     | Why it is the half worth running                                                                                                        | Blocked on                                                                                                                                                                             |
| ------------------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prob_sde_bruss`                      | state 2, **noise 4** (2×4, non-diagonal), `p = (1,1,2.5,1)`, `t ∈ (0,100)`                      | SciML's own weak-error test problem; immune to §0b                                                                                      | H3, H6                                                                                                                                                                                 |
| `prob_sde_lorenz`                     | state 3, diagonal noise `g ≡ 3.0`, `p = (10, 28, 2.66)`, `t ∈ (0,10)`                           | chaotic + additive noise; the SRA family's home ground                                                                                  | H6                                                                                                                                                                                     |
| `prob_sde_additivesystem`             | state 4                                                                                         | cheapest `d > 3` forcing case; truth is closed-form                                                                                     | H5                                                                                                                                                                                     |
| `prob_sde_oscilreact`                 | state 7, diagonal, 10 parameters, `t ∈ (0,4000)`                                                | chemical-Langevin square-root diffusions, long horizon                                                                                  | H5, H6, **H7 — exceeds the budget by ~10³**                                                                                                                                            |
| `oval2ModelExample`                   | **state 19**, diagonal with 4 non-zero entries, `t ∈ (0,500)`, ~60 named constants, no analytic | SciML's headline stiff result (_"the advantage grows with problem difficulty"_ `[W]`)                                                   | H5, H6, H7, **high transcription cost**                                                                                                                                                |
| `generate_stiff_stoch_heat(N=100)`    | state `N`, `WienerProcess` + `RSWM`                                                             | an SPDE graded under the SDE contract                                                                                                   | H5, H6, H7, and ~~it is **not exported** from the library~~ **the library's copy cannot run at all** (§2a.8); the notebook's own copy works and _does_ have a Stratonovich closed form |
| `LotkaVolterraSDE` (notebook-defined) | state 2, diagonal `g = 0.1u`, `p = [1.5,1,3,1]`, `t ∈ (0,10)`                                   | **rev 2, §2a.5:** the only published SDE comparison graded on `:final` — the terminal state this harness already grades — at 2500 steps | H6 only                                                                                                                                                                                |

`[N]` **The reference route is the one genuinely new piece of ground truth** (**H6**). The repo's
SDE routes are `exact` moments, `reference` (discretisation-free MC of a _known exact path_ —
unavailable here), `functional_truth` and `stability` `[R]`. Phase 2 needs a fourth: a harness-owned
converged reference, measured as a **pathwise difference under common random numbers** against a very
fine step, which is how SciML does it (`appxsol` = EM at `dt = 1/10000`, `test_dt = 1//10000` `[W]`)
and which kills the Monte-Carlo noise in the difference rather than fighting it. `crn.ladder_spec`
already builds nested coarsenings of one fine path, so this is an extension of shipped machinery.
It needs the same validation story the no-closed-form PDE suite has: two independent integrators
agreeing far below the gate (that suite reached 1.245e-09 … 1.375e-11 `[R]`), and nothing short of
that should be trusted.

---

### 2a. Step-2 source reconciliation `[rev2 2026-11-10, N]`

Sources, all fetched raw and independently re-checkable:

| Source                                                                     | Pin                                                                                | Digest                                                                                              |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `lib/SDEProblemLibrary/src/SDEProblemLibrary.jl`                           | tag `SDEProblemLibrary-v1.2.4` → commit `d77baf216b3a77e591a9aabd10032090053b84f1` | sha256 `58adc9b105d5c40fedf17ffd99ad403c160e4f905c2694437073ad5f4ac84dbb` (19 399 bytes, 663 lines) |
| `benchmarks/{NonStiffSDE,StiffSDE}/*.jmd`, `Project.toml`, `Manifest.toml` | `SciMLBenchmarks.jl` `master`, read 2026-11-10                                     | —                                                                                                   |
| `SciMLBase/src/solutions/rode_solutions.jl`                                | `master`, read 2026-11-10                                                          | — (the `:l2` / `:final` definitions)                                                                |

The two measurements below are reproducible from
[sciml-benchmark-data/](sciml-benchmark-data/) — seeded, numpy-only, no Julia and no repo imports —
which also lists every source read, with its pin.

The library is **one file**; there is no second module to read, and `Project.toml` confirms
`version = "1.2.4"`. Every `[L]` formula in §2 is confirmed **verbatim** except where struck through
above. Two confirmations were run beyond a textual diff, because a correctly transcribed formula
paired with the wrong equation still type-checks:

- **Numeric, S01–S08.** Milstein under common random numbers against each analytic formula at four
  nested steps, 4000 paths (1000 and 200 for S07/S08), one fine draw aggregated down. Observed
  strong order: S01 **1.03 / 0.97 / 0.94**, S02 **1.00 / 1.00 / 1.00**, S03 **1.04 / 0.94 / 0.96**,
  S04 **1.02 / 0.98 / 0.98**, S05 **1.00 / 1.00 / 1.00**. A mismatched (SDE, analytic) pair plateaus
  at O(1) instead, so each of these five pairs is confirmed to solve its own equation — this is the
  check that would have caught a transcription error, and it passed. S06–S08 returned a strong error
  of **exactly 0.0**, which is finding 1 rather than a pass.
- **Analytic, S09.** From the coefficients alone:
  `d(X₁²) = 2X₁·(−273/512)X₁ dt + (g₁₁² + g₁₂²) dt + dM = [−546/512 + 1/16 + 1/256]X₁² dt`, and
  `−546/512 + 34/512 = −1`, so `E[X₁(t)²]′ = −E[X₁(t)²]` and the published `E[X₁(t)²] = e^{−t}` is
  exact. The `[W]` moment is derived, not taken on trust — which matters, because it is the one
  Phase-1 ground truth with no formula to check it against.

**1. S06–S08 are vacuous at `T = 3`: the terminal state is the constant −1.** The exact solution
depends on the path only through `exp(−2αt + 2βW_t)`; at `α = 50, T = 3` that factor is `e^{−300}`,
so the solution has collapsed onto `u = −1`, which is also an **absorbing** point of the equation
(`f = g = 0` there, since both carry the factor `1 − u²`). Measured std of the exact terminal state
across 2×10⁵ paths:

| `(α, β)`                         | `t = 1e-4` | `t = 1e-3` | `t = 0.01` | `t = 0.05` | `t = 0.25`  | `t = 3` |
| -------------------------------- | ---------- | ---------- | ---------- | ---------- | ----------- | ------- |
| (1, 1) — _the library's own `p`_ | 7.5e-03    | 2.4e-02    | 7.6e-02    | 1.7e-01    | 4.0e-01     | 4.4e-01 |
| (50, 1) — _S06_                  | 7.5e-03    | 2.5e-02    | 9.9e-02    | 2.0e-02    | **1.8e-10** | **0.0** |
| (500, 1) — _S07_                 | 7.9e-03    | 3.2e-02    | 5.6e-05    | **0.0**    | **0.0**     | **0.0** |
| (5000, 70) — _S08_               | 5.2e-01    | 2.5e-01    | **0.0**    | **0.0**    | **0.0**     | **0.0** |

**0.0** is a bit-identical constant across all 2×10⁵ paths, not a small number. So on S06–S08 as §2
writes them, `return -np.ones(num_paths)` earns a perfect terminal error at any `dt`, having
consumed no increments and simulated nothing. H2 does flag it — the CRN ladder is degenerate for a
constant too — but reports it as `ANALYTIC_TRANSCRIPTION`, which is the wrong reason and would read
as a near-miss rather than as the suite's most trivial defect.

The fix is cheap and it pays for itself twice: **move the graded horizon into the transient** —
`T ≈ 0.05` (S06), `0.005` (S07), `5e-4` (S08) — keeping SciML's `(α, β)` and keeping `tspan = (0,3)`
for the comparator runs only. That also dissolves §4a.ii's budget problem on these entries (S08 at
`T = 5e-4`, `dt = 1e-5` is **50 steps**, not 3×10⁵) and puts the horizon where the harness step rule
`min(dt, 0.04/T)` stops binding, so each entry's declared `dt` is the step it is graded at. What it
costs is the claim that these are SciML's problems: they are SciML's _equations_ on a different
horizon, and every table has to say so.

**2. SciML's `:l2` is a timeseries error, and absolute — so H1 as written measures a different
quantity.** `SciMLBase` computes `errors[:l2] = norm(sqrt(recursive_mean((u − u_analytic).^2)))` over
**every saved timepoint**, gated by `timeseries_errors`, with `errors[:final]` as the terminal one;
`WorkPrecisionSet` then averages over `numruns`. H1's "relative `l2` strong error over paths" at the
terminal state is neither SciML's grid nor its normalisation, so a comparator table built from it
would be putting two different numbers in one column. Three consequences, all cheap now:

- H1's `pathwise_exact` must be evaluable **on a time grid**, not only at `T`:
  `pathwise_exact(u0, p, t, W_t)`. Every §2 formula already is such a function, so this costs
  nothing at authoring time and cannot be retrofitted later without rewriting all nine entries.
- The route records **both** the absolute timeseries `l2` (for comparability) and a relative terminal
  error (for the gate). H1's acceptance test is unaffected: order ≈ 0.5 for EM and ≈ 1.0 for
  Milstein holds in either norm.
- It is also what rescues S06–S08 as _SciML_ problems: their published comparison is informative
  precisely because the transient sits inside the norm. A terminal-only route cannot reproduce that
  at any horizon, which is why finding 1's horizon move and this grid change are one decision rather
  than two. Note the quirk that the `:l2` grid is each setup's **own** step grid when
  `save_everystep` is on, so the published norm is not a fixed-grid norm across `dt`.

**3. Provenance: three of §2's nine entries are not in the library, and two more are in no notebook.**

| Entry                                        | §2 claimed          | Source says                                                                                                                                   |
| -------------------------------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| S06–S08 `(α,β) = (50,1), (500,1), (5000,70)` | `[L]`               | **`[W]`** — the library's `prob_sde_stiffquadito` carries `p = (1.0, 1.0)`; the three sets are `remake`d in `StiffSDE/QuadraticStiffness.jmd` |
| S09, the entire problem                      | `[L]`               | **`[W]`** — defined inline in `NonStiffSDE/HighOrderWeakSDEWorkPrecision.jmd`; **no such problem exists in `SDEProblemLibrary.jl`**           |
| S03 `prob_sde_cubic`                         | a benchmark problem | in the library, used by **no SDE notebook** — adopting it means _authoring_ its comparator setups, not rerunning a published comparison       |
| S05 `prob_sde_nltest`                        | a benchmark problem | same, and additionally **unexported**                                                                                                         |

The §2 formulas for all four are still correct; it is their status that was wrong. The practical
consequence is on step 3: `harvest.jl` can _rerun_ a published comparison for S01, S02, S04 and
S06–S07, but for S03, S05 and S08 it has to **invent** the setups, and those rows are then "our
comparator runs of `StochasticDiffEq.jl`", not "SciML's published comparison". That distinction has
to survive into the table or the table overclaims.

`prob_sde_stiffquadito`, `prob_sde_stiffquadstrat`, `prob_sde_nltest` and
`generate_stiff_stoch_heat` are all unexported (reachable by `import SDEProblemLibrary: x`, not by
`using`), so §2's remark that stoch-heat "is **not exported**" singles out one of four.

**4. §1a.3 is withdrawn: the page prose does not disagree with the library.** The master `.jmd`
describes the _additive_ problem with `α = 1/10, β = 1/20` — exactly its `p = (0.1, 0.05)` — and
separately describes the linear/Black–Scholes problem as `α = 1.01, β = 0.87` with solution
`X₀e^{(α − β²/2)t + βW_t}`, which is right and not transposed. The summarising agent read one
section's coefficients into the other section's equation. §1a's conclusion loses one of its four
legs; the other three (almost no numbers, four erroring cells, 128 threads) stand, and §2a.7 adds a
fifth that is better evidenced than the one withdrawn.

**5. §1's table covers three notebooks of the nine, and two of the six it omits matter.** (The
nine are the 4 non-stiff and 5 stiff the Scope line already counts; the four that remain
uninteresting here are `Oval2LongRun`, `Oval2LongTimes`, `Oval2Timings` and `StochasticHeat`, all
of whose problems §4a.ii drops on budget.)

- **`NonStiffSDE/BasicSDEWeakWorkPrecision.jmd`** — the **weak** axis on `prob_sde_additive`,
  `prob_sde_linear` and `prob_sde_wave`: S02, S01 and S04 graded on a terminal **moment**, which is
  the `exact` truth route this repo has shipped since its first SDE problem. **Three Phase-1 entries
  therefore gain a published comparator axis that needs none of H1** — the single largest piece of
  good news in step 2, because it means the suite produces a comparator table even if H1 slips.
  Settings: `numruns_error = 1000`, `reltols = 10^-(1:5)`, `error_estimate = :weak_final`, the same
  `5^-(2:6)` and `5^-(3:7)` step sets, and `get_sample_errors` at
  `numruns = [10, 10², 10³, 10⁴]` (see finding 6).
- **`NonStiffSDE/LotkaVolterraSDE.jmd`** — `dx = ax − bxy`, `dy = −cy + dxy`, diagonal `g = 0.1u`,
  `u0 = [1,1]`, `p = [1.5, 1, 3, 1]`, `t ∈ (0,10)`, `N = 100`, `reltols = 4^-(2:4)`, seven schemes
  (SRIW1 adaptive and fixed, EM, RKMil, SRIW2, SOSRI, SOSRI2), reference
  `appxsol_setup = SRIW1 @ abstol = reltol = 1e-4`, `test_dt = 1e-2`, and — alone among the five —
  **`error_estimate = :final`**, for both its strong and its weak section. It is the only published
  SDE comparison whose axis is the quantity this harness already grades; at `T = 10` the step rule
  gives `dt ≤ 4e-3`, i.e. 2500 steps, which is affordable; and it needs **H6 only** (no closed form,
  `d = 2`, diagonal noise, so neither H3 nor H5). **It belongs in Phase 2 ahead of `bruss` and
  `lorenz`, and it is the cheapest honest entry in the adoption.** Its own prose records the one
  hazard: "larger `dt`s contained trajectories which would veer off to infinity."

**6. The one published noise floor is computed wrong upstream, and the page says so.** From the weak
notebook: `get_sample_errors` "currently divides by `√N` twice, so it is a factor of about `√1000`
below `E_S = √(V/N)` (a fix is proposed in
`https://github.com/SciML/OrdinaryDiffEq.jl/pull/4612`); treat comparisons against that line as
provisional." So `harvest.jl` must compute `√(V/N)` itself and record both values. This lands
directly on §10's prediction 4, which expects the Monte-Carlo floor rather than the scheme to set
the resolvable gate on S09: the prediction stands, but the published line cannot be used to check it.

**7. Page staleness is confirmed from a second direction, and it is the real reason to distrust the
pages.** Master's High-Order Weak conclusion reads "the observed convergence is slower than the
nominal order 2, `PL1WM` and `RDI2WM` stop improving at the smallest step size, and **`RDI2WM`**
reaches the smallest errors", against §0d's `[W]` quote of "**DRI1** in particular achieves the
smallest errors for fixed timesteps". The rendered pages read on 2026-11-09 are therefore behind
`master`. This also bears on §1a.2: in master every cell's `dts` and `reltols` lengths agree, so the
four `BoundsError` cells may simply not reproduce at `master` — which would make "regenerating the
incomplete comparison" a smaller contribution than §1a.2 claims. Only step 1's local rerun settles
it; §1a.4's instruction is unaffected. **§0d's quote should be replaced by master's wording before it
is used in any write-up**, since it names a different winning scheme.

**8. Smaller corrections and transcription traps, each confirmed in the source.**

- **The finer `dt` sets were missed.** Every non-stiff page runs its Rössler-comparison cells one
  notch finer than its headline cells — `5^-(3:7)` (`(1:5) .+ 2`), not only §1's `5^-(2:6)` — and
  Quadratic Stiffness likewise uses `8^-(3:5)` for its second problem. `work_precision_levels` has
  to carry both sets or half the published points have no counterpart.
- **S08 has no published work-precision curve at all.** Quadratic Stiffness builds
  `WorkPrecisionSet`s for `(50,1)` and `(500,1)` only; the mixed case `(5000,70)` appears solely in
  `@time` cells. §1's third stiff-set scheme, `ImplicitRKMil`, likewise appears in no
  `WorkPrecisionSet` on that page — only in `@time` cells and on the stochastic-heat page. §1's
  scheme list is a list of _mentioned_ schemes, not of compared ones.
- **The notebook's own stability limits are the `dt` the entries must declare.** Its `@time` cells
  run `(50,1)` at `dt = 0.01`, `(500,1)` at `0.002`, `(5000,70)` at `1e-5`, introduced as "the
  maximum dts which are allowed. Anything higher is mostly unstable." The harness cap
  `0.04/T = 0.0133` at `T = 3` is **coarser** than S06's stability limit, so each entry must declare
  its own `dt` and rely on the `min`.
- **The library's stoch-heat generator cannot run.** `generate_stiff_stoch_heat` calls `full(…)`
  (removed in Julia 0.7) and uses `mul!`, `Tridiagonal`, `WienerProcess` and `RSWM` without
  importing any of them, inside a module whose only import is `DiffEqBase: SDEFunction, SDEProblem`.
  It throws `UndefVarError` on call at v1.2.4. `StiffSDE/StochasticHeat.jmd` **defines its own
  working copy**, which also _has_ a closed form — `expm(At + WI)u₀`, valid in the **Stratonovich**
  sense, which is why that page benchmarks `EulerHeun`, `ImplicitEulerHeun` and
  `ImplicitRKMil(interpretation = Stratonovich)` and measures its errors at `t_end = 0.25`. §11's
  "no analytic" is wrong about the notebook version; the §4a.ii drop stands, on budget.
- **`prob_sde_bruss` is used by no notebook.** The weak page's "stochastic Brusselator" is a
  _different, inline_ problem: `dX₁ = [(p₁−1)X₁ + p₁X₁² + (X₁+1)²X₂]dt + p₂X₁(1+X₁)dW`,
  `dX₂ = −[…]dt − p₂X₁(1+X₁)dW`, `X₀ = [−0.1, 0]`, `p = [1.9, 0.1]`, `t ∈ (0,100)`, 100 paths,
  `dts = 2^-(1:4)`, `test_dt = 1/1000`. So §2's Phase-2 justification for the library's
  `prob_sde_bruss` — "SciML's own weak-error test problem" — attaches to the wrong problem. The
  notebook's version is also a **third** noise shape for H3: **scalar** noise, `m = 1 < d = 2`,
  where S09 is `m = d = 2` with a full matrix and the library's bruss is `m = 4 > d = 2`. And its
  noise cancels exactly between components, so `X₁ + X₂` obeys the noise-free
  `d(X₁+X₂) = −X₁ dt` — a structural invariant a wrong solver breaks, and a free constraint check of
  the kind `CONSTRAINT_VIOLATION` already exists for.
- **`prob_sde_bistable`** (`d = 1`, `m = 2` via `noise_rate_prototype = zeros(1,2)`, `p = (5,18,1)`,
  `u0 = [3]`, `t ∈ (0,300)`) is in the library, **is** exported, has no closed form and is in no
  notebook. It is the cheapest possible H3 exercise if one is wanted without a comparator.
- **Three library docstrings disagree with their own code**, worth knowing before any is quoted:
  `prob_sde_cubic`'s says `+¼u(1−u²)` where the code has `−0.25`; `prob_sde_lorenz`'s says
  `β = 8/3` where the problem is built with `2.66`; `prob_sde_2Dlinear`'s calls itself "8 linear
  SDEs" for a 4×2 state. §2's exclusion of `prob_sde_2Dlinear` stands.
- **Oval2 has a discontinuous drift**: `TGF0 = 0.5(t > 100)` inside `f`, so any order claim on it
  would be measured across a jump at `t = 100`, and its `σ` writes only 4 of 19 components (relying
  on the caller to pre-zero). Dropped anyway per §4a.ii; the reason is now two reasons.
- **The library's Brusselator diffusion is asymmetric in a way that reads like a typo**:
  `du[2,3] = 0.15·√(p₃·2.5·u₁)` against `du[1,3] = −0.15·√(p₃·u₁)`, i.e. a literal `2.5` alongside
  `p₃ = 2.5`. Transcribe as written and record it; do not "fix" it.
- **§2's blanket claim about the Stratonovich variants is half wrong — and the half that is wrong
  is the one §7 depends on.** `prob_sde_linear_stratonovich` and `prob_sde_2Dlinear_stratonovich` do
  share `f` and `g` with their Itô counterparts and differ only in `analytic`, as §2 says. But
  `stiff_quad_f_strat = −p₁(1−u²)` is a **different drift** (no `β²u`) paired with a
  **byte-identical** analytic function. That is self-consistent — the Itô drift is the Itô correction
  of the Stratonovich one, which is why both share a solution — and it is exactly why §7's proposed
  wrong-variant control works: dropping `β²` yields a problem whose exact solution is _the same
  formula_, so only integrating it with an Itô scheme reveals the error. **The control is sound, and
  its mechanism is subtler than §7 states** — it is not "the formula changes", it is "the formula
  does not change and the law does". §2's stated reason for excluding the Stratonovich entries
  ("indistinguishable from their Itô counterparts by problem text") is right for linear and wrong for
  stiffquad, where the text differs and the answer key does not; the exclusion still holds, for the
  stronger reason that a problem statement cannot say which interpretation it means.

### 2b. Step 6 as implemented — the nine entries `[rev4 2026-11-10, N]`

**Shipped: M01–M09 in `benchmark/problems.py`**, eight on the `pathwise` route and one
on `functional_truth`, with 66 tests in `benchmark/tests/test_sciml_suite.py` and a
calibration of every gate in `docs/sciml-benchmark-data/calibrate_sciml_suite.py`.
`validate_ground_truth.py` passes with the suite in it, and the other 66 problem
statements and their manifest entries are still **byte-identical**.

The entries as they ship. Note that `T` is SciML's for six of the nine and ours for
the stiff triple, and that the `dt` column is the step the harness actually grades at
(`sde_verify_dt`), which for every entry here is the entry's own declaration:

| # | slug | T | graded dt | paths | gate | fine grid | work-precision steps |
|---|---|---|---|---|---|---|---|
| M01 | `sde_sciml_linear_ito` | 1 | 1/625 | 1000 | 9.5e-02 | 62 500 | SciML's `5^-(2:6)` |
| M02 | `sde_sciml_additive` | 1 | 1/625 | 1000 | 1.2e-03 | 62 500 | SciML's `5^-(2:6)` |
| M03 | `sde_sciml_cubic` | 1 | 1/625 | 1000 | 4.9e-02 | 10 000 | ours (no published set) |
| M04 | `sde_sciml_wave` | 1 | 1/625 | 1000 | 5.7e-04 | 62 500 | SciML's `5^-(2:6)` |
| M05 | `sde_sciml_nltest` | 10 | 4e-03 | 1000 | 9.8e-03 | 10 000 | ours (no published set) |
| M06 | `sde_sciml_stiffquad_mild` | **0.01** | 1e-04 | 1000 | 1.5e-02 | 400 | ours (see finding 1) |
| M07 | `sde_sciml_stiffquad_det` | **0.001** | 1e-05 | 1000 | 3.9e-02 | 400 | ours |
| M08 | `sde_sciml_stiffquad_mixed` | **1e-04** | 1e-06 | 1000 | 1.1e-01 | 400 | ours |
| M09 | `sde_sciml_weak_noncommutative` | 3 | 2.5e-03 | 100 000 | per value | — | SciML's `1, ½, ¼, ⅛` |

**Every gate separates, and none was moved after seeing where a scheme landed.** Run
through the real sandbox against the shipped entries, four solvers one decision apart:

| entry | correct EM | one decision wrong | best constant | transcribed | gate clears correct by | separation |
|---|---|---|---|---|---|---|
| M01 | 0.0206 | 0.4600 | 0.7055 | 0.0 | 4.6x | 22.4x |
| M02 | 0.00026 | 0.2860 | 0.00455 | 0.0 | 4.5x | 17.2x |
| M03 | 0.0083 | 0.2897 | 0.6249 | 0.0 | 5.9x | 35.0x |
| M04 | 0.00008 | 0.0032 | 0.0281 | 0.0 | 7.3x | 41.5x |
| M05 | 0.0035 | 0.0275 | 0.1448 | 0.0 | 2.8x | 7.9x |
| M06 | 0.0074 | 0.0324 | 0.8949 | 0.0 | 2.0x | 4.4x |
| M07 | 0.0122 | 0.1260 | 0.5356 | 0.0 | 3.2x | 10.3x |
| M08 | 0.0434 | 0.2817 | 0.9984 | 0.0 | 2.5x | 6.5x |

The **narrowest separation is 4.4x** (M06), against BENCHOP's 6.8x — tighter, and said
rather than hidden. Each gate is the geometric mean of the worst correct scheme and the
best wrong one, rounded to two significant figures, so the margin is `sqrt(separation)`
on each side by construction rather than by choice. The `transcribed` column is 0.0 to
machine precision on all eight and is **excluded by name**, not passed.

The one-decision controls, which are *not* the ones §7 proposed on two of the nine:

| entry | the single decision changed |
|---|---|
| M01 | the Itô correction added to the drift (reading the solution as Stratonovich) |
| M02 | the ½ dropped from the mean-reversion rate |
| M03 | **the drift sign flipped — which is what the library's own docstring says** |
| M04 | `cos³` written as `cos` in the drift |
| M05 | the `⅓X^{1/3}` drift term dropped |
| M06, M08 | `β²` dropped from the drift (the Stratonovich twin) |
| M07 | diffusion `β(1−X)` instead of `β(1−X²)` |
| M09 | one increment shared across both noise channels |

#### Findings

**1. The stiff triple is graded in the transient, and the alternative was measured
before being rejected.** §2a.1 recommended moving the horizon; §2a.2 suggested instead
gating on SciML's own timeseries norm at `T = 3`, which would have kept the published
parameters *and* the published steps. **That alternative does not work, and the reason
is quantitative.** The transient occupies 0.3% of `[0, 3]`, so on an equally spaced
save grid a constant solver's relative timeseries error at `T = 3` is:

| saves | first save at | constant solver's relative timeseries error |
|---|---|---|
| 10 | t = 0.30 | **3.5e-13** |
| 100 | t = 0.03 | 2.7e-02 |
| 1000 | t = 0.003 | 8.6e-02 |

— inside any usable gate even at a thousand save points, because equally spaced saves
barely sample the transition. Making it bite needs saves at every step at `dt = 1e-5`,
i.e. 3×10⁵ points per path crossing the process boundary. So the horizon moves, and
`T` is a round number near `ln(3)/(2α)`, the transition midpoint, where the terminal
law is widest and a constant is therefore maximally wrong (its error is 0.89–1.00
there). The same rule for all three keeps them comparable with each other, which
matters because they are one equation at three parameter sets.

**What this costs, stated plainly:** M06–M08 are SciML's *equations* on a different
horizon, so their published `8^-(2:4)` steps are unusable (at `T = 0.01` a step of
1/64 is both coarser than the horizon and past the stability limit) and their
work-precision levels are ours. Combined with §2a.3 — M03 and M05 are in no notebook,
and M08 never had a published work-precision curve anyway — **five of the nine entries
have no published comparator steps**, and only M01, M02, M04 and M09 do. Any table has
to mark the difference between a rerun of a published comparison and a comparison we
constructed.

**2. Three entries were vacuous as the plan specified them, not one.** §2a.1 found the
stiff triple. Step 6's calibration found two more, by the same test applied to every
entry: *what does the best possible constant score?*

| entry | best constant's relative error | against the default 10% gate |
|---|---|---|
| M02 `additive` | **0.0046** | 22x **inside** it — a constant passes |
| M04 `wave` | **0.0281** | 3.6x **inside** it — a constant passes |

M02's terminal law has a spread of 0.5% of its magnitude, because its noise amplitude
is `αβ = 0.005`; M04's diffusion is `0.1cos²(X) ≈ 0.03` at `X(0) = 1`. Neither is
degenerate — both vary — but both are *answerable without simulating* at a gate set by
a universal number. Their gates are therefore 1.2e-03 and 5.7e-04, i.e. 80x and 175x
tighter than the default.

The general lesson, which is the one worth carrying: **a strong-error gate's scale is
set by the spread of the terminal law, not by a universal tolerance.** `SDE_VAR_TOL`
can be universal because a variance is normalised by itself; a strong error is
normalised by the solution's *magnitude*, and the ratio of those two is a property of
each problem. The default exists for a problem nobody has measured yet.
`validate_ground_truth.py` now refuses any `pathwise` entry whose best constant clears
its gate, and `test_sciml_suite.py` asserts it per entry, so this class cannot be
authored again.

**3. §7's one-decision control for M03 is not a wrong solver at all.** §7 proposes "a
dropped Milstein correction on S03". A dropped Milstein correction **is
Euler–Maruyama** — a correct scheme of lower order, not a wrong one — and it measures
0.0083 against Milstein's 0.00057, i.e. exactly the correct EM number. An accuracy gate
cannot separate it and must not be asked to; holding a plan to an order it *declared*
is the order study's job, not the strong-error gate's. The replacement is better than
the original: **the drift sign flipped, which is what `prob_sde_cubic`'s own docstring
says** (§2a.8 found the library's prose contradicts its code), so the control is the
single most plausible wrong reading there is, and it separates by 35x.

**4. §7's control for the stiff triple only bites where `β²` is comparable to `α`.**
Dropping `β²` from the drift turns the Itô problem into its Stratonovich twin, whose
exact solution is the *same formula* — which is what makes it an elegant control.
Measured, it separates on M06 (4.4x) and M08 (6.5x) and **not on M07** (0.0170 against
a 0.0200 gate, 1.4x from a correct scheme), because at `(α, β) = (500, 1)` the Itô
correction is 1/500 of the drift and the two problems are numerically near-identical.
M07 therefore ships with a different one-decision control, and the β²-drop is recorded
as measuring something real: **the Itô/Stratonovich distinction is only observable on
this family where the noise is stiff.**

**5. §10's prediction 4 is falsified, and the mechanism is the opposite of the one
predicted.** The prediction: "the Monte-Carlo noise floor at 10⁵ paths, not the scheme,
sets the resolvable gate" on the weak problem, and "expect the gate to need widening on
noise grounds". `X1` turns out to be **exactly lognormal** — its SDE involves only `X1`
and the single combination `(1/4)dW1 + (1/16)dW2` — so the floor is computable in closed
form rather than estimated: the relative standard deviation of `X1(3)²` is 1.1039, and

| paths | relative SE on `E[X1²]` | a 5% gate is |
|---|---|---|
| 25 000 | 0.698% | 7σ |
| 100 000 | 0.349% | 14σ |
| 1 000 000 | 0.110% | 45σ |

so no widening is needed. But the prediction was not simply wrong — it was right about
*which* quantity binds and wrong about the direction. A correct Euler–Maruyama's weak
error on `E[X1²]` at the graded step is **0.31%, which is below its own standard error
of 0.175%** at 4×10⁵ paths. So the noise floor does set that gate — by making the
scheme's error unmeasurable, not by making the gate too tight. The gate is 2.0%,
5.7σ clear of the floor, and the entry's other two observables (whose weak errors are
0.59% and 1.34%) are what actually resolve a scheme there.

That same fact broke the validator's first convergence check, which asked for the weak
error to halve with `dt` unconditionally and got 0.310% → 0.383% on `E[X1²]`. That is
two noise measurements, not a convergence failure, and the check now declines to assert
a ratio where the coarse error is inside 3x its own standard error — the discipline
`crn.weak_order` already applies for the same reason.

**6. M09's published value is derived rather than transcribed, and the derivation says
why one observable is not enough.** The second moments `m = (E[X1²], E[X1X2], E[X2²])`
satisfy a closed linear system `dm/dt = A m` from Itô's rule, with

```
A = [[ -1.0,       0.0,      0.0     ],
     [ -0.1142767, 0.8271296, 0.0    ],
     [  0.2189466, 0.0,      2.7167591]]
```

`A[0,0] = 2(−273/512) + (1/4)² + (1/16)² = −1` **exactly**, so `E[X1²] = exp(−t)` with
no quadrature anywhere and SciML's published `exp(−3)` is confirmed from the
coefficients. Two things that matrix shows:

- **`A`'s first row is `[−1, 0, 0]`: `E[X1²]` decouples completely.** It is a function
  of component 1's *total* diffusion variance and of nothing else — not of the two
  channels being separate, and not of the non-commutativity. SciML grades `h(z) = z²`
  on component 1, so **the published observable on this problem cannot distinguish the
  true two-channel noise from a single channel of matched variance.**
- `E[X1X2]` and `E[X2²]` do not decouple; their rows couple through `(1−2√2)/4` and
  `1/10`, the coefficients of the two *different* increments in row 2. Those are the
  observables that require the channels to stay separate.

So the entry carries all three, exactly known, each with its own gate, validated by a
route that shares nothing with it: a direct Euler–Maruyama Monte Carlo of 400 000
paths, Richardson-extrapolated across two nested step counts to separate the weak bias
from the noise, agreeing at **1.6σ, 0.7σ and 1.6σ**.

**7. M09 must *not* be handed increments, and `noise_dimension` cannot express what
makes it interesting.** Two limits on H3, both found by authoring the entry it was
built for:

- The high-order weak schemes SciML compares here — DRI1, PL1WM, RDI1WM, RDI2WM — draw
  auxiliary randomness per step and **cannot consume a supplied Gaussian increment**.
  Supplying `dW` would exclude the very methods the published page is a comparison
  between. It is the same structural fact `exact_sampling_components` exists for on the
  exact-sampling side. So M09 stays on the moment route, and H3's `(paths, Nt, m)`
  promise applies to the eight pathwise entries only.
- M09 is 2 states driven by 2 channels, so `m == d` and `noise_dimension` is
  indistinguishable from the implicit default every 2-D problem already had. **The
  field says how many channels there are, not whether the diffusion matrix is
  diagonal**, and non-commutativity is the latter. The manifest now records a
  *declared* `noise_dimension` even where it equals the state dimension, which at least
  preserves the statement that someone decided it; expressing the structure itself
  would need a field H3 does not have.

**8. Two defects in step 5's own work, both caught by step 6's calibration.**

- **The exclusion was read before the pass decision, so it rescued failing solvers.** A
  constant solver's ladder is degenerate too — a constant does not change with `dt`
  either — so `ANALYTIC_TRANSCRIPTION` was returned for solvers that had *also* missed
  their gate by up to 60x, discarding the fact that they were wrong. The exclusion now
  applies only to a run that would otherwise have passed: being accurate and not a
  scheme is a different thing from being wrong.
- **Only the pipeline verdict knew about the exclusion.** `compute_oneshot_verdict`
  maps the *baseline* path and had no branch for it, so a transcribing C0 baseline
  would have been recorded as a clean PASS on all eight pathwise problems — and §10's
  prediction 2 is precisely that the C0 baseline is **where transcription is most
  likely**, since a single completion with no tools is the setting in which writing
  down the closed form is the obvious move. Step 9 would have measured the opposite of
  what it set out to. Both verdict functions now read one shared helper.

**9. The nesting check earned its keep immediately.** M03's first draft reused M05's
work-precision steps at a tenth of its horizon, giving `dt = 0.016` at `T = 1` — a step
count of 62.5, which is not an integer and does not divide the fine grid.
`validate_ground_truth.py` refused the entry before any solver ran.

---

## 3. Environment setup `[rev1 2026-11-09]`

Nothing here needs a model call, and none of it touches the pipeline. **Julia is not installed on
this machine** `[R]`; everything in this section is prerequisite to step 3 of §7.

### 3a. Julia, pinned

~~`[W]` The three published pages ran on three different version sets (§1a.4), so there is no
"current" set to inherit — pick one and record it. Use the newest of the three, because two of the
three pages already ran on it:~~

**Rev 2: there is a current set to inherit, and inheriting it is strictly better than choosing one.**
`benchmarks/NonStiffSDE/Manifest.toml` and `benchmarks/StiffSDE/Manifest.toml` are committed upstream
and **agree with each other**, so one manifest reproduces all five notebooks. Read off `master` on
2026-11-10 `[N]`:

```
# Inherited verbatim from the upstream benchmark manifests, which agree across both folders.
julia 1.11.9            # julia_version in both Manifest.toml files
StochasticDiffEq     7.2.0     # not 7.1.5
SDEProblemLibrary    1.2.4     # confirmed: the version step 2 read
DiffEqNoiseProcess   5.36.3     # not 5.36.1
DiffEqDevTools       3.6.3     # not 3.6.0
SciMLBase            3.54.0
SciMLLogging         2.1.0     # needed for `verbose = SciMLLogging.None()`; omitted in rev 1
```

Take the upstream `Manifest.toml` itself rather than re-resolving: `Pkg.add` at these versions
resolves a _different_ dependency closure than the one the published pages ran on, and the manifest
is the only artefact that fixes the closure. ~~The `Pkg.add` recipe below stays as the fallback for
when the upstream manifest will not instantiate on this machine.~~ **Rev 5: the fallback is
withdrawn, not merely unused (§3e).** The manifests instantiate verbatim on a platform neither
published page ran on, and the recipe below pins the *stale* versions rev 1 read off the pages —
`7.1.5`, `5.36.1`, `3.6.0` — so falling back to it would have resolved a closure matching neither
the manifests nor any published run, while looking pinned. It is kept below only as the record of
what rev 1 proposed.

Install juliaup rather than a system package, so the version is pinnable and reproducible:

```bash
curl -fsSL https://install.julialang.org | sh      # juliaup
juliaup add 1.11.9 && juliaup default 1.11.9
julia +1.11.9 --version                            # must print 1.11.9
```

Then build a project **outside** the repo working tree first (it is third-party code; treat it as
untrusted data per the download rules), and commit only the resolved manifest:

```bash
mkdir -p ~/sciml-comparators && cd ~/sciml-comparators
julia +1.11.9 --project=. -e '
  using Pkg
  Pkg.add([Pkg.PackageSpec(name="StochasticDiffEq",  version="7.1.5"),
           Pkg.PackageSpec(name="SDEProblemLibrary",  version="1.2.4"),
           Pkg.PackageSpec(name="DiffEqNoiseProcess", version="5.36.1"),
           Pkg.PackageSpec(name="DiffEqDevTools",     version="3.6.0")])
  Pkg.status(); Pkg.instantiate()'
```

Copy the resulting `Project.toml` and `Manifest.toml` to
`benchmark/references/sciml/` and commit both. The manifest is the only thing that makes a rerun
reproducible, and it is small.

### 3b. Single-threaded, and no compilation in the clock

`[W]` Two conventions, both forced by what the published pages got wrong:

- **Threads.** The published runs used 128 threads; this machine has 8 cores `[R]`. Run the
  comparators with `JULIA_NUM_THREADS=1` and record it. A threaded comparator against a
  single-process numpy solver is not a measurement of anything.
- **Compilation.** Page 3's `@time` numbers are up to 99.98% compilation `[W]`. Every timed call is
  run once to warm up and then timed over at least 5 repetitions, reporting the **minimum**, with
  `@belapsed` (BenchmarkTools) or an explicit warm-up loop. Discard any number from a first call.

### 3c. The harvest script

`[N]` One Julia script, `benchmark/references/sciml/harvest.jl`, committed, that writes
**numbers** where the pages have plots. For each (problem, scheme, fixed `dt`) it records:

| Field                                                          | Why                                                               |
| -------------------------------------------------------------- | ----------------------------------------------------------------- |
| `problem`, `scheme`, `dt`, `paths`                             | the key                                                           |
| `strong_err_l2`, `weak_err`                                    | the accuracy axes (`:l2` and `:weak_final`, matching §1)          |
| `steps` = `round(T/dt)`, `evals` = `steps × stages(scheme)`    | the **work** axis — language-free (§5, H4)                        |
| `seconds_min_of_5`, `threads`, `julia_version`, `pkg_versions` | the anchored-time axis plus its caveats                           |
| `seed`                                                         | pages 1 and 3 set none `[W]`; this script sets one and records it |

It writes `benchmark/references/sciml/comparators.json`. Two rules about where that file lives,
both copied from the BENCHOP answer key `[R]`: it goes under `benchmark/references/`, **never** in
`workspace/`, and it is added to the `pipeline-settings.json` deny rules and to the leakage scan
(**H9**) — it contains both the schemes the pipeline is compared against and the error values it is
graded near.

The script must also record, per cell, whether it reproduced the published result or hit the
`BoundsError` of §1a.2. "Still broken at the pinned version" is a finding worth committing.

### 3d. Python side

Nothing new: `uv run python benchmark/...` as today. The EM anchor (§5, step 4) is two files — one
`.jl` and one `.py` implementing the _same_ Euler–Maruyama on S01 — committed beside the harvest
script so the ratio can be re-measured on other hardware.

---

### 3e. Step 1 as implemented — the pinned environment `[rev5 2026-11-10, N]`

juliaup installed from the official script; `juliaup add 1.11.9 && juliaup default 1.11.9`;
`julia +1.11.9 --version` prints `julia version 1.11.9` `[R]`. Note that juliaup installs
its `release` channel first, which on 2026-11-10 is **1.13.3** — so the pin is not
cosmetic, and every command in this track names `+1.11.9` explicitly rather than
trusting the default.

**Rev 2's instruction was right, and the reason was stronger than it knew.** Both
upstream `Manifest.toml` files instantiate **verbatim** on aarch64-darwin — a platform
neither published page ran on — and `Pkg.instantiate()` left both byte-identical
(md5 unchanged before and after). 396 dependencies precompiled in 283 s for the
non-stiff project; the stiff one needed nothing further, and the reason is the finding:

**The stiff closure is a strict subset of the non-stiff one.** Of 293 packages present in
both manifests, **zero** disagree on version; the non-stiff manifest holds 39 more
(`ParameterizedFunctions`' tree) and the stiff manifest holds none the non-stiff one
lacks. Rev 2 said the two manifests "agree with each other"; what is true is stronger —
*one of them contains the other* — so a single project reproduces all five notebooks
with the same bytes of every package, and that is a fact about the dependency graph
rather than a judgement about which versions matter. Both pairs are committed
(`benchmark/references/sciml/nonstiff/`, `.../stiff/`) because the two `Project.toml`
files differ in their *direct* deps and reproducing the stiff notebook needs its own;
the harvest runs under `nonstiff/`.

Every version §3a predicted is confirmed `[R]`: `StochasticDiffEq 7.2.0`,
`SDEProblemLibrary 1.2.4`, `DiffEqNoiseProcess 5.36.3`, `DiffEqDevTools 3.6.3`,
`SciMLBase 3.54.0`, `SciMLLogging 2.1.0` on `julia 1.11.9`, plus
`OrdinaryDiffEqCore 4.17.2`.

**The `Pkg.add` fallback in §3a is now struck through, because taking it would have been
a mistake rather than a fallback.** Its version pins are the *stale* ones read off the
published pages in rev 1 — `StochasticDiffEq 7.1.5`, `DiffEqNoiseProcess 5.36.1`,
`DiffEqDevTools 3.6.0` — so a run that fell back to it would have resolved a closure that
matches neither the manifests nor any published page, while looking pinned. The manifest
is not the *preferred* route to a reproducible environment here; it is the only one.

#### The `BoundsError` cells of §1a.2: the three it names now run; the fourth is unidentified

Run cell by cell in one session at `JULIA_NUM_THREADS=1`, with `GKSwstype=100` so GR is
headless. §1a.2 says *four* cells fail but names only three — "the 2-D linear and scalar
linear first cells, and the wave first cell", which are cells 6, 8 and 10 of
`BasicSDEWorkPrecision.jmd`. **All three pass** (84 s, 34 s, 27 s), as do cells 1–5
(~19 minutes together).

**The fourth cannot be matched to a cell from the published page text**, and the three
candidates left — cells 7, 9 and 11, the adaptive-set cells — were **not** run to
completion: cell 7 ran for over an hour without either finishing or raising, and was
stopped. So what is established is that the three named failures are fixed upstream, not
that the whole page now runs. The cost is adaptive timestepping, which §11 excludes from
the comparison anyway, so the honest reading is that **the §1a.2 repair is no longer
worth doing** — three of the four named cells demonstrably work and the harvest's value
is the *numbers*, not the repair — rather than that page 1 has been fully revalidated.

The other four notebooks were run in full and pass completely: `QuadraticStiffness`
**8/8**, `HighOrderWeakSDEWorkPrecision` **7/7**, `BasicSDEWeakWorkPrecision` **20/20**
(weave footers skipped in each). `StochasticHeat` was not run: §4a.ii drops its problem
from the adoption.

#### A live defect with the same signature, on a different code path `[R]`

Found while writing `harvest.jl`, and it is why that script records terminal errors
rather than SciML's own timeseries `:l2`:

> `saveat` with **three or more** save points, on an `SDEProblem` that carries an
> `analytic`, raises `BoundsError: attempt to access 2-element Vector{Float64} at index
> [3]` inside `SciMLBase.calculate_solution_errors!` (`rode_solutions.jl:221`) at
> SciMLBase 3.54.0 / StochasticDiffEq 7.2.0.

The mechanism is that the function indexes the saved noise `sol.W.W` with the
*solution's* save index, and `saveat` desynchronises the two. One and two save points
work; three do not; the index is always `[3]` into a 2-element vector however many
points are requested; and `calculate_error = false` does **not** suppress it, so the only
way past it is to solve a problem that carries no `analytic` and evaluate the closed form
oneself. `DiffEqDevTools`' `WorkPrecisionSet` is unaffected because it saves every step
rather than using `saveat`, which is exactly why the notebook cells pass while a direct
three-point `saveat` solve of the same problem does not.

**The shape is the §1a.2 shape** — an `n`-element vector indexed at `n+1`, against a
published `attempt to access 4-element Vector{Float64} at index [5]`. Whether it is the
same defect surviving a refactor onto a different path cannot be settled without
resolving the 6.87.0 closure the published page ran on, which this track does not do. It
is recorded in `comparators.json` under `upstream_defects` with its mechanism, its
versions and what it cost here, and the honest statement is that the published failure
does not reproduce and a failure with its signature is live elsewhere.

---

### 3f. Step 3 as implemented — `comparators.json` `[rev5 2026-11-10, N]`

**279 rows**, every one `status: ok` — no scheme failed on any problem at any step. One
row per (problem, scheme, `dt`) over the nine entries, 19 distinct schemes, and each
entry's own published step grid. `harvest.jl` is 800 lines, of which ~230 are a JSON
reader and writer: the pinned environment is inherited verbatim and contains no JSON
package, and adding one would rewrite `Manifest.toml` and destroy the one property that
makes the run reproducible (§3e). That trade was worth naming rather than silently
taking the other side of.

**The two stacks are provably solving the same problems.** Before any scheme runs,
`harvest.jl` evaluates the library's own `analytic` at nine sample points per entry —
three times over the horizon × three Brownian values including `W = 0` and both signs —
and compares them to this repo's closed forms, read from `harvest_spec.json` (generated
from `problems.py`, so the configuration cannot drift either). **all 72 checks pass,
worst relative difference 2.18e-16.** The run *aborts* rather than warns if they
disagree, because a comparator table between two different problems still looks like a
comparison. This is the join the plan did not have: §2a confirmed each closed form
solves its own equation, and this confirms the Julia problem is that equation.

#### Finding 1: measuring the stage counts was not a refinement, it was a correction

§5 says the work axis should use "the published stage count for each Rößler scheme".
`harvest.jl` measures them instead — wrapping `f` and `g` in counters and differencing
evaluation totals over 4 and 8 steps, so start-up costs cancel. The two anchors §5 names
come out exactly right, which is what validates the method:

| scheme | f/step | g/step | **stages** | |
| ------ | ------ | ------ | ---------- | - |
| EM, SimplifiedEM | 1 | 1 | **1** | §5 says 1 ✅ |
| RKMil | 1 | 2 | **2** | §5 says 2 ✅ |
| SRA1, SRA2, RDI1WM | 2 | 1–2 | **2** | |
| SRA3, SOSRA, SOSRA2 | 3 | 3 | **3** | |
| ImplicitEM, ImplicitRKMil | 3 | 1–2 | **3** | no fixed count: the solve iterates |
| SRIW1 | 2 | 4 | **4** | |
| SRIW2, SOSRI, SOSRI2 | 4 | 4 | **4** | |
| **DRI1, PL1WM, RDI2WM** | 2–3 | **9** | **9** | the weak schemes |
| **SRI** | 10 | 10 | **10** | |

The last two rows are the correction. `SRI` costs **10** evaluation rounds per step and
the three high-order weak schemes cost **9**, driven by `g` rather than `f` — they need
many diffusion evaluations to handle non-commutative noise. A stage count taken from a
paper's tableau would have put most of these at 2–4, under-charging the most expensive
schemes in the set by up to 5× on the axis the whole comparison is drawn against. The
measured counts are in every row of `comparators.json` beside `f_evals_per_step` and
`g_evals_per_step`, so a reader can check the scaling rather than trust it.

#### Finding 2: the work axis and the seconds axis disagree, and in the pipeline's favour

This is the plan-changing one. §5 calls work "a currency no runtime can inflate". True
across languages — and **false as a proxy for time within one**. Measured at each
entry's finest run, against Euler–Maruyama on the same entry:

| scheme | stages (× EM) | seconds (× EM) | the `evals` axis over-charges by |
| ------ | ------------- | -------------- | -------------------------------- |
| RKMil | 2.0× | 1.0–1.5× | 1.3–2.0× |
| SRIW1 / SOSRI / SRIW2 | 4.0× | 1.3–2.2× | 1.9–3.0× |
| **SRI** | 10.0× | **1.6–3.3×** | **3.0–6.2×** |
| **DRI1, RDI2WM** | 9.0× | **1.6×** | **5.7–5.8×** |
| ImplicitEM / ImplicitRKMil | 3.0× | 2.7–3.1× | **0.95–1.1×** |

A 10-stage scheme costs 1.6× the seconds of a 1-stage one. The mechanism is that these
right-hand sides are three or four flops, so the per-step cost is loop overhead and
noise generation — fixed costs a stage count does not see. The implicit schemes are the
control: their three drift evaluations are three *nonlinear-solve iterations*, real work,
and their over-charge is 1.0×.

**Why this matters for step 10 rather than just being a curiosity.** The pipeline will
ship a 1- or 2-stage scheme (§10 prediction 1), so the `evals` axis *flatters the
pipeline* by up to 6× against the schemes it is being compared with. Reporting work
alone would understate the gap; reporting seconds alone would be the cross-runtime
comparison §0c refuses. Both, with this table beside them, is the only honest form —
which is what §5 already prescribed, now for a measured reason rather than a cautious
one, and with the **direction** of the bias known.

#### Finding 3: §10's prediction 1 is not precise enough to be graded, and the two
readings differ by two orders of magnitude

Prediction 1: *"beaten by SRA1/SRIW1/DRI1 at matched work by ≥ 10× at relative error
10⁻⁴"*. That sentence holds two axes fixed at once and they cannot both be. Measured both
ways, on the same 279 rows:

**Reading A — hold work fixed, compare accuracy** (at Euler–Maruyama's finest run):

| entry | EM's error | best other scheme | its error | accuracy gain |
| ----- | ---------- | ----------------- | --------- | ------------- |
| M01 linear | 4.19e-3 | SRIW2 | 2.63e-6 | **1593×** |
| M02 additive | 1.06e-5 | SRA3 | 6.27e-12 | 1.7e6× *(roundoff floor)* |
| M03 cubic | 4.29e-3 | SRIW2 | 6.72e-6 | **638×** |
| M04 wave | 1.47e-5 | SRIW2 | 6.84e-11 | 2.1e5× *(roundoff floor)* |
| M05 nltest | 1.00e-3 | SOSRI2 | 8.91e-6 | **113×** |
| M06 stiff mild | 2.51e-3 | SRIW1 | 3.82e-5 | **66×** |
| M07 stiff det | 3.09e-3 | SRIW1 | 2.37e-5 | **131×** |
| M08 stiff mixed | 2.13e-2 | SRIW1 | 5.86e-4 | **36×** |
| M09 weak | 1.11e-1 | RDI1WM | 3.68e-2 | **3.0×** |

**Reading B — hold error fixed at 10⁻⁴, compare work** (which is prediction 1's own
phrasing; `>` means Euler–Maruyama never got there on the published grid, so the figure
is a lower bound):

| entry | EM's work | best other | its work | work saved |
| ----- | --------- | ---------- | -------- | ---------- |
| M01 | > 1.6e7 | SOSRI | 2.5e6 | > 6.2× |
| M02 | 3.1e6 | SRA1 | 5.0e4 | **62.5×** |
| M03 | > 2.5e6 | SOSRI2 | 5.0e5 | > 5.0× |
| M04 | 6.3e5 | RKMil | 5.0e4 | **12.5×** |
| M05 | > 1.0e7 | SOSRI2 | 2.5e6 | > 4.0× |
| M06 | > 4.0e5 | SRIW1 | 4.0e5 | > 1.0× |
| M07 | > 4.0e5 | SRIW1 | 2.0e5 | > 2.0× |
| **M08** | > 4.0e5 | SRIW1 | **1.6e6** | **0.2×** |
| M09 | > 2.4e6 | *none reached 10⁻⁴* | — | — |

So prediction 1 is **confirmed under reading A on all nine entries** (36×–1.7e6×, far
past the ≥10× it claimed) and **falsified under reading B on four of nine** — 6.2×, 5.0×,
4.0× and 1.0× fall short of 10×, and **on M08 the best Rößler scheme needs five times
more work than Euler–Maruyama to reach 10⁻⁴**. Both tables come from one set of rows; the
difference is entirely which axis is pinned. Reading A compares at the far end of the
grid where an order-1.5 scheme has had maximum room to pull away from an order-0.5 one;
reading B compares at a fixed error, where the five-fold step grid's granularity and the
4× stage cost dominate. **The lesson for step 10 is to fix the axis in advance and quote
both tables**, and the lesson for §10 is that a prediction naming two axes is not a
prediction.

M08 is the most interesting row in either table, and it is the one §10 prediction 3
anticipated: at high noise stiffness (`β² = 4900` against `α = 5000`) the high-order
scheme's order advantage is gone and its 4× per-step cost is all that is left. SciML's
own text says nothing wins there `[W]`; measured, on the work axis, **Euler–Maruyama
wins there**.

#### What is in the artefact

`benchmark/references/sciml/comparators.json` (224 KB): per row the key
(`problem`, `scheme`, `dt`, `paths`, `seed`), three error columns, the work columns
(`steps`, `f_evals_per_step`, `g_evals_per_step`, `stages`, `evals`), the time columns
(`seconds_min` over 5 reps after a discarded warm-up, `paths_timed`,
`ns_per_path_step`), and for M09 all three observables with their Monte-Carlo standard
errors and the error in units of those. Plus `environment` (julia 1.11.9,
`threads=1`, apple-m1, every package version), `analytic_cross_checks`,
`notebook_reproduction` (§3e's per-cell log, folded in so the artefact is
self-contained), and `upstream_defects`.

`docs/sciml-benchmark-data/comparator_tables.py` produces every table above from it; the
tables are not transcribed by hand.

**Deliberately not recorded: SciMLBenchmarks' own `error_estimate = :l2`.** It is an
*absolute* timeseries error (§2a.2), so it is not the suite's gate, and collecting it
needs `saveat` — which raises the §3e `BoundsError` at the pinned version. What is
recorded instead is `sde_err_final_abs`, SciMLBase's own `sol.errors[:final]`, which
agrees with this script's independently computed absolute terminal error to a ratio of
**exactly 1 on all 255 pathwise rows** (worst `|ratio - 1|` = 0). That is the cross-check that says the script is
reading the right field.

---

## 4. Harness work list `[rev1 2026-11-09; H1–H4, H8–H10 implemented rev3 2026-11-10 — see §4b]`

Ten items. **H1–H4 and H8 are Phase 1 and are the minimum to run anything**; H5–H7 are Phase 2;
H9–H10 are hygiene. Line numbers are from the tree at 2026-11-09 `[R]`.

**Seven of the ten are shipped.** §4b is the implementation record and is the authority on what each
one turned out to need; the item descriptions below are left as written so the plan can be graded
against the outcome rather than re-narrated. H5, H6 and H7 are Phase 2 and untouched.

### H1 — a pathwise (strong-error) truth route · **Phase 1, required**

Today `verify.py` grades terminal moments, PDE fields, scalar functionals and stability. There is no
route that compares a solver's path-by-path output to an exact solution. SciML's primary axis is
exactly that.

- **New `ground_truth_kind = "pathwise"`**, documented in the `problems.py` module docstring
  alongside `exact` / `reference` / `stability` (`benchmark/problems.py:19-41`).
- A problem of this kind carries `pathwise_exact(u0, p, T, W_T) -> ndarray`, where `W_T` is the
  summed supplied increments. All nine Phase-1 entries satisfy this with a one-line formula (§2).
- `verify.py` supplies `dW` to the solver (the contract already accepts it —
  `verification_manual.md` §18, `sandbox._crn_increments` at `verifylib/kernel/sandbox.py:206`),
  sums it, evaluates `pathwise_exact`, and gates the **relative `l2` strong error** over paths.
- **Build the ladder's increments from one fine draw, aggregated down** — reuse
  `crn.ladder_spec`'s `aggregate` trick (`verifylib/kernel/sde/crn.py:30-45`) rather than drawing
  per level, so every `dt` on the curve is driven by the same Brownian path and the levels are
  comparable rather than independently noisy.
- `setup.py`'s `contract_footer` gains the sentence that says so (`benchmark/setup.py:64-175`), and
  the existing rendering for non-pathwise SDE problems must stay **byte-identical** — the rule that
  file already states in a comment, because churn in a problem statement makes an old run
  incomparable for a reason unrelated to the solver.
- **It must extend to the work-precision path too**: `_fold_work_precision_sde`
  (`verify.py:1674`) and `_wp_solve_sde` (`verify.py:1570`) compute a _moment_ relative error
  today; for a pathwise problem the series point must carry the strong error, or the comparator
  table (§5) has nothing to compare.
- Acceptance: on S01, a hand-written EM reads strong order ≈ 0.5 and a Milstein ≈ 1.0 — checkable
  against the repo's own measured 0.488 / 1.006 on GBM `[R]`. A solver returning the exact formula
  reads a strong error at the rounding floor (which H2 then rejects).

### H2 — the degenerate-ladder verdict (`ANALYTIC_TRANSCRIPTION`) · **Phase 1, required**

§0b.2: the detector exists, the verdict does not. Without this, H1 grades transcription.

- `verify.py` records, per problem, whether the terminal states are identical (to rounding) across
  CRN levels. The reading already has a precedent: `s0 = 9.4e-15` on `sde_bm_with_drift` `[R]`.
- New verdict string threaded through `report.py`: `VERDICT_LABEL` (`benchmark/report.py:35-47`),
  `_VERDICT_ORDER` (`:53`), and `compute_verdict` (`:160-189`), which currently has no branch for
  it. It is **not** a pass and **not** a failure — it is an exclusion from the scheme-matched arm,
  so it must not land in `_PASS_VERDICTS` (`:51`) and must not be counted as a FAIL in the
  discrepancy table (`:494`).
- Acceptance: the pathwise-exact solver of H1's acceptance test is reported
  `ANALYTIC_TRANSCRIPTION`; a correct EM at the same `dt` is not; a _wrong_ solver is still `FAIL`
  rather than an exclusion.

### H3 — a noise dimension on benchmark problems · **Phase 1, required for S09**

`[R]` `verifylib/schema.py:372` has `check_noise_dimension` on the _plan spec_ side, but
`benchmark/problems.py` entries have no `noise_dimension` field at all: every existing SDE problem
has `m = d` implicitly. S09 is 2 states × 2 noise sources with a **non-commutative** diffusion
matrix, and `prob_sde_bruss` is 2 × 4.

- Add `noise_dimension` to the problem entry, defaulting to `state_dimension`.
- Thread it into `setup.py`'s contract text (the `dW` shape the solver is promised is
  `(num_paths, Nt, m)`, not `(num_paths, Nt)`), into H1's increment generation, and into
  `validate_ground_truth.py`.
- Acceptance: S09's brief states the `(paths, Nt, 2)` shape; `crn.ladder_spec(..., m=2)` returns
  payloads of that shape; a solver that uses one increment for both noise sources fails — the
  defect `verifylib/kernel/sde/driver.py:1016` already describes but which no problem has exercised.

### H4 — a work-measured cost axis · **Phase 1, required**

§0c: wall-clock alone is a language benchmark. The work-precision point currently records only
`rel_err`, `solve_s`, `wall_s`, `graded` (`_wp_point`, `verify.py:1504-1511`).

- Add `steps` (`round(T/dt)`), `paths`, and `evals` to each point. `evals = steps × stages`, where
  `stages` is the scheme's declared stage count read from the plan frontmatter (`scheme_family` is
  already declared there `[R]`) — **not** self-reported by the solver, and cross-checked against the
  measured per-step cost so a mis-declaration is visible rather than load-bearing.
- `_fmt_work_precision` (`verify.py:1920`) and `report.py` render the new columns.
- Acceptance: a series on S01 at four `dt` values shows `evals` scaling exactly 1:2:4:8 for a
  one-stage scheme, and the comparator table (§5) can be built from `comparators.json` and the
  series without either side quoting seconds.

### H5 — general state dimension · **Phase 2**

`[R]` A hard ceiling at `d = 3`, in eight places: `["X","Y","Z"][:d]` at `problems.py:263`,
`setup.py:80`, `verify.py:333`, `validate_ground_truth.py:381`, and `("X","Y","Z")` at
`verifylib/kernel/sde/driver.py:727` and `:790`, `verifylib/kernel/sde/dynkin.py:260`,
`verifylib/reference.py:580`.

- Replace with a generated labelling, exactly as `SYSTEMS_3D_EXTENSION_PLAN.md` did for PDE
  `fields` `[R]`. Keep the `X`/`Y`/`Z` spelling for `d ≤ 3` so no existing problem statement or
  archived record changes.
- Acceptance: `prob_sde_additivesystem` (`d = 4`) runs end to end; all three existing suites'
  manifests and problem statements are byte-identical before and after.

### H6 — the converged-reference route · **Phase 2, largest item**

§2's fourth truth route: a harness-owned fine-step reference, compared under common random numbers,
validated by two independent integrators agreeing far below the gate. Reuses `crn.ladder_spec`.
Acceptance: agreement at least two orders below the gate on `lorenz` and `bruss`, recorded, with the
second integrator named — the no-closed-form PDE suite's standard `[R]`.

### H7 — the step rule and the solver budget · **see §4a; blocks three Phase-2 problems**

### H8 — suite registration · **Phase 1, required, trivial**

`SUITES = ("main", "no-closed-form", "benchop")` at `benchmark/setup.py:434` gains `"sciml"`;
`problem_suite` (`:437`) and `apply_suite_filter` (`:447`) need nothing else; the `--suite` choices
(`:497`) pick it up automatically; `n_by_suite` (`:410`) reports it. The runner refusal that stops
one suite's records merging into another's baseline already covers it `[R]`. This is the one piece of
the adoption that is genuinely already built, and it is built because step 9 of the BENCHOP plan
found four defects here the hard way.

### H9 — leakage scan coverage · **Phase 1, required**

Extend `verifylib/leakage.py` to the new filenames (`benchmark/references/sciml/*`) and add the
directory to `benchmark/pipeline-settings.json`'s deny rules. §8 says why: those files hold both the
competitor schemes and the graded error values.

### H10 — the arm switch · **Phase 1, required**

§6's two arms need to be a recorded field, not a convention: each result carries which arm it was
run under, so the two can never be averaged together in a report. Acceptance: `compare.py` refuses
to put unrestricted-arm and scheme-matched-arm records in one column.

### 4a. Three findings from reading the harness `[rev1, N]`

**(i) The graded step is forced finer than every published point.**
`sde_verify_dt = min(problem["dt"], 0.04 / problem["T"])` (`verify.py:271-278`) `[R]`. The `min`
takes the _smaller_ value, so declaring a fine `dt` for a stiff problem works — S08 can declare
`dt = 1e-5` and be graded there. But the rule also _caps_ the graded step at `0.04/T`, and on S09
(`T = 3`) that is `0.0133`, while SciML's weak-error points are `dt = 1, 1/2, 1/4, 1/8` `[W]` — **all
four published points are coarser than the harness will ever grade at.** The same is true of the
coarse end of pages 1 and 3.
**Resolution, and it needs no code:** grade at the rule's step, and put the published steps in
`work_precision_levels`, which accepts arbitrary `(num_paths, dt)` pairs and is explicitly
_measurement, never grading_ `[R]`. The comparator table is then built from the **series**, not from
the verdict. This is why H1 must extend to the work-precision path and why H4 exists; without both,
the suite produces a verdict at a step no comparator ran at and no table at all.

**(ii) Two Phase-2 problems exceed the solver budget by orders of magnitude.**
`DEFAULT_SOLVER_TIMEOUT_S = 300`, `DEFAULT_SOLVER_MEM_MB = 4096` (`verify.py:78-79`) `[R]`. With the
step rule above: `prob_sde_oscilreact` (`T = 4000`) is graded at `dt ≤ 1e-5` — **4×10⁸ steps per
path**; `oval2ModelExample` (`T = 500`) at `dt ≤ 8e-5` — 6.25×10⁶ steps × 19 states. Neither fits.
`prob_sde_bruss` (`T = 100`, `dt ≤ 4e-4`, 2.5×10⁵ steps) does fit. **Decision: drop `oscilreact`,
`oval2` and the stochastic heat equation from the plan** unless a per-problem budget and a
justified step override are added, and say so in the result rather than discovering it in a run.
That removes SciML's headline stiff result from the comparison, which is a real loss and is the
strongest argument for Phase 2 being a separate plan.

**(iii) Phase 1's path counts are comfortable.** S01–S08 at SciML's `N = 1000` (strong) and S09 at
`numtraj = 1e5` (weak) `[W]`, against the repo's usual 10⁶–4×10⁶ `[R]`, are small. S08 at
`dt = 1e-5`, `T = 3`, 1000 paths is 3×10⁵ vectorised numpy steps on a 1000-element array — seconds.
The budget risk in Phase 1 is the work-precision _sweep_ multiplying the count, not any single solve.

### 4b. Step 5 as implemented `[rev3 2026-11-10, N]`

**Shipped: H1, H2, H3, H4, H8, H9, H10**, with the acceptance test of each in
`benchmark/tests/test_pathwise_route.py` (28 tests). H5-H7 remain Phase 2 and are
untouched. The existing suites are unaffected, measured rather than asserted: all **66
problem statements and the entire manifest hash identically** before and after
(`sha256 984bcb4e…`), and the invariant that produces that is a test rather than the
hash, so a legitimate future edit to one problem does not read as a regression here.

Unlike §2a, implementing this mostly confirmed the plan. Four things it did not:

**1. The benchmark's SDE contract did not accept `dW` at all, so H1 was bigger than
one route.** H1 said "the contract already accepts it -- `verification_manual.md` §18,
`sandbox._crn_increments`", and §18 does specify `dW`, but **that is the kernel's
contract, not the benchmark's**. `benchmark/setup.py`'s `contract_footer` rendered
`solve_sde(num_paths, dt, T, seed)` with no `dW` on every problem, and neither
`_run_and_read_sde` nor `runner.run_solver` had any way to supply one. So H1 needed
three things rather than one: the route, a contract extension rendered _only_ for the
new kind, and increment plumbing through the sandbox child. The child builds the
increments from a seed-and-shape payload rather than receiving an array, which is
`verifylib/kernel/sandbox.py`'s own convention; the parent builds the same payload to
evaluate its truth on the very path the solver integrated. **The two builders are
duplicated on purpose** -- the child imports numpy and the standard library only, which
is what makes the sandbox a sandbox -- and they are held to bit-equality by a test
rather than by hope.

**2. The gate had to be measured before it could be set, and the first value was
wrong.** `SDE_PATHWISE_TOL` was written as 5%, reasoning that a pathwise difference
under common random numbers has the Monte-Carlo error cancelled out of it and is
therefore better determined than a sampled moment. That reasoning is correct and
irrelevant: what sets this gate is the **scheme's discretization error**, not the
measurement's noise, and a strong error is much larger than a weak one at the same step
because the errors do not cancel across paths. Measured on SciML's own linear problem
at this harness's `dt = 0.01`, 4000 paths, against the exact solution on the supplied
path:

| Solver (one decision apart)                 | Relative strong error        |
| ------------------------------------------- | ---------------------------- |
| Milstein                                    | **1.64%**                    |
| Euler-Maruyama                              | **5.36%**                    |
| Euler-Maruyama, one wrong drift coefficient | **40.67%**                   |
| the closed form, transcribed                | **0.00%** (2.6e-16 absolute) |

So 5% **failed a correct Euler-Maruyama** by a hair -- the one thing this repo's
calibration rule forbids, and invisibly, because a false hard-fail on a correct solver
does not look like a false positive. The default is now 10%, where a correct EM clears
by 1.9x, the one-decision-wrong variant fails by 4.1x, and the two are separated by
**7.6x**. The number and its measurement are in the constant's own docstring, and
per-entry calibration in step 7 still overrides it: a default is the gate for a problem
nobody has measured yet, never a substitute for measuring it.

**3. H2 needed the distinction §2a.1 found, not the single verdict H2 specified.** The
ladder cannot tell a transcribed analytic solution from a _constant_ -- both return
bit-identical terminal states at every level -- and §2a.1 established that the
stiff-quadratic entries make the second case real rather than hypothetical. The record
now names which: `pathwise_exact` where the states vary across paths but not across
levels, `path_independent` where they do not vary across paths either. Both are
`ANALYTIC_TRANSCRIPTION`, because both are exclusions rather than defects, but calling
a constant a transcribed analytic solution would credit it with more than it did.
`validate_ground_truth.py` additionally refuses a _vacuous entry_ at authoring time, so
the stiff-quadratic problems cannot be declared at `T = 3` in step 6 and discovered in
step 9 -- which is the failure §2a.1 was one run away from.

**4. H9's deny-rule half was already done; the hole was elsewhere.**
`Read(./benchmark/**)` in `pipeline-settings.json` already covers
`benchmark/references/sciml/**`, so nothing needed adding there. What was missing is
that `harvest.jl` is a **`.jl` file under `benchmark/`** and every interpreter in the
deny list is a Python one: `julia benchmark/references/sciml/harvest.jl` was not
blocked. It is now, along with `juliaup`. The citation scan gained the artefact
filenames (`comparators.json`, `harvest.jl`, and `benchop_slv.json`, which had never
been listed either) -- but **not** `StochasticDiffEq`, `SDEProblemLibrary` or
`SciMLBenchmarks`, which were in the first draft of this change and removed: those are
public literature, a plan writing "an SRA1 scheme, after Rößler" is citing a paper
rather than reading an answer key, and flagging it is the false-positive class
`leakage.py`'s own docstring exists to refuse.

Four smaller things, recorded because step 6 depends on three of them:

- **SciML's `:l2` needs a saved history, so the schema gained one** -- `path_values`,
  at `pathwise_save_points` equally spaced times. It is **optional and never a gate**:
  a comparator axis must not be able to fail a solver that is accurate where it is
  graded, and a full history is not affordable anyway (S08 at the full step grid is
  ~800 MB). The absolute timeseries error is recorded per work-precision point as
  `abs_l2`, which is what makes the comparator table buildable from the series.
- **Nesting is now an authoring constraint on step 6.** Every level -- graded step,
  ladder, and each work-precision point -- is an aggregation of one Brownian draw, so a
  declared step that does not divide `pathwise_fine_steps` is reported `not_nested`
  rather than drawn fresh. SciML's published steps are `5^-k`, `8^-k` and `2^-k`
  (§2a.8), so `pathwise_fine_steps` has to be declared as a common multiple per entry;
  `validate_ground_truth.py` checks it without running anything.
- **The work axis records its own provenance.** `evals = paths x steps x stages`, with
  `stages` read from the plan's declared `scheme_family` and `stages_source` recording
  whether it was declared or defaulted to 1, beside `solve_s`. A curve whose work axis
  rests on a guess has to say so rather than look like a measurement.
- **`report.harness_error` needed a fourth branch, which is finding F10 again.** That
  function exists because the per-plan table read `rel_l2_err`, which no
  functional-graded problem has, and the whole BENCHOP suite printed `-` in its error
  column. A pathwise record carries neither a field error nor a moment error, so
  without a branch of its own **every entry in this suite would have printed `-` for
  exactly the same reason, one route later**. Caught here rather than in step 10 only
  because the earlier incident is written down; it is the clearest argument in this
  plan for why these documents record mechanisms and not just outcomes.

---

## 5. The timing problem, and the EM anchor `[N]`

**Primary axis: error vs work, not error vs seconds.** Work is counted in a currency no runtime can
inflate — `paths × steps × stages` (H4): 1 stage for EM, 2 for RKMil, the published stage count for
each Rößler scheme. This is what SciML's diagrams are really showing when the schemes are
fixed-step, and it is the only axis on which numpy and compiled Julia can share a plot without an
argument.

**Secondary axis: anchored wall-clock.** Implement **the same Euler–Maruyama** in both stacks — same
problem, same `dt`, same path count, single-threaded — and measure the ratio. That one number is the
language-and-implementation factor; quote raw seconds _and_ seconds divided by it, with the anchor's
value and the hardware printed beside every table. Measure the anchor per problem shape, not once:
`[N]` the factor on a scalar 1000-path problem (numpy vectorises across paths, Julia loops over
trajectories) and on a many-state few-path problem (Julia loops over floats, numpy pays interpreter
overhead per step) will not be the same number, and may not have the same sign.

**Three conventions, or the numbers mean nothing** (§3b): pin every version, run single-threaded,
and never time a first call.

---

### 5a. Step 4 as implemented — the measured anchor `[rev5 2026-11-10, N]`

`benchmark/references/sciml/em_anchor.jl` and `em_anchor.py`: the same
Euler–Maruyama, same step, same path count, single-threaded, warm-up discarded, minimum
of five. Increments are drawn **before** the clock starts in both stacks, because the
contract hands a solver its `dW` (H1), so stepping is the work being compared and a
random-number generator's speed is a different measurement. Each half is written the way
its stack is actually written: numpy vectorised across paths (as every SDE solver in this
benchmark is), Julia looping one trajectory at a time (as `StochasticDiffEq.jl` does).

**§10 prediction 5 is confirmed, and by more than it claimed** — it said the two factors
would differ by more than 3×; they differ by **8.1×**:

| shape | numpy | julia | factor |
| ----- | ----- | ----- | ------ |
| scalar, 1000 paths, 15 625 steps | 4.78 ns/path-step | 4.63 | **1.03×** |
| 8 states, 10 paths, 15 625 steps | 37.07 ns/path-step | 4.43 | **8.37×** |

#### The scalar parity is an artifact, and the anchor says so

1.03× reads as "numpy matches compiled Julia", and that reading is wrong. The
trajectory-at-a-time loop is **latency**-bound: `xp += a*xp*dt + b*xp*dW` is a serially
dependent chain of multiply-adds, so each step waits on the previous one, and 4.6 ns is
about four FMA latencies on this machine rather than a statement about Julia. Two
measurements establish it:

- **Interleaving four independent trajectories** in the same loop — breaking the
  dependency, changing nothing else — runs the identical arithmetic at **1.76 ns**, a
  **2.6× speedup** from instruction-level parallelism alone.
- **The array layout is not the cause**: `dW[path, step]` against `dW[step, path]`
  (contiguous inner loop) differ by **1.04×**, so neither stack is memory-bound here.

numpy has no choice but to vectorise across paths, which hides exactly the latency the
Julia loop exposes. Comparing against the single-chain loop therefore measures a
bottleneck one stack was forced out of and the other was not. `em_anchor.jl` ships the
interleaved variant as a third row and `em_anchor.py` prints the comparison against it,
labelled as the ceiling:

> **scalar shape, numpy vs the best available Julia: 2.71×** (4.78 ns vs 1.76 ns)

So the anchor is three numbers, not one and not two: **1.03× against the idiomatic
trajectory loop, 2.71× against the best Julia on the same shape, 8.37× on the many-state
shape.** The honest statement for a table footer is the middle one on scalar problems and
the last on systems.

**And it is not a constant even then.** The factor is set by which bottleneck binds, so
it moves with the cost of the right-hand side: a cheaper one favours numpy (whose
per-step overhead is amortised over the path axis), an expensive one favours Julia (whose
temporaries do not grow with the expression). That is §5's "measure it per problem shape"
instruction, with a mechanism attached — and it is the reason the primary axis stays
`evals` even though §3f finding 2 shows `evals` is biased too. The two biases are at
least in known and opposite directions: `evals` over-charges the multi-stage comparators
by up to 6×, and seconds over-charge numpy by 2.7×–8.4×.

**Where the stiff triple sits.** Those three entries are scalar with 1000 paths, so they
take the scalar factor; M09 is the only 2-state entry and the only one where the
many-state factor is even approximately right. No entry in this suite has the 8-state
shape — it was measured because §5 and §7 step 4 asked for a ≥7-state measurement, and
what it establishes is the *spread*, not a number any row of the table will be divided
by. Phase 2's `bruss` and `lorenz` are where it would bind, and they are not authored.

Measurements are committed at
`docs/sciml-benchmark-data/measurements/em_anchor_{julia,python}.json`
(julia 1.11.9, `threads=1`, apple-m1, 8 cores / 17 GB; python 3.14.7, numpy 2.4.6).

---

## 6. Two arms, and why the second one is the suite `[N]`

| Arm                | What the pipeline may do                                                                                                         | What it measures                                                                                          | Status                             |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **Unrestricted**   | anything, including the exact pathwise solution                                                                                  | whether the pipeline _recognises_ an exactly-solvable SDE — a real capability, and the one rule 0 rewards | a one-line result, not a benchmark |
| **Scheme-matched** | a discretised stepper that consumes the supplied increments; `exact_sampling_components` empty; degenerate ladders rejected (H2) | scheme quality against nine published schemes at matched work                                             | **this is the suite**              |

Report both, never merged (H10). The unrestricted arm is cheap, it will probably look spectacular,
and presenting it as a win over `StochasticDiffEq.jl` would be indefensible — so the plan writes
down in advance that it is a capability note, not a comparison, and that the headline table is the
scheme-matched arm.

---

## 7. Steps

| #   | Step                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Acceptance                                                                                                                                                                                                                                                                                                                                                                                          |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | ~~§3a–3b: juliaup, the pinned project, `Project.toml` + `Manifest.toml` committed to `benchmark/references/sciml/`~~ **DONE 2026-11-10 — §3e** | ✅ met, and the acceptance line's third clause came back negative in the useful direction: both upstream manifests instantiate **verbatim** on aarch64-darwin and were not re-resolved (md5 unchanged), the pinned version set is confirmed package by package, and **the three `BoundsError` cells §1a.2 actually names now run** — it claims four and names three, and the fourth cannot be matched to a cell from the page text, so what is established is that the named failures are fixed upstream rather than that page 1 is fully revalidated; the three adaptive-set cells were not run to completion, at over an hour each for results §11 excludes anyway. The other four notebooks pass in full (8/8, 7/7, 20/20). The stiff closure turns out to be a strict *subset* of the non-stiff one with zero version disagreements, which is stronger than rev 2's claim. A different `BoundsError` with the same signature is live at the pinned version on the `saveat` path, and is recorded rather than worked around silently |
| 2   | ~~Re-read `SDEProblemLibrary.jl` **raw**, at the pinned version, and reconcile every formula in §2 against it~~ **DONE 2026-11-10 — §2a**                                                                                                                                                                                                                                                                                                                                                                                               | ✅ met, and exceeded: the diff covers the library _and_ all five notebook sources, every `[L]` formula is confirmed or struck through in place, and each closed form was additionally confirmed to _solve_ its stated equation (Milstein order ≈ 1 under CRN for S01–S05; S09's moment derived from its own coefficients). Eight findings, four of which change the plan                            |
| 3 | ~~§3c: write and run `harvest.jl`; produce `comparators.json` with the per-(problem, scheme, dt) fields of §3c~~ **DONE 2026-11-10 — §3f** | ✅ met: **279 rows**, 19 schemes, every cell `ok`, numbers not plots. Three findings, two of which change how the result must be read — the stage counts are **measured** and `SRI` costs 10 rounds per step against a tableau's 2–4, the `evals` axis **over-charges** multi-stage schemes by up to 6× against measured seconds (so it flatters the pipeline), and §10's prediction 1 is confirmed under one reading of "matched work" and **falsified on four of nine entries** under the other. SciML's `:l2` column is deliberately absent, with the upstream defect that prevents it recorded |
| 4 | ~~§5: measure the EM anchor on S01 (scalar) and on a ≥7-state problem; commit both EM sources~~ **DONE 2026-11-10 — §5a** | ✅ met, and **§10 prediction 5 confirmed at 8.1× spread** against the >3× it claimed. Three numbers rather than two: the scalar factor of 1.03× is an artifact of a latency-bound dependency chain, proved by interleaving four trajectories for a 2.6× speedup and by ruling out memory layout at 1.04×, so the honest scalar ceiling is **2.71×** and the many-state factor is **8.37×** |
| 5   | ~~**H1, H2, H3, H4, H8, H9, H10**, with acceptance tests as written in §4~~ **DONE 2026-11-10 — §4b**                                                                                                                                                                                                                                                                                                                                                                                                                                   | ✅ both halves met: all seven items' acceptance lines pass as 28 tests in `benchmark/tests/test_pathwise_route.py`, run against four real solvers one decision apart through the real sandbox; and all **66** problem statements plus the entire manifest hash identically before and after. Four findings, in §4b — the largest being that the benchmark's SDE contract never accepted `dW` at all |
| 6 | ~~Author S01–S09 (Gauss–Hermite weak truth for the moment gates); validate every truth by a route independent of the entry (`validate_ground_truth.py`); set `work_precision_levels` to **SciML's own published steps** (§4a.i)~~ **DONE 2026-11-10 — §2b** | ✅ met, with two corrections to the row itself. The weak truth needs **no Gauss–Hermite**: the second moments satisfy a closed 3×3 linear system, so all three are exact and `E[X1²] = exp(−3)` to the bit — validated instead against a 4×10⁵-path Euler–Maruyama Richardson-extrapolated across two nested steps, agreeing at 1.6σ / 0.7σ / 1.6σ. And `work_precision_levels` are SciML's published steps only on **four of nine** entries, because three are in no notebook and the stiff triple's graded horizon moves (§2b finding 1). `--suite sciml` runs alone, and the refusal is tested |
| 7 | ~~**Calibrate every gate both ways**~~ **DONE 2026-11-10, as part of step 6 — §2b.** It could not be deferred: three entries were *vacuous* until calibration found them, so authoring an entry and calibrating it turned out to be one act rather than two. Every gate separates, the narrowest by **4.4x**, none was moved after the fact, and **two of this row's own proposed controls had to be replaced** because they do not separate (§2b findings 3 and 4). Original text, kept so the plan can be graded against it: calibrate every gate both ways, per the repo rule: a correct scheme clears it and a plausible wrong one fails it, at the resolution the entry ships with, with the separation factor recorded and **no gate moved after seeing where a scheme lands**. One-decision wrong variants: a dropped Milstein correction on S03; `β²` omitted from the S06–S08 drift (`[L]` the _only_ difference between the library's Itô and Stratonovich variants); a single Wiener increment shared by both components of S09's non-commutative noise | every gate separates; the narrowest factor recorded; failures recorded rather than re-cut                                                                                                                                                                                                                                                                                                           |
| 8 | ~~Smoke-test the suite without spending a pipeline call (the step that found four defects on BENCHOP `[R]`)~~ **DONE 2026-11-10 — §7a** | ✅ met, as **97 automated assertions** over the nine staged briefs, and it found four defects here too — plus a fifth the fix for the second one exposed. The per-plan ranking axis was **blank on eight of nine entries**; **nothing in the repo had ever rendered the work-precision series**, which is this suite's primary axis; the report named two suites; and the per-plan record did not say which scheme each plan chose, which is step 10's own acceptance line. One correction to the row: *every* brief cannot state the `dW` shape, because M09 is handed no `dW` |
| 9 | ~~Run the C0 one-shot baseline on S01–S09, both arms~~ **DONE 2026-11-10 — §9a** | ✅ met: C0@Opus, naked, 9 units per arm, **18/18 PASS** at $0.96 and $1.06. The arms are now genuinely different experiments (§7a defect 6) and the data says so — **0 of 8 degenerate ladders in the scheme-matched arm, 8 of 8 in the unrestricted one**, which confirms §10's prediction 2 at its maximum in one arm and falsifies its premise in the other: told plainly that transcription is detected and excluded, Opus wrote a discretised scheme nine times out of nine. The deterrent measured is the *sentence*, not the detector. Also unpredicted: the baseline ships a stochastic Heun predictor–corrector on two entries, reaching 5.4e-8 on M02 — better than §3f's measured `SRA1` at the same step — with no manual in the prompt |
| 10 | ~~Run the pipeline (C2), both arms~~ **DONE on the scheme-matched arm — §10a–10e; ~~8 of 9 entries completed, M09 a run-error at the old timeout~~ all 9 completed 2026-11-13** | ✅ met on one arm.  M01 scheme-matched: **19.6 min**, `VERIFIED_PASS`, 2.06% against the 9.5% gate — so ~3 h for the arm and ~6 h for both. The row's own acceptance line is met by that run (`scheme_family` per plan, as §7a defect 4 made possible) and it immediately produced the run's most interesting result: **`wrong_plan_won: True`** — the conductor shipped Euler–Maruyama at 0.217× gate over its own Heun-SRK plan at 0.012×, a 17.7× gap against a 0.0048 noise margin, and scored the worse plan 10 against the better ones' 8. That field was `null` before §7a defect 1 was fixed. The probe also exposed a vocabulary gap: there is no SRK family in `SDE_FAMILIES`, so an SRK plan declares `other` and its work axis falls back to 1 stage — understating its own cost, which the report now labels a lower bound. **Diagnosed rev 6:** the selection was made by *one* of fifteen checks, the terminal-law KS statistic, which Milstein lost by **1.1% of the critical value** — and on the run's own increments a **pathwise-exact** solver fails that same gate at 1.17×, worse than the Euler–Maruyama that passed it. The gate is not ranking schemes. Both losers also carry an incorrect reason string naming the mean, which passed 46–67× inside tolerance. **Probed further rev 7:** M06 and M09 both read `wrong_plan_won: true` as well, by a *different* mechanism — on both, every plan scored the same (10/10/10 and 8/8/8) and the winner was chosen by a key *below* `score`, shipping a plan 5.1× and 3.3× worse than the best. ~~by plan **index**~~ **rev 8 (10b) locates both:** on M06 the quality index decided it, `decided_by: sqi` by 0.014, because the best plan's error sat inside its own Monte-Carlo bar so `accuracy` returned None and its index was renormalised over three components against the winner's four — the best plan lost for being too accurate to resolve; on M09 the ranker never ran, the run having timed out before Phase 3, and the winner is `run.py`'s fallback taking the first plan in declaration order. Both fixed, with the corpus-wide effect measured: one winner improved by 5.1× and two moved inside their own noise margins. On M09 the shipped plan is the only one of three the harness does not grade 10, and the tie was manufactured by a single `null` in a declared `terminal_law` list disabling the rubric's top two rows for every plan on the problem. So the defect is the SDE evaluator's, not the pathwise route's: a tie is enough, and ties are the common case. M06 ran 21.5 min and passed; M09 hit the 60-min timeout mid-evaluation, so the per-problem budget must be raised. Three more harness defects fell out of these runs (§7a defects 8–10). **Rev 9:** M02 run under the fix at the raised timeout — `UNDERCLAIM`, 13.0 min, harness-verified 10 at a strong error of 1.34e-7 against a 1.2e-3 gate; the conductor shipped the ranker's winner rather than the first plan, which is the first case where those two differ, so compliance is confirmed. `wrong_plan_won: true` once more, but now between two plans 1980× and 4879× better than the Euler–Maruyama, decided on the fourth significant figure of a structural diagnostic because no plan's accuracy resolved (10c). **Rev 10:** the remaining five (M03, M04, M05, M07, M08) run in one invocation at 7200 s — all five completed, none timed out, **80.0 min** for five problems, **2 `VERIFIED_PASS` and 3 `UNDERCLAIM`, no overclaim**. Across the eight completed entries: 4 pass, 4 underclaim, **zero overclaims**. The fix flips no winner on the five and widens M07's margin **8.8×**; `wrong_plan_won: true` once more, on M05, by **622×** and 44.6× outside the margin, with the whole selection resting on `('structure',)` alone because that problem's order check never ran. And the vocabulary gap turns out to be an **incentive**: M05's order-1.5 plan withholds `scheme_family` on purpose, with the correct reason written above its metrics block — no kernel family key maps to order 1.5, so a correct scheme fails every band it could declare. Defects 9 and 10 now stand at **seven of ~~eight~~ nine** completed entries (10d, 10e). ~~Eight of nine entries run; M09 to re-run at the raised timeout~~ **Rev 11 `[2026-11-13]`: M09 completed on the third attempt in 312.1 s (10e), `VERIFIED_PASS` at 10/certified.** Neither earlier failure was the problem or the timeout — the second stopped on the Claude Pro 5-hour cap, on a window that had opened 40 min before launch, with the machine clamshell-sleeping 8 min in (`caffeinate -ims` does not prevent lid-close sleep, and `run.py:185`'s monotonic clock does not advance through it: 124.1 reported minutes across 4 h 18 min real). The re-run was cheap because the **conductor recovered its own crash** — it diagnosed the stale `STATE.md`, verified via `solver_sha256` that the two certified scores on disk belonged to the files there, and completed the one half-cycle by reusing saved kernel output instead of re-paying ~1500 s of ladder time. The winner is the plan a **pool gate** caused to exist: no plan had declared `exact_sampling_components: 1`, the conductor re-dispatched the plan-creator once, and the Rao-Blackwellised plan it added won — the first time a gap declared in the *pool* produced the shipped plan. First case too where the ranker's winner and `best_plan_by_harness` differ and `wrong_plan_won` is correctly **false**: 0.0590 against 0.0571 of gate, inside a 0.0617 margin, the ranker preferring a real 3.2× variance reduction a single realisation cannot see. The rev-8 fix is inert again, but for a new reason — `order` is `None` on every plan because the record's per-plan `kernel` block **does not carry `observed_order`** although `SOLUTION.md` does, so the fix's input is truncated by the recorder (second instance after M08's headroom). Step 10 complete: **9 of 9, 5 pass / 4 underclaim / 0 overclaims, 139.3 min**. **Rev 12 (10f)** then re-ranks all nine pools offline against the C0 baseline at equal budget — same `dt`, path count and evals on both sides — and the answer is the one the suite was built to get: the pipeline **as shipped is 1.412× worse than Opus cold** (better on 1 of 9, bit-identical on `M03`, within 1e-16 on two more), while **the best plan in each pool is 2.65× better** — `M05`'s pool held one **625×** better than Opus. Of the three remaining ranker misses, only one is the ranker's: `M01` is `decided_by: score` over an index that already ranks it right, and repairing that score alone takes the arm to **0.855×, i.e. better than the one-shot**; `M05` needs a measurement it skipped, not a weight. `abstain` and `discriminating` measure **inert**, and the resolution floor 10c called for is **withdrawn as harmful** — monotonic on sciml, opposite sign on the BENCHOP holdout, because a tie falls to declaration order, which is the worst plan on 9 of 9 sciml pools and the best on 5 of 9 BENCHOP ones |
| 11  | Phase 2 **if it is still wanted**: H5, H6, H7, then `additivesystem` → `bruss` → `lorenz`. §4a.ii drops `oscilreact`, `oval2` and stoch-heat                                                                                                                                                                                                                                                                                                                                                                                            | reference agreement two orders below the gate, with the second integrator named                                                                                                                                                                                                                                                                                                                     |

**Steps 1–10 are the plan. Step 11 is a second plan wearing a row**, and §4a.ii has now removed its
three most interesting problems, which strengthens that reading rather than weakening it.

---

### 7a. Step 8 as implemented — the smoke test, and ten defects `[rev5 2026-11-10; defects 8-10 rev7 2026-11-11, N]`

The step exists because the same step found four defects on BENCHOP `[R]`. It found four
here, a fifth while fixing the second, a sixth when step 9 was launched and a seventh when
step 9's second arm was launched — and — as there — every one of them was on the path step 10 is about to take
rather than in the entries, which is why authoring and calibrating the suite (steps 6
and 7) could not have caught any of them.

**What the acceptance line asked for, and what it got.** `setup.py` stages all **75**
problems (49 + 7 + 10 + 9) and `manifest.json` reads
`n_by_suite: {main: 49, no-closed-form: 7, benchop: 10, sciml: 9}`;
`run.py --suite sciml --list` selects exactly M01–M09; `validate_ground_truth.py` passes
with no failures. The brief checks are automated rather than read:
**97 assertions** over the nine staged files confirm that each brief states the call its
verifier will make (`solve_sde(num_paths=…, dt=…, T=…, seed=…)`, compared against
`sde_verify_dt` and the entry's own fields, not against the `description` string), that
each pathwise brief states the `dW` shape the CRN payload will actually produce and the
`path_values` shape beside it, that the graded step divides the horizon exactly, and that
no brief contains any graded value — tested against the exact terminal state on the
path the solver is handed, its mean and its spread, each at four roundings, plus
fourteen closed-form tells (`arctan`, `0.63155`, `np.exp`, `(1 + u0)`, `analytic`, …).

**Finding: the acceptance line over-counted its own suite.** It says *every* brief
states the `dW` shape. Eight do. **M09 has no `dW` at all** — it is the one entry not on
the pathwise route, so it draws its own randomness and the contract never mentions an
increment array. What it needs instead is for the *channel count* to be unmistakable,
and it says so in prose: "driven by **two independent** Brownian motions", and "using one
increment for both channels is a different system". That is the information the §7 control
depends on, so the suite is not short of anything — but a check written from the
acceptance line as published fails on a correct brief, which is worth recording as a
caution about acceptance lines written before the thing exists.

#### The four the smoke test itself found

Numbered as the smoke test found them. The fifth came out of fixing the second; the sixth and seventh are below, and neither was findable by testing — each took an actual run.

**1. The per-plan ranking axis was blank on eight of the nine entries.**
`run.plan_harness_error` had a branch for a PDE's field error, one for SDE moments and
one for a functional, and the pathwise route is none of the three — so it returned
`None`, `wrong_plan_won` had nothing to compare the eight on, and the "Every plan,
graded" table's error column rendered as `-`. This is the *same defect one route later*
that BENCHOP's step 8 recorded for prices, which is the useful part: adding a grading
route to this harness has now twice silently removed the per-plan comparison, and
nothing in the route's own tests noticed either time. Fixed with a pathwise branch
returning `rel_err / tol` (1.0 exactly at the pass line, as the other branches are), and
`harness_error_metric` — which had been answering "worst moment error / its gate" for
eight entries that have no moment gate — now answers "relative strong error / its gate".
Measured on the planted M01: `harness_error = 0.2166` (= 0.0206 / 0.095).

The margin needed deriving rather than copying. The other branches use a Monte-Carlo
standard error; the graded number here is a root-mean-square over `n` paths, whose
standard error is `rms/sqrt(2n)` to leading order, so that is what it uses — **0.0048** on
the same dimensionless scale at 1000 paths, i.e. two plans within 2.2% of each other's
error are not distinguishable at this path count.

**2. Nothing rendered the work-precision series. Nothing ever had.**
`report.py` and `compare.py` contained **zero** occurrences of `work_precision`. BENCHOP's
step 8 recorded this as something "step 11 inherits" and it stayed true, which was
survivable there because the series was a nice-to-have. Here it is not: error against
**work** is this adoption's primary axis (§5) — the only axis on which numpy and compiled
Julia can share a plot — so a step-10 sweep would have paid for nine five-point curves,
written them all to `results.json`, displayed none of them, and been unreadable without
re-running. `report.py` gains an **Error vs work** section: one table per entry with
`dt | steps | paths | stages | evals | rel err | solve s | graded`, the declared scheme
family and the provenance of its stage count, and the least-squares log-log slope of
error against work — reported only from three or more points, because two points make a
slope out of any pair of numbers. Measured on the planted Euler–Maruyama:

| entry | d log err / d log evals | observed order in `dt` |
| ----- | ----------------------- | ---------------------- |
| M01 (pathwise, strong) | −0.53 | **0.53** |
| M09 (weak, 3 moments)  | −0.83 | **0.76** |

M01's 0.53 is Euler–Maruyama's strong order 1/2, measured through the shipped entry at
the shipped resolutions by the renderer that step 10 will use — which is the check the
section had to pass before it could be trusted to display anything.

**3. The report described two suites.** The methodology section named `main` and
`--no-closed-form` and stopped, so a reader of a `sciml` report had no way to know which
denominator its pass rate was quoted against, that `benchop` is graded against published
values while `sciml` is graded against truth this repo owns and *compared* against
another library's schemes, or that the two arms exist and are never merged. It now names
all four, both arms and the exclusion.

**4. The per-plan record did not say which scheme each plan chose** — which is step 10's
own acceptance line, verbatim. The declared `scheme_family` reached the record only
inside the *winner's* work-precision points, so a comparison across plans — the entire
point of a scheme benchmark — had nothing to read. `harness_by_plan` now records it per
plan, declared and never inferred, for the reason `verify.SCHEME_STAGES` gives.

**A fifth, smaller one, found by the renderer the moment it existed.** The free
work-precision point — the coarse solve the two-level anti-hardcoding check has already
paid for (§BENCHOP step 9) — carried an error and a clock but **no work axis**: it drew
as a row of dashes beside five complete rows in M09's table, because
`_wp_sde_free_level` never asked for `_wp_sde_work`. It is the one point on the curve
that is free and was the only one that could not be plotted against work. Fixed; the
two points now agree about what one step of the same solver costs, which is pinned by a
test that changes the declared family and checks both points move together.

#### What the smoke test measured, end to end

With a correct Euler–Maruyama planted as a plan and **no pipeline call**
(`run.py --only … --skip-run --arm scheme-matched --results-json <scratch>`), the whole
non-agent path runs on both routes:

| entry | verdict | graded error | gate | margin |
| ----- | ------- | ------------ | ---- | ------ |
| M01 (pathwise) | VERIFIED_PASS, verified score 10 | rel strong err **2.06%** | 9.5% | 4.6× |
| M09 (weak) | VERIFIED_PASS, verified score 10 | 0.35% / 0.86% / **1.53%** | 2% / 1.6% / 2.9% | 1.9× |

Three things in that table are worth keeping. First, M09's gates had only ever been
checked against the validator's own Monte-Carlo; this is the first time a *solver* has
cleared them through the sandbox, and the narrowest of the three clears by 1.9× — the
tightest margin anywhere in the suite, tighter than the 4.4× of §2b. Second,
`second_moment_x1`'s error is **0.35%** against a predicted noise floor of 0.349% at
10⁵ paths (§2b finding 4): the correct scheme's measured error on that observable *is*
the sampling noise, to two figures, which is the prediction confirmed from the other
direction. Third, the free coarse point at 25 000 paths scores **1.37%** against the
graded point's 1.53% — four times fewer paths, *less* error — which is only possible
because neither number is resolving bias.

**M09's work-precision curve says something about SciML's published steps.** At its four
published weak step sizes (1, 1/2, 1/4, 1/8 over `[0, 3]`) a correct Euler–Maruyama's
relative error on `E[X₁²]` is **95%, 86%, 68%, 47%**. Those steps are chosen to display an
*order* on a log-log plot with no gate anywhere on it; they are roughly 30× too coarse
for any gate this repo would write. The series is still the right thing to collect — it
is what makes the comparison a work-precision comparison — but the curve and the gate
live at different resolutions, and a reader who takes the top of that curve for a result
would conclude that Euler–Maruyama fails M09, which it does not.

#### A sixth defect, found by trying to start step 9: **the second arm did not exist**

The step-8 smoke test ran `--arm scheme-matched`, which is the default, and that is why it
missed this. Before launching the step-9 baseline the other arm was staged for the first
time, and:

- **`setup.py` did not accept `--arm` at all** — the flag lives in `add_suite_arguments`,
  which only the two runners use.
- **`contract_footer` never took an arm**, so the brief was byte-identical in both arms:
  each one told the solver *"A discretised scheme is required"* and that a transcription
  would be excluded.
- **`_is_transcription` applied unconditionally**, so the exclusion fired in both arms.

So `--arm unrestricted` was a *label on the scheme-matched experiment*. Running both arms
as §7 steps 9 and 10 ask would have cost twice, measured once, and **met its own
acceptance line** — "baseline records, per arm, with the arm field set" — which is the
property that makes this the worst of the six. §6 anticipated the two arms being
*pooled* and made the arm a recorded field rather than a convention to prevent it; the
risk turned out to be the other one, that the field is all there is.

It also explains a result that should have looked wrong earlier: §6's table says the
unrestricted arm "will probably look spectacular", and nothing in the harness could have
produced that outcome, because the brief forbade the thing the arm is for.

**The fix makes the arm change the task and the verdict.** `contract_footer(problem, arm)`
renders one of two paragraphs on a pathwise problem — the existing one, now a named
constant, or a new one that says in as many words that evaluating the closed form on the
supplied increments is a correct and expected answer, that the ladder still measures
whether any discretization error was carried, and that recognising the equation as
exactly solvable is what is being measured. `_is_transcription(ver, arm)` returns `False`
in the unrestricted arm, so a transcription there is a PASS on its own merits rather than
an exclusion — there is nothing to exclude it *from*, since that arm is a capability note
and not a comparison against another library's schemes. Both runners and `setup.py` now
thread their arm into staging.

Three properties are pinned rather than assumed, because each of them would quietly
invalidate earlier work:

| Property | Why it has to hold |
| -------- | ------------------ |
| the two arms' briefs differ by **exactly one line** | the arm changes the permission, not the problem |
| the scheme-matched brief is **byte-identical** to the armless default | every number measured before the arms existed was measured against that text; verified across all **75** staged briefs, 0 differ |
| a record carrying **no arm** grades exactly as before | every frozen result predates the flag, so the exclusion is the default and not the exception |
| the unrestricted arm still **fails a wrong solver** | the permission is to use the closed form, not to be wrong — and §2b showed a constant also produces a degenerate ladder |
| all **67** non-pathwise problems are untouched by the arm | including M09, which is in this suite but not on that route |

**Tests.** `test_sciml_suite.py` gains **N8**, fifteen tests (eight parameterised over the
pathwise entries): the two briefs differ in one line and in the right direction, the
default is unchanged, no other problem moves, the verdict flips in both the pipeline and
baseline paths, an armless record is unaffected, a wrong solver still fails in the
permissive arm, and all three entry points can state an arm.

#### A seventh defect, found by running the second arm: **the arms shared their artifacts**

With the brief fixed, both arms ran — and the second one deleted the first one's
evidence. `oneshot.plan_dir_for` keyed the artifact directory on
`(problem, condition, manuals, model, rep)` and `unit_key` on the same tuple, neither
including the arm. So:

- **The unrestricted run wrote its `solver.py` over the scheme-matched run's.** On a
  *scheme* benchmark the solver source is the evidence — it is the only thing that says
  which scheme was chosen — so the first arm's result survived in the record while the
  artifact the record is about was gone. Noticed only because a survey of which schemes
  the baseline picked came back reporting closed forms on problems the records said had
  non-degenerate ladders.
- **`unit_key` collided across arms**, which is worse than lost evidence. Two arms
  written to one results file would have silently deduped to one unit, and `--resume`
  would have treated a unit already run in one arm as done in the other. §6 made the arm
  a *recorded field* rather than a convention precisely so that the two could not be
  pooled; the pooling it prevented was in `compare.py`'s columns, and this one was
  upstream of that, in the file.

Both now carry the arm, and the default arm's spelling is unchanged — `""` for
`scheme-matched` or no arm, `-<arm>` otherwise — so every frozen results file still
matches on `--resume` and every existing artifact directory is still the right one. Both
arms were re-run afterwards; the pre-fix records agreed with the re-runs on all nine
verdicts, and were pruned rather than kept, because the artifacts they pointed at no
longer existed.

**`run.py` has the same shape of hole** and it is recorded here rather than fixed
blind: the pipeline's workspace is `workspace/{slug}`, with no arm in the path, so a
second-arm pipeline run would overwrite the first arm's `plans/` — and on this suite the
plans are the evidence. It does not bite until step 10 is run in both arms, which is
where it will be fixed, measured against a real two-arm run rather than against a guess.

**Two things the smoke test confirmed rather than found.** `run.py` refuses the standing
results file for this suite — *"this selection includes the sciml suite, which must not
be written into the standing main-suite results file"* — which is BENCHOP step 8's
fix #2 working on a suite that did not exist when it was written. And the planted plan
carries no `problem_spec.json`, so the kernel block reads `unavailable` and the per-plan
verdict column reads `UNDERCLAIM`: an artifact of the plant, identical to the one BENCHOP
recorded, and the reason that column is not quoted as a result here either.

**Tests.** `test_sciml_suite.py` gains **N7**, nine tests — one per defect plus the
guards: the pathwise per-plan error and its margin (including that four times the paths
halves it, as a standard error must), 1.0 exactly at the pass line, the metric label on
all eight entries, the four suite names and both arms in the rendered report, the slope
refusing two points and refusing a degenerate series, the section rendering a full row
byte-for-byte, and a one-point series rendering no section at all.
`test_work_precision.py` gains **I6**, two tests on the free point's work axis.

#### Three more defects, found by the step-10 probe runs `[rev7 2026-11-11]`

The M06/M09 probes took three attempts — the first lost to an API/DNS outage
(`ENOTFOUND` on both problems), the second to a usage limit 21.9 minutes into M06. The
two lost attempts are the only reason the first of these was found at all.

**Defect 8: a dead workspace was graded as a live run.** Attempt 1 returned `RUN ERROR`
on both problems and M09's record still read `phase: complete, best_score: 10`,
`verified: true`, `passed: true`. It had graded the **step-8 smoke test's planted
`STATE.md` and `solver.py`, dated the previous day**. The existing guards 3a/3b are
conditioned on `run["status"] == "completed"`; the status was `"error"`, so they never
fired, and `clear_pipeline_outputs` only runs under `--fresh`. A conductor that dies
before writing anything therefore inherits whatever the last run left in
`workspace/{slug}` and reports it as this run's result — the one failure mode that
produces a *passing* record out of a run that did nothing.

`summarize_pipeline` now records the mtime of the `STATE.md` it read, and `process`
refuses to grade a workspace whose state predates the launch (`status:
stale_workspace`), skipping verify, kernel and `harness_by_plan` rather than recording
them from stale files. `--skip-run` is exempt, because re-grading old state is its
purpose. A first fix left a hole of its own — under `--fresh` there is no stale
`STATE.md`, so a conductor writing nothing at all would fall through the check — closed
by a second branch (`status: no_state`). A non-terminal phase is still graded, but
stamped `graded_incomplete` with a note, which is what M09's timed-out record carries
below. **Nine tests**, `test_orchestration.py` section **E9**.

**Defect 9: `score_cap` is recorded and never applied.** M06's verified record carries
an error-severity ledger violation and `score_cap: 3` — and grades **10 /
`VERIFIED_PASS`**. `ledger_audit` sets the cap (`benchmark/verify.py:2489`), `run.py:504`
folds it into the record, one wiring test asserts the folding, and **nothing reads it to
bound a score or a verdict**. The docstring says "the score is capped and the review
names the ledger entry"; only the second half happens. Recorded, not fixed: the fix
changes scores on a suite whose baseline is already measured, so it belongs with the
step-11 rework rather than between two probe runs.

**Defect 10: the violation defect 9 failed to apply is the formulator's, not the
solver's.** `audit_sde_output` compares `run_meta.num_paths` against
`evaluation_thresholds.num_paths` (`verifylib/audit.py:162`). M06 is graded by the suite
at `num_paths: 1000` (`benchmark/problems.py:4445`); its formulator wrote
`num_paths: 100000` into the spec it hands the solver, and put the real 1000 in
`graded_num_paths` — a field that means the **larger** count, used for the kernel's
memory projection (`agents/formulator.md:215`). Across the twenty live SDE specs,
`graded_num_paths >= num_paths` everywhere except the two SciML entries, which are the
only two where it was read as "the count that is graded". So the audit was right, it was
pointing at the spec rather than the run, and the cap that would have made it visible is
defect 9.

---

## 8. Leakage `[N]`

A different shape from BENCHOP's, and mostly milder — with one hard rule:

- **Recall of an exact solution is not cheating here**, it is the formulator's job. §0b is about
  _grading_, not about leakage.
- **The real vector is the comparator artefacts.** The Julia sources contain the SRK tableaux and,
  once step 3 runs, `comparators.json` contains the reference error values. If they land inside
  `workspace/`, the pipeline can read both the schemes it is compared against and the numbers it is
  graded near. H9 is that fix, and it is required rather than optional.
- **Scheme recall without justification.** SciML publishes _which scheme wins_; a model can name
  SOSRI without being able to derive it. The existing order study already charges a plan that
  claims an order it does not achieve — but note that a correct SRK implementation would be the
  first scheme in this repo's history whose declared order the CRN ladder has never verified, and
  §10's prediction 5 is about that.
- **Shifted-parameter controls are cheap here** (S06–S08 are already the same equation at three
  parameter sets) and worth one entry, but they defend against the wrong threat; H2 is the defence
  that matters.

---

## 9. Cost

| Item                                                  | Estimate                                                                       |
| ----------------------------------------------------- | ------------------------------------------------------------------------------ |
| §3a–3b Julia setup, pinned, notebooks rerun (step 1)  | 1 day, no model calls                                                          |
| Source reconciliation (step 2)                        | 0.5 day                                                                        |
| §3c harvest script + `comparators.json` (step 3)      | 1–1.5 days                                                                     |
| EM anchor (step 4)                                    | 0.5 day                                                                        |
| **H1 + H2** (pathwise route + transcription verdict)  | 1.5–2 days                                                                     |
| **H3 + H4 + H8 + H9 + H10**                           | 1–1.5 days                                                                     |
| Nine problem entries + GH truth + validation (step 6) | 1–1.5 days                                                                     |
| Calibration and smoke test (steps 7–8)                | 1–1.5 days                                                                     |
| **Phase 1 subtotal, no model calls**                  | **~8–10 days**                                                                 |
| C0 + C2 runs, two arms, nine problems (steps 9–10)    | the model-call budget; BENCHOP's ten-problem equivalent is the reference point |
| H5 + H6 + H7 + three Phase-2 entries (step 11)        | 4–6 days, plus runs                                                            |

**If the budget is one week:** steps 1–8 with S01–S04 and S06–S09 (drop S05, whose `u^{2/3}`
diffusion is a domain-guard problem more than a scheme problem), scheme-matched arm only, plus the
C0 baseline. That yields the comparator table and a baseline, and leaves the pipeline run for when
the manual has an SRK section to run _with_.

---

### 9a. Step 9 as implemented — the C0 baseline, both arms `[rev5 2026-11-10, N]`

`oneshot.py --suite sciml --condition c0 --manuals naked --model opus --c0-max-turns 4`,
once per arm. Nine units each, **$0.96** and **$1.06**, three minutes per arm, one turn
per unit. Opus rather than a cheap model because every other suite's baseline in this
repository was measured at C0@Opus and a baseline that is not comparable is not a
baseline; `--c0-max-turns 4` because at the default cap of 2 Opus spends both turns
probing its (denied) tools and commits no solver `[R]`. Naked only — the knowledge
contrast is deliberately skipped, and §0d says why it would have been predictable:
`sde_manual.md` contains no Rößler-family scheme at all.

**Every unit passed in both arms, 18/18.** And the two arms are completely different
experiments, which is the point and was not true until the defect in §7a was fixed:

| entry | scheme-matched | × gate | unrestricted | ladder |
| ----- | -------------- | ------ | ------------ | ------ |
| M01 linear | 2.48e-3 | 0.026× | 1.29e-15 | degenerate |
| M02 additive | 5.43e-8 | 4.5e-5× | 9.78e-17 | degenerate |
| M03 cubic | 1.94e-4 | 0.004× | 5.04e-16 | degenerate |
| M04 wave | 3.25e-7 | 0.001× | 5.37e-17 | degenerate |
| M05 nltest | 3.34e-3 | **0.341×** | 3.35e-16 | degenerate |
| M06 stiff mild | 6.26e-3 | **0.417×** | 7.83e-16 | degenerate |
| M07 stiff det | 8.58e-4 | 0.022× | 1.54e-15 | degenerate |
| M08 stiff mixed | 5.14e-3 | 0.047× | 3.67e-16 | degenerate |
| M09 weak | worst 1.9e-3 | 0.066× | worst 4.0e-3 | — |

**Degenerate ladders: 0 of 8 in the scheme-matched arm, 8 of 8 in the unrestricted one.**

#### §10's prediction 2, settled in both directions

The prediction: *"at least one of the C0 baseline and the pipeline returns a
pathwise-exact answer on ≥ 3 of S01–S08, and H2 catches every one of them. If it catches
**none**, H2 is broken, not vindicated."*

- **Unrestricted arm: 8 of 8, and H2 caught all 8.** Errors of 5e-17 to 1.5e-15 — not
  "accurate", *exact to rounding* — with the ladder flagging every one. The prediction's
  floor of 3 was raised to the maximum, and its failure condition (catching none) did not
  occur.
- **Scheme-matched arm: 0 of 8.** This is the result worth keeping, because it was not
  predicted at all. §0b's whole premise is that a transcribed closed form is the obvious
  move for a single tool-less completion, and §10 prediction 2 expected exactly that. What
  happened instead is that **one paragraph of brief text suppressed it completely**: told
  plainly that an exact evaluation is detected and excluded, Opus wrote a discretised
  scheme nine times out of nine. The detector was not what stopped the transcription; the
  *disclosure* was. H2 is still necessary — it is the thing that makes the disclosure
  credible, and it is what caught the other arm's eight — but the measured deterrent is
  the sentence, not the mechanism.

#### What the baseline chose, and what it means for prediction 1

In the scheme-matched arm Opus shipped **Milstein** on five of the eight (M01, M03, M04,
M06, M08), a **stochastic Heun / trapezoidal predictor–corrector** on M02 and M07, and
plain Euler–Maruyama on M05. Prediction 1 says "the pipeline ships Euler–Maruyama or
Milstein on at least 7 of 9 problems"; the *baseline* already does better than that on
two entries, with a scheme that is in neither `sde_manual.md` nor the suite's own
`SCHEME_STAGES` table — chosen with no manual in the prompt at all.

That makes M02 the sharpest row in the table: the predictor–corrector reaches **5.4e-8**,
which is 22 000× inside the gate and better than §3f's measured `SRA1` at the same step
count. The reason is the problem rather than the scheme — M02's solution is affine in `t`
and `W`, the class the SRA schemes integrate exactly (§3f finding 3) — but it means the
baseline is not the weak comparator §0d assumed, and step 10 has a real opponent on its
own side of the table.

**The two tightest margins are M06 at 0.417× and M05 at 0.341×**, so the gates set in §2b
are not trivially clearable: a correct scheme has about 2.4× of room on the hardest entry,
which is the margin §2b's calibration predicted for a correct Euler–Maruyama and is being
spent here on a *better* scheme at the same step.

Records: `benchmark/results/baselines/oneshot_sciml_{scheme_matched,unrestricted}.json`.
Both were re-run after the §7a defect-7 fix so that each arm's solver sources survive on
disk; the pre-fix records agreed with them on every verdict and were pruned rather than
kept, since the artifacts they referred to had been overwritten.

---

## 10. Predictions, to be checked against the outcome `[N]`

Written before the runs so the plan can be graded rather than re-narrated:

1. **Scheme-matched arm, non-stiff (S01–S05, S09):** the pipeline ships Euler–Maruyama or Milstein
   on at least 7 of 9 problems, and is beaten by SRA1/SRIW1/DRI1 at matched work by **≥ 10×** at
   relative error 10⁻⁴ on the additive and diagonal problems. Mechanism: §0d.
2. **Unrestricted arm:** at least one of the C0 baseline and the pipeline returns a pathwise-exact
   answer on ≥ 3 of S01–S08, and H2 catches every one of them. If it catches _none_, H2 is broken,
   not vindicated.
3. **S06–S08 (stiff):** no system wins, including `StochasticDiffEq.jl` — SciML's own text says the
   mixed case defeats adaptivity and implicitness `[W]`. If the pipeline merely _matches_ EM at a
   small `dt` here, that is a pass, and the suite's most interesting row.
4. **S09 (weak, non-commutative):** the Monte-Carlo noise floor at 10⁵ paths, not the scheme, sets
   the resolvable gate. Expect the gate to need widening on noise grounds, and expect that to be the
   defect the smoke test (step 8) finds — this repo has twice shipped a ranking number with no noise
   margin on a sampled quantity `[R]`.
5. **The EM anchor is not one number** (§5): the scalar and many-state factors differ by more than 3×.
6. **The suite's value is conditional.** If the manual gains an SRK section and prediction 1 is then
   falsified on a repeat run, the adoption paid for itself. If the suite is run once, loses as
   predicted, and the manual does not change, it measured a limitation that `sde_manual.md` rule 3
   already documents in one sentence — and this plan will have cost more than that sentence.

**Scoreboard as of rev 5** `[N]` — three settled, two settled against the plan, one open:

| # | about | verdict |
| - | ----- | ------- |
| 1 | the Rößler schemes beat the pipeline by ≥10× at matched work | **ungradable as written** — it pins two axes at once. Holding work fixed: 36×–1.7e6× on all nine (§3f reading A). Holding error fixed at 1e-4, which is its own phrasing: 6.2×, 62.5×, 5.0×, 12.5×, 4.0×, 1.0×, 2.0× and **0.2× on M08** — so ≥10× holds on two of five non-stiff entries and fails on the rest (reading B) |
| 2 | ≥3 pathwise-exact answers, all caught by H2 | **confirmed at the maximum in one arm, premise overturned in the other** (§9a): 8 of 8 in `unrestricted`, all caught; **0 of 8** in `scheme-matched`. The deterrent measured is the brief's sentence, not the detector |
| 3 | nothing wins on the stiff triple | **partly confirmed early** (§3f): on the work axis Euler–Maruyama *wins* on M08, where `β² = 4900` against `α = 5000`. Needs step 10 to finish |
| 4 | the MC floor sets M09's gate, and the gate needs widening | **falsified with its mechanism inverted** (§2b finding 4): the floor binds by putting a correct scheme's error *below its own standard error*, not by forcing the gate wider. No widening was needed |
| 5 | the EM anchor is not one number; the shapes differ by >3× | **confirmed at 8.1×** (§5a) — and the scalar factor needed a third number, because 1.03× parity is an artifact of a latency-bound dependency chain |
| 6 | the suite's value is conditional on the manual changing | **open by construction.** It cannot be graded until a second run after an `sde_manual.md` with an SRK section, and §10a has now added a second prerequisite: there is no SRK family in the kernel's vocabulary either |

---

### 10a. Step 10, probed on three problems `[rev5 2026-11-10; diagnosis rev6, probes M06/M09 rev7 2026-11-11; fix 10b rev8, M02 10c rev9, the other five 10d rev10 2026-11-12; M09 10e rev11, the offline re-rank 10f rev12 2026-11-13, N]`

One pipeline run on M01, scheme-matched arm, to measure what the step costs before
committing nine of them: **19.6 minutes of wall clock**, `VERIFIED_PASS`, verified score
10, graded relative strong error **2.06%** against the 9.5% gate. Three plans, three
solver↔evaluator iterations in total. So the scheme-matched arm is ~**3 hours** for nine
problems and both arms ~**6 hours**, which is the number the step-10 decision needs.

**The pipeline built better schemes than §10 predicted and then shipped the worst of
them.** The three plans were `1-euler-maruyama`, `2-milstein` and `3-heun-srk` (declared
`scheme: stochastic-heun-stratonovich`), and the harness grades them:

| plan | declared family | rel err / gate | the conductor's score |
| ---- | --------------- | -------------- | --------------------- |
| **1-euler-maruyama** | `euler-maruyama` | **0.2166** | 10 — **winner** |
| 2-milstein | `milstein` | 0.0261 | 8 |
| 3-heun-srk | `other` | **0.0122** | 8 |

**`wrong_plan_won: True`**, by 17.7× against a sampling-noise margin of 0.0048. The
conductor picked the least accurate of its own three plans on a problem where the other
two were an order of magnitude better, and scored it 10 while scoring them 8.

That single line is the whole argument for §7a's defect 1. **Before that fix this field
was `null`** — `plan_harness_error` had no pathwise branch, so the harness had no number
to rank the three plans on and the most interesting result of the run would not have been
recorded at all. It was fixed two hours before the run that produced it, on the strength
of a blank column in a smoke test.

#### A vocabulary gap the probe exposed

`3-heun-srk` declares `scheme_family: other`, and it has to: the kernel's SDE vocabulary
(`verifylib/kernel/plan_meta.SDE_FAMILIES`) is `euler-maruyama | milstein |
tamed-euler-maruyama | tamed-milstein | implicit | exact-transition |
quadratic-exponential | other` — **there is no Rößler or SRK family in it**, which is the
same absence §0d found in `sde_manual.md`, one layer down. `other` carries no cost per
step, so `_wp_sde_work` falls back to `stages = 1` and records
`stages_source: "default"`.

The fallback's *direction* is the problem. It **under**-states the work of the plan being
measured — a stochastic Heun costs 2 stages and an SRK 3 or 4 by §3f's measurements — and
understating your own side's cost flatters it against comparators whose stages were
measured honestly. Had `3-heun-srk` won, its curve would have been drawn at a quarter of
its true work against the very scheme family it is a member of.

Rather than guess a stage count for `other` (which would be the same error with better
manners), the report now says it: a series resting on a defaulted stage count is labelled
a **lower bound**, with the direction of the bias stated. Fixing the vocabulary is the
honest repair and it is not this plan's — it changes what every plan-creator declares — so
it is recorded here as the thing step 10's headline table depends on.

**And one test broke, legitimately.**
`verifylib/tests/test_schema.py::test_every_live_sde_spec_declares_a_floor_for_every_family_it_admits`
is a *census* of what is staged under `workspace/` — "the artifacts, not the rule", in its
own words — and it pinned "exactly two SDE specs admit a Milstein plan". The probe's
`problem_spec.json` is a third, and it is right to be: M01 is scalar with multiplicative
noise and `dg/dX = 0.87` in closed form, which is `agents/formulator.md`'s eligibility
rule exactly, and the spec declares `expected_strong_order_milstein: 1.0` like the other
two. So the **rule** the test is about held across a problem family that did not exist
when it was written, and only the count moved. Updated to the three, named, with a note
that both numbers are properties of a directory every pipeline run rewrites — a census of
mutable state will go stale again, and the assertion inside the loop is the invariant.

**Two smaller observations, both from `STATE.md`.** The formulator logged two ambiguities
of its own accord: *"the kernel SDE evaluation vocabulary has no pathwise-strong-error
key, so the formulator carried it as `rel_l2_err_max` plus the CRN strong-order study"*
and *"the stated pathwise gate has no evaluation_thresholds key; carried verbatim as
`rel_l2_err_max: 0.095`"*. The pathwise route reached the kernel as a relative-L2 bound
with a strong-order floor beside it — which is a reasonable translation and is *not* the
quantity the harness grades, so the pipeline's self-score and the harness's verdict agree
here by construction rather than by measurement. That is worth knowing before nine runs
are read.

#### What actually decided the run: one check, at its own noise floor

The three plans differ on **exactly one** of the kernel's fifteen checks.
`constraints_ok`, `invariants_ok`, `resolved`, `richardson_stable`,
`reported_moments_agree` and `provenance_as_expected` pass on all three, and so do
*both* moments, with room to spare:

| plan | mean rel err (tol 5%) | var rel err (tol 20%) | KS `D` | critical | ratio | `terminal_law_ok` | `moments_ok` | score |
| ---- | --------------------- | --------------------- | ------ | -------- | ----- | ----------------- | ------------ | ----- |
| **1-euler-maruyama** | 0.089% | 1.12% | 0.004602 | 0.006074 | **0.76×** | **true** | true | **10** |
| 2-milstein | 0.075% | 1.35% | 0.006141 | 0.006074 | **1.011×** | false | false | 8 |
| 3-heun-srk | 0.107% | 1.71% | 0.006754 | 0.006074 | **1.112×** | false | false | 8 |

The selection was made by the terminal-law Kolmogorov–Smirnov statistic
(`verifylib/kernel/sde/laws.ks_standardized`), and **Milstein lost it by 1.1% of the
critical value**.

**And the reason string the two losers carry is wrong.** Both read *"the variance
passes; the mean is slightly above threshold"* — `verifylib/kernel/score.py:458` —
while their means sit 67× and 46× *inside* tolerance. That row is a fallback, not a
diagnosis: `moments_ok` is a conjunction that includes `terminal_law_ok`, and the
rubric reads `orders_ok`, `invariants_ok` and the Dynkin gap only *inside*
`if passed(moments_ok) and passed(resolved) and passed(constraints_ok)`
(score.py:427). A plan that fails only the law test falls past every explanatory row
to the variance fallback and is told it failed on the mean. `3-heun-srk` also has
`orders_ok: false`, and that is swallowed by the same guard — two different failures,
one identical and incorrect sentence.

#### The statistic reproduces exactly, and a pathwise-*exact* solver fails it

`docs/sciml-benchmark-data/ks_gate_resolution.py` re-runs the kernel's own test —
same standardization, same `KS_C95 = 1.3581`, same n = 50 000, same `dt = 0.0016`,
increments drawn as one `(paths, steps)` block as every graded solver draws them — on
four solvers sharing one Brownian path. Three of them are the run's plans and the
fourth is the pathwise-exact map `X_T = X_0 exp((μ − σ²/2)T + σ W_T)`. At seed 42 it
reproduces the run to six digits, which is what licenses the fourth row:

| solver at seed 42 | rel strong err | KS `D` | verdict |
| ----------------- | -------------- | ------ | ------- |
| **exact** (zero error, same increments) | 0 | **0.007133** | **FAIL, 1.174×** |
| euler-maruyama | 2.18e-2 | 0.004602 | pass, 0.758× |
| milstein | 2.54e-3 | 0.006141 | FAIL, 1.011× |
| heun | 1.18e-3 | 0.006754 | FAIL, 1.112× |

**On the run's own draw, a solver with no error at all fails the gate — and fails it
worse than the Euler–Maruyama that passed.** The gate did not rank the three schemes
badly; it is not ranking schemes at all.

Over nine seeds:

| reading | over 9 seeds |
| ------- | ------------ |
| KS order == pathwise-error order | 6/9 |
| KS order == exactly **reversed** | 3/9 |
| the configuration this run produced (EM passes, both better schemes fail) | **1/9** |
| seeds where the **pathwise-exact** solver fails the gate | **5 of 9** (42, 3, 5, 7, 8) |
| the exact solver's own KS `D` | 0.0035–0.0138, against a bar of 0.0061 |
| between-scheme spread `\|D(milstein) − D(em)\|` | 0.00062–0.00154, i.e. **10–25% of the bar** |

The schemes separate by a tenth to a quarter of the critical value; the *draw* moves
the statistic by four times the critical value. The run's outcome was a coin landing
on its edge, and it is 1-in-9, not systematic — the earlier reading, that the gate
penalises accuracy, was wrong in that direction and is corrected here.

#### Why, with the number

`laws.py`'s docstring is explicit that the sample is standardized to remove the
location and scale bias "which is what the `O(dt)` error mostly is", leaving shape.
That is right, and its finding 4 is real — three defect classes pass every other SDE
check. But the standardization divides by an **estimate**, and on this law that
estimate is the noisy quantity. The relative standard error of a sample standard
deviation is `sqrt((κ − 1)/4n)`; M01's terminal lognormal has excess kurtosis 47.65,
so `κ = 50.65` and at n = 50 000 the divisor itself carries **1.58%** error. Measured
over the nine seeds the exact sample's `|sd/sd_exact − 1|` runs 0.04%–2.94%, as
predicted, and its KS `D` correlates with that error at **+0.70**. A random scale
error is not removed by the standardization — it *is* the standardization — and here
it dominates the shape signal the test is for.

Two consequences, both outside this plan's scope and recorded for the same reason the
vocabulary gap is:

1. `KS_C95/√n` is the distribution-free Kolmogorov value, and this statistic is not
   distribution-free once its divisor is estimated from the sample. The honest bar is
   a parametric-bootstrap critical value: sample the declared law at the same `n`,
   standardize the same way, take the 95th percentile. That is one cheap simulation
   per spec and it is *exact* rather than asymptotic.
2. Until then, `terminal_law_ok` should be **reported and not conjoined into
   `moments_ok`** — the same treatment `orders_ok`'s abstention already gets at
   score.py:447, and for the same stated reason: it is a property of the kernel's
   estimator, not of the solver.

`laws.py`'s own calibration table came within a hair of finding this — it records
correct Euler–Maruyama on GBM at `D = 0.00580` against `crit = 0.00607` and calls it
"pass, by a hair". What it never ran was the exact sampler as the control, which is
the row that shows the hair is noise.

#### The plan also lost for converging too fast

`3-heun-srk` declared `expected_strong_order: 0.5` and measured **0.983** (median
0.970, se 0.015) over `dt ∈ {0.02, 0.01, 0.005}`. The band is two-sided at ±0.4, so
`[0.1, 0.9]`, and 0.983 is *above* it: `strong_ok: false`. A Stratonovich Heun on
scalar commutative noise is strong order 1.0, so the plan-creator under-declared and
the two-sided band charged it for the discrepancy. The band is right to be two-sided
— a scheme converging faster than its design order usually means the error estimate
is broken, not that the scheme is better — but on a *conservatively declared* plan it
converts a correct result into a failed check. Had `moments_ok` passed, this alone
would have capped the best of the three plans at 8.

#### The pipeline's own self-estimate is the variance error, not an accuracy

`STATE.md` records `est_err` 1.12e-2 / 1.35e-2 / 1.71e-2 against measured relative
strong errors of 2.06e-2 / 2.48e-3 / 1.16e-3 — an ordering that is exactly reversed.
That is not a bad accuracy estimate; it is **not an accuracy estimate**:
`estimated_rel_error` is byte-identical to `variance_rel_err` in all three
`evaluation.json`s. At 50 000 paths every one of the three sits below its own
`mc_se_rel` of ~0.0316, which the conductor noticed and logged under
`unmeasured_accuracy` for two of the three plans. The pipeline had no number
describing how accurate any of its plans was, said so in its own record, and ranked
them anyway — by `score`, per `ranker: 'decided_by: score'`.

#### And the two better plans were never finished

`notes` in `STATE.md`: *"winner early-exit fired in round 1 (Path A, certified: true);
plans 2 and 3 left unfinished at iter 1, neither plateaued nor at max_iter."* Both
better plans are parked at `state: await_solver`. The early exit is a cost control and
it is triggered by `certified`, which here means "passed a KS test at 0.76× of a bar
an exact solver fails at 1.17×". One check at its noise floor both selected the
winner and stopped the work on its competitors.

#### What this changes about step 10

Nothing about the suite, and the headline it was built for still stands: the harness
graded all three plans correctly, ranked them correctly (`wrong_plan_won: true`, 17.7×
against a margin of 0.0048), and the comparator table is unaffected. What changes is
what the remaining runs *measure*. The step-10 table will read "the pipeline shipped
the worst of its own plans" on every entry whose spec declares a `terminal_law` the
kernel can build, and that sentence will be about `laws.py`'s critical value rather
than about scheme choice. Both must therefore be reported in the same table: the
plan the pipeline shipped, and the plan its own harness-graded errors say it should
have shipped. `harness_by_plan` now carries both, which is §7a defect 1.

~~The two probes the next runs are for — **M06** (tightest gate margin, 0.417×, and
stiff) and **M09** (the only functional-route entry, so a different instrument
entirely) — test exactly one thing: whether this is a property of the pathwise route
or of the SDE evaluator. M09 is the sharper of the two, because its gates are moment
gates and it has no pathwise error for the vocabulary to be missing.~~ **Both ran — see below. The answer is the evaluator.**

#### The answer: three problems, three mechanisms, one defect `[rev7 2026-11-11]`

Both probes ran. **`wrong_plan_won: true` on all three problems, by three different
routes, only one of which is the KS gate.** The question the probes were for — pathwise
route, or SDE evaluator — resolves to the evaluator:

| | route | plans | pipeline scores | winner / best by harness | ratio | margin | how the winner was chosen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **M01** | pathwise | 3 | 10 / 8 / 8 | `1-euler-maruyama` / `3-heun-srk` | **17.7×** | 0.0048 | scores *separated*, by a gate reading its own estimator's noise |
| **M06** | pathwise | 3 | **10 / 10 / 10** | `1-euler-maruyama` / `3-trapezoidal-drift-milstein` | **5.1×** | 0.0111 | ~~flat tie — plan **index**~~ **`decided_by: sqi`, by 0.014 — see 10b** |
| **M09** | functional | 3 | **8 / 8 / 8** | `1-euler-maruyama-antithetic` / `2-exact-lognormal-integrating-factor` | **3.3×** | 0.1468 | the ranker never ran — `run.py`'s fallback, in declaration order (10b) |

The common factor is not the gate, it is that **the pipeline's scores do not separate
plans its own harness separates cleanly**. On M01 one check out of fifteen did the
separating and did it backwards. ~~On M06 and M09 nothing separated them at all and the
tie-break is `plans/` in lexical order.~~ **Corrected in 10b:** the equal *scores* on
M06 and M09 do not mean nothing separated the plans — `score` is only the first of four
ranking keys. On M06 the quality index separated them and got it backwards, because the
best plan's accuracy was too small to resolve and its index was renormalised over fewer
components; on M09 the ranker never ran at all, and the winner is `run.py`'s fallback
taking the first plan in declaration order. All three shipped the worst or near-worst of
three plans.

**M06 — `PASS (verified)`, 21.5 min, score 10, strong error 0.743% against the 1.5%
gate.** Three plans, three iterations, every one certified at 10:

| plan | declared family | design order | kernel est. err | strong err / gate |
| --- | --- | --- | --- | --- |
| **1-euler-maruyama** | `euler-maruyama` | 0.5 | 1.43e-2 | **0.4955** — winner |
| 2-milstein | `milstein` | 1.0 | 1.43e-2 | 0.4171 |
| 3-trapezoidal-drift-milstein | `milstein` | 1.0 | 2.55e-3 | **0.0968** |

This is the cleanest demonstration of the defect, because here **no gate failed
anybody**. M06 deliberately declares no `terminal_law` — "tanh(Normal) is not a law the
kernel implements" — so the statistic that decided M01 cannot fire, and with every plan
certified at 10 the conductor had nothing to choose on. The semi-implicit plan, the one
that solved the actual stated challenge (`sde_manual.md` rule 3 calls implicit schemes
"not yet supported", and it built a trapezoidal-drift Milstein anyway), is **5.1× more
accurate** than the Euler–Maruyama that shipped, and placed third by filename. Note also
that plans 1 and 2 report an *identical* self-estimate, 1.43e-2, while their true errors
differ by 19% — §10a's note that the self-estimate is a variance error, not an accuracy,
again.

**M09 — `RUN ERROR`, timed out at 60.0 min, `graded_incomplete: true`, phase
`running`.** This is the sharpest result of the three and it is worth stating carefully,
because the pipeline's own grade is a partial run while the harness numbers are complete:

| plan | declared family | worst functional err | / its gate | harness verified score | pipeline score |
| --- | --- | --- | --- | --- | --- |
| **1-euler-maruyama-antithetic** | `euler-maruyama` | 1.50e-2 | **0.5490** — winner | **8** | 8 |
| 2-exact-lognormal-integrating-factor | `exact-transition` | 3.317e-3 | 0.1658 | **10** | 8 |
| 3-weak-order-2-ito-taylor | `other` | 3.318e-3 | 0.1659 | **10** | 8 |

The harness grades the winner **8** and the other two **10**, on gates of 0.020 / 0.016 /
0.029 for the three declared functionals. So on M09 the pipeline did not merely rank its
plans wrongly — it shipped the only one of the three its own harness does not score 10,
and the flat tie that let it do so was **manufactured by a problem-wide cap**:

- `verification.terminal_law` was declared `[{lognormal}, null]`. `laws.build(None)`
  returns `"terminal_law is not an object"` (`laws.py:118`), that component reports
  `unresolved`, and `_distribution` folds **any** `unresolved` into
  `moments_ok: "unavailable"` (`driver.py:1004-1005`) — which makes the rubric's 9 and 10
  rows unreachable for **every plan on the problem**. One `null` in a list disables the
  top of the rubric for all of them. All three evaluators diagnosed this independently;
  the conductor confirmed it against the kernel source and re-dispatched the formulator,
  and the spec on disk now carries a one-entry list, so the repair happened *inside* the
  run. The X2 law is a non-trivial mixture and is genuinely not declarable — the kernel
  only checks declared entries, so dropping the placeholder loses nothing.
- A second, independent 8-cap on plans 1 and 3: the two-sided strong-order band
  (0.5 ± 0.4) failed them for **overshooting** — measured 0.973 and 1.049 against a
  declared 0.5 — which `references/project_manual.md` itself calls noise rather than a
  defect, and which `ranking.order` no longer charges. Same shape as M01's `3-heun-srk`
  losing for converging too fast.

Two further observations from M09's record. The conductor logged **two more vocabulary
gaps** of its own: `R22`, a problem statement that both denies and asserts a
terminal-moment reference (resolved by deriving the moments and marking them
non-authoritative), and `R25`, *"no `evaluation_thresholds` key expresses a
cross-path-count non-identity test"* — the check that a value identical at 100 000 and
25 000 paths is a constant rather than an estimate is enforced **only by the external
harness**, by no spec field. That is the third and fourth independent formulator report
of the same class of gap — and a **fifth** turned up in the test suite afterwards: M09's
spec carries a top-level `terminal_moments_at_T` holding the exact values of all three
graded functionals, cross-checked against `solve_ivp` on the moment ODE at 3.8e-15, in a
field nothing reads. Five instances, four formulators, one missing vocabulary. (It is
also the **fourth** stale-census test failure in two days, all four from asserting
equality against a `workspace/` directory that every pipeline run rewrites; registered
with the others under `REGENERATED`.) And the final `kernel_metrics` re-run hit its own 900 s
timeout, so M09's top-level `kernel` block reads `{"status": "timeout"}` and the
per-plan `kernel_rerun.reason` fields are `null`; the per-plan `kernel` blocks from the
run itself are intact and are what the table above uses.

**The timeout is not a defect.** 60.0 min of monotonic time took 82 min of wall clock
because the machine slept mid-run; `run_conductor` measures on the monotonic clock
deliberately (`run.py:185`), which is the §8 finding about a laptop sleeping through a
40-minute timeout, working as intended. What it does mean is that **one hour is not
enough for a 2-D functional problem with three plans** — M06 finished in 21.5 min and
M09 was still inside its third evaluation at the cut — so the step-10 budget for the
remaining entries should be read off M09, not M06.

**What the three probes change about step 10.** The §10a conclusion stands and widens:
the step-10 table must carry both the plan the pipeline shipped and the plan its
harness-graded errors say it should have shipped, on **every** entry, because the
inversion does not need a `terminal_law` to happen — a tie is enough, and ties are the
common case once more than one plan certifies. Two concrete consequences for the
remaining seven runs: raise the per-problem timeout above 3600 s, and expect the
"shipped the worst of its own plans" column to be populated by plan index rather than by
any measurement.


#### 10b. The tie-break fix, and what it actually was `[rev8 2026-11-12]`

The fix went in, and **the diagnosis above was wrong about M06 in one respect that
matters.** §10a says M06 and M09 both tied flat and fell to plan index. That is right
for M09 and wrong for M06, and the ranker says so when you simply run it on the
finished workspace:

```
plan                           score cert provenance    sqi    acc    ord    str    res
1-euler-maruyama                10.0  yes  analytic    0.722  0.801  1.000  0.143  0.910
2-milstein                      10.0  yes  analytic    0.722  0.801  1.000  0.144  0.907
3-trapezoidal-drift-milstein    10.0  yes  analytic    0.708      -  1.000  0.244  0.908
best: 1-euler-maruyama   decided_by: sqi
```

**Not a tie: `decided_by: sqi`, by 0.014.** And the mechanism is sharper than a flat
tie. `accuracy` returned **None** for the winning-by-rights plan — correctly, under F16:
its `estimated_rel_error` of 2.545e-3 is *below its own error bar* of 8.82e-3
(`mc_se_rel` 4.410e-3 at `ci_mult` 2), so the error is not located and an unlocated
error is not a measurement. `sqi` then renormalised its index over the remaining three
components, and a weighted mean over three terms came out below a weighted mean over
four. **The most accurate plan in the pool lost because it was too accurate to
resolve.**

`sqi`'s own docstring names the half of this that is safe — a plan that *skipped* a
check forfeits a circularity break, `score.ceiling` caps it, and `rank_key` reads
`score` before quality, so "skipping a check can improve the index; it cannot improve
the rank". The other half has no backstop: nothing caps a plan whose component is
missing because the kernel **could not resolve** it, and the direction the index moves
is set by whether the absent component happened to sit above or below the mean of the
rest.

**The fix is `ranking.comparable_keys`:** a component may contribute to a *comparison*
only where every plan in the pool measured it. That is not a new rule —
`cost_admissible` is the same rule stated for one component, and this is it applied to
all of them. `sqi` keeps renormalising, because a single plan's index must; `rank_key`
now restricts the comparison, and `cli.py rank` prints the restricted index with a note
naming which components were dropped and each plan's own index beside it (otherwise the
table shows the winner with a *lower* number than the plans it beat). **F16 is
untouched** — a noise-limited accuracy is still None, and deliberately so.

M06 now ranks `3-trapezoidal-drift-milstein` first, `decided_by: sqi`, which is the
plan the harness measures as 5.1× more accurate.

**M09's inversion was something else entirely, and is the second fix.** Its STATE.md
carries **no `best_plan` at all** — the run timed out before the conductor's Phase 3, so
the ranker never ran on it. `run.py`'s `summarize_pipeline` fell back to the
highest-scoring plan; all three sat at score 8 on iteration 1; `max` returned the first
key in declaration order, and `harness_by_plan` then compared the harness errors against
it and recorded `wrong_plan_won: true`. The comparison was right and the *attribution*
was not: the pipeline had not chosen a plan, the grader had. The fallback now records
`best_plan_source: "state" | "fallback"` and `best_plan_tied` when its key is exhausted,
and `harness_by_plan` carries `winner_source`/`winner_untied`, so a step-10 table can
tell a ranking defect from an unfinished run. A ranking defect and a run that stopped
early are different findings and the record must not spell them the same way.

**What the change does across the corpus.** Nineteen multi-plan workspaces on disk;
**all nineteen** have at least one component missing on at least one plan, so the
restriction is live everywhere, and the winner moves on **three**:

| workspace | old winner | new winner | harness gap | pool noise margin | verdict |
| --- | --- | --- | --- | --- | --- |
| `sde_sciml_stiffquad_mild` | `1-euler-maruyama` | `3-trapezoidal-drift-milstein` | **0.4955 → 0.0968** | 0.0111 | **fixed** — 5.1×, 45× outside the margin |
| `sde_sciml_weak_noncommutative` | `3-weak-order-2-ito-taylor` | `2-exact-lognormal-integrating-factor` | 0.16588 → 0.16585 | 0.1468 | indistinguishable |
| `sde_heston_call_feller` | `3-full-truncation-euler` | `1-exact-v-conditional-mc` | 0.0391 → 0.0779 | **0.2973** | indistinguishable |

So: **one real improvement and two moves inside the noise, and no measurable
regression.** The B03 row is the one to read carefully, because it is the only one where
the change moves the winner *away* from the lower harness error and it is worth being
explicit that this is not a regression: plan 3's own margin is 0.2973, 7.6× its measured
error, so the 0.0388 gap is an eighth of the noise floor — which is why that record's own
`wrong_plan_won` reads `False`. What B03 does show is a *separate* defect the fix
exposes rather than causes: there the accuracy component is the declared functional (F3),
it reads 0.451 for plan 1 against 0.048 for plan 3, and the harness ranks them the other
way round. Under the old rule plan 3 won anyway — carried by an `order` component the
other two plans never measured, which is the M06 mechanism pointing the right way by
luck. **The old winner there was right for the wrong reason; the new one is wrong for
the right reason**, inside a margin where neither is measurable. The accuracy
component's calibration on a sampling-limited price is its own question and is not this
fix.

**Tests.** `test_benchop_fixes.py` gains **seven**, with M06's two real `<metrics>`
blocks transcribed as the fixtures — a hand-built pool does not reproduce the failure,
because it turns on the interaction between which component is missing and where the
remaining three sit, so any fixture chosen to make the point would be fitted to it. They
pin the premise (2.545e-3 is inside its bar and 0.0143 is not), the unrestricted indices
0.722/0.708 that got it backwards, the restricted order, the whole three-plan pool
through `rank_key`, that a component the pool *did* measure still decides, that a pool
where nobody measured one does not rank on nothing, and that `sqi_cert` is unchanged.
`test_orchestration.py` gains **E10**, three tests on the winner's provenance. The suite
reads **1287 passed, 3 skipped** against a 1277 baseline, and a collection diff confirms
all ten are additions with nothing displaced — worth checking rather than assuming, on a
day whose other findings were two silently-dropped measurements.

#### 10c. M02, the first run under the fix: compliance confirmed, and the index read past its own resolution `[rev9 2026-11-12]`

One of the six un-run entries, at the 7200 s timeout 10a asked for. **`UNDERCLAIM`,
13.0 min, pipeline 9, harness-verified 10**, graded strong error **1.34e-7** against a
1.2e-3 gate. M02 was chosen over the other five because §9a measured the C0 baseline
shipping a stochastic Heun on it and reaching 5.4e-8 — a known-wide scheme space and a
known-large accuracy spread, which is what was needed to make the ranker's answer differ
from declaration order.

**It did, and the conductor followed it.** The pipeline shipped
`3-drift-implicit-trapezoidal`, with `best_plan_source: state`:

| plan | declared | strong err | / the EM | kernel score | `est_err` | `mc_se_rel` | `invariant_worst_headroom` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1-euler-maruyama | `euler-maruyama` | 2.207e-1 | 1× | 9 | 1.1105e-2 | 9.996e-3 | 0.671 |
| 2-stochastic-heun | **(none)** | **4.524e-5** | 4879× | 9 | 0.0120 | 9.996e-3 | 2.2330 |
| **3-drift-implicit-trapezoidal** | **(none)** | 1.115e-4 | 1980× | 9 | 0.0120 | 9.996e-3 | **2.2379** — winner |

This is the first observation on this suite where a *correct* selection and a
declaration-order selection are distinguishable, and the question 10b left open is
answered: **the conductor obeys its ranker.** On M06 the ranker's answer happened to be
plan 1, so compliance and index order could not be told apart; here the ranker said plan
3 and plan 3 shipped.

**The fix was inert on this pool, and that is the right outcome.** All three plans are
missing `accuracy` *and* `consistency`, so :func:`comparable_keys` drops the same two
components for everybody and the restricted index equals each plan's own — 0.887 / 0.886
/ 0.655, identical either way. The old ranker would have picked plan 3 too. M02 therefore
confirms the fix does nothing where nothing is asymmetric, which is the property that
makes it safe to run on the remaining five; it is **not** evidence that the fix works,
and the only evidence for that remains M06.

**What M02 does find is the next layer of the same defect.** `wrong_plan_won: true`
again — plan 3 at 1.115e-4 against plan 2 at 4.524e-5, a factor of **2.46×** against a
margin of 2.49e-6, so the gap is real and 45× outside the noise. And the index that chose
between them reads **0.887 against 0.886**:

- `accuracy` did not resolve on **any** of the three. Every plan's `estimated_rel_error`
  sits at its own `mc_se_rel` — 1.1105e-2 against 9.996e-3, and **0.0120 against 9.996e-3
  for both of the good plans, identical to three digits while their true errors differ by
  2.46×**. So 0.35 of the index, the weight that is supposed to carry accuracy, ordered
  nothing at all.
- `order` is 1.000 for all three (and the one-sided fix is working as designed: plans 2
  and 3 read `observed_order` 2.0004 and 2.0000 against a declared 1.0, a full order of
  overshoot, and are no longer charged for it).
- Which leaves `structure`, at 0.746 against 0.744 — and that comes from
  `invariant_worst_headroom` **2.2379 against 2.2330**.

**The winner was chosen on the fourth significant figure of a structural diagnostic,
between two plans whose accuracies the pool could not measure.** That is 10b's defect
generalised: 10b stopped the index being compared across *different* component sets, and
M02 shows it is also being read to a precision it does not have when the sets are the
same. The discipline exists everywhere else in this repo and is missing here —
`run.py`'s `harness_error_margin` refuses a difference inside the sampling noise,
`ranking.uncertainty`'s snap refuses one inside the error bar, `cli.py compare` refuses
one between plans — and the *composite* index has no notion of its own resolution, so
`decided_by: sqi` is reported identically for M06's 0.014 gap, M02's 0.001 gap, and a
genuine separation. ~~Recorded rather than fixed: a resolution floor on the index changes
rankings corpus-wide, it is the one remaining input `benchmark/rank_study.py` was built
to fit against P90 regret, and guessing the floor here would be the same mistake as
asserting the weights.~~ **Superseded by 10f, in both halves.** `benchmark/rank_study.py`
did not exist when this was written — it does now, and that section builds it. And the
floor, fitted against regret rather than guessed, is **harmful**: monotonically on these
nine pools and with the opposite sign on the BENCHOP holdout, because a tie on the index
hands the decision to declaration order, which is `1-euler-maruyama` on 9 of 9 sciml
pools and the exact-sampling plan on 5 of 9 BENCHOP ones. Withdrawn.

**Defects 9 and 10 reproduce on a second problem, with a different wrong number.** M02's
record carries the same uncapped error-severity ledger violation — *"the run used
num_paths=1000, the spec asks for **20000**"* — and the same ignored `score_cap: 3`,
grading 10. A third formulator, a third path count the suite does not grade at (100000 on
M06, 20000 here, against the suite's 1000 both times). So neither defect is M06-specific,
and the formulator-side half of it is systematic rather than a slip.

**And the vocabulary gap is now worse than §10a recorded it.** M01's Heun-SRK plan at
least declared `scheme_family: other`; M02's two best plans — the two that beat the
Euler-Maruyama by 1980× and 4879× — declare **no `scheme_family` at all**, so
`harness_by_plan` reads `None` for both and this suite's primary axis has no scheme name
for the plans that won it. On a *scheme* benchmark that is the field the result is about.

**For the remaining five runs:** 13.0 min at the raised timeout, so 7200 s is ample and
M09's cut was specific to a 2-D functional problem with three plans. The step-10 table
will need the `scheme_family: None` column read as a defect rather than as a measurement.

#### 10d. The remaining five, under the fix: step 10 complete, and the vocabulary gap turns out to be an incentive `[rev10 2026-11-12]`

M03, M04, M05, M07 and M08, run in one invocation at the 7200 s timeout —
`--results-json benchmark/results/pipeline/sciml_rest/sciml_probe_rest.json`, so the four
earlier records and their logs are untouched. **All five completed, none timed out**, at
12.3 / 18.0 / 16.2 / 12.0 / 21.4 min — **80.0 min** for five problems, against the 60 min
M09 alone exhausted at the old budget. The raised timeout is settled.

| | slug | verdict | pipeline | verified | strong err | gate | inside | plans | iters | winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M03 | `cubic` | **VERIFIED_PASS** | 10 | 10 | 1.9385e-4 | 0.049 | 253× | 3 | 3 | `2-milstein` |
| M04 | `wave` | UNDERCLAIM | 9 | 10 | 3.2478e-7 | 5.7e-4 | 1755× | 3 | 6 | `2-milstein` |
| M05 | `nltest` | UNDERCLAIM | 6 | 10 | 3.3375e-3 | 9.8e-3 | 2.9× | 3 | 6 | `2-milstein` |
| M07 | `stiffquad_det` | **VERIFIED_PASS** | 10 | 10 | 8.5689e-4 | 0.039 | 45.5× | 3 | 3 | `3-trapezoidal-drift-milstein` |
| M08 | `stiffquad_mixed` | UNDERCLAIM | 5 | 10 | 5.1359e-3 | 0.11 | 21.4× | 3 | 6 | `2-milstein` |

**Zero overclaims, zero failures, zero blow-ups, and three of five score themselves below
what they earn.** Across the eight completed entries the suite reads 4 `VERIFIED_PASS`, 4
`UNDERCLAIM`, 1 `RUN_ERROR` (M09) and **no overclaim anywhere** — which is the opposite of
the failure mode §10 predicted to be worth watching, and says the pipeline's self-scoring
is conservative on this suite rather than optimistic. `best_plan_source: state` on all
five, so with M02 that is six consecutive runs in which the conductor reached Phase 3 and
the record says so.

**The fix: no flips, one margin rescued.** Each pool was re-ranked restricted and
unrestricted (`verifylib/ranking.py`, `comparable_keys`). The winner is identical on all
five. `accuracy` is dropped on four of them; M08 is the first sciml pool where all four
components resolve on every plan, so nothing is dropped at all.

| | dropped | restricted index (1st / 2nd / 3rd) | each plan's own index | effect |
| --- | --- | --- | --- | --- |
| M03 | accuracy | 0.6737 / 0.6729 / 0.6579 | identical | inert |
| M04 | accuracy | 0.6870 / 0.6852 / 0.6783 | identical | inert |
| M05 | accuracy, order, consistency | 0.1181 / 0.1002 / 0.0999 | identical | inert |
| M07 | accuracy | 0.7011 / 0.6337 / 0.6336 | 0.7011 / 0.6934 / 0.6930 | **margin 8.8× wider** |
| M08 | — | 0.3827 / 0.3642 / 0.3631 | identical | nothing to drop |

M07 is the one that moves. Restricted, the winner leads by **0.0674**; on each plan's own
index it led by **0.0077**, and second and third place swap. Same answer, but it stops
being a third-decimal call — which is precisely the complaint 10c raised. The corpus
tally over all nine multi-plan sciml pools is now **one winner flipped (M06), one margin
widened 8.8× (M07), seven inert, zero regressions.**

**M05 is the serious finding: `wrong_plan_won` by 622×.** The pipeline shipped
`2-milstein` at **0.3406** of its gate over `3-ito-taylor-1p5` at **5.4733e-4** — a factor
of **622×**, and **44.6× outside** the 7.6153e-3 noise margin. Both passed, so the verdict
costs nothing; but on a *scheme* benchmark this is the one entry of the five where the
schemes genuinely separate, and the ranking is backwards on it.

The mechanism is 10c's defect with a measured consequence attached. On M05 `order_check`
is **`not_run`** and `resolved: false` on all three plans, so `accuracy`, `order` **and**
`consistency` all drop out and `comparable_keys` is left with **`('structure',)` alone**.
The entire selection then rests on `invariant_worst_headroom` — **0.3542 / 0.3005 /
0.2997**, indices 0.1181 / 0.1002 / 0.0999 — one weakly separated diagnostic, pointing
622× in the wrong direction. 10b stopped the index being compared across different
component sets and 10c noted it is read past its own resolution; M05 is what that costs.
It also shows the restriction is not sufficient on its own: `comparable_keys` correctly
narrowed the comparison to the one honest component, and the one honest component was
wrong. **A floor would not have fixed M05 either — it would have abstained**, which is the
right answer here and is the argument for building one.

**The vocabulary gap is not missing taxonomy. It is an incentive to withhold.** M05's
`3-ito-taylor-1p5` ships no `scheme_family`, and says why in a comment above its own
metrics block:

> `scheme_family` is deliberately NOT declared. No SDE family maps to an order-1.5 key in
> the kernel (`FAMILY_ORDER_KEY` has only milstein / tamed-milstein), so any declared
> family other than milstein is judged two-sided at expected_strong_order 0.5 +/- 0.4, and
> `milstein` at 1.0 +/- 0.4. A plan-creator probe of the CRN ladder (dt = 0.004/0.002/0.001,
> 1000 paths, seed 42) measured this scheme's strong order at **1.585**, which fails both
> two-sided bands on a correct solver. Undeclared, the band is one-sided (order >= 0.1).

The reasoning is **correct on the mechanics**: a correct order-1.5 scheme does fail every
band the kernel can offer it, because no key exists for 1.5. So the kernel penalises
declaring the truth and rewards silence, and an agent that reads the kernel carefully
arrives at silence on purpose. That reframes §10a's and 10c's `scheme_family: None`
readings: they are not an oversight in the plan-writer, they are the kernel's own
incentive working as built. On this suite's primary axis the better scheme is the one with
no name attached, and `harness_by_plan` cannot report what it was.

**It is two plans, not one, and the mechanism is subtler than a blank field.** M05's
`1-euler-maruyama` carries a comment making the same argument from the other end — its
error is drift-dominated over the CRN ladder, measuring strong order **0.965** against a
two-sided 0.5 ± 0.4 that a *correct* EM would fail at this ladder's range. Its shipped
`<metrics>` block does read `scheme_family: euler-maruyama`, which looks at first like the
intent not being carried out; it is not. `verifylib/kernel/plan_meta.read` resolves that
plan as **`scheme_family_source: 'inferred'`, `declared: False`** — the family is the
kernel's own inference from the free-text `scheme:` field, and `expected_orders` widens an
*inferred* family to a one-sided band by design (an inference must never make a test
stricter). So **withholding the front-matter field while leaving the free-text `scheme:`
in place buys the one-sided band and leaves the block looking fully declared.** Both plans
got what their comments were after; `verify._plan_scheme_family` reads front-matter only,
which is why the record's column reads `None` for both.

One qualification, measured rather than assumed: on M05 `order_check` never ran at all, so
the band these two plans manoeuvred around had no effect on the run that actually shipped.
The incentive is real, documented and demonstrably acted on; its effect on *this* outcome
is not established.

**Defects 9 and 10 are now the common case, not the exception.** Four of these five carry
an uncapped error-severity ledger violation of the same shape, and across the eight
completed entries it is **seven of eight** — M05 alone is clean:

| | run at | spec asks | `score_cap` | graded |
| --- | --- | --- | --- | --- |
| M01 | 1000 | 50000 | 3 | 10 |
| M02 | 1000 | 20000 | 3 | 10 |
| M03 | 1000 | 20000 | 3 | 10 |
| M04 | 1000 | 50000 | 3 | 10 |
| M06 | 1000 | 100000 | 3 | 10 |
| M07 | 1000 | 50000 | 3 | 10 |
| M08 | 1000 | 50000 | 3 | 10 |

Seven formulators, three distinct wrong path counts, one actual path count, and the cap is
recorded and never applied every single time. Neither half is a slip: the formulator-side
`num_paths` and the unapplied `score_cap` are both systematic, and the grading consequence
is that **no sciml score on record has been capped** when seven of eight should have been.

**One new observation, on M08.** `invariant_worst_headroom` is **negative on all three
plans** — −0.7693 / −0.7726 / −0.7480 — i.e. the declared invariant is violated by every
plan, and the problem still grades 10 with `passed: true` on all three. `structure` scores
0.000 for everyone, so the component also cannot rank. A violated invariant that neither
fails a plan nor orders a pool is doing no work at all; recorded here, with defect 9,
rather than fixed on a measured baseline.

**The five runs broke two tests, and neither failure is in the code they were testing.**
The suite read `2 failed, 1285 passed` afterwards: `test_no_sde_spec_field_is_declared_and_unread_without_a_reason`
on **eleven** new orphan fields, and `test_the_milstein_cap_stays_off_every_live_spec`
on a transcribed `len(...) == 3` that is now 4. Both read `workspace/`, which is
gitignored (`.gitignore:41`, one tracked file: `.gitkeep`), so both pass on a clean
checkout and neither implicates the rev-8 fix. **This is the fifth and sixth instance of
the same anti-pattern** — an exact-equality assertion against a directory every pipeline
run rewrites — and the count is the point: listing the names one at a time has now failed
six times, so the fix this round is structural rather than another transcription.

- **The orphan fields.** All eleven were coined by the five runs' own formulators and
  appear nowhere in `benchmark/problems.py`. Seven are leaves of a block one formulator
  invented on the spot (`verification.pathwise_reference.*`,
  `verification.pathwise_exact_solution.*`), so `VALUE_SUBTREES` now exempts those two
  subtrees with the reason written once, as it already does for
  `verification.structural_facts.`. The remaining four are declared keys in their own
  right and are registered individually: the `pathwise_reference` container,
  `evaluation_thresholds.pathwise_strong_rel_err_max` and its `_definition`, and
  `output_contract.path_values`. All four join `REGENERATED`, because a field coined
  inside a workspace spec cannot have its liveness pinned.
- **They are all the same finding.** `pathwise_strong_rel_err_max` is the clearest yet: a
  formulator put **the right quantity in the right section** — a bound on the relative
  strong error, under `evaluation_thresholds` — and had to invent the key, because none
  exists. That is the seventh spelling of a missing threshold key across seven runs, and
  it is the same root the `pathwise_exact_solution` registry entry names. `path_values` is
  the one exception and is recorded as such: `benchmark/verify.py`, `setup.py` and
  `runner.py` all read it, and the census scans kernel source only, so it is a scoping
  artifact rather than a gap.
- **The milstein count is replaced by the property it stood in for.** The fourth
  `zero_diffusion_derivative` spec is `sde_sciml_additive`, from the M02 probe, and it is
  correctly classified — additive noise *has* a zero diffusion derivative. So the test now
  asserts the partition instead of the tally: every `zero_diffusion_derivative` spec
  declares `noise_structure: additive` and every `levy_area` one declares
  `multiplicative`, which is *why* one is cap-free and the other is not, plus membership
  for the two Levy specs the docstring names. That survives a re-run and fails if a spec
  ever declares a reason its own noise does not support.

~~**Step 10 is now complete on the scheme-matched arm**: nine entries, eight completed, one
(`M09`) a run-error at the old timeout and worth a re-run at 7200 s before the arm is
called done.~~ **Superseded by 10e: M09 completed on 2026-11-13 and step 10 is complete at
nine of nine.**

---

#### 10e. M09, on the third attempt: the conductor recovers its own crash, and a pool gate writes the winner `[rev11 2026-11-13]`

`M09` is the entry that failed twice. It is the suite's only 2-D system, the only one
handed no `dW`, and the only one graded purely on *weak* functionals. Neither failure was
the problem: the first hit the 60-minute timeout mid-evaluation (§10a), and the second
stopped on the **Claude Pro 5-hour usage cap** — the five-hour window had opened at 17:40,
forty minutes *before* the 18:21 launch, so the run never had a full budget. A second
cause is worth recording because it will recur: `pmset -g log` shows
`Entering Sleep state due to 'Clamshell Sleep'` at **18:29:46**, eight minutes in, on
battery. **`caffeinate -ims` does not prevent lid-close sleep**, and `run.py:185` measures
its timeout on a monotonic clock that does not advance while the machine is asleep, so
that attempt reported **124.1 minutes** across a real span of **4 h 18 min**. The timeout
was never the binding constraint in either direction.

**The third attempt cost 312.1 s.** `2026-11-13 00:00:31 → 00:05:44`, `status: completed`,
`timed_out: false`, `--resume` without `--fresh`. The verdict is **`VERIFIED_PASS`**:
winner `4-conditional-mc-rao-blackwell`, **score 10**, `certified: true`, tier A,
provenance `analytic`, worst functional error **9.43e-04 against a 1.6e-02 gate**
(`limited_by: sampling`, bias 4.3e-05 + sampling 8.2e-04). The authoritative record is
`benchmark/results/pipeline/sciml_m09/sciml_probe_m09.json`; the `RUN_ERROR` row for the
same slug in `sciml_probe.json` is the first attempt and is superseded.

**The 5.2 minutes are the finding, not the verdict.** The conductor wrote a
`recovery_note` diagnosing the stale-`STATE.md` shape that produced *both* earlier
failures, and it diagnosed it correctly without being told:

> `STATE.md` had been left at the pre-round-1 snapshot (all four plans iter 0 / score 0 /
> await_solver) while the disk showed round 1 had in fact run: plans 2 and 4 carry complete
> `<metrics>` blocks, both `certified: true` at score 10, and their `solver.py` hashes still
> match the `solver_sha256` the kernel scored, so both scores belong to the files on disk.

It then ran **Phase pre-2** on the one half-cycle — plan 1, whose `solver.py` had run and
whose `evaluation.json` the kernel had written before the evaluator agent was killed
mid-block — and **reused the saved kernel output rather than re-paying ~1500 s of ladder
time**, having first checked that `solver.py` was unmodified so the reuse is not a grading
violation. The branch is not new — `commands/conductor.md:120` has always said Phase pre-2
*"handles: (a) crash recovery mid-cycle"* — but this is the first time in this corpus it has
fired, and it is the answer to the bookkeeping defect §10a flagged: the state file lagging the disk is recoverable *because*
`solver_sha256` is recorded, which is a use for that field nothing had exercised before.

| | plan | scheme family | score | cert | verdict | harness err / gate | solve s |
|---|---|---|---|---|---|---|---|
| 1 | `euler-maruyama` | `euler-maruyama` | 8 | ✗ | `UNDERCLAIM` | 0.5517 | 3.813 |
| 2 | `exact-lognormal-exponential` | `exact-transition` | 10 | ✓ | `VERIFIED_PASS` | **0.0571** | 3.140 |
| 3 | `milstein-symmetric-levy` | `other` | — | — | never implemented | — | — |
| 4 | `conditional-mc-rao-blackwell` | `other` | **10** | ✓ | `VERIFIED_PASS` **(winner)** | 0.0590 | 10.692 |

**A pool gate wrote the winning plan.** The pool originally held three. The spec gate fired
—

> state component 1 is conditionally tractable, and no plan declared
> `exact_sampling_components: 1`, so no plan integrated it out by conditional Monte Carlo /
> Rao-Blackwellisation

— the conductor **re-dispatched the plan-creator once**, and the plan that came back,
`4-conditional-mc-rao-blackwell`, integrates the W1 channel out analytically by propagating
`E[X1²|W2]`, `E[X1X2|W2]`, `E[X2²|W2]` along each sampled W2 path. It won. Two other
workspaces record a re-dispatch (`sde_sabr_cev_call_i`'s formulator,
`sde_sabr_cev_call_ii`'s solver), but this is the only **plan-creator** re-dispatch in the
corpus and so the first entry where a declared gap in the *pool* — not in a plan, and not a
defect in one — produced the plan that was shipped. It is the strongest evidence so far
that the gate is worth its cost.

**`wrong_plan_won: false` with winner ≠ `best_plan_by_harness` — the first such case here.**
On the other eight entries the two either agreed (M03, M04, M07, M08) or disagreed and the
flag fired (M01, M02, M05, M06). Here the harness's best is plan 2 at **0.0571** of gate
against the winner's **0.0590**, a 3.3% gap inside the **0.0617** margin, so the flag
correctly stays down. The reason the ranker preferred plan 4 is a real one the harness
cannot see in one realisation: plan 4's self-reported sampling term is **8.152e-04** against
plan 2's **2.568e-03**, a ~3.2× variance reduction that is exactly what Rao-Blackwellising
a channel buys. **This is the margin doing the job it was added for** — absorbing a
difference between a single measured realisation and a genuinely lower-variance estimator.
Note also that the winner is the pool's **most expensive** plan by 3.4× and has a `cost`
component of **0.0**: `cost` is not in `CERT_COMPONENTS`, so it did not decide, which is
the F7 rule holding under a case that would have inverted on the old stopwatch tiebreak.

**The rev-8 fix is inert here, and inert for a reason worth separating from 10c's.** On
M02 it was inert because every plan was missing the same components. Here
`comparable_keys` returns `('accuracy', 'consistency')` — `order` and `structure` are
`None` on all three graded plans — and the restricted and unrestricted orders are
identical, `0.578 / 0.431 / 0.348`. But `order` is `None` **not because it was not
measured**: all three `SOLUTION.md`s carry one — `0.9514` with `order_check: true` on plan
1, `0.5265` and `0.5104` with `order_check: unresolved` on plans 2 and 4. The record's
per-plan `kernel` block simply does not carry `observed_order`, while it does carry
`design_order`, and `ranking.order()` reads the former. **So the ranking fix's input is
being truncated by the recorder** — the second instance of that shape, after 10d found
`invariant_worst_headroom` missing from M08's `kernel` block the same way, which is why
`audit_plan_claims.py` had to grow `_headrooms` to read `SOLUTION.md`.

**The truncation changes nothing here, and it is worth saying why rather than leaving it
as a worry.** Had `observed_order` reached the record, `ranking.order()` would still have
returned `None` for plans 2 and 4: `unresolved` is `crn.abstention`, `is_skip` catches it,
and ranking a plan on an estimate the kernel declined to gate on is the defect that branch
exists to prevent. `comparable_keys` would drop `order` for exactly the same reason it
drops it now. And plan 1's `0.9514` would not have helped it either, because `order()` is
**one-sided** since rev 8 — overshoot is not a penalty and not a bonus — so against a floor
of 0.5 all three would sit at the component's ceiling and order nothing. The defect is
therefore **latent, not active**: on a pool where every plan's order study resolves, the
component would go missing from the comparison silently, with no symptom in the record to
read it off. That is the version worth fixing, and it is a one-field change to the
recorder.

**Plan 1 is `UNDERCLAIM` for converging too fast, and the evaluator named the gaming move
and declined it.** Its measured order is **0.9514 ± 0.0111** against a declared design
order of **0.5**, and the two-sided band (half-width 0.4, accepted interval 0.1–0.9) fails
**by overshoot**, at 0.451 against 0.4 allowed. The per-component split is the tell: the X1
leg reads **0.5104 ± 0.0186**, textbook Euler–Maruyama, and only the X2 leg reads near 1.
Three things follow.

- **The kernel's reason string is wrong on its face.** It says the order "does not clear
  0.5" for a plan that measured 0.951. The evaluator says so in the file. That is the same
  class of defect as §10a's M01 losers carrying a reason string naming a mean that passed
  46–67× inside tolerance, and it is now two for two on reason strings that misreport
  *which direction* a check failed in.
- **The evaluator refused the move that 10d caught two M05 plans making.** Verbatim: *"I am
  not suggesting dropping `scheme_family` to widen the band: that would hide the finding
  rather than answer it."* 10d established that withholding the front-matter family buys a
  one-sided band; this is the same incentive, correctly identified and declined, which is
  evidence the incentive is legible to the agents rather than an artifact of how 10d read
  M05.
- **It is a declaration-ladder failure, not a solver failure**, and the evaluator says the
  lever belongs to the plan-creator or formulator. M05 was the same ladder failing in the
  opposite direction — a correct order-1.5 scheme with no family key to declare. Both
  entries lose points for the ladder's vocabulary rather than for their numerics.

**Defects 9 and 10 do not apply to M09, and the exception is diagnostic.** Its graded run
used `num_paths: 100000`, matching `evaluation_thresholds.num_paths` *and*
`graded_num_paths`, and no `evaluation.json` in the workspace carries a `score_cap` at all.
So the tally stands at **seven of nine** completed entries, with M05 and M09 the exceptions.
That M09's formulator wrote the full graded path count into its own spec, while seven others
wrote 1000 against specs asking 20000–100000, confirms defect 10 as a **formulator-discretion**
defect rather than a harness constant — there is nothing to fix in the harness's default,
only a missing constraint on what the formulator may write.

**The largest caveat on this run is a circularity the pipeline could not break.** `problem.md`
states *"this problem has no terminal-moment formula"*; the formulator declared
`analytic_moments.has_analytic_solution = true` anyway, arguing the moment hierarchy closes
exactly by linearity, which put the run on **Path A** with provenance `analytic`. Every
`analytic` tag and the whole 9–10 band therefore rests on the formulator's own derivation.
The kernel did report `reference_outcome: validated` on all three plans — against *the
spec's own operator*, and the REPORT says the quiet part: **"a reference derived from a
mis-transcribed equation would validate identically."** And the cross-plan check that would
have broken it was not run, by design:

> `agreement_gate`: not run — Path A does not require `cli.py compare`. `cross_plan_agreement`
> is unavailable on every plan and `agreements.json` does not exist, so no cross-plan
> circularity break was earned.

This is the clearest case in the suite for requiring the agreement gate on Path A **whenever
the closed form is formulator-derived rather than problem-given**, which is a distinction the
spec does not currently carry. Recorded as the open item, not fixed here.

**One pre-registered risk, confirmed with numbers and then accepted.** Before any solver ran,
the conductor logged a RISK that plan 4's gated functionals would be the conditional estimates
rather than the plain sample moments of `terminal_paths` that `implementation_notes`
prescribes, and named plans 1–3 as the fallback. The evaluator confirmed the gap at **4.9 and
9.4 pathwise standard errors** on two functionals — Euler–Maruyama weak `dt` bias, not noise.
The independent harness then graded plan 4 on its own re-run and passed it
(`functional_ok: true`, worst 9.43e-04), so it does not demand exact agreement and the
fallback was not needed. The pipeline predicted its own most likely failure mode, in advance,
in writing, and was wrong about the outcome in the safe direction.

**Two ambiguities carried, and one of them is a non-coinage.** R26 is the Path-A circularity
above. R25 is the non-constancy gate — `problem.md` requires `solve_sde` to be called a second
time at `num_paths=25000` and every reported value to *differ* — for which no
`evaluation_thresholds` key exists. 10d recorded the seventh invented spelling of a missing
threshold key across seven runs; **this formulator did not invent an eighth.** It carried the
requirement three ways (`num_paths`/`graded_num_paths = 100000`, the ungated
`estimator_stability` constraint, and a verbatim restatement in `implementation_notes`),
labelled the gap `not dropped` with the reason, and recorded that the check is enforced only
by the independent harness. That is the behaviour the registry wanted, and it is the first
run to produce it — which is worth knowing, because it means the fix for the missing-key root
cause can be a convention rather than a new schema.

**But the spec coined two fields anyway, and one of them is the registry's sharpest
evidence yet.** The census test failed again after this run — the seventh recurrence of the
anti-pattern 10d tried to end structurally — on `parameter_provenance` and
`analytic_moments.exact_values_at_T`. Neither is a leaf of an exempted subtree, which is why
10d's `VALUE_SUBTREES` fix did not catch them: that fix covers invented *blocks*, and these
are invented *keys*. Both are now in `UNREAD`/`REGENERATED` with their reasons. What they
show:

- **`exact_values_at_T` is `terminal_moments_at_T` re-spelled — by the same problem.** The
  first M09 formulation wrote the exact terminal moments at top level under
  `terminal_moments_at_T`, which the registry has carried since §10a as the fifth instance
  of the missing-key gap. This formulation **dropped that name entirely** — the live spec
  does not contain it anywhere — and wrote the same content inside `analytic_moments` under
  a new one. Seven spellings across seven different problems could be seven formulators'
  taste. **The same problem re-spelling itself cannot be**, and that is a much stronger
  argument for fixing the root cause than the count was.
- **`parameter_provenance` is the field this run's own largest caveat asks for.** It holds
  one line per coefficient saying where it came from in `problem.md`, including the sign
  reasoning on `a22` (`-(-785/512 + sqrt(2)/8) = 785/512 - sqrt(2)/8`, *"the coefficient
  multiplying X2 is +1.3564264297"*). Nothing reads it. Set that against R26 above: the
  whole Path-A provenance rests on the formulator's transcription, and the REPORT's own
  words are that *"a reference derived from a mis-transcribed equation would validate
  identically."* The formulator wrote the audit trail for exactly that risk, unprompted,
  into a field with no consumer. **A consumer for it is a design item, not an exemption** —
  it is the cheapest available partial answer to the circularity, since a transcription a
  reader can check is not a transcription nobody can.

**An eleventh defect, measured and not yet hit.** Plan 1's evaluator reports the CRN ladder
peaking at **4001 MB against the 4096 MB limit** (`ladder_mem_fraction` 0.977; nominal
finest-level increment block 7.68 GB) while the graded run itself used 497 MB. It completed,
but a slightly larger path count or `Nt` would exceed the budget and leave `orders_ok`
unavailable — which caps the plan, as this very entry shows. The solver cannot control it:
it receives `dW` as supplied. Logged against the kernel, not the suite.

**Step 10 is complete on the scheme-matched arm: nine of nine.** Verdicts: **5
`VERIFIED_PASS`, 4 `UNDERCLAIM`, zero overclaims.** Total **139.3 minutes** of pipeline wall
time for the nine completed runs (~2.3 h for the arm), excluding the two aborted M09
attempts. `wrong_plan_won` fired on four of the nine (M01, M02, M05, M06), was correctly
down on four more, and on M09 was correctly down *despite* the ranker and the harness naming
different plans. What remains before the suite is finished is the unrestricted arm and the
optional Phase 2 (step 11).

---

#### 10f. The nine pools re-ranked offline: the pipeline loses to its own baseline, and the floor is withdrawn `[rev12 2026-11-13]`

Step 10 is finished, so the question it was run to answer can be asked: **is the
pipeline better than Opus answering the same problem cold?** The C0 baseline (§9a) ran
every entry at the **same `dt`, the same path count and the same evaluation count** —
one nominal stage on both sides, so the errors compare with no work correction — which
makes this a clean equal-budget comparison on the harness's own error-vs-gate axis.

| | C0 Opus | C2 shipped | C2 best in pool | shipped ÷ C0 | best ÷ C0 |
|---|---|---|---|---|---|
| M01 | 0.02608 | 0.2166 | 0.01223 | **8.31× worse** | 0.47× |
| M02 | 4.525e-05 | 1.115e-04 | 4.524e-05 | **2.46× worse** | 1.00× |
| M03 | 0.003956 | 0.003956 | 0.003956 | 1.00× | 1.00× |
| M04 | 5.698e-04 | 5.698e-04 | 5.698e-04 | 1.00× | 1.00× |
| M05 | 0.3406 | 0.3406 | **5.473e-04** | 1.00× | **0.0016×** |
| M06 | 0.4171 | 0.4955 | 0.09675 | 1.19× worse | 0.23× |
| M07 | 0.02199 | 0.02197 | 0.02197 | 1.00× | 1.00× |
| M08 | 0.04669 | 0.04669 | 0.04669 | 1.00× | 1.00× |
| M09 | 0.06433 | **0.05895** | 0.05708 | **0.92× better** | 0.89× |
| | **$0.96 / 131.7 s** | **139.3 min** | | **geomean 1.412× worse** | **geomean 0.377×** |

**As shipped, the pipeline loses to a $0.11 one-shot** — better on one entry of nine,
tied on five, worse on three, geomean **1.412× worse** for roughly 65× the wall time.
The ties are not coincidence: on `M03` the two errors are **bit-identical**, and `M05`,
`M08` and `M04` agree to 1e-16, 7e-16 and 6e-12. On four entries of nine the whole
apparatus converged on numerics indistinguishable from the one-shot's.

**But the pools are good and the selection gives them away.** The best plan in each pool
is **2.65× better than C0** on geomean (0.377×), beating it on four entries and losing
on none. `M05`'s pool held a plan **625× better than Opus** and shipped one bit-equivalent
to Opus instead. So the deficit is not in what the pipeline writes; it is in which of its
own plans it ships — which is a cheap thing to fix, if the fix is the right one.

#### What was built, and a correction to 10c

`benchmark/rank_study.py`, which re-ranks finished pools from the recorded data and
scores a candidate ranker against the harness. Regret is the ratio the whole question
reduces to:

> `regret = harness_error(the plan the ranker puts first) / harness_error(the best plan in the pool)`

It costs nothing to run — every plan's independently measured error is already in the
results files — so a ranking change is measurable **before** it is shipped. ~~`benchmark/rank_study.py`
was built to fit against P90 regret~~ **Correction (10f): that file did not exist.**
`verifylib/ranking.py:55` says it *"exists to"* fit the weights and
`docs/plan-blind-evaluator.md:468` lists it as **pending**; 10c's phrasing above read as
though it were already there, and the only precedent was
`benchmark/results/benchop-step10/rankcheck.py`, a 42-line one-off that printed whether
the ranker's top agreed with the shipped plan. That answers a *compliance* question, not
a quality one — a ranker can agree with every shipped plan and still choose badly — which
is why regret is the measure here. Seven unit tests cover the module's reasoning
(`benchmark/tests/test_rank_study.py`); the study's own inputs are gitignored, so it is
not a CI test, for the same reason `audit_plan_claims.py` is not.

#### What the re-rank measures

Each variant is scored on all nine pools, with the ranker fed the plans' own `<metrics>`
blocks — guarded by `solver_sha256`, because `workspace/` is gitignored and every run
rewrites it.

| variant | picks the best plan | outside the margin | regret geomean | vs C0 |
|---|---|---|---|---|
| **as shipped** (at run time) | 4/9 | 4 | 3.741 | 1.412× worse |
| **`current`** (with the rev-8 fix) | **5/9** | **3** | **3.120** | **1.177× worse** |
| `abstain` | 5/9 | 3 | 3.120 | 1.177× — **inert** |
| `discriminating` | 5/9 | 3 | 3.120 | 1.177× — **inert** |
| `noscore` *(diagnostic)* | **6/9** | **2** | **2.267** | **0.855× better** |
| `truncated` *(diagnostic)* | 3/9 | 5 | 14.023 | 5.291× worse |
| oracle (the best plan, always) | 9/9 | 0 | 1.000 | 0.377× better |

**The rev-8 fix is confirmed and priced.** It is worth exactly `M06`: re-ranked today
that pool comes out right, which moves the arm from 1.412× to **1.177×** worse than C0.
That is the fix doing what 10b predicted, on a pool that ran before it landed.

**Two of the three candidate fixes are inert, and that retires them.** `abstain` —
dropping `comparable_keys`' fallback to all four keys when the pool shares none — never
fires: given its proper input, every one of the nine pools shares at least one component.
The M02 case 10c blamed on that fallback was not the fallback. `discriminating` — refusing
a component the whole pool scores identically — changes the key set on `M02` (dropping an
`order` that reads 1.000 for all three plans) and **changes no ranking**, because the
component that was already deciding goes on deciding. Both were recommended on the
strength of a mechanism; neither survives being measured.

#### The three remaining failures have three different causes, and only one is the ranker's

| | regret | `decided_by` | shared components | what is actually wrong |
|---|---|---|---|---|
| `M01` | 17.7× | **`score`** | `order`, `structure` | the **index already ranks it correctly** (0.6046 / 0.5972 / 0.5828, best plan first) and `score` overrides it: the evaluator scored Euler–Maruyama **10** and the two better plans **8** |
| `M05` | 622× | `sqi` | **`structure` alone** | `order_check: not_run` on all three (§24's cost skip) and accuracy unavailable, so the one shared component decides — and it is **anti-correlated** with quality here (0.1002 best / 0.1181 winner / 0.0999) |
| `M02` | 2.46× | `sqi` | `order`, `structure` | `order` is 1.000 for all three, so `structure` alone decides, at **0.7460 against 0.7443** — an index gap of **0.0007** against a 2.46× truth gap |

**`M01` is a scoring defect, not a ranking one, and it is the single most valuable fix
available.** The `noscore` diagnostic moves the index above `score` purely to price this:
`M01` flips to the best plan and the arm goes from **1.177× worse than C0 to 0.855×
better** — it crosses the line on that one entry. The variant is *not* a proposal;
`rank_key`'s docstring is right that a validated-analytic 9 must not be traded against a
weighted mean. What it establishes is that **the KS-gate defect §10a diagnosed is now
measured in the only currency that matters**: repairing the score would let the existing
ranker, unchanged, beat the one-shot baseline.

**`M05` is a missing measurement, not a mis-weighted one.** Nothing that re-weights three
unavailable components can fix a pool whose only available one points the wrong way. The
fix is to *acquire* a discriminating measurement — run the order study that §24 skipped
for cost, or earn a cross-plan break with `cli.py compare` — which is the same conclusion
10d reached from the other side and is now quantified at 622×.

#### The resolution floor is withdrawn, and the measurement is why

10c recorded a resolution floor on the index as the fix to build, noting that guessing it
would be the same mistake as asserting the weights. Fitted rather than guessed, **it is
harmful, monotonically**:

| floor | sciml: picks best / regret geomean | holdout: picks best / regret geomean |
|---|---|---|
| 0 (`abstain`) | 5/9 · **3.120** | 7/9 · **1.519** |
| 0.005 | 5/9 · 4.284 | 7/9 · 1.519 |
| 0.03 | 4/9 · 7.912 | 7/9 · 1.519 |
| 0.05 | 4/9 · 9.944 | **8/9 · 1.264** |
| 0.10 | 4/9 · 9.944 | **8/9 · 1.264** |
| 0.20 | 2/9 · 16.097 | **8/9 · 1.051**, 0 outside the margin |
| 0.30 | 1/9 · 41.358 | 7/9 · 1.646 |

**It has the opposite sign on the two corpora, and the reason is a naming convention.**
A floor turns a small index gap into a tie, so the keys *below* the index decide — `iter`,
then declaration order. Measured:

- **sciml: declaration-order plan 1 is the best plan on 0 of 9 pools**, mean rank **2.89 of
  3.** It is `1-euler-maruyama` on every single entry, because the plan-creator
  conventionally puts the baseline cross-check first.
- **holdout (BENCHOP SDE, 9 pools): plan 1 is the best plan on 5 of 9**, mean rank **1.56
  of 3** — `1-exact-chi2-bridge`, `1-exact-cir-conditional-mc`, `1-exact-besq0-poisson-gamma`.
  There the convention puts the exact or conditional-MC plan first.

So the floor's value is set entirely by whether a pool's *naming convention* correlates
with quality, which is a property of neither the ranker nor the problems. **A floor fitted
on one corpus has the opposite effect on the other, so it is not fittable** — and shipping
the 0.05 that the holdout likes would have taken sciml from 3.120 to 9.944. What is missing
below the index is a *fallback better than declaration order*, and until there is one, a
floor converts a noisy decision into a systematically bad one. Withdrawn, with the numbers,
rather than left as a to-do.

The holdout reading is worth keeping for its own sake: the live ranker scores **7/9 and
1.519** on the BENCHOP SDE pools against **5/9 and 3.120** on sciml. The ranker is not
uniformly bad; sciml is harder for it, and the three failures above say why.

#### §10e's truncation defect is active after all — for every reader but the conductor

10e called the recorder dropping `observed_order` **latent, not active**, which is right
for the live ranker: it reads `SOLUTION.md`. The `truncated` variant prices what it costs
anything that re-ranks from the **record** — `run.py`'s own `best_plan_source: fallback`
path, and every offline analysis including the table at the top of this section:

- picks the best plan on **3/9** instead of 5/9, outside the margin on **5/9** instead of 3
- regret geomean **14.023** against 3.120, max **4879×** against 622×
- against C0: **5.291× worse** instead of 1.177×

`M03` and `M08` are ranked *correctly* from the metrics blocks and **wrongly** from the
record, at 42.7× and 8.5×; `M02` degenerates to `decided_by: tie` and takes the first plan
in declaration order, which is the exact failure mode §10a found on M09's first attempt. So
the one-field fix 10e called cosmetic is load-bearing for the record's usefulness as
evidence, and this section's own numbers depend on not using it.

#### What step 1 settles

1. **The pipeline's generation is better than the baseline; its selection is worse.** Pool
   best 0.377× C0, shipped 1.412× C0. The gap is the deliverable.
2. **Fix the scorer before the ranker.** `M01` alone is the difference between losing and
   beating the one-shot, and no ranking change can reach it.
3. **Do not build the floor.** Two corpora, opposite signs, cause identified.
4. **`abstain` and `discriminating` are retired** — measured inert.
5. **Carry `observed_order` into the record.** Not for the ranker; for every reader of it.
6. **`M05` needs a measurement, not a weight.** Run the skipped order study, or earn the
   cross-plan break.

---

## 11. Deliberately not done `[N]`

- **Adaptive-tolerance comparison.** The contract fixes `dt` (_"Use the `dt` you are given"_) `[R]`;
  adding a tolerance-driven entry point changes every suite's problem statements and breaks
  comparability with the measured baselines. Fixed-step column only, stated in every table. This
  also means SciML's adaptive results (its Oval2 headline) are outside the comparison.
- **`oscilreact`, Oval2 and the stochastic heat equation** — §4a.ii: they exceed the solver budget
  by orders of magnitude under the current step rule, and Oval2 additionally costs ~60 constants and
  an undocumented 19th state `[L]`. Dropped rather than deferred, and the loss is named.
- **PDE entries** — out of scope by the owner's decision, as in the BENCHOP-SLV adoption.
- **Stratonovich variants** (§2) — indistinguishable from their Itô counterparts by problem text.
- **Citing any published SciML number.** §1a: there are almost none, four cells error, the prose
  disagrees with the source, and the runs used 128 threads. Everything is rerun locally or not used.
