# 07 — Review of this folder

**Status:** review, 2026-10-28. Written against the proposal in
[README.md](README.md)–[06-retest-protocol.md](06-retest-protocol.md) and checked against the
graded records in `benchmark/results/benchop-step10/` and the code on `main` at 85eead6.

Every number below was recomputed from the records, not read out of
[01-what-was-measured.md](01-what-was-measured.md). Where this review and that document disagree,
the recomputation is shown so it can be checked.

**Summary of the verdict.** The diagnosis is right about *where* the system loses — selection, not
solver generation — and the code tracing in [02-root-causes.md](02-root-causes.md) is accurate; F0,
F8 and F10 are real one-line defects and the line numbers check out. But the quantitative case is
inflated by roughly a factor of two by a statistic that selects on Monte-Carlo luck, three numbers
in the authoritative table are wrong, and the two highest-leverage fixes (F1, F3) rest on a
quantity that does not exist anywhere in the system and cannot be derived from what crosses the
sandbox boundary. F9, billed as "the change that pays for the retest", saves nothing. Half the
evidence the retest protocol calls free is no longer on disk.

---

## §1 Three numbers in the authoritative table are wrong

[01](01-what-was-measured.md) §2 is the table the rest of the folder argues from. Recomputing
`worst_functional_rel_err / functional_tol` over the 100 C0 attempts in `s1.json`–`s4.json`:

| | recomputed best | doc | recomputed median | doc | worst |
|---|---|---|---|---|---|
| B03 | **0.0072** | 0.0178 | **0.0709** | 0.1265 | 0.6549 ✓ |
| B08 | 0.0292 ✓ | 0.0292 | **0.0392** | 0.1430 | 0.2788 ✓ |

The other eight problems reproduce exactly, as do the distinct-answer counts, the `$16.86`, the
48 s mean wall, and `r = +0.041`. All 100 attempts carry `num_turns: 1`, so "a single completion,
no tools" is accurate, and the path budgets are identical between arms on all ten problems
(`num_paths`/`dt` match C0 and C2 exactly) — the two confounders a reader would check first are
genuinely ruled out.

Correcting B03 and B08 moves both headlines **against** the pipeline:

- geometric mean shipped ÷ C0-best: 8.40 → **8.69×**
- geometric mean shipped ÷ C0-median: 2.55 → **2.90×**

Separately, **the 2.18× selection-loss figure does not reproduce from the folder's own §3 table**
under any obvious convention: the ratios listed there give 2.38 (B04 cloud), 2.68 (B04 local) or
**2.43** (both rows, which is what the §3 wording implies). 2.18 is quoted in four places,
including as the baseline for a pre-registered target, so it needs to be re-derived or corrected.

Minor: "all 24 graded plans" is 23 graded plus one crash; the per-plan verdicts include one `FAIL`,
so "every record is UNDERCLAIM" is true of the 15 problem-level records only. Per-problem wall
times are 25–79 min, not 43–87 (that range came from dividing two-problem session totals).

---

## §2 More than half of the measured selection loss is a statistic on noise

This is the most important finding in this review, because it changes what the fixes are worth and
makes one pre-registered target unreachable by honest means.

`ranking.accuracy` already states the governing rule, and the repo implements it at
[`ranking.py:155`](../../verifylib/ranking.py#L155) (`err = max(err, uncertainty(summary))`):

> the error is first **snapped up to its own uncertainty** — a plan cannot claim accuracy finer
> than its own error bar.

The records carry that error bar per plan, as `harness_error_margin`. Apply the same rule to the
`best-in-pool` statistic — floor each plan's measured error at its own margin, then take the pool
minimum:

| | shipped | best in pool | best, floored at its own noise bar | plan |
|---|---|---|---|---|
| B01 | 0.2584 | 0.0259 | **0.2584** | *the shipped plan* |
| B02 | 0.9594 | 0.7592 | 0.7592 | 3-aks-decorrelated-em |
| B03 | 0.4606 | 0.4478 | 0.4478 | 3-qe-exact-variance |
| B04 💻 | 0.1741 | 0.0603 | **0.1668** | 3-qe-exact-variance |
| B05 | 0.1936 | 0.1315 | 0.1315 | 1-euler-maruyama-bessel |
| B06 | 0.5177 | 0.4897 | 0.4897 | 2-qe-martingale-corrected |
| B07 | 0.2952 | — | 0.2952 | (sibling crashed) |
| B08 | 0.3836 | 0.3350 | 0.3350 | 2-andersen-qe |
| B09 | 0.3017 | 0.2968 | 0.2968 | 1-euler-maruyama |
| B10 | 0.4250 | 0.0041 | **0.0573** | 2-qe-martingale-absorbing |

**Geometric mean shipped ÷ best-in-pool: 2.42×. Floored at each plan's own error bar: 1.34×.**

Three consequences:

1. **B01's 9.98× selection loss disappears entirely.** `2-exact-conditional-besq0` measured 0.0259
   against a noise bar of **0.529** gate units — its error is twenty times smaller than its own
   standard error. In expectation that plan is *worse* than the one the conductor shipped. B01 is
   cited in the fix table as evidence for F1 and F3 and listed in F3 as one of five winners the fix
   should change; a correct implementation of F3's own snap-up rule will not change it, and should
   not.
2. **B10's 103.7× is really about 5–7×.** The two plans' prices differ by 8.7e-5 to 1.5e-4
   absolute against a sampling SE of ~3.2e-5, so plan 2 *is* genuinely better — about 5σ, a real
   discretization-bias difference, not luck. But its measured 0.0041 is 0.07 of its own error bar.
   The honest statement is "perfect selection takes B10 from 0.425 to ≈0.06", a 7× gain. The
   headline 104× is the difference between a bias that is real and a draw that was lucky, and the
   folder quotes the lucky draw nine times.
3. **The pre-registered target "geometric mean shipped ÷ best-in-pool < 1.2×" is close to already
   met (1.34×) on the honest statistic, and chasing it on the raw statistic rewards a ranker that
   picks lucky seeds.** `best-in-pool` is a maximum over 2–3 noisy draws; it is biased upward by
   construction and gets more biased as the pool grows — which F9 proposes to do. A retest that
   adds a third plan per problem will see the raw statistic *rise* with no change in the ranker.

The same winner's-curse applies on the C0 side, and [01](01-what-was-measured.md) §2 already says
so by making the median the headline. It should say it on the pipeline side too. The fix is cheap:
**report shipped ÷ best-in-pool with each plan's error floored at its margin, as the primary
number**, and keep the raw one beside it.

How significant is the gap to C0 at all? Comparing each shipped plan against C0's median and
testing the difference against the pipeline plan's own margin: **6 of 10 C0-median wins survive**
(B01, B02, B03, B04, B08, B10). B05, B06, B07 and B09 do not. The qualitative finding holds — the
pipeline is behind, and never ahead — but four of the ten per-problem verdicts are not measurements
yet.

---

## §3 F1 cannot do the job it is given

F1 is billed as "the single most valuable change in this folder" and "the only oracle-free signal
that can separate the pools that are currently tied". Three independent reasons it is not:

**(a) A functional comparison at `rel_tol` cannot disagree on this suite.** `compare_pair`
([`cli.py:420`](../../verifylib/cli.py#L420)) judges a row by `rel_diff` against that row's
tolerance. Every plan on the suite passes the external gate, so any two plans' prices are within
`2 × tol` of each other by construction. B10, the acceptance test: `functional_tol` is **0.01**
(not the 0.003 the folder quotes throughout — the suite's tolerances are 0.003 to 0.015, and only
B06 is 0.003), the two plans' prices differ by 0.2–0.4% relative, and `0.004 + 2·se < 0.01`.
**F1's own acceptance test — "on B10's two plan directories `compare` must return DISAGREE" — is
unsatisfiable as specified.** Same for B04: the required DISAGREE is a 0.03% price difference
against a 0.6% tolerance.

**(b) An agreement verdict is symmetric, so it cannot order anything.** F1's stated payoff for B10
is "a disagreement caps both plans at 8". Both plans already score **2**; a cap at 8 is vacuous.
Nothing in the agree/disagree/inconclusive vocabulary says *which* plan to ship, which is the
decision that was made wrongly. F1 can only ever remove confidence from a pool, never redirect
selection within it.

**(c) The consensus F3 wants to build on it does not exist in a two-plan pool.** Six of the ten
pools held exactly two plans. With two values the midpoint is equidistant from both and carries no
ordering information; with three, "distance from consensus" is majority rule, which rewards two
same-family plans agreeing with each other over one better plan of a different family — the
opposite of what M4 says the suite needs.

**(d) It would make certification pool-dependent, which the repo forbids in writing.** F3 proposes
that "where F1 has recorded a cross-plan consensus, the distance from consensus … should take
precedence". `accuracy` is in `CERT_COMPONENTS`, and
[`sqi`'s docstring](../../verifylib/ranking.py#L395) rules this out explicitly:

> substituting a pool median makes one plan's index depend on its siblings', which nothing a
> certification reads may do.

`cost` is excluded from `sqi_cert` for exactly this reason. A consensus-based accuracy term would
reintroduce the property that exclusion exists to prevent.

---

## §4 F3's load-bearing ingredient does not exist

F3 defines the new accuracy error as "the larger of (a) the self-convergence estimate F2 produces
and (b) **the Monte-Carlo standard error of that functional** at the graded path count". Neither
term is available:

**(b) has no mechanism anywhere in the system.** `uncertainty()` reads `mc_se_rel`, which
[`sde/driver.py:133`](../../verifylib/kernel/sde/driver.py#L133) computes from `terminal_paths` —
it is the standard error of the *moments*. A price is a scalar; the sandbox carries one float per
functional and nothing else, so its standard error cannot be recovered on the kernel side at all.
**This is a solver-contract change, not a ranking change**: the solver must return per-functional
standard errors, or the per-path payoff samples they are computed from. That change is upstream of
F0 and is the real prerequisite for the whole selection program. It is not in the fix list.

Related and worth fixing with it: `functionals` appears nowhere in
[`agents/solver-sde.md`](../../agents/solver-sde.md), [`agents/evaluator-sde.md`](../../agents/evaluator-sde.md)
or [`references/sde_manual.md`](../../references/sde_manual.md). The BENCHOP solvers return prices
only because each `problem_spec.json`'s `output_contract` tells them to. The generic SDE contract
has no notion of a quantity of interest.

**(a) is measured at the wrong path count and on the wrong axis.** The ladder F2 would carry
functionals up is the CRN ladder, which runs at `conv_num_paths` = **20 000** paths
(`problem_spec.json`, all four surviving specs) against graded counts of 400 000 to 4 000 000 — a
factor of 20 to 200 in paths, so √(20–200) ≈ 4.5–14× coarser sampling. Because the ladder shares
Brownian paths across levels, the level-to-level difference is bias-dominated and the sampling
error largely cancels, so `functionals.estimate_error` would return a reasonable estimate of the
**discretization** part of the functional error and would be structurally blind to the
**sampling** part — which is the dominant term at the graded count (B10 plan 2's entire error bar
is sampling). An accuracy component built on it ranks plans by dt-bias alone and ignores variance,
which is precisely the axis C0 won B08 on with a control variate.

There is also a type error in reusing `functionals.estimate_error` as written: it infers an order
from `log2(|d10|/|d21|)` assuming a refinement ratio of 2 on **one** axis. The folder's own
cautionary measurement (12 500 paths at dt=0.02 beating 25 000 at dt=0.01) is a two-axis ladder,
where bias falls like dt and noise like n^(−1/2); an order inferred across that mixture is not a
number about anything. F2 correctly flags that a decreasing test will fail correct solvers and
proposes the harness's symmetric band — but a band yields pass/fail, and F3 then needs a
*continuous* quantity from the same estimator. The two fixes want incompatible things from one
function.

**What would actually work**, and is implementable with one contract field: estimate the functional
error as the sum of (i) a **dt-Richardson term at the graded path count under common random
numbers** — solve at `dt` and `dt/2` with the same `dW` block, which the solver contract already
supports via the `dW` hook, so the sampling error cancels and the difference is the bias — and (ii)
the **reported per-functional standard error**. That is oracle-free, it resolves differences below
the gate (B10's two plans are 5σ apart in price), it is measured where the problem is graded, and
it prices both error sources. It costs one extra half-step solve per plan. It should be validated
on B01–B04, whose plan directories still exist, before anything is built on it.

---

## §5 F4 points at the wrong cost term

F4's change and acceptance test target [`ranking.cost`](../../verifylib/ranking.py#L369), the
weight-0.10 component of `sqi`. But `solve_time_s` enters the decision **twice**, and the second
entry is the one that fires on an exact tie:
[`rank_key`](../../verifylib/ranking.py#L437) is

```python
(0, -score, provenance_rank, -sqi, solve_time_s, iter)
```

— a **lexicographic fifth key**. On B10, where the folder's own account is that accuracy tied, the
quality index is identical and key 5 decides outright; nothing done to the weight-0.10 component
changes that. F4's acceptance test ("the winner must not change if both plans' `solve_time_s` are
swapped") will still fail after F4 as described. Both paths have to be addressed, and they want
different treatments: a weighted component is a tiebreak, a lexicographic key is a decision rule.

Two further problems with `solve_time_s` as a cost signal on the SDE path, neither mentioned:

- **It is the sandbox child's wall time.** [`sde/driver.py:93`](../../verifylib/kernel/sde/driver.py#L93)
  sets `solve_time_s = got["wall_s"]`, although `ranking.cost`'s docstring says it reads the solve
  time and "never `wall_time_s`". The PDE and SDE paths mean different things by the same field.
- **The pipeline runs up to three plan cycles concurrently** (`conductor.md:26`,
  `parallelism = 3`). Plans in one pool therefore contend for CPU with each other, so their wall
  times are mutually contaminated and partly a scheduling artifact. [06](06-retest-protocol.md) §3
  argues the log-scale pool-relative form cancels a uniform slowdown — true, but concurrency is not
  uniform. Three of ten winners were decided on this number.
- And once F8 lands, a solver that forks workers (which is exactly what B07's plan was doing when
  it crashed) reports a wall time divided by its worker count. Fixing F8 without pinning the
  process model turns `cost` into a reward for parallelism. Either record CPU time, or require
  single-process solvers and say so in the contract.

---

## §6 F9 saves nothing

F9 is "the change that pays for the retest": cut `max_iter` 5 → 2 and claim "~Half the pipeline
cost, no measured loss", with [06](06-retest-protocol.md) §7 budgeting 20–45 min against today's
43–87.

The folder's own evidence refutes this. The plateau rule stops a plan at `iter >= 2` when its score
did not improve, and every plan on the suite stopped at exactly 2 — confirmed in the records:
`iter == 2` for all 24 plans, `total_iters == 2 × n_plans` in all 15. **`max_iter = 5` therefore
cost exactly what `max_iter = 2` would have cost.** The saving is zero. What `max_iter = 2` does
change is the counterfactual the plateau rule exists to permit: a plan whose score *does* improve
on cycle 2 would now be cut off.

The real content of F9 — more plans per problem — is a change to `n_plans`, and it is a cost
*increase*: measured per-problem wall time tracks plan count (the three 3-plan runs are the long
ones; B08 at 3 plans took 79 min against B09 at 2 plans in 28). Propose it on its own terms, with
the §2 caveat that a larger pool inflates the `best-in-pool` statistic the retest is graded on.

---

## §7 F6's evidence does not support F6

M5 and F6 are a good catch as *vocabulary*: `FAMILY_ORDER_KEY`
([`crn.py:221`](../../verifylib/kernel/sde/crn.py#L221)) does map only the two Milstein families,
everything else does fall through to `expected_strong_order`, and `expected_orders` does keep the
band two-sided when the family was declared — so `other` is genuinely a worse declaration than
silence. The `FAMILY_ORDER_REQUIRED` parity argument in [`schema.py:487`](../../verifylib/schema.py#L487)
is the right precedent and the fix is correctly specified.

But the measured case for it is misattributed. The folder cites B10's QE plan — "104× more
accurate and scored 2" — as the cost of declaring `other`. The record says that plan scored 2 with
`provenance: none`, `d1_outcome: unavailable`, `reference_outcome: unavailable`,
`dynkin_z: unavailable`, and `reason: "no test in the manual could be run"`. Its *sibling*, a plain
Euler plan, scored 2 for the same reason. Nothing on B10 was lost to the order band, and no plan on
the suite is shown losing a point to it. B07's QE plan is the one scored down on an order-study
outcome, and its recorded reason is "the estimator is erratic in dt", which is not a family-floor
miss either.

F6 is probably still right — it is cheap, it is a real defect, and M5's reasoning is sound — but it
should be presented as a defect found by reading the code, with the honest note that this suite
does not measure its cost. As written it borrows B10's 104× for a mechanism that did not fire there.

---

## §8 F7's evidence is confounded and its fix contradicts its own test

"`1-` is Euler in all ten pools and won seven" cannot be read as a positional bias, because plan
position and scheme family are **perfectly collinear on this suite**: plan 1 is always Euler, and
Euler is also always the fast plan, which is the signal M2 says decided those winners. Position,
family and cost are one variable here with n = 10. The folder spends the same ten observations
three times (M2, M4, F7) as if they were independent lines of evidence.

The proposed fix is also internally inconsistent: "a hash of the plan slug, or alphabetical" is
still a fixed function of the plan's identity, and the acceptance test asks that "the winner must
not change with the plan ids permuted" — a slug hash fails that test by construction. If the intent
is that a rank decided by a tiebreak is not a rank, the honest implementation is to **report the
tie** (`decided_by: tiebreak` in the record, which F10 should carry anyway) and let the conductor
spend one measurement breaking it, rather than to randomize invisibly.

---

## §9 Half the evidence the retest protocol calls free is gone

[06](06-retest-protocol.md) §1 is the right instinct — do the offline work before spending credits
— and its central factual claim does not hold:

> The ten pools' `<metrics>` blocks are on disk in `workspace/`.

They are not. `workspace/` holds plan directories for **B01–B04 only**; B05–B10
(`sde_qlsv_dnt`, `sde_hhw_call_corr`, `sde_hhw_call_zerocorr`, `sde_heston_call_hivar`,
`sde_heston_dnt_hivar`, `sde_sabr_cev_call_ii_lowvol`) contain `problem.md` alone — no plans, no
`SOLUTION.md`, no `solver.py`. Those six ran in cloud sessions and the artefacts did not come back.
Consequences:

- **F1's acceptance tests cannot be run.** B10 and B09 have no plan directories to compare.
- **F8's "free data point" cannot be collected.** B07's `2-qe-log-euler` has a recorded
  `solver_sha256` and no file; `--skip-run` parses STATE and re-verifies, which means re-solving,
  which needs the solver. There is nothing to re-grade.
- **The offline re-rank fixture is faithful for four problems, not ten.** The graded JSONs carry a
  reduced per-plan `kernel` block (score, provenance, `estimated_rel_error`, `d1_outcome`,
  `reference_outcome`, `invariant_max_drift`, `dynkin_z`, `solve_time_s`) with no order or
  structure fields and no `config`. `sqi` renormalises over whatever is measurable, so a fixture
  built from those blocks computes a *different* index from the one the live run computed, and
  "today's ranked order" recorded from it would not be today's ranked order.
- **F11 is fine.** B01's two plan directories survive, so the metrics diff is genuinely free.
- `--skip-run` is free of credits, not of compute: verification re-solves at 0.4–4M paths plus the
  two-level check, which is minutes per plan.

This should be the first line of [06](06-retest-protocol.md) §8, ahead of the backup step: decide
now whether B05–B10 are re-run to recover their artefacts, or whether every claim about them stays
frozen at what the JSON records.

---

## §10 What will not generalize beyond BENCHOP

**The whole selection program is scoped to specs that declare gated functionals, and almost nothing
does.** Of the 38 `problem_spec.json` files on disk, **6 declare `functionals`** — four SDE
(the BENCHOP problems that survive locally) and two PDE (`pde_gross_pitaevskii_1d`,
`pde_schrodinger_eigen_2d`). F3 is explicit that the moment path stays untouched where no
functional is declared, which is correct and conservative, and which also means **F1 and F3 do
nothing at all on roughly 85% of the main suite**. They are BENCHOP-shaped fixes. That is a
defensible choice, but the folder presents them as fixing "the ranker", and the retest's
pre-registered targets are all measured on the suite where they fire. Nothing in the plan predicts
or measures the effect on the 49-problem main suite, where the same ranker still decides every
winner on moments.

**`harness_error` is not comparable across problems the way the geometric means assume.** It is
relative error ÷ that problem's gate, and the gates on this suite span 0.003 to 0.015 — a 5×
spread, authored by this project. A plan on B06 is being judged five times harder than one on B03
or B08, and every geometric mean in the folder is a function of those authoring choices. Within a
pool the normalization is harmless; across problems it is a choice that should be stated. (It also
makes the "104×" and "8.4×" headlines sensitive to a gate nobody would defend to two significant
figures.)

**The multilevel / variance-reduction axis F5 proposes is blocked by the solver contract, not by
the planner's menu.** The harness calls `solve_sde(num_paths, dt, T, seed)` and grades
`terminal_paths` moments plus `functionals`. MLMC has no single path population at one `dt`, so
`terminal_paths` is ill-defined for it, and the two-level anti-hardcoding check
(`output_contract.called_with`: the same solver at two path counts, values required to differ)
assumes a plain estimator. A control variate does fit the contract — but then the kernel cannot see
what it bought: the variate changes the *functional*, not the terminal distribution, so
`empirical_mean`, `empirical_variance`, `mc_se_rel` and `estimated_rel_error` are all unchanged.
**A control-variate plan is invisible to every signal the ranker reads**, which is a sharper version
of M1 than M1 states, and it is the scheme family C0 beat the pipeline with on B08. Widening the
planner's menu to variance reduction without extending the contract and the kernel produces plans
the system cannot grade.

**F5's "the planner improvised past the rule, so the fix is to legitimise it" reasoning is
BENCHOP-specific in one way worth naming.** On a suite where every problem is a known
mathematical-finance SDE with a published exact-transition law, "propose the exact transition" is
sound advice. The corresponding instruction in the manual will apply to every multi-dimensional SDE
the system ever sees, most of which have no such law, and a planner that proposes exact-transition
sampling where none exists produces a plan the solver cannot write. The rewritten Step 1b needs a
guard keyed on a field the formulator actually sets — `analytic_moments`, or a new
`transition_law_known` — not on the planner's recall of the finance literature.

---

## §11 What is missing

**1. Seed replication. This is the cheapest high-value measurement available and it is not in the
plan.** Every conclusion in the folder rests on one seed (42) per plan. §2 shows what that costs:
two of the three headline selection failures are substantially noise. Re-solving the *shipped* and
*best-in-pool* plans at two more independent seeds costs minutes of compute, no credits, and no
pipeline call, and it converts "the pool contained a plan 104× better" into a confidence interval.
Do it on B01–B04, which still have solvers, before implementing anything.

**2. An equal-compute baseline.** C0 costs $0.17 and 48 s; the pipeline costs 25–79 min and, at
Opus, $4.88 for one problem. The pre-registered parity criterion compares one pipeline run against
one completion, which is the right *single-draw* comparison but not the comparison a user faces.
At the pipeline's budget a user could run 50–100 C0 attempts and select among them with the same
`rank_key` the folder already uses for the `kernel` cut — which is exactly the "matched attempts,
matched selection rule, no oracle" arm [06](06-retest-protocol.md) §4 defines, just at the right N.
Report that cut at matched spend. It is the bar that decides whether the pipeline is worth running,
and on this evidence it is a much harder bar than parity-with-one-completion.

**3. A guard on implausible error estimates.** B10's two plans reported
`estimated_rel_error` of 2.22e-14 and 2.34e-14 with `provenance: none` and no reference of any
kind. A Monte-Carlo estimate at 10^6 paths cannot be accurate to 1e-14; the number is a
non-measurement. The repo already has the concept — `baseline_select.looks_n_independent`, and
`headroom_decades`' own rule that an error of exactly zero comes back `None` because it is "the
signature of a solver whose output did not respond to the grid at all". Nothing applies it on the
SDE path. A cheap, general fix: an `estimated_rel_error` below the plan's own `mc_se_rel` is
unmeasured, not perfect. This generalizes well beyond BENCHOP and would have stopped two of the
three cost-decided ties from being ties.

**4. Power.** The central evidence for M1 is "8 of 15 within-pool pairs ordered correctly, a coin
flip gives 7.5". With n = 15 the 95% interval on 8/15 runs roughly 0.27–0.78: that observation
cannot distinguish chance from a usefully-informative signal. The cross-problem `r = +0.041`
reproduces exactly but is confounded by problem difficulty and by 8 of 9 pools being score-tied —
it mostly measures that the kernel score does not track problem difficulty, which it is not
designed to do. M1's conclusion is probably right; the folder should say it is underpowered rather
than quote a coin flip as a finding.

**5. Nothing changes the evaluator.** M6 establishes that the refine loop learns nothing, and the
plan's answer is that F1–F3 "should improve the loop as a side effect, by giving the evaluator a
signal that moves". No fix touches `evaluator-sde.md`, and no acceptance test checks whether the
loop then improves. If the second cycle is genuinely worthless the loop is ~50% of pipeline cost
for nothing, and that deserves its own measurement rather than a hoped-for side effect.

**6. B02 is not diagnosed.** At 0.9594 it is within 4% of failing the external gate and it is the
only such plan on the suite. It is dismissed as "an implementation-quality gap" and left out of the
retest subset. Whatever is wrong there is the closest thing on this suite to an actual failure.

---

## §12 Revised order of work

Reordered from [06](06-retest-protocol.md) §8 by what this review found. Steps 1–4 cost no credits
and two of them can disconfirm the plan.

1. **Back up `benchmark/results/benchop-step10/` outside the repo.** Unchanged, and still first.
2. **Decide the B05–B10 artefact question** (§9). Everything downstream depends on whether those
   six pools can be touched again.
3. **Correct §1's three numbers and re-derive 2.18×.** The targets in §6 of the retest protocol are
   stated against it.
4. **Re-state the selection loss with each plan floored at its own error bar** (§2), and
   **replicate B01–B04's shipped and best plans at two more seeds**. If the deflated loss is 1.34×
   and stays there under replication, F1–F4 are worth much less than the folder claims and the
   retest's primary target needs rewriting before anything is built.
5. **F11** — diff B01's two metrics blocks. Free, and the one place the primary ranking key fired
   backwards.
6. **F0**, then the **solver-contract extension**: per-functional standard errors (or per-path
   payoff samples) returned by `solve_sde`, declared in `agents/solver-sde.md` and
   `references/sde_manual.md`, not only in each problem's `output_contract` (§4). Without this F3
   has no error bar to snap to and F1 has no interval.
7. **The functional error estimator**: dt-Richardson at the graded path count under common random
   numbers, plus the reported standard error (§4). Validate on B01–B04 against the known harness
   errors — that is a real test of whether any oracle-free signal can order these plans — *before*
   wiring it into `accuracy`.
8. **F3**, scoped as the folder already scopes it, with the consensus clause dropped (§3d).
9. **F4 on both cost paths** — the `sqi` component and `rank_key`'s fifth key — plus a decision on
   CPU-time-vs-wall-time and the process model (§5).
10. **F10**, extended to carry `decided_by` and the per-plan margin. The formatter reads
    `harness["rel_l2_err"]`, which does not exist on a functional-graded problem; `harness_error`
    is already in the record one level up. One-line fix, and it is what made this review expensive.
11. **F8**, with the process-model decision from step 9 settled first. Note that the quoted fix —
    "register the module in `sys.modules` … and remove it in a `finally`" — reinstates the bug if
    the `finally` runs where the quoted `sys.path` block's does: pickling happens during the solve,
    long after `exec_module` returns. The sibling at
    [`sandbox.py:124`](../../verifylib/kernel/sandbox.py#L124) leaves the registration in place,
    which is why it works.
12. **F5 and F6**, with F6 presented as a code defect rather than as B10's 104× (§7), and F5's
   Step 1b keyed on a spec field rather than on the planner's knowledge of which SDEs have closed
   transition laws (§10).
13. **F7** as a reported tie rather than a hidden randomization (§8).
14. **Drop F9's `max_iter` change** (§6). Propose the plan-count increase on its own, and note it
    inflates the `best-in-pool` statistic.
15. Then the Sonnet C0 baseline — on all ten, as [06](06-retest-protocol.md) §4 rightly insists,
    plus the matched-spend cut from §11.2 — and only then the C2 subset.

---

## §13 What the folder gets right

Recorded because this review is mostly objections and the balance matters.

- **The central diagnosis.** Selection, not solver generation, is where the suite is lost. Every
  check in this review agrees, including the noise-deflated one: the shipped plan was the pool's
  most accurate on **1 of 10** problems.
- **The code tracing in [02](02-root-causes.md) is accurate.** Every line number, docstring quote
  and mechanism checked out: `accuracy` does read `estimated_rel_error`; `_tol_of` does ignore
  `rel_tol`; `functionals.run_declared` is called only from `evaluate_pde`; `grep -rn functional
  verifylib/kernel/sde/*.py` is empty; the scheme rule does admit Euler alone on every problem in
  the suite; `FAMILY_ORDER_KEY` does map only Milstein. The §1-correction about having earlier
  missed `kernel/functionals.py` is the right instinct applied in the right direction.
- **F0 is real** and is correctly identified as blocking everything else.
- **F8's cause is correctly diagnosed** down to the missing line and the sibling that has it.
- **F10 is real**, and its stated motive — that this study had to rebuild every table by hand — is
  the correct reason to prioritise a formatter fix.
- **[06](06-retest-protocol.md) §2's warning** that an all-Sonnet retest downgrades the planner and
  that the fixes will take the blame is the single most useful paragraph in the folder, and it is
  the kind of thing that only gets written by someone looking for ways to be wrong.
- **[06](06-retest-protocol.md) §3 and §4** — hold the gates fixed, do not re-cut a tolerance after
  seeing a result, keep the Opus column as a second yardstick, use one `rank_key` for both arms and
  state that F3 moves the baseline's `kernel` cut too — are all correct and all easy to get wrong.
- **The folder says what it does not claim**, including that perfect selection still loses four
  problems. That framing survives this review intact; the magnitudes do not.
