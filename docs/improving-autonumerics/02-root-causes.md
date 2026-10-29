# 02 — Root causes

**Status:** analysis, 2026-10-28; **revised the same day against [07-review.md](07-review.md)**,
which confirmed every line reference and docstring quote here and corrected two pieces of the
evidence. Six mechanisms. Each names the line of code that implements it
and the measurement in [01-what-was-measured.md](01-what-was-measured.md) that proves it fires on
this suite.

---

## M1 — The ranker ranks on a quantity that is not the one being graded

### What the quantity is

The problems declare what they are scored on. From `workspace/sde_hhw_call_corr/problem_spec.json`:

```json
"functionals": {
  "price_S75":  {"rel_tol": 0.003, "gate": true, "S0": 75.0,
                 "definition": "E[exp(-integral_0^T R dt) * max(S(T) - K, 0)] with S(0) = 75"},
  "price_S100": {"rel_tol": 0.003, "gate": true, "S0": 100.0, "...": "..."},
  "price_S125": {"rel_tol": 0.003, "gate": true, "S0": 125.0, "...": "..."},
  "note": "The quantity of interest is the whole external grade: each price within
           0.3% relative of an independently held reference"
}
```

and the solver contract requires them back: `"functionals": "dict with float entries price_S75,
price_S100, price_S125"`. The harness grades on exactly these, via
[`_functional_errs`](../../benchmark/verify.py#L1078).

### What the ranker reads instead

[`ranking.accuracy`](../../verifylib/ranking.py#L99) — 0.35 of the quality index, its largest
single weight — reads `summary["estimated_rel_error"]`. On an SDE that field is *the worst of the
mean and variance errors* and nothing else;
[`verifylib/kernel/metrics.py:210`](../../verifylib/kernel/metrics.py#L210) says so in as many
words: *"SDE: `estimated_rel_error` is the worst over mean and variance and does not say which."*
It is set at [`verifylib/kernel/sde/driver.py:194`](../../verifylib/kernel/sde/driver.py#L194) and
`:301`.

[`_tol_of`](../../verifylib/ranking.py#L463) picks the tolerance that error is judged against, and
it looks for `rel_l2_err_max` then `variance_rel_err_max`, falling back to `0.1`. A functional's
`rel_tol` of `0.003` is never consulted.

### Where the functional check does exist, and where it does not

**Correction to an earlier claim in this project's analysis:** the kernel *does* have a functional
check. [`verifylib/kernel/functionals.py`](../../verifylib/kernel/functionals.py) reads
`evaluation_thresholds.functionals` and asks, by self-convergence up the ladder, whether each
declared quantity has converged to its `rel_tol`; a gated one that has not makes the run
not-converged. An earlier grep that reported "the kernel reads it nowhere" had searched
`verifylib/*.py` only and missed the `kernel/` subdirectory.

The real situation is narrower and more actionable:

- `functionals.run_declared` is called from **one place**:
  [`driver.py:245`](../../verifylib/kernel/driver.py#L245), inside `evaluate_pde`.
- `grep -rn functional verifylib/kernel/sde/*.py` returns **nothing**. The SDE path has no
  functional check at all.
- Even on the PDE path the check is a **gate**, not a ranking term. It can stop a plan certifying;
  it never feeds `sqi`, so it cannot order two plans that both pass or both fail it.

So the machinery exists, is written, is tested, and is wired to the half of the system that was not
being graded on functionals.

### The measurement

Across the 23 graded plans, the correlation between the kernel's score and the harness error is
**r = +0.041** — zero, and if anything the wrong sign.

Sharper, because it is the comparison the ranker actually makes — within a single pool, does the
plan with the smaller moment error also have the smaller price error?

> **8 of 15 within-pool pairs.** A coin flip gives 7.5.

The seven inversions: B02 (1 vs 2), B03 (1 vs 3, 2 vs 3), B05, B06, B09, B10. B10 is the extreme —
moment errors of 2.22e-14 and 2.34e-14, tied at round-off, against prices of 0.4250 and 0.0041.

**[rev] This statistic is underpowered and should not be quoted as a finding on its own.** The
Wilson 95% interval on 8/15 runs **0.30 to 0.75**: the observation cannot separate chance from a
usefully informative signal. It establishes that the moment error is not a *strong* predictor of
the price error, which is what M1 needs, but the argument for M1 is the mechanism below — the
ranker reads a quantity the problem is not scored on, and on the SDE path the quantity it *is*
scored on never crosses the sandbox boundary at all (see [F0](03-fix-selection.md)). The
cross-problem `r = +0.041` is weaker still: with the kernel score tied inside 8 of 9 pools it
largely measures that the score does not track problem difficulty, which it is not designed to do.

**A third signal, which is not underpowered:** B10's two plans report `estimated_rel_error` of
2.22e-14 and 2.34e-14 with `provenance: none` and no reference of any kind. A Monte-Carlo estimate
at 10⁶ paths cannot be accurate to 1e-14. That is not a small error, it is a non-measurement being
fed to the component that carries 0.35 of the index — see
[F13](05-fix-loop-and-harness.md#f13).

**Fix: [F1, F2, F3](03-fix-selection.md).**

---

## M2 — When the accuracy term ties, cost decides, and cost is biased against the schemes that win

[`ranking.cost`](../../verifylib/ranking.py#L369) is pool-relative and log-scaled:

```python
return _clip01(math.log10(times[-1] / mine) / math.log10(times[-1] / times[0]))
```

**[rev]** And it is only *one* of two places `solve_time_s` enters the decision. The other is
[`rank_key`](../../verifylib/ranking.py#L437), where it is the **fifth lexicographic key**:

```python
(0, -score, provenance_rank, -sqi, solve_time_s, iter)
```

On an exact tie in score, provenance and index — which is B10 — key 5 decides outright and the
weighted component is irrelevant. Any fix has to address both, and they want different treatments:
a weighted component is a tiebreak, a lexicographic key is a decision rule.

Two further properties of the field itself, neither of which the design intends:

- **On the SDE path it is wall time, not solve time.**
  [`sde/driver.py:93`](../../verifylib/kernel/sde/driver.py#L93) sets
  `solve_time_s = round(float(got["wall_s"]), 3)` — the sandbox child's wall clock — although
  `cost`'s own docstring says it reads the solve time and *"never `wall_time_s`"*. The PDE and SDE
  paths mean different things by one field.
- **Plans in a pool contend for CPU with each other.** `parallelism = 3`
  ([`conductor.md:26`](../../commands/conductor.md#L26)), so up to three plan cycles run
  concurrently and their wall times are partly a scheduling artefact.
  [06](06-retest-protocol.md) §3's argument that the log-scale pool-relative form cancels a uniform
  slowdown is correct, but concurrency is not uniform. Three of ten winners were decided on this
  number.

The weight itself, 0.10, is correct as a tiebreak and is justified in the module docstring by
exactly the right sentence: *"when two methods are equally accurate within their own error bars,
take the cheaper one."* The premise fails here. The plans are not equally accurate — they differ by
by a resolvable margin on the graded price — 3.4σ to 4.5σ on B10 — they merely *look* equally
accurate to a term that cannot see the price. Cost is then no longer breaking a tie; it is making the decision.

Three of the ten shipped plans were chosen this way:

| | what tied | what decided | cost of the decision |
|---|---|---|---|
| **B05** | `est_err` 1.70e-02 vs 1.70e-02, score 7 vs 7, provenance equal, drift 0.1857 vs 0.1846 | `solve_time_s` **61.4 < 77.4** | shipped plan 1.47× less accurate |
| **B08** | score 9 vs 9, provenance analytic vs analytic; QE was *better* on `est_err` (2.36e-3 vs 2.53e-3) | `solve_time_s` **80.5 < 217.5** at weight 0.10 outweighing a 7% accuracy edge at weight 0.35 | shipped plan 1.15× less accurate |
| **B10** | score 2 vs 2, provenance none vs none, `est_err` tied at round-off *(both ≈2e-14 — a non-measurement, see [F13](05-fix-loop-and-harness.md#f13))* | `solve_time_s` **171.3 < 198.8**, via `rank_key`'s 5th key, not the weighted term | shipped plan 103.7× less accurate as measured, **≈7× after flooring at the plans' own error bars** — a real 3.4–4.5σ difference |

And the bias is systematic, not incidental. In the six pools containing both an Euler plan and an
exact/QE plan, the advanced plan is the slower one in **five**: 93.5 vs 28.7 (B01), 82.7 vs 20.9
(B03), 64.9 vs 57.9 (B04), 217.5 vs 80.5 (B08), 198.8 vs 171.3 (B10). B07 is the exception and its
faster, more accurate plan crashed in verification. Exact transition sampling and QE buy accuracy
with arithmetic per step — that is what they are *for* — so a cost term that fires whenever
accuracy ties will systematically select against precisely the schemes that would have closed the
gap to C0.

**Fix: [F4](03-fix-selection.md#f4).**

---

## M3 — The first two ranking keys are inert on this suite

The Phase-3 key is `score → provenance → sqi → solve_time_s → iterations`
([`ranking.rank_key`](../../verifylib/ranking.py#L437)). Keys 1 and 2 are deliberately
lexicographic and deliberately first, because they are claims about *what was established* and must
not be averaged against quality.

On BENCHOP-SLV they separate nothing:

- **Kernel score is identical across the pool in 8 of 9 pools.** (B01 is the exception.)
- **Provenance is identical across the pool in 9 of 9.**

So on this suite the architecture's two most carefully reasoned keys are no-ops, and 100% of the
decision load falls on `sqi` and `solve_time_s` — the two keys M1 and M2 show to be measuring the
wrong things. The design is sound and the suite is outside the regime it was designed for: these
problems are all Tier 3, all multi-dimensional, all scored on a functional, and the kernel's
evidence ladder cannot tell the plans apart.

B01 is worth its own line because it is the one case where key 1 *did* fire, and it fired
**backwards**: `2-exact-conditional-besq0` has 5.2× better moments and 9.98× better price than the
winner, and scored **8 against the winner's 9**. Both are `provenance: analytic`, both
`reference_outcome: validated`, and their invariant drifts are comparable (6.69e-4 vs 4.42e-4). No
field in the graded record explains the missing point. **This is an open question, not a diagnosis**
— see [F11](05-fix-loop-and-harness.md#f11) for the command that would answer it.

---

## M4 — The scheme menu excludes the schemes that win

[`agents/plan-creator-sde.md:21-29`](../../agents/plan-creator-sde.md#L21), quoted verbatim:

```
### Step 1: Determine which schemes to propose
Follow the scheme selection rules in `sde_manual.md`:
- **Additive noise OR multi-D** → 1 plan: Euler-Maruyama only
- **Multiplicative scalar noise** → 2 plans: Euler-Maruyama AND Milstein
- **Stiff** → document that implicit schemes are required but not yet supported;
  propose EM with a note on stability risk
```

backed by [`references/sde_manual.md:168`](../../references/sde_manual.md#L168):

```
1. `noise_structure == "additive"` OR `state_dimension > 1` → EM only.
```

**Every BENCHOP-SLV problem is multi-dimensional.** `state_dimension` is 2 or 3 throughout — B06
carries `state_dimension: 3`, `noise_structure: "multiplicative"`, `milstein_eligible: false`,
`milstein_ineligible_reason: "levy_area"`. So the rule as written admits exactly one scheme on the
entire suite: Euler–Maruyama.

The justification is sound *as far as it goes* and is stated at
[`sde_manual.md:38`](../../references/sde_manual.md#L38): multi-dimensional Milstein needs the Lévy
area. But the rule's conclusion is "use EM for all multi-D problems", which silently equates
"Milstein is unavailable" with "nothing but Euler is available". That is false, and the suite shows
how expensive the conflation is: the schemes C0 used on nine of ten problems — exact non-central-χ²
and Bessel transition sampling, Andersen QE — are all perfectly well defined in multiple
dimensions. None of them requires a Lévy area, because none of them is a Taylor scheme.

What actually happened is more interesting than a hard block: **the plan creator improvised past
its own rule in nine of ten pools** and wrote QE, exact-transition, AKS and Milstein-variant plans
anyway. Those improvisations were the more accurate plan in most pools. The rule therefore does not
prevent the good plan from being written — it prevents it from being *legitimate*, which is M5.

One consequence is visible in the record: **plan `1-` is a Euler-Maruyama variant in all ten
pools, and the conductor shipped plan 1 in seven of them.** **[rev, against
[07](07-review.md) §8]** That is *not* independent evidence of a positional bias. On this suite
plan position, scheme family and solver cost are **perfectly collinear** — plan 1 is always Euler
and Euler is always the fast plan — so M2, M4 and this observation are three readings of one
variable with n = 10, not three lines of evidence. [F7](03-fix-selection.md#f7) is re-scoped to a
reporting change rather than a fix for a measured bias. The
rule puts Euler first; the ranker ties; position decides.

**Fix: [F5](04-fix-planning.md#f5), [F7](03-fix-selection.md#f7).**

---

## M5 — `scheme_family: other` penalises the schemes that are off-menu

The frontmatter enum at
[`agents/plan-creator-sde.md:62`](../../agents/plan-creator-sde.md#L62):

```
scheme:        euler-maruyama | milstein
scheme_family: euler-maruyama | milstein | tamed-euler-maruyama | tamed-milstein | implicit | other
```

There is no value for quadratic-exponential, for exact transition sampling, or for a conditional /
mixed scheme. Every improvised plan had to declare `other`.

What `other` costs it: [`FAMILY_ORDER_KEY`](../../verifylib/kernel/sde/crn.py#L221) maps only
`milstein` and `tamed-milstein` to their own declared order. Everything else — including `other` —
falls through to `verification.expected_strong_order`, which the formulator sets to Euler's 0.5.
And because the family *was* declared, the band stays **two-sided**: 0.5 ± 0.4. A scheme that
integrates the variance exactly does not have Euler's strong order, so declaring `other` honestly
buys it a convergence check it can only fail or render `unresolved`.

`workspace/sde_sabr_cev_call_ii_lowvol/plans/2-qe-martingale-absorbing/SOLUTION.md` says this
about itself:

> **Expected strong order**: 0.5 (the spec's only declared floor; `scheme_family: other` selects it)

The plan creator is Opus, it understood the trap, documented it, and had no field to express the
truth in. This is a vocabulary defect, not a reasoning defect, and no model upgrade reaches it.

> **[C, against [07](07-review.md) §7] The suite does not measure what this defect costs, and this
> section previously claimed it did.** ~~It cited that plan as "104× more accurate and scored 2",
> as the price of declaring `other`.~~ The record says the plan scored 2 with `provenance: none`,
> `d1_outcome: unavailable`, `reference_outcome: unavailable` and `dynkin_z: unavailable`; its
> plain-Euler sibling scored 2 for the same reasons. **Nothing on B10 was lost to the order band.**
> No plan on the suite is shown losing a point to it — B07's QE plan is the only one scored down on
> an order-study outcome, and its recorded reason is that the estimator is erratic in dt, which is
> not a family-floor miss either.
>
> M5 stands as a defect found by **reading the code**: the mapping, the fall-through and the
> two-sided band are all exactly as described, and `FAMILY_ORDER_REQUIRED` is the repo's own
> precedent for fixing it. It does not stand on a measurement, and
> [F6](04-fix-planning.md#f6) is re-scoped accordingly.

**Fix: [F6](04-fix-planning.md#f6).**

---

## M6 — The loop stops learning after two cycles, and nothing certifies

Two numbers from the record:

- **Every one of the 24 plan records carries `iter: 2`**, and `total_iters == 2 × n_plans` in all
  15. Every plan ran exactly two solver↔evaluator cycles
  and then hit the plateau early-stop at
  [`conductor.md:117`](../../commands/conductor.md#L117) — *"a plan is plateaued when it has
  completed at least 2 cycles and its most recent evaluator score did not improve."* On this entire
  suite, **no plan's score improved on its second cycle**. The `max_iter = 5` budget was never
  approached, and the one refinement cycle that did run bought nothing the harness can measure.
- **Nothing reached `certified: true` on any problem**, so the winner early-exit at
  [`conductor.md:99`](../../commands/conductor.md#L99) never fired. Every pool was decided by the
  Phase-3 ranker — which is why M1–M3 account for the whole result.

And yet all **23 graded** plans pass the external gate (the 24th, B07's `2-qe-log-euler`, crashed
in verification) and every problem-level record is `UNDERCLAIM`. The kernel's
certification bar and the external bar are pointing in different directions on this suite: the
kernel certifies nothing that passes, and cannot order what it does score.

The practical reading for the next run: **the refine loop is the least productive part of the
pipeline on this suite.** It is not, however, the most expensive — **[C]** the budget it declines
to use costs nothing, and wall time tracks plan count instead. See
[F9](05-fix-loop-and-harness.md#f9), where the claim that cutting `max_iter` saves money is
withdrawn.

**Fix: [F9](05-fix-loop-and-harness.md#f9).**

---

## §7 Why a model upgrade does not reach most of this

The agents' declared models today:

| `model: opus` | `model: sonnet` |
|---|---|
| `formulator`, `plan-creator-pde`, `plan-creator-sde` | `solver-pde`, `solver-sde`, `evaluator-pde`, `evaluator-sde` |

[`benchmark/run.py:728`](../../benchmark/run.py#L728) notes that Claude Code honours each
subagent's own `model:` frontmatter over the session flag, so `--model opus` changes only the four
Sonnet agents — the solvers and evaluators. The planner and formulator are **already Opus**.

Map that onto the six mechanisms:

| Mechanism | Reachable by upgrading the solver/evaluator to Opus? |
|---|---|
| M1 ranker reads the wrong quantity | **No** — version-controlled Python |
| M2 cost decides ties | **No** — version-controlled Python |
| M3 first two keys inert | **No** |
| M4 scheme menu | **No** — the planner is already Opus and the rule is in the manual it is told to follow |
| M5 `scheme_family` vocabulary | **No** — an enum, and a kernel lookup table |
| M6 loop plateaus | **Partly** — a stronger solver might improve on cycle 2 where Sonnet did not |
| solver implementation quality (B02, B03, B06) | **Yes** — this is the one place it bites |

The B06 Opus salvage measured that last row directly: one plan, truncated at `iter: 0` with the
evaluator never run, reached **0.3999** against Sonnet's twice-iterated **0.5177** and C0's
**0.1445**. A better solver closed about a quarter of the gap on a problem where neither selection
nor the scheme menu was the binding constraint. That is real and it is not nothing — and it is also
comparable to the 1.3×–2.2× that selection is giving away for free — and selection is free to
fix.
