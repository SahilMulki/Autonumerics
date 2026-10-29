# 05 — Fixing the refine loop and the harness

**Status:** proposal, 2026-10-28; **revised the same day against [07-review.md](07-review.md)**.
F8's cause survives but its quoted fix was wrong and would have reinstated the bug. **F9 is
withdrawn** — it saves nothing, and the records prove it. F10 gains the exact line. Two fixes are
new: [F13](#f13), a guard on non-measurements, and [F14](#f14), the evaluator nobody is changing.

---

## F8 — The sandbox import does not register the module, and it voided a plan {#f8}

**Effort: S (one line). Confirmed cause; the correct version is already in the repo.**

B07's `2-qe-log-euler` was graded **FAIL** with:

```
PicklingError: Can't pickle <function _fork_work at 0x7fa80f81f6a0>:
import of module '_sandbox_solver' failed
```

That plan was *better than the winner on both of the ranker's inputs* — 5.0× better moments
(4.54e-4 vs 2.26e-3) and 2.2× faster (111.1 s vs 245.7 s) — and the harness never got a price out
of it, so B07 is the one problem where the pipeline's pool has no second graded point at all.

### The cause

[`benchmark/runner.py:157-168`](../../benchmark/runner.py#L157) loads the solver by path:

```python
spec = importlib.util.spec_from_file_location("_sandbox_solver", solver_path)
mod = importlib.util.module_from_spec(spec)
plan_dir = os.path.dirname(solver_path)
sys.path.insert(0, plan_dir)
try:
    spec.loader.exec_module(mod)
finally:
    sys.path.remove(plan_dir)
return mod
```

It never does `sys.modules["_sandbox_solver"] = mod`. `pickle` serialises a function by
`(module name, qualified name)` and then re-imports the module to resolve it — so any function
defined in this module is unpicklable, and anything that pickles one (a `multiprocessing` pool,
which is a reasonable thing for a QE solver to use to parallelise paths) fails with exactly this
message.

The sibling implementation gets it right.
[`verifylib/kernel/sandbox.py:124`](../../verifylib/kernel/sandbox.py#L124):

```python
spec = importlib.util.spec_from_file_location("plan_solver", path)
module = importlib.util.module_from_spec(spec)
sys.modules["plan_solver"] = module          # <-- the missing line
spec.loader.exec_module(module)
```

### The change

Register the module in `sys.modules` before `exec_module`, and **leave it registered**.

> **[C, against [07](07-review.md) §12]** ~~…and remove it in a `finally` so a second plan in the
> same process does not inherit the first one's module object.~~ That would reinstate the bug.
> Pickling happens **during the solve**, at [`runner.py:359`](../../benchmark/runner.py#L359) and
> after, long after `exec_module` returns — so a `finally` placed where the quoted `sys.path` block's
> is would unregister the module before the first worker is forked. The sibling at
> [`sandbox.py:124`](../../verifylib/kernel/sandbox.py#L124) leaves the registration in place, and
> that is why it works. If cross-plan contamination in one process is a real concern, the answer is
> to clear the entry *between plans*, not inside the import.

**And settle the process model first.** [F4](03-fix-selection.md#f4) notes the interaction: a
solver that forks workers reports a wall time divided by its worker count, and
[`sde/driver.py:93`](../../verifylib/kernel/sde/driver.py#L93) feeds that wall time straight into
`cost`. Fixing F8 without pinning the process model turns the cost term into a reward for
parallelism. Either record CPU time, or require single-process solvers and say so in the
contract.

**Acceptance test.** A fixture solver that uses `multiprocessing.Pool` with a module-level worker
function grades without crashing.

> **[C, against [07](07-review.md) §9] The "free data point" is not available.** ~~Re-grade B07's
> `2-qe-log-euler` with `run.py --only B07 --skip-run`.~~ `workspace/sde_hhw_call_zerocorr/` holds
> `problem.md` alone — no `plans/`, no `solver.py`. That run happened in a cloud session and the
> artefacts never came back. `--skip-run` parses STATE and re-verifies, which means re-solving,
> which needs the solver file. The record keeps the plan's `solver_sha256` and nothing to hash.
> **B07 stays blank unless it is re-run.** See [06](06-retest-protocol.md) §1.

---

## F9 — ~~Cut `max_iter` to 2~~ · Withdrawn. Propose more plans on their own terms {#f9}

**[C, against [07](07-review.md) §6]: the `max_iter` change saves nothing, and this section
previously claimed it halved pipeline cost.**

Every one of the 24 plan records carries `iter: 2`. The plateau rule stops a plan at `iter >= 2`
when its score did not improve, and it fired on every plan on the suite — so **`max_iter = 5` cost
exactly what `max_iter = 2` would have cost.** The saving is zero. What lowering it *would* change
is the counterfactual the plateau rule exists to permit: a plan whose score does improve on cycle 2
would be cut off. ~~Set `max_iter = 2` for the retest.~~ **Leave it at 5.**

The real content of this item is the other half — **more plans per problem** — and that is a cost
*increase*, not a saving. Wall time tracks plan count, not iterations: B08 at 3 plans took 78.7
min against B09 at 2 plans in 28.2. It is still probably the right trade, because
[01](01-what-was-measured.md) §3 shows the marginal value is in the pool rather than in the
iterations, but it must be proposed on its terms and with one caveat: **a larger pool inflates the
raw `best-in-pool` statistic the retest is graded on**, since that statistic is a minimum over
noisy draws. Grade the retest on the noise-floored version ([06](06-retest-protocol.md) §6).

The original reasoning about *why* the loop is unproductive stands and is worth keeping:

`iters == 2 × n_plans` in **all 15 records**. Every plan ran exactly two solver↔evaluator cycles
and then hit the plateau early-stop at
[`conductor.md:117`](../../commands/conductor.md#L117) — *"a plan is plateaued when it has
completed at least 2 cycles and its most recent evaluator score did not improve."*

So on this entire suite **no plan's score improved on its second cycle**, and the `max_iter = 5`
budget at [`conductor.md:25`](../../commands/conductor.md#L25) was never approached. The second
cycle also bought nothing the harness can measure: the B06 Opus salvage reached **0.3999** at
`iter: 0`, with the evaluator never run, against Sonnet's twice-iterated **0.5177**.

Two readings, and both point the same way:

- The refine loop is not converging on this suite. That is not surprising — the evaluator diagnoses
  against the kernel's metrics, and [02](02-root-causes.md) M1 shows those metrics cannot see the
  graded quantity. **An evaluator that cannot see the price cannot tell the solver how to improve
  the price.** F1–F3 should improve the loop as a side effect, by giving the evaluator a signal
  that moves.
- Until then, cycle 2 is pure cost. Pipeline wall-clock is **24.6–78.7 min** per problem against
  C0's 8 min for ten attempts. **[C]** That is the cost of the cycle that *ran*, not of the
  `max_iter` budget — cutting the budget saves nothing, because it was never reached. See
  [F14](#f14) for the measurement this section needs and no fix provides.

### The change

Raise the plan count per problem (`parallelism` is already 3) and leave `max_iter` alone. Record
both in the results JSON config block, which today carries neither, so the next analysis can tell
runs apart without re-deriving it.

**Acceptance test.** Nothing to assert — this is a budget change.

---

## F10 — `REPORT.md` drops the per-plan harness error {#f10}

**Effort: S.**

The "Every plan, graded" table in the generated report prints `-` in the **harness rel err** column
for every plan, on every run, although the JSON carries the values. From `REGRADE4.md`:

| problem | plan (* winner) | kernel score | provenance | harness rel err | verdict |
|---|---|---|---|---|---|
| sde_heston_dnt | 1-euler-maruyama * | 9.0 | analytic | **-** | UNDERCLAIM |
| sde_heston_dnt | 3-qe-exact-variance | 9.0 | analytic | **-** | UNDERCLAIM |

The JSON for those same two rows holds 0.1741 and 0.0603 — the 2.9× gap that is the whole finding
on that problem. The report's own header says this table exists because `wrong_plan_won` *"is the
number the Path-B ranking rules exist to drive to zero, and it is invisible in a winner-only
record"* — and then the column that would show it is empty.

**[rev] The line is [`benchmark/report.py:552`](../../benchmark/report.py#L552):**

```python
err = h.get("rel_l2_err")
```

`rel_l2_err` does not exist on a functional-graded problem — the field is
`worst_functional_rel_err`, and `harness_error` is already in the record one level up, on the plan
entry rather than on its `harness` block. One line.

Every table in [01-what-was-measured.md](01-what-was-measured.md) had to be rebuilt from the raw
JSON by hand because of it.

While there: add `est_rel_error`, `solve_time_s`, the plan's own `harness_error_margin`, and
**`decided_by` — which ranking key chose the winner** ([F7](03-fix-selection.md#f7)). The margin is
what [01](01-what-was-measured.md) §3's noise-floored statistic needs and it is already computed;
`decided_by` is the field that would have answered M2 immediately instead of after an afternoon of
re-derivation.

---

## F11 — ~~Open question~~ **Answered**: B01's more accurate plan scored one point lower {#f11}

> **[ANSWERED, 2026-10-29 — [08](08-free-measurements.md) §4]** The missing point is the CRN
> strong-order check. `2-exact-conditional-besq0` reads `observed_order: -0.0251` against a 0.5
> floor and `score.rubric_sde`'s `not order_ok` row applies. It is **not a defect**: the exact
> BESQ(0) transition is a Poisson-mixed chi-squared draw, so it cannot consume the supplied
> Gaussian increment, the CRN ladder decorrelates, and the measured order is meaningless. The
> plan predicted the penalty in writing before it was implemented. `crn.abstention` already
> excuses the *roundoff* version of this (exact **integration**); the **exact-sampling** version
> is invisible to it because the pathwise differences are large rather than tiny. See **F18** in
> the [README](README.md) fix list. The text below is the original open question, kept for the
> reasoning it records.

**Effort: unknown — this is a measurement, not a fix.**

| plan | harness err | kernel score | provenance | reference_outcome | est_rel_err | invariant drift |
|---|---|---|---|---|---|---|
| `1-euler-maruyama-absorbing` ★ | 0.2584 | **9** | analytic | validated | 3.68e-03 | 4.42e-04 |
| `2-exact-conditional-besq0` | **0.0259** | **8** | analytic | validated | **7.07e-04** | 6.69e-04 |

The exact-BESQ plan is 5.2× better on moments and 9.98× better on price, carries the same
provenance and a validated reference, and scored **one point lower**. B01 is the only pool on the
suite where the kernel score differed between plans — so the one time the ranker's primary key
fired, it fired backwards.

Nothing in the graded record explains the missing point. Candidates, in rough order of likelihood:
a capped check ([`score.ceiling`](../../verifylib/kernel/score.py)), an invariant reported by one
plan and not the other, an agreement cap, or a Dynkin / D1 outcome that is `unavailable` on one
side.

**How to answer it, for free:** re-run `verifylib/cli.py evaluate` on both plan directories with
the full metrics block retained, and diff the two `<metrics>` blocks field by field. The workspaces
and solvers are still on disk. Do this **before** F3, because if the kernel's score itself is
mis-ordering plans then F3's quality index — which sorts below `score` — will never get to vote.

---

## F13 — ~~Refuse an error estimate smaller than the plan's own sampling noise~~ — **it already does** {#f13}

> **[C, 2026-10-29 — [08](08-free-measurements.md) §2]** This is **already implemented**, in
> [`ranking.headroom_decades`](../../verifylib/ranking.py#L155): `err = max(err,
> uncertainty(summary))`, where `uncertainty` is `ci_mult × mc_se_rel`. And it is not a fix — it
> is the **saturation**. Because [`achievable_decades`](../../verifylib/ranking.py#L169) uses the
> same half-width as the ceiling, every snapped plan lands on `accuracy = 1.000`, measured on
> **11 of 11** B01–B04 plans. Re-scope F13 to the **reporting** layer, where B02's 25–38× gap is
> genuinely unflagged, and see **F16** for the component fix.

**Effort: S. New, from [07](07-review.md) §11.3. It generalizes well beyond BENCHOP.**

B10's two plans report `estimated_rel_error` of **2.22e-14** and **2.34e-14**, with
`provenance: none` and no reference of any kind. A Monte-Carlo estimate at 10⁶ paths cannot be
accurate to 1e-14. These are not small errors; they are non-measurements, and they are the input to
the component carrying 0.35 of the quality index. They are also why B10's pool tied and fell
through to cost.

The repo already has the concept in two places and applies it nowhere on the SDE path:

- [`headroom_decades`](../../verifylib/ranking.py#L124) returns `None` for an error of exactly
  zero, because that is *"the signature of a solver whose output did not respond to the grid at
  all"*.
- `baseline_select.looks_n_independent` makes the same judgement about a solver whose output does
  not move with `N`.

### The change

An `estimated_rel_error` below the plan's own `mc_se_rel` is **unmeasured, not perfect**: return
`None` from the accuracy component and let `sqi` renormalise, which is exactly what it does for
every other check that could not run. This is the same rule as `accuracy`'s existing snap-up,
applied one step earlier and as a refusal rather than a clamp.

**Acceptance test.** On a B10 fixture, both plans' accuracy components come back `None` and the
metrics record says why. Note that this alone does not fix B10 — it makes the tie *honest*, and
the tie then still falls to [F4](03-fix-selection.md#f4)'s key 5. Both are needed.

---

## F14 — Nothing in this folder changes the evaluator {#f14}

**Effort: unscoped. New, from [07](07-review.md) §11.5. Listed because leaving it unlisted was the
defect.**

M6 establishes that the refine loop learns nothing on this suite: no plan's score improved on its
second cycle, anywhere. The folder's answer is that F1–F3 *"should improve the loop as a side
effect, by giving the evaluator a signal that moves."*

That is a hope, not a plan. No fix touches [`agents/evaluator-sde.md`](../../agents/evaluator-sde.md),
and no acceptance test anywhere checks whether the loop improves afterwards. If the second cycle is
genuinely worthless it is a large fraction of pipeline cost for nothing, and that deserves its own
measurement.

**The minimum version:** after F0, F2 and F12 land, give the evaluator the functional block in its
diagnosis — it currently cannot see the graded quantity either, for the same reason the ranker
cannot — and **record per-cycle score deltas in the results JSON** so the next run can answer
"did cycle 2 help" without re-reading transcripts. If the answer is still no, cut the loop rather
than keep paying for it.

---

## F15 — ~~B02 is undiagnosed~~ **Diagnosed**, and worse than recorded {#f15}

> **[ANSWERED, 2026-10-29 — [08](08-free-measurements.md) §5]** B02's shipped plan **fails the
> external gate at 1 of 3 seeds** (1.1380 at seed 202) and `2-exact-vol-em-absorbed` fails at
> **2 of 3**. The recorded 0.9594 pass was the seed. The cause is not scheme quality: all three
> plans report `mc_se_rel` of **0.18–0.28** against an `estimated_rel_error` of 7.4e-3 — the
> estimate is 25–38× below its own sampling noise — and all three carry `resolved: false`, with
> an `INCONCLUSIVE` agreement gate. Nothing in the loop turns "unresolved" into "raise
> `num_paths`", so all three plateaued at iter 2 with the evaluator finding no defect. See **F19**
> in the [README](README.md) fix list. The `test_review` failure is a **true positive against the
> metrics block**: `observed_order: -1.0772` is the *abstaining* σ component's roundoff noise,
> while the prose's 0.235 is the real S-channel measurement.

**Effort: unscoped. New, from [07](07-review.md) §11.6.**

B02's shipped plan sits at **0.9594** — within 4% of failing the external gate, and the only plan
on the suite anywhere near it. Its whole pool is bad: 0.9594, 0.9098, 0.7592, against C0's best of
0.0116. That is a **65× gap at the pool's best**, the largest on the suite, and no fix in this
folder addresses it. It was dismissed as "an implementation-quality gap" and left out of the
retest subset on the grounds that F1–F4 will not move it — which is true and is the wrong
conclusion, because it is the one problem where the pipeline nearly produced a wrong answer rather
than a merely imprecise one.

It is also where the suite's other red test points: `test_review::test_no_false_positives_on_archived_reviews`
flags B02's `2-exact-vol-em-absorbed` for an `observed_order` of 0.235 recorded in its review prose
and disagreeing with its own `<metrics>` block ([06](06-retest-protocol.md) §7a). An observed
strong order of 0.235 against a 0.5 floor is itself a finding — that plan's convergence was not
what it claimed, on a problem whose whole pool is 65× off.

**Do this before the retest, and offline:** B02's three plan directories survive in
`workspace/sde_sabr_cev_call_ii/`. Read the solvers, read the order study, and find out what is
wrong. It needs no credits and it is the only open question on the suite about *correctness*
rather than about ranking.

---

## §5 Two calibration findings, not yet fixes

### Everything is `UNDERCLAIM`

All 15 records carry the verdict `UNDERCLAIM`: the independent harness passes, the pipeline's own
kernel scores 2–9 and **certifies nothing**. All 24 graded plans have `harness_error < 1.0` — every
one of them clears the external gate.

The kernel is therefore miscalibrated in both directions at once on this suite. It is **too harsh**
— it certifies nothing that in fact passes — and **too blind** — it scores B10's two plans 2 and 2,
one of which is at 0.4% of its gate and the other at 42.5%.

Two of the suite's problems make the "too harsh" half concrete:

- **B10**: both plans `provenance: none`, `d1_outcome: unavailable`, `dynkin_z: unavailable`,
  score 2. The Dynkin generator check is memory-bound at these path counts; `unavailable` already
  caps the score at 9, and here the pool is at 2.
- **B07**: both plans score 3 on `provenance: surrogate`.

The fix is not to loosen the kernel. It is F1–F3: give it a signal that discriminates, and the
scores will spread on their own. **But record the UNDERCLAIM rate in the retest** — if it is still
10/10 after the fixes, the certification ladder needs its own study.

### The margin is generous, so `wrong_plan_won: 0` understates the problem

`wrong_plan_won` reported **False** on 8 of 10 problems, which reads as "selection is mostly fine."
It is not, and the reason is in the estimator's own docstring
([`benchmark/run.py:555`](../../benchmark/run.py#L555)): it takes the larger of two available
divisors so that the comparison *"errs towards not calling a ranking wrong"*, and it is one
realization — a noise estimate with a single degree of freedom.

B04-local is the clean case: winner at 0.1741, sibling at 0.0603, a **2.9× accuracy gap**, and
`wrong_plan_won: False` because the gap of 0.1138 sits under a margin of 0.1668.

This is the right conservatism for the flag's stated purpose and the wrong statistic to manage the
pipeline by. **Report the geometric mean of shipped ÷ best-in-pool alongside it, in both forms** —
raw **2.18×** and noise-floored **1.33×** ([01](01-what-was-measured.md) §3) — which is a
continuous measure of selection quality that does not need a significance test to be informative.
Quoting only the raw one rewards a ranker that picks lucky seeds; quoting only the floored one uses
this same over-conservative margin as the floor. **[rev]** Both, or neither.
