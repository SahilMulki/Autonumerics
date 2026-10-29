> **[M, 2026-10-29 — read first]** [08-free-measurements.md](08-free-measurements.md) §2 measured
> the quality index on the shipped metrics blocks: `accuracy` is **1.000 on all eleven** B01–B04
> plans, `order` is `None` on **all 88** blocks in `workspace/`, and `consistency` is `None`
> throughout, leaving a live index of **⅔ structure + ⅓ cost**. **F0, F2 and F3 below are
> necessary and not sufficient** — a price error inside its own Monte-Carlo noise saturates
> `accuracy` exactly as the moment error does. Do **F16** (unsaturate `accuracy`) and **F17**
> (`design_order` on the SDE path) with them, or the change will not move a single ranking.

# 03 — Fixing selection

**Status:** proposal, 2026-10-28; **substantially revised the same day against
[07-review.md](07-review.md)**, which found that F1 and F3 as first written could not do the job
they were given. F1 is re-specified, F3 gains a prerequisite that was missing from the fix list
entirely ([F12](#f12)), F4 is doubled because it pointed at only one of two code paths, and F7 is
demoted from a fix to a reporting change. The superseded specifications are struck through where
the reasoning is worth keeping.

**What the selection program is worth, restated.** The earlier headline — *"worth a geometric mean
factor of 2.18×"* — is an upper bound that selects on Monte-Carlo luck. Floored at each plan's own
error bar it is **1.33×**, which is a lower bound because that bar is deliberately conservative.
The defensible claim is **1.3× to 2.2×, with one problem (B10) carrying a real ~7× at 3.4–4.5σ**,
and [01](01-what-was-measured.md) §3 has the derivation. It is still the highest-leverage work in
the folder and it still needs no model change — but the retest's primary target was written
against 2.18× and has to be rewritten first. See [06](06-retest-protocol.md) §6.

---

## F0 (prerequisite) — the SDE sandbox throws the price away {#f0}

**Confirmed by [07](07-review.md) §13. Nothing in F1–F3 can be built until this is fixed, and it
is a few lines.**

The solver returns its declared functionals. On the PDE path,
[`_dump_pde`](../../verifylib/kernel/sandbox.py#L273) captures them across the child-process
boundary and [`_reconstruct`](../../verifylib/kernel/sandbox.py#L538) hands them back. On the SDE
path:

- [`_dump_sde`](../../verifylib/kernel/sandbox.py#L305) captures `terminal_paths`,
  `empirical_mean`, `empirical_variance` and `path_integrals`. **It never looks at
  `functionals`.**
- [`_reconstruct`](../../verifylib/kernel/sandbox.py#L521)'s `kind == "sde"` branch returns the same
  four keys and `return`s before the line that would restore them.

So on every SDE problem in the repo, the quantity of interest the solver computed is **discarded at
the sandbox boundary before the kernel ever sees it**. Every downstream claim in
[02](02-root-causes.md) M1 rests on this one omission.

The benchmark harness hit the identical bug and already fixed it on its own boundary —
[`benchmark/verify.py:1753`](../../benchmark/verify.py#L1753) records it:

> Reaching parity took three fixes to what the boundary carries: `functionals`, both result shapes
> at once, and an absolute solver path.

`verifylib`'s sandbox never got the same fix. **The change is to mirror `_dump_pde`'s
`functionals` block into `_dump_sde`, and the `out["functionals"]` line into the SDE branch of
`_reconstruct`.**

### The repo already has a red test naming this defect

**[rev] `verifylib/tests/test_spec_field_coverage.py::test_no_sde_spec_field_is_declared_and_unread_without_a_reason`
is currently failing, and F0/F2 are its fix.** Criterion 11 asserts that no field declared in a
live SDE spec is read by nothing in `verifylib/`. Its failure message lists, among others:

```
price_S75, price_S100, price_S125        (evaluation_thresholds.functionals.*)
no_touch_S75, no_touch_S100, no_touch_S125
price_K1, price_K2, price_K3
```

> Give it a consumer, or record why it has none in `test_spec_field_coverage.UNREAD`

That is M1 stated by the test suite, in the repo's own words, since the BENCHOP specs landed. The
test gives the whole selection program a **free, pre-existing acceptance criterion**: it goes green
when the consumer exists, and it cannot be satisfied by anything less than a real one.

Six further fields come back in the same census, and three of them matter for
[06](06-retest-protocol.md) §1's replication work — the specs already declare replication knobs
that nothing reads:

| unread field | bearing |
|---|---|
| `num_paths_replicate`, `num_paths_secondary`, `second_call_num_paths` | re-solving at a second path count — the machinery seed replication wants |
| `replication_check` | a declared check with no consumer |
| `attainable_boundary` (`verification.structural_facts`) | a structural fact the kernel could gate on |
| `feller_condition` | the condition that defines half this suite's difficulty |

Decide for each: give it a consumer, or add it to `UNREAD` with a reason. Doing neither is what
left the test red.

**Acceptance test.** The criterion-11 test goes green for the nine functional names. Plus the
direct one: run a solved BENCHOP plan through `sandbox.run(plan, "sde", ...)` and assert
`result["functionals"]` is a dict with the names `problem_spec.json` declares, alongside the
existing PDE round-trip. **Only B01–B04 have plan directories on disk**
([06](06-retest-protocol.md) §1), so write it against those.

---

## F12 — The solver contract has no notion of a quantity of interest {#f12}

**Effort: M. New, from [07](07-review.md) §4. This is the real prerequisite for F3, and the
original fix list did not contain it.**

F3's error estimate needs a per-functional error bar. There is none, and it cannot be derived from
what crosses the boundary:

- `uncertainty()` reads `mc_se_rel`, which
  [`sde/driver.py:133`](../../verifylib/kernel/sde/driver.py#L133) computes from `terminal_paths`
  as `max(se_variance / |variance|)`. **It is the standard error of the moments.**
- A price is a single float. Even after F0, the sandbox carries one number per functional and
  nothing else, so its standard error cannot be recovered on the kernel side at all.

Worse, the contract does not mention functionals anywhere a solver would read. `grep -c functional`
returns **0** for [`agents/solver-sde.md`](../../agents/solver-sde.md),
[`agents/evaluator-sde.md`](../../agents/evaluator-sde.md),
[`references/sde_manual.md`](../../references/sde_manual.md),
`references/project_manual.md` and `references/verification_manual.md`. The BENCHOP solvers return
prices only because each problem's own `output_contract` tells them to. **The generic SDE contract
has no quantity of interest in it.**

### The change

Extend `solve_sde`'s return contract with per-functional uncertainty, and declare it in the manuals
rather than per problem:

```python
return {
    "terminal_paths": X,
    "functionals":    {"price_K1": 0.0499, ...},
    "functional_se":  {"price_K1": 1.25e-5, ...},   # <-- new
    ...
}
```

`functional_se` is the Monte-Carlo standard error of each returned functional — `std(payoff) /
sqrt(num_paths)`, which every estimator already has in hand. Returning the per-path payoff samples
instead is the more general option and lets the kernel compute higher moments, but it costs an
array per functional across the boundary; the standard error is enough for everything F1 and F3
need.

Carry it through `_dump_sde` / `_reconstruct` in the same edit as F0, and add it to `solver-sde.md`
and `sde_manual.md` beside the existing `dW` and `observables` hooks.

**Acceptance test.** A spec declaring a gated functional whose solver omits `functional_se` reports
`not_reported` — a measurement — rather than crashing. `sde_manual.md`'s solver template shows the
two lines that compute it.

---

## F1 — Compare the functionals across plans, at the sampling error rather than at the gate {#f1}

**Effort: M. Re-specified: the first version's acceptance tests were unsatisfiable.**

### What was wrong with the first version

~~Add a comparison row group judging each functional against its own `rel_tol`, and require
`compare` to return DISAGREE on B10 and B04.~~ **[C, against [07](07-review.md) §3a.]** Three
things were wrong, and the first is fatal:

1. **A comparison at `rel_tol` cannot disagree on this suite, by construction.** Every plan passes
   the external gate, so any two plans' values are within `2 × tol` of each other. On B10 the gate
   is **0.01** — not the 0.003 quoted throughout the first draft, which is B06's and B07's figure;
   the suite's per-value gates span **0.002 to 0.015** — and the two plans' prices differ by 0.2%
   to 0.4% relative. `compare_pair` ([`cli.py:420`](../../verifylib/cli.py#L420)) would return
   AGREE. **The acceptance test as written could not be satisfied by a correct implementation.**
2. **An agreement verdict is symmetric and cannot order anything.** The stated payoff for B10 was
   "a disagreement caps both plans at 8" — but both plans already score **2**, so the cap is
   vacuous, and nothing in the agree/disagree/inconclusive vocabulary says *which* plan to ship.
   F1 can remove confidence from a pool; it cannot redirect selection inside one.
3. ~~"Where F1 has recorded a cross-plan consensus, the distance from consensus should take
   precedence."~~ **Dropped.** `accuracy` is in `CERT_COMPONENTS`
   ([`ranking.py:73`](../../verifylib/ranking.py#L73)) and
   [`sqi`'s docstring](../../verifylib/ranking.py#L395) forbids exactly this:

   > substituting a pool median makes one plan's index depend on its siblings', which nothing a
   > certification reads may do.

   `cost` is excluded from `sqi_cert` for this very reason. A consensus accuracy term would
   reintroduce the property that exclusion exists to prevent. It is also empty in practice: six of
   the ten pools held two plans, and with two values the midpoint is equidistant from both.

### The change, re-specified

Compare the functionals, but **against the combined sampling standard error, not against the
gate**. With `functional_se` from [F12](#f12), the comparison between plans A and B on functional
`k` is

```
z_k = |a_k - b_k| / hypot(se_a_k, se_b_k)
```

and the pool's verdict is the worst `z_k`. This is the same conservative-interval discipline
`_compare_sde`'s moment rows already use — [`cli.py:360`](../../verifylib/cli.py#L360) forms the
interval as if the two runs were independent, because *"correlation can only make the true error
bar smaller, and the conservative bar is the one that refuses to call a break it has not earned"* —
applied to the quantity the problem is actually scored on.

On B10 that yields **z = 3.4 to 4.5**: the two plans disagree on the price at four sigma while
agreeing on both moments to fourteen digits. That is a real, resolvable signal below the gate, and
it is the one the current comparison cannot see.

### What it is and is not for

- **It is a disagreement detector.** A pool whose plans differ on the graded quantity by several
  sigma has an unresolved discretization bias in at least one of them, and the existing
  agreement-gate cap is the right response.
- **It is not a selector.** Ordering the pool is [F3](#f3)'s job, and F3 must do it from each
  plan's *own* properties, never from the pool's.
- **It fixes the circularity break.** An agreement on the price between two genuinely different
  families is far stronger evidence than agreement on two moments both schemes get right for free.
  B10's SOLUTION.md reaches for this explicitly — it declares `scheme_family: other` *"precisely so
  that such an agreement counts"* — and today it counts for nothing, both because the comparison
  ignores prices and because [F6](04-fix-planning.md#f6)'s missing vocabulary makes two
  different-family plans read as the same family.

**Acceptance test.** On B10's two plans, `compare` reports a worst functional `z` above 3 and marks
the pool disagreeing *at the sampling level*, while still reporting AGREE at the gate — both facts,
distinctly. On B09's two plans (0.2968 vs 0.3017) it reports neither. **B09 and B10 have no plan
directories on disk**, so this test needs the artefact decision in
[06](06-retest-protocol.md) §1 settled first; B01–B04 can be used instead.

---

## F2 — Run the functional convergence check on the SDE path {#f2}

**Effort: S. Unchanged in substance; one caveat added.**

[`verifylib/kernel/functionals.py`](../../verifylib/kernel/functionals.py) already does the right
thing: carry each declared quantity up the ladder and ask whether it has converged to its own
`rel_tol`, in the closed outcome vocabulary `clean` / `stalled` / `not_reported` / `unavailable`,
with `not_reported` on a *gated* functional capping the score at 9 because — as its docstring puts
it — *"a declared gate the kernel cannot evaluate is worse than no gate, because it reads as
evidence."*

It is called from exactly one place, [`driver.py:245`](../../verifylib/kernel/driver.py#L245),
inside `evaluate_pde`. `grep -rn functional verifylib/kernel/sde/*.py` returns nothing.

### The change

Call `functionals.run_declared` from `evaluate_sde`, with three SDE-specific adaptations:

1. **The ladder is a path-count / step-size ladder, not a grid ladder.** If only one level is
   available the outcome is `unavailable`, which is a no-op, so a partial implementation is safe.
2. **The convergence bar must respect sampling noise.** `plan-benchop-slv.md` §6 step 5 already
   established this and the caution is on the record: on `sde_gbm_2d_high_corr`, 12 500 paths at
   dt=0.02 scores 8.1e-3 while 25 000 at dt=0.01 scores 1.5e-2 — *worse for twice the work*. **A
   simple decreasing test will fail correct solvers.** Use the harness's shipped shape: a symmetric
   band plus the requirement that the two values differ at all.
3. **[rev] Do not reuse `estimate_error` as a continuous quantity.**
   [`functionals.estimate_error`](../../verifylib/kernel/functionals.py#L70) infers an order from
   `log2(|d10| / |d21|)`, which assumes a refinement ratio of 2 **on one axis**. The CRN ladder
   varies dt *and* runs at `conv_num_paths` — 8 000 or 20 000 against graded counts of 400 000 to
   1 000 000 — so bias falls like dt while noise falls like `n^(-1/2)`, and an order inferred
   across that mixture is not a number about anything. F2's output is a **pass/fail gate**; F3's
   continuous estimate comes from [a different estimator](#f3).

### What it buys

A gated functional that has not converged stops the run certifying, on the SDE side as on the PDE
side. On this suite nothing certified anyway, so this fix changes no winner by itself — its value
is making the kernel's bar mean the same thing on both halves of the system.

**Acceptance test.** `evaluate_sde` on a BENCHOP plan emits a `functionals` block with one outcome
per declared name, and a plan whose solver returns no `functionals` reports `not_reported` and caps
at 9, not `crashed`.

---

## F3 — Let the accuracy component read the functional where one is declared {#f3}

**Effort: M, and larger than first estimated: it now depends on [F12](#f12).**

### The problem

[`ranking.accuracy`](../../verifylib/ranking.py#L99) carries 0.35 — the largest weight in the index
— and reads `estimated_rel_error`, which on an SDE is the worse of the mean and variance errors.
[`_tol_of`](../../verifylib/ranking.py#L463) judges it against `variance_rel_err_max`, defaulting to
0.1. Neither the functional's value nor its own `rel_tol` is ever consulted.

### The change

Where `evaluation_thresholds.functionals` declares at least one **gated** entry, compute the
accuracy component from the functional:

- **`tol`** = that functional's own `rel_tol`, worst over gated functionals by `err / rel_tol`.
- **`err`** = the sum of two terms, both measured where the problem is graded:
  1. **A dt-Richardson bias term at the graded path count under common random numbers.** Solve at
     `dt` and `dt/2` with the same `dW` block — the solver contract already supports this through
     the `dW` hook — so the sampling error cancels and the difference is the discretization bias.
     ~~The self-convergence estimate F2 produces.~~ **[C]** That estimate is measured on the CRN
     ladder at 8 000–20 000 paths, a factor of 20–125 coarser in sampling than the graded run, and
     because the ladder shares Brownian paths it is structurally blind to the sampling term — which
     is the dominant one at the graded count. It would rank plans on dt-bias alone.
  2. **The reported per-functional standard error**, from [F12](#f12). This is the term that prices
     variance reduction, and it is the axis C0 beat the pipeline on at B08 with a control variate.
- **Then snap up to the plan's own uncertainty**, as `accuracy` already does: *"a plan cannot claim
  accuracy finer than its own error bar."* This is not a detail. It is what stops B10's 2.22e-14
  from reading as infinite headroom, and it is what correctly declines to call B01 a selection
  failure ([01](01-what-was-measured.md) §3).

Cost: one extra half-step solve per plan. Keep the moment-based path untouched where no functional
is declared.

### Why not just show the kernel the truth

Because that collapses the separation the benchmark rests on: the kernel grades against the plan's
own spec, the harness against independently authored truth, and `wrong_plan_won` is only meaningful
while the kernel has never seen a reference value. dt-Richardson under CRN and a reported standard
error are both oracle-free.

### Validate the estimator before wiring it in

**[rev]** The first version went straight to an acceptance test on the ranked order. The prior
question is whether *any* oracle-free estimate can order these plans at all. Build the estimator,
run it on B01–B04's eight surviving plans, and compare its predicted error ordering against the
known harness errors. If it cannot separate B04's `1-euler-maruyama` (0.1741) from
`3-qe-exact-variance` (0.0603), nothing built on it will improve selection and the whole program
needs rethinking. That test costs compute and no credits.

**Then** the fixture test: a regression fixture from the pools' metrics blocks asserting the ranked
order — with the caveat from [06](06-retest-protocol.md) §1 that the JSON's per-plan `kernel` block
is a *reduced* record (no order or structure fields, no `config`), so `sqi` renormalises over a
different set and a fixture built from it does not reproduce the live index. Use the four
workspaces that still exist for anything that must be faithful.

---

## F4 — Stop cost deciding ties it should not decide {#f4}

**Effort: S for the weighted component; M once both paths are covered.**

### The problem, corrected

`solve_time_s` decided three of the ten shipped plans — B05, B08 and B10 — and on B10 it chose the
plan that is 3.4–4.5σ worse on the graded price. **[C, against [07](07-review.md) §5]** The first
version pointed only at [`ranking.cost`](../../verifylib/ranking.py#L369), the weight-0.10
component. That is not where B10 was decided. `solve_time_s` enters **twice**:

| | where | fires when |
|---|---|---|
| weighted component, 0.10 | [`ranking.cost`](../../verifylib/ranking.py#L369) | always, as a term in `sqi` |
| **lexicographic key 5** | [`rank_key`](../../verifylib/ranking.py#L437) | score, provenance **and** `sqi` all tie exactly |

On B10 the index is identical and key 5 decides outright. **The original acceptance test — "the
winner must not change if both plans' `solve_time_s` are swapped" — would still have failed after
the fix as described.**

Two further defects in the field itself, neither previously mentioned:

- **On the SDE path it is wall time.** [`sde/driver.py:93`](../../verifylib/kernel/sde/driver.py#L93)
  sets `solve_time_s = round(float(got["wall_s"]), 3)`, contradicting `cost`'s docstring.
- **It is contaminated by concurrency.** `parallelism = 3`, so plans in a pool contend for CPU with
  each other. The log-scale pool-relative form cancels a *uniform* slowdown; contention is not
  uniform.
- **[F8](05-fix-loop-and-harness.md#f8) makes this worse before it makes it better.** Once a
  forking solver can run — and B07's plan was forking workers when it crashed — its wall time is
  divided by its worker count, and `cost` becomes a reward for parallelism. **Either record CPU
  time, or require single-process solvers and say so in the contract.** This decision blocks F8.

### The change

1. **F1 and F3 are the real fix.** Once accuracy can see the price, B05, B08 and B10 stop tying and
   cost returns to breaking genuine ties. Do not reweight before measuring what F3 alone does.
2. **Then both cost paths.** The weighted component gets a tie-credibility guard: cost should only
   contribute when the accuracy component was measurable for every plan in the comparison. Key 5
   gets a different treatment, because a lexicographic key is a decision rule, not a tiebreak — see
   [F7](#f7).
3. **Then the field.** Record CPU time, or pin the process model; and make the PDE and SDE paths
   mean the same thing by the same name.

**Do not simply lower the cost weight.** It is right that an equally accurate, 2.7× cheaper scheme
should win. The defect is upstream.

**Acceptance test.** On a B10 fixture the winner must not change when both plans' `solve_time_s`
are swapped — **and the test must exercise `rank_key`, not `sqi` alone**, or it passes while the
defect survives.

---

## F7 — Report the tie instead of breaking it invisibly {#f7}

**Effort: S. Demoted from a fix to a reporting change.**

### The evidence does not support the original claim

~~`1-` is Euler in all ten pools and won seven, so there is a positional bias to remove.~~
**[C, against [07](07-review.md) §8]** Plan position, scheme family and solver cost are **perfectly
collinear on this suite**: plan 1 is always Euler, and Euler is always the fast plan — which is the
signal M2 already identifies as deciding those winners. M2, M4 and F7 were three readings of one
variable with n = 10, presented as three lines of evidence.

The proposed fix was also self-contradicting: *"a hash of the plan slug, or alphabetical"* is still
a fixed function of the plan's identity, so it fails the very permutation test proposed beside it.

### The change

Treat an exhausted ranking key for what it is. When `rank_key` reaches the `solve_time_s` key with
score, provenance and `sqi` all tied, **the ranker did not rank** — it is reporting that the pool is
indistinguishable on everything it can measure. So:

- Record `decided_by: <key name>` on the winner in `STATE.md` and in the results JSON. This is one
  field, [F10](05-fix-loop-and-harness.md#f10) should carry it anyway, and it is the field whose
  absence made this whole analysis expensive.
- Where `decided_by` is a tiebreak rather than a measurement, let the conductor spend **one**
  measurement breaking it — [F1](#f1)'s functional `z` is exactly such a measurement, and it is
  cheap relative to a plan cycle.
- Do not randomize invisibly. A hidden tiebreak is the same defect with the audit trail removed.

**Acceptance test.** Re-rank all ten pools and assert that every winner carries a `decided_by`, and
that on B05 and B10 it reads `solve_time_s` today. After F1/F3 it should read `sqi` on both.
