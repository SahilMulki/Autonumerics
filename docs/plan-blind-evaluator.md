# Oracle-Free Candidate Ranking — Implementation Plan

**Status:** §§4, 7, 8 built (2026-09-28); §§5, 6, 9 proposal
**Scope:** how competing plans are ordered, how that ordering is measured, and what an oracle-free
ranker is actually for
**Related:** [plan-no-closed-form.md](plan-no-closed-form.md),
[plan-hallucination-guardrails.md](plan-hallucination-guardrails.md)

Provenance labels: **[A]** = already in this repo, **[B]** = adapted from the AutoNumerics
paper-release pipeline, **[N]** = new in this plan, **[X]** = built by this plan.

> **Revision note (2026-09-28).** The first draft of this document was written before the
> no-closed-form work landed. Three of its claims had gone stale and one had been inverted by the
> code; they are corrected in place and the corrections are marked. The two facts that changed the
> design most: cross-plan agreement now *certifies* (§4), and `wall_time_s` turned out not to be a
> cost metric at all (§8a).

---

## 1. What the paper-release Blind Evaluator is, and what it measured

It is **not** a solver and **not** an accuracy metric. It is a *selector study*: it re-ranks
already-executed candidates using only a sanitized PDE spec, sanitized plans, a code **summary**
(line count, imports, function names, hash — not the code), execution metadata, and solver-only
replay timing.

Critically, its `numerical_diagnostics` field is `{"available": {}, "missing": [...]}` — **empty in
practice**. It ranks solvers having never seen a number produced by any of them.

Measured result over 24 candidate pools:

| Metric | Result |
|---|---|
| Top-1 agreement with oracle-best | 6/24 (25%) |
| Blind pick passes the quality threshold | 12/24 (50%) |
| Blind pick within the lowest three diagnostics | 14/24 (58%) |
| Abstentions | 0 |

That is honestly reported and weak — which is itself the finding: **plan prose plus code metadata
does not select a good solver.** It is a useful negative result about LLM-as-judge, and it is the
right baseline to measure against, not a target to beat.

One caveat this document's own §5 supplies: **25% top-1 tells you nothing about whether the misses
cost 1.01× or 10⁴×.** The number is not interpretable without a regret measure, so the control arm
in §5 must be re-scored under one rather than merely reproduced.

---

## 2. How this repo ranks candidates today **[A]**

1. `plan-creator-{pde,sde}` emits **2–4 competing plans**, dispatched concurrently
   (`parallelism = 3`).
2. Each plan runs its own solver↔evaluator loop and is scored **1–10 by the kernel**
   (`verifylib/kernel/score.py`) — never by an agent.
3. **Winner early-exit** — the instant any plan certifies, the loop stops; no further cycles launch.
4. **Plateau early-stop** — `iter >= 2` with no improvement marks a plan `stopped`.
5. `commands/conductor.md` Phase 3 ranks lexicographically.

> **Correction.** The first draft gave the Phase 3 key as
> `score → estimated_rel_error → order margin → wall_time → iterations` and proposed moving
> provenance to position 2 as new work. **Provenance was already key 2**
> (`conductor.md`, and `benchmark/baseline_select.py:rank_key` as code). That part of §4 was
> describing a change that had shipped.

**This is oracle-informed ranking.** Every plan arrives carrying a measured error. The paper-release
blind evaluator ranks with no numerical measurement at all. They are different operations, and its
25% top-1 is not a bar this repo is failing to clear — it is a floor on what is achievable with zero
numerical evidence.

### What the run data says

Re-measured against `benchmark/results/results.json`, which now holds **28** records, not the 16 the
first draft cited. The pathology got worse, not better:

| | first draft | 2026-09-28 |
|---|---|---|
| Problems with **multiple plans tied at 10** | 8 / 16 | **18 / 28** |
| Problems with plans left `unrun` (`iter: 0`) | 7 / 16 | **14 / 28** |
| Plans created / cycles executed / plans never run | 42 / 36 / 9 | **83 / 66 / 20** |
| Sweep wall / per cycle | 3.02 h / 302 s | 4.89 h / 267 s |

And `pde_heat_2d` reproduces exactly, `workspace/pde_heat_2d/STATE.md`:

| Plan | Design order | Score | `est_err` | Solver time | State |
|---|---|---|---|---|---|
| `1-fd-explicit-rk2` | 2 | 10 | 1.023e-04 | 1.51 s | winner |
| `2-crank-nicolson-sparse` | 2 | 10 | **1.010e-04** | 0.81 s | **selected** |
| `3-adi-peaceman-rachford` | 2 | 10 | 1.019e-04 | **0.40 s** | winner |
| `4-fourth-order-rk3` | **4** | 0 | null | — | **`unstarted`** |

Three second-order schemes hitting the same discretization floor, spanning **1.3%** — noise, not a
difference — decided on a 0.9% margin, while the one plan with a different asymptotic rate never ran.
At `N = 127` a 4th-order scheme should land near `1e-8`.

The solver-time column is new, and it is the other half of the finding: the three tied plans differ
by **3.8× in cost**, and no key in the old ladder could see it.

**The weakness is not that the criteria are the wrong quantities.** It is that score 10 is a
*threshold* — `converged` is `gci < rel_l2_err_max`, so 9.9e-3 and 1e-10 both certify — so all the
discrimination was pushed onto tiebreakers reading inside the noise floor, while the early-exit
discarded the candidate most likely to win by orders of magnitude.

---

## 3. Three separate things

The paper-release design conflates them. Separating them is most of the work.

| | What | When it runs | Cost |
|---|---|---|---|
| **(a)** | Deterministic oracle-free ranker | In-loop, Phase 3 | Reads the metrics JSON the kernel already emitted. **Milliseconds.** |
| **(b)** | Agreement / regret experiment | **Offline, once, over frozen artifacts** | Zero marginal cost per production run |
| **(c)** | LLM anomaly detector | One call per plan | ~42 calls across a full 16-problem sweep |

Only (c) is an LLM, and it never touches a score.

---

## 4. (a) Deterministic oracle-free ranker **[X]**

Built as `verifylib/ranking.py`, imported by both the conductor (via `cli.py rank`) and
`benchmark/baseline_select.py`, so the pipeline and the baseline study cannot rank by different
rules. For problems with no ground truth you *must* rank without an oracle; this is the production
path for that slice.

### The key

```
1. score        ↓     unchanged
2. provenance   ↓     unchanged  [A, not [N] as first drafted]
3. sqi          ↓     [X] replaces estimated_rel_error, order margin, wall_time_s
4. solve_time_s ↑     [X] the solver at the graded grid
5. iterations   ↑     unchanged
```

Keys 1–2 stay lexicographic and stay first. They are claims about *what was established*, not
measures of quality, and a weighted sum must never trade a validated-analytic 9 against a
self-convergence 10 — averaging that away is the one thing the provenance vocabulary exists to
prevent. Everything below them is quality, and that is one number.

### The certification score resolves quality too **[X]**

`ceiling()` became a **band** and the accuracy headroom places the run within it:

| `ceiling()` | Band | |
|---|---|---|
| 10 — Tier A / A⁻ quoted / A′ / B + two breaks | **9–10** | certified; 10 also needs headroom |
| 9 — A⁻ unavailable / one break | **8–9** | one circularity break |
| 8 — Tier C / `unshadowable` | **6–8** | self-convergence only |
| 3 / 2 / 1 — gates / nothing ran / crash | unchanged | findings have no quality |

The top of a width-1 band needs **a third of the achievable headroom** — two decades below the
tolerance on a PDE. Three things make this safe, and each was found by a test rather than reasoned
out in advance:

1. **It reads accuracy alone, not the composite index.** The first cut compared the index against
   0.75, on the reasoning that the certifying row's other components sit at 1.0 so the bar reduces to
   two decades. They do not sit at 1.0 — they are often *absent*, and the index renormalises, so the
   effective accuracy bar moved with which checks happened to run. Where only accuracy and structure
   were measurable, accuracy carried 64% of the weight and the bar demanded ~3.6 decades; the canary
   suite's deliberately **honest** control solver scored 9. A bar that tightens because a plan's D1
   was `unavailable` is the exact failure the renormalisation rule exists to prevent, one level up.
2. **It is a fraction of what is achievable, not a fixed number of decades.** A Monte Carlo
   estimate's accuracy is bounded by `1/√M`, not by its scheme. Measured on the honest SDE canary: at
   50 000 paths against a 10% variance tolerance the ceiling is **0.90 decades** and the solver sits
   at 0.896 — already as accurate as the path count allows. A two-decade bar is unreachable there for
   any solver. The same correction had to be made inside the index's accuracy component, where an
   SDE's contribution could not exceed ~0.2 against a flat six decades.
3. **It applies only where the rubric put the run at the top of its band, and never reads the full-`T`
   error on a chaotic problem.** The rubric's lower rows are already accuracy-graded, so placing them
   again demotes one shortfall twice; and §28 waives pointwise accuracy at `T` precisely because
   trajectories separate past the Lyapunov time. Both were found by frozen tests going red — the KS
   plans moved down by one and two points before the second rule went in.

Measured: on `pde_heat_1d` FTCS (2.45 decades), Crank–Nicolson (2.59) and 4th-order RK4 (6.67) all
score 10; on `pde_heat_2d` all three second-order plans land at 1.99–2.00 and score **9**, with the
4th-order plan that would have cleared it never run.

**`certified` is what makes this cost nothing.** `benchmark/report.py` computed
`pipeline_pass = best_score >= 10`, so `VERIFIED_PASS` — and the whole C0/C1/C2 headline — turned on
one integer. `certify()` now emits `certified` (exactly the old `score >= 10`, settled off every cap
*except* placement, including `agent_cap`), and the winner early-exit and `compute_verdict` read that
instead. The verdict taxonomy is untouched, no frozen record's verdict moves, and the loop runs the
same number of cycles.

### The quality index

Five components, each in `[0, 1]` or `None` for "not measurable", weighted and **renormalised over
whichever were measurable**:

| | Component | w | What it measures |
|---|---|---|---|
| **A** | Accuracy headroom | **0.35** | `log10(tol / err) / 6`, err first snapped up to its own error bar |
| **B** | Order fidelity | **0.25** | distance from the scheme's **design** rate, two-sided |
| **C** | Structure headroom | **0.20** | worst declared invariant, against **its own** tolerance |
| **D** | Consistency rate | **0.10** | D1 slope margin, or Dynkin `z` against the family-wise bar |
| **E** | Cost | **0.10** | pool-relative `solve_time_s`, log-scaled |

The weights are asserted, not fitted; §5 exists to fit them. The ordering is the argument: accuracy
is what the problem asks for; order fidelity is what makes accuracy predictable rather than lucky,
and is the primary object of code verification; structure is what practitioners will not trade, since
a symplectic scheme with larger local error beats a higher-order one over long horizons; consistency
is real but already priced into the *score*, where a clean D1 is worth a full point of certification,
so weighting it heavily double-counts; cost is subordinate to correctness and decisive between
equals.

**Renormalisation is a hard rule, not an optimisation.** Substituting zero for an unmeasured
component charges a skipped check as a failed one, which manual §24/§25 and `evaluator-pde.md`'s Key
Rules all forbid; substituting a pool median makes one plan's index depend on its siblings', which
nothing a certification reads may do. Stated precisely — because the weaker version is the true one —
the index is a weighted **mean**, so dropping a component moves it toward the mean of the rest. It is
*not* invariant under dropping, and it cannot be. What it guarantees is that nothing is scored
against a bar it was never measured against.

That leaves one gap, closed a level up rather than here: a plan whose *weakest* component is the one
that did not run scores a higher index. The architecture already prices that — a D1 that did not run
forfeits a circularity break, so `score.ceiling` caps at 9 where the reporting plan reaches 10, and
the key reads `score` first. Skipping a check can improve the index; it cannot improve the rank.
`verifylib/tests/test_ranking.py` ties those two facts together so neither can move alone.

### Three corrections to the key this document first proposed

The first draft's §4 listed eight keys, three of which were wrong:

- **"D1 term-balanced residual ↑"** contradicted this repo's own manual. §25 puts "a residual
  reported as a magnitude" on the never-happen list and §26 explains there can be no calibrated
  threshold for one: a solver's residual is dominated by its own truncation error, so any constant bar
  either fails every correct coarse-grid solver or passes everything. The rankable quantity is the
  **slope**, `d1_slope`, and component D uses its margin past the `p_design − 1` bar `slope_test`
  applies.
- **"max invariant drift ↑"** had no source field — `invariant_max_drift` appeared only in
  `verification_manual.md`'s example block and in this document, never in `metrics.BLOCK_KEYS`. It is
  emitted now, but it is **not** what component C reads: raw drifts of different invariants are not on
  one scale (a mass drift of 1e-8 against a 1e-8 bar is marginal; a symmetry drift of 1e-8 against
  1e-4 is not). C reads `invariant_worst_headroom`, each drift against the tolerance *it* was judged
  against, which required moving the per-invariant defaults out of the evaluators and into the
  registry.
- **`observed_order − order_floor` descending was backwards on the SDE side.** `order_floor` there
  carries the *expected strong order* and `strong_ok` is the two-sided `|γ − γ_exp| < 0.4`, so within
  the passing band a margin of +0.39 sorted ahead of 0.00 — ranking by distance from the correct
  answer. On the PDE side the floor is a pass bar (1.0, and 0.0 on `pde_kuramoto_sivashinsky`), not a
  target, and `p_sane` already treats `p ≥ theoretical_order + 1` as evidence the ladder is *not*
  asymptotic. Component B compares against the **design** order, two-sided.

### What it does on the case that motivated it

Measured, `cli.py rank`, with every plan re-evaluated:

| `pde_heat_2d` | A | C | E | sqi |
|---|---|---|---|---|
| `1-fd-explicit-rk2` | 0.332 | 1.000 | 0.000 | 0.486 |
| `2-crank-nicolson-sparse` | 0.333 | 1.000 | 0.467 | 0.559 |
| `3-adi-peaceman-rachford` | 0.332 | 1.000 | **1.000** | **0.640** |

The 1.3% error spread becomes **9.25e-04** of a point: the three genuinely tie on accuracy and fall
through to cost, which separates them 3.8× and picks the ADI scheme. The old key picked
Crank–Nicolson on the 0.9% error gap. The new answer is the one a numerical analyst would give:
**equally accurate within the error bars, so take the cheaper one.**

On `pde_kdv_zabusky_kruskal` both keys agree, which is the other half of the requirement — the index
changes the answer only where the old key was reading noise. There the accuracy differences are real
(1.5e-10 / 2.4e-9 / 1.7e-5, five decades) and A correctly dominates.

### Cross-plan consensus does **not** move here

> **Correction — this is the claim the code inverted.** The first draft said consensus "leaves the
> scoring ladder and becomes a tiebreak and anomaly signal" and that "the existing rule holds
> unchanged: **corroborates, never certifies, never lifts a score**."
>
> That is no longer true and has not been since the no-closed-form work.
> `verifylib/kernel/score.py:agreement_is_break` makes cross-plan agreement a **certifying second
> circularity break** that lifts Tier B from 9 to 10, and `conductor.md` makes it a mandatory Path-B
> gate whose *disagreement* caps both plans at 8. The reasoning is in
> `plan-no-closed-form.md` §4c: D1 is structurally `unresolved` for a resolved spectral solver, so
> agreement with an independently written plan of a *different* `scheme_family` is the only second
> break such a plan can earn.
>
> So consensus stays in the scoring path, and `verification_manual.md` §11/§22 should stop deferring
> to this document for a relocation that is not going to happen. What §11's text still gets wrong is
> smaller and worth fixing separately: it defines agreement as
> `rms(u_a − u_b)/rms(u_b) < gci_a + gci_b`, while the shipped `cli.py compare` tests `rel_diff`
> against `rel_l2_err_max` at the graded grid.

---

## 5. (b) The agreement / regret experiment **[N, using B's machinery]**

This is the valuable measurement, and where the paper-release machinery earns its keep — the
sanitizer, alias maps, batch provenance and rank-serialization contract are all reusable.

### The oracle side already exists

> **Correction.** The first draft specified "compute the oracle rank by true error from
> `problems.py`" as new work. `benchmark/run.py:grade_every_plan` already grades **every** plan
> directory and emits `best_plan_by_harness`, `best_harness_err`, `winner_harness_err`,
> `harness_error_margin` and `wrong_plan_won`. The study reads `harness_by_plan`; it does not
> re-derive it. The draft also did not know about `baseline_select.py:selection_lost_a_pass`, which is
> the same selector-failure metric on the baseline side.

### Regret, redefined

`err(blind) / err(oracle)` breaks where it matters most: when the oracle pick sits at the round-off
floor the ratio explodes meaninglessly, and on `pde_heat_2d` it reports a regret at all when the
honest answer is that the choice was free. Work in log space and respect the noise floor:

```
digits_lost = max(0, log10(err_blind) − log10(err_oracle))
            = 0  when both picks pass AND |err_blind − err_oracle| < harness_error_margin
```

`harness_error_margin` is already computed by `run.py:plan_harness_error` — the reference's own error
for a PDE, the deciding moment's Monte-Carlo standard error for an SDE. Floor both errors at the
achievable precision. Report:

| Metric | Definition |
|---|---|
| Top-1 agreement | Blind pick == oracle-best |
| Top-3 containment | Blind pick within the three lowest true errors |
| **Median / P90 `digits_lost`** | How many digits of accuracy the selector gave up, and the tail |
| `wrong_plan_won` / `selection_lost_a_pass` | The pass-preserving binaries the repo already has |

### Fitting the weights

`benchmark/rank_study.py` should sweep the weight vector and report P90 `digits_lost` as a function
of it, plus the sensitivity to ±0.1 on each weight. That turns "these weights" from an assertion into
a measurement. **It is blocked on data, not on code**: only 3 of 28 result records currently carry
`harness_by_plan`, so a sweep has to populate it across the suite first.

### The baseline arm

Run the paper-release LLM judge as a **control condition** on the *same frozen pools*, so the
comparison is paired. That pairing, not reproducing 25%, is the acceptance criterion — a protocol
that reproduces a number has not thereby been validated. The ablation it yields:

> *"Oracle-free ranking from the verification battery achieves X% top-1 and median regret Y digits,
> against Z% top-1 and W digits from plan prose and execution metadata alone."*

This isolates the value of the **battery** from the value of the **judge**.

---

## 6. (c) LLM anomaly detector **[B, demoted]**

There is one thing a deterministic ranker cannot do: read a code summary against its plan and notice
*"claims WENO, has no limiter"* or *"plan says implicit, code has no linear solve."* That is
intent-versus-implementation pattern matching, and it is real value.

Keep it. Constrain it:

- Output is a list of **warnings**, not scores or ranks.
- **Never** an input to the ranking key — enforced structurally, since `verifylib/ranking.py` reads
  only the kernel's own metrics.
- Subject to the paper-release output-schema validation: named evidence sources only.

Cheaper than the draft implied: warnings already have a destination. `verifylib/gate.py` findings
flow into `REPORT.md`'s *Verification gaps* section, and the evaluator's `agent_cap` is already the
sanctioned route for a judgement no check measures — downward only, with a machine-readable reason.

---

## 7. The `oracle_derived` flag **[X, partial]**

Tag every field at the point of computation rather than blocklisting names, since names drift and a
legitimate oracle-free field can share a name with an oracle-derived one.

> **Correction — the threat model is narrower than first stated.** The draft said "the moment
> numerical evidence is fed to a blind ranker, the leakage scan has to run on the evidence itself",
> treating Path-A `estimated_rel_error` as leakage. It is not, *inside the pipeline*: it is derived
> from the **formulator's** claimed closed form, which is legitimate Tier-A evidence the pipeline is
> entitled to use. The oracle that must not leak is `benchmark/problems.py`, and
> `verifylib/leakage.py:PROHIBITED_MODULES` already blocks it. `oracle_derived` therefore matters for
> the **offline rank study of §5**, where the oracle is the harness, not for in-loop ranking.

Status: the *reader* shipped with the guardrails work — `verifylib/review.py:_coerce` parses
`{"value": x, "oracle_derived": bool}` and discards the flag, pinned by `test_review.py`. No kernel
module emits it yet. Retain the paper-release token scan as a second layer; they fail differently.

---

## 8. The dispatch-order fix — free, and it does most of the work **[N]**

`commands/conductor.md` sorts eligible plans by `(score asc, iter asc)`. At the start every plan is
`score: 0, iter: 0`, so that sort is a no-op and the pool fills in **plan-file order**. On
`pde_heat_2d` that put plans 1, 2, 3 in the first `parallelism = 3` wave and left the 4th-order scheme
unrun. Still true, still the single largest available win.

**Sort the first wave by design order descending, within a family-diverse cover.**

> **Correction — a pure design-order sort would break the Path-B gate.** The draft proposed sorting
> on design order alone. On Path B the early-exit does not fire until two plans of *different*
> `scheme_family` have been evaluated, so filling `parallelism = 3` with the three highest-order
> plans can put one family in the whole wave and *cost* a cycle on exactly the problems this is meant
> to help. Take the highest-design-order plan of each distinct family first, then fill by design
> order.

Cost: **zero.** It is a scheduling change, not an extra cycle. The best plan enters the first wave; if
it certifies first, the early-exit fires on the *best* plan rather than the first-listed one.

**The field mostly exists.** `spatial_order` and `scheme_family` are declared in `SOLUTION.md`
frontmatter and read by `plan_meta.read()`; what was missing was the conductor reading them at
dispatch, and their reaching the metrics block. Measured caveat: across `workspace/`, only **25 of
104** plans declare `scheme_family` and **22 of 104** declare `spatial_order` — both fields landed
with the no-closed-form work, so they are present on the newer pools only. Until a fresh sweep, B
renormalises out on most plans. Inferring a design order from a plan's *slug* is not an option: it
would let a plan rename itself into a better rank, against the declared-never-inferred rule the rest
of this repo is built on.

### 8a. `wall_time_s` is not a cost metric **[X]**

Not in the first draft at all, and it is why key 5 never did any work. `wall_time_s` is measured from
the top of `evaluate_pde` to the end — the *whole evaluation*, including the 2-vs-3-level ladder, MMS,
D1 and the temporal probe. Manual §24 prices that between **1.0× and 35×** by evidence path and
Stage-2 gating. Measured: `pde_heat_1d/3-fd4-rk4` solves in 0.28 s and evaluates in 1.20 s; within a
single Path-B pool (`pde_kdv_zabusky_kruskal`) the overhead ranges **7.9× to 12.5×**, so it varies
*within* a pool as well as across one.

`solve_time_s` — the solver at the grid the error is quoted on — is emitted instead. The measurement
already existed: `sandbox.run` has always returned `wall_s` per call.

### Why this matters more than the ranking itself

There is a genuine tension in the repo:

- If the deliverable is **"a solver that meets the spec"**, the early-exit is correct and free.
- If the deliverable is **"the best scheme for this problem"** — which `README.md` states as *"return
  the one with the lowest measured error"* — then the early-exit does not support the claim when three
  plans tie and a fourth never ran.

Dispatch ordering resolves this for free by making "first to certify" much more likely to *be* the
best.

---

## 9. Cost menu

From the re-measured sweep: 4.89 h total, 83 plans, 66 cycles, 20 plans never run, **267 s per cycle**.

| Option | Cost | What it buys | Status |
|---|---|---|---|
| **0.** Sort dispatch by design order, family-diverse | **0%** | Best plan in the first wave | pending |
| **1.** Order-guard: do not finalize while a strictly-higher-order plan is `unrun` | ~+8% *(estimate)* | Catches the `heat_2d` case exactly | held |
| **2.** Full bake-off: one cycle for every unrun plan | **+25%** | Complete pool, defensible "best plan" claim | held |
| **3.** Deterministic oracle-free ranker (§4) | ~0% | Ranking on no-ground-truth problems | **built** |
| **4.** Offline agreement/regret experiment (§5) | 0% in-loop | The headline paper number | pending |
| **5.** LLM anomaly detector (§6) | +1 call/plan | Intent-versus-implementation warnings | pending |

**Recommended: 0, 3, 4 now; hold 1 and 2.** Option 0 captures most of what the bake-off was for at no
cost. One further argument against option 2: `benchmark/compare.py` scores structure on accuracy and
cost, and inflating pipeline wall time by 25% weakens precisely the head-to-head the thesis depends
on.

---

## 10. Implementation — file by file

| File | Change | Section | Status |
|---|---|---|---|
| `verifylib/ranking.py` **(new)** | SQI components, weights, renormalisation, `rank_key` | §4 | **[X]** |
| `verifylib/cli.py` | `rank <workspace>` subcommand | §4 | **[X]** |
| `verifylib/kernel/score.py` | emit `certified` alongside `score` | §4 | **[X]** |
| `verifylib/kernel/metrics.py` | `certified`, `design_order`, `scheme_family`, `invariant_max_drift`, `invariant_worst_headroom`, `dynkin_z`, `dynkin_bar`, `solve_time_s` in `BLOCK_KEYS` | §4, §8a | **[X]** |
| `verifylib/kernel/invariants.py` | per-invariant `default_tol` in the registry; `headroom` / `worst_headroom` | §4 | **[X]** |
| `verifylib/kernel/{driver,sde/driver}.py` | surface the above; time the plain solves | §4, §8a | **[X]** |
| `benchmark/baseline_select.py` | import `verifylib.ranking`; local `rank_key` deleted | §4 | **[X]** |
| `benchmark/report.py`, `benchmark/run.py` | `compute_verdict` reads `certified` | §4 | **[X]** |
| `commands/conductor.md` | Phase 3 key; early-exit on `certified` | §4 | **[X]** |
| `commands/conductor.md` | family-diverse design-order dispatch | §8 | pending |
| `benchmark/rank_study.py` **(new)** | agreement + `digits_lost`; weight sweep | §5 | pending |
| `benchmark/blind_baseline.py` **(new)** | LLM-judge control arm | §5 | pending |
| `verifylib/kernel/metrics.py` | `oracle_derived` at point of computation | §7 | pending |
| `references/verification_manual.md` | §11/§22: stop deferring the relocation; fix §11's agreement formula | §4 | pending |

`verifylib/metrics.py` and `verifylib/kernel/score.py`'s paths in the first draft's table were wrong
(`verifylib/metrics.py` does not exist; the module is `verifylib/kernel/metrics.py`), and the draft
proposed `verifylib/ranking.py` without noticing that `baseline_select.rank_key` was already the
conductor's rule as code and should be extracted rather than duplicated.

---

## 11. Acceptance criteria

1. ~~On `pde_heat_2d`, the 4th-order plan is dispatched in the first wave.~~ pending (§8)
2. No plan is selected as `best_plan` while a plan with a strictly higher design order is `unrun`.
3. **`verifylib.ranking` produces a total order using no oracle-derived field.** ✅
4. **The three tied `pde_heat_2d` plans tie on accuracy and are separated by cost.** ✅ (9.25e-04 of
   a point apart on A; 3.8× apart on E)
5. **A component that could not be measured is never scored zero, and skipping a check cannot improve
   a plan's rank.** ✅ (`test_ranking.py`)
6. **Switching the early-exit and `VERIFIED_PASS` onto `certified` changes no frozen verdict.** ✅
   (`test_certified.py`, run against every record in `benchmark/results/`)
7. `benchmark/report.py` prints median and P90 `digits_lost` alongside top-1 and top-3. pending
8. The LLM control arm runs on the same frozen pools as the deterministic ranker. pending
9. Anomaly-detector output appears in reviews and never in a ranking key — enforced by the ranker
   reading only `verifylib` metrics. ✅ structurally

---

## 12. Provenance summary

| Element | Source |
|---|---|
| Multi-plan architecture; parallel dispatch; winner early-exit; plateau early-stop | **[A]** |
| Lexicographic Phase 3 ranking, including provenance as key 2 | **[A]** |
| Cross-plan consensus as a *certifying* second circularity break | **[A]**, §4c — **not** relocated |
| Oracle-free selector concept and the negative result that motivates it | **[B]** |
| Sanitizer, alias maps, batch provenance, rank-serialization contract | **[B]** |
| Judge-output schema validation; anti-grid-density rule; abstention gate | **[B]** |
| Prohibited-token scan as a second leakage layer | **[B]** |
| Weighted quality index over the oracle-free metrics vector | **[X]** |
| Accuracy snapped to its own error bar, so ties are real ties | **[X]** |
| Order fidelity against the *design* rate, two-sided | **[X]** |
| Structure as headroom against each invariant's own tolerance | **[X]** |
| `solve_time_s` separated from `wall_time_s`; cost as a ranking key | **[X]** |
| `certified` separated from `score` | **[X]** |
| Regret in log space, floored at the harness noise margin | **[N]** |
| LLM judge as a measured baseline arm rather than a component | **[N]** |
| `oracle_derived` flag set at point of computation | **[N]**, reader shipped |
| Dispatch ordering by design order, family-diverse | **[N]** |
