# Oracle-Free Candidate Ranking — Implementation Plan

**Status:** proposal
**Scope:** how competing plans are ordered, how that ordering is measured, and what an oracle-free
ranker is actually for
**Related:** [plan-no-closed-form.md](plan-no-closed-form.md),
[plan-hallucination-guardrails.md](plan-hallucination-guardrails.md)

Provenance labels: **[A]** = already in this repo, **[B]** = adapted from the AutoNumerics
paper-release pipeline, **[N]** = new in this plan.

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

---

## 2. How this repo ranks candidates today **[A]**

1. `plan-creator-{pde,sde}` emits **2–4 competing plans**, dispatched concurrently
   (`parallelism = 3`).
2. Each plan runs its own solver↔evaluator loop and is scored **1–10 against the same reference**.
3. **Winner early-exit** — the instant any plan reaches 10, the loop stops; no further cycles launch.
4. **Plateau early-stop** — `iter >= 2` with no improvement marks a plan `stopped`.
5. `commands/conductor.md` Phase 3 ranks lexicographically:
   `score` ↓ → `estimated_rel_error` ↑ → order margin (`observed_order - order_floor`) ↓ →
   `wall_time_s` ↑ → fewest iterations.

**This is oracle-informed ranking.** Every plan arrives carrying a measured error. The paper-release
blind evaluator ranks with no numerical measurement at all. They are different operations, and its
25% top-1 is not a bar this repo is failing to clear — it is a floor on what is achievable with zero
numerical evidence.

### What the run data says

From `benchmark/results/results.json`, 16 problems:

| | |
|---|---|
| Problems with **multiple plans tied at 10** | **8 / 16** |
| Problems with plans left `unrun` (`iter: 0`) | **7 / 16** |
| Plans created / cycles executed / plans never run | 42 / 36 / **9** |

So the tiebreakers do the real work more often than not. And `pde_heat_2d` in full:

| Plan | Design order | Score | `est_err` | State |
|---|---|---|---|---|
| `1-fd-explicit-rk2` | 2 | 10 | 1.023e-04 | winner |
| `2-crank-nicolson-sparse` | 2 | 10 | **1.010e-04** | **selected** |
| `3-adi-peaceman-rachford` | 2 | 10 | 1.019e-04 | winner |
| `4-fourth-order-rk3` | **4** | 0 | null | **`unstarted`** |

Three second-order schemes hitting the same discretization floor, spanning **1.3%** — noise, not a
difference — decided on a 0.9% margin, while the one plan with a different asymptotic rate never ran.
At `N = 127` a 4th-order scheme should land near `1e-8` rather than `1e-4`.

**The weakness is not the ranking criteria.** Score 10 is a *threshold*, not a measure, so once
several plans clear it the ranking is deciding on quantities inside the noise floor — while the
early-exit discards the candidate most likely to win by orders of magnitude.

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

## 4. (a) Deterministic oracle-free ranker **[N]**

For problems with no ground truth — Kuramoto–Sivashinsky today, more as the benchmark grows — you
*must* rank without an oracle. This is not optional; it is the production path for that slice.

The merged ladder ([plan-no-closed-form.md](plan-no-closed-form.md)) already emits an oracle-free
measurement vector. Rank on it deterministically. No LLM involved.

### Revised ranking key

Two changes from the current Phase 3 order.

> **Note (guardrails plan §14 C4).** Tier A splits once the analytic-reference check exists, so the
> tier list below gains `analytic_unvalidated` between `analytic` and `surrogate`. Without it an
> unvalidated quoted formula outranks a validated surrogate. Take the ordered vocabulary from
> [plan-hallucination-guardrails.md](plan-hallucination-guardrails.md) §14 C4, not from this table.

```
1. score                        ↓
2. provenance tier              ↓     [N]  analytic > analytic_unvalidated > surrogate
                                          > manufactured > manufactured_partial
                                          > self_convergence > none
3. estimated_rel_error          ↑
4. observed_order - order_floor ↓
5. D1 term-balanced residual    ↑     [N]
6. max invariant drift          ↑     [N]
7. wall_time_s                  ↑
8. fewest iterations            ↑
```

**Why provenance moves to position 2.** Today a `self_convergence` estimate and an `analytic`
measurement compete in the same `estimated_rel_error` field — different quantities in one column. A
10 earned against a manufactured solution is a stronger claim than a 10 earned by self-convergence,
and the ordering should say so before it compares magnitudes.

**Why D1 and invariant drift enter.** They are the signals that separate plans which tie on a
threshold. On `pde_heat_2d` the three tied plans differ by 1.3% in error but would differ far more in
residual and in symmetry drift.

### Cross-plan consensus moves here **[A, relocated]**

§11 / §22 consensus leaves the scoring ladder and becomes a **tiebreak and anomaly signal**. When
three plans agree to `1e-4` and a fourth disagrees, that is information no single plan's evidence
contains. The existing rule holds unchanged: **corroborates, never certifies, never lifts a score.**

---

## 5. (b) The agreement / regret experiment **[N, using B's machinery]**

This is the valuable measurement, and it is where the paper-release machinery earns its keep — the
sanitizer, alias maps, batch provenance, and rank-serialization contract are all reusable.

### Protocol

For every problem that **has** ground truth:

1. Compute the **oracle rank** by true error from `benchmark/problems.py`.
2. Compute the **blind rank** from the ladder's oracle-free vector only (§4).
3. Report agreement **and regret**.

### Regret is the metric that matters **[N]**

```
regret = (true error of the blind pick) / (true error of the oracle pick)
```

Neither source system reports it, and it changes the interpretation completely.

Top-1 agreement is a brutal measure. On `pde_heat_2d`, picking any of the three tied plans gives
**regret ≈ 1.01×** — the choice is free. Top-1 agreement scores that a 33% success rate and calls it
a failure. Regret correctly calls it irrelevant, and points instead at the plan that was never run,
which is where the entire loss on that problem came from.

It also reframes the paper-release headline: **25% top-1 tells you nothing about whether the misses
cost 1.01× or 10⁴×**, and the answer completely changes what the number means.

Report all four:

| Metric | Definition |
|---|---|
| Top-1 agreement | Blind pick == oracle-best |
| Top-3 containment | Blind pick within the three lowest true errors |
| **Median regret** | Median of `err(blind) / err(oracle)` across problems |
| **P90 regret** | The tail — how bad the worst selections get |

### The baseline arm

Run the paper-release LLM judge as a **control condition**: same problems, same alias maps, same
protocol, plan-prose-and-metadata only. That yields a clean ablation:

> *"Oracle-free ranking from the verification battery achieves X% top-1 and median regret Y×,
> against 25% top-1 and median regret Z× from plan prose and execution metadata alone."*

This isolates the value of the **battery** from the value of the **judge**, and it is a considerably
stronger claim than either source system can currently make. It is also the natural headline result
for the no-closed-form work: it puts a number on how much the ladder is worth.

### Cost

Effectively zero in-loop. The experiment is read-only over frozen artifacts and can use a batch API
(the paper-release archive was built exactly this way, with `batch_provenance/` recording the
submission). Compute it once after a sweep; it changes nothing about how the sweep executes.

---

## 6. (c) LLM anomaly detector **[B, demoted]**

There is one thing a deterministic ranker cannot do: read a code summary against its plan and notice
*"claims WENO, has no limiter"* or *"plan says implicit, code has no linear solve."* That is
intent-versus-implementation pattern matching, and it is real value.

Keep it. Constrain it:

- Output is a list of **warnings**, not scores or ranks.
- Warnings are routed into the `<review>` text and `REPORT.md`.
- **Never** an input to the ranking key.
- Subject to the paper-release output-schema validation: named evidence sources only, no invented
  diagnostics.

Same rule already applied to consensus: corroborates, never certifies.

---

## 7. The `oracle_derived` flag **[N]**

The moment numerical evidence is fed to a blind ranker, the leakage scan has to run on **the evidence
itself**.

A GCI estimate is legitimately oracle-free — computed without ground truth. But `estimated_rel_error`
on the *analytic* path is oracle-derived, and it lives under the same key in the same metrics block.
The paper-release `PROHIBITED_TOKENS` list blocks `l2_error` and `relative_l2` for exactly this
reason, but a name blocklist is fragile: names drift, and a legitimate oracle-free field can share a
name with an oracle-derived one.

Fix: **tag every field the kernel emits at the point of computation.**

```python
metrics = {
  "estimated_rel_error": {"value": 1.01e-4, "oracle_derived": True},   # analytic path
  "gci_bound":           {"value": 3.2e-4,  "oracle_derived": False},  # Richardson
  "d1_residual":         {"value": 3.1e-6,  "oracle_derived": False},
  "observed_order":      {"value": 3.98,    "oracle_derived": False},
  "invariant_max_drift": {"value": 4.2e-12, "oracle_derived": False},
}
```

The blind bundle filters on `oracle_derived is False`. A flag set where the value is computed cannot
drift the way a name blocklist does. Retain the paper-release token scan as a **second** layer —
belt and braces, since they fail differently.

---

## 8. The dispatch-order fix — free, and it does most of the work **[N]**

`commands/conductor.md` sorts eligible plans by `(score asc, iter asc)`. At the start every plan is
`score: 0, iter: 0`, so that sort is a no-op and the pool fills in **plan-file order**. On
`pde_heat_2d` that put plans 1, 2, 3 in the first `parallelism = 3` wave and left the 4th-order scheme
unrun.

**Change the initial sort key to expected accuracy — design order descending, then stability class.**

The plan-creator already knows each scheme's design order: it is in `verification.mms_probe.expected_order`
and implicit in the plan slug. It simply is not used for scheduling.

Cost: **zero.** It is a scheduling change, not an extra cycle. The best plan enters the first wave; if
it reaches 10 first, the early-exit fires on the *best* plan rather than the first-listed one, and
every bit of the current savings is preserved.

### Why this matters more than the ranking itself

There is a genuine tension in the repo today:

- If the deliverable is **"a solver that meets the spec"**, the early-exit is correct and free. A 10
  means the 1% threshold cleared; further optimization is waste.
- If the deliverable is **"the best scheme for this problem"** — which `README.md` states as
  *"return the one with the lowest measured error"* and `conductor.md` Phase 3 as *"Best plan
  recommendation and why"* — then the early-exit does not support the claim when three plans tie and
  a fourth never ran.

Dispatch ordering resolves this for free by making "first to 10" much more likely to *be* the best.

---

## 9. Cost menu

From the 16-problem sweep: 3.02 h total, 42 plans, 36 cycles, 9 plans never run, **302 s per cycle**.

| Option | Cost | What it buys |
|---|---|---|
| **0.** Sort dispatch by expected order | **0%** | Best plan in the first wave; early-exit stops on it |
| **1.** Order-guard: do not finalize while a strictly-higher-order plan is `unrun` | ~+8% *(estimate; fires on perhaps 3 of 9 unrun plans — worst case degenerates to option 2)* | Catches the `heat_2d` case exactly |
| **2.** Full bake-off: one cycle for every unrun plan | **+25%** (+45 min per sweep) | Complete pool, defensible "best plan" claim |
| **3.** Deterministic oracle-free ranker (§4) | ~0% | Ranking on no-ground-truth problems |
| **4.** Offline agreement/regret experiment (§5) | 0% in-loop | The headline paper number |
| **5.** LLM anomaly detector (§6) | +1 call/plan | Intent-versus-implementation warnings |

**Recommended: 0, 3, 4 now; hold 1 and 2.** Option 0 captures most of what the bake-off was for at no
cost.

One further argument against option 2: `benchmark/compare.py` scores structure on **accuracy and
cost** (C0 / C1 / C2). Inflating pipeline wall time by 25% weakens precisely the head-to-head the
thesis depends on. A free scheduling fix does not.

---

## 10. Implementation — file by file

| File | Change | Section |
|---|---|---|
| `commands/conductor.md` | Initial dispatch sort by design order; revised Phase 3 ranking key including provenance tier, D1, invariant drift | §4, §8 |
| `verifylib/metrics.py` | `oracle_derived` flag on every emitted field | §7 |
| `verifylib/ranking.py` **(new)** | Deterministic lexicographic ranker over the metrics vector | §4 |
| `benchmark/rank_study.py` **(new)** | Offline agreement + regret experiment; oracle rank from `problems.py`, blind rank from `verifylib.ranking` | §5 |
| `benchmark/blind_baseline.py` **(new)** | LLM-judge control arm; sanitizer, alias maps, batch submission | §5 |
| `benchmark/report.py` | Report top-1, top-3, median regret, P90 regret; print blind-vs-oracle per problem | §5 |
| `references/verification_manual.md` | Move §11 / §22 consensus out of the scoring ladder into the ranking layer | §4 |
| `references/project_manual.md` | Document the revised ranking key | §4 |
| `agents/plan-creator-{pde,sde}.md` | Emit an explicit `design_order` field per plan so dispatch can sort on it | §8 |

### Phasing

| Phase | Deliverable | Gate to proceed |
|---|---|---|
| **1** | `design_order` on plans + dispatch sort | `pde_heat_2d` runs the 4th-order plan in wave 1 |
| **2** | `oracle_derived` flags in `verifylib.metrics` | No oracle-derived field reaches a blind bundle |
| **3** | `verifylib/ranking.py` + revised Phase 3 key | Kuramoto–Sivashinsky ranks its plans without ground truth |
| **4** | `benchmark/rank_study.py` — agreement + regret | Numbers reproduce over a frozen sweep |
| **5** | `benchmark/blind_baseline.py` — LLM control arm | Ablation table renders |
| **6** | Anomaly detector, advisory only | Warnings appear in `REPORT.md`, never in a score |

Phases 1–3 are production changes. Phases 4–6 are measurement and are entirely offline.

---

## 11. Acceptance criteria

1. On `pde_heat_2d`, the 4th-order plan is dispatched in the first wave.
2. No plan is selected as `best_plan` while a plan with a strictly higher design order is `unrun`
   (either it ran, or the guard is explicitly disabled in config).
3. `verifylib.ranking` produces a total order for Kuramoto–Sivashinsky using no oracle-derived field.
4. Every field in a blind bundle has `oracle_derived: false`; asserted, not assumed.
5. `benchmark/report.py` prints median and P90 regret alongside top-1 and top-3.
6. The LLM control arm reproduces something close to the published 25% top-1, confirming the protocol
   is comparable.
7. Anomaly-detector output appears in reviews and never in a ranking key — enforced by the ranker
   reading only `verifylib` metrics.

---

## 12. Provenance summary

| Element | Source |
|---|---|
| Multi-plan architecture; parallel dispatch; winner early-exit; plateau early-stop | **[A]** |
| Lexicographic Phase 3 ranking on score / error / order margin / time / iterations | **[A]** |
| Cross-plan consensus; corroborates-never-certifies rule | **[A]** §11, §22 |
| Oracle-free selector concept and the negative result that motivates it | **[B]** |
| Sanitizer, alias maps, batch provenance, rank-serialization contract | **[B]** |
| Judge-output schema validation; anti-grid-density rule; abstention gate | **[B]** |
| Prohibited-token scan as a second leakage layer | **[B]** |
| Deterministic ranking over the oracle-free metrics vector | **[N]** |
| Provenance tier as a ranking key | **[N]** |
| D1 residual and invariant drift as tiebreaks | **[N]** |
| Regret (median and P90) as the primary selector metric | **[N]** |
| LLM judge as a measured baseline arm rather than a component | **[N]** |
| LLM judge demoted to advisory anomaly detection | **[N]** |
| `oracle_derived` flag set at point of computation | **[N]** |
| Dispatch ordering by design order | **[N]** |
| Consensus relocated from scoring to ranking | **[N]**, relocating **[A]** |
