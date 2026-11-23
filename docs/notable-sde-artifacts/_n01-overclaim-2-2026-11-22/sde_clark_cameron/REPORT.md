# sde_clark_cameron — Final Report

**Problem type:** SDE, family `clark_cameron` — the canonical Clark & Cameron (1980)
counterexample. 2 state components, **2 independent noise channels**:

```
dS = mu dt + dW2,   S(0) = 1
dU = S  dW1,        U(0) = 0          mu = 1,  T = 1
```

**Evidence path:** A (closed form exists). All four plans were graded against exact moments.

---

## Provenance of the scoring — read this first

**`analytic`, `reference_outcome: validated`** for all three evaluated plans. This is the strongest
tier: the closed-form moments were not merely quoted, they were validated against the spec's own
declared generator and moment hierarchy. The exact values are

| moment | exact |
|---|---|
| `E[S(T)]` | 2 |
| `Var[S(T)]` | 1 |
| `E[U(T)]` | 0 |
| `Var[U(T)]` | 17/6 = 2.83333… |
| `E[S(T)U(T)]` | 0 |

The formulator derived these from first principles (S is exactly integrable and normal; U is an L²
martingale, so `Var[U]` follows from the Itô isometry) rather than recalling them, and they are
backed three independent ways: the exactly-closing moment ODE, hand-checked generator images
(`L(S²) = 2 mu S + 1`, `L(U²) = S²`, `L(SU) = mu U`), and a 40 000-path simulation.

One qualification the reader should carry: the kernel's reference check pairs only against the
scalar moment keys, which here are **S**'s moments. `U`'s `17/6` rests on the same exactly-closing
hierarchy plus the numerical cross-check — strong evidence, but not the half the kernel itself
certified.

---

## Ambiguities — where the solved problem may differ from the one described

Two requirements were recorded `ambiguous` by the formulator. Both are real gaps in `problem.md`,
not formulator hedging.

**R12 — which noise channel is which.** Quote: *"a numpy array of shape `(1000, 25, 2)` holding the
Brownian increments, **already scaled by sqrt(dt)**, whose last axis indexes the 2 independent noise
sources"*. `problem.md` never says that index 0 is W1 and index 1 is W2. This matters more than a
labelling convention would: the grade is **pathwise**, so swapping the channels produces a *wrong*
terminal state against the same reference path rather than a distributionally equivalent one. The
formulator encoded the only natural reading — **channel 0 = W1 (drives U), channel 1 = W2 (drives
S)** — and every plan implements it. If the harness's convention is the opposite, every plan here is
wrong by the same amount, and nothing in this run would detect it. The one probe that bears on it is
the Dynkin test function `X[:,0]*X[:,1]`, which catches a solver that *correlates or swaps* the
channels; it passed on all three plans, and it is the function that came closest to its bar on all
three (see Verification gaps).

**R21 — the problem's actual accuracy gate is not in the spec vocabulary.** Quote: *"the harness
gates the relative strong error `sqrt(mean((X_num - X_exact)^2)) / sqrt(mean(X_exact^2))` over paths
at < 2%"*. This is a scoring rule, so it belongs in `evaluation_thresholds` — but there is **no
SDE-side key for a pathwise strong-error tolerance**: the accuracy keys are all moment-space
(`variance_rel_err_max`, `mean_rel_err_max`) or solver-reported scalars (`functionals`), and the
pathwise strong error is neither, since the solver does not report it. The formulator declined to
repurpose `rel_l2_err_max`, which would have been read in moment space. What carries the requirement
instead is the pair of strong-order floors (0.5 for a `dW`-only scheme, 1.0 with the Lévy area) plus
the 2% figure verbatim in `implementation_notes`.

**This gap shapes the result below and is the single most important thing in this report.** The
quantity the problem is scored on was never measured by the pipeline. Everything that was measured —
moments, strong order, Dynkin, invariants — is a proxy for it.

---

## A spec defect was found and fixed mid-run

The first grading round returned `score: 9`, `certified: false` for **all three** plans, for the
identical reason: `dynkin_ok` came back `unavailable`, so the generator check never reached a
verdict. Under the SDE rubric that caps every plan on the problem at 9 and leaves it uncertified,
because nothing else can establish that the paths came from *this* SDE — exact terminal moments
cannot substitute, since two different processes can share two moments.

Three independent evaluators traced it to the spec and attributed it to the string-valued
`initial_condition.value`. On re-dispatch the formulator found that diagnosis was wrong on the
mechanism — the kernel's Dynkin reader *does* resolve symbolic names against `parameters` — and that
the real cause was **location**: `dynkin.initial_moments` reads `spec.get("initial_condition")` at
the **top level**, and the field had been declared under `verification`, where nothing reads it. With
no top-level key the reader fell through to a branch that accepts only `X_0`/`Y_0`/`Z_0`, found
`S_0`/`U_0`, and returned `None`.

The field was moved to the top level as explicit floats `[1.0, 0.0]` — mathematically identical to
what it already said. `check-spec` still exits 0, no plan directory was modified, and the Dynkin
check now reaches a verdict and **passes**. All three plans were then re-evaluated on
**byte-identical solvers** (sha256 verified unchanged before and after) and all three moved to 10 /
`certified: true`. The jump owes nothing to any solver change.

---

## Plans

Four plans were created; three were solved and evaluated.

### 1-levy-area-milstein-trapezoid — score 10, certified

- **Scheme:** `milstein`, `levy_area_treatment: sub-increments`, design order 1. S takes its exact
  step `S + mu*dt + dW[:,n,1]`. U takes `U + S_n*dW1 + mu*(dt/2)*dW1 + I12`, where `I12` is the
  iterated integral assembled exactly from `dW_sub` by an exclusive prefix sum of the W2
  sub-increments contracted against the W1 sub-increments. The `mu*(dt/2)*dW1` term is a trapezoidal
  surrogate for the drift-noise integral `I_{0,1}`, which cuts the remaining per-step residual
  variance from `dt³/3` to `dt³/12`.
- **Provenance:** `analytic` (validated). **Error: measured, not estimated** (`error_is_estimate:
  false`).
- `estimated_rel_error: 0.0114` — `variance_rel_err` 0.0114 on U, `mean_rel_err` 3.675e-03, against
  tolerances of 0.10 and 0.05.
- **Observed order: 0.9343 ± 0.0351** against a floor of 1.0 (one-sided band), `order_check: true`.
- `mc_se_rel: 0.0150`, `resolved: true`. Dynkin z = 2.203 vs bar 2.531, replicated over 3 seeds,
  median z 1.91.
- `invariant_max_drift: 0.0186`, worst headroom 0.157 decades (the `martingale` check on U).
- **iter: 2** (one solve, two evaluations — the second on the corrected spec).
- `solve_time_s: 1.663`.

### 2-levy-area-milstein — score 10, certified

- **Scheme:** `milstein`, `levy_area_treatment: sub-increments`, design order 1. The canonical scheme
  exactly as the spec's `milstein_correction` states it: `I_{1,2}` only, no further correction. It is
  the independent confirmation that plan 1's extra term is a refinement rather than the whole effect.
- **Provenance:** `analytic` (validated). Error measured, not estimated.
- `estimated_rel_error: 9.803e-03` — the **lowest moment error in the pool**. `mean_rel_err`
  3.675e-03.
- **Observed order: 0.9827 ± 0.0307** against a floor of 1.0 — the closest to its design rate of the
  two order-1 plans.
- `mc_se_rel: 0.0150`, `resolved: true`. Dynkin z = 2.203 vs bar 2.531, median z 1.91 over 3 seeds.
- `moment_bias_binding: variance`, `moment_bias_frac_tol: 0.2131`, `moment_bias_actionable: true` —
  a real, resolved `dt` bias in `Var[U]` worth 21% of its own tolerance (graded 2.843 against a
  Richardson limit of 2.904, ≈2.1% of the value). It comes from the deliberately omitted
  `mu*I_{0,1}` term — which is precisely the term plan 1 adds.
- **iter: 2**. `solve_time_s: 1.762`.

### 3-euler-maruyama-control — score 10, certified — **the ranker's pick**

- **Scheme:** `euler-maruyama`, design order 0.5. `S += mu*dt + dW[:,n,1]` (exact);
  `U += S_n*dW[:,n,0]` with the left-endpoint Itô integrand. `dW_sub` is accepted and **deliberately
  ignored**.
- **Purpose, from its own SOLUTION.md:** this is the pool's *declared control*. It exists to measure
  the Clark–Cameron `c·√dt` lower bound rather than assert it, and it was **numbered last so that a
  tie could not ship it**. Its author expected it to miss the problem's 2% pathwise gate by
  construction (predicted 5.1–8.4%).
- **Provenance:** `analytic` (validated). Error measured, not estimated.
- `estimated_rel_error: 0.0162` — the **highest moment error of the three**. `mean_rel_err`
  3.675e-03. `E[U]` was not graded, because its exact value of 0 falls under the near-zero skip.
- **Observed order: 0.4965 ± 0.0348** against its declared floor of 0.5 — the √dt bound, reproduced
  to within one standard error. This is the plan working exactly as designed.
- `mc_se_rel: 0.0149`, `resolved: true`. Dynkin z = 2.226 vs bar 2.531, median 1.942 over 3 seeds.
- `moment_bias_binding: variance`, `moment_bias_frac_tol: 0.3874` — a `Var[U]` `dt` bias of about
  −3.9% at `dt = 0.04`, the largest in the pool.
- **iter: 2**, plus the Phase 2b cycle below. `solve_time_s: 1.379`.

### 4-conditional-mc-projection — **unfinished, never solved**

Written in response to a pool-gate warning (below). It Rao-Blackwellises U as far as the pathwise
contract permits: U's increment is replaced by its exact conditional expectation given the resolved
S/W2 sub-path, `U_{n+1} = U_n + (S_n + mu*dt/2 + B̄_n)*dW1_n`. It is provably L2-optimal in the
`dW1`-only class and was also expected to miss the 2% gate (predicted ≈2.95%), hence its position
last. **It was never dispatched to a solver**: the three plans above all certified first and the
winner early-exit fired. It has no score and no measurements — nothing in this report rests on it.

---

## Best plan

The ranker's verdict, verbatim:

```
best: 3-euler-maruyama-control   decided_by: sqi
note: the index above is over order, structure, consistency only. accuracy did not resolve on
every plan in this pool, and a component no comparison was made on does not decide one --
see ranking.comparable_keys.
```

| plan | score | cert | provenance | sqi | acc | ord | str | res | solve_s |
|---|---|---|---|---|---|---|---|---|---|
| 3-euler-maruyama-control | 10 | yes | analytic | **0.494** | – | 0.993 | 0.057 | 0.120 | 1.379 |
| 2-levy-area-milstein | 10 | yes | analytic | 0.490 | – | 0.983 | 0.053 | 0.130 | 1.762 |
| 1-levy-area-milstein-trapezoid | 10 | yes | analytic | 0.467 | – | 0.934 | 0.052 | 0.130 | 1.663 |

`best_plan` is recorded as **3-euler-maruyama-control**, because that is what the ranker — the same
`verifylib/ranking.py` the baseline study uses — returns, and the conductor does not rank by hand.

**But this pick should not be read as "the most accurate plan", and the reader needs to know why.**
Three things line up against it, and all three are visible in the table above:

1. **All three plans are tied on everything the ranker treats as a claim about what was
   established** — `score: 10`, `certified: true`, `provenance: analytic`. The separation is
   entirely the quality index, and the index's spread is 0.494 / 0.490 / 0.467: a 0.027 gap decided
   the pool.
2. **Accuracy dropped out of the comparison.** The `acc` column is empty for *every* plan: each
   plan's moment error (0.0114 / 0.0098 / 0.0162) sits below its own Monte-Carlo error bar
   (`mc_se_rel ≈ 0.015`, with a variance standard error of ≈0.043), so `ranking.accuracy` returned
   None under F16 and `comparable_keys` correctly restricted the index to order, structure and
   consistency. The component that would have separated these plans on *how right they are* was not
   measurable at 16 000 paths, so it did not vote.
3. **What remained — order fidelity — is measured against each plan's own *design* rate, and the
   control has the easiest target.** Plan 3 hits 0.4965 against a design rate of 0.5, for a fidelity
   of 0.993. Plans 1 and 2 aim at 1.0 and reach 0.9343 and 0.9827, for 0.934 and 0.983. The control
   wins the deciding component by being the scheme that promised least and delivered it, not by
   converging faster: its *actual* convergence rate is half of its rivals'.

Put together with R21: the quantity the problem is actually gated on — relative pathwise strong
error < 2% — is unmeasurable in the current spec vocabulary, and on the plans' own derivations
**the ranker's pick is the one expected to fail that gate** (≈5.1–8.4%), while plans 1 and 2 are
expected to clear it (≈0.41–0.69% and ≈0.82–1.37%). The pool's author anticipated exactly this and
numbered the control last so a tie could not ship it; the tie was not broken by a tie-break but by
a quality component on which a half-order scheme scores well.

**Recommendation for anyone consuming this result rather than the ranking machinery:
`2-levy-area-milstein` or `1-levy-area-milstein-trapezoid`.** Both certify at 10 with the same
`analytic` provenance, both converge at order ≈1 where the control converges at 0.5, and plan 2 has
the lowest measured moment error in the pool (9.803e-03). Plan 1 is the more accurate scheme by
construction — it adds the trapezoidal `I_{0,1}` surrogate that plan 2 omits, and its `Var[U]` bias
is small enough that the kernel does not report `moment_bias_actionable` at all, where plan 2 carries
0.2131 of tolerance — but its measured order, 0.9343 ± 0.0351, sits further from the design rate of
1.0 than plan 2's 0.9827 ± 0.0307, which is what cost it the index. The honest statement is that
plans 1 and 2 are not separable on the evidence this run produced, and that both are separable from
plan 3 by a full half-order of convergence.

**The cheapest measurement that would settle it** is a solver cycle raising `num_paths` until the
moment error resolves above its own standard error — at which point `acc` re-enters the comparable
set and the index ranks on accuracy instead of on design-rate fidelity. That was not reachable
inside this run's rules: all three plans certified, and a certified plan is not eligible for a
further cycle except the single bias-removal pass in Phase 2b.

---

## Phase 2b — the winner's bias-removal cycle: **reverted**

It fired on `3-euler-maruyama-control`, which entered with `moment_bias_binding: variance` and
`moment_bias_frac_tol: 0.3874`.

**Outcome: reverted — the solver changed nothing, deliberately, and gave its reasons.** It named the
sub-step carrying the bias (U's increment, advanced with the single-interval `I_{0,1} ≈ ΔW·dt/2`,
which picks up the O(dt) Euler error measured at −3.9%) and argued it is the floor for this scheme
class: a higher-order U update requires the iterated integral within each step, that information
exists only in `dW_sub` and not in `dW`, and consuming `dW_sub` would (a) convert the control into a
duplicate of plan 2, (b) destroy the baseline that measures the `√dt` lower bound, and (c) risk
exclusion under R22, since assembling `I_{0,1}` exactly from `dW_sub` makes the update algebraically
identical to a full sub-grid Euler sum. It explicitly declined the cosmetic route of rescaling U's
variance toward the Richardson limit.

| | before | after |
|---|---|---|
| `moment_bias_binding` | variance | variance |
| `moment_bias_frac_tol` | 0.3874 | 0.3874 (unchanged) |
| `score` | 10 | 10 |

`solver.py` is byte-identical to the certified artifact (sha256 `9ecba458…` before and after), so
no restoration was needed and the shipped file is the one the score describes. **This is a result,
not a no-op:** the bias the kernel measured is real and resolved, and it is a property of the
`dW`-only scheme class rather than a defect the solver left unfixed. Reducing it means not being
this plan. Note that plan 2 carries the *same* kind of bias at 0.2131 and plan 1 carries none the
kernel judges actionable — i.e. the pool already contains the schemes that remove it, which is the
pool working as designed.

---

## Verification gaps

Things this run did **not** establish. Each bounds the result.

1. **The problem's own accuracy gate was never measured (R21).** The relative pathwise strong error
   that `problem.md` gates at 2% has no key in the SDE spec vocabulary and no plan reports it. It is
   carried only indirectly, by the strong-order floors. Every accuracy number in this report is a
   moment-space or order-space proxy for it. On the plans' own derivations the shipped plan is
   expected to miss it — see Best plan.

2. **Accuracy did not resolve on any plan**, and the ranker says so itself. Each plan's moment error
   lies below its own Monte-Carlo error bar: `estimated_rel_error` 0.0114 / 0.0098 / 0.0162 against
   `mc_se_rel ≈ 0.0150` and a `Var[U]` standard error of ≈0.043. Note what this is and is not: the
   kernel reports `resolved: true` on all three — the *variance estimate* separated from its
   reference in moment space, which is the claim `resolved` makes and which the score reads — while
   `ranking.accuracy` returned None because the *error* is not located inside its own noise. **No
   plan's accuracy was measured precisely enough to rank on, and none of these three numbers should
   be quoted as "the error" without its error bar.** More paths, not a better scheme, is what that
   would take.

3. **`zero_channel_cross_variation` was declared and never checked.** It appears in
   `invariants_not_reported` on all three plans. The spec labels it in its own note as declared but
   not checked — it is not a registry name and carries no `tol` — and its gateable form is the
   Dynkin test function `X[:,0]*X[:,1]`. A declared invariant the kernel has no evaluator for is a
   property this run did not establish, however clean `invariants_ok: true` reads beside it.

4. **Dynkin passed, but marginally, on all three plans, and on the function that matters most for
   R12.** First-seed z values were 2.203 / 2.203 / 2.226 against a family-wise bar of 2.531 — inside
   the kernel's 0.7 margin, which is why it replicated over 3 seeds and decided on the medians (1.91
   / 1.91 / 1.942). The binding test function in every case was `X[:,0]*X[:,1]`, which is precisely
   the probe for a solver that correlates or swaps the two noise channels — the R12 ambiguity. It
   passed, but it is the check to watch at any higher path count. On plan 2 the evaluator
   additionally reported that the **pooled** z of 3.03 disagrees with the median of 1.91; the rule
   decides on the median, and the evaluator recorded the disagreement rather than treating it as
   decisive. That disagreement is a gap in the evidence, not a failure.

5. **Dynkin cannot catch a mis-transcribed equation.** All three evaluators flagged this
   independently and it is worth repeating: the Dynkin check builds its generator from the **spec's
   own** drift and diffusion. If the formulator mis-read `problem.md`, every plan descends from the
   same wrong operator, all of them pass unanimously, and nothing downstream detects it. Combined
   with gap 1 and the R12 channel ambiguity, the common-mode exposure of this run is the spec, not
   the schemes.

6. **Pool gate warning, persisting (recorded at both pool checks, exit 0 — a warning, never a
   block).** The gate reports that state component 1 (U) is *conditionally tractable* — its own
   drift and diffusion are of degree at most one in it and no other component's coefficients read it
   — so conditionally on the driving components its law is closed form, and no plan integrates it
   out (none declares `exact_sampling_components: 1`). The plan-creator was re-dispatched once, as
   the protocol requires, and added `4-conditional-mc-projection`; it deliberately did **not** declare
   `exact_sampling_components`, and its argument is that the technique is **not fully dischargeable
   under this contract**: `E[U(T) | sigma(W2)] = 0` exactly, so returning it pathwise gives ≈100%
   relative strong error against the supplied-path reference (R6/R21); drawing `U(T)` from
   `Normal(0, ∫S²ds)` needs fresh randomness and violates R13; and more basically
   Rao-Blackwellisation removes the sampling variance of a path the *solver draws*, whereas here the
   path is supplied at 256× resolution and the grade is pathwise, so for the Lévy-area plans that
   variance is already zero. **The pool's ceiling, stated plainly:** only U's within-step W1 fine
   structure can be integrated out, not the component. Whether that reasoning is right is not
   something this run established — plan 4 was never solved — so the warning stands as an open gap.

7. **Plan 4 has no measurements at all.** It was left unfinished because three plans certified before
   it was dispatched. Its predicted numbers quoted above are its author's derivations, not results.

8. **One solver deviation from its plan text, on plans 1 and 2.** Both solvers report that when
   `dW_sub` is withheld they use the conditional-mean area `0.5*dW1*dW2` where their SOLUTION.md
   text specifies zero, and both say SOLUTION.md still says zero. This does not affect the graded
   runs, which supply `dW_sub`, but the plan text and the code disagree in that fallback.

9. **The kernel reduced the sub-grid below the declared resolution.** It cut `dW_sub` from the
   declared 256 sub-steps to 167 at 16 000 paths, and to 64 on the CRN order ladder, to fit its 1 GiB
   memory cap. This is a kernel memory decision rather than a plan defect, and plan 3 is unaffected
   since it ignores `dW_sub` — but the Lévy-area plans' graded runs were not made at the
   resolution their plans declare.

10. **Path count departs from the contract, by design.** The contract's `graded_num_paths` is 1000;
    the spec sets `num_paths: 16000`, because U's measured excess kurtosis of 1.32 puts the
    resolution floor near 10 600 paths and copying 1000 would have left every plan MC-inconclusive.
    `conv_num_paths` is 2000 to keep each CRN level's `dW_sub` near 0.2 GB. This is recorded as
    `mapped` in the ledger, not an ambiguity, but it means the graded path count is not the one
    `problem.md` names.

**Clean, for the record:** the requirements ledger carried all 24 entries with **zero `dropped`**;
`check-spec` exits 0 both before and after the mid-run fix; both pool gates exit 0; no
`GUARDRAIL_UNSATISFIED` file was written; no `agent_cap` was applied by any evaluator; and the
solver hash was verified unchanged across every grading pass, so no evaluator edited code it was
grading. `reported_moments_agree` is true on all three plans, confirming `ddof=1` is used
consistently. No plan recorded `resolved: false`.
