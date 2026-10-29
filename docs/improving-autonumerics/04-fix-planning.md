# 04 — Fixing the plan menu

**Status:** proposal, 2026-10-28; **revised the same day against [07-review.md](07-review.md)**,
which confirmed both defects by reading the code and withdrew the measured evidence for F6. Two
changes, F5 and F6. Lower leverage than
[03](03-fix-selection.md) on the measured record, but they are what stands between the pipeline and
the four problems selection alone cannot win.

---

## §1 The measured case

C0's best attempt per problem, by scheme family:

| scheme family | problems | count |
|---|---|---|
| exact transition law (non-central χ², BESQ, Bessel) | B01, B02, B03, B05, B10 | 5 |
| Andersen QE | B04, B07, B09 | 3 |
| conditional Black–Scholes + control variate | B08 | 1 |
| Euler–Maruyama, full truncation | B06 | 1 |

**C0 chose an exact transition or QE scheme on nine of ten problems.** It won seven and tied three.

The pipeline's planner is instructed, in writing, to propose Euler–Maruyama alone on every
multi-dimensional problem. All ten BENCHOP-SLV problems are multi-dimensional.

---

## F5 — Widen the SDE scheme menu past Euler–Maruyama {#f5}

**Effort: M.**

### What the rule says

[`agents/plan-creator-sde.md:21-29`](../../agents/plan-creator-sde.md#L21):

```
- **Additive noise OR multi-D** → 1 plan: Euler-Maruyama only
- **Multiplicative scalar noise** → 2 plans: Euler-Maruyama AND Milstein
- **Stiff** → document that implicit schemes are required but not yet supported;
  propose EM with a note on stability risk
```

[`references/sde_manual.md:166-172`](../../references/sde_manual.md#L166):

```
1. `noise_structure == "additive"` OR `state_dimension > 1` → EM only. Milstein adds
   no value for additive noise (dg/dX = 0) and is not well-defined for multi-D
   without Lévy areas.
2. `noise_structure == "multiplicative"` AND `state_dimension == 1` → propose both
   EM and Milstein.
3. `stiff == true` → implicit or semi-implicit scheme required. Not yet supported;
   document the limitation.
```

### What is wrong with it

The premise is correct and the conclusion does not follow. Multi-dimensional Milstein does need the
Lévy area ([`sde_manual.md:38`](../../references/sde_manual.md#L38)), so "no Milstein here" is
right. But the rule then says *"Use EM for all multi-D problems"*, which quietly equates **"Milstein
is unavailable"** with **"nothing but Euler is available."**

That is false. The entire family of schemes C0 beat the pipeline with is well defined in multiple
dimensions and needs no Lévy area, because none of them is a stochastic Taylor scheme:

| scheme | what it is | why multi-D is fine |
|---|---|---|
| **Exact transition sampling** | draw the CIR/BESQ variance from its exact non-central χ² law, then the asset conditionally | it is a sampler, not a discretization; no iterated integrals |
| **Andersen QE** | moment-matched quadratic / exponential approximation to the same transition | same |
| **Broadie–Kaya and its truncations** | exact conditional scheme for Heston | same |
| **Conditional / mixed schemes** | integrate one leg exactly (e.g. Black–Scholes conditional on realised variance) and simulate the rest | same |
| **Variance reduction** — control variates, antithetics, MLMC | orthogonal to the discretization altogether | — |

The rule's three branches also only cover the *discretization* axis. Path construction, variance
reduction and conditioning are not mentioned anywhere in the planner's instructions. There is no
control variate, no antithetic sampling, no multilevel scheme anywhere in the 24 plans the pipeline
wrote. C0 used a control variate on B08 and beat the pipeline 13×; the BENCHOP authors' own SDE
entries (MCA, MLMC, SGBM, mSABR) are built on precisely this axis.

### The rule did not actually block the good plan

Worth saying, because it changes the fix. The plan creator **improvised past its own rule in nine
of ten pools** and wrote QE, exact-transition, AKS and Milstein-variant plans anyway — 9 of the 24
plans are non-Euler. Opus read the rule, judged it wrong for the problem, and went around it.

So F5 is not "unblock a blocked planner." It is:

1. **Make the good plan legitimate** instead of an off-menu improvisation that has to misdeclare
   itself (F6).
2. **Make it reliable** instead of dependent on the planner choosing to disobey — which it will do
   less often at Sonnet than at Opus, and the next run is all-Sonnet.
3. **Stop putting Euler first by construction** in every pool ([F7](03-fix-selection.md#f7)).

### The change

Restructure the rule along two axes instead of one. Sketch:

```
### Step 1a: the discretization
- Additive noise OR multi-D  → Euler-Maruyama is the baseline plan
- Multiplicative scalar noise → Euler-Maruyama AND Milstein
- Stiff                       → drift-implicit / semi-implicit

### Step 1b: is the transition law known in closed form?
Read `analytic_moments`, the SDE's family, and `problem_spec.json`'s notes.
- A CIR / squared-Bessel variance leg → propose exact non-central-chi-squared
  sampling, or Andersen QE, as a SECOND plan. Neither needs a Levy area and both
  are well defined in any dimension; the multi-D restriction in Step 1a is about
  stochastic Taylor schemes only.
- A leg that can be integrated exactly conditional on another (log-Euler for the
  asset given the variance path) → propose the conditional scheme.

### Step 1c: variance reduction
Where the graded quantity is an expectation (`evaluation_thresholds.functionals`),
consider a control variate with a known mean -- the discounted asset, a
Black-Scholes price at the same parameters -- as a plan modifier. State the
control and its known expectation in the plan.
```

Keep the Lévy-area reasoning in the manual; it is correct. Add one sentence saying what it does
**not** rule out, since that is the inference the current text invites.

### Two constraints on how Step 1b and 1c are written

**[rev, against [07](07-review.md) §10]**

**1b must key off a spec field, not off the planner's recall of the finance literature.** On this
suite every problem is a known mathematical-finance SDE with a published exact transition law, so
"propose the exact transition" is sound advice. The instruction will apply to **every**
multi-dimensional SDE the system ever sees, most of which have no such law, and a planner that
proposes exact-transition sampling where none exists produces a plan the solver cannot write. Gate
it on something the formulator actually sets — `analytic_moments`, or a new `transition_law_known`
with a named law — in the same way Step 1a is gated on `milstein_eligible`.

**1c is blocked by the solver contract, not by the planner's menu.** The harness calls
`solve_sde(num_paths, dt, T, seed)` and grades `terminal_paths` moments plus `functionals`:

- **MLMC does not fit at all.** It has no single path population at one `dt`, so `terminal_paths`
  is ill-defined for it, and the two-level anti-hardcoding check — the same solver at two path
  counts, values required to differ — assumes a plain estimator.
- **A control variate fits the contract but is invisible to the ranker.** It changes the
  *functional*, not the terminal distribution, so `empirical_mean`, `empirical_variance`,
  `mc_se_rel` and `estimated_rel_error` are all unchanged by it. **A control-variate plan is
  invisible to every signal the ranker currently reads** — which is a sharper statement of M1 than
  M1 makes, and it is the scheme family C0 beat the pipeline with on B08.

So Step 1c must not land before [F0](03-fix-selection.md) and [F12](03-fix-selection.md#f12).
Widening the menu to variance reduction without extending the contract and the kernel produces
plans the system cannot grade, and they will rank *below* the plans they beat.

**Acceptance test.** Re-run the planner alone (no solver) on all ten BENCHOP `problem_spec.json`
files and count: at least one non-Euler plan proposed per problem, and at least one exact-transition
or QE plan on the five problems with a CIR/BESQ variance leg. This costs one plan-creator call per
problem and no solver time — it is the cheapest possible check that the rule change landed.

### What it will not fix

B02 and B03 are not scheme-menu failures. On both, the pool *contained* a non-Euler plan — AKS on
B02, QE on B03 — and the best plan in the pool was still 65× and 25× behind C0's best. Those are
implementation-quality gaps, and they are where a stronger solver model actually helps.

---

## F6 — Give `scheme_family` real values for QE and exact transitions {#f6}

**Effort: M.**

### The problem

[`agents/plan-creator-sde.md:62`](../../agents/plan-creator-sde.md#L62):

```
scheme:        euler-maruyama | milstein
scheme_family: euler-maruyama | milstein | tamed-euler-maruyama | tamed-milstein | implicit | other
```

No value exists for quadratic-exponential, exact transition sampling, or a conditional scheme. Every
improvised plan declared `other`.

[`FAMILY_ORDER_KEY`](../../verifylib/kernel/sde/crn.py#L221) maps `milstein` and `tamed-milstein` to
`expected_strong_order_milstein`. Everything else — `other` included — falls through to
`verification.expected_strong_order`, which the formulator writes as Euler's **0.5**. And because
the family *was* declared, `expected_orders` keeps the band **two-sided**: 0.5 ± 0.4.

So a plan that samples the variance from its exact law is judged against Euler–Maruyama's strong
order, two-sided, and is penalised for being better than it.

The plan creator saw this coming. From
`workspace/sde_sabr_cev_call_ii_lowvol/plans/2-qe-martingale-absorbing/SOLUTION.md`:

> **Expected strong order**: 0.5 (the spec's only declared floor; `scheme_family: other` selects it)

This is a vocabulary defect. Opus understood the problem, documented it, and had no field to say
the true thing in. **No model upgrade reaches it.**

> **[C, against [07](07-review.md) §7] This is a defect found by reading the code, and the suite
> does not measure what it costs.** ~~The plan quoted above was "104× more accurate and scored 2",
> and that is what declaring `other` cost it.~~ It scored 2 for unrelated reasons —
> `provenance: none`, `d1_outcome: unavailable`, `reference_outcome: unavailable`,
> `dynkin_z: unavailable` — and its plain-Euler sibling scored 2 for the same ones. **No plan on
> the suite is shown losing a point to the order band.** B07's QE plan is the only one scored down
> on an order-study outcome and its recorded reason is that the estimator is erratic in dt.
>
> F6 should still be done: the mechanism is real, `FAMILY_ORDER_REQUIRED` is the repo's own
> precedent for fixing it, and it is cheap. It should be argued from the code, not from a 104× it
> did not cause.

### The change

Three coordinated edits, and they must land together or the schema checks will fail:

1. **Extend the enum** in [`plan-creator-sde.md:62`](../../agents/plan-creator-sde.md#L62) with
   `quadratic-exponential`, `exact-transition` and `conditional`. Keep `other` as the honest
   fallback it is meant to be.
2. **Extend `FAMILY_ORDER_KEY`** so each new family selects its own declared floor — e.g.
   `expected_strong_order_exact`, `expected_strong_order_qe`. A scheme that integrates a leg
   exactly has no Euler strong order to measure; `unresolved` is the correct outcome and it already
   costs the plan nothing.
3. **Extend `FAMILY_ORDER_REQUIRED`** in [`verifylib/schema.py:487`](../../verifylib/schema.py#L487)
   with the matching gate so a spec that admits a QE plan must declare its floor — the same rule
   `milstein_eligible` already enforces, and for the same reason it was added:

   > Without this key the order study judges it against `expected_strong_order` — Euler-Maruyama's
   > floor — and a correct Milstein solver measuring 1.0 against a 0.5 floor is a two-sided miss of
   > 0.5.

   That sentence is already in the repo, describing this exact failure mode one family over. The
   fix is to apply it to the families that are missing.
4. **Give the formulator a rule** for when to declare the new floors, keyed off the same fields it
   already reads to set `milstein_eligible`. A floor nobody declares is an inference, and
   `expected_orders`' own rule (`crn.py:266`) is that an inference must never make a test stricter.

### Interaction with F1

These two fixes compound. `cli.py compare` reads `scheme_family` to decide whether an agreement is
*cross-family* and therefore a circularity break
([`cli.py:484`](../../verifylib/cli.py#L484)). Today a QE plan and an exact-transition plan both
declare `other`, so an agreement between them reads as same-family and **earns nothing** — the
strongest evidence the pool can produce is discarded on a naming technicality. Fix the vocabulary
and that agreement starts counting.

**Acceptance test.** A QE plan on a problem whose spec declares a QE floor measures its order and
is not failed for it; `cli.py compare` on B10's two plans reports `different_family: true`. Both
are checkable on the existing workspaces without a pipeline call.
