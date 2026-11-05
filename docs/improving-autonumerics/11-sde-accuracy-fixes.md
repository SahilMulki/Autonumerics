# 11 — SDE accuracy: the fixes, re-derived

**Status:** written 2026-11-04, replacing [10](10-sonnet-retest.md) §6's fix list; the four open
decisions settled 2026-11-05 and recorded in §9.1. Every claim below is measured by **re-running the
shipped kernel on the solvers already on disk** at the current commit (`c832572`), not inferred from
the run records. **Nothing here is implemented yet** — §7 is the order of work and §9 is what has to
be true before it starts.

The goal this list is written against: **AutoNumerics should be at least as accurate as a single
model draw on the same problem, and ideally better.** [10](10-sonnet-retest.md) §8 says it is not —
noise-floored, the pipeline's pool and three naked Sonnet draws tie at 0.99×, and Opus is 1.70× ahead
of both. §6 of that document proposed six fixes to close it. **Four of the six were wrong about the
cause**, and the reason is in §1.

---

## §0 How this was measured, and how to reproduce it

One command per plan, against the solver the run graded:

```bash
uv run python verifylib/cli.py evaluate workspace/{slug}/plans/{id}-{plan_slug} --json
```

Re-run for all three plans of `sde_hhw_call_corr` (B06), `sde_sabr_cev_call_i` (B01),
`sde_sabr_cev_call_ii_lowvol` (B10), and plan 1 of `sde_heston_call_feller` (B03). Rankings are
recomputed by calling `verifylib.ranking.rank_key` on the kernel's **own JSON output**, so the
ranker sees every field the kernel produced — which is exactly the difference this document is about.

**One caveat up front, and it is not small.** B03 plan 1's recorded `<metrics>` block does not
reproduce. The run recorded `estimated_rel_error: 5.942e+04`, `resolved: false`, `score: 6`; the same
solver at the same commit today gives `estimated_rel_error: 1.213e-03`, `resolved: true`, `score: 8`.
The 5.942e+04 is the *variance* error against a reference paired to the wrong state component
(component 0's variance against component 1's reference value); today the reference resolves
`variance: [1]` correctly. The code is unchanged and the working tree is clean, so the difference is
in `workspace/`, which is gitignored and may have been edited after the run. **Treat B03's recorded
kernel numbers as unreliable.** Its *harness* numbers come from an independent grader and stand. §5
carries this as the first open item, because a run whose records do not reproduce cannot be used to
calibrate anything.

---

## §1 The finding that reorganises the whole list

**The `<metrics>` block is a whitelist, it is the ranker's only input, and four of the ranker's
inputs are not on it.**

`verifylib/kernel/metrics.py` renders the block from `BLOCK_KEYS`:

```python
for key in BLOCK_KEYS:
    if key not in flat:
        continue
```

`BLOCK_KEYS` has 38 entries. **None of them contains the string `functional`.** And
`cli.py rank` → `review.parse_metrics(SOLUTION.md)` → `{"summary": block}` → `ranking.rank_key` is
the only path by which a measurement reaches selection. Cross-referencing the keys `ranking.py`
actually reads against the whitelist:

| ranker input | in the block? | what is lost |
|---|---|---|
| `functional_rel_error` | **no** | `accuracy` on every problem with a declared functional |
| `functional_rel_tol` | **no** | the tolerance that error is measured against |
| `order_band` | **no** | the one-sided `order` branch — **including the fix [10](10-sonnet-retest.md) §6 proposed** |
| `kind` | **no** | the SDE branch of `order`, except where `scheme_family` happens to be in `_SDE_FAMILIES` |
| the other 15 | yes | — |

So `ranking._functional_fraction` returns `None` on **every plan, always, by construction**, and
`accuracy` falls through to the state-moment error — which then abstains via `noise_limited` or reads
something meaningless. F3 and F12 were implemented in the kernel and have never once reached the
ranker.

**This is not a BENCHOP artefact.** It fires on any SDE whose problem statement names a quantity of
interest with a tolerance, which is most of them: a price, a hitting probability, an expected
functional, a passage time. The state moments the ranker falls back on are a proxy at best and on this
suite they are worse than that — the recorded `estimated_rel_error` across the thirty plans is either
`8.47e-16` (a martingale identity satisfied by construction, carrying no information about anything),
or `3.9e+07` (a moment the problem's own notes say does not close), or a legitimate moment error
drowned in sampling noise. **In none of the thirty cases does it read the quantity the problem is
scored on**, which the kernel had computed and dropped.

### What it costs, measured

Re-ranking from the kernel's full output instead of the block:

**B06 (`sde_hhw_call_corr`) — the 15.7× miss, recovered outright.** The functional error ranks the
pool *perfectly* against truth:

| plan | `functional_rel_error` | bias / sampling split | `accuracy` | sqi | true harness error |
|---|---|---|---|---|---|
| 1 EM-FT antithetic | **0.000910** | 3.36e-05 / 8.76e-04 | 0.2591 | **0.1134** | **0.0474** |
| 3 EM-FT + CV | 0.002276 | 9.34e-04 / 1.34e-03 | 0.0600 | 0.0262 | 0.5017 |
| 2 QE + CV | 0.002994 | 1.61e-03 / 1.38e-03 | 0.0004 | 0.0049 | 0.7445 ← shipped |

Monotone, in the right direction, and the winner becomes the pool's best plan. Note that `order` is
*unchanged* — still 0.0000 / 0.0000 / 0.0152, still rewarding the worse-converging plan — and it no
longer matters, because `accuracy` at weight 0.35 outvotes it 0.2591 to 0.0004.

**B01 (`sde_sabr_cev_call_i`) — the 2.3× miss, recovered.** Pool ties on score 9/9/8;
`accuracy` becomes 0.1087 / 0.1859 / 0.2972 and `rank_key` ships
`2-euler-maruyama-absorbing-antithetic` at **0.0473** instead of `1-exact-besq-time-change` at
0.1108. `decided_by: sqi`.

Against [10](10-sonnet-retest.md) §6's counterfactual table, B06 + B01 is a **1.43×** gain on the
shipped geomean (0.1693 → 0.1181) and takes the suite from 1.16× behind a single Opus draw to
**0.81× ahead of it**.

**The functional error is a much better signal than the moment error, and it is not an oracle.**
Across the four pools re-ranked here it orders B06 perfectly, changes which plan ships on B06 and
B01 in both cases to the pool's best, leaves B07 unchanged (`decided_by: score`, §6), and on B10
inverts. Two concrete limits worth knowing before leaning on it:

- **Its resolving power is coarse.** B07's two score-3 plans read `functional_rel_error` 0.0012 and
  0.0013 — a tie — while their true errors differ by 4.5× (0.1760 against 0.0388). A
  self-convergence estimate measures how far a solver has stopped moving, which bounds the error
  without locating it.
- **It can invert**, and §2 is the two measured reasons why. Both are estimator defects with
  specific fixes, not an argument against the signal.

Nothing above is a reason to withhold Fix 1: the component it replaces carried *no* information on
any of the thirty plans.

### Fix 1 — put the ranker's inputs on the block, and add the test that keeps them there

1. Add `functional_rel_error`, `functional_rel_tol`, `functional_name`, `functional_bias_rel`,
   `functional_se_rel`, `functionals_ok`, `order_band` and `kind` to `BLOCK_KEYS`.
2. **Add a test that fails when `ranking.py` reads a summary key `BLOCK_KEYS` does not carry.** This
   is the fix that matters more than the eight names, because the bug is structural: the ranker and
   the block are two lists that must agree and nothing made them agree. It recurs every time a kernel
   field is added — it has already happened twice (F3's functional fields, D2's `order_band`).
3. `functional_bias_rel` and `functional_se_rel` are needed on the block for Fix 4 even though
   `ranking` does not read them.

Why this is general, not a benchmark patch: it restores the component that measures *the quantity the
problem asked about* on every problem that declares one. No part of it names a model, a payoff or a
tolerance.

---

## §2 The functional error is not trustworthy everywhere — two measurable reasons

Fix 1 recovers B06 and B01. It does **not** recover B10, and the reason generalises in a way worth
fixing before leaning on the functional error.

B10 (`sde_sabr_cev_call_ii_lowvol`), re-ranked: `functional_rel_error` 0.0127 / 0.0193 / 0.0270 for
plans 1 / 2 / 3, against true errors 0.4628 / 0.1612 / 0.1783. **Anti-correlated**, and all three
exceed their own tolerance of 0.0100 while the independent harness passes all three at a sixth of the
gate. The estimate is both over-conservative by ~10× and wrongly ordered. Two distinct causes.

### Fix 2a — the Richardson denominator explodes as the observed order goes to zero

`functionals.estimate_error` infers an order from the ladder and divides by `2**p - 1`. On B10's
plan 3:

```
d21 = 6.906e-05,  d10 = 7.333e-05   ->  p = log2(d10/d21) = 0.0866
denominator = 2**0.0866 - 1 = 0.0619
error estimate = 6.906e-05 / 0.0619 / 0.0500 = 0.0223      (16.2x inflation)
```

Plan 3 holds **the tightest ladder in its pool** — `d21` is 7.3× smaller than plan 1's 5.062e-04 —
and is reported as the pool's *worst* plan, because its differences have stopped shrinking. Flat
differences at a small absolute level mean converged, not diverging. The estimator already guards the
extreme case (`if d21 <= ROUNDOFF_FLOOR * scale: return 0.0`) and nothing guards the common one.

**Fix (settled 2026-11-05):** require a meaningfully positive inferred order before trusting the
denominator; below it, fall back to `d21 / scale`, which the function's own docstring already names
as the safe bound — *"the difference stands in for the error, which overstates it for any convergent
scheme and is the safe direction."*

**The floor is derived from the plan's declared `design_order`, at half of it**, not set to a
constant. A bare 0.5 reproduces the effect on this suite — it moves B10's plan 3 from last (2.70) to
first (0.34) in the pool ordering — and is exactly the kind of number that goes silently wrong on a
first-order scheme, where half the design rate is 0.5 and the floor would reject a correctly
converging ladder. Where no design order is declared, fall back to a constant and say so in the
detail block rather than inferring one.

This is general: any solver that converges well on a short ladder, or that carries variance
reduction tight enough to flatten the differences, is currently punished for it. It is the opposite
of the behaviour wanted.

### Fix 2b — exact-sampling components must abstain in the functional ladder, as they already do in the order study

An exactly-sampled transition carries **no discretisation error at any `dt`**, so its ladder
differences are sampling noise, not bias. F18 recognised this and made the CRN order study abstain on
declared `exact_sampling_components`. **`functionals.run_declared` does not consult the field at
all.** Measured: B10's plan 2 declares `exact_sampling_components: 0` in its frontmatter and is still
charged a ladder bias of 0.0135 — 2.3× its tolerance — on a leg that cannot have one.

B07 (`sde_hhw_call_zerocorr`) shows the same thing more starkly. Its plan 1,
`1-exact-chi2-gaussian-rate`, declares `exact_sampling_components: 0, 1, 2` — **every** component
exactly sampled, so it carries no discretisation error anywhere — and the ladder charges it a bias of
**0.0130** against a tolerance of 0.003, which drives `accuracy` to 0.0000. The independent harness
grades that same solver at **0.0780**, i.e. 7.8% of its gate. The bias term is wrong by roughly fifty
times, in the direction that condemns the plan.

Thirteen of the thirty plans declare the field, so this is not a corner case; it penalises exactly the
family of schemes that is most accurate, which is the opposite of what Fix 4 is trying to encourage.

**Fix (settled 2026-11-05):** `run_declared` takes the declared components and abstains on the bias
term, leaving the sampling term to stand alone.

**Scope: abstain whenever *any* state component is exactly sampled.** This is the conservative cut,
it is what `crn.py` already does, and it is a few lines. The precise version — abstaining only for
functionals that actually *depend* on an exactly-sampled component — is correct but requires knowing
which components each functional reads, and nothing declares that today. Ship the conservative cut
and **record the narrower version as a known approximation** in the detail block, so a plan whose
functional reads only discretised legs is visibly over-abstained rather than silently so. Abstention
costs the bias evidence; it does not fabricate a pass.

While there: the field is spelled two ways on disk, `exact_sampling_components: 0, 1` and
`exact_sampling_components: [0, 1]`. Accept both or reject one loudly; silently parsing one of them
as empty would reintroduce the whole defect.

### And a figure to retire: B10 was never a selection miss

[10](10-sonnet-retest.md) §6 counted B10 as a 1.1× miss and put it in the counterfactual. Its noise
bar is **0.170** and the gap between shipped and pool-best is 0.1783 − 0.1612 = **0.017** — ten times
inside the bar. There is nothing there to fix, and chasing it would be fitting to a seed. The real
misses are B06 (15.7×), B01 (2.3×) and B07 (2.0×, architectural — see §4).

---

## §3 `order` penalises converging better than required

Kept from [10](10-sonnet-retest.md) §6 Fix 1, but the diagnosis there was wrong and the fix as
written would not have fired.

B06's three schemes measured `observed_order` ≈ 1.0 against a declared floor of 0.5, and
`ranking.order`'s two-sided branch scored the plan **furthest below** 1.0 highest:

```
plan 1: 1 - |1.0078 - 0.5| / 0.5 = -0.0156  ->  clipped to 0.0000
plan 2: 1 - |0.9924 - 0.5| / 0.5 =  0.0152   <- wins
```

[10](10-sonnet-retest.md) §6 proposed "default `order_band` to one-sided when `order_floor` was
defaulted". **That rule is already implemented** — `crn.py`'s D2 sets `strong_band` to one-sided
whenever the floor's source is the kernel default, and the driver copies it to
`summary["order_band"]`. It does not fire on B06 for two independent reasons: B06's spec *declares*
`expected_strong_order: 0.5` and the plan declares `scheme_family: euler-maruyama`, so D2 correctly
reads the floor as declared and keeps the band two-sided; **and** `order_band` is not in `BLOCK_KEYS`,
so even a one-sided verdict would never have reached the ranker. §10's "the fix exists and has never
once executed" was right about the symptom and wrong about why.

### Fix 3 — one-sided unconditionally, for ranking

For a *quality ranking*, over-performance should cap at 1.0 and never be charged. The module's own
comment already states the principle — *"§19's [rule is] that an order above the theoretical one is
noise rather than a bonus"* — and a floor is a pass bar, not a target to hit on the nose. The
legitimate worry behind two-sidedness, that an order far above design means the ladder is outside its
asymptotic range, is a *diagnostic* and is already handled elsewhere: `richardson.guards`' `p_sane`
caps the score. A rank key should not re-litigate it.

This is cheap, needs no agent change, and fixes every plan already on disk. On B06 it is now
redundant with Fix 1 — `accuracy` outvotes `order` there — but it removes a component that is
actively anti-correlated with quality on any problem where the declared floor sits below the scheme's
real rate, which is the common case for additive and near-additive noise.

---

## §4 Generation: the pool has to contain a good solver

Selection fixes cannot reach B03. Even with per-functional cherry-picking the pool's ceiling is
0.2679 against **0.0178** for one Opus draw and **0.0579** for the best of three Sonnet draws. The
pool simply did not contain the technique.

### The mechanism, from a within-current-code comparison

B03 (`sde_heston_call_feller`) and B08 (`sde_heston_call_hivar`) are both Heston with the Feller
condition violated, both run under the same code, and **both specs declare the same structured
hook**: `verification.terminal_law = {"law": "noncentral_chi2", ...}` on the variance component.

- **B08's pool contains `1-exact-chi2-broadie-kaya-cv`** — exact non-central χ² transition, plus a
  control variate.
- **B03's pool is Euler full-truncation, Andersen QE, and EM antithetic.** No exact χ² draw, no
  conditioning on the variance path.

The difference between the two is not the structured field, which is identical. It is the
*free-text* `implementation_notes`:

- B08: *"Exact V transition (noncentral chi-squared / Andersen QE) avoids sqrt of negative variance
  and a clipping bias"* and *"The variance process is independent of S, so one set of V paths serves
  all three spots."* — the scheme and the conditioning, spelled out.
- B03: entirely about verification provenance — martingale identities, the moment ladder,
  `expected_provenance`. Nothing about scheme choice.

**So plan quality is currently a lottery on whether the formulator happened to write scheme advice
in prose.** `verification.terminal_law` — the field `plan-creator-sde.md`'s rule 0 is supposed to key
on — was present in both and decided neither. The pipeline's own work-precision table confirms what
that cost on B03: 500k → 2M paths made the error **worse**, 0.003647 → 0.004019. The estimator is
bias-dominated and no path count can reach it.

### Fix 4 — make the structured field an obligation, and make non-compliance visible

1. **Where `verification.terminal_law` is declared for a component, at least one plan in the pool
   must sample it exactly** and declare `exact_sampling_components` for that component. Today this is
   a recommendation in prose competing with other prose.
2. **Have the gate say so when no plan does.** `cli.py gate` already reads the spec and all three
   plans; a pool that leaves a declared closed-form transition unused is a *pool* defect and should
   be reported as one, the same way an undeclared exact-sampling scheme is reported today.
3. **Require one variance-reduction-by-conditioning plan where the state admits one.** Conditional
   Monte Carlo (Rao–Blackwellisation) — integrating a sub-vector out in closed form given the rest —
   appeared in **zero** of the thirty plans and in C0's best B03 draw, where it removed the entire
   asset-leg sampling variance and is responsible for a 37× gap on the out-of-the-money strike. The
   triggering condition is structural and readable from the declared drift and diffusion: one
   component's diffusion depends only on the *other* components, so conditionally on their path it
   has a closed-form law. The kernel already builds a symbolic generator from those same two
   expressions for the Dynkin check, so the machinery to detect it exists.

4. **Give the orphaned structural facts a consumer, in this same change.** `test_spec_field_coverage`
   (§9.2) is currently red because three fields are declared in live SDE specs and read by nothing:
   `verification.structural_facts.barrier`, `verification.structural_facts.absorbing_boundary` and
   `verification.expected_cross_check`. The first two are the *same defect this fix is about*, one
   problem class over: a declared structural fact that decides a scheme's accuracy, with no consumer
   and no obligation attached. A barrier or absorbing problem needs its crossing probability handled
   — a Brownian-bridge correction, or an exact absorption treatment — and several plans on this suite
   do precisely that, unprompted and unchecked. Where `barrier` or `absorbing_boundary` is declared,
   at least one plan should treat the crossing explicitly, and the gate should say so when none does.
   Filing this as test maintenance would miss that it is Fix 4's own rule applied to a second field.

None of this names Heston, a non-central χ², an option or a barrier. The conditions are "a declared
transition law exists", "a component is conditionally tractable given the others" and "the problem
declares a boundary the payoff is sensitive to" — all properties of a declared SDE.

### Fix 5 — measure pool diversity, because the pools are smaller than they look

Four of the ten pools contain a **bit-identical pair**:

| problem | pair | `observed_order` | `estimated_rel_error` |
|---|---|---|---|
| B02 | plans 1 & 3 | 0.4862 | 1.600e-15 |
| B06 | plans 1 & 3 | 1.0078 | 3.648e-06 |
| B07 | plans 2 & 3 | 1.0036 | — |
| B10 | plans 1 & 3 | 0.5445 | 8.473e-16 |

In every case the pair is *scheme X* and *scheme X plus a control variate or antithetic pairing*.
Under the CRN ladder a variance-reduction wrapper does not change the pathwise dynamics, so every
statistic the kernel computes on them is identical to the last digit. **A three-plan pool is
effectively a two-plan pool, and nothing detects it.**

This closes a loop with Fix 1. The *only* field that distinguishes such a pair is
`functional_se_rel` — the sampling error of the graded quantity, which is what a control variate
actually buys — and that is one of the fields the block drops. So the pipeline currently proposes
variance reduction, builds it, and then ranks on a set of statistics in which it is invisible. On
B06, plans 1 and 3 differ only there (8.76e-04 against 1.34e-03) and that difference is what orders
them correctly.

**Fix (settled 2026-11-05):** have the conductor flag an identical-statistics pair as a pool defect,
and ask the plan creators for a third *scheme*, not a third *estimator*, where one is available.

**No new spec field** (§9.1 decision 4). A declared `variance_reduction:`, so the kernel could see
what was attempted, was the obvious instrument and is the wrong one. `functional_se_rel` — which Fix 1 puts on the
block — *measures* what a variance-reduction plan buys, and a measurement beats a declaration for
anything the kernel can observe directly. The declared-never-inferred rule exists for facts the
kernel must **trust** (a closed form, an exact transition, an invariant), not for quantities it can
compute. So the duplicate-pair signature is read straight off the statistics already in hand:
**identical `observed_order` and `estimated_rel_error` with differing `functional_se_rel`** is
exactly "same scheme, different estimator", and needs nothing the formulator has to write.

---

## §5 The refine loop gives the wrong instruction on a bias-dominated plan

The kernel already splits the functional error into its two terms — `functional_bias_rel` (what `dt`
costs) and `functional_se_rel` (what the path count costs). F19's retry instruction is unconditional:
*"the retry's single instruction is to raise `num_paths`"*.

On a bias-dominated plan that is the wrong advice, and B03 is the measurement: 500k → 2M paths made
the error worse. More paths tighten the second term and cannot touch the first.

### Fix 6 — branch the retry on the split

| measured | diagnosis | instruction |
|---|---|---|
| `se ≫ bias` | sampling-limited | raise `num_paths`, or add variance reduction |
| `bias ≫ se` | discretisation-limited | reduce `dt`, or move to a scheme with less bias (an exact transition where one exists) |
| comparable | both bind | both, and say so |

Measured splits available today: B06's plan 1 is 3.36e-05 / 8.76e-04 — strongly sampling-limited, so
more paths is right there; its plan 2 is 1.61e-03 / 1.38e-03 — balanced; B03's plan 1 is
1.2524e-03 / 1.2504e-03 — balanced, and raising paths alone cannot get it under a 0.002 gate. The
same split is what should be quoted in REPORT.md in place of a bare error, and it is a far more
actionable thing to tell a solver than a score.

This needs Fix 1 (the two fields on the block) and is strictly more useful once Fix 2a and 2b make
the bias half trustworthy.

---

## §6 What was dropped from [10](10-sonnet-retest.md) §6, and why

| old fix | verdict |
|---|---|
| Fix 1, one-sided `order` when the floor was defaulted | **Superseded.** The rule is already implemented in `crn.py` D2 and does not fire on B06 for a reason §10 did not identify. Replaced by Fix 1 (put `order_band` on the block) + Fix 3 (one-sided unconditionally) |
| Fix 2, minimum measured weight for the index | **Demoted to hygiene.** It was written because `accuracy` abstained on B06 and `order` ranked alone. With Fix 1, `accuracy` is measurable on every functional-graded plan and the single-component case largely disappears. Admitting `invariant_max_drift` into `structure` is still worth doing — it ranked B06 perfectly — but it is no longer an accuracy fix |
| Fix 3, grade accuracy on the declared functional | **Right conclusion, wrong cause.** It said `functional_rel_error` "is not being written on these problems". It *is* written, on every plan, and `render_block` drops it. Fix 1 is the same intent with the actual cause |
| Fix 4, plan diversity | **Kept and sharpened** into Fix 4 and Fix 5, with the B03-vs-B08 mechanism and the duplicate-pair measurement replacing the appeal to what C0's draws happened to do |
| Fix 5, per-functional best-of-pool (1.52× on B03) | **Removed, deliberately.** It is a benchmark-metric exploit: it works only because the grade is a max over functionals, and it ships an answer no single plan computed. The ask here is fixes that help solve SDE problems in general, and assembling one number per strike from three different solvers is not one. [10](10-sonnet-retest.md) §6 asked for this to be ruled out in writing rather than left unnoticed; this is that |
| Fix 6, reporting and hygiene | **Kept unchanged**, still with no accuracy effect: split B01's three collapsed generator-failure causes, close `reference.py`'s `_moment_observables` pairing gap, checksum before `override_agent_models` restores, investigate the `observed_order` bias |
| "Certification will not improve accuracy" | **Still true.** `certified` is not a rank key. Unchanged |
| "Provenance will not improve accuracy" | **Still true, and now better evidenced.** It separated no pool and was uniform in all four misses. B07 remains the one miss `score` causes — re-ranked from the kernel's full output it still reports `decided_by: score`, 6 against 3. **Settled (§9.1 decision 1): the key order stays, and the question re-opens after Fix 2b.** An accuracy-first key measures 1.30× *worse* over the four re-ranked pools, picks the pool best on neither B07 nor B10, and B07's motivating gap is inside both plans' noise bars |

---

## §7 Order of work

| # | fix | cost | expected accuracy effect |
|---|---|---|---|
| **0** | persist the kernel's `--json` per plan (§9.2) | small | none directly — it is what makes every row below verifiable by diff instead of by re-running solvers |
| **1 + 2a + 2b** | ranker inputs on `BLOCK_KEYS` + the consistency test; derived Richardson floor; exact-sampling abstention. **One change set** | hours | **1.43×** on the shipped geomean; recovers B06 and B01 outright; flips the suite to 0.81× against a single Opus draw. 2a removes a 16× estimate inflation and 2b a ~50× spurious bias, both of which Fix 1 would otherwise promote into the rank key |
| **3** | `order` one-sided unconditionally | hours | removes an anti-correlated component; redundant on B06 after Fix 1, not redundant in general |
| **6** | branch the refine retry on bias vs sampling; settle what `resolved` tracks | hours | stops spending a solver cycle on paths when `dt` is what binds |
| — | re-measure B07, then re-open §9.1 decision 1 | free | decides whether the score/accuracy conflict is real |
| **4** | `terminal_law` and the orphaned structural facts as obligations + gate report + a conditioning plan | agent + gate work | the only fix that can reach B03, where the pool ceiling is 23× behind one Opus draw |
| **5** | pool-diversity check off the existing statistics | small | makes three plans worth three plans |
| — | test maintenance: the `UNREAD` dead exemptions, and the load-dependent work-precision flake (§9.2) | small | none; it restores a trustworthy green baseline |

**1, 2a and 2b ship together** (§9.2), and with 3 and 6 are version-controlled Python with no agent
change — every one testable against the plans already in `workspace/`, with the re-runs in §0 as the
acceptance test. 4 and 5 need agent and gate changes and a fresh pipeline run to evaluate.

**The honest expectation.** Fixes 1–3 and 6 are selection and measurement; [10](10-sonnet-retest.md)
§7 put the whole remaining shipped-vs-pool gap at 1.55×, so that is roughly the ceiling they can
reach — enough to pass a single draw, not enough to pass best-of-three. Beating a model draw outright
needs Fix 4: the pool has to contain a solver as good as the one the model writes unaided. That is
the harder half and the one with no measured gain attached to it yet.

## §8 Open items

1. **B03's kernel record does not reproduce** (§0). Find out why before using any run's kernel
   numbers to calibrate a threshold. If `workspace/` can drift under a completed run, the records
   are not evidence.
2. **The `observed_order` bias.** Six harness-passing solvers read 0.2688–0.3226 against a 0.5
   floor ([10](10-sonnet-retest.md) F11). Fix 2a is the same estimator in its functional form, so
   these may share a cause worth finding once.
3. **`resolved` is computed on the moment error**, not on the functional. With Fix 1 the functional
   is the graded quantity; `resolved` should follow it, and F19's gate reads `resolved`.
4. ~~Whether to require a `variance_reduction:` declaration.~~ **Settled**: no new field, see §9.1
   decision 4.
5. **Whether `score`'s primacy survives a working accuracy component.** §9.1 decision 1 defers this
   to a re-measurement of B07 after Fix 2b. It is the one open *design* question here.
6. **Three documents have now failed to reproduce against `workspace/`**: [09](09-implementation.md)
   §4's test baseline (§9.2), [10](10-sonnet-retest.md) §6's claim that `functional_rel_error` was
   not being written (§1), and B03's kernel record (§0). Each was accurate when written and each was
   invalidated by a gitignored corpus turning over underneath it. That is the standing argument for
   Fix 0, and a reason to date every measurement in this folder and name the artefact it came from.

---

## §9 Before any of this is implemented

### §9.1 Four decisions, settled 2026-11-05

**1. Score keeps its primacy over accuracy — deferred, not affirmed.** The question was whether a
large accuracy gap should outvote a score gap, which is B07: `decided_by: score`, 6 against 3,
shipping 0.0780 over a 0.0388. The decision is **leave the key order alone and re-open the question
after Fix 2b**, for three measured reasons.

*First, reordering does not fix B07 and does cost B01.* An accuracy-first key, run on the four pools
re-ranked here:

| pool | scores | score-first ships | accuracy-first ships | |
|---|---|---|---|---|
| B01 | 9, 9, 8 | **0.0473** ← pool best | 0.0592 | worse |
| B06 | 3, 3, 3 | **0.0474** ← pool best | 0.0474 | same |
| B07 | 6, 3, 3 | **0.0780** | 0.1760 | worse |
| B10 | 2, 2, 2 | 0.1783 | 0.1783 | same |
| **geomean** | | **0.0747** | **0.0969** | **1.30× worse** |

Identical on the two pools where score ties, worse on both where score separates. On B07 the
accuracy component ranks the genuinely best plan (0.0388) *second*, so accuracy-first hands the win
to plan 2 at 0.1760. **The pool best is picked by neither ordering**, which is the tell that the key
order is not what is broken.

*Second, B07's motivating gap is not resolved.* Its shipped plan and its pool best are **both below
their own noise bars** — 0.0780 against 0.1892 and 0.0388 against 0.1632 — and B01's 0.0592-vs-0.0473
sits inside a bar of 0.437. So the 1.30× above is itself unresolved. The honest statement is that
there is no evidence *for* accuracy-first and what evidence exists points against it. Reordering a
rank key on a difference smaller than its own error bar is the overfitting §9.4 warns about.

*Third, B07's accuracy reads 0.000 for a reason Fix 2b removes.* Plan 1 declares all three components
exactly sampled and the ladder charges it a bias wrong by ~50× (§2). That is not the accuracy
component losing an argument to score; it is the accuracy component being broken on exactly the plan
in question. Fix it, then re-measure B07 — if the conflict survives a working component, the design
question is real and can be re-opened on evidence better than one unresolved pool.

**What this decision does not concede:** the instinct that accuracy is under-weighted is correct.
The cause is the component being broken and structurally invisible (§1, §2), not the key order.
Fixes 1–3 give accuracy 0.35 of a working index where it currently has none.

**2. The Richardson order floor is derived from `design_order`, at half of it** — not a constant.
Recorded in §2's Fix 2a with the reason: a bare 0.5 works here and breaks on a first-order scheme.

**3. Fix 2b abstains whenever *any* state component is exactly sampled** — the conservative cut,
matching `crn.py`, with the per-functional version recorded as a known approximation. Recorded in
§2's Fix 2b.

**4. No `variance_reduction:` spec field.** `functional_se_rel` *measures* what variance reduction
buys once Fix 1 puts it on the block, and a measurement beats a declaration for anything the kernel
can observe directly. Declared-never-inferred is for facts the kernel must trust, not for quantities
it can compute. Recorded in §4's Fix 5, which now reads the duplicate-pair signature straight off
the existing statistics.

### §9.2 Five things to do first, in this order

**Fix 0 — persist the kernel's own JSON next to each plan.** `cli.py evaluate --json` produces
everything in §1 and §2; `render_block` keeps 38 whitelisted keys and the rest is discarded at the
end of the evaluator's turn. Nothing on disk records what the kernel measured. That is why answering
"did F3 ever fire?" cost ten solver re-runs rather than a `grep`, and why B03's discrepancy (§0)
cannot be attributed. Writing `evaluation.json` into the plan directory is cheap, is the same root
cause as Fix 1 — information thrown away at a reporting boundary — and should land first so that
every fix below can be verified by diffing instead of re-deriving.

**Land Fix 2a and 2b with Fix 1, not after it.** Fix 1 promotes the functional error into the rank
key; 2a and 2b fix its two measured failure modes. On this suite Fix 1 alone regresses nothing —
B07's spurious bias is masked by `score`, and B10's inversion is inside its own noise bar — but
shipping a known-broken estimator into the selection path is not worth the two days of separation.

**Move the block's three homes together.** `verification_manual.md` §2 says the block's fields are
specified *once*, in `project_manual.md` — *The `<metrics>` Block* — and emitted by `BLOCK_KEYS`;
it says so because the two copies had already drifted once. Adding eight keys therefore touches:

| file | what changes |
|---|---|
| `verifylib/kernel/metrics.py` | `BLOCK_KEYS` |
| `references/project_manual.md` | the block's field specification |
| `agents/evaluator-sde.md` | the `<review>` template, so the functional error is what gets reported |
| `commands/conductor.md` | Phase 3's ranking read and REPORT.md's required lines, which currently name `estimated_rel_error` and `resolved` |

**Decide what `resolved` tracks.** It is computed on the moment error, and F19's retry gate reads
it. Once accuracy is the functional's, `resolved` is gating on a different quantity than the one
being ranked. This has to be settled with Fix 6, not after it.

**The test baseline has drifted from what [09](09-implementation.md) §4 records, and the three
failures need triage first.** Measured 2026-11-05, `uv run pytest verifylib/tests benchmark/tests`:

```
3 failed, 1070 passed, 3 skipped, 15 deselected in 649.21s
```

(The 15 deselected are `-m 'not slow'` from `pyproject.toml`'s `addopts`, as intended.) None of the
three is the failure 09 §4 predicted, and that document's own account is now wrong in both
directions:

| test | 09 §4 says | actually |
|---|---|---|
| `test_review::test_no_false_positives_on_archived_reviews` | "still red, by design" | **green** (22 passed) |
| `test_work_precision::test_an_sde_sweep_moves_both_axes_...` | repaired | **flake** — fails under full-suite load on `[0.0136, 0.1079, 0.0457, 0.1439]`, passes 3/3 in isolation. The tolerance 09 §4 added, `max(0.05, 0.25 * earlier)`, is not enough for sub-100 ms solves under CPU contention. Not a regression; the test is timing-dependent and should be made load-independent or marked slow |
| `test_spec_field_coverage::test_no_sde_spec_field_is_declared_and_unread_without_a_reason` | repaired | **red again**, and doing its job: three fields are declared in live SDE specs and read by nothing — `verification.structural_facts.barrier`, `verification.structural_facts.absorbing_boundary`, `verification.expected_cross_check` |
| `test_spec_field_coverage::test_the_unread_table_names_only_fields_some_live_spec_declares` | not mentioned | **red.** Four `UNREAD` exemptions name fields no live spec declares any more: `feller_condition`, `num_paths_replicate`, `num_paths_secondary`, `replication_check` — the last three being exactly the replication-gate fields 09 §3 recorded with an argument written out. They have aged out of the corpus |

Both `spec_field_coverage` failures are corpus drift rather than code regressions, and both are
cheap. But the first one is **substantive for Fix 4**: `barrier` and `absorbing_boundary` are
declared *structural facts* about the problem that nothing in the kernel reads. A barrier problem
needs its crossing probability handled — a Brownian-bridge correction or an exact absorption
treatment — and several plans on this suite do exactly that, unprompted and unchecked. That is the
same defect Fix 4 names for `terminal_law`: a structured field stating something that decides the
accuracy of a scheme, with no consumer and no obligation attached. It should be fixed in the same
change, not filed as test maintenance.

The corrected baseline to judge the implementation against is therefore **1070 passed with three
known failures, one of them load-dependent** — and §9.3's check matters more than it looked, because
`test_review` is a *passing* test to protect rather than an already-broken one not to worsen.

### §9.3 One risk checked and cleared

Adding fields to `BLOCK_KEYS` is not inert for `review.py`. Its drift check partitions a review's
claims by whether the block *carries* the key:

```python
drift     = [c for c in stray if c[0] in metrics]      # counted as a review defect
uncarried = [c for c in stray if c[0] not in metrics]  # counted as a gap in the block
```

So adding a key reclassifies any archived review that cites it from "gap in the block" to
"transcription drift" — and `test_no_false_positives_on_archived_reviews` counts only the first
kind, over every `SOLUTION.md` in `workspace/`. **Checked: zero keyed citations of the eight new
fields appear inside any archived `<review>` section**, so the change will not fire it. The
module's own comment documents this trap from the last time it was hit, which is the reason to
re-check rather than assume after the evaluator template starts quoting the functional error.

### §9.4 Not blockers — do not wait on these

- **A fresh pipeline run.** Fixes 1, 2a, 2b and 3 are deterministic re-rankings of plans already on
  disk; §0's re-runs are the acceptance test and cost compute, not credits. A run is needed only to
  evaluate Fix 4 and Fix 5, which change what gets generated.
- **The four unresolvable problems.** B01, B04, B05 and B10 cannot separate any arm
  ([10](10-sonnet-retest.md) §8); they are not evidence for or against any fix here.
- **Certification and provenance.** Still no accuracy leverage, for the reasons in §6.
