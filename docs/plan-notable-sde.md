# Notable SDE Problems — a Recognition Suite

**Status:** **drafted 2026-11-12; rev 1 2026-11-13; rev 2–3 2026-11-14 — H-N2 and H-N3
implemented, step 2 done for three of its four remaining entries.** No problem entry and no run yet.

**Rev 3 `[2026-11-14]` runs step 2** (§2b): the truth derivations for N05, N06 and N07, plus H-N2
(suite registration). Four of its six findings overturn the plan. **N06's premise was wrong** — the
"explicit solution" its route rested on is for a *telegraph-noise ODE*, proved with ordinary
calculus, and the Brownian model does not reduce; but the logit chart makes the noise **additive**,
which is a better truth route than the one assumed, and is also the technique the entry punishes a
solver for missing. **N06 is nearly vacuous at `T = 1`** — the mean moves 3.59% from `I0` against a
5% gate, so returning the initial condition passes. **N05's mean is exactly vacuous** (0.00e+00
sensitivity to all four jump and volatility parameters) and **its proposed control is not a
control** (0.28× the gate on the mean; 19× on the price, where it belongs). And **N07's
`(γ, ω)` is `(1, 4)`, not the `(2, 5)` the plan proposed**, because only 3 of 30 pairs clear all
three constraints at once and the plan's pair is near-stationary. N03 remains blocked on step 1's
parameters. Every truth route is validated against something independent — the generalised
Kolmogorov solver reproduces the shipped double-well truth to **2.8e-15**.

**Rev 2 `[2026-11-14]` ships H-N3** — §4a has the measured record. The SDE routes now run structure
diagnostics and a declared gate fails the run, which is the item rev 1 left as the only
record-corrupting one open. The gate separates a correct scheme from a one-decision-wrong one by
**34.5×**, the widest in the repo (BENCHOP's narrowest is 6.8×, SciML's 4.4×), and the statistic it
gates was **chosen by measurement**: the per-path amplitude drift, which is the stronger
mathematical statement, turns out to be the weaker instrument at 4.7×, because it charges a correct
but non-conservative scheme for its own `O(dt)` variance. Three pre-existing defects came out of
building it, and the middle one matters beyond this suite: **a solver returning the wrong number of
components passes the moment route** — an `(n, 3)` sample on a `state_dimension: 2` problem scores
`ok`/`passed: True` on all 24 SDE problems on record. It is recorded with its test and not fixed
here. 1343 tests pass; the fold is inert on all 75 entries and every manifest entry hashes
identically.

**Rev 1 `[2026-11-13]` revises the plan against the SciML rev 11–20 work, and withdraws one of its
own recommendations.** Three changes, in order of weight. **§2a is new**: SciML ended on one
unshipped candidate — the pool-relative order reading, measured at 7 of 9 pools and 0.419× against
the one-shot — which it declined to ship because settling it *"needs a corpus whose pools are
discretisation schemes rather than exact samplers"*, `order` being comparable on 0 of 9 BENCHOP
pools. This suite can be that corpus, but four of its ten entries have an exact sampler as the best
answer and would reproduce the blindness, so the ten are now split and the **order-bearing five run
first**. **H-N6 is withdrawn**: rev 1 of this plan told the owner to apply `score_cap` before the
first run; SciML rev 16 then retired that defect as a **false positive** — the `num_paths` it
objected to is the harness's own call parameter, which a solver cannot choose, and applying the cap
would have taken seven of nine entries from a harness-verified 10/10 to a 3. The real fault was the
attribution, fixed by a third severity (`findings.SPEC`) that reports without capping. **And one of
this suite's design risks was removed before it was run into**: the strong-order gate was two-sided,
capping plans for converging *faster* than their floor, which is the exact shape of N01's and N07's
correct schemes (§10, prediction 3). After rev 1 the only item in §4 still of the
record-corrupting kind is **H-N3**, re-verified open today. Every number below is either derived here and marked
`[derived]`, read from the shipped code and marked with its file and line, or named as a
measurement a step has to make and marked `[to measure]`. Nothing is quoted from recall without
saying so.

**The five `[derived]` truths are measured, not asserted.** N01's exact variance, N02's
Itô–Stratonovich factor, N04's moment hierarchy and N08's CIR transformation were each checked by
simulation, and N09's quadrature bias computed exactly, on 2026-11-12 —
[`notable-sde-data/derivations_check.py`](notable-sde-data/derivations_check.py), seeded, numpy and
scipy only, with the output quoted in its docstring. One draft claim was wrong and the check caught
it: N09's bias has the opposite sign to what this plan first said (§2, N09).

**Scope**, set by the request: ten SDE problems that a person who works with SDEs recognises on
sight, each with a citation an SDE referee will accept, to be run through the pipeline and reported
in the paper. Difficulty is explicitly *not* the selection criterion — recognition is. §0 is about
what that buys and what it costs, because the two are not the same thing and the difference is the
whole reason this document is longer than a list.

**Provenance labels** as the folder's convention ([README.md](README.md), "Conventions inside a
plan"): **[A]** already in this repo or the agent manuals, **[N]** new in this plan, **[W]** from
the literature, **[X]** to be built by this plan, **[derived]** worked out here and checkable by
hand.

---

## 0. Verdict

**Feasible, and cheaper than either external suite.** Nine of the ten entries need no new truth
route: the harness already grades exact terminal moments, a scalar functional at two path counts, a
discretization-free Monte-Carlo reference, a pathwise strong error against an exact solution, and a
stability criterion. Eight of the ten have truth that is *exactly* known — four of them by the two
machineries already in `problems.py` (a closed moment hierarchy integrated by matrix exponential,
and a Kolmogorov solve declared `exact`). One harness extension is needed and it is small (§4,
H-N1). One entry is expensive and is phase-gated out of the headline (N10).

Three things about this suite have to be said before the problem list, because they decide how the
paper may describe the result.

### 0a. Recognition and difficulty are different axes, and this suite buys the first one

The repo's existing SDE problems are selected for *mechanism*: a violated Feller condition, a
superlinear drift that makes Euler diverge, thirteen non-commuting noise channels, a singular
spring. That is the right axis for showing a pipeline earns its keep, and it is the axis
`benchmark/design/hard_sde_candidates_from_lit.md` was written on. It is not the axis a referee
reads. `multichannel_stiff_m13` is a harder problem than the Kubo oscillator and a weaker exhibit,
because the referee has met the Kubo oscillator and has not met M13.

So this suite optimises for the second axis and accepts the first one where it comes free. The
consequence is that **a clean sweep here is weaker evidence than a clean sweep on the tier-3 set**,
and the paper has to say so in the same breath as the number. What a clean sweep here *does*
establish is the thing a hard-problems-only result cannot: that the pipeline is not tuned to this
repo's taste in problems. Seven of the ten (§3) require a technique that appears nowhere in
`references/sde_manual.md`, which is where the difficulty turns out to live anyway — not in the
equations, but in the distance between each equation and the manual the agents are given.

### 0b. Notable means the answer may be recalled rather than computed — and BENCHOP's control does not transfer

This is the methodological risk, and it is sharper here than anywhere else in the repo. A famous
SDE comes with a famous answer. Merton's call price is a Poisson-weighted sum of Black–Scholes
values; the geometric-average Asian is Black–Scholes at `sigma/sqrt(3)`; the Kubo oscillator is a
rotation through `a t + sigma W(t)`. A model can produce the graded number on all three without
discretising anything.

The externally anchored suite solved its version of this with **shifted twins**: three problems
identical in model and payoff to a published sibling, with the initial condition moved far enough
to shift the answer by many multiples of its own gate, and the key computed here
(`benchmark/README.md`, "The externally anchored suite"). That control defends against a recalled
*number*. It does nothing here, because almost nothing here is graded against a published number —
the keys are all computed in `problems.py` — and the recall at risk is of a *formula*, which moves
with the parameters. **Twins are therefore not part of this design, and §7 says what is instead.**

The replacement is the machinery `plan-sciml-benchmark.md` already built for exactly this
situation: the two arms (`scheme-matched` and `unrestricted`, `benchmark/setup.py:521–574`) and the
degeneracy reading on the pathwise route that records whether a solver carried *any* discretization
error (`benchmark/verify.py:2255–2269`, `reason: "pathwise_exact"` / `"path_independent"`). On this
suite the reading is not an exclusion — recognising that the Kubo oscillator is a rotation is a
correct and good answer, and `sde_manual.md` rule 0 asks for exactly that. It is a *label*, and the
paper reports the two columns separately: how many entries were solved by integration, and how many
by recognising a closed form. A table that does not separate them is not measuring what its caption
would claim.

### 0c. The pipeline's own manual forbids the winning method on at least one entry

`references/sde_manual.md:38` reads, in full: *"Multi-dimensional Milstein requires the Lévy area
(the iterated stochastic integral ∫∫dW_i dW_j), which is not computable without approximation. Use
EM for all multi-D problems."* N01 is the problem that statement is about, and on N01 the statement
is a ceiling: Clark and Cameron proved that a scheme using only the increments cannot beat strong
order ½ **[W]**, and the way past it is to approximate the Lévy area, which the manual tells the
agents is not done. The suite will therefore measure a rule rather than a pipeline on that entry,
and that is a finding worth having — it is the same shape as the finding
`plan-benchop-slv.md` recorded when rule 1 said "EM only" ten times on a suite of ten
multi-dimensional problems and an untooled completion beat it on seven (`sde_manual.md:168–176`).
§10 predicts the outcome; §11 records what this plan deliberately does not do about it.

### 0d. Its own suite, for the reason the other two are

`setup.py:521` — `SUITES = ("main", "no-closed-form", "benchop", "sciml")`. Each carries its own
C0/C1/C2 baseline over its own denominator, and `apply_suite_filter` plus `guard_results_path`
(`setup.py:587–625`) exist because a sweep that silently joined two suites once overwrote a live
baseline. This becomes `"notable"`, the fifth. It must not join `main`: the main suite's rates are
quoted in the root README against a 49-problem denominator, and ten recognisable additions would
move every one of them while measuring something else.

---

## 1. Selection rule

A problem qualifies for this suite only if it passes all four:

1. **Named.** It has a name a referee will recognise without the equation — "the Kubo oscillator",
   "Wright–Fisher", "Merton jump diffusion" — or it is the canonical counterexample in its corner
   of the numerical-SDE literature (Clark–Cameron).
2. **Cited.** There is a specific paper or textbook section to cite, not a folklore attribution.
3. **Independently checkable inside this harness**, by one of the five routes `verify.py` already
   has, or by a route this plan builds and validates. A problem whose truth we would have to take
   from the pipeline is excluded by definition.
4. **Not already in the 66.** Where it overlaps an existing entry in *mechanism*, the overlap is
   stated in the entry rather than hidden (N07 against `sde_stochastic_oscillator` and
   `sde_ou_stiff`; N08 against `sde_cir_feller_violated`).

Two further rules shape the set rather than each entry:

5. **One mechanism per entry, ten different mechanisms.** The suite is worthless as evidence if
   eight entries punish the same decision. §2's table has a mechanism column and no value repeats.
6. **No entry may be answerable by a constant.** `validate_ground_truth.py:434–443` already refuses
   a pathwise entry whose terminal state does not vary across paths, after three of the nine SciML
   entries turned out vacuous at the horizon their source used (`plan-sciml-benchmark.md` §2a.1).
   The moment and functional routes have no such authoring-time guard, so §6 adds the check per
   entry by hand: the graded quantity must depend on the dynamics at the graded horizon, not be the
   initial condition, the stationary law, or zero.

---

## 2. The ten problems

| ID | Problem | Citation **[W]** | State × noise | Truth route | The decision it punishes |
|---|---|---|---|---|---|
| **N01** | Clark–Cameron system | Clark & Cameron 1980 | 2 × 2 | `pathwise` + exact variance | the Lévy area; the strong-order ½ barrier |
| **N02** | Kubo oscillator (Stratonovich) | Kloeden–Platen §4.4; Milstein–Repin–Tretyakov 2002 | 2 × 1 | `pathwise` (shipped route) + exact moments + invariant | Itô vs Stratonovich; amplitude conservation |
| **N03** | Aït-Sahalia interest-rate model | Aït-Sahalia 1996; Szpruch, Mao, Higham & Pan 2011, *BIT* 51:405–425 | 1 × 1 | Kolmogorov solve (`exact`) + stability | positivity at a repelling `1/x` drift |
| **N04** | Wright–Fisher / Jacobi diffusion with mutation | Kimura 1964; Jacobi-diffusion literature | 1 × 1 | exact — closed moment hierarchy | staying inside `[0,1]` when the boundary is attainable |
| **N05** | Merton jump diffusion | Merton 1976, *J. Financial Economics* 3:125–144 | 1 × (1 + Poisson) | exact moments + exact functional | the compensator; a jump channel at all |
| **N06** | Stochastic SIS epidemic | Gray, Greenhalgh, Hu, Mao & Pan 2011, *SIAM J. Appl. Math.* 71:876–902 | 1 × 1 | `reference` — MC of the explicit solution | staying inside `(0,N)` under a superlinear diffusion |
| **N07** | Underdamped Langevin, moderate friction | Leimkuhler & Matthews 2012/2013 (BAOAB) | 2 × 1 | exact — 2-D Gaussian by matrix exponential | invariant-measure / splitting bias |
| **N08** | 3/2 stochastic-volatility model | Ahn & Gao 1999, *Rev. Financial Studies* 12:721–762 | 1 × 1 | exact — CIR moments of `1/V` | a `V^{3/2}` diffusion; Euler's moments diverge |
| **N09** | Geometric-average Asian option | Kemna & Vorst 1990; Giles 2008 (MLMC) | 1 × 1 | exact functional (Kemna–Vorst) | the quadrature of a path integral |
| **N10** | Stochastic Duffing–van der Pol | Baxendale 1980s / Arnold; Kloeden–Platen §17 | 2 × 1 | `reference` — two method families | bistability with no closed form — **Phase 2** |

IDs `N01`–`N10` are free: existing prefixes are `S`, `P`, `B` and `M` (`problems.py`). Slugs are
`sde_clark_cameron`, `sde_kubo_oscillator`, `sde_aitsahalia_rate`, `sde_wright_fisher_mutation`,
`sde_merton_jump_call`, `sde_sis_epidemic`, `sde_langevin_baoab`, `sde_three_halves_vol`,
`sde_geometric_asian`, `sde_duffing_van_der_pol`, all with `"suite": "notable"`.

Each subsection below gives the equation, the parameters, the truth, the one-decision-wrong control
that calibration must fail (§5), and the vacuity check (§1 rule 6). Parameters marked `[to fix]`
are chosen in step 2 from the cited source rather than invented here, because a parameter set a
referee can look up is worth more than one that suits the gate.

### 2a. The split that makes this suite answer SciML's open question `[rev1 2026-11-13, N]`

This section was added after `plan-sciml-benchmark.md` reached rev 20, and it changes how the suite
should be run rather than what is in it.

SciML's step-10 work ends on one unshipped candidate. Its §10m prices three fixes for the ranking
defect that cost M05 a 622× regret; two are withdrawn as measured-harmful, and the third — the
**pool-relative order reading**, which judges a plan's strong order against the best rate anyone
demonstrated on that problem rather than against a floor every plan clears — takes the arm to
**7 of 9 pools and 0.419×** against the one-shot. It is **recorded as the strongest candidate the
project has measured and deliberately not shipped**, for a reason that is structural and stated as a
measurement:

> `order` is in the comparable set of **8 of 9** sciml pools and **0 of 9** holdout ones (every
> holdout pool holds an exact sampler with no order to measure) … settling it needs a corpus whose
> pools are discretisation schemes rather than exact samplers.

**This suite can be that corpus, but only if it is built to be.** Four of the ten entries have an
exact sampler as their best answer — which is precisely the property that makes BENCHOP structurally
incapable of settling the question. If those four dominate, the notable suite inherits BENCHOP's
blindness and the rule stays unpriceable.

So the ten split in two, and the split costs nothing but a label:

| | Entries | Why `order` is comparable, or not |
|---|---|---|
| **Order-bearing** — every plausible plan is a discretisation | **N01, N03, N04, N06, N07**, and N10 | No exact transition law is reachable. N01 *is* a strong-order problem by construction; N03, N04 and N06 have no samplable law at all; N07's splitting schemes differ precisely in order of invariant-measure bias |
| **Exact-sampler** — the best answer carries no order | **N02, N05, N08, N09** | N02 is a rotation in `W(T)`; N05 is lognormal × a Poisson product; N08 steps the CIR variable exactly; N09 samples `ln A` as a Gaussian. All four are `sde_manual.md` rule 0 situations |

Three consequences, in order of how much they change the plan:

1. **Run the order-bearing five (six with N10) first, in `scheme-matched`.** They are the measurement
   SciML could not make, and they are worth more than the suite's recognition value on their own. A
   corpus of five pools where `order` is comparable is not large, but it is five more than BENCHOP
   offers and it is *not* the nine the candidate rule was fitted on — which is exactly the holdout
   property §10h and §10m hold their fixes to.
2. **The exact-sampler four are the control group, not the headline.** That matches §7's reading of
   them: they measure recognition. They also form a second, smaller holdout with BENCHOP's structural
   property, so the pool-relative rule must leave them bit-identical for the same reason it leaves
   BENCHOP alone — a prediction with a sharp right answer, added to §10 as prediction 8.
3. **The suite's framing in the paper gains a second leg.** Not only "ten recognisable problems",
   but "the corpus on which the ranking rule the project could not settle was settled, in either
   direction". A negative result there is publishable too, and cheaper than it looks: the pools come
   from runs §8 is doing anyway.

### 2b. Step 2 as implemented — the derivations `[rev3 2026-11-14, X]`

**Done for N05, N06 and N07; N03 is blocked on step 1.** Five truths were already
measured in rev 1 (`derivations_check.py`: N01, N02, N04, N08, N09). Rev 3 adds
[`n07_langevin.py`](notable-sde-data/n07_langevin.py),
[`n05_merton.py`](notable-sde-data/n05_merton.py) and
[`n03_n06_kolmogorov.py`](notable-sde-data/n03_n06_kolmogorov.py). **H-N2 is also
shipped** — `setup.py:521` now reads `SUITES = ("main", "no-closed-form", "benchop",
"sciml", "notable")`, `--suite notable --list` answers, and `--suite main` still
selects 49.

Six findings, and four of them overturn the plan rather than confirming it.

**Finding 1 — N06's premise was wrong, and the replacement is better.** `[C]` The plan
said *"The equation is reducible — a stochastic logistic, solvable in closed form"*.
Read at source, the explicit solution that line rests on
([the 2012 JMAA paper](https://strathprints.strath.ac.uk/40698/1/SIS_MS_final.pdf),
Theorem 2.1, eq. 2.5) is for **equation (2.3)**, `dI/dt = I[α_{r(t)} − β_{r(t)} I]` — a
*deterministic ODE with piecewise-constant random coefficients*, whose proof
substitutes `y = 1/I` and applies the **ordinary** variation-of-constants formula.
There is no Itô calculus in it and it does not apply to the Brownian model. And the
Brownian model does not reduce that way: with `y = 1/I` and Itô,

    dy = [(σ²N² − α) y + (β − 2σ²N) + σ²/y] dt − σ(N y − 1) dW

the noise is linear in `y` but the `σ²/y` term is not affine, so it does not
linearise; the logit chart fails the same way in `I`. The live literature on
positivity-preserving schemes for this exact model is consistent with there being no
closed form.

**The plan's pre-registered fallback is not needed, because a better route exists.**
The logit chart `Z = ln(I/(N−I))` does not linearise the equation but makes its noise
**additive** — constant `σN` — so the Kolmogorov solve runs in a constant-diffusion
operator on the real line instead of one that degenerates at both ends of `(0,N)`.
**The chart is also the technique the entry punishes a solver for not finding**
(`Lamperti` appears 0 times in `sde_manual.md`, §3), so the truth route and the lesson
coincide. N06 keeps its 2011 SIAM citation and its mechanism; only its route changes,
from "MC of the explicit solution" to "Kolmogorov in the Lamperti chart".

**Finding 2 — N06 is nearly vacuous at `T = 1`, which is rule 6 earning its keep.**
At the scale taken from the 2012 paper's Example 6.1 (`N = 100`, `β = 0.004`,
`μ = 0.05`, `γ = 0.15`, so `α = 0.2` and the deterministic equilibrium is 50), with
`I0 = 60`, `σ = 0.0025`:

    E[I_T] = 57.8453     I0 = 60     equilibrium = 50

The mean moves **3.59% from `I0`, which is 0.72× the 5% mean gate** — so a solver that
returns the initial condition unchanged *passes the mean gate*. The variance (28.09)
is not vacuous. **`T` must be raised** (the equilibrium is 16.7% from `I0`, so the
mean has somewhere to go) or the variance must be the deciding gate. Flagged for step
5 rather than fixed here, because the horizon interacts with the step budget
(`dt = 0.04/T`).

**Finding 3 — N06's truth is a `reference`, not an `exact`.** The three solves
(`N = 160, 192, 256` over `z ∈ [−6,6]` and `[−8,8]`) spread by **5.9e-07 relative**,
not the 1e-10 the double-well reaches, because the `(μ+γ)/(N−I)` term in the chart
drift stiffens as `I → N`. That is still 160× inside the independent MC cross-check
and three orders inside any gate, but it must be **declared as a measured
`reference_error`** — `verify.py` refuses a reference without one, and claiming
`exact` here would be the kind of unearned precision §6 exists to catch. Two
independent fine-step Monte Carlos, one in each chart, agree with it to 9.4e-05 and
1.2e-04 relative.

**Finding 4 — N05's mean is *exactly* vacuous, and its proposed control is not a
control.** `[C]` Both halves of what §2 (N05) proposed are wrong. The mean: `E[X_T] =
S0·e^{μT}` for every `λ`, `m`, `δ` and `σ`, and perturbing each by 10% moves it by
**0.00e+00** while moving the variance by 0.5–8.3%. The control: dropping the
compensator moves the mean by **1.385%, i.e. 0.28× the 5% gate — it passes**. Measured
on the gates that actually decide:

| control | mean | price | price error |
|---|---|---|---|
| compensator dropped | 1.385% (0.28× gate, **passes**) | 3.557893 | **18.98%** |
| jump channel omitted entirely | 0.000% | 3.635070 | **17.22%** |

against a 1% functional gate, **19.0× and 17.2×**. So N05 is graded on the price and
the second moment, the mean is recorded, and the controls are priced on the price. (At
`λ = 1`, `T = 1` the mean control would work — 73.3%, 14.7× the gate — but that is not
the published parameter set, and the published one is worth more than a convenient
control.)

**Finding 5 — N05's price is confirmed by two method families and a sampler**, which
is the discipline `benchop.py` applies to a published number. At the
d'Halluin–Forsyth–Vetzal set (`S0 = K = 100`, `r = 0.05`, `σ = 0.15`, `T = 0.25`,
`λ = 0.1`, `m = −0.90`, `δ = 0.45`; `[to confirm against the source in step 6]`):

| route | value |
|---|---|
| Merton's Poisson-weighted Black–Scholes series | 4.3912456892 |
| direct quadrature of the lognormal mixture (no BS formula) | 4.3912456892 — **1.6e-14** apart |
| Monte Carlo of the exact solution, 4M paths | 4.393054 ± 0.002755 — **0.66 σ** |

**Finding 6 — N07's `(γ, ω)` is `(1.0, 4.0)`, not the `(2.0, 5.0)` the plan proposed,
and the sweep is a three-way constraint rather than a one-way one.** `[C]` Three
things must hold at once and they pull against each other: Euler–Maruyama must be
**linearly stable** (`ρ(I + A·dt) < 1`, i.e. `γ > ω²dt`) or its error is a divergence
and the control is loud, which §5 rules out; EM's variance error must **exceed** the
gate; and `Var(q,T)` must sit **far from the stationary value** or the entry is
answerable by returning the Gibbs variance. **Only 3 of 30 pairs satisfy all three.**

| | γ=1, ω=4 (**chosen**) | γ=2, ω=5 (the plan's) | γ=1, ω=8 (widest separation) |
|---|---|---|---|
| `ρ(EM)` | 0.9928 — stable | 0.9798 — stable | 1.0307 — **unstable** |
| from stationary | **42.0%** = 4.2× gate | 13.6% = 1.4× gate | 3.65× gate |
| EM error | 0.2816 = **2.8× gate** | 0.4995 = 5.0× gate | 2.47 = 24.7× gate |
| BAOAB error | 0.00079 = gate/127 | 0.0003 | 0.0006 |
| verdict | **admissible** | **near-stationary** | EM diverging |

The plan's pair is rejected for vacuity: at `(2,5)` the terminal variance sits 1.4
gate-widths from the Gibbs value, so returning the stationary law would pass. And the
eye-catching separations — up to 3852× — are **all** in the regime where EM is
linearly unstable, so they measure a blow-up rather than a bias. Separation at the
chosen pair is 356×, with EM failing by 2.8× and BAOAB clearing by 127×.

**The truth routes themselves are validated, not asserted.** N07's exact moments agree
between Gauss–Legendre quadrature of the stochastic convolution and `solve_ivp` on the
covariance ODE to **1.3e-15**, are flat between 200 and 400 quadrature nodes to
4.6e-16, and reproduce the closed-form Gibbs covariance as `t → 200` to **1.5e-12**.
And the generalised Kolmogorov solver — `problems._kolmogorov_1d` extended from a
polynomial drift and constant `σ` to callable `b(x)`, `s(x)` — **reproduces the
shipped `sde_double_well_langevin` truth to 2.8e-15 (mean) and 1.4e-14 (variance)**,
which is the point: the generalisation is checked against an answer
`validate_ground_truth.py` already cross-checks three other ways.

**N03 is the one entry step 2 could not finish.** Its machinery runs — the Kolmogorov
solve is resolution- and domain-independent to 8 figures on a provisional parameter
set — but the **published parameters are still owed by step 1**, and a provisional set
must not reach `problems.py`. One conditioning finding is recorded for whoever
finishes it: the `α₋₁x⁻¹` drift makes the operator singular as `x → 0`, so the left
end of the interval cannot simply be pushed to zero the way the double-well's can.

---

### N01 — Clark–Cameron

    dS = mu dt + dW2,        S(0) = s0
    dU = S dW1,              U(0) = 0

with `W1`, `W2` independent. Proposed: `s0 = 1`, `mu = 1`, `T = 1`, `state_dimension: 2`,
`noise_dimension: 2`.

**Why it is notable.** It is *the* counterexample of the strong-convergence literature: Clark and
Cameron (1980) showed that for any approximation of `U(T)` built from the Brownian increments
alone, the mean-square error is bounded below by a constant times `T·dt` — strong order ½, for every
scheme, no matter how clever **[W]**. It is the reason multi-dimensional Milstein needs the Lévy
area, and it is still the test system in the current literature on Lévy-area generation (Foster et
al. 2023/2024 use it in the form `dU = S dW1`, `dS = mu dt + dW2`) **[W]**.

**Truth.** Two gates, and they measure different things.

- *Exact variance* **[derived]**: `U(T)` is a Wiener integral of a process independent of `W1`, so
  `E[U(T)] = 0` and

      Var[U(T)] = ∫₀ᵀ E[S(t)²] dt = s0²T + s0·mu·T² + mu²T³/3 + T²/2

  which at `s0 = mu = T = 1` is `1 + 1 + 1/3 + 1/2 = 17/6 = 2.8333…` — measured
  `2.8159 ± 0.0126` at 100 000 paths and 2000 steps, 1.4 standard errors low, as the left-point
  discretization of `S` predicts. The mean is exactly zero, so
  the mean gate must be skipped — `verify.py` has the `near_zero_mean` flag for precisely this
  (`verify.py:2495–2500`, and `sde_manual.md:160–162` records the `sde_stochastic_oscillator`
  precedent). Step 6 checks that the flag fires rather than assuming it.
- *Strong error* on the pathwise route, which is the gate that discriminates. The exact variance
  does **not**: Euler–Maruyama on `dU = S dW1` with the left-point `S` is the Itô sum itself, so it
  is unbiased in law to `O(dt)` and clears a variance gate comfortably. Only the pathwise route
  sees the barrier.

**Harness work.** `pathwise_exact` is a callable of `(t, W(t))` — `validate_ground_truth.py:426`
calls it as `V._sde_pathwise_truth(prob, T, dW.sum(axis=1))`, i.e. the truth must be a function of
the Brownian motion's *value*. `U(T)` is a functional of the whole path, so N01 does not fit the
shipped route. H-N1 (§4) adds a truth computed from the fine increments the CRN ladder already
generates.

**The reference estimator, and why it is good enough** **[derived]**. On the fine grid, the
left-point sum `Σ S(t_i)·ΔW1_i` has error variance `Σ ∫(t−t_i)dt = T·h/2`, i.e. `O(sqrt(h))` — the
same rate as the thing being graded, which would make it a useless reference. The **trapezoidal**
sum `Σ ½(S(t_i)+S(t_{i+1}))·ΔW1_i` is a different matter: because `S` and `W1` are independent their
cross-variation vanishes, so the Stratonovich and Itô integrals coincide here, and the trapezoid
converges at `O(h)` instead. At `h = 1/62500` that is a reference error around `5e-6` relative
against a graded Euler error around `1e-2` — three orders down. Step 5 measures both rather than
trusting this derivation, and stores the measured disagreement between the two estimators as the
reference error, which is the rule `verify.py` enforces: it refuses to grade against a reference
with no measured error (`benchmark/README.md`, "Ground truth").

**One-decision-wrong control.** Correct: Milstein with the Lévy area sampled (Foster/Davie, or
exact on a sub-grid). Wrong by one decision: the same Milstein with the Lévy area set to zero —
which is Euler–Maruyama, and is the approximation Clark–Cameron says is optimal among
increment-only schemes. The separation between them is the whole content of the entry, and if it is
smaller than `pathwise_tol` the entry is not worth shipping; step 5 decides that by measurement.

**Vacuity.** `U(T)` varies across paths by construction; `std/|mean|` is infinite because the mean
is zero, so the guard at `validate_ground_truth.py:437` passes trivially — and is therefore not
evidence. Step 6 checks instead that `Var[U]` at the graded step is resolved well above the
Monte-Carlo floor at the declared path count.

### N02 — Kubo oscillator (Stratonovich)

    dX1 = -a X2 dt - sigma X2 ∘ dW
    dX2 =  a X1 dt + sigma X1 ∘ dW      X(0) = (1, 0)

Proposed `a = 1`, `sigma = 1`, `T = 1`, `state_dimension: 2`, `noise_dimension: 1`.

**Why it is notable.** It is the textbook Stratonovich system — Kloeden–Platen's rotating-noise
example — and the standard test problem for stochastic geometric integrators, where the point is
that the exact flow conserves `|X|` and a naive scheme does not (Milstein, Repin & Tretyakov,
*SIAM J. Numer. Anal.* 2002) **[W]**.

**Truth, and why this entry is nearly free** **[derived]**. With `theta(t) = a·t + sigma·W(t)`, the
exact solution is the rotation `X(t) = R(theta(t))·X(0)`, a function of `W(t)` alone — so it fits
the **shipped** `pathwise_exact` contract with no harness work at all. `|X(t)| = |X(0)|` exactly,
which is a `diagnostics` structural gate. And the terminal moments are exact in closed form,
because `E[cos theta] = cos(a t)·e^{-sigma²t/2}` and `E[cos 2theta] = cos(2a t)·e^{-2sigma²t}`:

    E[X1(T)] = e^{-sigma²T/2} ( x1 cos aT - x2 sin aT )
    E[|X(T)|²] = |X(0)|²                       (exact, all T)

Measured at `a = sigma = T = 1`: `E[X1] = 0.3297` against the formula's `0.3277`, and
`E[|X|²] = 1.0004` against `1`.

**One-decision-wrong control, and it is a large separation.** The single decision is the
Itô–Stratonovich conversion. Reading the coefficients above as Itô coefficients and stepping them
solves a different equation, whose second moment *grows*: `E[|X|²] = |X(0)|²·e^{sigma²t}`, which at
`sigma = T = 1` is `e = 2.718×` the correct value **[derived]** — measured `2.7188` against
`e = 2.7183`, a ratio of `2.718` against the correct scheme's `1.0004`. A 172% error on a conserved
quantity, caught by the structural gate as well as the moment gate. This is the sharpest separation
in the suite and the cheapest to measure.

**Why it earns a slot despite being easy.** `sde_manual.md` contains the string "Stratonovich"
zero times (measured: `grep -ic`, §3). The entry is easy for someone who knows the distinction and
unanswerable for someone who does not, which makes it a test of the manual, not of the arithmetic.

**Vacuity and the transcription reading.** The exact solution is a one-line function of `W(T)`, so
this is the entry most likely to be answered by recognition rather than integration. That is the
`unrestricted`-arm situation `setup.py:566–574` was written for, and §7 says how it is recorded.
The `scheme-matched` arm is the headline.

### N03 — Aït-Sahalia interest-rate model

    dX = ( alpha_{-1} X^{-1} - alpha_0 + alpha_1 X - alpha_2 X^{rho} ) dt + sigma X^{theta} dW

on `X > 0`, with `rho > 1` and `theta > 1`. Parameters `[to fix]` from Szpruch, Mao, Higham & Pan
(2011), *BIT Numerical Mathematics* 51:405–425, whose numerical section is the one a referee will
check against **[W]**.

**Why it is notable.** Aït-Sahalia's (1996) nonlinear short-rate model is the standard example of a
"strongly nonlinear" SDE in the numerical literature: the `X^{-1}` term repels the origin, the
`X^{rho}` term is superlinearly mean-reverting, and the diffusion is superlinear too. The 2011
paper proves that *backward* Euler–Maruyama preserves positivity and converges strongly, under
`kappa + 1 > 2rho`, while explicit Euler–Maruyama does not preserve positivity **[W]** — and a
decade of follow-up work on explicit positivity-preserving schemes for this exact model (through
2024) is why the name is current rather than historical **[W]**.

**Truth.** A Kolmogorov/Fokker–Planck solve, declared `exact`, which is a route this repo already
owns: `sde_double_well_langevin` does precisely this (`validate_ground_truth.py:1422`,
`_dw_fd_moments`), with the entry's docstring recording why a PDE solve may be declared `exact`
when it evaluates as an ordinary `moments` callable. N03 is a different drift and diffusion in the
same 1-D machinery. Second gate: a stability criterion (positivity and finiteness), which is what
explicit EM gets wrong first.

**One-decision-wrong control.** Correct: backward EM, or an explicit positivity-preserving scheme
from the later literature. Wrong by one decision: the same scheme with the positivity projection
removed — i.e. explicit EM — which leaves the domain and then takes `X^{-1}` of a negative number.
Note that this control fails *loudly* (NaN) rather than with a wrong number, and `plan-sciml`
§2b found that loud controls are the ones that calibrate badly. Step 5 therefore also measures a
*quiet* control: explicit EM with a `max(X, eps)` clamp, which stays finite and biases the moments.

**Vacuity.** The horizon must be short enough that the answer is not the stationary density, which
is available in closed form up to a normalising constant for this model and could be transcribed.
Step 2 picks `T` from the Kolmogorov solve: the graded moments must differ from both the initial
condition and the stationary values by many multiples of the gate.

### N04 — Wright–Fisher / Jacobi diffusion with mutation

    dX = ( a(1-X) - b X ) dt + sigma sqrt( X(1-X) ) dW,     X(0) = x0 ∈ (0,1)

Proposed `a = b = 0.3`, `sigma² = 1`, `x0 = 0.5`, `T = 1` `[to fix in step 2]`.

**Why it is notable.** The Wright–Fisher diffusion is the central object of mathematical population
genetics (Kimura), and under its finance name — the Jacobi diffusion — it is a standard bounded
process. It is also one of the two canonical test problems for boundary-preserving schemes, beside
CIR.

**Truth: exact, and by machinery already in the repo** **[derived]**. The generator maps
polynomials to polynomials, so the moments close in a *triangular* system:

    dm_n/dt = [ n·a + (sigma²/2)n(n-1) ] m_{n-1} - [ n(a+b) + (sigma²/2)n(n-1) ] m_n

with `m_0 = 1`. The first two moments therefore come from a 2×2 lower-triangular linear ODE, exact
by matrix exponential — the same route `sde_log_heston_feller_violated` and
`sde_multichannel_stiff_m13` already use (`problems.py:3211`, `:3260`). Nothing new is needed.
Measured at `a = 0.3`, `b = 0.45`, `sigma² = 1`, `x0 = 0.5`, `T = 1`: the hierarchy gives
`m1 = 0.44724`, `m2 = 0.29119` against a fine-step simulation's `0.44655` and `0.29050`. The
asymmetric `b` is deliberate — see the vacuity note below, which is why the check was run that
way.

**The decision it punishes.** With `2a/sigma² = 0.6 < 1` the boundary `0` is attainable, exactly
parallel to a violated Feller condition, and a scheme must say what it does there. Euler–Maruyama
steps outside `[0,1]` and then takes the square root of a negative number. The entry is the
`[0,1]`-interval sibling of `sde_cir_feller_violated`, and the overlap is deliberate and stated:
what differs is that *two* boundaries are attainable and a reflecting fix at one of them is wrong at
the other.

**One-decision-wrong control.** Correct: a boundary-preserving step (split-step, or the Lamperti
transform with an implicit drift). Wrong by one decision: the same scheme with the projection
replaced by `clip(X, 0, 1)`, which is finite, stable, and biases `Var[X(T)]`.

**Vacuity.** `a = b` makes the stationary mean `0.5`, which is also `x0` — so the *mean* would be
vacuous. Step 2 breaks the symmetry (`a ≠ b`) or grades the variance only. This is exactly the
failure §1 rule 6 exists to catch, found at the drafting table rather than after a run, and it is
recorded here rather than silently fixed because the symmetric parameters are the ones in the
textbooks.

### N05 — Merton jump diffusion

    dX = (mu - lambda·kappa) X dt + sigma X dW + X dJ

where `J` is compound Poisson with intensity `lambda` and multiplicative jumps `Y = e^Z`,
`Z ~ N(m, delta²)`, and `kappa = E[Y] - 1 = e^{m + delta²/2} - 1`. Parameters `[to fix]` from a
published set so the figures are checkable.

**Why it is notable.** Merton (1976) is the first jump-diffusion option-pricing model and one of the
most cited papers in the field; the Poisson-weighted Black–Scholes series is in every textbook
**[W]**. It is also the only non-Gaussian driver anywhere in this repo's 66 problems.

**Why it needs no harness change.** A solver draws its own randomness on every route except
`pathwise` (`setup.py:186–206` supplies `dW` only on pathwise problems), so a compound-Poisson
channel is the solver's business. The harness grades moments and a functional, both of which are
distributional. **A jump channel on the pathwise route would be a harness extension; this entry
deliberately does not ask for one** (§11).

**Truth** **[derived]**. `X(T) = X0 exp((mu - lambda·kappa - sigma²/2)T + sigma W(T))·∏Y_i`, so

    E[X(T)]   = X0 e^{mu T}
    E[X(T)²]  = X0² exp( 2(mu - lambda kappa)T + sigma²T ) · exp( lambda T (e^{2m + 2delta²} - 1) )

both exact. Plus a `functional_truth` European call from Merton's series, which step 2 derives and
step 6 cross-checks by a second method family — a Fourier/COS inversion, the discipline
`benchop.py` already applies to the 24 published values.

**One-decision-wrong control.** Correct: a jump-adapted or fixed-grid scheme with the compensator in
the drift. Wrong by one decision: the same scheme with `lambda·kappa` dropped. ~~Parameters are
chosen in step 2 so that factor is several times the 5% mean gate.~~ **Measured in step 2 and
corrected (§2b finding 4): at the published parameter set the control moves the mean by 1.385%,
which is 0.28× the gate — it passes.** On the price it is 18.98%, and omitting the jump channel
altogether is 17.22%, against a 1% functional gate. The control is priced on the price, which is
also the gate the entry is decided on.

**Vacuity — predicted here, then measured.** `E[X(T)] = X0 e^{mu T}` is independent of `lambda`,
`m`, `delta` and `sigma`: the mean alone is answerable without simulating a single jump. Step 2
confirms it exactly — perturbing each of the four by 10% moves the mean by **0.00e+00** and the
variance by 0.5–8.3% (§2b finding 4). So the mean is recorded and the gates that decide this entry
are the second moment and the functional. Naming this here is the point of rule 6: an entry graded
on its mean would have looked fine and measured nothing.

### N06 — Stochastic SIS epidemic

    dI = I [ (beta N - mu - gamma) - beta I ] dt + sigma I (N - I) dW,   I(0) = I0 ∈ (0, N)

Parameters `[to fix]` from Gray, Greenhalgh, Hu, Mao & Pan (2011), *SIAM J. Appl. Math.*
71(3):876–902 **[W]**.

**Why it is notable.** It is the standard stochastic SIS model, the paper that introduced it is a
staple of the mathematical-biology literature, and the authors prove extinction/persistence
thresholds and derive the mean and variance of the stationary distribution **[W]**.

**Truth `[rev3 2026-11-14, C]`: a Kolmogorov solve in the Lamperti chart.**
~~A discretization-free Monte-Carlo of the explicit solution. The equation is reducible — a
stochastic logistic, solvable in closed form.~~ **Measured in step 2: it is not** (§2b finding 1).
The explicit solution this route rested on belongs to a *deterministic ODE with telegraph noise*,
and the Brownian model does not reduce — `y = 1/I` leaves a `σ²/y` term in the drift. **The
pre-registered fallback to the stochastic Verhulst equation is not needed**, because the logit chart
`Z = ln(I/(N−I))` makes the noise **additive**, which gives a better-conditioned Kolmogorov solve
than the plan assumed: in `I` the diffusion vanishes at both ends of `(0,N)` and the operator
degenerates there. N06 keeps its citation and its mechanism; only the route changes. The truth is a
**`reference` with a measured error of 5.9e-07 relative**, not an `exact` — the `(μ+γ)/(N−I)` term
stiffens as `I → N` — cross-checked by two independent fine-step Monte Carlos, one in each chart, to
9.4e-05 and 1.2e-04.

**The decision it punishes.** The diffusion `sigma·I(N-I)` is superlinear in `I`, so plain
Euler–Maruyama is not merely inaccurate: Hutzenthaler–Jentzen–Kloeden divergence applies and the
moments blow up. The domain `(0,N)` has two attainable-looking ends. Correct: a Lamperti transform
(the process is reducible, so one exists), taming, or an implicit step. Wrong by one decision: the
untamed explicit step.

**Vacuity `[rev3 2026-11-14, N]`.** At `T = 1` and the scale of the 2012 paper's Example 6.1, the
mean moves only **3.59% from `I0`, which is 0.72× the 5% mean gate** — a solver returning the
initial condition unchanged passes it (§2b finding 2). `T` must be raised, or the variance must be
the deciding gate. Settled in step 5, because the horizon trades against the step budget
(`dt = 0.04/T`).

**Overlap, stated.** `sde_quintic_drift_noise` and `sde_ginzburg_landau_s4/s6` already punish
untamed Euler on a superlinear coefficient. N06 adds the citation and the bounded domain; it does
not add a mechanism. If the suite has to lose an entry to cost, this is the one to lose — though
step 2 has now made it the only entry whose truth route *is* the technique it tests.

### N07 — Underdamped Langevin at moderate friction (BAOAB)

    dq = p dt
    dp = ( -omega² q - gamma p ) dt + sqrt(2 gamma / beta) dW,      (q,p)(0) = (1, 0)

~~Proposed `omega = 5`, `gamma = 2`, `beta = 1`, `T = 1`.~~ **Set by the step-2 sweep
(§2b finding 6): `gamma = 1`, `omega = 4`, `beta = 1`, `T = 1`.** The proposed pair is rejected for
vacuity — at `(2, 5)` the terminal variance sits 1.4 gate-widths from the stationary value, so
returning the Gibbs law would pass.

**Why it is notable.** Underdamped Langevin dynamics is the workhorse of molecular sampling, and
BAOAB is a household name in that field: Leimkuhler and Matthews (2012/2013) compared Langevin
integrators by the *invariant measure* they sample rather than by their trajectory error, and showed
BAOAB's configurational marginal is unusually accurate — the search-confirmed statements are that
BAOAB is strong order one with an asymptotic bias of order two, and that for the configurational
marginal a superconvergence occurs **[W]**. The entry does not need to adjudicate the order of that
superconvergence; it needs the comparison to be the one the field makes.

**Truth: exact.** The system is a 2-D linear SDE with additive noise, so the mean is `e^{At}X(0)`
and the covariance solves the matrix Lyapunov equation — exact by matrix exponential, the machinery
`sde_oscillator_long_horizon` and `sde_multichannel_stiff_m13` already use. Stationary values
`Var(q) = 1/(beta omega²)`, `Var(p) = 1/beta` are a second, independent check at long horizon.

**The decision it punishes.** Splitting. Euler–Maruyama's bias on `Var(q)` is `O(dt)` with a
coefficient that grows with `gamma` and `omega`; BAOAB's, and any Strang-type splitting's, is far
smaller at the same step. Correct: BAOAB (or OBABO, or an exact Ornstein–Uhlenbeck propagator for
the `p` leg). Wrong by one decision: the same splitting with the `O` step replaced by an explicit
Euler drag update.

**The design risk, stated plainly — and it was real, in a direction the plan did not predict.**
The verifier re-runs at `min(dt, 0.04/T)` (`verify.py:271–278`), so the graded step is at most
`0.04` and whether EM's bias exceeds the 10% variance gate is a measurement. It does, easily — but
**the binding constraint turned out to be vacuity and EM's linear stability, not the separation**
(§2b finding 6). Only **3 of 30** `(gamma, omega)` pairs clear all three at once: EM stable
(`ρ(I + A·dt) < 1`, i.e. `gamma > omega²·dt`), EM's error above the gate, and `Var(q,T)` far from
the Gibbs value. The eye-catching separations — up to 3852× — are all in the regime where EM is
*diverging* rather than biased, which §5 rules out as a loud control. At the chosen pair the
separation is 356×, EM failing by 2.8× and BAOAB clearing by 127×.

**Overlap, stated.** `sde_stochastic_oscillator` (2-D, noise in the velocity) and `sde_ou_stiff`
each cover part of this. What is new is the friction term and the invariant-measure framing, which
is the thing BAOAB is known for.

### N08 — The 3/2 stochastic-volatility model

    dV = V ( a - b V ) dt + sigma V^{3/2} dW,       V(0) = v0 > 0

Parameters `[to fix]` from Ahn & Gao (1999) **[W]**.

**Why it is notable.** The 3/2 (Ahn–Gao) model is the standard alternative to Heston's square-root
variance, it is the canonical example in the moment-explosion literature, and it is a recurring test
case in work on tamed and implicit schemes for superlinear diffusions **[W]**.

**Truth: exact, through the inverse** **[derived]**. For `U = 1/V`, Itô gives

    dU = ( (b + sigma²) - a U ) dt - sigma sqrt(U) dW

which is a CIR process with `kappa = a`, `theta = (b + sigma²)/a` and vol `sigma`. So every moment
of `1/V(T)` is a CIR moment in closed form, and `2 kappa theta = 2(b + sigma²) > sigma²` always, so
`U` never reaches zero and `V` never explodes. Measured at `a = 1`, `b = 2`, `sigma = 0.6`,
`v0 = 0.3`, `T = 1`: `E[1/V(T)] = 2.7156` by simulation of the `V` equation against the CIR mean's
`2.7181`. The entry is graded on `E[1/V(T)]` and
`Var[1/V(T)]`, which are bulletproof, with `E[V(T)]` added only if step 2 confirms the inverse-moment
condition `2(b+sigma²)/sigma² > 1` holds at the chosen parameters and derives the value by a second
route.

**The decision it punishes.** A `V^{3/2}` diffusion is superlinear, so explicit Euler–Maruyama has
unbounded moments here in the Hutzenthaler–Jentzen–Kloeden sense. Correct: step the CIR variable
`U`, or tame. Wrong by one decision: explicit EM on `V` directly.

**Overlap, stated.** The *mechanism* is shared with `sde_quintic_drift_noise` (untamed explicit
Euler on a superlinear coefficient). What N08 adds is a named finance model and a truth route
through an exact transformation — which is also the solution, so this entry rewards recognising the
transformation rather than discovering a scheme.

### N09 — Geometric-average Asian option

    dS = r S dt + sigma S dW,     A = exp( (1/T) ∫₀ᵀ ln S_t dt ),     payoff (A - K)^+

Parameters `[to fix]`: a standard set (`S0 = 100`, `K = 100`, `r = 0.05`, `sigma = 0.2`, `T = 1`).

**Why it is notable.** Kemna–Vorst (1990) is the closed-form geometric Asian, and the Asian option on
GBM is the standard worked example of the multilevel Monte-Carlo literature (Giles 2008) **[W]**.

**Truth: exact functional** **[derived]**. `ln A` is Gaussian with mean
`ln S0 + (r - sigma²/2)T/2` and variance `sigma²T/3`, so the price is a Black–Scholes formula at the
adjusted drift and volatility `sigma/sqrt(3)`. Graded as `functional_truth`, which brings the SDE
side's two-path-count defence automatically (`benchmark/README.md`, "On the SDE side a functional is
also checked at two path counts") — the check that a returned constant cannot pass. `E[A]` and
`Var[A]` are exact as well and are the second gate.

**The decision it punishes: the quadrature of a path integral** **[derived, measured]**. Nothing
else in the 66 grades a time integral along the path. An endpoint Riemann sum of `∫ ln S dt` gives
the discretely sampled geometric average, whose variance differs from the continuous `sigma²T/3` at
`O(1/n)`. At the `n = 25` steps `0.04/T` implies, computed exactly — it is a quadratic form in the
Brownian covariance, so no simulation is involved:

    left-point sum  (i = 0 … n-1)   Var = 0.313600   -5.92%
    right-point sum (i = 1 … n)     Var = 0.353600   +6.08%
    trapezoid                       Var = 0.333200   -0.04%

**This plan's first draft had the sign wrong**: it said `+6%` for the left-point sum, which in fact
*under*states the variance. The direction matters, because it is the direction of the price error a
reviewer would check. Note also that the trapezoid is not merely an order better here — at `n = 25`
it is already `150×` closer than either endpoint rule, so the separation is wide and the gate is
easy to place. Correct: trapezoid (or sample `ln A` from its exact Gaussian law). Wrong by one
decision: the same solver with the left-point sum.

**Vacuity.** The price depends on every parameter and on the quadrature; the two-path-count check
catches a constant. Clean.

### N10 — Stochastic Duffing–van der Pol — **Phase 2**

    dx1 = x2 dt
    dx2 = [ x1(alpha - x1²) - beta x2 ] dt + sigma x1 ∘ dW

**Why it is notable.** The stochastic Duffing–van der Pol oscillator is the standard example in
stochastic bifurcation theory (Baxendale; Arnold's *Random Dynamical Systems*) and appears as a
worked system in Kloeden–Platen **[W]**.

**Why it is phase-gated.** There is no closed form, so the truth is a `reference`, and this repo's
rule for a reference is two *independent method families* with a measured disagreement
(`make_references.py`, and the six existing reference problems). For a 2-D SDE that means a
Fokker–Planck finite-difference solve on `(x1, x2)` plus a high-resolution Monte Carlo with an
order-1 scheme, both offline, plus the `reference_error` that `verify.py` refuses to proceed without.
That is the largest single item in this plan and it is the only entry whose absence does not weaken
the suite's claim — the other nine already span nine mechanisms. **It ships only if steps 1–9 land
with budget left.**

---

## 3. What the manuals do not cover — the measured gap

This is the section that decides whether the pipeline can plausibly do well, and it is a
measurement, not an opinion. `references/sde_manual.md` is 399 lines. Counting the lines that
match, case-insensitively (`grep -ic <term> references/sde_manual.md`, run 2026-11-12):

| Term | Matching lines | Entry that needs it |
|---|---|---|
| `Stratonovich` | **0** | N02, N10 |
| `tamed` / `taming` | **0** | N06, N08 |
| `splitting` | **0** | N07 |
| `BAOAB` | **0** | N07 |
| `Langevin` | **0** | N07 |
| `compound Poisson` | **0** | N05 |
| `jump-adapted` | **0** | N05 |
| `Lamperti` | **0** | N06, N08 |
| `invariant` | **0** | N07 |
| `implicit` | 3 | N03 |
| `Lévy` | 2 — lines 38 and 266, both telling the agents **not** to | N01 |

So **seven of the ten entries turn on a technique the manual does not name**. One (N01) turns on a
technique the manual mentions twice and rules out both times: line 38 says the Lévy area is "not
computable without approximation", and line 266 is rule 1, which sends every multi-dimensional
problem to Euler–Maruyama. The remaining two (N04's boundary handling, N09's quadrature) turn on
reasoning the manual gestures at without naming the method: rule 5 on boundary treatment
(`sde_manual.md:289–309`) and rule 0 on exact transition laws (`:168–200`).

Two readings of that, and the plan commits to neither in advance:

- **The suite measures the pipeline.** The conductor's job is to find a method, not to recite one,
  and `sde_manual.md:168–176` records that an untooled completion already beat rule 1 on nine of ten
  option-pricing problems. If the pipeline proposes a splitting scheme for N07 without being told
  about BAOAB, that is a strong, paper-worthy result.
- **The suite measures the manual.** If the plans come back Euler-and-Milstein on all ten, the
  result is a ceiling, and the honest reporting is a table of what each entry needed against what
  the manual offers — which is this table, and which is worth publishing either way.

**This is why the suite is worth running even if it goes badly**, and it is the reason to prefer it
over ten more tier-3 problems whose mechanisms the manual already covers.

---

## 4. Harness work list

Small, because the four routes this suite leans on are shipped. Each item names its files and its
acceptance test.

### H-N1 — a pathwise truth that is a functional of the whole path · **required by N01**

`verify.py:473` (`_sde_pathwise_truth`) calls `problem["pathwise_exact"](t, W)` where `W` is the
Brownian motion's *value*; `validate_ground_truth.py:426` pins that contract. N01's truth is
`∫ S dW1`, which no function of `W(T)` can give.

**Change.** A second, optional key — `pathwise_reference(dW_fine, dt_fine)` — which receives the
fine increments the CRN payload already carries (`sde_pathwise_crn`, `verify.py:398`) and returns
the reference terminal state. A problem declares one or the other, never both. The reference's own
error must be declared (`pathwise_reference_error`) and `verify.py` must refuse to grade against a
pathwise reference that has none, exactly as it refuses a field reference without one.

**Acceptance.** (1) All nine existing pathwise/SciML entries hash identically in `manifest.json`
before and after. (2) On N01 with the *exact-variance* route as cross-check, the trapezoidal
reference and a 4× finer left-point reference agree to within the declared error. (3) A declared
`pathwise_reference_error` of `None` is refused at authoring time in `validate_ground_truth.py`.

### H-N2 — register the suite · **trivial**

`setup.py:521` — add `"notable"` to `SUITES`. Everything downstream is generic: `problem_suite`,
`apply_suite_filter`, `guard_results_path`, `add_suite_arguments`, `n_by_suite`
(`setup.py:488–497`), the four-suite paragraph in `report.py:817–826`. **Acceptance:**
`run.py --suite notable --list` prints ten problems and `--suite main` still prints 49.

### H-N3 — a structural invariant gate on an SDE problem · **required by N02** · **SHIPPED 2026-11-14, see §4a**

`diagnostics` with `gate=True` exists (`benchmark/README.md`, "A problem may also declare"), but
every current user is a PDE. N02 needs `|X(T)| == |X(0)|` to within a tolerance as a gate on an SDE
problem's terminal paths. **Acceptance:** the Itô-misreading control fails the diagnostic gate at
`sigma = T = 1` by the derived factor `e^{1/2}` on the norm, and a correct Stratonovich scheme
clears it; both measured, not asserted.

**The acceptance test has to be the gate *failing*, because the existing mechanism is measured
inert.** `plan-sciml-benchmark.md` §10d, on M08: `invariant_worst_headroom` was **negative on all
three plans** — the declared invariant was violated by every one of them — and the problem graded
`10` with `passed: true` on all three, while `structure` scored `0.000` for everyone so the
component could not rank either. An invariant that neither fails a plan nor orders a pool is doing
no work. N02 is the entry whose design rests on an invariant gate, so H-N3 is not "wire a diagnostic
in": it is *make the gate bite*, and a version of H-N3 that reproduces M08's behaviour has shipped
nothing. This is the one item in §4 to build and smoke-test before the rest of the suite is
authored — if an SDE invariant cannot be made to fail, N02 loses its structural gate and keeps only
its moment gate, which is still a 172% separation (§2, N02) but no longer a second, independent
check.

**Premise re-verified 2026-11-13** `[rev1, N]`, after the rev 11–20 SciML work closed everything
else in §4: `invariant_worst_headroom` is written by `kernel/driver.py:339` and
`kernel/sde/driver.py:296`, read by `ranking.py:431` as the `structure` component and by `cli.py:765`
for display, and **by nothing that fails a plan** — `grep -rn 'invariant_worst_headroom\|invariant_max_drift'`
over `verifylib/` returns ranking, recording and display sites only. So M08's behaviour is intact and
H-N3 is now **the only item in §4 of the second kind** (a defect that corrupts the record rather than
defining the measurement). It is also newly interesting rather than merely necessary: rev 20 measured
`structure` to be *simultaneously* what ruins M05's ranking and what carries M02, M03, M06 and M07,
which withdrew one candidate fix as harmful. N02 would add the first entry whose `structure`
component is an exactly known invariant rather than a diagnostic of convenience — i.e. the first
pool where that component's value can be checked against a right answer.

### H-N4 — the vacuity guard on the moment and functional routes · **required by rule 6**

`validate_ground_truth.py:434–443` refuses a vacuous *pathwise* entry. N05's mean is independent of
four of its five parameters, and N04's symmetric mean equals its initial condition: both are vacuity
of a kind the current guard cannot see. **Change:** at authoring time, for every moment and
functional gate, compute the truth at the declared parameters and at a perturbation of each
parameter, and refuse a gate that no parameter moves by more than its own tolerance.
**Acceptance:** N05's mean gate and N04's symmetric mean gate are both *refused* by the new check
before they are fixed, and the 66 existing entries all pass it. *(If an existing entry fails, that
is a finding and it goes in the plan, not a patch to the check.)*

### H-N5 — nothing for jumps · **deliberate**

N05 needs no harness change (§2, N05). A jump channel on the pathwise route would need one and is
listed in §11 as not done.

### H-N6 — the SciML prerequisites · **resolved 2026-11-13, and not the way this plan asked for**
`[rev1 2026-11-13, C]`

**Withdrawn.** Rev 1 of this plan said: *"fix defects 9 and 10 before step 8"*, reasoning that a
suite with no measured baseline has nothing for the fix to invalidate, so fixing is free now and
expensive later. The reasoning about *timing* was right and the premise under it was wrong, and
`plan-sciml-benchmark.md` rev 16–17 `[2026-11-13]` is the measurement that shows it.

**Defect 9 is retired as a false positive.** The `num_paths = 1000` the cap objected to is the
**harness's own call parameter**: `problems.py`, the brief's own `solve_sde(num_paths=1000, …)` line
and the executed run agree on all nine entries, and `runner.run_solver` builds `spec["call"]` and
then copies `run_meta` out of it — so a solver cannot influence the field it was being capped on.
The cap tracks the *formulator's* `evaluation_thresholds.num_paths` instead, at **cap ⟺ declared ≠
executed, 9 of 9**, with M05 and M09 — the only two whose formulator declared correctly — the only
two uncapped. **All seven capped entries were graded 10/10 by the harness independently.** So
applying the cap, which is exactly what this plan asked for, would have taken seven of nine entries
from a harness-verified 10 to a 3 on a number they never chose.

**Defect 10 was the real one, and the fix is an attribution rather than a cap.** Rev 17 adds
`findings.SPEC`, a third severity for a finding that is certain and is about *someone other than the
solver* — the gate/warn split had no room for it, both halves being claims about the solver
differing only in reliability. All four SDE run-parameter findings now raise that way and land in
`ledger_spec_defects` with no cap (`run.py:732`'s whitelist, with the reasoning in the comment).
Re-audited: **7/9 capped → 0/9, with the same 7/9 still reporting the defect.** PDE output findings
still gate.

**What this suite inherits is therefore a fixed harness, and one lesson.** The lesson is the one
worth keeping: *a defect report is itself a measurement, and this plan quoted one without checking
whose fault it named.* Three greps established that nothing read `score_cap`; no grep establishes
that it *should* be read. Step 0 is reduced to confirming the fix is in and recording it.

**The distinction that decides every item in this section, and §11's, still stands**: does the
defect change what is being measured, or does it corrupt the record of the measurement? The manual's
coverage gaps are the first kind and §11 leaves them alone. What has changed is that the second
category is now empty but for H-N3.

### 4a. H-N3 as implemented `[rev2 2026-11-14, X]`

**Shipped.** The SDE routes now run structure diagnostics and a declared `gate` fails the run.
`benchmark/verify.py` gains `_run_sde_diagnostics`, `_fold_sde_diagnostics`, `_sde_terminal_paths`
and `_orient_paths`; `benchmark/problems.py` gains the two Kubo diagnostics;
`benchmark/tests/test_sde_diagnostics.py` is 19 tests. **No verdict code changed** —
`report.compute_verdict` and `compute_oneshot_verdict` already return `CONSTRAINT_VIOLATION` ahead
of every pass branch (`report.py:317`, `:254`), so folding the same record shape the PDE side
produces was enough to make the gate bite on both the pipeline and the one-shot verdict.

**The gate is calibrated both ways and the statistic was chosen by measurement, not by taste.**
`docs/notable-sde-data/kubo_invariant_calibration.py`, at exactly N02's configuration and the step
the verifier re-runs at (`min(dt, 0.04/T) = 0.04`, 25 steps, 50 000 paths, seed 42):

| scheme | `\|E\|X\|²/\|X₀\|²−1\|` **(gated)** | RMS per-path drift | `E\|X(T)\|²` |
|---|---|---|---|
| exact rotation `[correct]` | 0.000e+00 | 4.791e-17 | 1.0000 |
| implicit midpoint `[correct]` | 0.000e+00 | 4.124e-16 | 1.0000 |
| Stratonovich Heun `[correct]` | 3.302e-02 | 1.952e-02 | 1.0330 |
| Itô–Euler + conversion `[correct]` | 5.133e-02 | 1.451e-01 | 1.0513 |
| **Itô–Euler, misread** `[WRONG]` | **1.771e+00** | 6.873e-01 | 2.7706 |
| | **34.5× separation** | 4.7× separation | |

**The gate is 0.30**, the geometric mean of the worst correct scheme and the wrong one: the worst
correct scheme clears by 5.8× and the misread fails by 5.9×. The separation is **34.5×**, against
BENCHOP's narrowest 6.8× and SciML's 4.4×, and it was fixed from this table and not moved after.

**Finding 1 — the per-path statistic, which is the stronger statement, is the weaker instrument.**
The obvious gate is the per-path amplitude drift, since the exact flow conserves `|X|` *pathwise*.
Measured, it separates correct from wrong by **4.7×** where the mean-square statistic separates
them by **34.5×**, because a correct but non-conservative scheme (Heun, Euler) has a genuine `O(dt)`
amplitude *wobble*: large in RMS, small in the mean. Gating the per-path number charges a correct
scheme for its own variance. So the mean square is the gate and the per-path RMS ships beside it
un-gated — it is the one that would see a scheme whose amplitude wanders symmetrically, which the
gated statistic cannot. One `gate: True` and one `gate: False`, the Taylor-Green pattern.

**Finding 2 — a solver returning the wrong number of components passes the moment route.** `[N]`
Measured: an `(n, 3)` sample on a `state_dimension: 2` problem is scored `status: ok`,
`passed: True`. The route reads the first `d` columns and never asks whether there are more
(`_run_and_read_sde`'s label loop). This is **pre-existing and applies to all 24 SDE problems on
record**, not only this suite, and it is the second kind of defect by §4's own test — a malformed
output recorded as a pass. It is **recorded and not fixed here**, because the fix is a contract
check in `_run_and_read_sde` rather than anything H-N3 owns, and slipping it into a diagnostics
change is how a one-line behaviour change becomes invisible. H-N3's gate catches it only where a
diagnostic is declared, which is nowhere yet. **Recommended as its own one-line change, with the
test already written** (`test_a_gate_that_cannot_be_evaluated_has_not_been_cleared` pins both
halves: the gate refuses it, and without the gate the same output is a clean pass).

**Finding 3 — `verify_sde_stability`'s documented fallback is unreachable.** `[N]` Its "No terminal
states exposed: use the reported moments' finiteness as the blow-up signal" branch cannot run,
because `_run_and_read_sde` raises `KeyError('terminal_paths')` before the route sees the output
(measured directly). Pre-existing, recorded, not fixed — it is a dead branch rather than a wrong
answer, and it changes nothing about what is graded today.

**Finding 4 — the path-orientation rule had three copies.** The contract accepts `(n, d)` or
`(d, n)`, so every reader of `terminal_paths` has to apply the same transpose rule, and the moments
route, the pathwise route and the new fold each had their own. Extracted to `_orient_paths`, for
the reason `RECORDED_HARNESS` was extracted: a rule with three copies is a rule that drifts. The
one behaviour worth pinning is that a square sample is left alone, since nothing can disambiguate
it (`test_orient_paths_leaves_a_square_sample_alone`).

**The kernel-side half is deliberately still open.** The M08 finding this item quoted is about
`verifylib`'s `invariant_worst_headroom`, which feeds `ranking.structure` and fails nothing. That is
the *self* grader, and this change is the independent one; the two are kept ignorant of each other
on purpose (`benchmark/README.md`, "Two graders, one solver"). Making the kernel's invariant fail a
plan would move scores on three suites whose baselines are already measured — the situation SciML
rev 16 is a cautionary tale about — so it needs its own before-and-after. Recorded as an open item
rather than done. **What this suite can now do that it could not is make the kernel's number
checkable**: N02 would be the first pool whose `structure` component is an exactly known invariant
rather than a diagnostic of convenience, so the kernel's value has a right answer to be compared to.

**Regression surface: nil, and verified rather than assumed.** No SDE problem on record declares
`diagnostics` (measured; pinned by `test_no_sde_problem_on_record_declares_diagnostics`), so the
fold is inert on all 75 entries, and every one of the 75 manifest entries hashes identically before
and after. `ruff` clean on all three changed files.

---

## 5. Gate calibration — both ways, per entry

The repo's standing rule (`benchmark/README.md`, "Every gate in this suite is calibrated both
ways"): a correct scheme must clear the gate *at the resolution the entry ships with*, and a
plausible wrong one must fail it, and the two must differ by exactly one decision so that what
separates them is that decision. BENCHOP's narrowest separation is 6.8×; SciML's is 4.4×. **No gate
on this suite is set before both numbers exist, and no gate is moved after seeing where a scheme
landed.**

| ID | Correct scheme | Wrong by exactly one decision | Expected separation |
|---|---|---|---|
| N01 | Milstein + sampled Lévy area | the same, Lévy area set to zero (= EM) | `[to measure]` — the entry is dropped if it is under 3× |
| N02 | Stratonovich (midpoint / conversion term) | the same coefficients read as Itô | **34.5×** on the invariant gate, measured — §4a |
| N03 | backward EM / positivity-preserving explicit | *quiet* variant: explicit EM with `max(X, eps)` | `[to measure]` |
| N04 | split-step / Lamperti + implicit drift | the same with `clip(X, 0, 1)` | `[to measure]` |
| N05 | jump-adapted or fixed-grid with compensator | the same with `lambda·kappa` dropped; also, jumps omitted entirely | **19.0× and 17.2×** of a 1% functional gate, measured — and **0.28× on the mean, which passes** (§2b) |
| N06 | Lamperti chart + implicit/tamed drift | untamed explicit EM | `[to measure]`; the control may diverge, so a quiet variant is needed too. Truth route settled in §2b; horizon still open (near-vacuous mean) |
| N07 | BAOAB | Euler–Maruyama (and the `O` step by explicit Euler drag) | **356×** at `gamma=1, omega=4`: EM 2.8× the gate, BAOAB gate/127 — measured, §2b |
| N08 | CIR step on `U = 1/V` (or tamed) | explicit EM on `V` | `[to measure]`; likely loud |
| N09 | trapezoid (or the exact Gaussian law of `ln A`) | the same with the left-point sum | `-5.92%` on `Var[A]` at `n = 25` against the trapezoid's `-0.04%` — `150×`, measured |
| N10 | — | — | Phase 2 |

**Step 2 has already filled four of these rows and moved two parameter sets** (§2b). It also added
a fourth lesson to the three below, from N07: **a control can be loud for a reason the gate cannot
see.** Euler–Maruyama's variance error on an oscillator exceeds any gate once `ρ(I + A·dt) > 1`, but
then it is diverging rather than biased, and the separation measures the blow-up. The admissibility
test is therefore a *property of the scheme* (`gamma > omega²·dt`) and not of the error, and it has
to be computed, not eyeballed — 11 of 30 pairs failed it while separating beautifully.

Three lessons from the two previous calibrations are adopted rather than rediscovered:

- **A loud control calibrates badly.** `plan-sciml` §2b found that two of its one-decision controls
  were not wrong schemes at all (a dropped Milstein correction *is* Euler–Maruyama). Here the
  opposite failure dominates: N03, N06 and N08's natural controls produce NaN, which separates
  everything from everything and sets no scale. Every loud control gets a quiet sibling.
- **The Monte-Carlo floor can make a correct scheme's error unmeasurable.** `plan-sciml` §2b,
  finding 4. Every gate here must sit several standard errors above zero at the declared path count,
  and step 5 records the standard error beside the error.
- **The step the solver is graded at is `min(dt, 0.04/T)`**, not the entry's `dt`
  (`verify.py:271`). Every separation above is measured at that step, as BENCHOP's were.

---

## 6. Ground-truth validation — routes that share nothing with the entry

`validate_ground_truth.py` exists because an answer key and the thing it grades must not come from
the same derivation. Per entry:

| ID | The entry computes | The validator computes, independently |
|---|---|---|
| N01 | `Var[U(T)] = 17/6` in closed form | the moment ODE `d/dt Var[U] = E[S²]` integrated numerically; and a direct EM simulation at 4 step sizes, Richardson-extrapolated |
| N02 | rotation + `e^{-sigma²t/2}` moments | direct simulation of the Stratonovich SDE by midpoint; and the exact `|X|` invariant on every sampled path |
| N03 | 1-D Kolmogorov FD solve | a second grid and a second order (the `_dw_fd_moments` precedent refines `N`), plus a backward-EM Monte Carlo at a very fine step |
| N04 | triangular moment ODE by matrix exponential | the same hierarchy integrated by `solve_ivp` with tight tolerances; plus a boundary-preserving MC |
| N05 | closed-form moments + Merton series | moments by direct MC of the exact solution; the price by Fourier/COS inversion — two families, as `benchop.py` requires |
| N06 | MC of the explicit solution | the paper's stationary mean and variance at long horizon; plus a Lamperti-transformed fine-step MC |
| N07 | matrix exponential + Lyapunov | `solve_ivp` on the covariance ODE; plus the closed-form stationary values as `T → ∞` |
| N08 | CIR moments of `1/V` | the CIR moment ODEs integrated numerically; plus an exact non-central-χ² sampler for `U` |
| N09 | Kemna–Vorst closed form | the Gaussian law of `ln A` sampled directly (no SDE at all); plus quadrature of `E[(A-K)^+]` against the lognormal density |
| N10 | FD Fokker–Planck | high-resolution MC — the measured disagreement **is** `reference_error` |

Five of these routes were already exercised at draft time —
[`notable-sde-data/derivations_check.py`](notable-sde-data/derivations_check.py) is the N01, N02,
N04, N08 and N09 columns of the right-hand side, at a step far finer than any graded step. It
confirms the formulas and nothing about the gates: §5's separations are a different measurement.

Also per entry, and this is the part §1 rule 6 exists for: **the vacuity table**. For each gate,
the validator records the truth at the graded horizon, the initial condition, the stationary value
where one exists, and the value under a `±10%` perturbation of each parameter. A gate that cannot
be moved by any parameter is refused (H-N4).

---

## 7. The two arms, and the transcription reading

`setup.py:525–574` already defines `ARMS` and the paragraph that makes `unrestricted` a different
experiment rather than a relabelling of the same one. This suite uses both, and the reason is §0b:

- **`scheme-matched`** is the headline. The brief asks for a discretised stepper that consumes the
  supplied increments where the route supplies them, and the pathwise degeneracy fold
  (`verify.py:2255–2269`) records `degenerate: true` with `reason: "pathwise_exact"` when a solver
  carried no discretization error at all.
- **`unrestricted`** is the capability note: any method, including the exact law. On N02, N05, N08
  and N09 the exact law or an exact transformation *is* the best answer, and `sde_manual.md` rule 0
  asks for it in as many words.

**What the paper reports is both columns, labelled.** The sentence the suite licenses is of the form
"*n* of 10 verified under `scheme-matched`, of which *k* integrated and *n−k* recognised a closed
form". Any sentence that merges those two numbers is overclaiming, and this plan's position is that
the honest split is more interesting than the merged total — because "recognised that the Kubo
oscillator is a rotation and that `1/V` is CIR" is a real capability and an unusual one to be able to
measure.

One precaution the two previous suites paid for: `plan-sciml` §7a found the two arms had been staging
a byte-identical brief and sharing artifact paths, so the second run deleted the first's solvers.
**Step 3 verifies arm separation on this suite before any graded run** — different brief text,
different artifact paths, different unit keys.

---

## 8. Steps

| # | Step | Depends on | Output |
|---|---|---|---|
| **0** | ~~Fix SciML defects 9 and 10~~ **Done 2026-11-13, differently — H-N6 withdrawn.** Confirm `ledger_spec_defects` reaches the record and that `score_cap` is uncapped for spec defects by design | — | one dated paragraph in §4 **— written** |
| **1** | Read the ten sources; fix every `[to fix]` parameter from the cited paper, not from taste | — | §2 updated with sourced parameters, and a `[W]`-labelled note per entry where the source disagrees with this draft |
| **2** | Derive each truth as a standalone, seeded, numpy-only script under `docs/notable-sde-data/` | 1 | **Done for 8 of 9 — §2b.** `derivations_check.py` (N01/N02/N04/N08/N09), `n07_langevin.py`, `n05_merton.py`, `n03_n06_kolmogorov.py`. The N06 go/no-go is answered (premise wrong, better route found) and the Merton series is derived and triple-checked. **N03 blocked on step 1** |
| **3** | ~~H-N2 (suite registration)~~ **done 2026-11-14** and arm separation; verify the existing entries hash identically | — | `--suite notable --list` answers and `--suite main` still selects 49; all 75 manifest entries hash identically. **Arm separation still to verify** |
| **4** | H-N1 (pathwise reference), ~~H-N3 (SDE invariant gate)~~ **H-N3 done 2026-11-14 — §4a**, H-N4 (vacuity guard), with the acceptance tests in §4 | 3 | the three items shipped; any existing entry the new guards refuse is a **finding** |
| **5** | Calibrate every gate both ways at `min(dt, 0.04/T)`, with standard errors; set `gamma`, `omega` for N07 and the jump parameters for N05 **from** this sweep | 2, 4 | the §5 table filled with measured numbers; any entry under 3× separation is dropped and said to be dropped |
| **6** | Author the nine Phase-1 entries in `problems.py`; add the independent validation routes of §6 and the vacuity table | 5 | nine entries, `validate_ground_truth.py` green, no gate set after seeing a scheme |
| **7** | Smoke-test end to end on one entry (N02 — cheapest, sharpest control) in both arms | 6 | one graded record; defects found here are recorded as defects, as §7a's ten were |
| **8** | C0 baseline (and C1 if the budget allows) on all nine, both arms, with `--results-json benchmark/results/baselines/oneshot_notable.json` | 6 | the baseline denominators this suite's rates are quoted against |
| **9a** | Pipeline runs, **the order-bearing five first** (N01, N03, N04, N06, N07), `scheme-matched` | 7, 8 | five pools in which `order` is comparable — the corpus §2a is for — plus their verdicts |
| **9b** | Pipeline runs, the exact-sampler four (N02, N05, N08, N09), both arms | 9a | the recognition column, and the second structural holdout |
| **9c** | Re-rank both groups offline against the C0 baseline with `rank_study.py`, and price the pool-relative order rule on them | 9a, 9b | prediction 8 scored; the rule shipped, or withdrawn with a number |
| **10** | Write the result up: the §3 gap table against the outcome, and §10's predictions scored | 9c | the paper's table and the plan's own revision |
| **11** | **Optional** — N10 (Duffing–van der Pol) and its two-family reference | 9c | the tenth entry, or a recorded decision not to |

Step ordering has one non-obvious constraint, learned from `plan-sciml` step 6: **calibration before
authoring**, because three of nine SciML entries were vacuous until calibration found them, and an
entry authored first is an entry a later measurement has to argue with.

---

## 9. Cost

Rough, and in the units the other plans use.

- **Offline derivation and calibration** (steps 1–6): the dominant *human/agent* cost, and almost no
  model cost. Nine truths, nine correct schemes, eleven controls, all numpy. N03's Kolmogorov solve
  and N06's exact-solution MC are the two that take real compute, and both are minutes, not hours.
- **Harness work** (steps 3–4): H-N2 is a one-line change; H-N1 and H-N3 are each localised to one
  function plus tests; H-N4 is the largest and touches `validate_ground_truth.py` only.
- **Graded runs** (steps 8–9): nine problems × two arms × (pipeline + baseline). The SciML
  precedent is the closest anchor — five problems completed in 80.0 minutes of wall clock at a
  7200 s timeout — and this suite's entries are cheaper per problem than the BENCHOP ones (no
  400 000-path barrier problems). N01 is the exception: a pathwise ladder at `62500` fine steps,
  and step 2 measures its wall clock before step 9 commits to it.
- **What is cheap here and was not elsewhere:** no Julia environment (`plan-sciml` §3), no
  re-derivation of published values (`plan-benchop-slv`), no new reference-field machinery — because
  N10, the only entry that needs it, is phase-gated.

---

## 10. Predictions, to be checked against the outcome

Written before any run, so the plan can be wrong in public. The convention is `plan-sciml` §10's.

1. **The pipeline passes N09, N05 and N04 and does well on N02** — N09 and N05 because the exact law
   is reachable from `sde_manual.md` rule 0, N04 because the boundary rule (rule 5) points at it, N02
   because the Stratonovich circle is such a strong hint that even a manual silent on the word may be
   recovered from. **Confidence: moderate.**
2. **N01 fails, and fails because of `sde_manual.md:38`** — not because the Lévy area is hard, but
   because the manual instructs EM for all multi-D problems. The failure mode to look for is a plan
   that *names* the Lévy area and declines it, citing the manual. **Confidence: high.** If this is
   right it is the suite's most useful single result, and the fix is a manual change, not a pipeline
   change.

   **Sharpened `[rev1 2026-11-13]`:** `plan-sciml-benchmark.md` §10f measured that the pipeline's
   deficit against Opus cold was **selection, not generation** — as shipped 1.412× worse, while the
   best plan *in its own pools* was 2.65× better, one pool holding a plan 625× better than the
   baseline. So the prediction splits, and the suite should report which half binds: either the pool
   never contains a Lévy-area plan (generation, i.e. the manual) or it contains one and the ranker
   ships something else (selection, i.e. §2a). The second is now the more likely of the two on the
   project's own evidence, and it is the one the manual fix would *not* address.
3. **N07 is the coin flip.** Either a plan proposes a splitting scheme unprompted — which would be
   the strongest result in the suite — or every plan proposes EM and the entry measures the step
   budget. **Confidence: none, deliberately.**

   One risk here was removed while this plan sat unimplemented: SciML rev 18 found `crn.guards`
   judging the strong order against a **two-sided** band, so a plan was capped for converging
   *faster* than its floor — M01's best plan scored 8 at order 0.9831 against a 0.5 floor while the
   two it beat by 2.1× scored 10. N01's correct scheme (order 1.0 against Euler–Maruyama's 0.5) and
   N07's splitting are both exactly that shape, so **both would have been false-failed by the gate
   as it stood when this plan was drafted.** Fixed; the gate's one true positive — a Milstein with
   its correction dropped, at 0.504 against 1.0 — is a shortfall and survives.
4. **`wrong_plan_won` will be true on at least two entries, and at least one will be decided below
   `score`** — the SciML pattern, where on three of three probed problems no score separated
   anything and the winner was chosen by a quality index or by declaration order
   (`plan-sciml` §10a–§10d). The entries at risk here are N03 and N08, whose correct and wrong
   schemes differ by NaN-vs-number rather than by a graded error.
5. **At least one entry's answer will be produced by recalling a formula rather than integrating**,
   and the degeneracy fold will say so. N02 and N09 are the candidates. **Confidence: high** — and
   this is a prediction about the *measurement working*, not a failure.
6. **No `OVERCLAIM` anywhere.** Eight of the nine Phase-1 truths are exact, and the kernel cannot
   certify against them; the SciML suite produced no overclaim on eight completed entries.
   **Confidence: high.** An overclaim here would be the single most valuable finding in the plan.
7. **The vacuity guard (H-N4) refuses at least one of the 66 existing entries.** The guard is new,
   the rule it enforces is not, and `plan-sciml` §2a.1 found three vacuous entries in nine.
   **Confidence: moderate.** If it fires, the finding belongs in this plan and the fix belongs in a
   revision of whichever plan authored the entry.
8. **The pool-relative order rule (§2a) leaves the exact-sampler four bit-identical and moves at
   least one of the order-bearing five.** The first half is structural, for the same reason the rule
   leaves BENCHOP alone, and a violation would mean the rule does something nobody has
   characterised. **Confidence: high on the first half, none on the second** — which is the point of
   running it on a corpus it was not fitted to. `[rev1 2026-11-13]`

---

## 11. Deliberately not done

- **Shifted twins.** §0b: they defend against a recalled *number*, and this suite is graded against
  keys computed here. The two arms and the degeneracy reading replace them.
- **A jump channel on the pathwise route.** N05 needs no harness change because the solver draws its
  own jumps (§2, N05). Grading a *strong* error against a jump path would mean injecting a Poisson
  stream through the sandbox child alongside `dW` — a real extension of the increment plumbing
  `plan-sciml` H1 built, for one entry, measuring a quantity nobody publishes for this model.
- **Stochastic Lorenz 63.** The recognition is excellent and the truth route is not: a chaotic
  system's finite-time statistics need a reference whose own error is below the gate, and
  `pde_kuramoto_sivashinsky` only got one because two spectral integrators agreed to `1.2e-9`. For an
  SDE the Monte-Carlo floor sits far above that. Revisit only with an invariant-measure framing, and
  then it is N07's mechanism with a worse truth route.
- **α-stable / heavy-tailed drivers.** Out of scope for a Brownian harness; the repo already dropped
  `alpha-CIR` once (memory, 2026-09).
- **SPDEs** (stochastic heat, stochastic Allen–Cahn). The request is SDE-only, as `plan-sciml` rev 1
  settled for that suite.
- **Lookback and discretely monitored barrier options.** Notable (Giles 2008), but the mechanism —
  monitoring bias at `O(sqrt(dt))` — is already the content of `sde_heston_dnt` and `sde_qlsv_dnt`,
  and rule 5 of §1 forbids a repeat.
- **Fixing `sde_manual.md`.** Prediction 2 says the manual is the ceiling on N01, and the temptation
  will be to add Lévy-area guidance before the run. That would make the measurement unreadable: the
  suite's value is in showing what the shipped manual does. The manual change belongs to whatever
  plan follows this one, with the before-and-after measured — which is the structure
  `docs/improving-autonumerics/` already uses.

---

## 12. What the paper may claim, and what it may not

Stated here so the write-up in step 10 cannot drift.

**May claim**, if the runs support it:

- Ten (nine, if N10 is deferred) SDE problems named and cited from the primary literature, spanning
  nine distinct numerical mechanisms, each with an answer key authored and independently validated
  outside the pipeline, and each gate calibrated to clear a correct scheme and fail a scheme that
  differs from it by exactly one decision.
- The pipeline's verdict on each, against the same C0/C1/C2 baselines, on the suite's own
  denominator, with the integrate-vs-recognise split shown. **Quote the rate over
  `_VERIFIED_VERDICTS`, not `_PASS_VERDICTS`** `[rev1 2026-11-13]`: SciML rev 15 found the old set
  was a conjunction — the harness passed *and* the pipeline claimed 10 — while the baseline it is
  compared against has no self-score at all, so `UNDERCLAIM` counted as a failure on one side and
  was a verdict the other side could not receive. On that arm it read 5/9 against the one-shot's 9/9
  where the harness independently passed **9 of 9 on both**. An `UNDERCLAIM` is the pipeline being
  too modest about a success the harness confirmed, which is the one error direction this benchmark
  exists to reward.
- `wrong_plan_won` and the kernel-vs-harness cross-check, which is the thing this benchmark exists to
  produce and which a conventional benchmark table does not contain.

**May not claim:**

- Any **cost or work-axis** comparison on N01 or N07 without the caveat `[rev1 2026-11-13]`.
  `plan_meta.SDE_FAMILIES` is `euler-maruyama, milstein, tamed-euler-maruyama, tamed-milstein,
  implicit, exact-transition, quadratic-exponential, other` — verified 2026-11-13 — with no key for
  a stochastic Runge–Kutta, a splitting scheme, a Stratonovich midpoint, a jump-adapted step or a
  boundary projection. A multi-stage scheme therefore declares `other` and its work axis **falls
  back to one stage**, understating its own cost. That bias *flatters* N01's Lévy-area Milstein and
  N07's BAOAB, which is the direction a reviewer will not forgive, so either the vocabulary is
  extended first or the cost figures carry the caveat as a lower bound — which is how
  `report.py` already labels it on the SciML suite.
- That passing these ten shows the pipeline handles *hard* SDEs. §0a: this suite is selected for
  recognition. The hardness claim rests on the tier-3 set and the discriminator subset, and the two
  results should appear in the paper as two tables, not one.
- That a pass on N02, N05, N08 or N09 is a numerical-integration result where the record says the
  solver used an exact law. §7.
- That any figure here is comparable to a published benchmark value. Unlike BENCHOP, none of these
  keys was published by a third party; what is external is the *problem*, not the answer.
- Any suite-wide rate quoted against the main suite's 49-problem denominator. §0d.

---

## 13. Open questions for the project owner

1. **Nine or ten.** N10 (Duffing–van der Pol) is the only entry needing a new reference route. Ship
   nine and name the tenth as deferred, or spend the extra step?
2. **N06's citation vs its truth route.** If step 2 finds the SIS explicit solution does not reduce
   to a samplable form, the fallback is the stochastic Verhulst equation — same mechanism, textbook
   citation instead of a 2011 SIAM paper. Accept the fallback in advance, or drop the entry and run
   nine?
3. **Both arms, or `scheme-matched` only?** Both doubles the graded-run cost and produces the split
   §7 argues is the most interesting number in the suite. The SciML precedent ran both.
4. **Does the paper want the §3 gap table?** It is the most publishable thing in this plan and it is
   a table of this repo's own manual's omissions.
