# BENCHOP-SLV — External Benchmark Adoption Plan

**Status:** **steps 1-9 implemented 2026-10-21**; **step 10 in progress** — four of
its ten pipeline runs graded, the C0 baseline not started; step 11 planned. Research
and feasibility measurement done 2026-10-16. The suite exists: **ten SDE problems**
in their own `benchop` suite, seven anchored on published values and three shifted
controls whose truth the harness computes. **The PDE half was dropped** (step 7 rev,
and §0) — this is an SDE-only adoption, so §0's point (c) is not being collected.
Every gate is now calibrated **both ways** — a correct scheme clears it and a
plausible wrong one fails it, measured at the resolution each entry ships with —
and no gate was moved to make that true. The suite is staged and smoke-tested: the
ten briefs agree with the verifier that grades them, and the four defects step 9 found
— a manifest that did not name the suite, two runners that would have merged it into
the standing baseline, a work-precision point already paid for and discarded, and a
mislabelled ranking metric — are fixed. What remains is the runs. Three things are
deliberately undone and named in §8: `HHWeuCallI` rests on one derivation family
rather than two, so whether its one-sided residual is the sampler's or the answer
key's is unsettled; `sde_sabr_cev_call_ii`'s gate separates a *wrong* scheme by four
orders of magnitude but lands within 1% of a *generic* one's bias; and MLMC's
reading of `num_paths` is still unmeasured. ~~What remains is the runs.~~ The runs
are under way, and have already cost two more harness defects — see step 10's
in-progress record.
**Scope:** adopting the BENCHOP-SLV option-pricing benchmark (8 problem functions,
24 published reference values) as an externally-anchored suite alongside
`benchmark/problems.py`, plus the two harness changes it requires — solver timing
instrumentation and a functional tolerance convention.
**Related:** [plan-no-closed-form.md](plan-no-closed-form.md),
[plan-sde-verification-gaps.md](plan-sde-verification-gaps.md),
[`benchmark/README.md`](../benchmark/README.md)

Every number in §2 was measured while writing this plan, not quoted. The scripts are
reproducible from the formulas given; nothing here rests on recall.

---

## 0. Why an external benchmark at all

The repo's existing argument is about circularity. The pipeline grades itself — the
formulator writes down the analytic solution and the evaluator scores against it — so
`benchmark/problems.py` exists to hold an independently-authored answer key the agents
never read, and `pipeline-settings.json` enforces that they cannot.

That closes the loop one notch. It does not close it completely, because the answer key
and the pipeline share an author and a repo. **A published benchmark closes it the rest
of the way**: the reference value was computed by a third party, before this project
existed, by methods with no relationship to anything here. If a pipeline answer matches
it, no amount of shared-authorship skepticism explains the match away.

BENCHOP-SLV was selected over the other candidates surveyed because it is the only one
that is simultaneously (a) fully open — problem formulations *and* reference values to
15 digits, (b) stated as equations rather than shipped as experimental data, so it fits
`solve_pde(N)` / `solve_sde(...)` without a modelling layer in between, (c) solvable by
*both* halves of the pipeline, since every problem has a PDE formulation and an SDE
formulation and BENCHOP itself compares the two, and (d) aligned with capability the
repo already has — `pde_heston_2d`, `pde_black_scholes_call`,
`sde_log_heston_feller_violated` are all in the same family.

Point (c) is the one that matters most. The repo's cross-plan consensus machinery
(verification manual §11 and §22, `cli.py compare`) currently checks two plans against
*each other*. On BENCHOP it can check a Monte-Carlo plan and an ADI plan against each
other **and** against an external number, on the same problem.

> **[rev, step 6]** Point (c) holds for seven of the eight, and **fails for
> `QLSVeuCall`** — measured, not suspected. With `f(s) = 0.01s²` the asset is a strict
> local martingale, the pricing PDE therefore has more than one solution, and the
> published values follow the one the usual linear far-field condition selects, which
> is *not* `E[(S_T−K)^+]`. An honest sampler of the stated SDE converges 0.24 below the
> published value at S₀ = 125 — 0.8% relative, eight times the gate. §6 step 6 records
> the closed-form arbiter that settles which is which. The consequence for step 7 is
> that this one problem is a PDE problem or it is nothing: posing it to both halves
> would fail a correct Monte-Carlo solver against a reference it cannot reach.

> **[rev, step 7]** ~~Point (c) is the one that matters most.~~ **Point (c) is not
> being used.** The suite as built is **SDE-only** — ten problems, no PDE entries —
> so the cross-formulation comparison this section argued for is not what the suite
> delivers, and the paragraph above should be read as the case for a benchmark that
> *could* support it rather than a description of one that does. What survives
> intact is (a), (b) and (d), and they are the whole of the argument that still
> applies: the reference values are open, third-party and fifteen digits, the
> problems are equations rather than data, and the families are ones this repo
> already has. The external anchor is the point; the second formulation was a
> bonus, and it is not being collected.
>
> Two consequences follow. The memorization defence can no longer be the PDE order
> check §5 leans on — see the rev there. And `QLSVeuCall` drops out entirely rather
> than becoming PDE-only, leaving seven published problems; the suite reaches ten
> with the three shifted twins.

---

## 1. What BENCHOP-SLV is

A 2018 benchmarking project from Uppsala University (von Sydow, Milovanović, Larsson,
in 't Hout, Wiktorsson, Oosterlee, Shcherbakov, Wyns, Leitao, Jain, Haentjens, Waldén),
published in *International Journal of Computer Mathematics*,
DOI [10.1080/00207160.2018.1544368](https://doi.org/10.1080/00207160.2018.1544368).
It compares eight numerical methods on stochastic- and local-volatility option pricing,
scoring **error against CPU time**.

### Sources

| What | Where | State |
|---|---|---|
| Problem formulations (5 pp.) | `https://it.uu.se/research/scientific_computing/project/compfin/benchop/SLV/benchop_slv_0622.pdf` | open; authoritative spec |
| **Reference values** | `https://it.uu.se/research/scientific_computing/project/compfin/benchop/SLV/unitTestBenchopSLV.m` | open; plain text, 15 digits |
| Paper | `https://ir.cwi.nl/pub/28249/28249.pdf` | open |
| Landing page | `https://it.uu.se/research/scientific_computing/project/compfin/benchop/SLV.html` | open |
| The eight methods' code | same page, `*.zip` | **encrypted MATLAB p-code** — runnable, not readable |

The reference values are *not* in the paper or the formulations PDF. They are in the
MATLAB unit-test harness, as `pr(i).refval`. Results in the paper are error-vs-CPU-time
plots only.

### The eight problem functions

Each returns a 1×3 row vector. 24 reference values total.

**SABR** (forward SDE, no drift; `dS = σS^β dW_S`, `dσ = ασ dW_σ`, corr ρ).
Strikes `K_i = S(0)·exp(0.1·√T·δ_i)` for `δ_i = −1, 0, 1`. European call.

| Set | Parameters | Strikes | Reference values |
|---|---|---|---|
| I | T=2, r=0, S₀=0.5, σ₀=0.5, α=0.4, β=0.5, ρ=0 | 0.434061722697, 0.5, 0.575954955084 | 0.221383196830866, 0.193836689413803, 0.166240814653231 |
| II | T=10, r=0, S₀=0.07, σ₀=0.4, α=0.8, β=0.5, ρ=−0.6 | 0.051022538988, 0.07, 0.096035989138 | 0.052450313614407, 0.046585753491306, 0.039291470612989 |

**Quadratic local stochastic volatility.**
`dS = rS dτ + √V f(S) dW¹`, `dV = κ(η−V)dτ + σ√V dW²`, `f(s) = ½αs² + βs + γ`.
Heston is `α=0, β=1, γ=0` (so `f(s)=s`); QLSV is `α=0.02, β=0, γ=0` (so `f(s)=0.01s²`).
Shared: **T=1, r=0, κ=2.58, η=0.043, σ=1, ρ=−0.36, V₀=0.114**, spots S₀ = 75, 100, 125.

| Function | Contract | Reference values |
|---|---|---|
| `HSTeuCall` | European call, K=100 | 0.908502728459621, 9.046650119220969, 28.514786399298796 |
| `HSTdnTouch` | double-no-touch, L=50, U=150 | 0.834539127387590, 0.899829293208866, 0.668399975738358 |
| `QLSVeuCall` | European call, K=100 | 0.527472759419533, 8.902347915743665, 29.159828965633729 |
| `QLSVdnTouch` | double-no-touch, L=50, U=150 | 0.933800903110254, 0.914799140676374, 0.592983062889906 |

**Heston–Hull–White** (3-D: asset, variance, short rate).
`dS = RS dτ + √V S dW¹`, `dV = κ(η−V)dτ + σ₁√V dW²`, `dR = a(b(τ)−R)dτ + σ₂ dW³`.
Shared: **T=10, κ=0.5, η=0.04, σ₁=1, σ₂=0.09, ρ₁₂=−0.9, a=0.08, b(τ)≡0.10**,
V₀=0.04, R₀=0.10, K=100, spots S₀ = 75, 100, 125.

| Function | Correlations | Reference values |
|---|---|---|
| `HHWeuCallI` | ρ₁₃=0.6, ρ₂₃=−0.7 | **37.52, 57.41, 78.58 — two decimals only** |
| `HHWeuCallII` | ρ₁₃=ρ₂₃=0 | 35.437896876285350, 54.728065308229503, 75.397596834993621 |

**The hardest problem has the weakest reference.** `HHWeuCallI` — full correlation,
genuinely 3-D — is quoted to 1e-2, annotated `%1E-02` in the source. Everything else
carries 15 digits.

**The Feller condition is violated in every problem**, which the formulations state
explicitly as a note. Measured: LSV set `2κη = 0.221880` against `σ² = 1`; HHW
`2κη = 0.04` against `σ₁² = 1`. The variance process hits zero, and a naive `√V` Euler
goes negative.

### What the formulations also say

- SABR set I has `ρ = 0`, for which an exact simulation formula exists (Islah 2009).
- Time-discretised MC on SABR **loses the martingale property**; a correction is required.
- For the LSV set, semi-closed formulas exist only when `α=0, β=1, γ=0, ρ=0`. Here
  `ρ = −0.36`, so the formula covers the European call (whose CF route works for any ρ)
  but **not** the double-no-touch.
- For HHW, `ρ₁₃ = ρ₂₃ = 0` admits "a semi-closed analytic formula akin to Heston."
- HHW may be reduced to a 2-D PDE with time-dependent coefficients; the formulations
  ask that you state which you solved.

---

## 2. What was measured before planning this

### 2a. The reference values are real — two independent derivations

**`HSTeuCall`**, via the Heston characteristic function in Albrecher "little trap" form,
integrated with `scipy.integrate.quad`:

| S₀ | independent derivation | BENCHOP reference | abs diff |
|---|---|---|---|
| 75 | 0.908502728455683 | 0.908502728459621 | 3.94e-12 |
| 100 | 9.046650119218924 | 9.046650119220969 | 2.04e-12 |
| 125 | 28.514786399294692 | 28.514786399298796 | 4.10e-12 |

**[rev, step 6]** The 4e-12 was *this derivation's* quadrature error, not BENCHOP's.
The same characteristic function fed to Lewis' **single**-integral form at the same
tolerance agrees to **2.7e-15, 2.7e-14, 5.3e-14** — machine precision on values of
order 1 to 30. The two-integral `P1`/`P2` form evaluates two oscillatory integrals and
differences them; the single-integral form does not, and the three orders of magnitude
are the difference between the two formulations rather than anything about the
reference.

**`HHWeuCallII`.** With ρ₁₃=ρ₂₃=0 the Hull–White rate is independent of (S,V), so
writing `M = ∫₀ᵀ R dτ` — Gaussian, measured `mean 1.0000000000, var 1.5457975016` —
the price collapses to `E_M[ ZeroRateHestonCall(S₀, K·e^{−M}) ]`, a 1-D Gauss–Hermite
quadrature over Heston calls at shifted strikes:

| S₀ | independent derivation | BENCHOP reference | abs diff | rel diff |
|---|---|---|---|---|
| 75 | 35.436815841033734 | 35.437896876285350 | 1.08e-03 | 3.05e-05 |
| 100 | 54.712482227869124 | 54.728065308229503 | 1.56e-02 | 2.85e-04 |
| 125 | 75.395112101906307 | 75.397596834993621 | 2.48e-03 | 3.30e-05 |

~~The route is sound; reaching the reference's own precision needs tighter quadrature
(140 Gauss–Hermite nodes over a σ=1.24 Gaussian, with the inner Heston integral
evaluated at strikes spanning `K·e^{−1±6}`, is not enough).~~ **[rev, step 6]** The
route is sound and this diagnosis was wrong. Tightening the quadrature does not help:
measured 2026-10-17, the Gauss–Hermite route still sits at **3.9e-5 relative at 320
nodes** and is no longer converging, because the error is in the *inner* Heston call at
strikes spanning nine orders of magnitude, not in the outer node count.

The fix is not a tighter quadrature but one fewer of them. In Lewis' integrand the
strike enters only as `√K'` and `e^{iu·ln(S₀/K')}`, so substituting `K' = K·e^{−M}` and
taking the expectation over `M` *inside* turns the whole Hull–White factor into the
Gaussian moment generating function `E[e^{zM}]` at `z = −½ + iu`. One integral instead
of two — and its integrand now carries `e^{−Var(M)·u²/2}`, so it terminates by `u = 12`
instead of needing the far tail. Measured agreement with the published values:
**5.7e-14 absolute, 1.4e-15 relative**, which settles §8's "needs tightening from 3e-5"
outright.

### 2b. Feasibility — the PDE route works inside the sandbox

`HSTdnTouch` solved by implicit finite differences on (S,v), upwinded v-drift,
central cross term, Dirichlet 0 at both barriers, degenerate `u_τ = κη u_v` at v=0.
**numpy + `scipy.sparse` only** — exactly the [solver-pde.md](../agents/solver-pde.md)
import grant:

| grid (N_S × N_v × N_t) | S₀=75 | S₀=100 | S₀=125 | max abs err | time |
|---|---|---|---|---|---|
| 100 × 40 × 100 | 0.835804 | 0.896819 | 0.684614 | 1.62e-02 | 0.0 s |
| 200 × 80 × 200 | 0.835279 | 0.898471 | 0.676351 | 7.95e-03 | 0.3 s |
| 400 × 160 × 400 | 0.834929 | 0.899212 | 0.672123 | 3.72e-03 | 3.2 s |
| *reference* | 0.834539 | 0.899829 | 0.668400 | | |

Clean first-order convergence toward the published value, well inside the 120 s sandbox
cap. A Crank–Nicolson/ADI scheme — what BENCHOP's own best PDE entry used — should
clear 1e-3 relative comfortably. **The monotone refinement is what makes the existing
2-grid order check a working anti-hardcoding defence** (see §5).

### 2c. Feasibility — the SDE route reaches 1e-2, not 1e-3

Full-truncation Euler on (S,V) for `HSTeuCall`:

| S₀ | paths × steps | MC price | abs err | MC stderr | time |
|---|---|---|---|---|---|
| 75 | 200k × 250 | 0.910321 | 0.0018 | 0.0126 | 1.1 s |
| 75 | 1M × 500 | 0.910107 | 0.0016 | 0.0054 | 11.0 s |
| 100 | 200k × 250 | 9.083097 | 0.0364 | 0.0342 | 1.1 s |
| 100 | 1M × 500 | 9.059893 | 0.0132 | 0.0152 | 11.0 s |
| 125 | 200k × 250 | 28.576609 | 0.0618 | 0.0568 | 1.1 s |
| 125 | 1M × 500 | 28.531227 | 0.0164 | 0.0253 | 11.2 s |

The error sits inside the MC standard error at 1M paths, so the scheme is unbiased at
this step — the limit is sampling. Reaching 1e-3 needs ~100× the paths, roughly 20
minutes, **outside the 120 s default in [runner.py:354](../benchmark/runner.py#L354)**.
This matches BENCHOP's own finding that Monte Carlo is competitive only on the 3-D problem.

### 2d. `HSTdnTouch` is the best discriminator in the suite

Discrete barrier monitoring biases MC **high**, systematically, at the classic
Broadie–Glasserman–Kou rate of O(√dt):

| S₀ | 200k × 250 steps | 200k × 1000 steps | reference |
|---|---|---|---|
| 75 | 0.844560 (err 0.0100) | 0.838710 (err 0.0042) | 0.834539 |
| 100 | 0.907230 (err 0.0074) | 0.902585 (err 0.0028) | 0.899829 |
| 125 | 0.687375 (err 0.0190) | 0.677655 (err 0.0093) | 0.668400 |

Quadrupling the steps roughly halves the error — √dt, converging far too slowly to ever
clear a tight gate by brute force.

> **[rev, step 8]** Re-measured at the resolution the entry actually ships with
> (400 000 paths × 500 steps, against the 6e-3 gate): the bridge reaches **0.35×**
> the gate and per-step monitoring **3.6×**, so the gate sits between them as this
> section predicted. One thing it did not predict is *where*: monitoring misses on
> **two of the three spots**, not all three, and the one that does the
> discriminating is S₀ = 125 — nearest the upper barrier, where survival is lowest
> and the bias largest. On the quadratic-local-volatility sibling it misses on one
> of three. That is enough, because the harness fails a problem when any reported
> value misses its gate, but it is worth knowing that the discriminator is a single
> value rather than the problem as a whole.

The fix is a *named technique*: the Brownian-bridge
crossing correction. That makes this problem structurally the same kind of test as the
tamed scheme on `sde_quintic_drift_noise`, the Il'in fitting on
`pde_convection_diffusion_bl`, and Godunov on `pde_burgers_inviscid` — a problem where
the generic answer is wrong and a specific, nameable one is right. It is also a case
where the PDE route beats MC outright, which is what the cross-plan machinery is for.

### 2e. `pde_heston_2d` is a weaker sibling of `HSTeuCall`

The existing problem uses κ=2.0, θ=0.04, σ=0.3, ρ=−0.7, r=0.03, K=1 on S ∈ [0,4] —
`2κθ = 0.16` against `σ² = 0.09`, so **Feller is satisfied**. BENCHOP's is the same
operator with the condition violated and σ=1, graded at three points instead of over the
whole surface. Adopting it is a parameter change plus `functional_truth`, not a new
problem family.

---

## 3. How it maps onto the harness

BENCHOP's answer is a **scalar price at three points**, to a tolerance. Not a field, not
a terminal moment.

| Need | PDE side | SDE side |
|---|---|---|
| Scalar quantity of interest | `functional_truth` **exists** ([problems.py:2398](../benchmark/problems.py#L2398), `pde_schrodinger_eigen_2d`) | ~~**missing** — contract returns terminal moments only~~ **done 2026-10-17, step 5**: `functionals` is in the SDE contract, and `moments` stopped being required, so a problem may be scored on a price alone |
| All three values must pass | `_pde_converged` already requires every functional within tol | **done**, same step — `_score_sde_functionals` requires every one within its own gate, through the same two helpers |
| Anti-hardcoding defence | 2-grid order check: a constant shows zero convergence and fails | ~~**none** — no refinement ladder on a price~~ **done**, same step: the price is taken at two path counts and must move, and must move by less than the sampling noise the two counts imply |
| Scalar QoI survives the sandbox | ~~**no** — `runner.py` dropped `functionals`, so every sandboxed functional solve failed `contract`~~ **fixed 2026-10-17, step 2** | ~~n/a until step 5~~ **done**, step 5 — both result shapes now go through one `_extract_functionals`, so the two cannot drift apart again |
| Per-problem tolerance | ~~`_pde_functional_tol` — one relative number for every value~~ **done 2026-10-17, step 4**: still per problem, but floored per *value* by the reference's own absolute uncertainty (`reference_abs_err`) | **done**, step 5 — the same helper, renamed `_functional_tol` because it is no longer PDE-only; only the default differs (5% sampled moment, not 1% resolved field) |
| Solve timing | ~~`wall_s` only (whole subprocess, incl. import)~~ **done** — `solve_s` beside it, step 1 | ~~same~~ **done**, same step |
| Error-vs-time curve | ~~not produced — the ladder's errors and its clocks were never joined~~ **done 2026-10-17, step 3**; free at the graded grids, extended by `work_precision_levels` | **done**, same step: a graded anchor plus declared `(num_paths, dt)` levels — but the SDE curve is sampling-noise-dominated and non-monotone |

~~Two gaps to close, and they are the first two implementation steps: **timing**
and **SDE functionals**.~~ **Both closed 2026-10-17** — timing in step 1, SDE
functionals in step 5. Everything else is problem authoring.

~~One naming hazard: on a functional-only problem,
[verify.py:797](../benchmark/verify.py#L797) stores a *relative* error in a field called
`max_abs_err`. Fix that before introducing a genuine absolute error or the two will be
silently conflated.~~ **Fixed 2026-10-17, step 4.** It was worse than "a hazard": the
field named for an absolute error held a relative one on the functional route and an
absolute one on the field route, so its meaning depended on which truth route the problem
took — and on the functional route it was the *primary* value's relative error, so on a
problem with several quantities of interest it did not even describe the worst one.
Measured on the unpatched verifier: two functionals with references 1.0 and 10.0, the
solver 0.01 and 0.5 off them, recorded `max_abs_err = 0.01` against a true largest
absolute deviation of 0.5. Nothing read the field, which is why it survived.

---

## 4. Tolerance: absolute or relative

BENCHOP gates on **absolute** error (`err = abs(Uref-U); if max(err) > tol`). The
harness is **relative** everywhere — PDE L2, SDE moments, `_functional_errs`, and both
scoring rubrics. The decision matters because the 24 values span 0.0393 to 78.58.

What each convention demands:

| Function | range | abs 1e-3 ⇒ relative | rel 1e-3 ⇒ absolute |
|---|---|---|---|
| `SABReuCallI` | 0.166–0.221 | 4.5e-3 … 6.0e-3 | 1.7e-4 … 2.2e-4 |
| `SABReuCallII` | 0.039–0.052 | 1.9e-2 … 2.6e-2 | 3.9e-5 … 5.3e-5 |
| `HSTeuCall` | 0.909–28.51 | 3.5e-5 … 1.1e-3 | 9.1e-4 … 2.9e-2 |
| `HSTdnTouch` | 0.668–0.900 | 1.1e-3 … 1.5e-3 | 6.7e-4 … 9.0e-4 |
| `QLSVeuCall` | 0.527–29.16 | 3.4e-5 … 1.9e-3 | 5.3e-4 … 2.9e-2 |
| `QLSVdnTouch` | 0.593–0.934 | 1.1e-3 … 1.7e-3 | 5.9e-4 … 9.3e-4 |
| `HHWeuCallI` | 37.52–78.58 | 1.3e-5 … 2.7e-5 | 3.8e-2 … 7.9e-2 |
| `HHWeuCallII` | 35.44–75.40 | 1.3e-5 … 2.8e-5 | 3.5e-2 … 7.5e-2 |

Three observations drove the decision:

1. **The 2000× range is between functions, not within them.** Six of eight have an
   internal spread of 1–2×. A per-problem tolerance handles that under either
   convention, so it is not the deciding factor.
2. **The decision is really about `HSTeuCall` and `QLSVeuCall`**, which span 31× and 55×
   internally because each pairs a deep-OTM call (0.91, 0.53) with a deep-ITM one
   (28.5, 29.2). Measured on the MC run in §2c, relative error was the more uniform
   measure: absolute errors spanned 10× across the three spots (0.0016 / 0.0132 /
   0.0164) while relative spanned 3× (0.18% / 0.15% / 0.06%).
3. **`HHWeuCallI` forces absolute reasoning somewhere.** Its ±0.005 rounding, under the
   repo's existing "widen to 3× the reference's own error" rule, is a floor of 0.015
   *absolute* — which is 1.9e-4 to 4.0e-4 *relative* depending on the spot. No single
   relative number expresses that, and no single absolute number suits the rest.

### Decision — **implemented 2026-10-17, step 4**

**Gate on relative, per problem, with a per-value absolute floor derived from the
reference's own uncertainty. Record both errors.**

```python
tol_i = max(functional_tol, 3.0 * reference_abs_err / max(abs(ref_i), 1e-12))
```

This is not a new pattern. It is what [`_score_sde_moments`](../benchmark/verify.py#L323)
already does for reference ground truth
(`var_tol = max(0.10, 3.0*se_v/max(abs(xv),1e-12))`), so the two tolerance surfaces stay
consistent.

Recording absolute error alongside is what lets a BENCHOP-comparable error-vs-time chart
be rendered without the *verdict* depending on the absolute number. It is also the only
way the absolute number survives at all: `functional_truth` is never written into a
record, so a relative error alone cannot be converted back.

Starting value: `functional_tol = 1e-3` relative for the Heston/QLSV family — §2b shows
a crude first-order scheme at 5.6e-3, so a good one should clear it. SABR needs
calibration: 1e-3 relative there means 3.9e-5 absolute, which §2c suggests is hard.

---

## 5. The memorization problem

[verify.py:721](../benchmark/verify.py#L721) states the repo's rule outright: a
functional's truth is "computed here rather than quoted from a paper, so a recalled
constant is not a passing answer." BENCHOP is nothing *but* published constants, online
since 2018, and `9.046650119220969` is exactly the kind of string a frontier model may
carry. This is the central governance tension in adopting it.

Two things resolve it, and one is already built.

~~**The order check defends the PDE route.** When functionals are the only truth,
`_pde_eval_run` makes the functional error the primary error, so the 2-grid order check
measures convergence *of the price itself*. A hardcoded constant has zero convergence
and fails. §2b's refinement table is what a real solver's trace looks like. **The SDE
route has no equivalent**, which is an independent reason to require a convergence check
in step 5 below, and to build the PDE route first.~~

> **[rev, step 7]** This paragraph was wrong twice over, and the suite is SDE-only
> now, so neither error reaches a shipped problem.
>
> **It did not describe the code.** `_pde_order_check` required a *field* ground
> truth, so on a functional-only problem — the shape every BENCHOP PDE entry would
> have had — the second grid was never run. Measured 2026-10-20 on a sandboxed
> solver returning a hardcoded constant: `passed=True`, **score 10**,
> `order_checked=False`. The defence this section leaned on did not exist.
>
> **And the fix is not the one the paragraph assumes.** "A hardcoded constant has
> zero convergence and fails" is false even with the check running: a constant has
> zero *error* at both grids, which trips the superconvergence waiver — the waiver
> that exists so a spectral scheme resting on its round-off floor is not failed for
> having no observable order — and hands it `observed_order = inf` and
> `order_ok = True`. Convergence and movement are different questions, and only the
> second catches a constant. Both are now checked; see step 7.

**Run a shifted-parameter control.** Pair each published problem with a twin at a
de-memorized parameter ~~(κ = 2.58 → 2.43)~~ whose truth the harness computes itself. A
baseline that scores well on the published set and badly on the shifted twin has been
caught recalling rather than solving. This converts the risk into a measurement, which
is what the benchmark is for. The twins also give the SDE route an anti-hardcoding
signal it otherwise lacks.

> **[rev, step 7]** The control is built, on three of the seven, and the parameter
> in the struck text above would not have worked. Measured: κ 2.58 → 2.43 moves the
> Heston call by 2.4e-2, 2.2e-3 and 1.9e-3 relative at the three spots, and a
> sampled price has to be gated at 1.5e-2, 4e-3 and 2e-3 — so at two of the three
> values the twin's answer moves **less than the tolerance**, and a recalled
> constant would pass it. A twin whose answer moves by less than the gate is not a
> control; it is a second copy of the same question.
>
> What is shifted instead is the **initial** variance or volatility, and nothing
> else: `V0` 0.114 → 0.164 for the Heston pair, `sigma0` 0.4 → 0.34 for SABR II.
> That keeps the model, the payoff, the spot, the strikes, the maturity and — for
> Heston — the Feller ratio 2κη/σ² = 0.2219 that makes the problem hard all
> bit-identical to the published sibling, so a solver that genuinely solves one
> solves the other at the same cost. Only the answer moves, and measured against
> the gates actually shipped it moves by **16–35×** the gate on the call, **7.4–18×**
> on the double-no-touch and **4.8–10×** on SABR — the last being the narrowest
> because SABR II's own gate is the loosest in the suite, for the reason step 7
> records.

---

## 6. Steps

**1. Time the solve call, not the process.** `run_solver` returns `wall_s`, but that is
the whole subprocess including interpreter startup and imports — noise in a
work-precision diagram. In [`_child_main`](../benchmark/runner.py#L261), wrap the two
`res = fn(...)` sites (PDE ≈ line 300, SDE ≈ line 317) and record
`run_meta["solve_s"]`. Keep `wall_s`; the difference is the overhead measurement.

> **Implemented 2026-10-17.** `_timed` in [runner.py](../benchmark/runner.py) wraps both
> call sites — three in fact, since the PDE branch has `fn(N=N)`, its `TypeError`
> fallback `fn(N)`, and the legacy `fn()` — and `run_solver` returns `solve_s` beside
> `wall_s` as well as inside `run_meta`. `perf_counter`, not `monotonic`: the same
> guarantee at finer resolution, and a coarse grid solves in under a millisecond.
>
> **The premise checks out, and by more than the step assumed.** Measured on a
> Crank–Nicolson `pde_heat_1d` through `runner.run_solver` (medians of 3, idle
> machine, 2026-10-17): the solve is **0.0053 s inside a 0.200 s subprocess at N=64**
> and **0.0228 s inside 0.220 s at N=128** — 97% and 90% overhead. A work-precision
> diagram drawn against `wall_s` would have been a picture of the harness; worse, it
> would have shown the 4× solve between the two grids as a 1.1× *total*, flattening the
> curve the diagram exists to show. The SDE side is the same shape: `sde_gbm`
> Euler–Maruyama, 50 000 paths at dt=0.01, **0.0449 s of solve inside 0.130 s** — 66%,
> lower only because the SDE sandbox denies scipy, so the child has less to import.
>
> **One thing the step under-specified.** Recording `run_meta["solve_s"]` is enough on
> the SDE side, where `run_meta` is injected into `verify_problem` and lands in the
> record as `verify.run.solve_s`. On the PDE side `_produce_pde_runs` kept only
> `run1["result"]` and threw `run_meta` away, so the number was measured and discarded
> — on the very route §5 says to build first. It now returns `(runs, timings, err)` and
> `_fold_timing` writes `verify.timing = {path, solve_s, wall_s, runs: [...]}`: one
> clock per graded grid, with the fine grid's as the headline, mirroring how
> `rel_l2_err` sits beside `grid_errors`. That is the two free (error, time) points
> step 3 asks for, though not the extended ladder or the error pairing.
>
> **The inline path is deliberately left untimed**, and `path` is recorded rather than
> assumed, which is step 2 arriving early in the only form that costs nothing: a
> sandboxed time and a warm in-process time are different measurement conditions, and
> a field that silently mixed them is exactly the hazard step 2 names. Until step 2
> routes both through the sandbox, the honest pipeline-side record is no record.
> Covered by [`benchmark/tests/test_solve_timing.py`](../benchmark/tests/test_solve_timing.py)
> (13 tests): the clock starts after the import, it does see the solve, both contracts
> carry it, a crash and a timeout carry none, and the inline path stores nothing.

**2. Make pipeline and baseline timings comparable.** Baseline solvers run sandboxed;
pipeline solvers are imported and run inline (`sandbox=False` in `verify_problem`).
Those are different measurement conditions, so a pipeline-vs-C0 *time* comparison would
be measuring the harness. Either route both through `runner.run_solver` for scoring, or
record which path produced each number and never compare across them. Routing both
through the sandbox is cleaner.

> **Implemented 2026-10-17.** Routed, as the step preferred. `run.py` now grades through
> `runner.run_solver` — winner and also-rans alike, via one `_verify_kwargs` — and
> `verify.py`'s CLI defaults to it too, so a human spot-check cannot disagree with the
> recorded verdict. `--no-verify-sandbox` restores the inline path for debugging.
>
> **The step is one sentence; the work was almost entirely in earning the right to
> flip it.** Routing is only sound if the boundary is faithful, and it was not. Measured
> by verifying all 34 solved workspaces both ways before changing anything:
> **29 of 34 agreed, 5 did not** — and separately the one functional-scored problem was
> broken for *every* sandboxed solver. Three defects, all in what `runner.py` carried
> across the process boundary:
>
> | Defect | Symptom | Who it hit |
> |---|---|---|
> | `functionals` never crossed | `status: contract`, score 1, "solver returned no `functionals` dict" — for a solver that returned one | every sandboxed run of `pde_schrodinger_eigen_2d`, i.e. **every baseline on the only functional-scored problem** |
> | only one result shape crossed | `fields` kept, `numerical_solution` dropped; a *scalar* problem then raised `KeyError` → `inconclusive` | the 5 disagreeing workspaces: `pde_wave_1d`, `pde_laplace_2d`, `pde_anisotropic_diffusion`, `pde_cahn_hilliard_2d`, `pde_cahn_hilliard_2d_coarsening` |
> | a relative `plan_dir` was sent as-is | the child runs with `cwd=benchmark/`; `FileNotFoundError` | any caller not already using absolute paths |
>
> After the three fixes: **34 of 34 agree**, on status, pass, score, field error,
> functional errors, order and the structure gates.
>
> **The first defect has already corrupted a recorded result.** Both archived C0 Opus
> baselines on `pde_schrodinger_eigen_2d` returned `lambda_1` and both are recorded
> `FAIL (contract)`. Re-verified with the fix they are **PASS, 10/10** — `lambda_1`
> accurate to 1.7e-7 and 1.4e-13, 4th-order convergence, nodeless gate clean. So the
> C0 baseline on that problem is 2/2, not 0/2. That moves a number in the
> no-closed-form comparison *against* the pipeline, which is the direction that makes
> it worth reporting rather than quietly fixing. `benchmark/results/` is regenerable
> and gitignored, so the recorded JSONs are not edited here; the correction lands when
> that suite is next re-verified (`run.py --skip-run` / `oneshot.py --skip-agent`).
>
> **Sandboxing the pipeline costs 2.4%.** 184.7 s against 180.3 s inline across the 34
> workspaces, which is the number that makes "route both" the easy call rather than a
> trade. The per-solve overhead is a fixed ~0.2 s of interpreter and import (step 1),
> so it is paid per graded solve and not per unit of work.
>
> **One budget, not two.** `verify.DEFAULT_SOLVER_TIMEOUT_S` is now the single default
> for `run.py`, `oneshot.py` and the CLI, because two defaults in two files drift and a
> pipeline measured under one budget is not comparable with a baseline measured under
> another. It was **raised 120 → 300 s**: the slowest solve in the corpus is
> `pde_cahn_hilliard_2d` at N=128, **71.1 s**, so 120 left a 1.7× margin on a *passing*
> problem, where the failure mode is a correct solver recorded `CRASHED` on a slower
> machine. 300 s is 4.2×. No recorded run has ever hit a verification timeout
> (the only recorded crash reason across every results JSON is `no_solve_fn`, 72 of
> them), so nothing already measured is affected. `run.py` now writes
> `verify_sandbox`, `solver_timeout_s` and `mem_mb` into its results config, as
> `oneshot.py` already did — the `c0_max_turns` precedent.
>
> **What parity does *not* extend to.** Only the sandboxed path carries a clock, by
> design, so a sandboxed time can never be compared with an inline one. Covered by
> [`benchmark/tests/test_execution_parity.py`](../benchmark/tests/test_execution_parity.py)
> (17 tests), which pins the contract-key set against the list grepped out of
> `verify.py` and `verifylib/audit.py`, so a key added to the scorer without being
> added to the boundary fails rather than being silently dropped.

**3. Get the work-precision curve from the ladder already being run.** The PDE path
already solves at `N` and `2N` — two (error, time) points for free. Add
`work_precision_levels` (e.g. `[N/2, N, 2N, 4N]`) reusing the ladder in
[`_score_pde_runs`](../benchmark/verify.py#L907), and emit the series into
`results.json`. For SDE, sweep `(num_paths, dt)` the same way.

> **Implemented 2026-10-17.** Both halves were already being produced and kept
> apart: `_score_pde_runs` recorded a per-resolution error in `grid_errors` for the
> order check, `_produce_pde_runs` recorded a per-resolution clock (step 1), and
> nothing joined them. `verify.work_precision` is that join plus the optional sweep
> past it, with three properties that are each a test: it is **sandbox-only** (an
> inline time and a sandboxed time are different conditions, step 2), it is
> **measured, never modelled** (no interpolation and no fitted slope — a fitted
> exponent is a claim, the points are evidence), and **it cannot move a verdict**.
>
> **The free points are genuinely free, and universal.** Re-verifying all 34 solved
> workspaces with the sweep on and off: **34 of 34 identical verdicts**, 179.7 s
> against 180.5 s — **−0.4%, i.e. noise**, because with no problem declaring levels
> the two runs execute the same solves. And **34 of 34 now carry a series** without
> any problem declaring anything: 25 two-point (PDE with the order check on), 2
> one-point (`pde_burgers_inviscid`, `pde_kuramoto_sivashinsky`, order check off), 7
> one-point SDE anchors.
>
> **What the ladder sees that the graded check does not.** Driven at `[N/2, N, 2N,
> 4N]` on six real pipeline solvers, two findings came straight out of the extra
> points:
>
> | Problem | graded 2-grid order | ladder orders | what the extra points show |
> |---|---|---|---|
> | `pde_advection_1d` | **4.00**, pass | 3.89, 4.00, **2.71** | the order collapses at the finest doubling; the graded pair happens to sit on the one clean doubling |
> | `pde_helmholtz_2d` | **inf** (waived) | 4.09, 4.05, 4.03 | `ORDER_SUPERCONV_FLOOR` blanks the order at 2.6e-10, and the ladder recovers the 4.0 it declined to state |
>
> Neither is a harness bug — the 2-grid check is doing what it was built to do, and
> the superconvergence waiver is correctly conservative — but both are cases where
> *the number the verdict rests on is less informative than the number the ladder
> produces for two more solves*. That is the argument for the sweep, made on this
> repo's own solvers rather than on BENCHOP's.
>
> **A declared four-level sweep costs 2.2x–2.9x the verification wall time**
> (measured across those six), and the cost is almost entirely the `4N` solve: in
> *solve* seconds alone, `pde_heat_1d` goes from 0.148 s graded to 0.702 s swept,
> 4.7x. `--no-work-precision` turns off the extra solves on both runners and in the
> CLI; the free points survive it, because they are already paid for.
>
> **The SDE side came back with a caution for step 5.** Its curves are not monotone.
> On `sde_gbm_2d_high_corr`, 12 500 paths at dt=0.02 scores 8.1e-3 while 25 000 at
> dt=0.01 scores 1.5e-2 — *worse for twice the work* — and `sde_ornstein_uhlenbeck`
> flattens from 8.2e-3 to 7.3e-3 for a 4x spend. The error axis there is dominated
> by sampling noise, not discretization, so step 5's "price at two path counts,
> checked to be converging" **cannot be a simple decreasing test**: it would fail
> correct solvers at the rates the suite actually runs at. It has to be a check
> against the standard error, as `_score_sde_moments` already does for reference
> ground truth. **[rev, step 5]** That is how it shipped: a symmetric band of
> `tol_abs × (1 + √(n_graded/n_coarse))` around the two reported prices, plus the
> requirement that they differ at all. The caution held — nothing in the shipped
> check reads the sign of the difference.
>
> **Two defects found on the way, both pre-existing.** `_run_and_read_sde` decided
> whether a `(d, num_paths)` return was transposed by comparing against the
> *problem's* path count; a swept level varies that count, so the check missed and
> the array would have been scored component-for-path — silently, with plausible
> numbers. It now keys off the run that was executed. And step 1's `solve_s` was
> rounded to 4 decimals, which recorded a 7 ms coarse-grid solve as `0.0` — a time
> that cannot be plotted on a log axis; it is now rounded to the microsecond.
> Covered by [`benchmark/tests/test_work_precision.py`](../benchmark/tests/test_work_precision.py)
> (27 tests), including the blunt one: grade the same solver with the sweep on and
> off and require the two records to differ in exactly one key.


**4. Tolerance plumbing.** Per-value absolute floor in `_functional_errs`;
`reference_abs_err` carried on the problem; record relative *and* absolute per
functional; fix the `max_abs_err` naming slip.

> **Implemented 2026-10-17.** Three pieces of plumbing and one defect.
>
> **The gate.** `_pde_functional_tols(problem)` returns a tolerance *per value* —
> `max(functional_tol, REFERENCE_TOL_FACTOR x reference_abs_err_i / |ref_i|)` — applied
> in [`_pde_converged`](../benchmark/verify.py#L912) rather than inside
> `_functional_errs` as this step's text proposed. **[C]** The error function computes
> errors and the convergence check applies tolerances; folding a tolerance into the
> former would have put the gate in two places, which is the shape of defect this very
> step exists to clean up. `reference_abs_err` is declared as one number covering every
> functional or as `{name: abs_err}`; the dict form is what the Monte-Carlo reference
> route needs, since §2c measured absolute errors spanning 10x across three values of
> one problem against 3x for the same values' relative errors.
>
> **The record.** `functional_abs_errs` beside `functional_errs`, and
> `functional_tol_used` — the gate each value was actually held to — beside the declared
> `functional_tol`, with `functional_tol_widened` naming the values that moved and
> `reference_abs_err` recording what moved them. The effective gate is written even when
> it equals the declared one, so that reading a near-threshold record does not require
> knowing the widening rule. Recording the absolute error is not redundant with the
> relative one: `functional_truth` never enters a record, so nothing downstream can
> convert between them.
>
> **The defect** is the `max_abs_err` conflation struck through in §3 above.
>
> **Nothing moved, and the no-op is structural rather than lucky.** Re-graded across all
> 34 solved workspaces with the pre-step-4 verifier and this one, same plans, same
> sandbox: **34/34 identical verdicts**, and the full record diff adds exactly two keys
> (`functional_abs_errs`, `functional_tol_used`) on exactly one problem — the suite's
> only functional problem, `pde_schrodinger_eigen_2d`, whose λ₁ now reads `rel 2.93e-04
> abs 2.47e-03` where before it read the relative figure alone. Nothing in the suite
> declares `reference_abs_err`, so every gate is still its declared value; a test asserts
> exactly that over the corpus and will fail the first time a BENCHOP problem lands,
> which is when the claim has to be re-measured rather than quietly outgrown.
> `manifest.json` and the problem statements handed to the agents are byte-identical,
> deliberately — the statements are *run inputs*, and churn in them would make a future
> run incomparable with every run before it, so the per-value wording appears only on a
> problem that actually declares an uncertainty.
>
> Covered by
> [`benchmark/tests/test_functional_tolerance.py`](../benchmark/tests/test_functional_tolerance.py)
> (29 tests), 24 of which fail without this step's changes.

**5. Add SDE functional support.** Four of BENCHOP's eight methods are SDE-side (MCA,
MLMC, SGBM, mSABR), so the full benchmark needs it. Port the PDE pattern:
`functional_truth` on the problem, a `functionals` dict in the SDE return contract, a
functional branch in `_score_sde_moments`. **Include a convergence requirement** — the
price at two path counts / two steps, checked to be converging — or the SDE route has no
defence against a hardcoded constant.

> **[rev, from step 3's measurement]** The requirement stands, but *"checked to be
> converging"* cannot be a decreasing test. Measured 2026-10-17, SDE error-vs-work
> curves at this suite's path counts are not monotone — `sde_gbm_2d_high_corr` scores
> *worse* at twice the work — because the error axis is dominated by sampling noise
> rather than by discretization. The check has to be stated against the standard
> error, the way `_score_sde_moments` already treats reference ground truth. The
> hardcoding defence it is there to provide survives either way, and step 3 added a
> second one for free: a solver that ignores the path count it was called with is
> detected and recorded as `level_ignored`.

> **Implemented 2026-10-17.**
>
> **The route.** `functional_truth` is no longer a PDE key. `functionals` crosses the
> sandbox on the SDE path, through the *same* `_extract_functionals` the PDE path now
> calls, so the two shapes cannot drift apart the way they did before step 2 fixed the
> PDE one. `_score_sde_moments` gained the functional branch the step asks for, and
> `moments` stopped being required: `_sde_moment_truth` returns `None` where there is
> no formula, the component loop runs zero times, and a problem can be graded on a
> price alone — which is the shape every BENCHOP problem will arrive in. Where a
> problem declares both, they are independent gates. The gate itself is step 4's,
> reached through the same two helpers, which were renamed `_functional_tol` /
> `_functional_tols` **[C]** because the `_pde_` prefix had become false. Only the
> *default* is type-aware: 1% for a resolved field, 5% for a sampled moment, since
> that is the gate each side would otherwise be held to.
>
> **The convergence requirement, as the rev above demands.** Stated as a symmetric
> band rather than a decreasing test. The solver is called a second time at
> `functional_level_paths` (default `num_paths // 4`) and each reported value must
> (a) **differ** between the two counts and (b) differ by less than
> `tol_abs × (1 + √(n_graded/n_coarse))`. The quarter is not a round number picked for
> looking reasonable: a standard error falls as `1/√n`, so four times fewer paths is
> exactly twice the noise and the band is exactly `3 × tol_abs` — the same factor
> `REFERENCE_TOL_FACTOR` carries everywhere else in this harness. Nothing requires the
> finer run to be the better one.
>
> **It catches what it was built to catch.** Measured on a Black–Scholes call
> (`S0=K=1, r=0, σ=0.2, T=1`, reference `0.07965567455405798` from `ndtr`, 20000 paths,
> seed 7), against a solver that simulates the right law and returns the reference to
> twelve digits:
>
> | | relative error | `functional_ok` | levels 20000 vs 5000 | verdict |
> |---|---|---|---|---|
> | honest MC pricer | 1.867e-02 | True | Δ 9.398e-04 of band 1.195e-02 | **PASS**, score 8 |
> | hardcoded constant | **0.000e+00** | True | **Δ 0.000e+00** | **FAIL**, score 1 |
>
> The hardcoded solver is *more* accurate than the honest one on every number the rest
> of the module looks at. It fails on the one thing that distinguishes an estimate from
> a constant. A solver that instead ignores the path count it is handed fails the same
> way, and the record separates the two diagnoses with `level_ignored`.
>
> **What cannot fail a verdict.** Only `constant` and `inconsistent` do. A crashed
> coarse solve, a coarse run that returns no functionals, a problem with no coarser
> level, or an exception anywhere in the check is recorded as `not_run` with a reason
> and leaves the verdict alone — a flaky sandbox must not be able to fail a correct
> solver. The inline path cannot re-solve at all, so it records
> `{"status": "not_run", "reason": "inline"}` rather than nothing: a passing record
> silent about the defence would read as one that had it and cleared it.
>
> **Zero churn, measured not asserted.** All **56/56** problem statements and **56/56**
> manifest entries render byte-identical to `HEAD` (`contract_footer` and
> `manifest_entry` re-run under both versions and compared).
>
> > **[C, step 7]** ~~byte-identical~~ for the manifest entries this was
> > *content*-identical: the comparison was over parsed dicts, which is
> > order-insensitive, and hoisting the functional block to the end of
> > `manifest_entry` moved two keys on `pde_schrodinger_eigen_2d` past
> > `diagnostics`. Same keys, same values, different order, visible as a diff in
> > the tracked `manifest.json` and in nothing an agent reads. The statements were
> > byte-identical and still are. A problem statement is an
> agent *input*; churn in it makes a later run incomparable with an earlier one for a
> reason that has nothing to do with the solver. The one key that did change shape is
> `SDE_CONTRACT_KEYS` in the parity test, which is the enumeration that pins the
> boundary.
>
> **Three defects found on the way**, all of them things that would have surfaced as
> confusing behaviour on the first BENCHOP problem rather than as an error here:
>
> 1. `manifest_entry` took `has_ground_truth` to imply a moment formula and called
>    `problem["moments"]` unconditionally, so the first functional-only SDE problem
>    would have raised `KeyError: 'moments'` at setup time, before any solver ran.
> 2. `plan_harness_error` — the number `wrong_plan_won` ranks plans on — returned
>    `None` whenever an SDE record had no moment components, which is *every* BENCHOP
>    SDE problem. It now folds the functional in on the same dimensionless scale the
>    moments use (1.0 exactly at the pass line), with the reference's own
>    `reference_abs_err` as the margin, which is the functional counterpart of the
>    Monte-Carlo standard error the moment branch already uses. This is the same
>    defect the moment branch's own comment records, one problem type later.
> 3. `report.key_metric` printed `varErr=0.0%, meanErr=0.0%` for a price-only
>    problem — the 0.0 that "no moment is wrong" puts there, reading as a flawless
>    solver. It now prints the price where there is no moment, and both where there
>    are both, and flags a two-level check that did not run.
>
> `run.py` also copies a fixed key list from the harness result into `results.json`,
> and the functional was not on it: a price-only verdict would have been recorded
> without the number it rests on.
>
> **Tests.** `benchmark/tests/test_sde_functional.py`, 42 tests in five sections
> (K1 boundary, K2 the grade, K3 the gate, K4 the two-level defence, K5 record /
> summary / statement / manifest / ranking). The K1 and K4 tests run a real
> `solve_sde` in a real subprocess, because the thing under test *is* the subprocess
> boundary and a stub would assume away the bug. Whole suite after the step:
> `uv run --extra dev python -m pytest verifylib benchmark/tests -q` →
> **817 passed, 1 failed, 3 skipped, 4 deselected in 632.86s**. The one failure is
> `test_certified.py::test_frozen_records_fall_back_to_the_old_test_unchanged`,
> which predates this plan: commit c21077e moved the result JSONs into
> `benchmark/results/{pipeline,studies,…}/` and the test's `os.listdir` is not
> recursive, so it now re-grades 28 frozen records and asserts it saw more than 50.

**6. Author the ground truth.** Independently derivable, implementations verified in §2a:
`HSTeuCall` (Heston CF) and `HHWeuCallII` (Gauss–Hermite over the HW integral;
~~tighten the quadrature first~~ **[C]** collapse the two quadratures into one — see
the §2a rev). The other six have no formula and need the
[`make_references.py`](../benchmark/make_references.py) route — two independent method
families, disagreement recorded as `reference_error`. For `HHWeuCallI` the published
±0.005 is the floor however good the reference is.

> **Implemented 2026-10-17.** **[X]** [`benchmark/benchop.py`](../benchmark/benchop.py)
> (1200 lines), [`benchmark/references/benchop_slv.json`](../benchmark/references/benchop_slv.json),
> [`benchmark/tests/test_benchop_truth.py`](../benchmark/tests/test_benchop_truth.py).
>
> **What "author the ground truth" turned out to mean.** Not writing an answer key —
> BENCHOP published one. Writing the *derivation* of each of its 24 numbers, because a
> transcribed constant is only as good as the reading of the problem it came from, and
> misreading the problem is precisely the failure a published benchmark does not protect
> against. So the ledger records **three** numbers per value and keeps them apart: the
> published figure, the uncertainty *that figure itself* carries, and how closely this
> repo reproduced it. Only the second is `reference_abs_err`. Conflating it with the
> third would widen a gate because of our own effort rather than the answer key's
> precision, which is the opposite of what step 4 built the field for.
>
> | problem | routes (family) | confirmed | relative |
> |---|---|---|---|
> | `HSTeuCall` | Lewis CF *(quadrature)* · CN grid | **5.3e-14** | 3.0e-15 |
> | `HHWeuCallII` | collapsed CF *(quadrature)* · Euler MC | **5.7e-14** | 1.6e-15 |
> | `HSTdnTouch` | CN grid · bridge MC *(sampler)* | **3.8e-06** | 5.6e-06 |
> | `QLSVdnTouch` | CN grid · bridge MC *(sampler)* | **4.5e-06** | 7.5e-06 |
> | `QLSVeuCall` | CN grid · MC **disagrees — see below** | **2.7e-05** | 4.4e-05 |
> | `SABR` set I | conditional BESQ(0) *(quadrature)* · CN grid · 2-factor MC | **4.6e-06** | 2.8e-05 |
> | `SABR` set II | 2-factor MC *(sampler)* · CN grid | **3.5e-06** | 9.0e-05 |
> | `HHWeuCallI` | Euler MC *(sampler)* only | **4.3e-02** | 8.1e-04 |
>
> The first route named is the one `confirmed` is measured against; the ledger carries
> every route, its disagreement per spot, and its standard error where it has one.
> Seven of the eight have two independent *families* — a quadrature and a grid, or a
> grid and a sampler — which is what the step asks for and what a test now pins.
>
> **The transcription is checked against the rule that produced it, not read twice.**
> The formulations give the SABR strikes both as `K_i = S₀·exp(0.1√T·δ_i)` and as six
> tabulated numbers; recomputing them from the spot and the maturity reproduces all six
> to **4.6e-13**, which is how we know the spot, the maturity and the strikes were read
> off the same problem.
>
> **Two numbers the plan treated as accuracy problems were formulation problems.** `HSTeuCall`'s 4e-12 in §2a was *this
> repo's* error, not BENCHOP's: the two-integral `P1`/`P2` form differences two
> oscillatory integrals, and Lewis' single-integral form at the same tolerance gives
> **5.3e-14**. `HHWeuCallII` was worse diagnosed — §2a asked for more Gauss–Hermite
> nodes, and more nodes do not help, because the error is in the *inner* Heston call at
> strikes spanning nine orders of magnitude. The fix is one fewer quadrature, not a
> finer one: the expectation over the rate can be taken *inside* Lewis' integrand,
> where it becomes the Gaussian moment generating function at `z = −½ + iu`, and the
> resulting `e^{−Var(M)u²/2}` makes the integral terminate by `u = 12`. **5.7e-14.**
>
> **`QLSVeuCall` has two different right answers, and a closed form says which is
> which.** The grid matched the published values to 7e-5 while an honest sampler of the
> stated SDE sat **0.24 below at S₀ = 125** — 0.8% relative, eight times the gate — and
> neither route would move. Freezing the variance settles it, because QLSV then
> acquires a closed form: `100/S` is a Bessel process of dimension 3, so the transition
> density is explicit. Measured at S₀ = 125:
>
> | | value | distance from the closed form |
> |---|---|---|
> | closed form, BES(3) density | **29.98063** | — |
> | Monte Carlo, 1600 steps | 29.86666 ± 0.0882 | 1.3 standard errors |
> | finite differences, `u_SS = 0` far field | **32.20765** | **25 standard errors** |
>
> The grid's disagreement *grows with the spot* — 0.3 standard errors at S₀ = 75, 5.7
> at 100, 25 at 125 — which is what a local-martingale defect looks like and what a
> discretization error does not.
>
> The grid is not converging slowly; it is converging to something else. With
> `f(s) = 0.01s²` the asset is a **strict local martingale** — `E[S_T] = S₀·erf(r₀/√(2θ))`
> is strictly below `S₀`, by 2.23 here — so the pricing PDE has more than one solution,
> and the usual linear far-field condition selects the one that is not the expectation.
> The published values follow the grid. That is a statement about the answer key rather
> than about either route, and the consequence is recorded against §0 and step 7:
> this problem is a PDE problem or it is nothing. `QLSVdnTouch` is untouched by it, as
> the measurement shows — a barrier at 150 kills the asset long before it reaches the
> region where the defect lives, and there the two families agree to 4.5e-6.
>
> **The one value this repo cannot check to its own quoted precision is `HHWeuCallI`**,
> which is the one §8 predicted. Three correlated factors leave no quadrature, and the
> grid route needs the T-forward reduction, whose coefficients depend on τ — so the
> only family here is a sampler, and it sits +0.030 to +0.043 above the published
> values against their own ±0.005. The step ladder says this is not discretization
> (2000 → 8000 steps moves the answer by 0.005 while the gap stays at 0.04), and on
> `HHWeuCallII`, where the same sampler *can* be held against a closed form, its bias
> at those settings is +0.010 — which is what the bias-controlled route subtracts.
>
> **A negative result worth keeping, because it would otherwise be tried again.**
> Andersen's QE step is the obvious cure for a Feller ratio of 0.04, and it does not
> work here. QE returns the variance without the Brownian increment that produced it,
> and `HHWeuCallI` has a third factor correlated with that same Brownian motion, so the
> increment has to be reconstructed from the exact identity and divided by `√V̄`. With
> the variance sitting on zero for a whole step — **69% of steps**, measured — the
> numerator is the drift term alone while the denominator vanishes, and the quotient
> overflows the asset. Guarding those steps with a fresh normal removes the overflow
> and leaves a worse fault: keeping only the steps where the variance moved is a
> *selection*, the surviving reconstructions have sample mean **−0.049** instead of
> zero, and with `ρ₂₃ = −0.7` that becomes a drift of order 1 in the short rate's
> Brownian motion. Set I came out **4.9 high on a price of 78.58**. The part worth
> stating plainly is that set II cannot catch any of this: its `ρ₂₃` is 0, so the
> reconstruction is multiplied by zero and never runs, and "validated against the one
> set that has a closed form" would have been a false claim.
>
> **The routes, and why each one is the shape it is.** Four families, chosen per
> problem rather than uniformly:
>
> - **Quadrature** — Lewis' single integral, for the two problems with a
>   characteristic function.
> - **A grid** — Crank–Nicolson with a Rannacher start on a `(S, V)` or `(S, ln σ)`
>   tensor mesh, `scipy.sparse` direct solves, one factorization for the whole march
>   because the operators are τ-independent. Two things in it were corrections rather
>   than refinements. The V axis uses Fornberg weights so that the *one-sided* stencils
>   at the degenerate `V = 0` row are second order — that row is where a
>   Feller-violating CIR keeps its density, and going from first order to second there
>   took the double-no-touch error at 400×160 from **4.5e-4 to 1.4e-4**. And the end
>   treatments are named (`dirichlet` / `degenerate` / `linear`) rather than assumed,
>   which is what made the QLSV far-field problem visible at all; keeping the cross
>   term on the boundary rows and taking `u_VV = 0` rather than `u = 0` at the top of
>   the V axis took the same problem from **4.5e-5 to 1.1e-5 at 800×320**.
> - **A sampler** — Andersen's QE variance step, and the asset in its Lamperti chart,
>   where the diffusion coefficient becomes the constant `√V`. The chart is not a
>   nicety: it is what makes a per-step **Brownian bridge** valid for the barriers, so
>   the O(√dt) discrete-monitoring bias §2d measured at 1.0e-2 is simply absent; and
>   for QLSV it is what reveals the Bessel(3) drift, whose explicit step walks through
>   the origin and hands back a *negative asset price*. The step is drift-implicit
>   instead, which cannot: the root it takes has the sign built in. The two differ by
>   0.02 at S₀ = 125, the difference does not shrink with the step count, and the
>   implicit one is the one that reproduces the frozen-variance closed form.
> - **Conditioning** — SABR set I has `ρ = 0`, so given the volatility path the asset is
>   a CEV(1/2) process on the clock `∫σ²dt`, which is a squared Bessel process of
>   dimension 0 with an explicit density. The payoff expectation is then exact and the
>   only Monte Carlo left is over a scalar.
>
> **Control variates moved more than any of the schemes did.** Four of them, each an
> exact identity rather than a fitted quantity: on Heston–Hull–White the discounted
> asset's expectation `S₀` and the discount factor's own `P(0,T)`, which is
> Gaussian-closed; and on SABR two more, which are the difference between confirming it
> at the level of the gate and confirming it two decades below:
>
> | | before | after |
> |---|---|---|
> | SABR I — `E[v]`, `E[v²]` of the clock | 9.6e-05 | **4.6e-06** |
> | SABR II — `E[S_T] = S₀` (CEV with β<1 is a true martingale) | 4.9e-05 | **3.5e-06** |
>
> The second one also changed what the step ladder says. Before it, set II's two step
> counts straddled the published value and differed by about their own standard error,
> which read as noise; after it, quartering the step divides the gap by **sixteen** —
> second order, not the first order an Euler scheme would give. That is recorded but is
> *not* the primary route, because the order was inferred from the same two points the
> extrapolation is applied to.
>
> **Tests.** `benchmark/tests/test_benchop_truth.py`, 28 tests in six sections (B1
> transcription, B2 the exact families, B3 each piece on its own, B4 the convention that
> splits the QLSV call, B5 the ledger, B6 every derivation end to end, marked `slow`).
> The fast ones re-derive from the published parameters in under a second, and three of
> them are identities rather than comparisons against the same answer key: the
> characteristic function converges to Black–Scholes at the measured second-order rate
> as the vol-of-vol vanishes, the killed heat kernel collapses onto the one-barrier
> reflection formula, and the Bessel(3) density integrates to 1 and to its own mean.
>
> Whole suite after the step:
> `uv run --extra dev python -m pytest verifylib benchmark/tests -q` →
> **837 passed, 1 failed, 3 skipped, 12 deselected in 616.67s**. The one failure is
> the same `test_certified.py` count assertion step 5 recorded, and still predates
> this plan.

**7. Write the problem entries.** ~~Eight PDE entries — SABR is 2-D in (S,σ), HHW is
3-D or its 2-D reduction (state which, the formulations ask) — plus~~ SDE variants,
~~once step 5 lands~~ **unblocked 2026-10-17**, plus the shifted twins from §5. Three
functionals each. An SDE entry carries `functional_truth` with no `moments` and may
set `functional_level_paths`; the default quarter costs 25% more solve time, which is
worth checking against the 300 s budget for the heavier methods.

> **Implemented 2026-10-20, and narrowed first.** **[X]** ten entries in
> [`problems.py`](../benchmark/problems.py), a third suite in
> [`setup.py`](../benchmark/setup.py), three twins in
> [`benchop.py`](../benchmark/benchop.py), and two corrections in
> [`verify.py`](../benchmark/verify.py) — the order check that was skipping these
> problems entirely, and a gate that can now be stated per value.
>
> **The scope changed before the work started.** The suite as built is **SDE-only**:
> seven published problems and three shifted twins, no PDE entries. The eight PDE
> entries this step called for are struck above rather than deferred, and §0's
> point (c) — the cross-formulation comparison — is not being collected; see the
> rev there. `QLSVeuCall` drops out with them, because step 6 measured that its
> published value answers a question its own SDE does not ask.
>
> | # | slug | T | paths | step | gate | its answer key |
> |---|---|---|---|---|---|---|
> | B01 | `sde_sabr_cev_call_i` | 2 | 4 000 000 | 0.005 | 4e-3 | published |
> | B02 | `sde_sabr_cev_call_ii` | 10 | 1 000 000 | 0.004 | 1e-2 | published |
> | B03 | `sde_heston_call_feller` | 1 | 2 000 000 | 0.002 | 1.5e-2 / 4e-3 / 2e-3 | published |
> | B04 | `sde_heston_dnt` | 1 | 400 000 | 0.002 | 6e-3 | published |
> | B05 | `sde_qlsv_dnt` | 1 | 400 000 | 0.002 | 8e-3 | published |
> | B06 | `sde_hhw_call_corr` | 10 | 1 000 000 | 0.004 | 3e-3 | published, ±5e-3 |
> | B07 | `sde_hhw_call_zerocorr` | 10 | 1 000 000 | 0.004 | 3e-3 | published |
> | B08 | `sde_heston_call_hivar` | 1 | 2 000 000 | 0.002 | 1.5e-2 / 4e-3 / 2e-3 | **harness** |
> | B09 | `sde_heston_dnt_hivar` | 1 | 400 000 | 0.002 | 6e-3 | **harness** |
> | B10 | `sde_sabr_cev_call_ii_lowvol` | 10 | 1 000 000 | 0.004 | 1e-2 | **harness** |
>
> Nothing in the table was typed twice: every value a problem is graded on is read
> out of `references/benchop_slv.json` and a test fails if an entry and the ledger
> disagree.
>
> **A third suite, because the other two carry measured baselines.** `apply_suite_filter`
> split on one boolean (`closed_form is False`); it now splits on a named `suite`,
> defaulted from that boolean so every problem authored before there was a third
> suite keeps the membership it had. Measured after the change: **main 49,
> no-closed-form 7, benchop 10**, and the first two are the same 56 problems they
> were.
>
> **The PDE route's anti-hardcoding defence did not exist, and §5 was leaning on
> it.** Found while checking what the entries would land on, and fixed even though
> no PDE entry is now being written, because the hole is in shipped code.
> `_pde_order_check` required a **field** ground truth, so on a functional-only
> problem the second grid was never run at all. Measured on a sandboxed solver that
> returns a hardcoded constant and nothing else:
>
> | | before | after |
> |---|---|---|
> | hardcoded constant | **passed, score 10**, `order_checked=False` | **failed, score 1** |
> | honest O(1/N) solver | passed, score 10 | passed, score 10, observed order 1.000 |
>
> The second half of the fix is the one §5 did not anticipate. "A hardcoded constant
> has zero convergence and fails" is false even with the check running: a constant
> has zero *error* at both grids, which trips the superconvergence waiver — there so
> that a spectral scheme resting on its round-off floor is not failed for having no
> observable order — and hands the constant `observed_order = inf` and
> `order_ok = True`. So movement is checked separately from convergence, which is
> the SDE side's step-5 rule stated on the grid the PDE side already runs: there the
> price is taken at two path counts, here at two resolutions, and in both it has to
> move. It applies only where the functional is the *whole* truth; a problem with a
> field has the field as its defence, and nobody memorizes a surface.
>
> **The gate is now per value, and it had to be.** BENCHOP's three Heston call
> values span 31× (0.91, 9.05, 28.51) and are priced from the same paths, so their
> relative standard errors span 10×: measured **4.2e-3, 1.0e-3, 4.6e-4** at a
> million paths. One relative gate over all three either fails the out-of-the-money
> value or stops testing the in-the-money one — the trade §8 names and declines.
> `functional_tol` therefore takes the same two shapes `reference_abs_err` already
> took, a number or `{name: tol}`, and the two compose: the declared gate per value,
> floored by what the answer key itself knows. The distinction is worth keeping
> straight — the floor is about how precisely a value is *known*, the declaration is
> about how precisely it can be *computed*.
>
> **The twins' shift is not the one §5 proposed**, and the measurement that rejected
> it is in the rev there: κ 2.58 → 2.43 moves two of the three Heston call values by
> **less than their own tolerance**. What is shifted instead is the initial variance
> or volatility alone — `V0` 0.114 → 0.164, `sigma0` 0.4 → 0.34 — which leaves the
> model, the payoff and the Feller ratio bit-identical and moves the answer by
> **4.8× to 35×** its own gate, value by value. The twins' truth is harness-owned and follows `make_references.py`'s
> rule, with the error estimate named per twin and **validated on its published
> sibling**, where a third party adjudicates.
>
> **The gates are measured, not chosen.** For each entry a validated sampler was
> run at the settings the entry ships with, *without* variance reduction, since an
> agent's solver is unlikely to use any; the gate is then set a few times above
> what that reaches. Where the error is dominated by sampling rather than by bias
> the number that matters is three standard errors, so both are given, and the
> margin is the gate over whichever is larger. Two rows (†) come instead from the
> plain solvers run end to end through the sandbox, which is the closer analogue of
> what an agent writes and is the worse number in both cases.
>
> | problem | paths × steps | worst \|error\| | worst 3·s.e. | gate | margin |
> |---|---|---|---|---|---|
> | `sde_sabr_cev_call_i` | 4M × 400 | 1.7e-3 † | 2.3e-3 | 4e-3 | 1.7× |
> | `sde_sabr_cev_call_ii` | 1M × 2500 | 3.2e-4 | 6.1e-3 | 1e-2 | 1.6× |
> | `sde_heston_call_feller` | 2M × 500 | 4.8e-3 † | 6.3e-3 | 1.5e-2 | 2.4× |
> | `sde_heston_dnt` | 400k × 500 | 2.1e-3 | 2.0e-3 | 6e-3 | 2.9× |
> | `sde_qlsv_dnt` | 400k × 500 | 3.6e-3 | 2.1e-3 | 8e-3 | 2.2× |
> | `sde_hhw_call_corr` | 1M × 2500 | 8.4e-4 | 9.3e-4 | 3e-3 | 3.2× |
> | `sde_hhw_call_zerocorr` | 1M × 2500 | 2.0e-4 | 1.1e-3 | 3e-3 | 2.6× |
>
> Three readings worth keeping. **Time is not the binding constraint** — a plain
> full-truncation-Euler Heston solver returns all three prices at two million paths
> in 24 s against a 300 s cap, and the path counts above are set by the standard
> error a gate has to clear rather than by the clock. **Except on the ten-year
> pair**, where `sde_verify_dt`'s `0.04/T` cap forces 2500 steps whatever the entry
> declares; that caps the affordable path count, which is why SABR set II carries
> the loosest gate in the suite. And **the quadratic local volatility is harder for
> a sampler than the linear one** — 3.6e-3 against 2.1e-3 on the same corridor at
> the same settings — because the Lamperti chart that makes a per-step Brownian
> bridge exact is the logarithm for `f(s)=s` and `-100/s` for `f(s)=0.01 s²`, where
> the bridge is only approximate. Its gate is looser for that reason and says so.
>
> **Three solvers were run against the real entries, in the real sandbox**, which
> is the only way to find out whether an entry is well posed rather than merely
> well typed:
>
> | solver | entry | result |
> |---|---|---|
> | plain log-Euler Heston, 100k paths | `sde_heston_call_feller` | **fails**, score 8 — `price_S75` at 1.53e-2 against its 1.5e-2 gate, the other two clear |
> | the same at the shipped 2M paths | `sde_heston_call_feller` | **passes**, score 10 — 4.8e-3 / 6.4e-4 / 2.6e-4, solve 23.9 s |
> | plain absorbed-CEV SABR, shipped 4M paths | `sde_sabr_cev_call_i` | **passes**, score 10 — 1.4e-3 / 1.6e-3 / 1.7e-3, solve 43.6 s |
>
> So the gate bites an under-resolved solver and clears an ordinary one, at a cost
> well inside the 300 s cap and inside the 4 GB rlimit with a four-million-row
> array crossing the process boundary. And the defence works on a real entry: a
> solver that simulates the right law and reports all three *published* constants
> to fifteen digits scores relative error **0.000e+00**, `functional_ok=True`, and
> **fails with score 1**, because none of its three values moved when the path
> count changed.
>
> **One more place assumed a moment formula.** `validate_ground_truth.py` iterates
> every scored SDE problem and called `prob["moments"]` unconditionally, so the
> validator crashed with `KeyError: 'moments'` on the first entry — the same
> assumption step 5 removed from `manifest_entry`, in a third file. It now skips a
> problem with no moment formula and checks the thing that *is* checkable without
> running anything: that every value a number-scored problem reports carries a
> gate, that the two-level check has a coarser level to run at, and that the step
> the verifier will use is the step the entry declares. `has_ground_truth` has not
> implied `moments` since step 5, and this is the second time that has had to be
> found by something breaking rather than by looking.
> `uv run python benchmark/validate_ground_truth.py` → **186 checks, all passed**,
> thirty of them new.
>
> **Zero churn, measured against `HEAD`.** All **56/56** pre-existing problems
> render a **byte-identical contract footer** — the only thing an agent reads —
> and a manifest entry with the same keys and the same values, across the
> per-value tolerance, the third suite and the order-check fix. `manifest.json`
> gains the ten new entries and changes no existing one; the diff on it also shows
> two keys moving position on `pde_schrodinger_eigen_2d`, which step 5 caused and
> recorded imprecisely as byte-identical. Three
> tests written in earlier steps to fail exactly here — the suite partition, step
> 5's "every SDE problem is called once", step 4's "nothing declares
> `reference_abs_err` yet" — were re-stated rather than deleted, which is what
> their own docstrings asked for.
>
> **Tests.** Two new files: `test_benchop_suite.py` (**78 collected**, most of them
> one per problem — the entries match the ledger, the contract states every gate,
> the twins pair and move far enough, the step the verifier uses is the one the
> entry declares) and `test_pde_functional.py` (**12** — the order-check fix, with
> the constant and the honest solver both run in a real subprocess, because what is
> under test is the two-grid ladder and a stub would assume it away). Four more in
> `test_leakage_ncf.py` and seven in `test_functional_tolerance.py`. The leakage
> rules bind harder here than anywhere: a statement must not quote a reference value, must not
> name the benchmark, and must not reveal that a twin is a twin — a model warned
> off the memorized value would make the control come back clean for the wrong
> reason.
>
> **What step 8 still owes.** Every gate above is calibrated against a *correct*
> scheme only. The repo's own rule has a second half — a plausible wrong scheme
> must fail — and it is measured for exactly one of the seven: on the continuously
> monitored barrier, a Brownian bridge reaches 2.1e-3 while per-step monitoring at
> four times the step count reaches 5.0e-3, 3.1e-3 and 1.4e-2 (§2d), and the 6e-3
> gate sits between them. The other six have no such pair measured.
>
> Whole suite after the step:
> `uv run --extra dev python -m pytest verifylib benchmark/tests -q` →
> **943 passed, 1 failed, 3 skipped, 15 deselected in 614.32s**. The one failure is
> the same `test_certified.py` count assertion steps 5 and 6 recorded, and still
> predates this plan.

> **[rev, from step 6's measurement]** Three constraints this step now inherits.
> **(a)** `QLSVeuCall` is a PDE problem or it is nothing — its two formulations have
> different answers and the published one is the grid's (§0 rev). **(b)** `HHWeuCallI`
> cannot be given a 1e-3 relative gate on the strength of anything measured so far
> (§8). **(c)** The numbers themselves come from
> [`benchmark/references/benchop_slv.json`](../benchmark/references/benchop_slv.json)
> rather than being retyped: it carries the published value, its own uncertainty, and
> this repo's agreement with it, per problem and per spot.

**8. Cross-check and calibrate.** Add the derivations to
`validate_ground_truth.py`, then apply the repo's own calibration rule: a correct scheme
must clear the gate at the benchmark's own resolution and a plausible wrong one must
fail. For `HSTdnTouch` the wrong-scheme case is already characterised — naive discrete
monitoring, biased high by the §2d amounts.

> **Implemented 2026-10-21.** Both halves. The derivations are in the validator,
> and the calibration rule is applied to all ten entries rather than the one it
> had. Every gate separates; none was moved to make it.
>
> **The pair is measured at the resolution the entry ships with**, not at a
> convenient one: `ship_settings()` reads the path count from `problems.py` and
> the step count from `sde_verify_dt`, so a gate cannot be calibrated at settings
> the benchmark does not use. The two schemes differ by **one decision**, so what
> separates them is that decision rather than the sampler, the seed, the paths or
> the step. Each runs through `benchop.py --calibrate`, which stores the result in
> the ledger beside the derivations; the whole sweep is about 35 minutes.
>
> | entry | the decision | correct | wrong | values missed |
> |---|---|---|---|---|
> | `sde_sabr_cev_call_i` | the CEV boundary at S = 0 | absorbing Euler **0.29×** | the same step reflecting **1.97×** | 3 of 3 |
> | `sde_sabr_cev_call_ii` | the CEV boundary at S = 0 | Lamperti, absorbing **0.03×** | reflecting **2.4e4×** | 3 of 3 |
> | `sde_heston_call_feller` | the variance below zero | full truncation **0.23×** | reflection **19.9×** | 3 of 3 |
> | `sde_heston_dnt` | barrier monitoring | Brownian bridge **0.35×** | once per step **3.6×** | **2 of 3** |
> | `sde_qlsv_dnt` | barrier monitoring | Brownian bridge **0.45×** | once per step **4.2×** | **1 of 3** |
> | `sde_hhw_call_corr` | the variance below zero | full truncation **0.28×** | reflection **81.9×** | 3 of 3 |
> | `sde_hhw_call_zerocorr` | the variance below zero | full truncation **0.07×** | reflection **45.6×** | 3 of 3 |
> | `sde_heston_call_hivar` | the variance below zero | full truncation **0.27×** | reflection **14.0×** | 3 of 3 |
> | `sde_heston_dnt_hivar` | barrier monitoring | Brownian bridge **0.40×** | once per step **4.6×** | 3 of 3 |
> | `sde_sabr_cev_call_ii_lowvol` | the CEV boundary at S = 0 | Lamperti, absorbing **0.05×** | reflecting **1.9e4×** | 3 of 3 |
>
> Ratios are of the entry's own gate, worst value. **The narrowest separation in
> the suite is SABR set I** — 0.29× against 1.97×, a factor of 6.8 — and it is the
> only one under 9×, which is the answer to §8's long-open question about what
> `functional_tol` on SABR should be: 4e-3 is doing real work there and is not
> loose.
>
> **A wrong scheme does not have to miss every value, and on the barriers it does
> not.** Per-step monitoring misses two of three on the Heston corridor and
> **one of three** on the quadratic one; the harness fails a problem when any
> reported value misses its gate, which is the bar used here, and `min` is stored
> beside `max` in the ledger so the difference does not have to be reconstructed
> later. The value doing the discriminating is S₀ = 125 in both — the spot nearest
> the upper barrier, where survival is lowest and the O(√dt) bias largest. Its
> twin, at initial variance 0.164 instead of 0.114, misses **all three**: more
> paths reach the barrier, so the same error grows. That is the clearest single
> piece of evidence that the twins are the same problem with a different answer.
>
> **The gate that does not resolve the generic scheme.** On SABR set II a plain
> absorbing Euler **in S** — no Lamperti chart — reaches 9.0e-3, 9.4e-3 and
> 9.9e-3 relative against a 1e-2 gate: **0.90, 0.94 and 0.99 of it**. That is a
> coin flip, and it is the one calibration result in the suite that is not clean.
> The gate was **left where step 7 put it**, for two reasons. Re-cutting a line
> after seeing which scheme falls on it is how a benchmark stops meaning anything;
> and the scheme in question is not *wrong* — it absorbs at the right boundary and
> converges to the right answer — it is under-resolved at a step the verifier caps
> from above at 0.004, which 2500 steps and a 300 s budget cannot push further.
> What the gate does separate there is reflection, by four orders of magnitude,
> and what clears it is the Lamperti scheme the problem is built around, at 0.03×.
> So B02's calibration is recorded with three rows rather than two, and the middle
> one is the finding. The same measurement on its twin reads 0.82–0.96×.
>
> **`HHWeuCallI`'s 3e-3 gate stands, and the reason is now a number.** §8 asked
> step 8 to justify it or drop the problem. The primary derivation in the ledger
> sits **8.1e-4, 7.3e-4 and 5.5e-4** *above* the three published values — so at the
> 1e-3 gate the rest of the suite gets, this repo's own best answer would already
> consume **81%, 73% and 55%** of the budget before a solver was written, and the
> same sampler without its bias control consumes **96%** at S₀ = 75. At 3e-3 it
> consumes 27% at worst, the plain sampler measured at the shipped settings
> consumes 28%, and reflection misses by 82×. The gate separates and the problem
> is kept. What has **not** improved is the thing underneath: all of those numbers
> come from one sampler at different settings, not from independent families, so
> whether the offset is the sampler's bias or the answer key's error is exactly as
> unresolved as step 6 left it. The budget a correct solver may be losing to the
> answer key is up to 27% of the gate, and that is the honest statement.
>
> **The derivations are in `validate_ground_truth.py`.** Three of the ten have a
> quadrature cheap enough for a validator — the Heston call, its twin and the
> decorrelated three-factor call — and those are **recomputed there, now**, rather
> than read back from the ledger, which is what catches a ledger regenerated at
> the wrong settings or edited by hand. The other seven are checked against the
> ledger for digit-for-digit agreement with the entry, for two independent method
> families (`hhw_eucall1` is the one exception, and is allowed by name), and for
> two things no earlier check asked: that the gate is wider than the **answer
> key's own uncertainty**, and wider than **this repo's disagreement with it** —
> otherwise the gate is grading the reference rather than the solver. The twins,
> which no third party adjudicates, are held to every route landing inside half
> the gate and to their error rule having been validated on the published sibling.
> The calibration is then checked for still applying: same paths, same steps, same
> gate as the entry declares today.
> `uv run python benchmark/validate_ground_truth.py` → **271 checks, all passed**,
> **85 of them new** (186 after step 7).
>
> **And the same pair run end to end, through the real sandbox**, because a
> relative error is not a verdict. One barrier solver, written as an agent would
> write it — log-Euler Heston, full truncation, the killed heat kernel over each
> step — run twice on `sde_heston_dnt` at the shipped 400 000 paths with **one line
> changed**:
>
> | monitoring | verdict | score | worst error | solve |
> |---|---|---|---|---|
> | over the step (Brownian bridge) | **passes** | 10 | 2.4e-3 | 65.5 s |
> | once per step | **fails** | 8 | 2.2e-2 | 10.1 s |
>
> The failing values are `no_touch_S75` at 8.5e-3 and `no_touch_S125` at 2.2e-2
> against their 6e-3 gate, with the middle spot clearing at 5.5e-3 — two of three,
> exactly as the calibration at the same settings said. Both runs pass the
> two-level anti-hardcoding check, so what fails is the answer and not the shape of
> it. The last column is the one worth keeping for step 11: **the wrong scheme is
> 6.5× faster**. On a work-precision axis it would sit well to the left of the
> right one, and only the accuracy gate tells them apart.
>
> **Tests.** `test_benchop_suite.py` gains an M5 section — every entry's stored
> calibration refers to the paths, steps and gate that entry declares *today*, a
> correct scheme clears it, a plausible wrong one fails it, and a twin is
> calibrated on the same decision as its sibling — taking the file from 78 to
> **111**. `test_benchop_truth.py` gains B8, which asks the cheap question the
> expensive measurement cannot: is each wrong scheme wrong in the *direction*
> claimed? Discrete monitoring must let strictly more paths survive than the
> bridge on the same paths; reflecting a forward off zero must raise its mean,
> which the martingale identity says should only fall; reflecting a variance must
> raise it where truncation holds it at zero. Those run in a second at four
> thousand paths and would catch a sign error in the calibration that a
> half-hour sweep would report as a clean separation.
>
> Whole suite after the step:
> `uv run --extra dev python -m pytest verifylib benchmark/tests -q` →
> **981 passed, 1 failed, 3 skipped, 15 deselected in 662.39s**. The one failure is
> the same `test_certified.py` count assertion steps 5, 6 and 7 recorded, and still
> predates this plan.

**9. Stage and smoke-test.** ~~`--only pde_benchop_hst_eucall`~~ — **[rev, step 8]**
there is no such slug: the suite is SDE-only and the entries are named in step 7's
table. Note also that `--suite` is a *runner* flag — `setup.py` has no suite filter
and stages whatever `--only` selects, writing a manifest for all 66 either way — so
the two halves are spelled differently. The timeout is the harness default now that
step 3 raised it to 300 s, and the calibration already measured what every scheme
costs inside it, so what step 9 is smoke-testing is staging rather than time.

```bash
uv run python benchmark/setup.py                         # stages all 66
uv run python benchmark/validate_ground_truth.py         # 271 checks
uv run python benchmark/benchop.py --calibrate --check   # recompute, write nothing
uv run python benchmark/run.py --suite benchop \
    --results-json benchmark/results/pipeline/benchop.json   # [rev, step 9]
```

> **Implemented 2026-10-21.** Staging itself was sound, and the smoke test was
> worth running anyway: four defects came out of it, three of them on the path
> step 10 is about to take and one of them the kind that cannot be repaired after
> the fact.
>
> **What was right.** `setup.py` writes all 66 workspaces (56 before, ten new),
> and all ten briefs agree with the verifier that will grade them: the call each
> one states is `solve_sde(num_paths=N, dt=sde_verify_dt(p), T, seed)` for its own
> entry, it names the second path count the two-level check uses, and it names
> every one of the three values by the key the harness reads. None of the thirty
> reference values appears anywhere in a staged file — checked on the staged
> artifact and not on the `description` string, since the file is what the
> pipeline is handed and the footer that completes it is generated.
> `run.py --suite benchop --list` selects exactly B01-B10, and
> `validate_ground_truth.py` passes all **271** checks.
>
> `benchop.py --calibrate --check` re-ran all 23 scheme/entry pairs and wrote
> nothing: **`CHECK: all 10 entries separate`**, every relative error reproducing
> the stored ledger digit for digit. Only the wall times moved, and they moved a
> lot — `sde_heston_dnt_hivar`'s bridge took 1152 s under load against the 113 s
> the ledger records — which is the useful negative result of the re-run: the
> values are seed-deterministic, and the timing column is the one property of the
> machine rather than of the scheme. A work-precision comparison in step 11 has to
> come from one sweep, not from numbers taken on different days.
>
> **1. The manifest did not say which suite an entry is in.** `closed_form: False`
> made the no-closed-form suite visible in `manifest.json`; the externally anchored
> one was invisible, so all ten entries read as members of the 49 whose baseline
> denominator they must never join. `manifest_entry` now writes `suite` where
> `problems.py` names one, and the header carries
> `n_by_suite: {main: 49, no-closed-form: 7, benchop: 10}`. Exactly ten entries
> changed; the other 56 are byte-identical, which is the claim `test_sde_functional`
> makes about them.
>
> **2. Both runners would have merged this suite into the standing baseline.** The
> step-9 and step-10 commands as written passed no `--results-json`, so
> `run.py --suite benchop` would have merged ten benchop records into
> `results/results.json` — the August 28-problem main-suite run — and *replaced
> its `config` wholesale*, since the file carries one config for all its records.
> `oneshot.py` would have done the same to `results/oneshot_results.json`, which is
> the file the 2026-09-27 accident already clobbered once. The protection was a
> convention written in `results/README.md` after that accident, which is to say a
> thing a person has to remember while launching a run that costs hours. It is now
> a refusal: `setup.guard_results_path` stops any selection that reaches outside
> `main` and names a path that follows the convention. `--list` still answers, and
> a main-suite run — including a one-problem smoke test, the invocation most
> likely to forget and least able to do harm — is untouched.
>
> **3. The work-precision series was throwing away a point it had already paid
> for.** `_fold_work_precision_sde` said "unlike the PDE side there is no free
> second point", which stopped being true in step 5: a functional-scored SDE
> problem is re-solved at `functional_level_paths` to prove its answer is an
> estimate, in the same sandbox, at the same step, and the harness times that
> solve. Since no entry in this suite declares `work_precision_levels`, every one
> of the ten was emitting a **one-point curve**, and a step-10 sweep would have
> paid for ten curves and recorded none of them — information no amount of later
> analysis could recover without re-running the sweep. The coarse level is now
> scored and folded in. Measured on `sde_heston_dnt` with a correct solver:
>
> | paths | solve | worst rel err |
> |---|---|---|
> | 100 000 | 15.1 s | 2.57e-3 |
> | 400 000 | 52.2 s | 2.37e-3 |
>
> Four times the work for 8% of the error, because at these counts what is left is
> discretization bias and not sampling noise — which is the same reading §2d gave
> the barrier and the reason the two-level band is symmetric. Where a problem
> carries moment gates *as well*, the point is recorded under `skipped` instead:
> the graded error is then max(mean, variance, functional) and the coarse level
> scores only the functional, so plotting them together would draw a curve out of
> two different summaries of one solver.
>
> **4. The per-plan record named the wrong quantity.** `harness_error_metric` said
> `"worst moment error / its gate"` on every SDE, including the ten that have no
> moment gates at all — their number comes from the price. The ranking axis itself
> was right (step 7 taught `plan_harness_error` to read a functional); only the
> label beside it was stale. It now says what it measured.
>
> **The smoke test, end to end.** With the step-8 bridge solver planted as a plan
> and no pipeline call (`run.py --only sde_heston_dnt --skip-run --results-json
> <scratch>`), the whole non-agent path runs: `VERIFIED_PASS`, verified score 10,
> `funcErr[no_touch_S75: 2.2e-4, no_touch_S100: 1.8e-4, no_touch_S125: 2.4e-3]`
> against the 6e-3 gate, the two-level check passing with all three values moved,
> and `REPORT.md` rendering the entry with its per-value errors. The planted plan
> carries no `problem_spec.json`, so the kernel block reads `unavailable` and the
> per-plan table shows `UNDERCLAIM` — an artifact of the plant, not of the
> harness, and the reason that row is not quoted as a result.
>
> **Recorded, not changed: the manual is not neutral across this suite.**
> `--manuals both` is step 10's knowledge contrast, and `references/sde_manual.md`
> mentions none of barriers, bridges, Heston, QE, SABR, CEV or absorption. It does
> cover one of the three decisions the calibration showed the grade turns on: under
> *CIR* it prescribes `np.maximum(X_n, 0.0)` before the square root — full
> truncation, the correct half of the variance pair — but as a floating-point
> guard under a *satisfied* Feller condition, where all four of the entries it
> bears on violate it and the choice is a modelling decision rather than a rounding
> one. So the C1 arm is handed the answer on the four variance problems, in the
> right form for the wrong stated reason, and nothing at all on the barrier (two
> entries plus a twin) or the CEV boundary (three). The manual is a pipeline input and editing one makes
> earlier runs incomparable, so it stays as it is and step 11 reads the contrast
> knowing this.
>
> **What step 11 inherits.** `report.py` and `compare.py` contain no occurrence of
> `work_precision` — the series reaches `results.json` and nothing renders it —
> and the report's methodology section describes two suites, not three. Both are
> step 11's own scope; they are named here because the smoke test is what
> confirmed them rather than assumed them.
>
> **Tests.** `test_benchop_suite.py` gains **M6**: what staging writes is the call
> the verifier will make, carries no value the entry is graded on, and lands in a
> manifest that says which suite it belongs to (111 → 142).
> `test_orchestration.py` gains **E7**, five tests on the refusal, including
> through `run.main` so that the guard's position — after `--list`, before
> anything is read or written — is pinned. `test_work_precision.py` gains **I5**,
> five tests on the free point: two points for two solves, the free point carrying
> the coarse level's own error and not the graded one's, the sweep's off switch
> unable to decline a point that costs nothing, and a declared level that names it
> not re-solving it. `benchmark/tests` goes from 422 to **464 collected (449
> passed, 15 deselected, 66 s)**.
>
> Whole suite after the step:
> `uv run --extra dev python -m pytest verifylib benchmark/tests -q` →
> **1023 passed, 1 failed, 3 skipped, 15 deselected in 795 s**. The one failure is
> the `test_certified.py` count assertion steps 5 to 8 each recorded, and still
> predates this plan; the 795 s is inflated by the calibration re-run sharing the
> machine.

**10. Pipeline run, then the Opus C0 baseline.**
[oneshot.py:189](../benchmark/oneshot.py#L189) records that the default
`--c0-max-turns 2` **is not enough for Opus — 36 of 44 units produced no solver at all**
on the 2026-09-27 run, because Opus spends both turns writing convergence studies and
probing its environment. The cap is recorded in the results config, and only runs
sharing it may be compared.

```bash
uv run python benchmark/oneshot.py --suite benchop --condition c0 \
    --model opus --c0-max-turns 8 --reps 5 --manuals both \
    --results-json benchmark/results/baselines/oneshot_benchop.json   # [rev, step 9]
uv run python benchmark/oneshot.py --redo CRASHED ...   # if any still produce nothing
```

**[rev, step 9]** The results path is not optional any more — both runners refuse a
selection outside the `main` suite without one, and the refusal names this path. The
`--suite benchop` spelling replaces `--only <benchop slugs>`: `oneshot.py` applies the
suite filter before the discriminator and type filters, so the suite selects itself.
Step 9 also measured what the `--manuals both` arm is actually contrasting here, which
is less than it is on the main suite: see the staging record above.

> **In progress, 2026-10-24.** The pipeline arm is part-run and the C0 baseline has not
> started; this entry exists so that two harness changes already in the tree are not
> undocumented. The full record replaces it when the step finishes.
>
> **Where the runs are.** Four of ten graded, all **UNDERCLAIM** — the pipeline stopped
> at 9, 7, 9 and 9 of 10 while the harness confirmed all four against the published
> values, worst value between 17% and 96% of its gate. B05 and B06 returned
> `API Error: Connection closed mid-response` after the machine slept mid-run
> (`wall_seconds` 1140.4 against `wall_clock_seconds` 2774.6 — the gap
> `run_conductor` records precisely so this is visible), so they are environment and
> not results; `--resume` retries them. B07-B10 are untouched. A Pro session window
> yields about two problems.
>
> **[new, step 10] The CLI cut the conductor off before it could score itself.**
> Headless `claude -p` terminates still-running background tasks 600 s after the main
> thread closes. One evaluator pass costs minutes here, so B01's first run ended on
> "Background tasks still running after 600s; terminating" and recorded phase
> `running`, self-score **0** and both plans at iter 0 — while the harness graded those
> same two plans a verified pass at 1.0e-3 against a 4e-3 gate. A zero meaning "the CLI
> stopped waiting" is indistinguishable in the C2 column from a zero meaning "the
> pipeline failed", and `best_plan` among plans tied at 0 is arbitrary, which is what
> `wrong_plan_won` is computed from. `run.py` now sets
> `CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0` (in 2.1.226, `ceilingExceeded = Uh > 0 &&
> ...`, so 0 disables the sweep rather than setting a zero deadline) and records it in
> `config.conductor_env`, leaving `--timeout` the one bound on a run. The same problem
> re-run came back phase `done`, self-score 9, in **36.4 m against 48.7 m** — faster,
> because the old run spent its last 600 s waiting on tasks it then killed. The
> truncated record is kept as
> `results/superseded/benchop.bg-ceiling-truncated-2026-10-22.json`.
>
> **[new, step 10] The ranking metric had no margin on a price.** `plan_harness_error`
> gives the moment branch a margin — "a difference smaller than the sampling noise
> never reads as a wrong plan", measured on `sde_double_well_langevin` — and the
> functional branch took its from the record's `reference_abs_err`, which `verify.py`
> exports **only where the reference floor widened a gate**, which on this suite it
> never does. The margin was therefore 0.0 on every plan and any difference at all read
> as a mis-ranking: the report's headline was `wrong_plan_won` **3/3**. The missing term
> is the solver's own Monte-Carlo error, which a returned scalar does not carry — but
> step 5's two-level check already re-solves at `r = n_hi/n_lo`, and for nested
> sampling `Var(graded - coarse) = sigma^2 (r-1)/n_hi`, so `SE(graded) =
> |delta|/sqrt(r-1)`. Independent draws would give `sqrt(r+1)`, which is smaller;
> whether a solver's streams nest is unknowable, so the margin takes the larger
> estimate and errs towards *not* accusing the conductor. Combined in quadrature with
> the reference's own uncertainty, read from the problem rather than the record.
> Recomputed on the first three: **1/3**, and the survivor clears its margin by 5% of a
> one-degree-of-freedom noise estimate, which is not a finding. B01's gap is 0.233
> against a margin of 0.529.
>
> **Tests.** `test_orchestration.py` gains **E8** (29 → 31): the env var reaches
> `subprocess.run` with the rest of the environment inherited rather than replaced.
> `test_sde_functional.py` gains **K6** (44 → 52): the sampling term, the quadrature,
> the conservative divisor, four degenerate inputs that must not fabricate a margin,
> and the comparison itself on two plans a hair apart.
>
> **Two findings held for the full record**, since both want the remaining six runs
> behind them: the kernel reads none of the nine `evaluation_thresholds.functionals`
> keys these specs declare, so the conductor ranks its plans without ever seeing the
> price it is graded on; and three of the four cap at 9/10 on `dynkin: unavailable`,
> the ladder's finest level needing 12.8 GB against a 16 GB machine.

**11. Compare and report** with the work-precision axis folded into `compare.py` and
`report.py`. **[rev, step 9]** Two specifics, both confirmed by the smoke test rather
than assumed: neither file contains the string `work_precision` today, so the series
reaches `results.json` and nothing reads it; and the report's methodology section
describes two suites, so a benchop report would not say what anchors it. Step 9 made
the series worth rendering — every entry now emits two points instead of one, at no
extra compute — and left the rendering here.

---

## 7. What this can and cannot claim

**It cannot claim to beat BENCHOP's methods on speed.** The published curves are MATLAB
p-code on 2017 hardware; these will be Python/NumPy on 2026 hardware. Those are not
comparable, and no measurement here will support "we beat ADI."

**The timing axis is still worth having**, for the comparison it *can* make: pipeline vs
C0 vs C1 at matched accuracy, and scheme vs scheme within a problem — which the
conductor already decides qualitatively and could now decide quantitatively.

**Accuracy against the reference values is a real external claim**, and it is the point.

---

## 8. Open questions

- ~~**SABR is the least de-risked problem.** β=0.5 gives a degenerate diffusion with an
  absorbing boundary at S=0; set II runs to T=10 with α=0.8; and the formulations warn
  that time-discretised MC loses the martingale property and needs a correction. None of
  this was measured. Expect it to be the hardest of the eight and budget accordingly.~~
  **[rev, step 6] Measured, and it is not the hardest — `HHWeuCallI` is.** All three
  warnings turned out to have clean answers. The absorbing boundary is a *property* once
  the right chart is used: in `U = 2√S` the asset is a Bessel process of dimension 0, so
  the step is drift-implicit and absorbs exactly when its quadratic loses the real root.
  The lost martingale property is real and was measured — `E[S_T] = 0.069861` against
  `S₀ = 0.07` at 1600 steps — and because the loss is measurable per path it is also
  removable: `E[S_T] = S₀` as a control variate improved set II's agreement by 14×.
  Both sets are now confirmed by two independent families, a sampler and a grid, to a
  few times 1e-6.
- ~~**QLSV's diffusion is superlinear in S** (`f(s) = 0.01s²`), so the martingale
  property is at risk there too — the same failure mode as the existing quintic/tamed
  problems, which is a useful continuity but also an unmeasured risk.~~ **[rev, step
  6] Measured 2026-10-17, and it is not a risk, it is the state of the problem.** The
  asset is a strict local martingale: `100/S` is a Bessel process of dimension 3, so
  `E[S_T] = S₀·erf(r₀/√(2θ))` is strictly below `S₀`, by 2.23 at S₀ = 125 with the
  variance frozen and by 0.27 with it live. The published values follow the
  grid-with-linear-far-field convention rather than the expectation. See §0's rev and
  §6 step 6; the consequence is that `QLSVeuCall` is a PDE-only problem.
- ~~**`HHWeuCallII`'s derivation needs tightening** from 3e-5 to something near the
  reference's own precision before it can serve as harness-owned truth.~~
  **Closed 2026-10-17, step 6**: 5.7e-14 absolute, by collapsing the two quadratures
  into one rather than by refining either (§2a rev).
- **`functional_tol` for SABR is unknown.** 1e-3 relative is 3.9e-5 absolute on set II;
  calibrate before committing. **[rev, step 4]** Still unknown — step 4 built the
  mechanism, not the number — but it now has somewhere to land that is not a single
  loosened tolerance: if the reference values themselves turn out to be the limit, that
  belongs in `reference_abs_err`, where it widens the small values and leaves the rest
  alone. A `functional_tol` loosened to cover the worst value is a gate the other values
  no longer feel. **[rev, step 6]** The reference values are *not* the limit on
  SABR, which is the question step 4 left open: on set II the two independent families
  agree with the published value to 3.5e-6 (sampler) and 1.7e-6 (grid) absolute, so
  1e-3 relative — 3.9e-5 absolute — sits about twenty times above anything the answer
  key's own precision would impose. Whatever `functional_tol` ends up being on SABR, it
  will be set by what a solver can reach inside the budget, not by the reference.
  **[rev, step 8] Closed, and the two sets closed differently.** Set I's **4e-3**
  separates an absorbing Euler (0.29× the gate) from the same step reflecting off
  zero (1.97×) — the narrowest margin anywhere in the suite, so that gate is doing
  real work rather than sitting loose. Set II's **1e-2** separates reflection by
  four orders of magnitude but does not resolve a plain absorbing Euler in S,
  which lands at 0.90–0.99× of it; what clears set II is the Lamperti scheme, at
  0.03×. See the new item below.
- ~~**Whether the shifted twins double the suite or sample it.** Sixteen problems is a
  lot of pipeline runs; a subset of twins may be enough to detect recall.~~
  **Settled in step 7: they sample it.** Three twins against seven published
  problems, chosen to cover the three distinct failure modes rather than the three
  hardest — the European call (where recall of `9.046650119220969` is likeliest),
  the continuously monitored barrier (the suite's best discriminator) and the
  ten-year SABR set (the hardest). Ten problems, not sixteen.
- **[new, step 5] What `num_paths` means to MLMC.** The two-level check varies the path
  count and requires the price to move, which is exactly right for a plain Monte-Carlo
  estimator and less obviously right for MLMC and SGBM, where `num_paths` is a budget
  split across levels rather than a sample size. The price will still move — the
  allocation changes — but the `1/√n` scaling the band is derived from is a plain-MC
  law, so the band may be the wrong width there rather than the wrong shape. Measure it
  on the first MLMC entry before trusting the default; `functional_level_paths` exists
  so a problem can widen the gap instead of being graded at a band that does not fit.
- ~~**[rev, step 7] The gate on `HHWeuCallI` is 3e-3**, three times the suite's
  ordinary 1e-3 and set by the item below rather than by anything a solver did. It
  is the one gate in the suite whose number comes from the weakness of the answer
  key rather than from what Monte Carlo can resolve, and step 8 should say so when
  it calibrates, or drop the problem.~~ **[rev, step 8] It calibrated, and the
  problem is kept.** The gate's number is no longer a judgement about the answer
  key's weakness; it is what the measurement leaves room for. The ledger's primary
  derivation sits **8.1e-4 / 7.3e-4 / 5.5e-4 above** the three published values, so
  a 1e-3 gate would be spent **81% / 73% / 55%** before a solver wrote a line — and
  96% at S₀ = 75 for the same sampler without its bias control. At 3e-3 the worst
  of those is 27%, a plain full-truncation Euler at the shipped settings reaches
  28%, and reflecting the variance misses by 82×. What the calibration does *not*
  settle is the item below: every one of those numbers is the same sampler at
  different settings.
- **[new, step 6] `HHWeuCallI` is the one value this repo cannot check to its own
  quoted precision, and it is the one most likely to need a loosened gate.** The
  independent derivation sits **+0.030 to +0.043 above** the three published values —
  six to nine times the ±0.005 those values claim for themselves. It is not the step
  size: quadrupling the step count from 2000 to 8000 moves the estimate by 0.005 while
  the gap stays at 0.04, and on set II, where the same sampler can be checked against a
  closed form, its bias at those settings is +0.010. What cannot be settled from here is
  whether the residual is this sampler's bias under set I's correlation structure or the
  published values' own error, and the two call for opposite responses. `reference_abs_err`
  stays at the published 5e-3, because that field means *the answer key's* uncertainty and
  not ours — but **step 8 must treat a 1e-3 relative gate on this problem as unproven**,
  since 1e-3 of 37.52 is 0.0375 and the best derivation this repo has does not clear it.
  **[rev, step 8]** Unproven is what it stayed; the gate is 3e-3 and the item above
  records what that buys. The residual is unchanged and still one-sided — every
  route sits *above* the published value, never below — and still from a single
  sampler, which is the item below this one and the thing that would settle it.
- **[new, step 6] A second method family for `HHWeuCallI` exists and was not built.**
  Under the T-forward measure the three factors collapse to two: with `F = S/P(t,T)`,
  `dF/F = √V dŴ¹ + σ₂B(τ)dŴ³` and
  `dV = [κ(η−V) − σ₁σ₂ρ₂₃B(τ)√V]dτ + σ₁√V dŴ²`, where `B(τ) = (1−e^{−aτ})/a`, and the
  price is `P(0,T)·E^T[(F_T−K)^+]`. That is an exact two-dimensional problem and the
  reduction the formulations refer to. It was not built because its coefficients depend
  on τ, and `fd2d_operator` factorizes once and reuses the factorization for every step —
  which is what makes the other grids cheap. Building it needs either a vectorized
  re-assembly per step or an ADI splitting. This is the one piece of step 6 left
  undone, and it is the obvious way to settle the item above.
- ~~**[new, step 5] The 25% solve-time surcharge** on every functional-scored SDE
  problem, against the 300 s budget. Harmless at this suite's path counts and not
  measured on BENCHOP's heavier methods.~~ **[rev, step 7] Measured, and it is not
  the binding constraint.** A plain full-truncation-Euler Heston solver returns all
  three prices at two million paths in **24 s**, against a 300 s cap; the second
  call at a quarter of the paths adds about 6 s. What binds is *noise*, not time —
  the path count is set by the standard error a gate has to clear, and on every
  problem here it was chosen from measurement rather than from the clock. The one
  place the clock does bind is the ten-year pair, where `sde_verify_dt`'s `0.04/T`
  cap forces 2500 steps whatever the entry declares, which caps the affordable path
  count and is why `sde_sabr_cev_call_ii` carries the loosest gate in the suite
  (1e-2) rather than the 4e-3 its one-year siblings get.
- ~~**[new, step 7] The gates are calibrated against a correct scheme only.** Step 8
  owes the other half of the repo's own rule — that a *plausible wrong* scheme must
  fail. It is already done for one problem: on the continuously monitored barrier,
  a Brownian-bridge estimator measures 2.1e-3 at worst while per-step monitoring at
  four times the step count measures 5.0e-3, 3.1e-3 and 1.4e-2 (§2d), and the 6e-3
  gate sits between them. The other six have no such pair measured yet.~~
  **Done in step 8, for all ten rather than the seven this item counted** — the
  table is in §6 step 8. Every gate separates; the narrowest is SABR set I at
  0.29× against 1.97×. One qualification survives into the item below.

- **[new, step 9] The knowledge contrast is uneven across this suite.** Step 10's
  `--manuals both` arm hands the solver `references/sde_manual.md`, which names none
  of barriers, bridges, Heston, QE, SABR, CEV or absorption, and does prescribe
  `np.maximum(X_n, 0.0)` before a CIR square root — full truncation, the correct
  half of one of the three decisions, given as a rounding guard under a *satisfied*
  Feller condition rather than as the modelling choice it is on the four entries
  that violate it. So C1 is handed the answer on four entries and nothing on the
  other six, and the four it helps are the ones whose wrong scheme was already
  missing by 14–82×. A C1-over-C0 margin on this suite is therefore not one
  measurement; it should be read per decision, not pooled. The manual is a pipeline
  input, so correcting it would make every earlier run incomparable — the
  alternative, if the pooled number matters, is a third knowledge arm rather than an
  edit.

- **[new, step 8] One gate returns a coin flip on one scheme, and it was left
  there.** On `sde_sabr_cev_call_ii` a plain absorbing Euler in S measures 0.90,
  0.94 and **0.99** of the 1e-2 gate, so whether that scheme passes is decided by
  noise. The gate was not re-cut: moving a line after seeing which scheme lands on
  it is how a benchmark stops meaning anything, and the scheme is not wrong — it
  absorbs at the right boundary and converges — it is under-resolved at a step the
  verifier caps at 0.004. What would fix it properly is more paths or more steps,
  and the 300 s budget has neither: `sabr_mc` already takes 151 s at 1M × 2500. So
  the honest position is that B02 separates the *wrong* scheme (reflection, by four
  orders of magnitude) and does not resolve the *generic* one, and a reader of a
  B02 verdict should know which of the two a solver wrote.
