# BENCHOP-SLV — External Benchmark Adoption Plan

**Status:** planned, not implemented. Research and feasibility measurement done 2026-10-16.
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

**`HHWeuCallII`.** With ρ₁₃=ρ₂₃=0 the Hull–White rate is independent of (S,V), so
writing `M = ∫₀ᵀ R dτ` — Gaussian, measured `mean 1.0000000000, var 1.5457975016` —
the price collapses to `E_M[ ZeroRateHestonCall(S₀, K·e^{−M}) ]`, a 1-D Gauss–Hermite
quadrature over Heston calls at shifted strikes:

| S₀ | independent derivation | BENCHOP reference | abs diff | rel diff |
|---|---|---|---|---|
| 75 | 35.436815841033734 | 35.437896876285350 | 1.08e-03 | 3.05e-05 |
| 100 | 54.712482227869124 | 54.728065308229503 | 1.56e-02 | 2.85e-04 |
| 125 | 75.395112101906307 | 75.397596834993621 | 2.48e-03 | 3.30e-05 |

The route is sound; reaching the reference's own precision needs tighter quadrature
(140 Gauss–Hermite nodes over a σ=1.24 Gaussian, with the inner Heston integral
evaluated at strikes spanning `K·e^{−1±6}`, is not enough).

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
clear a tight gate by brute force. The fix is a *named technique*: the Brownian-bridge
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
| Scalar quantity of interest | `functional_truth` **exists** ([problems.py:2398](../benchmark/problems.py#L2398), `pde_schrodinger_eigen_2d`) | **missing** — contract returns terminal moments only |
| All three values must pass | `_pde_converged` already requires every functional within tol | — |
| Anti-hardcoding defence | 2-grid order check: a constant shows zero convergence and fails | **none** — no refinement ladder on a price |
| Per-problem tolerance | `_pde_functional_tol` ([verify.py:569](../benchmark/verify.py#L569)) | — |
| Solve timing | `wall_s` only (whole subprocess, incl. import) | same |

Two gaps to close, and they are the first two implementation steps: **timing** and
**SDE functionals**. Everything else is problem authoring.

One naming hazard: on a functional-only problem,
[verify.py:797](../benchmark/verify.py#L797) stores a *relative* error in a field called
`max_abs_err`. Fix that before introducing a genuine absolute error or the two will be
silently conflated.

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

### Decision

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
be rendered without the *verdict* depending on the absolute number.

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

**The order check defends the PDE route.** When functionals are the only truth,
`_pde_eval_run` makes the functional error the primary error, so the 2-grid order check
measures convergence *of the price itself*. A hardcoded constant has zero convergence
and fails. §2b's refinement table is what a real solver's trace looks like. **The SDE
route has no equivalent**, which is an independent reason to require a convergence check
in step 5 below, and to build the PDE route first.

**Run a shifted-parameter control.** Pair each published problem with a twin at a
de-memorized parameter (κ = 2.58 → 2.43) whose truth the harness computes itself. A
baseline that scores well on the published set and badly on the shifted twin has been
caught recalling rather than solving. This converts the risk into a measurement, which
is what the benchmark is for. The twins also give the SDE route an anti-hardcoding
signal it otherwise lacks.

---

## 6. Steps

**1. Time the solve call, not the process.** `run_solver` returns `wall_s`, but that is
the whole subprocess including interpreter startup and imports — noise in a
work-precision diagram. In [`_child_main`](../benchmark/runner.py#L261), wrap the two
`res = fn(...)` sites (PDE ≈ line 300, SDE ≈ line 317) and record
`run_meta["solve_s"]`. Keep `wall_s`; the difference is the overhead measurement.

**2. Make pipeline and baseline timings comparable.** Baseline solvers run sandboxed;
pipeline solvers are imported and run inline (`sandbox=False` in `verify_problem`).
Those are different measurement conditions, so a pipeline-vs-C0 *time* comparison would
be measuring the harness. Either route both through `runner.run_solver` for scoring, or
record which path produced each number and never compare across them. Routing both
through the sandbox is cleaner.

**3. Get the work-precision curve from the ladder already being run.** The PDE path
already solves at `N` and `2N` — two (error, time) points for free. Add
`work_precision_levels` (e.g. `[N/2, N, 2N, 4N]`) reusing the ladder in
[`_score_pde_runs`](../benchmark/verify.py#L907), and emit the series into
`results.json`. For SDE, sweep `(num_paths, dt)` the same way.

**4. Tolerance plumbing.** Per-value absolute floor in `_functional_errs`;
`reference_abs_err` carried on the problem; record relative *and* absolute per
functional; fix the `max_abs_err` naming slip.

**5. Add SDE functional support.** Four of BENCHOP's eight methods are SDE-side (MCA,
MLMC, SGBM, mSABR), so the full benchmark needs it. Port the PDE pattern:
`functional_truth` on the problem, a `functionals` dict in the SDE return contract, a
functional branch in `_score_sde_moments`. **Include a convergence requirement** — the
price at two path counts / two steps, checked to be converging — or the SDE route has no
defence against a hardcoded constant.

**6. Author the ground truth.** Independently derivable, implementations verified in §2a:
`HSTeuCall` (Heston CF) and `HHWeuCallII` (Gauss–Hermite over the HW integral; tighten
the quadrature first). The other six have no formula and need the
[`make_references.py`](../benchmark/make_references.py) route — two independent method
families, disagreement recorded as `reference_error`. For `HHWeuCallI` the published
±0.005 is the floor however good the reference is.

**7. Write the problem entries.** Eight PDE entries — SABR is 2-D in (S,σ), HHW is 3-D
or its 2-D reduction (state which, the formulations ask) — plus SDE variants once step 5
lands, plus the shifted twins from §5. Three functionals each.

**8. Cross-check and calibrate.** Add the derivations to
`validate_ground_truth.py`, then apply the repo's own calibration rule: a correct scheme
must clear the gate at the benchmark's own resolution and a plausible wrong one must
fail. For `HSTdnTouch` the wrong-scheme case is already characterised — naive discrete
monitoring, biased high by the §2d amounts.

**9. Stage and smoke-test.**
```bash
uv run python benchmark/setup.py --only pde_benchop_hst_eucall
uv run python benchmark/validate_ground_truth.py
uv run python benchmark/run.py --only pde_benchop_hst_eucall --timeout 1800
```

**10. Pipeline run, then the Opus C0 baseline.**
[oneshot.py:189](../benchmark/oneshot.py#L189) records that the default
`--c0-max-turns 2` **is not enough for Opus — 36 of 44 units produced no solver at all**
on the 2026-09-27 run, because Opus spends both turns writing convergence studies and
probing its environment. The cap is recorded in the results config, and only runs
sharing it may be compared.

```bash
uv run python benchmark/oneshot.py --only <benchop slugs> --condition c0 \
    --model opus --c0-max-turns 8 --reps 5 --manuals both
uv run python benchmark/oneshot.py --redo CRASHED ...   # if any still produce nothing
```

**11. Compare and report** with the work-precision axis folded into `compare.py` and
`report.py`.

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

- **SABR is the least de-risked problem.** β=0.5 gives a degenerate diffusion with an
  absorbing boundary at S=0; set II runs to T=10 with α=0.8; and the formulations warn
  that time-discretised MC loses the martingale property and needs a correction. None of
  this was measured. Expect it to be the hardest of the eight and budget accordingly.
- **QLSV's diffusion is superlinear in S** (`f(s) = 0.01s²`), so the martingale property
  is at risk there too — the same failure mode as the existing quintic/tamed problems,
  which is a useful continuity but also an unmeasured risk.
- **`HHWeuCallII`'s derivation needs tightening** from 3e-5 to something near the
  reference's own precision before it can serve as harness-owned truth.
- **`functional_tol` for SABR is unknown.** 1e-3 relative is 3.9e-5 absolute on set II;
  calibrate before committing.
- **Whether the shifted twins double the suite or sample it.** Sixteen problems is a lot
  of pipeline runs; a subset of twins may be enough to detect recall.
