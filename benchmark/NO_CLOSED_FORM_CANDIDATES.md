# Problems With No Closed Form — sourcing playbook and candidate catalogue

Companion to [`hard_sde_candidates_from_lit.md`](hard_sde_candidates_from_lit.md) and
[`HARDNUMERICS_IMPORT_LIST.md`](HARDNUMERICS_IMPORT_LIST.md). Design:
[`docs/plan-no-closed-form.md`](../docs/plan-no-closed-form.md).

## Why this file exists

Of the 50 benchmark problems, **46 are `exact`**, 2 are SDE `reference`, 2 are SDE `stability`, and
exactly **one has no closed form** — `pde_kuramoto_sivashinsky`, which the harness could not grade
at all until the `reference` PDE path landed. Worse, most of the *hard* tier-3 PDEs are **MMS**: a
manufactured problem hands the formulator a closed form, so a pipeline built to score without one
never gets exercised.

That is the gap this catalogue fills. It is written for the *sourcing* question — where do
no-closed-form problems come from, and what grades them — not for any particular pipeline
capability. A problem that outruns the current tiers is a finding, not a reason to drop it.

---

## 1. The constraint that shapes everything: "no closed form" ≠ "no ground truth"

A problem with no independent check reports `SELF_ONLY` and measures nothing — the benchmark's whole
value is being able to say `OVERCLAIM`. So every candidate must arrive with a route to truth that
does not go through a formula the pipeline could also write down.

| Route | Truth is | Harness support | Cost to author |
|---|---|---|---|
| **R** — harness-owned reference field | a stored high-resolution solution; two independent schemes agreeing far below the problem's tolerance | `ground_truth_kind: "reference"` (PDE + SDE) | you compute and validate it |
| **S** — semi-analytic exact | a quadrature, series, root-find, eigen-expansion or stationary density evaluated to machine precision | none — reuses the `exact` path | **cheapest; best value** |
| **F** — functional (scalar QoI) | an eigenvalue, extremum, drag coefficient, Nusselt number | `functionals` return key + `functional_truth` | cheap, but see §3 |
| **I** — structure only | conservation, positivity, a maximum principle, a known steady state, non-triviality | the existing `diagnostics` gate | cheapest, weakest |

Route I never ships alone: a wrong-but-conservative scheme passes it. Pair it with R, S or F.

---

## 2. Seven ways to source one

Ordered by cost. The first two are nearly free and generate **matched pairs** — a new problem whose
only difference from an existing one is that the closed form is gone, which is the controlled
experiment for anything that claims to score without a formula.

1. **Delete the manufactured source.** Every MMS entry in `problems.py` becomes a no-closed-form
   problem by dropping `f` and evolving a deterministic initial condition instead. Same operator,
   same diagnostics, same `problem.md` skeleton. `cahn_hilliard_2d`, `keller_segel_2d`, `mhd_2d`,
   `elasticity_2d`, `fichera_3d`, `acoustic_3d_layered` are all sitting there waiting.
2. **De-integrate a solvable problem.** Swap a constant coefficient for `κ(x)`; use a non-separable
   domain; start from a non-soliton initial condition; add a quintic term to a cubic; break Feller.
   The difficulty story and the family write-up already exist.
3. **Compute the truth yourself** (R). Two independent *method families* — spectral ETDRK4 against
   IMEX-SBDF3, pseudospectral against high-order compact FD — refined until they agree several
   orders below the problem's own tolerance. This generalises to almost any well-posed problem.
   Measured on KS in `plan-no-closed-form.md` §0: **1.26e-9** between ETDRK4 and IMEX-SBDF3 at
   t = 50, six orders below the 1% target, in about ten seconds of compute.
4. **Semi-analytic truth** (S). Cole–Hopf plus quadrature; a Sturm–Liouville eigen-expansion;
   Feynman–Kac quadrature; a transcendental root-find; a stationary density `∝ exp(-2V/σ²)`. The
   repo already leans on this — `_stefan_1d`'s `brentq`, `_heston_call`'s characteristic-function
   quadrature, `mittag_leffler_series` — it just never named it as the no-closed-form route it is.
   ~~None of these can be written in the restricted expression namespace, so the pipeline genuinely
   has no formula while the harness has machine-precision truth.~~ **False, measured 2026-09-19:**
   the `pde_burgers_viscous_1d` formulator wrote the Cole–Hopf quadrature as a 2,010-character
   nested-lambda expression over `np.linspace(…, 4001)` and `np.sum`, it passed the expression gate,
   evaluated in the restricted namespace, agrees with the harness's own `_burgers_viscous_1d` to
   2.2e-14, and put the problem on Path A with provenance `analytic`
   (`benchmark/tests/test_leakage_ncf.py::test_staged_workspace_specs_take_the_route_the_problem_allows`
   reports it). A quadrature program *is* expressible. The kernel now flags such a formula
   `analytic_numerical` (with its node count) rather than demoting it — the formulator did
   something genuinely strong, and the residual check remains the ceiling — so a Route-S problem
   sits on the boundary of "no closed form" and the `closed_form: False` flag must not be read as
   "Path B" in any report. The experimental fix is on the benchmark side (§2.2: `ν = ν(x)` kills
   Cole–Hopf while keeping the operator and the layer), still undone.
5. **Published benchmark values** (F) — see §3 for the rule that governs them. Where to look:
   Schäfer–Turek DFG flow around a cylinder (drag, lift, Strouhal); Ghia et al. and Botella–Peyret
   for the lid-driven cavity (the latter spectral, ~1e-8); de Vahl Davis and Le Quéré for natural
   convection (Nusselt); Trefethen–Betcke's method of particular solutions for L-shape, sector and
   Fichera eigenvalues (~14 digits); the SIAM 100-Digit Challenge (several answers are PDE or
   probability values to 10 digits); Brown & Minion's double shear layer; the Orszag–Tang vortex;
   the deep-BSDE catalogue already logged as Tier D in the SDE lit file.
6. **Structure only** (I). Conserved mass and energy, positivity, a maximum principle, a known decay
   rate, non-triviality. Cheap, and it is what the `stability` kind already does for SDEs.
7. **Cross-route agreement.** The same quantity computed two unrelated ways — a Feynman–Kac Monte
   Carlo of an SDE against a spectral solve of its Kolmogorov PDE. Expensive, and the strongest
   independence available: it validates the *problem statement*, not just the solver.

---

## 3. The memorization trap

A famous published constant — the L-shape's `λ₁ = 9.6397238440219…`, DFG cylinder drag, de Vahl
Davis Nusselt numbers — can be **recalled** by a one-shot LLM that solved nothing. That corrupts
precisely the C0-vs-C2 comparison the benchmark exists to make, and it is invisible in the score.

> **Literature values validate our reference; they never grade the solver.** Grade at a
> de-memorized parameter — a sector angle, a wavenumber, a Reynolds number no paper tabulates —
> whose truth we computed ourselves and cross-checked against the published case.

The same reasoning is why Route F problems should carry a field check alongside the functional
wherever one is affordable: a returned constant with no solution behind it then fails on the field.

---

## 4. Pilot slate (built) and the second-wave additions

Four pilot problems (2026-09-19), all four routes, three matched pairs, one conversion; three
second-wave problems added 2026-09-21 as the **held-out set** the findings document's §10 asks for
-- every fix in that document was designed on KS and Cahn-Hilliard, and these are the problems it
is accepted on, including the first real problem through `kernel/sde/` Path B. Every one is
validated in `validate_ground_truth.py` in both directions -- a correct scheme clears every gate at
the benchmark's own resolutions, and a deliberately broken one is caught by the gate it is meant
to trip. For the second wave the two directions are run with *real* compact solvers through the
real scoring path (FD4+RK4 passes, FD2+RK4 fails, a dissipative flux trips the conservation gate;
stochastic Heun passes, Euler-Maruyama fails), not with synthetic fields.

| Slug | Route | Relationship | What a solver has to get right | Measured |
|---|---|---|---|---|
| `pde_kuramoto_sivashinsky` | R | existing P15, `SELF_ONLY` -> graded | chaotic, stiff, fourth-order | ETDRK4 vs IMEX-SBDF3 agree to **1.2e-09**; a correct solve is 30% off at N=64 and 6.6e-05 at N=128. **Graded at N=128 alone since 2026-09-20** (findings §11.2): the 64/128 order gate measured garbage → resolved and could not reject anything, so it is off, and the 1% accuracy gate stays at the grid that exposed the 7.2e-02 FD4 OVERCLAIM |
| `pde_burgers_viscous_1d` | S | pair with P11 `burgers_inviscid` (exact) | an internal layer of width `~2 nu` at Re = 100 | Cole-Hopf quadrature converged to **5.6e-16**; spectral 8.8e-03 -> 1.1e-04, conservative FD2 4.8e-02 -> 8.0e-03 |
| `pde_cahn_hilliard_2d_coarsening` | R + I | pair with P20 `cahn_hilliard_2d` (**MMS**) | coarsening dynamics, not just stability | agreement **5.9e-10**; ETDRK4 stays inside 1% even at dt = 2e-3, while the first-order stabilized IMEX is 5.8e-02 at dt = 1e-3 |
| `pde_schrodinger_eigen_2d` | S + F | pair with P19 `poisson_lshape` (masked domain) | an eigen*problem*: a number and a mode, from the right end of the spectrum | `lambda_1 = -8.442746362964` to **1e-11**; FD2 is 1.19e-03 at N=64 and 2.93e-04 at N=128 |
| `pde_kdv_zabusky_kruskal` | R + I | standalone (second wave) | dispersive third-order operator; a cosine fissions into solitons of width `~delta` whose phases need fourth-order accuracy at the graded grid | ETDRK4 (M=1024) vs a fully explicit RK4 on the spectral ODE agree to **3.3e-10**, invariants held to 1e-11; at t = 1, FD4+RK4 is 2.8e-03 at N=256 and 1.8e-04 at N=512 (order 3.97), FD2+RK4 7.7e-02 and 1.9e-02 -- outside the gate at both; first-order upwinding of `u u_x` loses 52% of the L2 invariant, which the gate catches on its own |
| `pde_gross_pitaevskii_1d` | R + I | standalone (second wave); the suite's first **non-periodic** reference and second multi-field problem | a trapped cubic NLS as (re, im): the error is a phase error, invisible to `|psi|` | Yoshida-4 split-step, ETDRK4 and explicit RK4 in the Dirichlet sine basis agree to **1.2e-11**; FD4+RK4 8.4e-03 at N=65 and 5.3e-04 at N=130 (order 3.97), FD2+RK4 6.2e-02 and 1.5e-02; a 0.2 rad global phase error is an 18% field error with the mass gate intact |
| `sde_double_well_langevin` | S | standalone (second wave); the first no-closed-form SDE with exact truth | `dX = k(X - X^3)dt + sigma dW` from one well, k = 10, sigma = 2: the mean at T = 1 is the un-hopped fraction, and a first-order weak scheme misprices the escape rate by O(k dt) | backward Kolmogorov by Chebyshev + `expm`, cross-checked by FD4 (**4e-10**), the stationary quadrature (1e-10 at t = 20) and Monte Carlo; at the verifier's `(200000, dt=0.01, seed=42)` Euler is 7.1% off on the mean (fails the 5% gate), stochastic Heun 0.24% (passes) |

Notes that decided the authoring, recorded because each was measured rather than assumed:

- **The eigenvalue problem is not on a re-entrant corner**, which is what the plan first proposed.
  A corner singularity is the more interesting numerics, but its eigenvalue needs the method of
  particular solutions to reach benchmark precision, and a reference I cannot certify is worse than a
  less picturesque one I can. The smooth-potential ground state is exact by a convergent sine
  expansion, is not a number any model can recall, and still punishes a low-order scheme. The
  L-shape variant is in §5.
- **`pde_burgers_viscous_1d` sits on the boundary of the definition.** Cole-Hopf is an explicit
  integral formula, so a solver could evaluate the quadrature instead of discretizing the PDE and
  would pass. That is a legitimate numerical solution rather than a hard-coded answer, and at
  `nu = 0.01` it is not the easy route -- but it is stated rather than hidden.
- **Gray-Scott's trivial attractor is not hypothetical.** At the first parameters tried, the seed
  simply dissolved and the true solution *was* `u = 1, v = 0`. The seed width has to be tied to the
  diffusion length for the problem to have a pattern to get wrong.
- **A reference is only as good as its own error.** Gray-Scott's first build at `M = 256` had a
  spatial check of 3.2e-04, only ~30x below its own 1% gate; `validate_ground_truth.py` rejects that
  by rule, and the grid was doubled rather than the rule loosened.

Second-wave notes (2026-09-21), each a measurement that changed the authoring:

- **The shared ETDRK4 phi-functions were wrong for a complex operator.** `_etdrk4_coefficients`
  took the real part of a half-circle contour average, which is exact for the real operators of KS
  and Cahn-Hilliard and wrong by O(1) for the purely dispersive `i delta^2 k^3` of KdV: the first
  KdV "reference" lost 4% of its L2 invariant by t = 1 and moved by 0.3 under step-halving. The
  helper now uses the full circle and the complex mean when `Lop` is complex; the real path is
  unchanged (the KS and CH artifacts re-check to 5e-14 and 2e-15, round-off). The invariants are now in the
  agreement block precisely so this class of error cannot pass as a small number again.
- **The nesting rule is `lcm(N-1, 2N-1)`, not `N-1`.** The verifier calls the solver at `grid_N` and
  `2 * grid_N` nodes -- `N-1` and `2N-1` intervals -- so a non-periodic reference must have both
  as divisors of its interval count. The earlier note in `problems.py` ("M = 1025 admits N = 65 and
  N = 129") covered only the coarse grid. Gross-Pitaevskii at `grid_N = 65` stores 8256 intervals
  (8257 nodes), filled by zero-padding the sine coefficients of a 1025-node solve, which is exact.
- **Periodic is not Dirichlet at the 1e-7 level.** The trapped condensate is `e^-40` at the box edge
  initially, but by t = 2 the nonlinear dynamics have radiated fast, low-amplitude ripples that
  reach the wall at `|psi| ~ 6e-7` and reflect. A Fourier reference on the same box is therefore a
  different problem (measured ~1e-6 apart), and the kink of `V = x^2/2` at the periodic seam gives
  it an algebraic spectral tail (M=512 vs 1024: 1.3e-09, only 4x per doubling). The sine basis
  carries the stated Dirichlet wall exactly and converges spectrally (n=255 vs 2047: 3e-11).
- **A 1-D additive-noise double well is not an "Euler fails" discriminator on its own.** Every
  first-order weak scheme has an O(dt) bias inside the gates at any step where a correct one passes
  (k = 1, sigma = 1, T = 2: Euler -0.2% on the mean at the verifier's dt). What makes it one is the
  well curvature: with `2k = 20` the escape-rate bias is O(k dt) and at dt = 0.01 Euler is 6-7% off
  on the mean while a weak-order-2 scheme is under 0.3%. Too stiff and nothing standard passes at a
  step the verifier accepts (k = 10, sigma = 3: tamed Euler +31% on the variance at dt = 0.04); the
  shipped `(k, sigma, dt, num_paths) = (10, 2, 0.01, 200000)` is the window where Euler's failure
  is 2.2 sigma of MC noise past the gate and a correct scheme's pass is 6.6 sigma inside it.
- **The graded grid decides who passes.** Gross-Pitaevskii at `grid_N = 129` would let FD2 through
  at the fine grid (3.7e-03 at N=258 with order 2.0); at `grid_N = 65` it cannot (1.5e-02 at
  N=130). The coarser grid is the one that discriminates, and a fourth-order scheme still has an
  order of magnitude of margin there.

### Not shipped, and why — two worked cases

Both were attempted, both were measured, and the numbers are recorded here because the next attempt
should start from them rather than rediscover them.

**`pde_gray_scott_2d` — the reference and the discrimination pull in opposite directions.** This was
meant to be the sharpest instance of the whole category: `u = 1, v = 0` is an *exact* homogeneous
steady state, so a scheme that over-diffuses and destroys the pattern satisfies every term of the
operator to machine precision, having eliminated the only phenomenon the problem is about. The
generator is kept, parked, in `make_references.py`. What the build measured:

| Horizon | Reference certifiable? | Does anything discriminate? |
|---|---|---|
| `T = 1000` | **no** — pattern dynamics amplify spatial truncation error, so doubling the grid only halved the M/2-vs-M disagreement (3.2e-04 → 1.5e-04) instead of collapsing it | — |
| `T = 300` | **yes** — spatial check 4.0e-05, two integrator families agreeing to 2.0e-07 | **no** — a correct solve is already 7.1e-04 at `M = 32` (two points per feature) and only 6.9e-05 at `M = 192`. Every resolution clears the 1% gate; the order check is waived because both grids pass |

The cause is the *metric*, not the physics: the pattern occupies a small fraction of the domain while
`u ~ 1` fills the rest, so relative L2 over the whole field dilutes exactly the error that matters. A
shippable version needs a pattern-weighted metric or a regime whose features fill the domain —
neither of which is a change one can make while calling the result the same problem. Shipping it
as-is would have added a problem that everything passes, which is worse than not shipping it: it
inflates the pass rate without measuring anything.

**`pde_navier_stokes_2d_shear_layer`** (Brown & Minion) — the matched pair to the exact Taylor-Green
problem, where under-resolution grows a **spurious extra vortex** that looks entirely plausible. Not
attempted. It is the best "the wrong answer looks right" case available and needs no new harness
machinery — the multi-field contract, the `div_u` diagnostic and the reference path all exist. What
it needs is the reference build and the resolution calibration, on the pattern above. Given
Gray-Scott's result, calibrate the discrimination **before** paying for the reference: vorticity
fills that domain, so relative L2 should not dilute the way it does here, but that is a prediction
until measured. It is first in §5.

### What the pipeline runs on these problems then surfaced

The three second-wave problems all passed (2026-09-24 to 09-26, `no_closed_form_v2.json`), and on
both PDEs the kernel ranked every plan the way the harness does. What they exposed was on the
pipeline side, and is recorded here because the next problem authored against this playbook will
meet the same things:

- **The graded tolerance decides whether a declared boundary condition is load-bearing.**
  Gross-Pitaevskii is stated with Dirichlet walls, and the winning plan used a *periodic* Fourier
  k-space propagator — passing at 2.0e-06. The wall matters at ~1e-06 (§4, second-wave notes), which
  is four orders below the 1% gate, so the BC is not something this problem tests. That is not a
  defect: the problem is a phase-accuracy test and it worked as one. But an author who wants the
  boundary treatment graded has to either tighten the gate or push the horizon until the radiated
  ripples matter, and should not assume that stating a BC makes it scored.
- **An invariant whose name the kernel's registry does not know is not checked.**
  `kdv_third_invariant` and `gaussianity` were both declared, both ungated, and both silently
  unevaluated (`not_reported`). KdV's third invariant is an *exact* invariant of the equation and one
  plan's own solver measured it drifting 170%/42%/10% across the ladder. The kernel now lists such
  names under `invariants_not_reported` in the metrics block and `formulator.md` says to map the
  invariant onto a registry name, give it an `expr`, or declare it knowingly as unchecked. The
  benchmark side is unaffected — `problems.py` owns its own diagnostics — but a harness-side
  invariant that matters should be a **gate** there rather than a note here.
- **A Monte-Carlo problem needs its standard error recorded, not just its error.** On the
  double-well the three plans' mean errors (1.065%, 0.955%, 0.941%) are all inside a 0.76% standard
  error, i.e. indistinguishable. `verify.py` now records `empirical_se_mean` / `empirical_se_variance`
  and `mean_resolved` / `variance_resolved` per component, and `run.py`'s `wrong_plan_won` — which
  did not exist for SDEs at all — uses the standard error as the margin, so a ranking difference
  smaller than the sampling noise is not reported as the wrong plan winning. When authoring an SDE,
  check that a *correct* scheme's error is resolved at the chosen `num_paths`, or the problem cannot
  distinguish plans however good its truth is.

## 5. Second wave — PDE

Not built. Each is one authored ground truth away.

- **2-D Navier-Stokes double shear layer** (R + I), Re ~ 1e4, Brown & Minion. The matched pair to
  the exact Taylor-Green problem. Under-resolution grows a spurious extra vortex, which is smooth,
  plausible, and wrong -- so the field reference does the work no structural check can. Reuses the
  multi-field contract and the existing `_diag_divergence_free` checker; gate `div_u` here rather
  than merely reporting it, since with no closed form the structure carries more weight.
- **Helmholtz scattering off a non-circular obstacle** (R). A boundary-integral Nyström solver with
  Kress-graded quadrature reaches ~1e-12 and is a genuinely different method family from anything
  the pipeline will write — the strongest form of independence short of §2.7. The circle case has a
  Mie series (S) to validate it against. Needs a radiation boundary condition on the truncated
  rectangle, which is the real work.
- **Lid-driven cavity, steady NS at a de-memorized Re** (F + R). Corner singularity plus a
  nonlinear steady solve. Botella–Peyret validate the method at Re = 1000; grade elsewhere.
- **Natural convection in a differentially heated cavity** (F + R). Nusselt number; de Vahl Davis
  and Le Quéré validate the method at Ra = 1e5.
- ~~KdV from a non-soliton initial condition~~ — **built** as `pde_kdv_zabusky_kruskal` (§4).
- ~~Non-integrable NLS as a two-field system~~ — **built** as `pde_gross_pitaevskii_1d` (§4), with
  a harmonic trap rather than a quintic term; the trap is what makes the problem non-periodic.
- **Semilinear blow-up**, `u_t = u_xx + u²` (F). The truth is the blow-up time. Genuinely hard to
  author to the precision a benchmark needs — listed for completeness, not recommended next.

## 6. Second wave — SDE (the "no closed-form moments" half)

Mostly Route S, which is cheap, and it complements the existing 22 SDEs without needing new tiers.

- ~~Double-well Langevin, stationary moments~~ — **built** as `sde_double_well_langevin` (§4), and
  not at stationarity: the backward Kolmogorov equation gives the *transient* moments at any `T` to
  machine precision, and the stationary quadrature is what validates that solve. The same
  `_kolmogorov_1d` route turns any scalar diffusion with polynomial drift and additive noise into a
  no-closed-form problem with exact truth; `sde_quintic_drift_noise` and `sde_fene_blowup`
  (currently `stability` kind) could be graded on moments the same way, but they sit in the main
  suite and changing their kind would move them out of the 49-problem denominators.
- **Exit time / exit probability** (S). `E[τ]` for a 1-D diffusion solves an ODE boundary-value
  problem whose solution is a double quadrature — exact truth, and it exercises the
  `path_integrals` return contract the Dynkin check already uses.
- **Feynman–Kac cross-check** (§2.7). `E[exp(-∫V(X_s)ds)]` against a spectral eigenvalue
  computation of the corresponding Schrödinger operator: two unrelated formulations of one number.
- **Two-scheme reference moments** (R) for the Tier-C superlinear catalogue already logged in
  `hard_sde_candidates_from_lit.md` — FitzHugh–Nagumo mean-field, the 3-species chemical Langevin
  equation, stochastic Lotka–Volterra, Duffing–van der Pol. A tamed-Milstein and a split-step scheme
  at tiny `dt` under common random numbers, with the MC standard error **and** a discretization-bias
  budget in the tolerance. This is the SDE analogue of Route R and needs the same widening the PDE
  side got.
- **Rough-volatility weak order** and the **Lévy-area method audit** stay where they are — Tier D in
  the SDE lit file. Their ground truth is a convergence *rate* or a method choice, not a value, and
  they need a verifier type that does not exist yet.

---

## 7. Authoring checklist

For every new no-closed-form problem, in order:

1. **Pick the route** (§1) before writing the entry. If none applies, the problem is not ready.
2. **Author the truth in `problems.py`** — never in the pipeline's reach. Reference artifacts go in
   `benchmark/references/{slug}.npz`, which `pipeline-settings.json` already denies.
3. **Two method families, and record the agreement** as `reference_error`. A candidate that cannot
   clear the bar at the chosen horizon does not ship — that judgment belongs in
   `validate_ground_truth.py`.
4. **Write `problem.md` leakage-free.** No formula, no reference values, no benchmark file paths.
5. **Check the grids nest.** Periodic axes use band-limited Fourier interpolation; non-periodic ones
   need a reference grid whose interval count is a multiple of `lcm(N-1, 2N-1)`, so that the
   verifier's `N` and `2N` nodes are both exact subsets (§4, second-wave notes).
6. **Prove both directions in `validate_ground_truth.py`**: a correct scheme clears both gates at
   the benchmark's resolutions, and a seeded defect fails by the intended gate.
7. **Set `closed_form: False`** so the problem joins `no_closed_form_problems()` and the suite runs
   on its own, leaving the 50-problem baseline denominators untouched.
