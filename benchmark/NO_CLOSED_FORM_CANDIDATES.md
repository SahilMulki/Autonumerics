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
   None of these can be written in the restricted expression namespace, so the pipeline genuinely
   has no formula while the harness has machine-precision truth.
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

## 4. Pilot slate (built)

Four problems, all four routes, three matched pairs, one conversion. Every one is validated in
`validate_ground_truth.py` in both directions -- a correct scheme clears every gate at the
benchmark's own resolutions, and a deliberately broken one is caught by the gate it is meant to trip.

| Slug | Route | Relationship | What a solver has to get right | Measured |
|---|---|---|---|---|
| `pde_kuramoto_sivashinsky` | R | existing P15, `SELF_ONLY` -> graded | chaotic, stiff, fourth-order | ETDRK4 vs IMEX-SBDF3 agree to **1.2e-09**; a correct solve is 30% off at N=64 and 6.6e-05 at N=128 |
| `pde_burgers_viscous_1d` | S | pair with P11 `burgers_inviscid` (exact) | an internal layer of width `~2 nu` at Re = 100 | Cole-Hopf quadrature converged to **5.6e-16**; spectral 8.8e-03 -> 1.1e-04, conservative FD2 4.8e-02 -> 8.0e-03 |
| `pde_cahn_hilliard_2d_coarsening` | R + I | pair with P20 `cahn_hilliard_2d` (**MMS**) | coarsening dynamics, not just stability | agreement **5.9e-10**; ETDRK4 stays inside 1% even at dt = 2e-3, while the first-order stabilized IMEX is 5.8e-02 at dt = 1e-3 |
| `pde_schrodinger_eigen_2d` | S + F | pair with P19 `poisson_lshape` (masked domain) | an eigen*problem*: a number and a mode, from the right end of the spectrum | `lambda_1 = -8.442746362964` to **1e-11**; FD2 is 1.19e-03 at N=64 and 2.93e-04 at N=128 |

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
- **KdV from a non-soliton initial condition** (R + I). Zabusky–Kruskal soliton fission and a
  dispersive shock; three exact invariants (mass, momentum, energy) make the I half strong.
- **Non-integrable NLS** (R + I) as a two-field real/imaginary system, so no complex harness is
  needed. Phase error is the dominant failure and `|ψ|²` and the Hamiltonian are exactly conserved.
- **Semilinear blow-up**, `u_t = u_xx + u²` (F). The truth is the blow-up time. Genuinely hard to
  author to the precision a benchmark needs — listed for completeness, not recommended next.

## 6. Second wave — SDE (the "no closed-form moments" half)

Mostly Route S, which is cheap, and it complements the existing 22 SDEs without needing new tiers.

- **Double-well Langevin, stationary moments** (S). `dX = -V'(X)dt + σdW` with `V` a double well.
  The transient has no closed form, but the stationary moments are a 1-D quadrature of
  `exp(-2V/σ²)` to machine precision. Euler–Maruyama's invariant measure is `O(dt)` wrong, and a
  superlinear `V` makes plain EM outright transient — a sharp one-shot trap with exact truth.
  **The best value-per-unit-work candidate on the SDE side.**
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
   need `grid_N` chosen so `N` and `2N-1` are exact subsets of the reference grid.
6. **Prove both directions in `validate_ground_truth.py`**: a correct scheme clears both gates at
   the benchmark's resolutions, and a seeded defect fails by the intended gate.
7. **Set `closed_form: False`** so the problem joins `no_closed_form_problems()` and the suite runs
   on its own, leaving the 50-problem baseline denominators untouched.
