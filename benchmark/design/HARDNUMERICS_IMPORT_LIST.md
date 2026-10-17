# HardNumerics Import List — the 13 problems (benchmark 37 → 50)

The 13 PDE problems to fold in from `HardNumerics-LowDim`
(`~/HardNumerics-LowDim/templates/lowdim25_prompt_ready_solve_specs_list.json`), one
instance per template (not the 360 resolution-sweep instances). All are hard tier-3 by
our scheme (beyond the reference manuals). Two batches by harness readiness:

- **Batch 1 (6) — scalar, ~current harness.** Drop into the existing PDE path (with two
  small additions noted below). Can be added *before* the systems/3D extension.
- **Batch 2 (7) — systems / higher-`d`.** Exercise the systems/3D extension across all
  three phases; add *after* that extension is built (see
  `SYSTEMS_3D_EXTENSION_PLAN.md`).

Every import is authored **leakage-free** the way the existing PDEs now are: the exact /
manufactured solution stays hidden in `problems.py`; MMS problems ship the explicit
**source term** in `problem.md`; scoring is the `solve_pde(N)` 2-grid order check
(per-field for systems), with structural diagnostics added for the systems.

## The 13 at a glance

| # | Problem (HardNumerics id) | Dim | Axis | GT kind | Fields | Batch / phase | What makes it hard |
|---|---|---|---|---|---|---|---|
| 1 | `heston_2d_european_call_semiclosed` | 2D | C1 | reference (quadrature) | scalar | 1 | mixed 2nd derivative + non-smooth payoff kink; reference is a semi-closed characteristic-function integral |
| 2 | `cahn_hilliard_2d_mms` | 2D | C1 | MMS | scalar | 1 | 4th-order operator, stiff |
| 3 | `monge_ampere_2d_convex_exact` | 2D | B4 | exact | scalar | 1 | fully nonlinear; needs a convexity-preserving solve |
| 4 | `stefan_1d_similarity_front` | 1D | A2 | exact (similarity) | scalar | 1 | moving free boundary (phase change) |
| 5 | `porous_medium_2d_barenblatt_exact` | 2D | C1 | exact | scalar | 1 | degenerate diffusion, compact support; L1 metric |
| 6 | `poisson_2d_reentrant_corner_exact` | 2D | A1 | exact | scalar | 1 | L-shaped domain, `u=r^{2/3}sin(2θ/3)` ∉ H²; kills uniform low-order rates |
| 7 | `fichera_3d_singularity_mms` | 3D | A1 | MMS | scalar | 2 / **P1** | 3-D corner singularity (Fichera cube-minus-octant) |
| 8 | `acoustic_3d_layered_mms` | 3D | C2 | MMS | scalar | 2 / **P1** | 3-D wave in layered (discontinuous-`c`) media; phase accuracy at 8π modes |
| 9 | `navier_stokes_2d_taylor_green_exact` | 2D | B3 | exact | u,v,p | 2 / **P2** | incompressible; divergence-free constraint |
| 10 | `mhd_2d_manufactured_divergence_free` | 2D | B3 | MMS | u,v,B | 2 / **P2** | resistive MHD; **div B = 0** structure preservation |
| 11 | `keller_segel_2d_positive_mms` | 2D | D1 | MMS | ρ,c | 2 / **P2** | chemotaxis; positivity + near-blow-up (min_order 1.4) |
| 12 | `elasticity_2d_nearly_incompressible_mms` | 2D | B2 | MMS | u,v | 2 / **P2** | volumetric locking / inf-sup (saddle point) |
| 13 | `maxwell_3d_periodic_plane_wave` | 3D | B3 | exact | E,H (3-vec) | 2 / **P3** | 3-D EM; **div E = div H = 0**, phase accuracy over propagation |

Axis legend (HardNumerics): A1 geometric singularity · A2 moving front · B2 saddle-point ·
B3 structure preservation · B4 fully-nonlinear/viscosity · C1 stiff/high-order ·
C2 wave propagation · D1 stiff-reaction/positivity · E1 heterogeneity.

---

## Batch 1 — scalar (add first)

The six I recommended earlier. Pure `solve_pde(N)` scalar problems on the current 2-D/1-D
path, each authored leakage-free with an authored, cross-checked ground truth.

- **`heston_2d_european_call_semiclosed`** — 2-D option price `V(S,v)`. Reference =
  Heston's semi-closed characteristic-function formula (deterministic Gauss quadrature to
  tolerance) → our **`reference`** kind. A genuine 2-D upgrade of the existing 1-D
  `black_scholes`. *No harness change.*
- **`cahn_hilliard_2d_mms`** — `u_t = Δμ + s`, `μ = -ε²Δu + u³ - u`. MMS: hand the solver
  the explicit source `s`, hide `u*`. *No harness change.*
- **`monge_ampere_2d_convex_exact`** — `det(D²u) = f`, convexity constraint. Exact `u`
  hidden. Fully nonlinear (solver challenge). *No harness change.*
- **`stefan_1d_similarity_front`** — 1-D phase change with a moving front; similarity
  (erf-based) exact temperature field hidden. *No harness change.*
- **`porous_medium_2d_barenblatt_exact`** — degenerate `u_t = Δ(uᵐ)`, compact-support
  Barenblatt profile. Metric is **L1** → needs a small `verify.py` L1-error option
  (`pde_metric="l1"`).
- **`poisson_2d_reentrant_corner_exact`** — Laplace on the **L-shaped** domain
  `(-1,1)² \ [0,1]×[-1,0]`, singular `u=r^{2/3}sin(2θ/3)`. Needs a **masked-domain**
  option (rectangular grid + an in-domain mask; error/order computed over masked nodes).
  Its `min_order` should be its published floor (~1.25) so a uniform low-order scheme
  correctly fails and a graded/adaptive mesh passes — this problem is the poster child
  for the order check.

**Batch-1 harness prerequisites (both small):** (a) an **L1 metric** option; (b)
**masked (non-rectangular) domain** support. Four of the six need neither.

---

## Batch 2 — systems / higher-`d` (add after the extension)

Chosen to exercise the systems/3D extension across **all three phases** (two Phase-1
scalar 3-D problems, four Phase-2 2-D systems, one Phase-3 3-D system) and to stress its
**structure-diagnostic machinery** with four *distinct* 2-D checkers (div-u, div-B,
positivity, locking) plus Maxwell's 3-D divergence-free constraint.

**Phase 1 — higher-`d` scalar** (two problems — validate the general-`d` grid path on
both a singularity and a wave, and give 3-D more weight in the final set):
- **`fichera_3d_singularity_mms`** — general-`d` path (§3.5 of the extension plan) on a
  real 3-D corner singularity. Scalar, MMS. Needs the **masked-domain** option
  (Fichera = cube minus an octant) and a small `grid_N` (3-D cost). Diagnostics:
  `energy_error`, `localized_singularity_error`.
- **`acoustic_3d_layered_mms`** — 3-D acoustic wave `p_tt = c(x,y,z)²Δp + s` with a
  **layered, discontinuous** wave speed `c = 1 + 0.5·1_{z>0.5}`. Scalar, MMS (source `s`
  given, `p_exact` hidden). Adds the **C2 (wave-propagation)** and **E1 (heterogeneity)**
  axes and a second Phase-1 validation. Box domain (no masking); small `grid_N`.
  Diagnostics: `pressure_L2_error`, `phase_error`, `energy_error`, `points_per_wavelength`.

**Phase 2 — 2-D systems** (multi-field contract + per-field L2 + one distinct structural
diagnostic each — the diversity is the point):
- **`navier_stokes_2d_taylor_green_exact`** — incompressible NS, fields `u,v,p`. Hard
  gate: **`divergence_norm` ≈ 0**. Also `pressure_L2_error_mean_removed`,
  `energy_decay_error`. The canonical exact NS test.
- **`mhd_2d_manufactured_divergence_free`** — resistive MHD, fields `u,v,B`. Hard gate:
  **`div_B_norm` ≈ 0** (a *different* constraint than div-u). Also `div_u_norm`,
  `magnetic_energy_error`.
- **`keller_segel_2d_positive_mms`** — chemotaxis, fields `ρ,c`. Hard gate: **positivity**
  (`min_rho ≥ 0`, `min_c ≥ 0`) near blow-up; `mass_source_balance_error`. Published
  `min_order` = **1.4**.
- **`elasticity_2d_nearly_incompressible_mms`** — near-incompressible elasticity, fields
  `u,v`. Hard gate: **`locking_indicator`** (volumetric locking / inf-sup); also
  `strain_energy_error`, `div_u_error`.

**Phase 3 — 3-D system** (both axes at once):
- **`maxwell_3d_periodic_plane_wave`** — Maxwell, fields `E,H` (3-vectors) in 3-D. Hard
  gate: **`divE_norm = divH_norm ≈ 0`** (structure preservation in 3-D); also
  `phase_error` over propagation. The single problem that proves the full Phase-3 stack.

**Distinct structural diagnostics stress-tested:** divergence-free velocity (NS),
divergence-free `B` (MHD), positivity (Keller–Segel), locking (elasticity),
divergence-free `E`/`H` in 3-D (Maxwell). Each needs its own independent checker in
`problems.py` — exactly the machinery the extension introduces.

---

## Sequencing

1. **Now / independent of the extension:** add the small Batch-1 harness bits (L1 metric,
   masked domain), then import the **6 scalar** problems (authored GT + leakage-free
   `problem.md`, validated in `validate_ground_truth.py`).
2. **After the systems/3D extension is built:** import the **7** in phase order —
   `fichera_3d` + `acoustic_3d` (P1) → the four 2-D systems (P2) → `maxwell_3d` (P3). Each
   new structural diagnostic gets a validation like the order check: a correct reference
   solver PASSes, a deliberately-broken one (e.g. non-div-free) trips
   `CONSTRAINT_VIOLATION`.

## Coverage of the final set

- **Dimensions:** 1-D ×1, 2-D ×9, 3-D ×3 — three 3-D problems (a scalar singularity, a
  scalar wave, and a system) give the general-`d` path and Phase 3 solid coverage.
- **Failure-mode axes added:** A1 (2-D + 3-D singularity), A2 (free boundary), B2
  (locking), B3 (three distinct structure-preservation constraints: div-u, div-B, div-E/H),
  B4 (nonlinear), C1 (4th-order / degenerate / finance), C2 (3-D wave propagation),
  D1 (positivity), E1 (layered heterogeneity). Complements the existing benchmark's shock
  (inviscid Burgers), fractional, chaos (KS), and boundary-layer PDEs.
- **Domains:** finance, phase-field, mechanics, aerospace/CFD, plasma, biology,
  electromagnetics, acoustics.

## Notes / possible adjustments

- **Composition choice:** this set carries **three 3-D problems** (`fichera_3d`,
  `acoustic_3d`, `maxwell_3d`) for strong general-`d` / Phase-3 coverage, at the cost of
  the shallow-water **well-balancedness** diagnostic — the one structural check from the
  candidate pool not represented in the final set (div-free, positivity, and locking all
  are). Easy to revisit: swap `acoustic_3d` back for `shallow_water_2d` to restore
  well-balancedness at the price of a 3-D problem.
- **Deliberately excluded** and why: `shallow_water_2d_lake_at_rest` (swapped out for
  `acoustic_3d` to gain a 3rd 3-D problem + the C2/E1 axes; its well-balancedness is the
  only diagnostic left uncovered), `anisotropic_diffusion_2d` (overlaps existing P09),
  `euler_2d_isentropic_vortex` (smooth compressible — its check is accuracy, not a
  distinct structural constraint; B1/shock is already covered by the existing inviscid
  Burgers), `swift_hohenberg` / `allen_cahn` / `eikonal` / `viscous_burgers_1d` /
  `obstacle_2d` (fine problems, but the six chosen scalars already cover their axes),
  `schrodinger_1d` (complex-valued — would need a complex-field harness tweak),
  `stokes_3d` (redundant with Maxwell for proving Phase-3; its B2/saddle axis is already
  covered by 2-D elasticity).
- **Effort is real:** 13 authored + cross-checked ground truths, plus the systems/3D
  extension for the 7. Batch 1 is small; Batch 2 rides on the extension.
