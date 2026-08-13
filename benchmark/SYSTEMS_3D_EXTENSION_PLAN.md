# Systems / 3D Extension — Full Plan

How to extend the AutoNumerics benchmark harness (and pipeline) from **single scalar
fields on 1-D/2-D rectangular grids** to **coupled multi-field systems** and **3-D**
problems — the class that HardNumerics-LowDim is mostly made of (Navier–Stokes,
Stokes, Maxwell, MHD, shallow water, elasticity, 2-species reaction–diffusion) and
that AutoNumerics has always aimed to solve.

This is a design + phasing plan, not yet implemented. It assumes the PDE contract is
already `solve_pde(N)` with the 2-grid convergence-order check (done).

---

## 1. Why

The current PDE side maxes out at a scalar field in 2-D. HardNumerics-LowDim is 1D×3,
2D×18, 3D×4, and most of the interesting 2-D problems are **systems** with
structure-preservation requirements (divergence-free constraints, inf-sup stability,
well-balancedness, positivity). Supporting them is what lets the benchmark — and the
pipeline — demonstrate competence on genuinely hard, realistic PDEs.

The extension is **two orthogonal axes**, deliberately separable:

- **Multi-field (systems):** the unknown is several coupled fields (e.g. velocity
  `u, v` + pressure `p`; electric/magnetic `E, B`; two chemical species) rather than one
  scalar `u`.
- **Dimensionality:** more spatial axes. Built **general-`d`** (not a hard-coded 3-D
  case), with 3-D as the first target — see §3.5. This covers the whole low-dim
  executable track up to the `N**d` grid wall; true high-dimensional PDEs (`d ≫ 3`) are a
  separate, mesh-free track discussed in §3.6.

A problem may need one axis (fichera_3d = 3-D scalar; keller_segel_2d = 2-D system) or
both (maxwell_3d, stokes_3d). Building them as separate phases keeps each change small
and testable.

## 2. What the current harness assumes (the blockers)

The PDE path is built around **one scalar field on a 1-D/2-D rectangular grid**:

1. **`verify.py`** — `verify_pde` / `_pde_eval_run` / `_extract_coords` handle
   `dims ∈ {1, 2}` only. The exact solution is called as `analytic(t, x)` or
   `analytic(t, X, Y)`; `numerical_solution` is reshaped to one array; `rel_l2` and the
   order check operate on that single array.
2. **The contract** (`setup.py` footer, `solver-pde.md`) — `numerical_solution` is one
   array of shape `(N,)` / `(N, N)`; `grid` is `{x}` / `{x, y}`.
3. **`runner.py`** — `_extract_pde` coerces exactly `numerical_solution` (one array) +
   `grid` axes + `t_final`.
4. **`problems.py`** — `analytic` is a single callable returning one array; the only PDE
   metric is relative L2 (+ the derived order).
5. **Verdicts** — `PASS/FAIL` (+ `NO_GT`) assume a single accuracy number.

Each of the three items 1–4 must generalize; the changes are additive and backward
compatible (scalar 2-D problems keep working unchanged).

## 3. Design

### 3.1 Multi-field return schema

**Decision needed (Q1).** Two clean options:

- **A — `fields` dict (recommended):** `solve_pde(N)` returns
  `{"fields": {"u": arr, "v": arr, "p": arr}, "grid": {...}, "t_final": ...}`.
  Scalar problems keep returning `numerical_solution` (a single array); the harness
  treats a scalar return as `fields = {"u": numerical_solution}`. Cleanest: one obvious
  place for multiple fields, scalar path untouched.
- **B — `numerical_solution` becomes a dict** when multi-field. Less explicit; more
  overloading of one key. Not recommended.

Field **names are fixed by the problem** (declared in `problems.py`, echoed in the
contract footer so the solver knows exactly which keys to return, and in which
component order for vector fields).

### 3.2 Ground truth per field

`problems.py` grows a per-field exact solution: `analytic` returns a dict
`{field_name: array}` given `(t, *coords)`, authored and cross-checked here as today.
For reference-kind systems (e.g. a semi-closed formula), the same `(value, se)`
convention extends per field.

### 3.3 Metrics and structure diagnostics

Two tiers of check, both independent (authored in `problems.py`, never the solver's
self-report):

- **Per-field accuracy:** relative L2 per field (`velocity_L2`, `pressure_L2`, …), and a
  `global_L2` (e.g. RMS across fields). The order check runs on a designated primary
  field (or the global norm). Pass requires each required field within tolerance **and**
  the order floor, exactly as scalar today but vectorized.
- **Structure diagnostics (the hard part — what a naive scheme gets wrong):** each is a
  small independent checker, declared per problem with a name + a pass predicate:
  - **divergence constraint** — `div_u_norm` (incompressible NS/Stokes), `div_B_norm`
    (MHD/Maxwell): must stay ≈ 0.
  - **well-balancedness** — `lake_at_rest` residual (shallow water): a steady state must
    not spawn spurious velocity.
  - **volumetric locking** — a locking indicator for nearly-incompressible elasticity.
  - **positivity / max-principle** — `min(rho) ≥ 0`, `max_principle_violation`
    (Keller–Segel, Gray–Scott).
  - **conservation** — mass/energy drift for reaction–diffusion / phase-field.

  These map directly onto HardNumerics' `required_outputs`. Each problem picks the
  subset that defines its failure mode.

### 3.4 Verdict vocabulary

Add one verdict analogous to the SDE `BLOWUP`: **`CONSTRAINT_VIOLATION`** — the solver
was accurate but broke a required structural constraint (div ≠ 0, negative density,
locked). It counts as a failure but is reported distinctly, because "accurate yet
structurally wrong" is exactly the finding these problems exist to surface. `PASS`
requires accuracy **and** order **and** all declared hard constraints.

**Decision needed (Q2):** which diagnostics are *hard pass/fail gates* vs.
*reported-only*? Recommendation: the defining constraint of each problem is a hard gate
(div-free for MHD/Stokes, positivity for Keller–Segel, well-balancedness for shallow
water); secondary diagnostics are reported.

### 3.5 Dimensionality — write it for general `d` (3-D is the first target)

Rather than hard-code a `dims == 3` branch, generalize the spatial handling to an
**arbitrary number of axes**. This is barely more code than a 3-D special case, it
removes the existing 1-D/2-D branching, and it future-proofs the grid regime to 4-D
without another rewrite:

- `problems.py` `analytic` is already `analytic(t, *coords)`, so it is dimension-agnostic
  already. Keep it that way.
- `verify.py`: replace the 1-D/2-D (and would-be 3-D) branches with one path that reads
  an **ordered list of `d` axis arrays** from `grid`, builds `coords = np.meshgrid(*axes,
  indexing="ij")`, calls `analytic(t, *coords)`, and reshapes `numerical_solution` to
  `(N,)*d`. `rel_l2` and the order check are already shape-agnostic (they flatten), so
  they work for any `d` untouched.
- `runner.py` / `setup.py`: `grid` is an ordered dict of `d` named axes (`x, y, z, …` or
  problem-declared names); the contract footer lists them and the `(N,)*d` shape.

So the scalar **grid-based** path becomes genuinely `d`-generic, and 3-D is just its
first instance. **Cost:** a full tensor grid has `N**d` points and `2N` refinement costs
`2**d`× — so `grid_N` defaults must shrink with `d` (≈ 24→48 at `d=3`), and the sandbox
mem cap / timeout genuinely bind. See §5.

### 3.6 Beyond `d ≈ 3–4`: the `N**d` wall and a separate high-dim track

Making the harness `d`-generic (§3.5) does **not** make dense grid methods *work* past
~3–4 dimensions — that is a computational wall, not a coding one:

- A tensor grid is `N**d` points: at `N=64`, `d=3` is 2.6e5 (fine), `d=4` is 1.7e7
  (heavy), `d=5` is 1.1e9 (infeasible under any sane budget). Dense FD/FEM/spectral
  solving, a dense `numerical_solution` array, and pointwise-L2-on-a-tensor-grid scoring
  all break down.
- Genuinely high-dimensional PDEs (`d ≫ 3`) are solved by **mesh-free** methods —
  Feynman–Kac / Monte-Carlo, sparse grids, tensor-train decompositions, deep-BSDE /
  physics-informed neural nets — none of which produce a dense field array.

Supporting `d ≫ 3` therefore needs a **different contract**, not a bigger grid — most
likely a **new high-dim track** (mirroring how HardNumerics itself split a "high-D
reasoning track" out from this low-D executable one):

- the solver returns a **queryable solution** — a `u(points)` callable / evaluator (or a
  sampler for a distribution / a specific functional), not an `(N,)*d` array;
- scoring shifts from "L2 on a full tensor grid" to **error at Monte-Carlo sample
  points**, or a **target quantity of interest** (a functional/expectation), against a
  reference (analytic, or a high-accuracy MC / semi-analytic value);
- the "order check" is replaced by a sample-count / accuracy-vs-budget curve.

**Recommendation:** do the `d`-generic grid harness now (§3.5) — it is cheap and covers
everything up to the `N**d` wall (i.e. the entire low-dim executable track, incl. any 4-D
that appears). Treat the query-based high-dim track as a **separate future effort** with
its own contract, scoped only if/when high-D problems are actually in hand (the shared
HardNumerics-LowDim set has none — it is explicitly the low-dimensional track).

## 4. Where the changes land (by file)

| File | Multi-field | Dimensionality (general `d`) |
|---|---|---|
| `problems.py` | `analytic` returns per-field dict; declare field names + structure-diagnostic checkers | `analytic(t, *coords)` (already `d`-agnostic); `grid_N` shrinks with `d` |
| `verify.py` | per-field L2 + diagnostics; `CONSTRAINT_VIOLATION`; order on primary field | one `d`-generic coords/eval path (`meshgrid(*axes)`) replacing the 1-D/2-D branches |
| `runner.py` | `_extract_pde` handles a `fields` dict | ordered list of `d` grid axes |
| `setup.py` | contract footer lists field names + which diagnostics gate | `(N,)*d` schema, `grid` = `d` named axes |
| `report.py` | render per-field metrics + `CONSTRAINT_VIOLATION`; oneshot verdict | — |
| pipeline: `solver-pde.md`, `evaluator-pde.md`, `formulator.md`, `plan-creator-pde.md` | multi-field I/O + diagnostic scoring | higher-`d` guidance, cost-aware N |
| `pde_manual.md` | multi-field template + a div-free / saddle-point section | higher-`d` stencil notes |

Everything is additive: a scalar 2-D problem exercises none of the new branches.

## 5. Resource budget (Q3)

HardNumerics scores under a hard **60 s / 2 GB / 1 thread** budget on a fixed machine,
and its `cpu_scaling_contract` even checks wall-time growth as `N` doubles. Our harness
currently reports wall time but does not gate on it. For 3-D and heavy systems this
matters (a dense 3-D solve at `2N` can blow memory/time). Options:
- **Report-only** (current behavior) — simplest; the sandbox timeout + mem cap still
  prevent runaways.
- **Soft gate** — record wall/peak-memory and flag over-budget runs without failing them.
- **Hard gate** — adopt a per-problem budget as a pass condition (closest to
  HardNumerics; most work).
Recommendation: start report-only + the existing sandbox caps; revisit a soft gate if
3-D runs prove unstable.

## 6. Mapping HardNumerics-LowDim onto the phases

- **Phase 1 — higher-`d` scalar** *(smallest)*: `fichera_3d_singularity_mms`,
  `acoustic_3d_layered_mms`. Single field, existing metrics; implement the spatial
  handling **general-`d`** (§3.5), with 3-D as the first instance.
- **Phase 2 — 2-D systems** *(the main effort)*: `navier_stokes_2d_taylor_green`,
  `euler_2d_isentropic_vortex`, `shallow_water_2d_lake_at_rest`,
  `elasticity_2d_nearly_incompressible`, `keller_segel_2d_positive`,
  `gray_scott_2d_stiff`, `mhd_2d_manufactured_divergence_free`. Multi-field contract +
  per-field metrics + structure diagnostics.
- **Phase 3 — 3-D systems** *(largest)*: `maxwell_3d_periodic_plane_wave`,
  `stokes_3d_manufactured_divfree_pressure`. Both axes at once, with structure
  preservation in 3-D.

(The scalar 2-D / 1-D HardNumerics problems — reentrant corner, Monge–Ampère, Cahn–
Hilliard, Swift–Hohenberg, Allen–Cahn, porous-medium, eikonal, Stefan, viscous Burgers,
Heston, obstacle — need **none** of this; they drop into the current harness and are the
right first imports, independent of this extension.)

## 7. Recommended phasing & effort

1. **Phase 1 (higher-`d` scalar) — small.** Replace the 1-D/2-D branches in `verify.py`
   with one `d`-generic path (`meshgrid(*axes)`, `analytic(t, *coords)`) + a schema line.
   High signal (a 3-D corner singularity), low risk, no contract shape change, and it
   future-proofs the grid regime to 4-D. Do first.
2. **Phase 2 (2-D systems) — medium/large.** The multi-field contract + per-field
   scoring + a reusable structure-diagnostic mechanism is the bulk of the work; then each
   problem adds its exact fields + its one defining diagnostic. This is where the
   "complex problems" goal really lands.
3. **Phase 3 (3-D systems) — large.** Only after 1 and 2 are proven; combines both plus
   3-D structure preservation.

Validate each phase the way the scalar order check was validated: a hand-written correct
reference solver must PASS (accuracy + order + constraints), and a deliberately wrong one
(e.g. non-div-free) must trip `CONSTRAINT_VIOLATION`.

## 8. Open design questions (decide at kickoff)

1. **Multi-field schema** — the `fields` dict (recommended) vs. overloading
   `numerical_solution`.
2. **Diagnostics as hard gates vs. reported-only** — recommendation: each problem's
   defining constraint is a hard gate; the rest are reported.
3. **Resource budget** — report-only (recommended to start) vs. soft/hard gate.
4. **`grid_N` defaults per dimension** — shrink with `d` given the `2**d`× cost of `2N`
   (start ≈ 24→48 at `d=3`).
5. **Pipeline scope** — do we extend the pipeline's PDE agents for multi-field in the
   same pass, or first support systems in the harness/baseline only and extend the
   pipeline agents once the metrics stabilize? (Recommendation: harness + baseline first,
   pipeline agents once the contract is settled — the same order we did the scalar work.)

## 9. Concrete first step (when greenlit)

Implement **Phase 1** end-to-end on `fichera_3d` (or a synthetic 3-D heat MMS): make the
`verify.py` + `setup.py` spatial path `d`-generic (§3.5), author the 3-D `analytic`, and
smoke-test with a hand-written 3-D solver (correct → PASS with order ≈ scheme order;
under-resolved → FAIL)
before touching the multi-field machinery. That proves the 3-D axis in isolation and
de-risks Phase 2.
