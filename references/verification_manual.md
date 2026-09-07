# Verification Manual — Scoring Without Ground Truth

This manual is the authority on **how a solution is judged when there is no closed-form answer**.
Both evaluators must read it. It defines the provenance taxonomy, the objective tests available at
each tier, and the reference implementations for each test.

---

## 1. The Ladder of Evidence

Ground truth is not binary. Each tier is objective, but each certifies something weaker than the one
above it.

| Tier | Evidence | What it certifies | Provenance tag |
|---|---|---|---|
| **A** | Closed-form solution, **validated** against the spec's own operator / IC / BC | The answer is right | `analytic` |
| **A⁻** | Closed form accepted on provenance alone — quoted verbatim, or the check could not run | The answer is right *if the formula is* | `analytic_unvalidated` |
| **A′** | Deterministic surrogate (moment ODE, Kolmogorov solve, stationary density) | The answer is right, to a controlled tolerance | `surrogate` |
| **B** | Manufactured solution, degenerate limit | The **scheme and code** are right | `manufactured` |
| **C** | Self-convergence + Richardson extrapolation | It converges, and by how much it is off | `self_convergence` |
| **D** | Invariants, residuals, symmetries, constraints | It is not wrong in specific detectable ways | (supporting) |
| **E** | Cross-plan agreement | Independent methods concur | (supporting) |

**The central caution.** Tiers C, D and E are *self-referential*: they all derive from the same
`problem_spec.json`. If the formulator misread a boundary condition, every plan converges, every
plan agrees, every invariant holds — and every plan solves the wrong problem. Only tiers A, A′ and B
break that circularity. This is why a no-ground-truth certification requires the **whole battery**,
never a single substitute metric, and why the provenance tag must be carried all the way into
`REPORT.md`.

**Never report a bare pass.** Every score carries `provenance`. A 10 earned against a closed form and
a 10 earned by self-convergence are different claims and must remain distinguishable downstream.

**Tier A is not automatic.** A closed form in `problem_spec.json` is a *claim*, and a wrong one is
the single most damaging thing that can enter the pipeline: every plan for a problem descends from
one spec, so all plans are wrong identically, cross-plan consensus confirms it, and nothing
downstream can see it. `A⁻` exists so that an unchecked closed form cannot silently outrank a
*checked* surrogate — see §2b.

---

## 2b. `verification.operator` — the residual form

**Required on every PDE spec, with or without a closed form.** It is what makes Tier A checkable;
it is also what plan-no-closed-form's residual gate consumes, so one declaration serves both.

```json
"operator": {
  "fields": ["u"],
  "terms": {"time_derivative": "dt(u)", "diffusion": "-alpha*lap(u)"},
  "source": "0"
}
```

A true solution satisfies `sum(terms) - source == 0`. For a system, one entry per equation:

```json
"operator": {
  "fields": ["u", "v"],
  "equations": {
    "u": {"terms": {"time_derivative": "dt(u)", "advection": "u*u_x + v*u_y",
                    "pressure": "p_x", "viscous": "-nu*lap_u"}, "source": "0"},
    "v": {"terms": {"time_derivative": "dt(v)", "advection": "u*v_x + v*v_y",
                    "pressure": "p_y", "viscous": "-nu*lap_v"}, "source": "0"}
  },
  "combine": "rms"
}
```

**Why terms rather than one string.** The residual is normalised by the largest term at each point,
never by an absolute scale. Helmholtz at large `k` and 100:1 anisotropic diffusion both have
individually enormous terms that cancel; an absolute tolerance rejects correct formulas on both.

**Helpers**, for every declared field: `u`, `u_t`, `u_tt`, `u_x`, `u_xx`, `u_xy`, `grad_u`, `lap_u`,
`lap_lap_u` — built from `spatial_variables`, so non-Cartesian axes come free (`["S", "v"]` gives
`u_SS`, `u_Sv`, `u_vv`). Callables: `dt(·)`, `lap(·)`, `lap2(·)`, `d_x(·)`, `d_xx(·)`.

There is deliberately **no** `adv_u`, `grad_p` or `lorentz_u`. A composite means something different
in every system that uses it; write the term out.

**Outcomes.** The check reports one of `validated`, `validated_off_singularity` (a shock, kink or
layer keeps the max above tolerance while the median passes — Tier A on the smooth complement, with
the gap recorded), `quoted`, `unavailable`, or `failed`. **`failed` nulls the claimed solution
unconditionally, including a quoted one.**

**What it cannot catch.** You write the operator and the analytic solution in one pass from one
reading. Misread the equation and both come out consistent, so the residual is ~0 and the check is
blind to it. The requirements ledger is the only place the operator is compared against `problem.md`
itself, which is why an `equation` entry must name `verification.operator` in its `spec_path`.

`"operator": null` is permitted only with an `operator_note` giving the reason — a non-local Caputo
derivative is the genuine case. A recorded gap is fine; a silent omission is not.

---

## 2. Shared: Reporting Metrics

Every evaluator writes a `<metrics>` block immediately before the `<review>` block in `SOLUTION.md`.
The conductor parses it to rank plans when scores tie.

```
<metrics>
provenance: analytic | surrogate | manufactured | self_convergence | none
estimated_rel_error: 3.10e-03      # measured vs reference, or Richardson/GCI estimate
error_is_estimate: true            # true when no true reference was available
observed_order: 1.98
order_floor: 1.50
invariants_ok: true
invariant_max_drift: 4.2e-12
constraints_ok: true
mc_se_rel: 0.0041                  # SDE only — relative standard error of the checked moment
resolved: true                     # SDE only — false means MC-inconclusive
wall_time_s: 12.4
</metrics>
```

Omit fields that do not apply. `estimated_rel_error` is always populated — it is the primary
tiebreaker, so an evaluator that cannot measure a true error must still supply the Richardson
estimate.

---

# PART I — PDEs

---

## 3. Nested Grid Ladders

Self-convergence requires differencing two *numerical* solutions, so the coarse grid nodes must be a
subset of the fine grid nodes. **`np.linspace(a, b, N)` and `np.linspace(a, b, 2N)` do not share
nodes** — differencing them requires interpolation, whose error contaminates the order estimate.

Halving `dx` gives the correct ladder, and it depends on the endpoint convention:

| Grid convention | `dx` | Refinement | Nesting |
|---|---|---|---|
| Endpoint-inclusive (Dirichlet/Neumann): `linspace(a,b,N)` | `(b−a)/(N−1)` | `N → 2N−1` | `fine[::2] == coarse` |
| Endpoint-exclusive (periodic): `linspace(a,b,N,endpoint=False)` | `(b−a)/N` | `N → 2N` | `fine[::2] == coarse` |

Do not guess which convention the solver used — detect it from the grid the solver returns.

```python
import numpy as np

def refine(N, periodic):
    """Next resolution up the ladder (exactly halves dx)."""
    return 2 * N if periodic else 2 * N - 1


def detect_periodic(axis, bounds):
    """True if the returned axis excludes its right endpoint."""
    return not np.isclose(axis[-1], bounds[1], rtol=0, atol=1e-12 * max(1.0, abs(bounds[1])))


def build_ladder(N0, periodic, levels=3):
    Ns = [N0]
    for _ in range(levels - 1):
        Ns.append(refine(Ns[-1], periodic))
    return Ns                      # e.g. [64, 127, 253]  or  [64, 128, 256]
```

### Restricting a fine solution to a coarse grid

```python
def restrict(u_fine, fine_axes, coarse_axes, atol=1e-10):
    """Sample u_fine at the coarse nodes. Returns None if the grids do not nest."""
    idx = []
    for xc, xf in zip(coarse_axes, fine_axes):
        j = np.abs(xf[None, :] - xc[:, None]).argmin(axis=1)
        if not np.allclose(xf[j], xc, rtol=0, atol=atol * max(1.0, float(np.ptp(xc)))):
            return None                       # grids do not nest
        idx.append(j)
    return u_fine[np.ix_(*idx)]
```

If `restrict` returns `None`, fall back to interpolation **of an order strictly higher than the
scheme under test** (`scipy.interpolate.RegularGridInterpolator(..., method="cubic")`), and set
`interpolated: true` in the metrics — the order estimate is then only indicative and must not be
used to certify a 10.

---

## 4. Richardson Extrapolation and the GCI

Run three grids `N0 < N1 < N2` with refinement ratio `r = 2`. All norms are RMS (`np.mean`-based), so
they are directly comparable across grids.

```python
def rms(a, mask=None):
    a = a if mask is None else a[mask]
    return float(np.sqrt(np.mean(a ** 2)))

# u0, u1, u2 = primary field at N0, N1, N2
u1_on_0 = restrict(u1, ax1, ax0)
u2_on_1 = restrict(u2, ax2, ax1)

d10 = rms(u1_on_0 - u0)          # difference on the N0 nodes
d21 = rms(u2_on_1 - u1)          # difference on the N1 nodes

p = np.log2(d10 / d21)           # observed order, from differences only
```

Richardson estimate of the exact solution, on the `N1` nodes:

```python
u_star   = u2_on_1 + (u2_on_1 - u1) / (2 ** p - 1)
e_est    = d21 / (2 ** p - 1) / (rms(u_star) + 1e-14)   # est. rel. error of the FINEST solution
gci      = 1.25 * e_est                                  # Roache's grid convergence index
```

`gci` is the conservative error bar and is what the pass test uses. Report both.

### Asymptotic-range guards

A Richardson estimate is meaningless outside the asymptotic range. Reject the estimate — do **not**
certify — unless all of these hold:

```python
shrinking   = d21 < d10 * 0.95           # differences actually decrease
p_finite    = np.isfinite(p) and p > 0
p_sane      = (p >= theoretical_order) if spectral else (p < theoretical_order + 1.0)
coarse_resolved = d10 < rms(u1)          # the coarse grid solves the same problem
not_at_roundoff = d21 > 1e-13 * (rms(u1) + 1e-30)
asymptotic  = shrinking and p_finite and p_sane and coarse_resolved and not_at_roundoff
```

**`p_sane` depends on the scheme family**, which the plan declares as `scheme_family` in its
`SOLUTION.md` frontmatter (and which the kernel infers from the free-text `scheme:` when it is
absent, recording that it guessed). The upper bound reads a large `p` as noise, and for a
finite-difference scheme it is. A **spectral** method converges exponentially, so `log2(d10/d21)` is
not an algebraic order at all and has no upper bound — measured on `pde_kuramoto_sivashinsky`, whose
three plans are all spectral: 10.47, 10.92 and 9.98 against a `theoretical_order` of 4, and at a
horizon where the ladder is fully resolved the order *rises* to 11.54 rather than falling. Applying
the algebraic guard there fails every correct spectral solver. For a spectral scheme the meaningful
test is the other direction: it must converge at least as fast as the algebraic design order.

Note that `theoretical_order` describes the **equation** — KS's biharmonic term makes it 4 — not the
scheme's rate. Conflating the two is what made the algebraic guard look right.

**`coarse_resolved` is a fifth guard, not in the original four.** A `d10` larger than the solution
itself means the coarse grid is not approximating the same function, and no ratio formed from it is a
convergence rate. Measured across every Path-B ladder in `workspace/`: the three Heston plans sit at
0.005% of the field and the three KS plans at 210–407%. Five orders of separation with nothing in
between, so this is not a tuned threshold. Without it, the spectral branch above would certify an
order of 10.47 built from a `d10` four times the size of the field.

If `not_at_roundoff` is false the scheme has converged to machine precision — treat as
`e_est = 0`, `order = inf`, `asymptotic = True`. If any other guard fails, the plan cannot exceed
score 4 (converging erratically) regardless of how small the differences look.

---

## 5. Method of Manufactured Solutions (MMS)

The strongest evidence available without a closed form. Pick a smooth `u_mms` unrelated to the real
initial/boundary data, substitute it into the operator to get a source term `f = ℒu_mms`, then solve
the *sourced* problem with the same code and check it recovers `u_mms` at the design order.

MMS verifies the **discretization**, not the problem's solution — but it is the same code that then
runs the real problem, so it eliminates the "converged to the wrong operator" failure that tiers
C–E cannot see.

The spec supplies the probe (see `verification.mms_probe`); the evaluator runs it through the
`override` hook (§7):

```python
mms = spec["verification"]["mms_probe"]
u_ex = make_callable(mms["exact"])        # (coords, t) -> array or {field: array}
src  = make_callable(mms["source"])       # (coords, t) -> array or {field: array}

errs = []
for N in Ns[:2]:
    res = solve_pde(N, override={"ic": lambda c: u_ex(c, 0.0), "source": src, "bc": u_ex})
    coords = mesh_from(res["grid"])
    errs.append(rel_err(primary_of(res), u_ex(coords, res["t_final"])))
mms_order = np.log2(errs[0] / errs[1])
mms_ok    = errs[-1] < tol and mms_order >= min_spatial_order
```

### Guarding against a faulty probe

A wrong hand-derived source term makes a correct solver fail. Before trusting the probe, verify the
source numerically: apply a **high-order** stencil for `ℒ` to `u_mms` sampled on a fine grid and
confirm it reproduces `f`.

```python
# spec supplies verification.mms_probe.operator_check: an expression in `u`, `lap`, `grad`, `t`
resid = operator_check(u_ex, coords_fine, t) - src(coords_fine, t)
probe_valid = rms(resid) < 1e-6 * (rms(src(coords_fine, t)) + 1e-14)
```

If `probe_valid` is false the probe is faulty. **Skip the MMS test and note it in the review — do not
fail the plan for it.** A broken probe is the formulator's defect, not the solver's.

---

## 6. Degenerate-Limit Checks

Cheaper than MMS and needs no source derivation. The spec names a parameter setting under which the
problem *does* have a closed form (kill the nonlinearity, freeze a variable coefficient, set the
reaction rate to zero). Run the solver with those parameters and compare against that closed form.

```python
deg = spec["verification"]["degenerate_limit"]     # {"params": {...}, "exact": "...", "tol": 1e-3}
res = solve_pde(N1, override={"params": deg["params"]})
deg_err = rel_err(primary_of(res), eval_expr(deg["exact"], coords, res["t_final"], deg["params"]))
deg_ok  = deg_err < deg.get("tol", tol)
```

This is Tier B evidence and counts alongside MMS for certification purposes.

---

## 7. The `override` Hook — Solver Contract Extension

MMS, degenerate limits, translation-invariance and temporal-isolation all need to run the *same
discretization* on a *modified problem*. One optional argument covers all four.

```python
def solve_pde(N: int, override: dict | None = None) -> dict:
```

`override` is `None` (the default — solve the real problem) or a dict with any subset of:

| Key | Type | Meaning |
|---|---|---|
| `ic` | `f(coords) -> array \| {field: array}` | Replaces the initial condition |
| `source` | `f(coords, t) -> array \| {field: array}` | Added to the RHS of the equation |
| `bc` | `f(coords, t) -> array \| {field: array}` | Dirichlet boundary data (BC *type* is unchanged) |
| `params` | `dict` | Overrides entries of `parameters` |
| `dt_factor` | `float` | Multiplies the solver's internally computed `dt` |

Every key is optional; an absent key means "use the problem's own". The discretization, mesh
strategy, stencil and time integrator must be **identical** to the unmodified run — that is the whole
point. A solver that ignores `override` cannot be certified above Tier C.

`coords` is the tuple of meshgrid arrays the solver already builds, in `indexing="ij"` order.

### Optional return keys

`override` is an **input** dict and has no return channel, so two checks that need more than
the final field take optional **return** keys instead. Both are optional; an absent key makes the
check that needs it `unavailable`, which is **skipped, never failed**.

| Key | Shape | Consumed by |
|---|---|---|
| `invariant_trace` | `{name: 1-D array, one entry per step}` | §8's trace invariants (`energy_decay`, `energy_conservation`, `mass_flux_balance`, `energy_bounded`) |
| `snapshots` | `[{"t": float, "fields": {name: array}}, ...]` — the last `K` states, ending at `t_final` | §26's D1 residual, for the time term |

```python
"snapshots": [{"t": 0.480, "fields": {"u": u_nm2}},
              {"t": 0.490, "fields": {"u": u_nm1}},
              {"t": 0.500, "fields": {"u": u_n}}]      # t_final = 0.5
```

**Why a return key and not a second solve.** D1 needs `u_t` of the *produced* field. The two routes
that do not require a contract change both fail. A second solve at `t_final = T - dt` picks a
different internal step sequence, so each run approximates its own exact solution to `O(dt^p)`
independently and the difference quotient carries `O(dt^{p-1})` — `O(1)` for a first-order scheme —
and it costs a full extra solve. The solver's own last step is what the scheme computes, so the time
term is consistent by construction and D1 degenerates to a test of the spatial operator alone.

The kernel forms the derivative by Fornberg weights over the returned `t` values, so the spacing may
be arbitrary and an adaptive final step is fine. `K >= 3` gives `O(dt^2)`; `K = 2` runs and records
its own truncation floor.

---

## 8. Invariants and Structural Diagnostics

The cheapest tests with the highest falsification value, and the only Tier-D layer that catches
"converged to the wrong equation". The spec declares which apply in `verification.invariants`; each
entry has a `gate` flag (hard fail) or not (reported only).

| Name | Applies to | Check | Computed from |
|---|---|---|---|
| `mass_conservation` | periodic/Neumann diffusion, conservation laws | `∫u dx` constant | final field + IC |
| `mass_flux_balance` | Dirichlet diffusion | `∫u(T) − ∫u(0)` equals integrated boundary flux | solver trace |
| `maximum_principle` | diffusion without source | `u(T) ∈ [min(u₀,BC), max(u₀,BC)]` | final field + IC |
| `energy_decay` | heat / dissipative | `∫u² ` non-increasing | solver trace |
| `energy_conservation` | wave, Hamiltonian | `E = ½∫(u_t² + c²|∇u|²)` constant | solver trace |
| `total_variation` | scalar conservation laws | `Σ|Δu|` non-increasing (TVD) | final field + IC |
| `positivity` | densities, chemotaxis | `min(u) ≥ −tol` | final field |
| `symmetry` | symmetric IC **and** BCs | `‖u − P u‖/‖u‖ ≈ 0` for the declared reflection `P` | final field |
| `translation_invariance` | periodic | solve with IC shifted by `k` cells, shift back, compare | `override.ic` |
| `divergence_free` | incompressible flow, solenoidal B/E | `‖∇·F‖` ≈ 0 | final fields |

Reference implementations for the ones computed from the final field:

```python
def integral(u, axes, periodic=False):
    """Quadrature over d dimensions. MUST be periodicity-aware."""
    if periodic:
        # Endpoint-exclusive grid: trapezoid half-weights the ends and drops the wrap-around
        # cell, which fakes a mass drift of ~1e-5 on a scheme that conserves mass exactly.
        # The rectangle rule is exact (spectrally accurate) on a periodic grid.
        dx = np.prod([float(ax[1] - ax[0]) for ax in axes])
        return float(u.sum() * dx)
    out = u
    for ax in axes:
        out = np.trapezoid(out, ax, axis=0)
    return float(out)

def check_mass(u_T, u_0, axes, tol):
    m0, mT = integral(u_0, axes), integral(u_T, axes)
    drift = abs(mT - m0) / (abs(m0) + 1e-14)
    return drift < tol, drift

def check_maximum_principle(u_T, u_0, bc_values, tol):
    lo = min(float(u_0.min()), *bc_values) - tol
    hi = max(float(u_0.max()), *bc_values) + tol
    over = max(0.0, float(u_T.max()) - hi, lo - float(u_T.min()))
    return over <= 0.0, over

def check_tv(u_T, u_0, tol):
    tv = lambda a: float(np.sum(np.abs(np.diff(a, axis=0))))
    growth = (tv(u_T) - tv(u_0)) / (tv(u_0) + 1e-14)
    return growth < tol, growth

def check_symmetry(u_T, axis, parity):
    P = np.flip(u_T, axis=axis)
    if parity == "odd":
        P = -P
    return rms(u_T - P) / (rms(u_T) + 1e-14)

def check_translation(solve_pde, N, ic_fn, shift_cells, axis=0):
    base    = primary_of(solve_pde(N))
    shifted = primary_of(solve_pde(N, override={
        "ic": lambda c: np.roll(ic_fn(c), shift_cells, axis=axis)}))
    return rms(np.roll(shifted, -shift_cells, axis=axis) - base) / (rms(base) + 1e-14)
```

Invariants requiring a **time trace** (`energy_decay`, `energy_conservation`, `mass_flux_balance`)
cannot be computed from the final snapshot. The solver may return them under an optional key:

```python
"invariant_trace": {"energy": np.array([...]), "mass": np.array([...])}   # one entry per step
```

If the spec declares such an invariant and the solver did not return the trace, mark it
`not_reported` — **do not fail the plan**, but it cannot then be certified above score 9.

---

## 9. Residual and Recovery Estimators

### Residual (steady-state / elliptic problems only)

Substitute the numerical solution back into the strong form using a stencil of **higher order than
the solver used**, and take the norm. For elliptic problems `‖e‖ ≤ C‖r‖`; even without a computable
`C`, the residual is a valid *ranking* signal across plans on the same grid.

This needs `u_t`, which a single snapshot does not provide, so it is **restricted to steady-state
problems**. For time-dependent problems the analogous check is temporal isolation (§10) plus the
invariants above.

```python
def laplacian_4th(u, h):                       # 4th-order 1-D/N-D Laplacian, periodic-safe interior
    out = np.zeros_like(u)
    for ax in range(u.ndim):
        um2, um1 = np.roll(u, 2, ax), np.roll(u, 1, ax)
        up1, up2 = np.roll(u, -1, ax), np.roll(u, -2, ax)
        out += (-um2 + 16*um1 - 30*u + 16*up1 - up2) / (12 * h[ax] ** 2)
    return out

interior = tuple(slice(2, -2) for _ in range(u.ndim))
r = (residual_expr(u, laplacian_4th(u, h), coords))[interior]
residual_rel = rms(r) / (rms(f[interior]) + 1e-14)
```

`residual_expr` comes from `verification.residual_operator` in the spec (e.g. `"-lap_u - f"`).

### Gradient recovery (ZZ) — any elliptic problem, no spec support needed

```python
def zz_indicator(u, axes):
    g  = np.gradient(u, *axes, edge_order=2)
    gs = [smooth(gi) for gi in g]              # 3-point averaging = recovered gradient
    num = sum(rms(gi - gsi) ** 2 for gi, gsi in zip(g, gs)) ** 0.5
    den = sum(rms(gsi) ** 2 for gsi in gs) ** 0.5 + 1e-14
    return num / den
```

Non-gating, but it **localizes** the error — report where the indicator peaks, because
"error concentrated at the reentrant corner, grade the mesh there" is far more actionable solver
feedback than a global norm.

---

## 10. Temporal-Error Isolation

Nothing else checks the plan-creator's instruction to keep the temporal error subdominant. A plan
whose spatial order looks low *because time-stepping error dominates* will otherwise send the solver
chasing the wrong fix for all five iterations.

```python
u_base = primary_of(solve_pde(N1))
u_half = primary_of(solve_pde(N1, override={"dt_factor": 0.5}))
temporal_share = rms(u_half - u_base) / (rms(u_base) + 1e-14)
temporal_ok = temporal_share < 0.1 * max(e_est, tol)     # temporal error is subdominant
```

If `temporal_ok` is false, the feedback must say so explicitly: *reduce dt, the spatial order you are
measuring is masked by time-stepping error.*

---

## 11. Cross-Plan Consensus

> **Moving to the ranking layer.** Consensus is a *cross-plan* comparison, so it belongs to how
> plans are ranked against each other rather than to how one plan is scored. It is documented here
> until [plan-blind-evaluator.md](../docs/plan-blind-evaluator.md) lands, and the kernel does not
> compute it: nothing that can never lift a score needs to run inside the scoring path.

The multi-plan architecture makes this nearly free, and it is the one check that uses information no
single plan has. Run **only** when `provenance` is `self_convergence` (with a real reference there is
nothing to gain), and only at the base resolution `N0`, and only against sibling plans whose
`SOLUTION.md` shows a completed run.

```python
import importlib.util, glob, pathlib

def load_sibling(path):
    spec = importlib.util.spec_from_file_location(f"sib_{pathlib.Path(path).parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.solve_pde

peers = {}
for p in glob.glob(str(plan_dir.parent / "*" / "solver.py")):
    if pathlib.Path(p).parent == plan_dir:
        continue
    try:
        peers[pathlib.Path(p).parent.name] = primary_of(load_sibling(p)(N0))
    except Exception as exc:
        peers[pathlib.Path(p).parent.name] = exc      # record, do not crash
```

Agreement is measured against each plan's own Richardson error bar: two plans *agree* when
`rms(u_a − u_b) / rms(u_b) < gci_a + gci_b`. Report `consensus: {n_agree, n_total, max_disagreement}`.

**Weight it below self-convergence.** All plans descend from the same `problem_spec.json`, so they
share a common-mode failure. Agreement is corroborating evidence; it is never certifying evidence,
and it can never by itself lift a score.

---

## 12. PDE Scoring Rubric

> **This table is the specification for `verifylib/kernel/score.py`, not an instruction to an
> agent.** Every input below is a boolean the kernel already computed, so applying the table is a
> lookup with no judgement in it — and handing those booleans to a language model adds no
> information while adding a drift surface. The measured shape of that drift is
> `workspace/pde_kuramoto_sivashinsky`, where three independently regenerated evaluators reported
> observed orders of 10.467, 10.925 and 0.003 and all three scored 10. Changing this rubric means
> editing `kernel/score.py` and its tests. That is a real cost, and it is the point: a rubric that
> can be reinterpreted per cycle is not a rubric.
>
> Two additions the code makes to the table below, both recorded here rather than left implicit:
> a run that is accurate at the finest grid but under-converging scores **7** (the table had no row
> for it and such a run used to fall through to 4); and §26's certification rule *caps* the result,
> so a clean self-convergence run with no circularity break tops out at **8**, not 9.

Scores below are for the **no-analytic-solution** path. When `analytic_solution` is present, the
existing analytic rubric in `project_manual.md` applies unchanged (with the metrics block added).

Define:

- `verified` — MMS **or** degenerate-limit passed at design order (Tier B evidence present)
- `converged` — `asymptotic` guards all pass **and** `gci < rel_l2_err_max`
- `order_ok` — `p >= min_spatial_order`
- `invariants_ok` — every declared invariant within tolerance; no `gate` invariant violated
- `constraints_ok` — no hard structural gate violated (`div u = 0`, positivity, ...)
- `temporal_ok` — §10

| Condition | Score |
|---|---|
| `verified` **and** `converged` **and** `order_ok` **and** `invariants_ok` **and** `constraints_ok` **and** `temporal_ok` | **10** |
| `converged` and `order_ok` and `invariants_ok` and `constraints_ok` (no Tier-B evidence available) | **9** |
| `order_ok` and `invariants_ok` but `gci >= tol` (converging, not yet accurate enough) | **7** |
| Converges, but a non-gate invariant drifts materially, or `temporal_ok` is false | **5** |
| Asymptotic guards fail — order estimate not stable across the ladder | **4** |
| A hard-gate structural constraint or gate invariant is violated | **3** |
| Ran, but differences do not shrink at all under refinement | **2** |
| Crash | **1** |

A 10 without a closed form is reachable, but only through the **full** battery — that is precisely
what makes Tier B non-optional for certification. Tag it `provenance: manufactured` so the claim
stays legible.

**Never score on plausibility alone.** "The code ran and the output looks physically reasonable" is
not a measurement. If no test in this manual could be run, the score is capped at **2** and the
review must say which tests were attempted and why each was unavailable.

---

# PART II — SDEs

---

## 13. Why SDEs Are Different

Two things separate this from the PDE case:

1. **Monte Carlo error sits on top of discretization error.** Every comparison — to a reference, to
   another plan, to a threshold — must be made against a confidence interval. Without one you cannot
   distinguish "biased" from "noisy", and with no reference that distinction is load-bearing.
2. **A real reference is more often recoverable than it looks.** Three routes (§15–§17) produce
   genuine Tier-A′ ground truth for SDEs that have no closed-form solution. Exhaust all three before
   falling back to self-convergence.

---

## 14. Monte Carlo Confidence Intervals

Required on **every** moment comparison, with or without ground truth.

```python
def moment_stats(X):
    """Point estimates and standard errors for the mean and the variance."""
    M   = X.shape[0]
    m   = float(np.mean(X))
    v   = float(np.var(X, ddof=1))
    se_m = float(np.std(X, ddof=1) / np.sqrt(M))
    mu4  = float(np.mean((X - m) ** 4))                 # 4th central moment
    se_v = float(np.sqrt(max(mu4 - v**2, 0.0) / M))     # general; do NOT assume Gaussian
    return m, v, se_m, se_v
```

Use the general `se_v`, not the Gaussian `v·√(2/(M−1))` — GBM, CIR and Exp-OU are heavy-tailed and
the Gaussian form badly understates the error there.

### The three-way outcome

A point estimate that lands inside a threshold band proves nothing if its own error bar is wider than
the band. Every check therefore returns one of **pass / fail / inconclusive**:

```python
ci_mult  = thresholds.get("ci_mult", 2.0)      # ~95%
resolved = ci_mult * se_v / (abs(v_ref) + 1e-14) < 0.5 * var_tol
rel_err  = abs(v_emp - v_ref) / (abs(v_ref) + 1e-14)

if not resolved:
    outcome = "inconclusive"      # MC noise too large to decide — raise num_paths
elif rel_err < var_tol:
    outcome = "pass"
else:
    outcome = "fail"
```

`inconclusive` is a distinct result and gets its own score. Reporting it as a pass, or as a solver
bug, are both wrong: the fix is more paths, and the feedback must say exactly that.

---

## 15. Surrogate 1 — Moment ODEs

The largest recoverable class. For **polynomial diffusions** — affine drift with `g²` affine or
quadratic in `X` — the moment equations **close exactly**, so the moments satisfy a deterministic ODE
system integrable to machine precision even though the SDE has no closed-form solution.

`dE[X]/dt = E[f(X)]`, and `dE[X²]/dt = E[2X f(X) + g(X)²]`.

Spec form (`verification.moment_ode`):

```json
{
  "state": ["m1", "m2"],
  "rhs": ["mu*m1", "2*mu*m2 + sigma**2*m2"],
  "initial": ["X_0", "X_0**2"],
  "closes_exactly": true,
  "mean_from": "m1",
  "variance_from": "m2 - m1**2"
}
```

```python
from scipy.integrate import solve_ivp

def moments_from_ode(mo, params, T):
    names = mo["state"]
    def rhs(t, y):
        env = {**params, **dict(zip(names, y)), "t": t, "np": np}
        return [float(eval(e, {"__builtins__": {}}, env)) for e in mo["rhs"]]
    y0  = [float(eval(e, {"__builtins__": {}}, {**params, "np": np})) for e in mo["initial"]]
    sol = solve_ivp(rhs, (0.0, T), y0, rtol=1e-11, atol=1e-13, dense_output=True)
    env = {**params, **dict(zip(names, sol.y[:, -1])), "np": np}
    return (float(eval(mo["mean_from"], {"__builtins__": {}}, env)),
            float(eval(mo["variance_from"], {"__builtins__": {}}, env)))
```

Only usable when `closes_exactly` is true. A moment-closure *approximation* is not ground truth — if
the formulator marks `closes_exactly: false`, treat the result as a sanity band (Tier D), not a
reference.

---

## 16. Surrogate 2 — The Kolmogorov Route

The elegant one, and the natural synthesis of the repo's two halves: the moments of an SDE solve a
deterministic PDE, so the PDE machinery produces a **noise-free** reference.

For `u(x,t) = E[φ(X_T) | X_t = x]`, the backward Kolmogorov equation is

```
u_t + f(x) u_x + ½ g(x)² u_xx = 0,        u(x,T) = φ(x)
```

Substituting `τ = T − t` turns it into a forward parabolic problem `u_τ = f u_x + ½ g² u_xx`, and the
answer is `u(X₀, τ=T)`. Take `φ(x) = x` for the mean and `φ(x) = x²` for the second moment.

```python
def kolmogorov_moment(f_expr, g_expr, params, X0, T, phi, bounds, Nx=4001, Nt=4000):
    """Backward Kolmogorov via Crank-Nicolson on a truncated domain."""
    import scipy.sparse as sp, scipy.sparse.linalg as spla
    a, b = bounds
    x  = np.linspace(a, b, Nx); dx = x[1] - x[0]; dtau = T / Nt
    ev = lambda e: np.asarray(eval(e, {"__builtins__": {}}, {**params, "X": x, "np": np}), float)
    f, g2 = ev(f_expr) * np.ones_like(x), ev(g_expr) ** 2 * np.ones_like(x)
    lo = 0.5 * g2 / dx**2 - f / (2 * dx)
    di = -g2 / dx**2
    up = 0.5 * g2 / dx**2 + f / (2 * dx)
    L  = sp.diags([lo[1:], di, up[:-1]], [-1, 0, 1], format="csc").tolil()
    L[0, :] = 0; L[-1, :] = 0                       # linear-extrapolation far field
    I  = sp.identity(Nx, format="csc")
    A  = (I - 0.5 * dtau * L).tocsc()
    B  = (I + 0.5 * dtau * L).tocsc()
    lu = spla.splu(A)
    u  = phi(x)
    for _ in range(Nt):
        u = lu.solve(B @ u)
        u[0]  = 2 * u[1] - u[2]                     # u_xx = 0 at the truncation boundary
        u[-1] = 2 * u[-2] - u[-3]
    return float(np.interp(X0, x, u))
```

### Truncation guard — mandatory

The far-field boundary is artificial, so the answer must be shown insensitive to it:

```python
m_a = kolmogorov_moment(..., bounds=B0)
m_b = kolmogorov_moment(..., bounds=widen(B0, 1.5))
trustworthy = abs(m_a - m_b) / (abs(m_a) + 1e-14) < 0.1 * tol
```

If `trustworthy` is false, **demote the provenance to `self_convergence`** and say so in the review.
An untrustworthy surrogate presented as ground truth is worse than no surrogate at all.

Tractable in 1-D, feasible in 2-D, hopeless beyond 3-D — the formulator should not propose it above
2 state dimensions.

---

## 17. Surrogate 3 — Stationary Density

For an ergodic **scalar** SDE the invariant density has a closed form essentially always, even when
the transient dynamics have none:

```
p_s(x) ∝ (1/g(x)²) · exp( 2 ∫^x f(y)/g(y)² dy )
```

```python
def stationary_moments(f_expr, g_expr, params, support, Nx=20001):
    a, b = support
    x  = np.linspace(a, b, Nx)
    ev = lambda e: np.asarray(eval(e, {"__builtins__": {}}, {**params, "X": x, "np": np}), float)
    f, g2 = ev(f_expr) * np.ones_like(x), ev(g_expr) ** 2 * np.ones_like(x)
    Phi = 2 * np.concatenate([[0.0], np.cumsum(0.5 * (f[1:]/g2[1:] + f[:-1]/g2[:-1]) * np.diff(x))])
    logp = Phi - np.log(g2)
    p = np.exp(logp - logp.max())
    Z = np.trapezoid(p, x)
    p /= Z
    return (float(np.trapezoid(x * p, x)),
            float(np.trapezoid(x**2 * p, x) - np.trapezoid(x * p, x) ** 2), x, p)
```

Applies only when the spec declares `stationary_density.ergodic: true` with a normalizable support —
so it covers the mean-reverting families (OU, CIR, Exp-OU and their nonlinear cousins) and not the
divergent ones (GBM, BM). Compare against a long-`T` run after burn-in, using both the moments and a
Kolmogorov–Smirnov statistic against the CDF of `p_s`.

```python
res = solve_sde(num_paths, dt, T=stat["T_stat"], seed=seed)
cdf = np.concatenate([[0.0], np.cumsum(0.5*(p[1:]+p[:-1]) * np.diff(x))])
ks  = float(np.max(np.abs(np.searchsorted(np.sort(res["terminal_paths"]), x) / num_paths - cdf)))
```

---

## 18. Common Random Numbers — SDE Contract Extension

Self-convergence in `dt` is only meaningful when every refinement level is driven by the **same**
Brownian path. If each level draws fresh noise, MC variance swamps the discretization bias entirely
and the order estimate is garbage.

The `seed` argument cannot deliver this: `standard_normal((M, Nt))` and `standard_normal((M, 2Nt))`
from the same seed are unrelated draws, so coarse increments are not sums of fine ones. The contract
therefore takes the increments explicitly:

```python
def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
```

| Argument | Meaning |
|---|---|
| `dW` | Pre-generated Brownian **increments** (already scaled by `√dt`), shape `(num_paths, Nt)` for scalar noise or `(num_paths, Nt, m)` for `m` noise sources. When given, the solver **must** use exactly these and take `Nt = dW.shape[1]`, `dt = T / Nt` — it must not draw its own. |
| `observables` | `{name: φ(X)}`. The solver accumulates `∫₀ᵀ φ(X_s) ds` along each path by trapezoid and returns it under `path_integrals[name]`, shape `(num_paths,)`. |

Canonical fallback when `dW is None` — draw as a single block, path-major:

```python
rng = np.random.default_rng(seed)
dW  = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))     # or (num_paths, Nt, m)
```

Building a nested CRN ladder in the evaluator:

```python
def crn_ladder(num_paths, dt0, T, levels=3, seed=42):
    """Increments for dt0, dt0/2, dt0/4 ... all driven by one Brownian path."""
    Nt0 = max(1, round(T / dt0))
    Nf  = Nt0 * 2 ** (levels - 1)
    dtf = T / Nf
    dW  = np.sqrt(dtf) * np.random.default_rng(seed).standard_normal((num_paths, Nf))
    out = []
    for k in range(levels):
        agg = 2 ** (levels - 1 - k)
        out.append((T / (Nt0 * 2 ** k), dW.reshape(num_paths, -1, agg).sum(axis=2)))
    return out                                    # [(dt, dW), ...] coarse -> fine
```

**Memory note.** The finest level dominates: `num_paths × Nt_fine × 8` bytes. With 50 000 paths and
400 steps that is 160 MB, which is fine; two more halvings is not. Because CRN removes most of the
variance from *differences*, the convergence study needs far fewer paths than the accuracy
measurement — use `num_paths_conv = min(num_paths, 20000)`.

---

## 19. Weak and Strong Convergence

With the CRN ladder in hand, both orders are measurable with no reference at all. This is also
exactly what distinguishes the Euler–Maruyama and Milstein plans, so it produces genuinely
differentiating feedback.

```python
levels = crn_ladder(num_paths_conv, dt0, T, levels=3, seed=seed)
runs   = [solve_sde(num_paths_conv, dt, T, seed=seed, dW=dWk) for dt, dWk in levels]
X      = [r["terminal_paths"] for r in runs]

# strong: pathwise, same Brownian path at each level
s0 = float(np.mean(np.abs(X[0] - X[1])))
s1 = float(np.mean(np.abs(X[1] - X[2])))
strong_order = np.log2(s0 / s1)          # ~0.5 for Euler-Maruyama, ~1.0 for Milstein

# weak: difference of moments, also variance-reduced by CRN
w0 = abs(float(np.mean(X[0])) - float(np.mean(X[1])))
w1 = abs(float(np.mean(X[1])) - float(np.mean(X[2])))
weak_order = np.log2(w0 / w1)            # ~1.0 for both EM and Milstein

# Richardson-extrapolated moment estimate (weak order p)
p = max(weak_order, 0.5)
mean_star = float(np.mean(X[2])) + (float(np.mean(X[2])) - float(np.mean(X[1]))) / (2**p - 1)
```

### The strong order is the gate; the weak order is a diagnostic

These two estimates are **not** equally reliable, and treating them as equal will fail correct
solvers.

- **Strong order is clean.** CRN makes `X^dt − X^{dt/2}` a pathwise difference, so its mean is
  measured with very little variance. Measured on GBM at 20 000 paths: **0.488** for
  Euler–Maruyama, **1.006** for Milstein. Gate on this.
- **Weak order is noisy, and can be degenerate.** It is a difference *of means*, so the MC error
  does not cancel, and for some `(scheme, φ)` pairs the leading weak-error term vanishes outright —
  the difference then sits at the noise floor and `log2(w0/w1)` returns garbage. Measured on GBM
  with Euler–Maruyama: `φ = x` gave 1.67, `φ = x²` gave 1.46, and `φ = tanh(x)` gave **4.18** purely
  because `w1` had fallen below its own standard error.

So estimate the weak order across several `φ`, discard any that is at the noise floor, and take the
median of what survives:

```python
def weak_order(Xs, phis, floor_mult=2.0):
    orders = []
    for phi in phis:
        a, b, c = (phi(X) for X in Xs)
        w0, w1 = abs(np.mean(a) - np.mean(b)), abs(np.mean(b) - np.mean(c))
        se = float(np.std(c - b, ddof=1) / np.sqrt(len(c)))     # SE of the finer difference
        if w1 < floor_mult * se or w1 <= 0 or w0 <= w1:
            continue                                            # degenerate — not informative
        orders.append(np.log2(w0 / w1))
    return (float(np.median(orders)) if orders else None), len(orders)
```

Guards, mirroring §4:

```python
strong_ok = (s1 < s0) and np.isfinite(so) and abs(so - expected_strong_order) < 0.4
w, n_ok   = weak_order(Xs, phis)
weak_ok   = (w is None) or (0.5 * expected_weak_order <= w <= expected_weak_order + 0.75)
```

`weak_ok` is **true when the estimate is unavailable** (`w is None`) — an indeterminate weak order is
not evidence of a defect. An order far *above* the theoretical one is noise, not a bonus; that is
what the noise-floor filter exists to remove.

**And so the weak order does not gate.** The heading above says the strong order is the gate and the
weak order a diagnostic, and the kernel implements exactly that: `orders_ok = strong_ok`. An
above-band weak order is the same noise the `w is None` branch already forgives, and it just did not
happen to be caught by the floor filter — measured, a correct tamed Euler-Maruyama plan on
`sde_ginzburg_landau_s6` lands at **1.7514** against an upper limit of 1.75. It is recorded as
`weak_order_out_of_band` and reported, never scored on.

**The expected strong order is a property of the scheme, so it depends on the plan.** A spec may
declare more than one — `sde_ginzburg_landau_s6` carries `expected_strong_order: 0.5` for tamed
Euler-Maruyama *and* `expected_strong_order_milstein: 1.0` — and the plan's `scheme_family`
frontmatter selects which applies. The plan chooses among the values the formulator set; it does not
set its own. Where the plan declares no family, the band widens to a one-sided bar at
`min(declared) - 0.4`: an order below the least of the declared expectations is a defect, an order
above the greatest is noise.

---

## 20. Dynkin's Identity — Reference-Free Residual

The SDE counterpart to the PDE residual check, and the reason it matters: it is available for
**every** SDE, with no closed form, no surrogate and no assumptions.

For any smooth test function `φ`, with generator `ℒφ = f φ′ + ½ g² φ″`:

```
E[φ(X_T)]  =  φ(x₀) + E[∫₀ᵀ (ℒφ)(X_s) ds]
```

```python
obs = {f"L{i}": make_generator(phi, f_expr, g_expr, params) for i, phi in enumerate(test_fns)}
res = solve_sde(num_paths, dt, T, seed=seed, observables=obs)
lhs = np.mean([phi(res["terminal_paths"]) for phi in test_fns], axis=1)
rhs = np.array([phi(x0) + np.mean(res["path_integrals"][f"L{i}"])
                for i, phi in enumerate(test_fns)])
resid = lhs - rhs
```

**The trapezoid path integral carries its own `O(dt)` bias**, so the residual does not go to zero at
fixed `dt` — the meaningful test is that it *shrinks at the expected rate*. Run at two `dt`,
Richardson-extrapolate the residual to `dt → 0`, and require the extrapolated value to sit inside the
MC confidence interval of zero:

```python
r_extrap = r_fine + (r_fine - r_coarse) / (2**1.0 - 1)        # residual is O(dt)
dynkin_ok = np.all(np.abs(r_extrap) < ci_mult * se_resid)
```

Three things this sketch leaves out, each of which was measured to matter on a **correct** solver:

1. **The two `dt` levels must share a Brownian path**, by §18's own argument. With independent
   draws, `r_fine - r_coarse` is mostly Monte Carlo noise and the extrapolation amplifies it: on a
   correct Euler–Maruyama OU solver the extrapolated residual moved from −3.9e-03 (CRN, passing) to
   −6.5e-03 (independent draws, failing) against the same 5.0e-03 bar.
2. **Extrapolate path by path, then take the standard error of the result.** `2·r_fine − r_coarse`
   has neither the variance nor the correlation structure of `r_fine`, so `se_resid` taken from the
   fine level alone is wrong in both directions at once.
3. **Correct for the number of test functions.** Requiring all `K` to sit inside a 95% interval is a
   ~86% test at `K = 3`, so one correct solver in seven fails — and the failure moves a Path-B score
   from 9 to 7. Bonferroni: `ci_mult` becomes 2.28 at `K = 2`, 2.43 at `K = 3`.

Even corrected, this check's resolution is set by the path count, because the residual's dominant
term is the sample mean of the **martingale part** `∫φ′g dW` — a property of the seed, not of the
scheme. Measured across six seeds on a correct solver at 40 000 paths, the extrapolated residual
lands between 0.2 and 3.2 standard errors of zero and one seed in six fails. Report `z` in the
metrics so a reader can see the margin, and raise `num_paths` or `ci_mult` rather than reading a
marginal excursion as a defect.

Choose test functions that probe different parts of the state space — `x`, `x²`, and one bounded
nonlinear function such as `np.exp(-x²)` or `np.tanh(x)`. A scheme that is right on polynomials and
wrong on a bounded function has a tail problem worth reporting.

---

## 21. SDE Constraints and Identities

| Check | Applies to | Test |
|---|---|---|
| `positivity` | CIR, GBM, Exp-OU, any density | fraction of paths with `X < 0` — an objective defect count |
| `support` | bounded processes | fraction outside the declared `[a,b]` |
| `martingale` | zero-drift, or a declared transform | `E[M_T] = M_0` within `ci_mult × SE` |
| `finite` | all | no NaN/Inf in `terminal_paths` |
| `estimator_stability` | all | batch-means / bootstrap across path subsets — confirms `num_paths` has settled |

```python
def estimator_stability(X, n_batches=20):
    b = np.array_split(X, n_batches)
    means = np.array([np.mean(bi) for bi in b])
    return float(np.std(means, ddof=1) / np.sqrt(n_batches) / (abs(np.mean(X)) + 1e-14))
```

Positivity violations are counted, not judged: report `neg_fraction` and the most negative value.
A bare `np.maximum(X, 0)` clip hides the defect without fixing the scheme — flag it if you see it in
`solver.py`.

---

## 22. Cross-Plan Consensus (SDE)

> Moving to the ranking layer with §11, and for the same reason.

Same rules as §11, with one addition specific to Monte Carlo: sibling plans must be run with
**independent seeds**, and agreement is tested against the *combined* standard error:

```python
agree = abs(m_a - m_b) < ci_mult * np.sqrt(se_a**2 + se_b**2)
```

Two plans agreeing on a shared seed proves nothing — they saw the same noise.

---

## 23. SDE Scoring Rubric

> Like §12, this is the specification for `kernel/score.py` rather than an instruction to an agent.

Applies when `analytic_moments.has_analytic_solution` is false. When exact moments are present, the
analytic rubric in `project_manual.md` applies unchanged — **plus** the confidence-interval logic of
§14, which is mandatory on both paths.

Define:

- `surrogate` — a §15/§16/§17 reference was obtained and passed its own trust guard
- `moments_ok` — every checked moment within tolerance of the reference
- `resolved` — MC error bars narrow enough to decide (§14); false ⟹ inconclusive
- `orders_ok` — `strong_ok and weak_ok` per §19 (the strong order gates; an indeterminate weak
  order does not fail the plan)
- `dynkin_ok` — extrapolated Dynkin residual inside the CI of zero (§20)
- `constraints_ok` — no gate constraint violated (§21)

| Condition | Score |
|---|---|
| `surrogate` and `moments_ok` and `resolved` and `constraints_ok` | **10** |
| No surrogate: `orders_ok` and `dynkin_ok` and `constraints_ok` and Richardson moments stable across the ladder | **9** |
| `orders_ok` and `constraints_ok`, but Richardson-extrapolated moments unstable, or `dynkin_ok` false | **7** |
| `resolved` is false — MC-inconclusive (raise `num_paths`; not a solver bug) | **6** |
| Runs and converges, but a gate constraint is violated (negative paths, support breach) | **4** |
| Estimator erratic in `dt` — orders far from expectation, or differences not shrinking | **3** |
| Crash or non-finite output | **1** |

**The formulator's `null` is not the end of the inquiry.** Reaching this rubric at all means §15–§17
were all attempted and all failed. If the evaluator finds that a surrogate *was* available and the
spec simply did not declare it, say so in the review — that is a formulator defect and should be
reported back, not silently absorbed.

---

## 24. Evaluation Cost Discipline

Verification is not free, and the expensive parts of it exist to substitute for ground truth. A
problem that *has* a closed form should not pay for the machinery that replaces one.

Two rules keep the cost where the value is.

### Rule 1 — the ladder is only as long as the measurement needs

| Path | Levels | Why |
|---|---|---|
| A (analytic solution present) | **2** | The order comes from `log2(e₀/e₁)` measured directly against the exact solution. A third grid adds robustness, not information. |
| B (no closed form) | **3** | Richardson differences two *numerical* solutions, so a third point is what makes the estimate exist at all. |

The third level dominates everything else in the evaluation — for an explicit scheme (`work ∝ N^(d+2)`)
it costs `2^(d+2)`× the second, so 8× in 1-D, 16× in 2-D, **32× in 3-D**. Running it on Path A turns a
9-unit evaluation into a 73-unit one and buys nothing the exact solution does not already give you.

`evaluation_thresholds.refinement_levels` overrides the default when a Path-A problem genuinely wants
three grids — an error suspected of stalling at a floor from a boundary treatment, say. Make that an
opt-in, never the default.

### Rule 2 — cheap checks every cycle, expensive verification once, at certification

The evaluator runs on every refine cycle, but most checks only change the outcome when a plan is
about to be *certified*. An expensive probe cannot turn a 6 into a 10, and a plan scoring 6 already
has actionable feedback without it.

So: compute the cheap checks, form a **provisional score**, and only then decide whether the
expensive ones can still change the answer.

```python
# --- always, every cycle (all of these reuse data already in hand) ---
#   grid ladder (2 or 3 levels), per-field error, observed order,
#   invariants, structural gates, confidence intervals (SDE)
provisional = score_from_stage_1(cheap_checks)

would_certify = provisional >= (10 if analytic is not None else 9)

# --- conditionally, and only for a reason ---
run_temporal   = (not order_ok) or would_certify   # attribute a low order, or confirm a pass
run_mms        = would_certify and analytic is None        # Tier B is what lifts 9 -> 10
run_crn_ladder = (analytic is None) or (not overall_pass) or (
                     would_certify and expected_strong_order >= 1.0)   # SDE, see below
run_dynkin     = analytic is None                          # SDE, Path B only
run_consensus  = analytic is None
```

Each gate has a distinct rationale — do not collapse them into one flag:

- **Temporal isolation** earns its cost twice: as *attribution* when the order is already low (it tells
  the solver to reduce `dt` rather than rewrite the stencil — the single most common misdiagnosis), and
  as a *confirmation* that a passing plan is not passing by accident. It is wasted when the plan fails
  on error magnitude with a healthy order, because `dt` is then not the suspect.
- **MMS / degenerate limit** is pure certification evidence. It is the only thing that lifts a Path-B
  plan from 9 to 10, and it is meaningless on Path A, where the closed form is strictly better
  evidence.
- **The CRN order study** is the measurement itself on Path B. On Path A it is a diagnostic when the
  moments miss, plus one special case: **Euler–Maruyama and Milstein have the same weak order**, so the
  moment comparison cannot tell them apart. A Milstein plan with a broken correction term will pass
  the thresholds and still be mislabelled in `REPORT.md`. The strong-order check is the only thing that
  catches it — but that is a correctness-of-*claim* issue, not correctness-of-*answer*, so run it once
  at certification rather than on every cycle.
- **Dynkin** adds nothing on Path A that exact moments do not already do better.

### Reuse the runs you already made

Temporal isolation compares against the solution at `Ns[1]`, which the ladder already computed. Cache
the ladder results and pass them in; recomputing costs a full level-1 solve for nothing.

```python
runs = run_ladder(...)                       # keep the results
u_base = primary_of(runs[1]["res"])          # NOT solve_pde(Ns[1]) again
u_half = primary_of(solve_pde(Ns[1], override={"dt_factor": 0.5}))
```

### What this costs

In units of a two-grid evaluation, for an explicit scheme:

| | Path A, passing | Path A, failing | Path B, most cycles | Path B, certifying |
|---|---|---|---|---|
| 1-D | **1.0×** | 2.8× | 8.1× | 10.9× |
| 2-D | **1.0×** | 2.9× | 16× | 19× |
| 3-D | **1.0×** | 2.9× | 32× | 35× |

Path A returns to baseline when the plan passes. Path B stays expensive because measuring an error
with no reference genuinely requires a third point — that cost is paid by the problems that actually
lack ground truth, which is where it belongs.

---

## 25. What Must Never Happen

- A missing reference reported as a solver crash (score 1). The absence of ground truth is a property
  of the problem, not a defect in the code.
- A pass claimed on plausibility. "Ran cleanly and looks physical" is not a measurement.
- A confidence interval omitted from a Monte Carlo comparison.
- A `10` whose `provenance` is not carried into `<metrics>`, `SOLUTION.md` and `REPORT.md`.
- Cross-plan agreement used to lift a score. It corroborates; it never certifies.
- **An evaluator assigning its own score.** The kernel computes `score` and `provenance`; the
  evaluator transcribes them. The one authority it keeps is asymmetric: it may cap the score
  *downward* with a machine-readable reason (`agent_cap: {score, reason}`), never raise it. An agent
  that cannot inflate cannot launder a 4 into a 10; one that can deflate can still stop a wrong 10
  from shipping.
- **A residual reported as a magnitude.** There is no calibrated passing value for a residual on
  solver output (§26). Report the rate.

---

## 26. D1 — The Operator Residual as a Slope Test

The PDE counterpart of §20's Dynkin check, and it took the same shape for the same reason.

**There is no calibrated magnitude threshold, and there cannot be.** A solver's residual is
dominated by its own truncation error, `O(h^p) + O(dt^q)`; it does not approach zero at fixed
resolution. The only calibrated number in this repo is `reference.TOL = 1e-6`, measured for *a
closed form on a 1025-point grid whose own discretization error is 1e-10* — a solver cannot do that.
Any constant threshold either fails every correct coarse-grid solver or passes everything.

So D1 tests the **rate**, exactly as §20 does for Dynkin:

```python
r(h)  = median over the interior of  |sum(terms) - source| / max_k |term_k|
p_res = log2(r(h) / r(h/2))          # both grids are already on the ladder
```

| Outcome | Condition | Effect |
|---|---|---|
| `clean` | `p_res >= theoretical_order - 1`, **or** `r` has reached the round-off floor | Counts as a circularity break |
| `slow` | `r` falls, but under-performs the design order | Reported; neither gates nor certifies |
| `stalled` | `r` does not fall and is not at round-off | **Gate** (score 3) when the operator is validated; a warning naming both suspects otherwise |
| `unresolved` | The two kernel stencils disagree | Skipped; never counts against the plan |
| `unavailable` | No `snapshots`, no operator, a non-uniform mesh, or an unsupported masked domain | Skipped |

`p_design` is `verification.theoretical_order` — **not** `evaluation_thresholds.min_spatial_order`,
which is `0.0` on `pde_kuramoto_sivashinsky` and would make every guard vacuous. It is the problem's
property, so a plan cannot lower its own bar by claiming a lower order.

**The median, never the max.** `reference.check_operator_residual` gates on the median because the
max-plus-"singular set < 2%" design was measured to be fragile (real singular cases land at
0.8/1.2/1.6%) and could not classify `pde_black_scholes_call` at all. Solver fields have shocks and
layers too. D1 inherits the rule.

**Difference at higher order, and in a different family.** Rule 1 (manual §9, extended to the
time-dependent case): use a stencil of higher order than the solver did. Rule 2: use a different
discretization *family* — if the solver was spectral, difference with high-order FD and vice versa.
Reusing the solver's own stencil makes agreement a tautology; the paper-release pipeline's
`5.03e-15` viscous-Burgers artifact is what that looks like. Both rules read `scheme_family` and
`spatial_order` from the plan's `SOLUTION.md` frontmatter.

**The stencil-insensitivity guard.** Compute `r` twice with two stencil families (FD8 and FD6, or
FD8 and spectral on a periodic grid). Disagreement by more than ~2x means the residual is measuring
the *kernel*, not the solver: report `unresolved` and skip. This is the escalation idiom
`reference.PROBE_N = (257, 513, 1025)` already uses to separate "under-resolved" from "wrong".

**Validating the operator itself.** D1's residual and the solver both descend from
`verification.operator`; if it is wrong, both are wrong consistently and D1 is not an independent
check at all. Three routes, descending in strength: the reference check passing on Path A; applying
the operator to `mms_probe.exact` and requiring it to reproduce `mms_probe.source`; applying it with
the degenerate parameters to `degenerate_limit.exact` and requiring the residual to vanish. **D1
gates only when one of them has run.** None of them catches an operator misread the same way
`problem.md` was — that is the requirements ledger's job, and it is the only check that reaches
outside the spec.

**The three trivial-attractor guards.** A residual rewards any field sitting on a stable steady state
of the operator, whether or not it is the state the initial condition evolves to. `u = 1, v = 0` is
an exact homogeneous Gray-Scott solution, so a solver that over-diffuses and destroys the pattern
satisfies every term to near machine precision, having eliminated the only phenomenon the problem is
about. Worse, if `u -> 0` the numerator and denominator both vanish and the epsilon leaves you
dividing zero by `1e-300` — a perfect score for solving nothing.

```python
nontrivial    = rms(u_T) > 1e-3 * rms(u_0)          # not collapsed
ic_consistent = rel_err(solve_to(t=0), u_0) < 1e-10 # the run started where it should
structural    = all(gate_invariants_pass)           # D2, §8
```

**What a residual is still blind to**, and what covers each: no error bound (`‖e‖ <= ‖L_h^-1‖·‖r‖`,
and the stability constant is unbounded near a Helmholtz resonance) — Tier C. Amplitude and phase:
for a linear homogeneous PDE `c·u` is also a solution, and for a translation-invariant problem so is
`u(x-s)`, so the residual is *identically zero* on both — D2's invariants. Boundary conditions: a
stencil needs ghost points, so the residual is interior-only and a BC violation lives exactly where
it cannot be computed — D2's `boundary_consistency`. Under-resolution: on a coarse grid every
derivative is small and mutually consistent, the residual small, and the answer 30% wrong — Tier C.

---

## 27. The Invariant Registry

`verification.invariants` and `verification.constraints` entries are **catalogue names with
parameters**, not free expressions:

```json
{"name": "divergence_free", "gate": true, "tol": 1e-08, "fields": ["Bx", "By"]}
{"name": "energy_bounded",  "gate": true, "bound": 100.0, "requires_trace": true}
```

`verifylib/kernel/invariants.py` owns a **closed registry** of the §8 and §21 names, and
`verifylib/schema.py` errors on a gated entry that neither names a registry entry nor carries an
evaluable `expr`. A gate nobody implements is worse than no gate, because it reads as evidence:
`{"name": "physically_reasonable", "gate": true}` used to pass the schema, match nothing, and render
downstream as a satisfied gate.

Ungated entries stay free-form notes — eight distinct names are in use that way, and requiring an
implementation for a diagnostic nobody gates on would buy nothing.

**A declared gate the kernel cannot evaluate reports `not_reported`, never `pass`, and caps the
score at 9** — the standing rule for a missing `invariant_trace` (§8), generalised.

One measured trap, worth stating because two real specs write it: a tolerance of `tol: 0.0` for a
quantity that is *exactly* monotone or *exactly* conserved. The reference snippets in §8 test
`drift < tol`, which on `tol = 0.0` reports a violation for a drift of exactly zero — a perfectly
conserving scheme failing its own conservation check. Compare `drift <= tol + 1e-12`.

---

## 28. Chaotic Problems and the Tier-C Waiver

`chaotic: true` on the spec, with `chaotic_T_ref`. When set:

- Tier **C** at full `T` is **waived, not failed** — trajectories separate past the Lyapunov time,
  so the asymptotic guards cannot pass and their failure carries no information about the scheme.
- Tier **C** runs instead at the declared `chaotic_T_ref`, inside the predictable window. That means
  a **second full ladder**, solved through `override={"params": {"t_final": chaotic_T_ref}}`. There
  is no way to measure convergence at a horizon the solver did not run to, and that second ladder is
  the price of a chaotic problem.
- Tiers **B**, **D1** and **D2** run unchanged at full `T`.
- The waiver is recorded as `order_check: waived_chaotic` so it never reads as a pass.
- `reference.check_reference` short-circuits: no closed form can apply, so `unavailable` would
  misreport the metrics.

**It is a self-relaxation vector.** It waives a scoring gate, so it requires a `requirements` ledger
entry naming `chaotic` in its `spec_path` and quoting the clause of `problem.md` that establishes it,
and `chaotic_T_ref` must be **declared** rather than chosen by the evaluator at scoring time. Both
are schema errors when absent.

**The solver must honour the horizon override, and the kernel checks that it did.** A solver that
hard-codes `t_final` runs cleanly, returns `status: ok`, and hands back the full-`T` field — so the
shifted ladder would measure convergence at `t = 50` while reporting it as `t = 5`. Measured on
`pde_kuramoto_sivashinsky`: **two of its three solvers do exactly that.** The kernel compares the
returned `t_final` against what it asked for, and where they disagree it stays on the full-`T` ladder
and names the defect. §7 already said a solver that ignores `override` cannot be certified above Tier
C; this is what makes that detectable rather than assumed.

Note also what the waiver is *not* for. It moves the horizon Tier C is measured at; it does not
relax a guard. A spectral scheme failing `p_sane` is a separate problem with a separate fix (§4), and
setting `chaotic` will not rescue it — measured, it changes the label and leaves the score at 4.

A second, narrower waiver exists and is deliberately distinct: `evaluation_thresholds.order_check:
false` records `order_check: waived_spec`. There the problem contract simply imposes no
convergence-order requirement (`pde_kuramoto_sivashinsky` lists only a relative-L2 target). The
asymptotic guards still run, against `verification.theoretical_order`; `order_ok` is vacuously true;
and Tier C rests on the GCI alone. Keeping the two waivers separate is what stops a spec-level
opt-out from silently becoming a tier-level one.

---
