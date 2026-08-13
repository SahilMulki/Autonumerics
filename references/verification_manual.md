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
| **A** | Closed-form solution / exact moments | The answer is right | `analytic` |
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
p_sane      = p < (theoretical_order + 1.0)   # a wild p signals noise, not super-convergence
not_at_roundoff = d21 > 1e-13 * (rms(u1) + 1e-30)
asymptotic  = shrinking and p_finite and p_sane and not_at_roundoff
```

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

Same rules as §11, with one addition specific to Monte Carlo: sibling plans must be run with
**independent seeds**, and agreement is tested against the *combined* standard error:

```python
agree = abs(m_a - m_b) < ci_mult * np.sqrt(se_a**2 + se_b**2)
```

Two plans agreeing on a shared seed proves nothing — they saw the same noise.

---

## 23. SDE Scoring Rubric

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
