"""Independent ground-truth verification of a pipeline solution.

The pipeline scores itself (formulator writes the analytic solution, evaluator
grades against it). This module re-checks that verdict *independently*: it imports
the best plan's ``solver.py``, runs it, and compares the raw numerical output to
the benchmark's own validated ground truth (from ``problems.py``) -- never to the
formulator's extracted solution. That catches two failure modes the pipeline
cannot catch itself:

  * a formulator that extracts the wrong analytic solution or wrong parameters,
  * an evaluator that grades against that wrong reference and still reports 10/10.

For SDE problems it recomputes the terminal moments; for PDE problems it recomputes
the relative L2 error. Problems with no closed form (``has_ground_truth=False``)
return ``status="no_ground_truth"`` -- the pipeline's self-score is all we have.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import sys

import numpy as np

# --- PDE evaluation configuration -------------------------------------------
# The PDE contract is ``solve_pde(N)``: the harness controls the resolution so it
# can measure the *observed convergence order* (run at N and 2N), not just the
# error at one grid -- a solver can hit a small error at a fine grid while
# converging at the wrong rate (e.g. a low-order scheme on a corner singularity).
# A correct solution's discretization error must both fall below the L2 tolerance
# AND fall at least at the problem's minimum order. Legacy no-argument solvers
# (``solve_pde()``) are still verified single-grid for backward compatibility.
DEFAULT_PDE_N = 64          # base resolution (points per dimension) if unset
DEFAULT_PDE_MIN_ORDER = 1.0  # observed-order acceptance floor if unset
PDE_L2_TOL = 0.01           # relative-L2 pass tolerance (matches pde_manual.md)
ORDER_SUPERCONV_FLOOR = 1e-9  # below this, error is at noise; order is "passed"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import problems as P  # noqa: E402, I001


# --- STATE.md parsing (targeted; avoids a YAML dependency) ------------------


def parse_state(workspace_dir):
    """Parse workspace/{slug}/STATE.md frontmatter.

    Returns {phase, problem_type, best_plan, blocked_reason,
    plans: {id: {iter, score, one_sentence}}} or {} if STATE.md is absent.
    """
    path = os.path.join(workspace_dir, "STATE.md")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        lines = fh.read().splitlines()

    state = {"phase": None, "problem_type": None, "best_plan": None,
             "blocked_reason": None, "plans": {}}
    cur_plan = None
    in_plans = False
    for raw in lines:
        if raw.strip() in ("---", ""):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()

        if indent == 0:
            in_plans = False
            cur_plan = None
            if line.startswith("phase:"):
                state["phase"] = line.split(":", 1)[1].strip()
            elif line.startswith("problem_type:"):
                state["problem_type"] = line.split(":", 1)[1].strip()
            elif line.startswith("best_plan:"):
                state["best_plan"] = line.split(":", 1)[1].strip()
            elif line.startswith("blocked_reason:"):
                state["blocked_reason"] = line.split(":", 1)[1].strip()
            elif line.startswith("plans:"):
                in_plans = True
        elif in_plans and indent == 2 and line.endswith(":"):
            cur_plan = line[:-1].strip()
            state["plans"][cur_plan] = {"iter": None, "score": None, "one_sentence": None}
        elif in_plans and cur_plan and indent >= 4:
            if line.startswith("iter:"):
                state["plans"][cur_plan]["iter"] = _to_int(line.split(":", 1)[1])
            elif line.startswith("score:"):
                state["plans"][cur_plan]["score"] = _to_int(line.split(":", 1)[1])
            elif line.startswith("one-sentence:"):
                state["plans"][cur_plan]["one_sentence"] = line.split(":", 1)[1].strip()
    return state


def _to_int(s):
    try:
        return int(s.strip())
    except (ValueError, AttributeError):
        return None


def best_plan_dir(workspace_dir, state=None):
    """Locate the best plan directory: STATE.best_plan if set, else the
    highest-scoring plan on disk, else the first plan directory found."""
    state = state if state is not None else parse_state(workspace_dir)
    plans_root = os.path.join(workspace_dir, "plans")
    if not os.path.isdir(plans_root):
        return None

    if state.get("best_plan"):
        cand = os.path.join(plans_root, state["best_plan"])
        if os.path.isdir(cand):
            return cand

    # Fall back to the highest STATE score.
    if state.get("plans"):
        ranked = sorted(
            state["plans"].items(),
            key=lambda kv: (kv[1].get("score") or -1),
            reverse=True,
        )
        for pid, _ in ranked:
            cand = os.path.join(plans_root, pid)
            if os.path.isdir(cand):
                return cand

    dirs = sorted(d for d in os.listdir(plans_root) if os.path.isdir(os.path.join(plans_root, d)))
    return os.path.join(plans_root, dirs[0]) if dirs else None


# --- solver import ----------------------------------------------------------

_import_counter = 0


def import_solver(plan_dir):
    """Import solver.py from a plan directory as an isolated module."""
    global _import_counter
    solver_path = os.path.join(plan_dir, "solver.py")
    if not os.path.exists(solver_path):
        raise FileNotFoundError(f"no solver.py in {plan_dir}")
    _import_counter += 1
    modname = f"_bench_solver_{_import_counter}"
    spec = importlib.util.spec_from_file_location(modname, solver_path)
    mod = importlib.util.module_from_spec(spec)
    # Let the solver resolve sibling imports if any.
    sys.path.insert(0, plan_dir)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(plan_dir)
    return mod


# --- scoring helpers (mirror the pipeline's rubric) -------------------------


def _rel(emp, exact):
    return abs(emp - exact) / max(abs(exact), 1e-10)


def sde_score(var_rel, mean_rel, near_zero_mean):
    """Reproduce evaluator-sde's 1-10 rubric for a single component."""
    var_ok = var_rel < 0.10
    mean_ok = near_zero_mean or (mean_rel < 0.05)
    if var_ok and mean_ok:
        return 10
    if var_ok and (near_zero_mean or mean_rel < 0.10):
        return 8
    if var_ok:
        return 7
    if var_rel < 0.20:
        return 5
    if np.isfinite(var_rel):
        return 3
    return 1


def pde_score(rel_l2):
    if not np.isfinite(rel_l2):
        return 1
    if rel_l2 < 0.01:
        return 10
    if rel_l2 < 0.05:
        return 8
    if rel_l2 < 0.20:
        return 6
    if rel_l2 < 0.50:
        return 4
    return 2


# --- SDE verification -------------------------------------------------------


def sde_verify_dt(problem):
    """The step the verifier calls ``solve_sde`` with. Scaled to the horizon so the
    discretization error of a *correct* scheme cannot masquerade as a correctness
    failure (explicit schemes accumulate O(dt*T) bias); never coarser than the
    problem's own dt. This is the single source of truth for the evaluation step --
    ``setup.py`` states it in ``problem.md`` so the solver is told the step it is
    judged at, rather than the verifier silently overriding the solver's choice."""
    return min(problem["dt"], 0.04 / problem["T"])


def _scalar(v):
    """Tolerant scalar coercion: accepts a python/numpy scalar or a size-1 array.
    Returns identical values for a correct scalar; lets a solver that reported a
    1-element array for a scalar moment be scored rather than errored."""
    return float(np.asarray(v, dtype=float).item())


def _run_and_read_sde(problem, plan_dir, injected=None):
    """Re-run the best plan's solver well-resolved and read the terminal moments.

    Returns ``(comps, run_meta, result, err)`` where ``comps`` is a list of
    ``(label, empirical_mean, empirical_variance)`` (label ``"X"`` for scalar state),
    ``result`` is the raw solver output, and ``err`` is ``None`` or an error dict.

    If ``injected`` is ``(result, run_meta)`` -- the output of a sandboxed run
    (``runner.run_solver``) -- that solver output is scored directly instead of
    importing and calling ``solve_sde`` inline. The scoring downstream is byte-for-
    byte identical either way, which is what makes the sandboxed one-shot path and
    the inline pipeline path a fair comparison.
    """
    if injected is not None:
        result, run_meta = injected
    else:
        mod = import_solver(plan_dir)
        if not hasattr(mod, "solve_sde"):
            return None, None, None, {"status": "error", "error": "solver.py has no solve_sde()"}
        T = problem["T"]
        dt_verify = sde_verify_dt(problem)
        result = mod.solve_sde(num_paths=problem["num_paths"], dt=dt_verify, T=T, seed=problem["seed"])
        run_meta = {"num_paths": int(problem["num_paths"]), "dt": float(dt_verify),
                    "T": float(T), "seed": int(problem["seed"])}
    d = problem["state_dimension"]
    if d == 1:
        comps = [("X", _scalar(result["empirical_mean"]), _scalar(result["empirical_variance"]))]
        return comps, run_meta, result, None
    paths = np.asarray(result["terminal_paths"], dtype=float)
    n = problem["num_paths"]
    if paths.ndim != 2:
        return None, run_meta, result, {"status": "error",
                                        "error": f"terminal_paths ndim={paths.ndim}, expected 2"}
    if paths.shape[0] != n and paths.shape[1] == n:
        paths = paths.T  # normalize to (num_paths, d)
    labels = ["X", "Y", "Z"][:d]
    comps = [(lab, float(paths[:, i].mean()), float(paths[:, i].var(ddof=1)))
             for i, lab in enumerate(labels)]
    return comps, run_meta, result, None


def _score_sde_moments(problem, comps, gt, run_meta, reference):
    """Score per-component terminal moments against ground truth. For ``reference``
    ground truth ``gt`` maps each key to a ``(value, standard_error)`` pair and the
    pass tolerance is widened to at least 3x the reference SE so a correct solution
    within reference noise is not failed."""
    d = problem["state_dimension"]
    components = []
    scores = []
    all_pass = True
    # Values are coerced to native Python types (float/bool) so the record is always
    # JSON-serializable — numpy scalars (np.bool_ in particular) are not.
    for lab, em, ev in comps:
        suffix = "" if d == 1 else f"_{lab}"
        gm, gv = gt["mean" + suffix], gt["variance" + suffix]
        if reference:
            xm, se_m = float(gm[0]), float(gm[1])
            xv, se_v = float(gv[0]), float(gv[1])
        else:
            xm, xv, se_m, se_v = float(gm), float(gv), 0.0, 0.0
        em, ev = float(em), float(ev)
        mean_rel = float(_rel(em, xm))
        var_rel = float(_rel(ev, xv))
        near_zero = bool(abs(xm) < 0.01)
        var_tol = max(0.10, 3.0 * se_v / max(abs(xv), 1e-12))
        mean_tol = max(0.05, 3.0 * se_m / max(abs(xm), 1e-12))
        var_ok = bool(var_rel < var_tol)
        mean_ok = bool(near_zero or (mean_rel < mean_tol))
        passed = bool(var_ok and mean_ok)
        all_pass = all_pass and passed
        scores.append(sde_score(var_rel, mean_rel, near_zero))
        rec = {
            "component": lab,
            "empirical_mean": em, "exact_mean": xm, "mean_rel_err": mean_rel,
            "empirical_variance": ev, "exact_variance": xv, "variance_rel_err": var_rel,
            "near_zero_mean": near_zero, "mean_ok": mean_ok, "variance_ok": var_ok,
            "passed": passed,
        }
        if reference:
            rec["reference_se_mean"] = se_m
            rec["reference_se_variance"] = se_v
        components.append(rec)

    finite = all(np.isfinite([c["empirical_mean"], c["empirical_variance"]]).all() for c in components)
    # Report the worst mean error only over components whose mean is actually checked;
    # a near-zero exact mean is skipped by the rubric and would otherwise divide by
    # ~0 and display an astronomical, meaningless percentage.
    checked = [c for c in components if not c["near_zero_mean"]]
    worst_mean = float(max((c["mean_rel_err"] for c in checked), default=0.0))
    return {
        "status": "ok" if finite else "nonfinite",
        "gt_kind": "reference" if reference else "exact",
        "passed": bool(all_pass and finite),
        "verified_score": int(min(scores) if finite else 1),
        "worst_variance_rel_err": float(max(c["variance_rel_err"] for c in components)),
        "worst_mean_rel_err": worst_mean,
        "components": components,
        "run": run_meta,
    }


def verify_sde(problem, plan_dir, injected=None):
    """Exact-moment SDE verification."""
    comps, run_meta, _result, err = _run_and_read_sde(problem, plan_dir, injected=injected)
    if err is not None:
        return err
    gt = problem["moments"](problem["T"])
    return _score_sde_moments(problem, comps, gt, run_meta, reference=False)


def verify_sde_reference(problem, plan_dir, injected=None):
    """Reference-moment SDE verification: ground truth is a discretization-free MC of
    the exact solution, returned as (value, standard_error) per moment."""
    comps, run_meta, _result, err = _run_and_read_sde(problem, plan_dir, injected=injected)
    if err is not None:
        return err
    gt = problem["moments"](problem["T"])  # {key: (value, standard_error)}
    return _score_sde_moments(problem, comps, gt, run_meta, reference=True)


def verify_sde_stability(problem, plan_dir, injected=None):
    """Stability verification: no ground-truth moments, only that a correct scheme
    stays finite / in its domain. Confirms the terminal states are all finite (and,
    for a ``domain`` check, all within |X| < bound) -- exactly what a naive scheme
    that diverges or escapes the domain gets wrong."""
    _comps, run_meta, result, err = _run_and_read_sde(problem, plan_dir, injected=injected)
    # A blow-up can itself corrupt the output shape; only a genuinely missing solve_sde
    # is a hard error. Otherwise treat unreadable output as a stability failure below.
    if err is not None and err.get("error", "").endswith("no solve_sde()"):
        return err
    check = problem.get("stability_check", {"type": "finite"})
    bound = float(check.get("abs_bound", 1.0))

    vals = None
    if isinstance(result, dict) and "terminal_paths" in result:
        vals = np.asarray(result["terminal_paths"], dtype=float)

    if vals is not None and vals.size:
        finite_mask = np.isfinite(vals)
        finite_frac = float(finite_mask.mean())
        all_finite = bool(finite_mask.all())
        max_abs = float(np.max(np.abs(vals[finite_mask]))) if finite_mask.any() else float("inf")
    elif isinstance(result, dict):
        # No terminal states exposed: use the reported moments' finiteness as the
        # blow-up signal (a single Inf/NaN path poisons the empirical mean).
        m = float(result.get("empirical_mean", np.nan))
        v = float(result.get("empirical_variance", np.nan))
        all_finite = bool(np.isfinite(m) and np.isfinite(v))
        finite_frac = 1.0 if all_finite else 0.0
        max_abs = None
    else:
        all_finite, finite_frac, max_abs = False, 0.0, None

    in_domain = True
    if check["type"] == "domain":
        # Without terminal states we can only confirm finiteness, not the bound.
        in_domain = all_finite if max_abs is None else bool(all_finite and max_abs < bound)

    passed = bool(all_finite and in_domain)
    return {
        "status": "ok",
        "gt_kind": "stability",
        "check": check["type"],
        "passed": passed,
        "finite": bool(all_finite),
        "finite_fraction": finite_frac,
        "in_domain": bool(in_domain),
        "max_abs": max_abs,
        "domain_bound": bound if check["type"] == "domain" else None,
        "run": run_meta,
    }


# --- PDE verification -------------------------------------------------------


def _extract_coords(grid, dims):
    """Heuristic fallback for the original scalar problems: pull ``dims`` ordered
    1-D coordinate arrays out of the solver's grid dict, tolerant of the coordinate
    key names (x/y, S, etc.). New problems declare an explicit ``axes`` order (see
    ``_pde_axes``); this only covers the legacy x/y/S problems that predate that."""
    arrays = []
    if isinstance(grid, dict):
        for key in ("x", "X", "S", "y", "Y", "z", "Z"):
            if key in grid and np.ndim(grid[key]) == 1:
                arrays.append(np.asarray(grid[key], dtype=float))
        if not arrays:
            arrays = [np.asarray(v, dtype=float) for v in grid.values() if np.ndim(v) == 1]
    return arrays[:dims] if len(arrays) >= dims else None


def _pde_axes(problem, grid):
    """Return the ordered list of ``d`` 1-D axis arrays for the problem's grid.

    A problem may name its axes explicitly (``problem["axes"] = ["S", "v"]`` or
    ``["x", "y", "z"]``); this is required once the conventional-name heuristic no
    longer suffices (a second axis called ``v``, a third axis, etc.). Absent an
    explicit list we fall back to the x/y/S heuristic that the original 1-D/2-D
    scalar problems rely on."""
    if not isinstance(grid, dict):
        return None
    names = problem.get("axes")
    if names:
        arrs = [np.asarray(grid[n], dtype=float) for n in names
                if n in grid and np.ndim(grid[n]) == 1]
        return arrs if len(arrs) == len(names) else None
    return _extract_coords(grid, problem["dims"])


def rel_l2(u_num, u_exact):
    """RMS-based relative L2 error, matching pde_manual.md's convention."""
    return pde_error(u_num, u_exact, metric="l2", mask=None)


def _field_norm(a, metric="l2", mask=None):
    v = a if mask is None else a[mask]
    return float(np.mean(np.abs(v))) if metric == "l1" else float(np.sqrt(np.mean(v ** 2)))


def pde_error(u_num, u_exact, metric="l2", mask=None, ref_norm=None):
    """Relative error of a numerical field against the exact field.

    ``metric="l2"`` is the RMS-based relative L2 error (pde_manual.md's convention);
    ``metric="l1"`` is the relative L1 error, appropriate for compact-support /
    free-boundary solutions (e.g. porous medium) where the L2 rate is polluted by
    the moving front. ``mask`` (a boolean array over the grid) restricts the error
    to the in-domain nodes, which is how a non-rectangular domain (L-shape, Fichera
    corner) is scored on a rectangular grid. ``ref_norm`` overrides the denominator
    (the exact field's own norm) -- used to normalize an identically-zero vector
    component (e.g. a polarization with a zero axis) by the global field scale
    instead, so machine noise over a ~zero norm is not a spurious 100% error."""
    diff = u_num - u_exact
    if mask is not None:
        diff = diff[mask]
    err = float(np.mean(np.abs(diff))) if metric == "l1" else float(np.sqrt(np.mean(diff ** 2)))
    nrm = ref_norm if ref_norm is not None else _field_norm(u_exact, metric, mask)
    return err / (nrm + 1e-14)


def _pde_grid_N(problem):
    return int(problem.get("grid_N", DEFAULT_PDE_N))


def _pde_min_order(problem):
    return float(problem.get("min_order", DEFAULT_PDE_MIN_ORDER))


def _pde_l2_tol(problem):
    """Accuracy gate for the (primary) relative error. Defaults to 1% but a
    per-problem override lets a genuinely harder problem (a free boundary, a
    corner singularity) set the floor a correct reference scheme can actually
    clear at the harness resolution."""
    return float(problem.get("pde_l2_tol", PDE_L2_TOL))


def _pde_order_check(problem):
    """Whether to run the 2-grid order check: on by default, but requires an exact
    solution to measure error against, and can be turned off per problem (e.g. a
    shock, where the L2 order is inherently fractional/ill-defined)."""
    return bool(problem.get("pde_order_check", True)) and problem.get("analytic") is not None


def _pde_call_mode(fn):
    """'N' if ``solve_pde`` takes a resolution argument, 'noarg' for the legacy
    ``solve_pde()``, or None if the signature is neither."""
    sig = inspect.signature(fn)

    def binds(*a, **k):
        try:
            sig.bind(*a, **k)
            return True
        except TypeError:
            return False

    if binds(N=DEFAULT_PDE_N) or binds(DEFAULT_PDE_N):
        return "N"
    if binds():
        return "noarg"
    return None


def _run_pde(fn, N, mode):
    if mode == "noarg":
        return fn()
    try:
        return fn(N=N)
    except TypeError:
        return fn(N)


def _crashed(run, plan_dir):
    return {"status": "crashed", "reason": run.get("reason", "exception"),
            "error": run.get("error"), "wall_s": run.get("wall_s"),
            "plan_dir": os.path.relpath(plan_dir, REPO_ROOT)}


def _produce_pde_runs(problem, plan_dir, sandbox, timeout_s, mem_mb):
    """Execute solve_pde at the resolution(s) needed. Returns ``(runs, err)`` where
    ``runs`` is a list of ``(N_or_None, result_dict)`` (one element for single-grid
    / legacy, two for the order check) and ``err`` is a status dict or None."""
    order = _pde_order_check(problem)
    grid_N = _pde_grid_N(problem)

    if sandbox:
        import runner
        run1 = runner.run_solver(problem=problem, plan_dir=plan_dir,
                                 timeout_s=timeout_s, mem_mb=mem_mb, pde_N=grid_N)
        if run1["status"] != "ok":
            return None, _crashed(run1, plan_dir)
        accepted = bool(run1.get("accepted_resolution"))
        runs = [(grid_N if accepted else None, run1["result"])]
        if accepted and order:
            run2 = runner.run_solver(problem=problem, plan_dir=plan_dir,
                                     timeout_s=timeout_s, mem_mb=mem_mb, pde_N=2 * grid_N)
            if run2["status"] != "ok":
                return None, _crashed(run2, plan_dir)
            runs.append((2 * grid_N, run2["result"]))
        return runs, None

    # Inline (trusted pipeline path): import and call directly.
    mod = import_solver(plan_dir)
    if not hasattr(mod, "solve_pde"):
        return None, {"status": "error", "error": "solver.py has no solve_pde()"}
    mode = _pde_call_mode(mod.solve_pde)
    if mode is None:
        return None, {"status": "error",
                      "error": "solve_pde signature is neither solve_pde(N) nor solve_pde()"}
    if mode == "noarg":
        return [(None, mod.solve_pde())], None
    Ns = [grid_N, 2 * grid_N] if order else [grid_N]
    return [(N, _run_pde(mod.solve_pde, N, mode)) for N in Ns], None


def _field_num(result, problem):
    """Normalize a solver's PDE output to a ``{field_name: ndarray}`` dict.

    A multi-field (systems) problem declares ``problem["fields"]`` and its solver
    returns ``{"fields": {...}}``; a scalar problem returns a bare
    ``numerical_solution`` array. Either is coerced to a dict keyed by the problem's
    field names, so the same per-field scoring path serves both."""
    fields = problem.get("fields")
    if fields:
        got = result.get("fields")
        if isinstance(got, dict):
            return {k: np.asarray(got[k], dtype=float) for k in fields if k in got}, fields
        # tolerate a scalar-style return for a single-field problem
        if len(fields) == 1 and "numerical_solution" in result:
            return {fields[0]: np.asarray(result["numerical_solution"], dtype=float)}, fields
        raise KeyError(f"multi-field problem expects a `fields` dict with keys {fields}")
    return {"u": np.asarray(result["numerical_solution"], dtype=float)}, ["u"]


def _domain_mask(problem, coords):
    """Boolean in-domain mask over the (meshgridded) coordinates for a
    non-rectangular domain, or None for a full rectangular grid."""
    fn = problem.get("domain_mask")
    if fn is None:
        return None
    return np.asarray(fn(*coords), dtype=bool)


def _run_diagnostics(problem, num_fields, exact_fields, axes, mask):
    """Evaluate the problem's structure diagnostics on the *numerical* fields
    (divergence-free, positivity, locking, ...), with the exact fields available
    for comparison (e.g. a div-u error against the exact divergence). Each is a
    ``fn(num, exact, axes, mask) -> (value, passed)`` callable authored in
    problems.py — never the solver's self-report. A ``gate`` diagnostic that fails
    turns the verdict into a CONSTRAINT_VIOLATION; the rest are reported."""
    out = []
    for d in problem.get("diagnostics", ()):
        try:
            value, passed = d["fn"](num_fields, exact_fields, axes, mask)
            value, passed = float(value), bool(passed)
        except Exception as exc:  # noqa: BLE001 -- a broken diagnostic must not crash scoring
            value, passed = float("nan"), False
            out.append({"name": d["name"], "value": value, "passed": passed,
                        "gate": bool(d.get("gate", False)), "error": f"{type(exc).__name__}: {exc}"})
            continue
        out.append({"name": d["name"], "value": value, "passed": passed,
                    "gate": bool(d.get("gate", False))})
    return out


def _pde_eval_run(problem, result):
    """Relative error of one solve against the exact solution, per field, plus the
    structure diagnostics. Returns ``(eval_dict, err)`` with ``err`` a status dict
    or None. Handles any spatial dimension (via ``meshgrid(*axes)``), scalar or
    multi-field output, an optional in-domain mask, and the L1 or L2 metric."""
    grid = result.get("grid", {})
    t_eval = problem["t_eval"]  # trust the benchmark's evaluation time, not the solver label
    metric = problem.get("pde_metric", "l2")

    axes = _pde_axes(problem, grid)
    if axes is None:
        return None, {"status": "inconclusive",
                      "error": f"could not extract {problem['dims']}-D axes from grid keys "
                               f"{list(grid) if isinstance(grid, dict) else grid}"}
    coords = np.meshgrid(*axes, indexing="ij")  # d-generic; for d=1 this is [x]
    mask = _domain_mask(problem, coords)

    try:
        num, field_names = _field_num(result, problem)
        exact = problem["analytic"](t_eval, *coords)
        exact = exact if isinstance(exact, dict) else {"u": exact}
        pairs = {}
        for name in field_names:
            ue = np.asarray(exact[name], dtype=float)
            un = np.asarray(num[name], dtype=float).reshape(ue.shape)
            pairs[name] = (un, ue)
    except (ValueError, TypeError, KeyError) as exc:
        return None, {"status": "inconclusive",
                      "error": f"field/shape mismatch vs exact grid — {type(exc).__name__}: {exc}"}

    # Non-finite output anywhere in-domain is a contract failure, not a score.
    for _name, (un, _ue) in pairs.items():
        vals = un if mask is None else un[mask]
        if not np.all(np.isfinite(vals)):
            return None, {"status": "nonfinite", "passed": False, "verified_score": 1,
                          "rel_l2_err": float("inf"), "primary_err": float("inf")}

    # Gauge fields (e.g. incompressible pressure) are defined only up to a constant;
    # compare them mean-removed over the domain so a valid gauge choice is not failed.
    gauge = set(problem.get("gauge_fields", ()))
    shifted = {}
    for name, (un, ue) in pairs.items():
        if name in gauge:
            sel = slice(None) if mask is None else mask
            un = un - float(np.mean(un[sel]))
            ue = ue - float(np.mean(ue[sel]))
        shifted[name] = (un, ue)
    # A field that is identically ~0 (e.g. a vector component along a zero polarization
    # axis) has no meaningful self-relative error, so normalize it by the global field
    # scale instead of its own ~0 norm.
    scales = {name: _field_norm(ue, metric, mask) for name, (un, ue) in shifted.items()}
    global_scale = max(scales.values(), default=0.0)
    field_errs = {}
    for name, (un, ue) in shifted.items():
        rn = global_scale if (global_scale > 0 and scales[name] < 1e-6 * global_scale) else None
        field_errs[name] = pde_error(un, ue, metric=metric, mask=mask, ref_norm=rn)

    primary = problem.get("primary_field", field_names[0])
    max_abs = max(float(np.max(np.abs((un - ue) if mask is None else (un - ue)[mask])))
                  for un, ue in pairs.values())
    diagnostics = _run_diagnostics(problem, {n: p[0] for n, p in pairs.items()},
                                   {n: p[1] for n, p in pairs.items()}, axes, mask)

    ev = {"rel_l2_err": field_errs[primary], "primary_err": field_errs[primary],
          "primary_field": primary, "metric": metric, "field_errs": field_errs,
          "max_abs_err": max_abs, "grid_shape": list(pairs[primary][0].shape),
          "diagnostics": diagnostics, "t_final": result.get("t_final")}
    return ev, None


def _fold_diagnostics(out, ev):
    """Attach the finest grid's diagnostics to the score dict and, if any hard-gate
    constraint failed, mark the run a constraint violation (and a fail)."""
    diags = ev.get("diagnostics", [])
    if len(ev.get("field_errs", {})) > 1:
        out["field_errs"] = ev["field_errs"]
    if diags:
        out["diagnostics"] = diags
        gate_fail = [d["name"] for d in diags if d["gate"] and not d["passed"]]
        if gate_fail:
            out["constraint_violation"] = True
            out["violated_constraints"] = gate_fail
            out["passed"] = False
    return out


def _pde_converged(problem, ev):
    """Accuracy gate: every *required* field within the (primary-metric) tolerance."""
    tol = _pde_l2_tol(problem)
    required = problem.get("required_fields") or list(ev["field_errs"])
    return bool(all(ev["field_errs"][f] < tol for f in required)), tol


def _score_pde_runs(problem, runs):
    evals = []
    for _N, result in runs:
        ev, err = _pde_eval_run(problem, result)
        if err is not None:
            return err
        evals.append(ev)
    t_eval = problem["t_eval"]

    if len(evals) == 1:  # single-grid (legacy no-arg, or order check disabled)
        N, ev = runs[0][0], evals[0]
        converged, _tol = _pde_converged(problem, ev)
        out = {"status": "ok", "passed": bool(converged),
               "verified_score": pde_score(ev["primary_err"]),
               "rel_l2_err": ev["primary_err"], "max_abs_err": ev["max_abs_err"],
               "grid_shape": ev["grid_shape"], "metric": ev["metric"],
               "primary_field": ev["primary_field"], "t_eval": t_eval,
               "solver_t_final": ev["t_final"], "order_checked": False}
        if N is not None:
            out["grid_N"] = N
        return _fold_diagnostics(out, ev)

    # Two grids: observed convergence order p = log2(e_N / e_2N) on the primary field.
    (N0, _r0), (N1, _r1) = runs[0], runs[1]
    e0, e1 = evals[0]["primary_err"], evals[1]["primary_err"]
    min_order = _pde_min_order(problem)
    converged, tol = _pde_converged(problem, evals[1])
    # The order floor exists to reject a solver that is accurate at ONE grid without
    # actually converging. It is waived when the coarse grid is ALSO within tolerance
    # (the solution is genuinely resolved at two independent resolutions -- a
    # spectral / high-order scheme plateaus at its time or round-off floor, so its
    # "observed order" is ~0 yet it is plainly correct), or at the noise floor. The
    # floor still bites a solver that only reaches tolerance at the fine grid.
    coarse_ok, _t0 = _pde_converged(problem, evals[0])
    if e1 < ORDER_SUPERCONV_FLOOR or e0 < ORDER_SUPERCONV_FLOOR:
        p_hat = float("inf")
        order_ok = True
    else:
        p_hat = float(np.log(e0 / e1) / np.log(2.0))
        order_ok = bool(p_hat >= min_order or coarse_ok)
    passed = bool(converged and order_ok)
    out = {"status": "ok", "passed": passed,
           "verified_score": pde_score(e1) if order_ok else min(pde_score(e1), 5),
           "rel_l2_err": e1, "max_abs_err": evals[1]["max_abs_err"],
           "grid_shape": evals[1]["grid_shape"], "metric": evals[1]["metric"],
           "primary_field": evals[1]["primary_field"], "t_eval": t_eval,
           "solver_t_final": evals[1]["t_final"], "order_checked": True,
           "observed_order": p_hat, "order_min": min_order, "order_ok": order_ok,
           "converged": converged, "grid_N": N0, "fine_grid_N": N1,
           "grid_errors": {str(N0): e0, str(N1): e1}}
    return _fold_diagnostics(out, evals[1])


def verify_pde(problem, plan_dir, sandbox=False, timeout_s=120, mem_mb=4096):
    runs, err = _produce_pde_runs(problem, plan_dir, sandbox, timeout_s, mem_mb)
    if err is not None:
        return err
    return _score_pde_runs(problem, runs)


# --- dispatch ---------------------------------------------------------------


def verify_problem(problem, workspace_dir, plan_dir=None, sandbox=False,
                   timeout_s=120, mem_mb=4096):
    """Independently verify one solved problem. Returns a result dict with a
    ``status`` field: no_ground_truth | ok | inconclusive | nonfinite | error |
    crashed.

    Dispatch is on ``ground_truth_kind`` (default ``exact``): ``reference`` compares
    against a discretization-free MC reference within an SE-aware tolerance;
    ``stability`` checks only that a correct scheme stays finite / in-domain (and is
    checkable even though ``has_ground_truth`` is False).

    ``sandbox=True`` runs ``solver.py`` in an isolated subprocess (wall-clock
    timeout, memory cap, numpy+stdlib-only import policy) via ``runner.py`` and
    scores the returned output. Use it for untrusted (one-shot baseline) solvers.
    A sandbox failure -- timeout, out-of-memory, blocked import, wrong signature,
    or a return value that violates the contract -- yields ``status="crashed"``
    with a ``reason``; it is a genuine failure of that solver, never an
    "unverified" of the benchmark. The default ``sandbox=False`` keeps the vetted
    pipeline path importing the solver inline, unchanged."""
    kind = problem.get("ground_truth_kind", "exact")
    if kind != "stability" and not problem["has_ground_truth"]:
        return {"status": "no_ground_truth",
                "note": "no closed form; pipeline self-score is the only signal"}

    if plan_dir is None:
        plan_dir = best_plan_dir(workspace_dir)
    if plan_dir is None or not os.path.isdir(plan_dir):
        return {"status": "error", "error": f"no plan directory under {workspace_dir}"}

    # SDE is a single run, so it is pre-executed here (sandboxed) and injected. PDE
    # may need two runs (the N/2N order check), so verify_pde orchestrates its own
    # sandboxed execution.
    injected = None
    if sandbox and problem["type"] == "sde":
        import runner  # local import: keeps the pipeline path free of the dependency
        run = runner.run_solver(problem=problem, plan_dir=plan_dir,
                                timeout_s=timeout_s, mem_mb=mem_mb)
        if run["status"] != "ok":
            return _crashed(run, plan_dir)
        injected = (run["result"], run["run_meta"])

    try:
        if problem["type"] == "sde":
            if kind == "stability":
                out = verify_sde_stability(problem, plan_dir, injected=injected)
            elif kind == "reference":
                out = verify_sde_reference(problem, plan_dir, injected=injected)
            else:
                out = verify_sde(problem, plan_dir, injected=injected)
        else:
            out = verify_pde(problem, plan_dir, sandbox=sandbox,
                             timeout_s=timeout_s, mem_mb=mem_mb)
    except Exception as exc:  # noqa: BLE001 -- verification must never crash the runner
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                "plan_dir": os.path.relpath(plan_dir, REPO_ROOT)}
    out["plan_dir"] = os.path.relpath(plan_dir, REPO_ROOT)
    return out


def _fmt(out):
    lines = [f"status: {out['status']}"]
    if out["status"] == "no_ground_truth":
        lines.append("  " + out["note"])
    elif out["status"] == "crashed":
        lines.append(f"  sandbox crash ({out.get('reason')}): {out.get('error', '')}")
    elif out["status"] in ("error", "inconclusive"):
        lines.append("  " + out.get("error", ""))
    elif out.get("gt_kind") == "stability":  # SDE stability
        fin = ("all-finite" if out.get("finite")
               else f"NON-FINITE (finite_frac={out['finite_fraction']:.5f})")
        dom = ("" if out["check"] != "domain"
               else (", in-domain" if out["in_domain"] else ", ESCAPED-DOMAIN"))
        mx = f", max|X|={out['max_abs']:.4g}" if out.get("max_abs") is not None else ""
        lines.append(f"  stability ({out['check']}): {fin}{dom}{mx}")
        lines.append(f"  verdict       : {'STABLE (no blow-up)' if out['passed'] else 'BLOW-UP'}")
        lines.append(f"  plan          : {out['plan_dir']}")
    elif "rel_l2_err" in out:  # PDE
        metric = out.get("metric", "l2").upper()
        pfld = out.get("primary_field", "u")
        lines.append(f"  rel {metric} error : {out['rel_l2_err']:.3e}  (field '{pfld}', "
                     f"pass < {_pde_l2_tol({}):g} -> {'PASS' if out.get('passed') else 'FAIL'})")
        if out.get("field_errs"):
            per = "  ".join(f"{k}={v:.2e}" for k, v in out["field_errs"].items())
            lines.append(f"  per-field    : {per}")
        if out.get("order_checked"):
            p = out.get("observed_order")
            pstr = "inf" if p == float("inf") else f"{p:.2f}"
            lines.append(f"  observed order: {pstr}  (min {out.get('order_min')}, grids "
                         f"N={out.get('grid_N')}/{out.get('fine_grid_N')} -> "
                         f"{'ok' if out.get('order_ok') else 'TOO LOW'})")
        for d in out.get("diagnostics", []):
            gate = "GATE" if d["gate"] else "info"
            verdict = "ok" if d["passed"] else ("VIOLATED" if d["gate"] else "off")
            lines.append(f"  [{gate}] {d['name']:<22}= {d['value']:.3e}  -> {verdict}")
        if out.get("constraint_violation"):
            lines.append(f"  !! CONSTRAINT VIOLATION: {', '.join(out['violated_constraints'])}")
        lines.append(f"  verified score: {out['verified_score']}/10")
        lines.append(f"  plan          : {out['plan_dir']}")
    else:  # SDE moments (exact or reference)
        for c in out["components"]:
            lines.append(
                f"  [{c['component']}] mean {c['empirical_mean']:.6g} vs {c['exact_mean']:.6g} "
                f"(rel {c['mean_rel_err']:.3%}{' skip' if c['near_zero_mean'] else ''}) | "
                f"var {c['empirical_variance']:.6g} vs {c['exact_variance']:.6g} "
                f"(rel {c['variance_rel_err']:.3%})")
        lines.append(f"  verified score: {out['verified_score']}/10 "
                     f"({'PASS' if out['passed'] else 'FAIL'})")
        lines.append(f"  plan          : {out['plan_dir']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Independently verify a solved benchmark problem.")
    ap.add_argument("--slug", required=True, help="benchmark problem slug (e.g. sde_gbm)")
    ap.add_argument("--workspace", default=os.path.join(REPO_ROOT, "workspace"),
                    help="workspace root (default: <repo>/workspace)")
    ap.add_argument("--workspace-dir", default=None,
                    help="explicit workspace dir (overrides --workspace/--slug pairing)")
    ap.add_argument("--plan-dir", default=None, help="explicit plan dir to verify")
    ap.add_argument("--sandbox", action="store_true",
                    help="run solver.py in an isolated subprocess (timeout, mem cap, "
                         "numpy+stdlib-only) -- use for untrusted one-shot solvers")
    ap.add_argument("--timeout", type=int, default=120,
                    help="sandbox wall-clock timeout in seconds (default 120)")
    ap.add_argument("--mem-mb", type=int, default=4096,
                    help="sandbox address-space cap in MB, best-effort (default 4096)")
    args = ap.parse_args(argv)

    problem = P.by_slug(args.slug)
    wdir = args.workspace_dir or os.path.join(args.workspace, args.slug)
    plan_dir = os.path.abspath(args.plan_dir) if args.plan_dir else None
    out = verify_problem(problem, wdir, plan_dir=plan_dir, sandbox=args.sandbox,
                         timeout_s=args.timeout, mem_mb=args.mem_mb)
    print(f"=== {problem['id']} {args.slug} ({problem['type'].upper()}, tier {problem['tier']}) ===")
    print(_fmt(out))


if __name__ == "__main__":
    main()
