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

A PDE need not have a closed form to be checked independently. ``ground_truth_kind:
"reference"`` compares against a **harness-owned reference field** -- a stored
high-resolution solution computed offline by two independent method families that
agree far below the problem's own tolerance (``benchmark/make_references.py``). The
tolerance is widened by the recorded agreement so the reference's own error can never
fail a correct solver, exactly as the SDE reference path widens by the Monte-Carlo
standard error. ``functional_truth`` adds the other no-closed-form route: the truth is
a scalar quantity of interest (an eigenvalue, an extremum, an option price) rather
than a field, gated relatively and widened per value by whatever absolute uncertainty
the reference number itself carries (``reference_abs_err``), which is the same
widening one notch finer.

That route runs on **both** problem types. It is orthogonal to the type and to the
ground-truth kind: an SDE problem may be scored on a price with no terminal-moment
formula behind it, on moments alone, or on both as independent gates. The one thing
the SDE side needs that the PDE side gets free is a defence against a solver that
simply returns the published number -- the PDE 2-grid order check already fails a
constant, an SDE price graded from one run does not -- so a functional-scored SDE
problem is solved at a second, coarser path count and the two prices are required to
differ and to agree within the sampling noise the two counts imply.
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
# A harness-owned reference field carries its own error (the measured disagreement
# between the two independent schemes that produced it). The pass tolerance is
# widened to at least this multiple of it, so a solver can never be failed by the
# answer key's own inaccuracy -- the same rule the SDE reference path applies to the
# Monte-Carlo standard error.
REFERENCE_TOL_FACTOR = 3.0

# The sandbox budget for ONE graded solve, shared by every caller -- `run.py`
# (pipeline), `oneshot.py` (baseline) and this module's CLI -- because a pipeline
# and a baseline measured under different budgets are not comparable, and two
# defaults in two files drift. 300 s is 4.2x the slowest solve measured over the 34
# solved workspaces (`pde_cahn_hilliard_2d` at N=128, 71.1 s, 2026-10-17), so a
# machine four times slower than that one still grades correctly; the previous 120 s
# left a 1.7x margin on a *passing* problem, where the failure mode is a correct
# solver recorded CRASHED. No recorded run has ever hit a verification timeout. Both
# runners write the value they used into their results config, so a sweep taken at a
# different budget is identifiable rather than silently pooled.
DEFAULT_SOLVER_TIMEOUT_S = 300
DEFAULT_SOLVER_MEM_MB = 4096

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)  # verifylib is not installed; package = false

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


class OracleLeak(Exception):
    """A solver reached for the answer key it is graded against."""


def scan_for_leakage(path):
    """Refuse to execute a solver that reads or cites the answer key.

    Called from **both** execution paths. ``runner.run_solver`` is only reached
    when ``sandbox=True``, which is the one-shot baseline; the pipeline path is
    ``sandbox=False`` and imports inline through this module, so a scan wired only
    into the runner would never see a pipeline-produced solver.
    """
    from verifylib.leakage import leakage_findings

    with open(path) as fh:
        findings = leakage_findings(fh.read(), path=os.path.relpath(path, REPO_ROOT))
    hard = [f for f in findings if f.severity == "error"]
    if hard:
        raise OracleLeak("; ".join(f.message for f in hard))


def import_solver(plan_dir):
    """Import solver.py from a plan directory as an isolated module."""
    global _import_counter
    solver_path = os.path.join(plan_dir, "solver.py")
    if not os.path.exists(solver_path):
        raise FileNotFoundError(f"no solver.py in {plan_dir}")
    scan_for_leakage(solver_path)
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
# The moment gates, named because a third caller now reads them: `_functional_tol`
# takes the mean gate as the default for a quantity of interest on an SDE problem,
# the way it takes `_pde_l2_tol` on a PDE one. They are deliberately loose -- a
# sampled moment is not a resolved field -- which is exactly why a BENCHOP-style
# price declares its own `functional_tol` rather than inheriting this.
SDE_VAR_TOL = 0.10          # relative gate on a terminal variance
SDE_MEAN_TOL = 0.05         # relative gate on a terminal mean (skipped near zero)


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
    paths = np.asarray(result["terminal_paths"], dtype=float)
    # The path count comes from the run that was actually executed, not from the
    # problem: the graded run uses the problem's own count, but a work-precision
    # level deliberately varies it, and a (d, num_paths) array would then fail to be
    # recognized as transposed and be scored component-for-path.
    n = int((run_meta or {}).get("num_paths") or problem["num_paths"])
    if d == 1:
        # The scored moments stay the solver's own reported values -- that is the
        # contract, and recomputing them here would silently forgive a solver whose
        # report disagrees with its paths. The standard errors come from the paths,
        # because the report does not carry them; a malformed `terminal_paths` costs
        # only the error bars, so it must not change the verdict.
        comps = [("X", _scalar(result["empirical_mean"]),
                  _scalar(result["empirical_variance"]), *_empirical_ses(paths.ravel()))]
        return comps, run_meta, result, None
    if paths.ndim != 2:
        return None, run_meta, result, {"status": "error",
                                        "error": f"terminal_paths ndim={paths.ndim}, expected 2"}
    if paths.shape[0] != n and paths.shape[1] == n:
        paths = paths.T  # normalize to (num_paths, d)
    labels = ["X", "Y", "Z"][:d]
    comps = [(lab, float(paths[:, i].mean()), float(paths[:, i].var(ddof=1)),
              *_empirical_ses(paths[:, i])) for i, lab in enumerate(labels)]
    return comps, run_meta, result, None


def _empirical_ses(col):
    """``(se_mean, se_variance)`` of one component's terminal sample, or ``(None,
    None)`` if it cannot be formed.

    Reporting only -- the gates are fixed relative tolerances and no verdict moves.
    But without them a reader cannot tell a resolved 0.9% mean error from an
    unresolved one, and neither can ``run.py``, whose ``wrong_plan_won`` would
    otherwise rank two plans on a difference smaller than the sampling noise
    (measured on `sde_double_well_langevin`: 0.955% against 0.941% with a standard
    error of 0.76%)."""
    col = np.asarray(col, dtype=float).ravel()
    if col.size < 2 or not np.all(np.isfinite(col)):
        return None, None
    v = float(col.var(ddof=1))
    se_m = float(col.std(ddof=1) / np.sqrt(col.size))
    # se of a variance estimate, from the fourth central moment.
    se_v = float(np.sqrt(max(float(((col - col.mean()) ** 4).mean()) - v ** 2, 0.0) / col.size))
    return se_m, se_v


def _score_sde_moments(problem, comps, gt, run_meta, reference, result=None):
    """Score per-component terminal moments against ground truth, and any scalar
    quantity of interest beside them. For ``reference`` ground truth ``gt`` maps each
    key to a ``(value, standard_error)`` pair and the pass tolerance is widened to at
    least 3x the reference SE so a correct solution within reference noise is not
    failed.

    ``gt`` may be empty, which is new in step 5: an SDE problem whose truth is a
    *price* has no elementary terminal moment to check, and ``functional_truth``
    carries the whole grade. Where a problem declares both they are independent
    gates -- the moments say the simulated law is right, the functional says the
    number read off it is -- and failing either fails.

    ``result`` is the raw solver output and is read only for ``functionals``. It
    defaults to ``None`` so the callers that drive scoring from hand-built
    components, ``validate_ground_truth.py`` and the tests, keep working unchanged.
    """
    d = problem["state_dimension"]
    components = []
    scores = []
    all_pass = True
    # Values are coerced to native Python types (float/bool) so the record is always
    # JSON-serializable — numpy scalars (np.bool_ in particular) are not.
    for comp in (comps or []) if gt else []:
        # 3-tuples are still accepted: validate_ground_truth.py and the tests build
        # components by hand from a scheme they drive themselves, and the standard
        # errors are reporting-only.
        lab, em, ev = comp[0], comp[1], comp[2]
        emp_se_m, emp_se_v = (comp[3], comp[4]) if len(comp) >= 5 else (None, None)
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
        var_tol = max(SDE_VAR_TOL, REFERENCE_TOL_FACTOR * se_v / max(abs(xv), 1e-12))
        mean_tol = max(SDE_MEAN_TOL, REFERENCE_TOL_FACTOR * se_m / max(abs(xm), 1e-12))
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
        if emp_se_m is not None:
            rec["empirical_se_mean"] = float(emp_se_m)
            rec["empirical_se_variance"] = float(emp_se_v)
            rec["mean_resolved"] = bool(abs(em - xm) > 2.0 * float(emp_se_m))
            rec["variance_resolved"] = bool(abs(ev - xv) > 2.0 * float(emp_se_v))
        components.append(rec)

    finite = all(np.isfinite([c["empirical_mean"], c["empirical_variance"]]).all() for c in components)
    # Report the worst mean error only over components whose mean is actually checked;
    # a near-zero exact mean is skipped by the rubric and would otherwise divide by
    # ~0 and display an astronomical, meaningless percentage.
    checked = [c for c in components if not c["near_zero_mean"]]
    worst_mean = float(max((c["mean_rel_err"] for c in checked), default=0.0))
    # A functional-only problem has no components, so every aggregate over them needs
    # an identity rather than an exception: no moment is wrong, which is a 10 and a
    # zero error, and the functional branch below is then the only thing that can
    # lower either.
    out = {
        "status": "ok" if finite else "nonfinite",
        "gt_kind": "reference" if reference else "exact",
        "passed": bool(all_pass and finite),
        "verified_score": int(min(scores)) if (finite and scores) else (10 if finite else 1),
        "worst_variance_rel_err": float(max((c["variance_rel_err"] for c in components),
                                            default=0.0)),
        "worst_mean_rel_err": worst_mean,
        "components": components,
        "run": run_meta,
    }
    return _score_sde_functionals(problem, result, out)


def _score_sde_functionals(problem, result, out):
    """Fold the quantity-of-interest gate into a scored SDE record.

    The same gate as the PDE route, reached through the same two helpers --
    ``_functional_errs`` for the relative and absolute error, ``_functional_tols``
    for the per-value floored tolerance. That is not code thrift. BENCHOP states
    every problem twice, once as a PDE and once as an SDE, and compares the two
    (plan-benchop-slv.md §0, point c); a price graded by one rule on one formulation
    and another rule on the other would make that a comparison of two graders rather
    than of two methods.

    A contract or type failure on the functional returns in place of the moment
    record rather than beside it, exactly as on the PDE route: a solver told to
    return a number and not returning it has failed, whatever its moments did. The
    two PDE-shaped error fields are dropped on the way out, because ``rel_l2_err`` on
    an SDE record would send ``_fmt`` and every reader down the PDE branch."""
    if not problem.get("functional_truth"):
        return out
    errs, abs_errs, err = _functional_errs(problem, result or {})
    if err is not None:
        rec = {k: v for k, v in err.items() if k not in ("rel_l2_err", "primary_err")}
        rec["gt_kind"] = out.get("gt_kind")
        rec["components"] = out.get("components", [])
        rec["run"] = out.get("run")
        return rec
    base, tols = _functional_tol(problem), _functional_tols(problem)
    worst = float(max(errs.values(), default=0.0))
    out["functional_errs"] = errs
    out["functional_abs_errs"] = abs_errs
    out["functional_tol"] = base
    out["functional_tol_used"] = tols
    out["worst_functional_rel_err"] = worst
    widened = sorted(n for n, t in tols.items() if t > base)
    if widened:
        out["functional_tol_widened"] = widened
        out["reference_abs_err"] = {n: _functional_reference_abs_err(problem, n)
                                    for n in tols}
    out["functional_ok"] = bool(all(e < tols.get(n, base) for n, e in errs.items()))
    out["passed"] = bool(out["passed"] and out["functional_ok"])
    # ``pde_score`` bands a relative error of a scalar against truth, which is what
    # this is; the PDE functional route already scores a quantity of interest with
    # it. A second rubric here would make the two formulations of one BENCHOP problem
    # score differently for a reason that has nothing to do with either solver.
    out["verified_score"] = int(min(out["verified_score"], pde_score(worst)))
    return out


def _sde_level_paths(problem, num_paths):
    """The coarse path count the two-level functional check re-solves at.

    A quarter of the graded count by default, and a problem may name its own with
    ``functional_level_paths``. The quarter is chosen rather than a half because a
    Monte-Carlo standard error falls as ``1/sqrt(n)``: four times fewer paths is
    exactly twice the noise, which makes the consistency band below ``3 * tol_abs``
    -- a factor that is derived rather than picked -- while costing 25% more solve
    time than grading alone."""
    declared = problem.get("functional_level_paths")
    n = int(declared) if declared else int(num_paths) // 4
    return max(n, 0)


def _sde_functional_levels(problem, plan_dir, graded_meta, timeout_s, mem_mb):
    """Re-solve at a second path count and compare the reported quantities of
    interest. This is a **gate**, not measurement: it can fail a problem.

    It is the anti-hardcoding defence for the SDE functional route
    (plan-benchop-slv.md §6, step 5). The PDE route gets one free from the 2-grid
    order check -- a constant has zero observed order and fails -- but an SDE price
    is graded from a single run, so without this a solver that returns the published
    number and never reads its own paths passes every other check in this module.

    Two things are checked and a third is deliberately not:

      * **Movement.** Two Monte-Carlo estimates of one expectation, taken at
        different path counts, must differ. Equal to the bit is a constant, and that
        is the finding. A solver that ignores the path count it was handed lands
        here too, by the same arithmetic, and ``level_ignored`` records which of the
        two it was.
      * **Consistency.** ``|f_coarse - f_graded|`` is held inside
        ``tol_abs * (1 + sqrt(n_graded / n_coarse))``, where ``tol_abs`` is that
        value's own absolute gate. The graded run is within ``tol_abs`` of the truth
        or it has already failed; the coarse run is ``sqrt(n_graded/n_coarse)``
        times noisier, since a standard error falls as ``1/sqrt(n)``; so sampling
        alone cannot separate them by more than the sum. Anything that does is not
        sampling.
      * **Not a decreasing test.** The band is symmetric, and nothing here requires
        the finer run to be better. Step 3 measured these curves non-monotone at
        this suite's path counts -- ``sde_gbm_2d_high_corr`` scores *worse* at twice
        the work, because the error axis is dominated by sampling noise rather than
        by discretization -- so "finer must be better" would fail correct solvers.
        The rev to step 5 in the plan says this in as many words; the hardcoding
        defence does not need it.

    Everything that merely prevents the check from running -- a crashed coarse
    solve, a coarse run that returns no functionals, a problem with no coarser level
    to take -- is reported as ``not_run`` with a reason and leaves the verdict
    alone. Only ``constant`` and ``inconsistent`` fail. That asymmetry is deliberate:
    a flaky sandbox must not be able to fail a correct solver, while the two
    findings this exists to produce are both positive evidence about the solver
    itself.
    """
    import runner  # local import: keeps the pipeline path free of the dependency
    n_hi, dt = int(graded_meta["num_paths"]), float(graded_meta["dt"])
    n_lo = _sde_level_paths(problem, n_hi)
    if not 2 <= n_lo < n_hi:
        return {"status": "not_run", "reason": f"no coarser level below {n_hi} paths"}
    run = runner.run_solver(problem=problem, plan_dir=plan_dir, timeout_s=timeout_s,
                            mem_mb=mem_mb, sde_num_paths=n_lo, sde_dt=dt)
    if run["status"] != "ok":
        return {"status": "not_run", "reason": run.get("reason", "crashed"),
                "num_paths": n_lo}
    got = (run.get("result") or {}).get("functionals")
    if not isinstance(got, dict):
        return {"status": "not_run", "reason": "coarse run returned no functionals",
                "num_paths": n_lo}
    graded_got = (graded_meta.get("functionals") or {})
    tols = _functional_tols(problem)
    base = _functional_tol(problem)
    widening = 1.0 + float(np.sqrt(n_hi / n_lo))
    values, constant, inconsistent = {}, [], []
    for name, ref in problem["functional_truth"].items():
        if name not in got or name not in graded_got:
            return {"status": "not_run", "reason": f"functional {name!r} absent from a level",
                    "num_paths": n_lo}
        try:
            f_lo, f_hi = float(got[name]), float(graded_got[name])
        except (TypeError, ValueError):
            return {"status": "not_run", "reason": f"functional {name!r} is not a number",
                    "num_paths": n_lo}
        # A relative gate on a reference of 0.0 is an absolute gate of 0.0, which
        # would make every nonzero difference "inconsistent" and fail every solver
        # on the problem. No reference in the suite is zero and a test says so, but
        # the failure that an authoring slip would produce here is baffling rather
        # than legible, so the consistency half is simply not formed: the movement
        # half still runs, and that is the half the hardcoding defence rests on.
        band = tols.get(name, base) * abs(float(ref)) * widening
        delta = abs(f_hi - f_lo)
        moved = f_hi != f_lo
        consistent = True if band == 0.0 else bool(delta <= band)
        if not moved:
            constant.append(name)
        if not consistent:
            inconsistent.append(name)
        values[name] = {"graded": f_hi, "coarse": f_lo, "delta": float(delta),
                        "band": float(band), "moved": bool(moved),
                        "consistent": consistent}
    out = {"status": "ok", "num_paths_graded": n_hi, "num_paths_coarse": n_lo,
           "dt": dt, "band_factor": float(widening), "values": values,
           "passed": not (constant or inconsistent)}
    # What the coarse level got wrong, and what it cost [step 9]. The check above
    # only needs the two values to *differ*; these two numbers make the solve the
    # check has already paid for usable as a second point on the work-precision
    # curve instead of being discarded. Nothing here can move the verdict.
    coarse_rel = [abs(values[name]["coarse"] - float(ref)) / abs(float(ref))
                  for name, ref in problem["functional_truth"].items() if float(ref) != 0.0]
    if coarse_rel:
        out["worst_functional_rel_err"] = float(max(coarse_rel))
    out["solve_s"] = run.get("solve_s")
    out["wall_s"] = run.get("wall_s")
    # Why the price did not move, when it did not: the solver may be returning a
    # constant, or it may be ignoring the path count it was handed and so running
    # the same simulation twice. Both fail, and the record says which.
    paths = np.asarray((run.get("result") or {}).get("terminal_paths", []), dtype=float)
    if paths.size:
        got_n = paths.size // max(int(problem["state_dimension"]), 1)
        out["level_ignored"] = bool(got_n != n_lo)
    if constant:
        out["constant"] = constant
    if inconsistent:
        out["inconsistent"] = inconsistent
    return out


def _fold_sde_functional_levels(out, problem, plan_dir, graded_result, *, timeout_s,
                                mem_mb):
    """Run the two-level check and let it fail the verdict.

    Unlike the work-precision sweep this one is allowed to move a verdict, so the
    blanket ``except`` records ``not_run`` instead of swallowing: a defence that
    crashed did not run, and a record that said nothing would read as a defence that
    passed."""
    if out.get("status") != "ok" or not problem.get("functional_truth"):
        return out
    try:
        meta = dict(out.get("run") or {})
        meta["functionals"] = (graded_result or {}).get("functionals") or {}
        lv = _sde_functional_levels(problem, plan_dir, meta, timeout_s, mem_mb)
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        lv = {"status": "not_run", "reason": f"{type(exc).__name__}: {exc}"}
    out["functional_levels"] = lv
    if lv.get("status") == "ok" and not lv.get("passed"):
        out["passed"] = False
        out["verified_score"] = min(int(out.get("verified_score", 10)), 1)
    return out


def _sde_moment_truth(problem):
    """The problem's terminal moments, or ``None`` when it has none.

    ``moments`` stopped being mandatory in step 5: a problem scored on a price
    declares ``functional_truth`` and no moment formula, and ``.get`` is what lets
    the same two entry points serve both rather than a third one appearing."""
    fn = problem.get("moments")
    return fn(problem["T"]) if callable(fn) else None


def verify_sde(problem, plan_dir, injected=None):
    """Exact-moment SDE verification."""
    comps, run_meta, result, err = _run_and_read_sde(problem, plan_dir, injected=injected)
    if err is not None:
        return err
    return _score_sde_moments(problem, comps, _sde_moment_truth(problem), run_meta,
                              reference=False, result=result)


def verify_sde_reference(problem, plan_dir, injected=None):
    """Reference-moment SDE verification: ground truth is a discretization-free MC of
    the exact solution, returned as (value, standard_error) per moment."""
    comps, run_meta, result, err = _run_and_read_sde(problem, plan_dir, injected=injected)
    if err is not None:
        return err
    # {key: (value, standard_error)}
    return _score_sde_moments(problem, comps, _sde_moment_truth(problem), run_meta,
                              reference=True, result=result)


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


def _pde_truth(problem):
    """The field-valued ground truth, ``(t, *coords) -> array | {field: array}``.

    ``analytic`` is a closed form; ``reference`` is a harness-owned high-resolution
    solution evaluated on the requested nodes (``ground_truth_kind="reference"``).
    They are interchangeable from here down -- every per-field error, gauge removal,
    mask and diagnostic works the same on either. ``None`` when the problem has no
    field truth at all (a pure ``functional`` problem, or a self-score-only one)."""
    return problem.get("analytic") or problem.get("reference")


def _pde_reference_error(problem):
    """The reference field's own relative error: the measured disagreement between
    the two independent schemes that produced it. Zero for a closed form."""
    return float(problem.get("reference_error", 0.0) or 0.0)


def _pde_l2_tol(problem):
    """Accuracy gate for the (primary) relative error. Defaults to 1% but a
    per-problem override lets a genuinely harder problem (a free boundary, a
    corner singularity) set the floor a correct reference scheme can actually
    clear at the harness resolution.

    Against a harness-owned reference the gate additionally never drops below
    ``REFERENCE_TOL_FACTOR`` times the reference's own error -- otherwise a solver
    more accurate than the answer key would fail for being right."""
    tol = float(problem.get("pde_l2_tol", PDE_L2_TOL))
    return max(tol, REFERENCE_TOL_FACTOR * _pde_reference_error(problem))


def _functional_tol(problem, name=None):
    """The declared *relative* gate on a quantity of interest. Defaults to the
    problem's own field gate, so a problem scored only on a number needs one
    tolerance, not two -- ``_pde_l2_tol`` for a PDE, the terminal-mean gate for an
    SDE, because on each side that is the gate the problem would otherwise be held
    to. The two defaults are far apart (1% against 5%), which is the point: a
    sampled moment is not a resolved field, and inheriting the wrong one would
    silently grade a Monte-Carlo price at a field solver's tolerance.

    ``functional_tol`` may be one number or ``{name: tol}``, the same two shapes
    ``reference_abs_err`` takes. The per-value form is for a problem whose values
    differ in how precisely they can be *computed* -- distinct from the floor
    below, which is about how precisely they are *known*. A Monte-Carlo price
    carries a standard error that is a fixed fraction of the payoff's spread, so a
    deep out-of-the-money value thirty times smaller than its sibling has thirty
    times its relative noise at the same path count: one relative gate across both
    either fails the small one or stops testing the large one.

    With no name, returns the loosest declared gate -- a headline for the record,
    never the thing a value is graded against. That is ``functional_tol_used``."""
    default = SDE_MEAN_TOL if problem.get("type") == "sde" else _pde_l2_tol(problem)
    declared = problem.get("functional_tol", default)
    if isinstance(declared, dict):
        if name is not None:
            return float(declared.get(name, default))
        return float(max(declared.values())) if declared else float(default)
    return float(declared)


def _functional_reference_abs_err(problem, name):
    """One reference value's own *absolute* uncertainty, or 0.0 if it has none.

    ``reference_abs_err`` is the functional counterpart of ``reference_error``, and the
    two are deliberately different shapes. A field reference's disagreement is naturally
    relative -- one number over a whole grid -- while a scalar quantity of interest is
    known to an *absolute* precision: a published figure rounded to a digit, a quadrature
    bound, a Monte-Carlo standard error. Declared as one number covering every functional,
    or as ``{name: abs_err}`` when they differ, which they do whenever the reference came
    from sampling: plan-benchop-slv.md §2c measured absolute errors spanning 10x across
    three values of one problem while their relative errors spanned 3x."""
    ra = problem.get("reference_abs_err")
    if isinstance(ra, dict):
        return float(ra.get(name, 0.0) or 0.0)
    return float(ra or 0.0)


def _functional_tols(problem):
    """The gate actually applied to each quantity of interest, keyed by name:

        tol_i = max(functional_tol, REFERENCE_TOL_FACTOR * reference_abs_err_i / |ref_i|)

    The gate stays **relative** -- the decision in plan-benchop-slv.md §4, taken because
    relative error was measured to be the more uniform of the two across a problem whose
    values span 30x -- but an absolute uncertainty on the answer key does not convert to
    a single relative number, and it is the small values where it bites. A reference
    published as ``37.52 +/- 0.005`` is known to 1.3e-4 relative; the same +/-0.005 on a
    value of 0.0393 is known to 0.13, and a solver graded at 1e-3 there would be failed
    for missing a digit the answer key does not have.

    This is the rule ``_pde_l2_tol`` already applies to a field reference and
    ``_score_sde_moments`` to a Monte-Carlo standard error, with the same factor, taken
    per value because per value is where the information is. It never tightens: the
    ``max`` leaves the gate exactly where a problem with a precise reference put it."""
    tols = {}
    for name, ref in (problem.get("functional_truth") or {}).items():
        floor = (REFERENCE_TOL_FACTOR * _functional_reference_abs_err(problem, name)
                 / max(abs(float(ref)), 1e-12))
        tols[name] = max(_functional_tol(problem, name), floor)
    return tols


def _pde_order_check(problem):
    """Whether to run the 2-grid order check: on by default, needs *something* to
    measure an error against, and can be turned off per problem (e.g. a shock, where
    the L2 order is inherently fractional/ill-defined).

    A field truth is one such thing; a declared ``functional_truth`` is another, and
    used to be missed. ``_pde_eval_run`` already makes the quantity of interest the
    primary error when it is the only truth, so the two-grid machinery below works
    on it unchanged -- but this predicate asked for a *field* and so skipped the
    check entirely on exactly the problems that need it most. Measured on a
    functional-only problem before the fix: a solver returning a hardcoded constant
    scored 10, with ``order_checked`` false and no observed order recorded."""
    return bool(problem.get("pde_order_check", True)) and (
        _pde_truth(problem) is not None or bool(problem.get("functional_truth")))


def _pde_functional_only(problem):
    """A problem whose entire ground truth is a handful of numbers. That is the one
    shape where a solver can hold the answer without having solved anything, so it
    is the one shape where the two grids must be required to *disagree*."""
    return bool(problem.get("functional_truth")) and _pde_truth(problem) is None


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
    """Execute solve_pde at the resolution(s) needed. Returns ``(runs, timings, err)``
    where ``runs`` is a list of ``(N_or_None, result_dict)`` (one element for
    single-grid / legacy, two for the order check), ``timings`` is the parallel list
    of per-run clocks, and ``err`` is a status dict or None.

    ``timings`` is empty on the inline path, which is deliberate rather than an
    omission: the sandboxed and inline routes are different measurement conditions
    (a fresh interpreter per solve against a warm in-process one), so a time from
    one is not comparable with a time from the other. Recording nothing is honest;
    recording an inline number under the same key would quietly invite the
    comparison. ``_fold_timing`` stamps the path onto whatever is recorded."""
    order = _pde_order_check(problem)
    grid_N = _pde_grid_N(problem)

    if sandbox:
        import runner

        def clock(run, N):
            return {"N": N, "solve_s": run.get("solve_s"), "wall_s": run.get("wall_s")}

        run1 = runner.run_solver(problem=problem, plan_dir=plan_dir,
                                 timeout_s=timeout_s, mem_mb=mem_mb, pde_N=grid_N)
        if run1["status"] != "ok":
            return None, None, _crashed(run1, plan_dir)
        accepted = bool(run1.get("accepted_resolution"))
        N1 = grid_N if accepted else None
        runs, timings = [(N1, run1["result"])], [clock(run1, N1)]
        if accepted and order:
            run2 = runner.run_solver(problem=problem, plan_dir=plan_dir,
                                     timeout_s=timeout_s, mem_mb=mem_mb, pde_N=2 * grid_N)
            if run2["status"] != "ok":
                return None, None, _crashed(run2, plan_dir)
            runs.append((2 * grid_N, run2["result"]))
            timings.append(clock(run2, 2 * grid_N))
        return runs, timings, None

    # Inline (trusted pipeline path): import and call directly.
    mod = import_solver(plan_dir)
    if not hasattr(mod, "solve_pde"):
        return None, None, {"status": "error", "error": "solver.py has no solve_pde()"}
    mode = _pde_call_mode(mod.solve_pde)
    if mode is None:
        return None, None, {"status": "error",
                            "error": "solve_pde signature is neither solve_pde(N) nor solve_pde()"}
    if mode == "noarg":
        return [(None, mod.solve_pde())], [], None
    Ns = [grid_N, 2 * grid_N] if order else [grid_N]
    return [(N, _run_pde(mod.solve_pde, N, mode)) for N in Ns], [], None


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


def _eigen_normalize(un, ue, metric, mask):
    """Put an eigenfunction pair in a common gauge before comparing them.

    An eigenfunction is defined only up to a nonzero scalar multiple, so a correct
    solver's field can differ from the reference by an arbitrary amplitude and sign
    and still be exactly right. Normalize each to unit norm over the in-domain nodes
    and align the sign on the largest-magnitude reference node (which is far from any
    zero of the mode, so the choice is stable under refinement)."""
    sel = slice(None) if mask is None else mask
    nn, ne = _field_norm(un, metric, mask), _field_norm(ue, metric, mask)
    if nn <= 0.0 or ne <= 0.0:
        return un, ue
    un, ue = un / nn, ue / ne
    k = int(np.argmax(np.abs(np.asarray(ue)[sel])))
    if float(np.asarray(un)[sel].ravel()[k]) * float(np.asarray(ue)[sel].ravel()[k]) < 0.0:
        un = -un
    return un, ue


def _functional_errs(problem, result):
    """Relative *and* absolute error of each scalar quantity of interest the solver
    reported, as ``(rel_errs, abs_errs, err_status)``.

    The other no-closed-form route: for an eigenvalue, an extremum, or a drag
    coefficient the truth is a *number*, authored in ``problems.py`` (and, per
    NO_CLOSED_FORM_CANDIDATES.md §3, computed here rather than quoted from a paper,
    so a recalled constant is not a passing answer).

    Both errors are returned because they answer different questions and only one of
    them grades. The relative error is what the gate is stated in and what
    ``_pde_converged`` reads. The absolute error is what an externally published
    benchmark quotes its own values and its own tolerances in, so a result cannot be
    compared against one without it -- and it cannot be recovered afterwards from the
    relative number alone, since the record does not carry the reference value.

    A problem that declares ``functional_truth`` and gets no ``functionals`` back has
    violated the contract, which is a failure of that solver, not an inconclusive."""
    truth = problem.get("functional_truth")
    if not truth:
        return {}, {}, None
    got = result.get("functionals")
    # A missing dict or a missing name is a *failure*: the contract told the solver
    # to return this number, and a solution without the number is not a solution
    # to the problem as posed (findings §7.7). A value that is present but not a
    # real number stays `inconclusive` -- that is a type slip, not a refusal.
    if not isinstance(got, dict):
        return None, None, {"status": "contract", "passed": False, "verified_score": 1,
                            "rel_l2_err": float("inf"), "primary_err": float("inf"),
                            "error": "solver returned no 'functionals' dict; this problem is "
                                     f"scored on {sorted(truth)}, and a solver told to return "
                                     f"them and not doing so has failed"}
    errs, abs_errs = {}, {}
    for name, ref in truth.items():
        if name not in got:
            return None, None, {"status": "contract", "passed": False, "verified_score": 1,
                                "rel_l2_err": float("inf"), "primary_err": float("inf"),
                                "error": f"solver's 'functionals' is missing {name!r}; this "
                                         f"problem is scored on it"}
        try:
            val = float(got[name])
        except (TypeError, ValueError):
            return None, None, {"status": "inconclusive",
                                "error": f"functional {name!r} is not a real number: "
                                         f"{got[name]!r}"}
        if not np.isfinite(val):
            return None, None, {"status": "nonfinite", "passed": False, "verified_score": 1,
                                "rel_l2_err": float("inf"), "primary_err": float("inf")}
        abs_errs[name] = abs(val - float(ref))
        errs[name] = abs_errs[name] / (abs(float(ref)) + 1e-14)
    return errs, abs_errs, None


def _functional_values(result, names):
    """The raw numbers the solver reported, for the names ``_functional_errs`` has
    already validated. Kept beside the errors because two different values can have
    the same absolute error -- one above the reference and one below -- so "did this
    move between the grids?" cannot be answered from the errors alone."""
    got = result.get("functionals") or {}
    return {n: float(got[n]) for n in names}


def _pde_eval_run(problem, result):
    """Relative error of one solve against the ground truth, per field, plus the
    structure diagnostics. Returns ``(eval_dict, err)`` with ``err`` a status dict
    or None. Handles any spatial dimension (via ``meshgrid(*axes)``), scalar or
    multi-field output, an optional in-domain mask, and the L1 or L2 metric.

    The truth is whichever of ``analytic`` (closed form) or ``reference``
    (harness-owned high-resolution solution) the problem carries -- everything below
    is identical for the two. A problem may additionally, or instead, be scored on
    ``functional_truth``; when it is the *only* truth, the functional error becomes
    the primary error, so the 2-grid order check measures the convergence of the
    quantity of interest itself."""
    grid = result.get("grid", {})
    t_eval = problem["t_eval"]  # trust the benchmark's evaluation time, not the solver label
    metric = problem.get("pde_metric", "l2")
    truth_fn = _pde_truth(problem)

    axes = _pde_axes(problem, grid)
    if axes is None:
        return None, {"status": "inconclusive",
                      "error": f"could not extract {problem['dims']}-D axes from grid keys "
                               f"{list(grid) if isinstance(grid, dict) else grid}"}
    coords = np.meshgrid(*axes, indexing="ij")  # d-generic; for d=1 this is [x]
    mask = _domain_mask(problem, coords)

    func_errs, func_abs, ferr = _functional_errs(problem, result)
    if ferr is not None:
        return None, ferr

    if truth_fn is None:
        # Functional-only problem: no field truth exists, so the QoI carries the score.
        primary = problem.get("primary_functional") or sorted(func_errs)[0]
        # ``max_abs_err`` is the largest *absolute* deviation, the same quantity the
        # field branch below records under that name. It used to hold the primary
        # *relative* error here, so the one field named for an absolute error meant two
        # incompatible things depending on which truth route a problem took -- harmless
        # while nothing read it, and a silent conflation the moment anything did
        # (plan-benchop-slv.md §3).
        ev = {"rel_l2_err": func_errs[primary], "primary_err": func_errs[primary],
              "primary_field": primary, "metric": "functional", "field_errs": {},
              "functional_errs": func_errs, "functional_abs_errs": func_abs,
              "functional_values": _functional_values(result, func_errs),
              "max_abs_err": max(func_abs.values()),
              "grid_shape": [len(a) for a in axes], "diagnostics": [],
              "t_final": result.get("t_final")}
        return ev, None

    try:
        num, field_names = _field_num(result, problem)
        exact = truth_fn(t_eval, *coords)
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
    # An eigenproblem's fields are defined up to a scalar multiple, which is the same
    # idea one gauge wider.
    gauge = set(problem.get("gauge_fields", ()))
    eigen = bool(problem.get("normalize_fields", False))
    shifted = {}
    for name, (un, ue) in pairs.items():
        if name in gauge:
            sel = slice(None) if mask is None else mask
            un = un - float(np.mean(un[sel]))
            ue = ue - float(np.mean(ue[sel]))
        if eigen:
            un, ue = _eigen_normalize(un, ue, metric, mask)
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
    # Reported, never gated. Measured on the raw pair, except under an eigen gauge
    # where the raw amplitudes are arbitrary and the difference would be meaningless.
    for_max = shifted if eigen else pairs
    max_abs = max(float(np.max(np.abs((un - ue) if mask is None else (un - ue)[mask])))
                  for un, ue in for_max.values())
    diagnostics = _run_diagnostics(problem, {n: p[0] for n, p in pairs.items()},
                                   {n: p[1] for n, p in pairs.items()}, axes, mask)

    ev = {"rel_l2_err": field_errs[primary], "primary_err": field_errs[primary],
          "primary_field": primary, "metric": metric, "field_errs": field_errs,
          "max_abs_err": max_abs, "grid_shape": list(pairs[primary][0].shape),
          "diagnostics": diagnostics, "t_final": result.get("t_final")}
    if func_errs:
        ev["functional_errs"] = func_errs
        ev["functional_abs_errs"] = func_abs
        ev["functional_values"] = _functional_values(result, func_errs)
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
    """Accuracy gate: every *required* field within the (primary-metric) tolerance,
    and every declared quantity of interest within its own."""
    tol = _pde_l2_tol(problem)
    required = problem.get("required_fields") or list(ev["field_errs"])
    ok = all(ev["field_errs"][f] < tol for f in required)
    # Per value, not one number for all of them: the gate on a quantity of interest is
    # raised where the reference itself is uncertain, and that is a property of the
    # individual value rather than of the problem.
    ftols, ftol = _functional_tols(problem), _functional_tol(problem)
    ok = ok and all(e < ftols.get(name, ftol)
                    for name, e in ev.get("functional_errs", {}).items())
    return bool(ok), tol


def _fold_reference(out, problem, ev):
    """Record what the score rested on: the quantity-of-interest errors, and -- for a
    harness-owned reference -- the answer key's own error and the tolerance widening
    it bought, so a near-threshold result can be read without re-deriving them."""
    if ev.get("functional_errs"):
        out["functional_errs"] = ev["functional_errs"]
        out["functional_tol"] = _functional_tol(problem)
        if ev.get("functional_abs_errs"):
            out["functional_abs_errs"] = ev["functional_abs_errs"]
        # The effective gate is always recorded, even when it equals the declared one:
        # a reader of the record should not have to know the widening rule, or go back
        # to problems.py, to see what a value was actually held to.
        out["functional_tol_used"] = _functional_tols(problem)
        widened = sorted(n for n, t in out["functional_tol_used"].items()
                         if t > out["functional_tol"])
        if widened:
            out["functional_tol_widened"] = widened
            out["reference_abs_err"] = {n: _functional_reference_abs_err(problem, n)
                                        for n in out["functional_tol_used"]}
    ref_err = _pde_reference_error(problem)
    if ref_err > 0.0:
        out["reference_error"] = ref_err
        out["tol_used"] = _pde_l2_tol(problem)
        out["tol_widened"] = bool(out["tol_used"] > float(problem.get("pde_l2_tol", PDE_L2_TOL)))
    return out


def _fold_timing(out, timings):
    """Record what the solve cost, for the runs where it was actually measured.

    ``solve_s`` is the ``solve_pde`` call alone and ``wall_s`` the whole subprocess,
    so their difference is the harness's own overhead -- on a fast problem that is
    most of ``wall_s``, which is why the process clock alone cannot carry a
    work-precision axis. The headline pair is the *graded* run (the finest grid,
    the one ``rel_l2_err`` comes from) with the per-grid series beside it, mirroring
    how ``rel_l2_err`` sits beside ``grid_errors``. ``path`` names the measurement
    condition, because a sandboxed time and an inline one must never be compared
    (plan-benchop-slv.md §6, step 2)."""
    if not timings:
        return out
    out["timing"] = {"path": "sandbox", "solve_s": timings[-1]["solve_s"],
                     "wall_s": timings[-1]["wall_s"], "runs": timings}
    return out


def _pde_functional_constant(problem, evals):
    """Quantities of interest that came back *identical* at both resolutions.

    A number that does not move when the grid is halved did not come off that grid.
    On the field route this cannot happen by accident, and the field is its own
    defence -- nobody memorizes a surface. On the functional-only route the whole
    answer is two or three numbers, and on an externally published problem those
    numbers have been on the internet since 2018, so the refinement ladder is the
    only thing standing between a recalled constant and a score of 10. This is the
    SDE side's two-level check (step 5) stated on the grid the PDE side already
    runs: there the price is taken at two path counts, here at two resolutions, and
    in both it has to move."""
    if not _pde_functional_only(problem) or len(evals) < 2:
        return []
    lo = evals[0].get("functional_values") or {}
    hi = evals[1].get("functional_values") or {}
    return sorted(n for n, v in hi.items() if n in lo and lo[n] == v)


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
        return _fold_reference(_fold_diagnostics(out, ev), problem, ev)

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
    # Checked after the order, and separately from it: the waiver above hands a
    # perfect ``order_ok`` to anything whose error is tiny at both grids, which is
    # exactly what a hardcoded constant looks like.
    constant = _pde_functional_constant(problem, evals)
    passed = bool(converged and order_ok and not constant)
    out = {"status": "ok", "passed": passed,
           "verified_score": 1 if constant else (
               pde_score(e1) if order_ok else min(pde_score(e1), 5)),
           "rel_l2_err": e1, "max_abs_err": evals[1]["max_abs_err"],
           "grid_shape": evals[1]["grid_shape"], "metric": evals[1]["metric"],
           "primary_field": evals[1]["primary_field"], "t_eval": t_eval,
           "solver_t_final": evals[1]["t_final"], "order_checked": True,
           "observed_order": p_hat, "order_min": min_order, "order_ok": order_ok,
           "converged": converged, "grid_N": N0, "fine_grid_N": N1,
           "grid_errors": {str(N0): e0, str(N1): e1}}
    if _pde_functional_only(problem):
        # Recorded either way. A passing record that is silent about the check reads
        # as one that had it and cleared it.
        out["functional_constant"] = constant
    return _fold_reference(_fold_diagnostics(out, evals[1]), problem, evals[1])


def verify_pde(problem, plan_dir, sandbox=False, timeout_s=DEFAULT_SOLVER_TIMEOUT_S,
               mem_mb=DEFAULT_SOLVER_MEM_MB, work_precision=True):
    runs, timings, err = _produce_pde_runs(problem, plan_dir, sandbox, timeout_s, mem_mb)
    if err is not None:
        return err
    out = _fold_timing(_score_pde_runs(problem, runs), timings)
    out = _fold_work_precision_pde(out, problem, plan_dir, timings,
                                   extra_levels=bool(work_precision and sandbox),
                                   timeout_s=timeout_s, mem_mb=mem_mb)
    # The finest run, carried to the ledger re-audit and dropped before the record
    # is written -- it holds full solution arrays and has no place in results.json.
    out["_run_result"] = runs[-1][1]
    return out


def verify_pde_reference(problem, plan_dir, sandbox=False, timeout_s=DEFAULT_SOLVER_TIMEOUT_S,
                         mem_mb=DEFAULT_SOLVER_MEM_MB, work_precision=True):
    """A PDE with no closed form, scored against a harness-owned reference field.

    The scoring path is `verify_pde`'s, unchanged: ``_pde_eval_run`` takes the truth
    from ``problem["reference"]`` instead of ``problem["analytic"]`` and every
    per-field error, mask, metric and diagnostic below that is identical. What this
    wrapper adds is the *dispatch* -- ``plan-no-closed-form.md`` §9d says
    ``benchmark/verify.py`` needs "No change" while §8c requires this path; §8c is
    right, and the disagreement is the reason it is named rather than folded in --
    and the guarantee that the reference actually carries a measured error, since a
    reference whose accuracy nobody established is an answer key, not ground truth."""
    if problem.get("reference") is None:
        return {"status": "error",
                "error": f"{problem['slug']}: ground_truth_kind='reference' but no "
                         "reference callable is authored in problems.py"}
    if _pde_reference_error(problem) <= 0.0:
        return {"status": "error",
                "error": f"{problem['slug']}: reference has no measured reference_error; "
                         "run benchmark/make_references.py and record the two-scheme "
                         "agreement before scoring against it"}
    return verify_pde(problem, plan_dir, sandbox=sandbox, timeout_s=timeout_s,
                      mem_mb=mem_mb, work_precision=work_precision)


# --- work-precision sweep ---------------------------------------------------
# A work-precision diagram plots error against the time that error cost. It is how
# the numerical-analysis literature -- BENCHOP included -- compares methods, because
# "more accurate" is not a claim until "at what cost" is attached: a scheme that wins
# at a loose tolerance routinely loses at a tight one, and the crossing point is the
# result. Two things make the first points nearly free here. The PDE path already
# solves at N and 2N for the observed-order check, and since step 1 it already times
# each of those solves -- so the first two (error, time) pairs cost no extra compute
# at all, only the join of two records the harness had been keeping separately.
# Beyond that a problem may declare ``work_precision_levels`` and pay for the rest.
#
# Declared levels, both optional and both on the problem dict:
#   PDE: a list of resolutions, e.g. ``[32, 64, 128, 256]``. Levels already covered
#        by the graded ladder are reused rather than re-solved.
#   SDE: a list of ``(num_paths, dt)`` pairs, e.g. ``[(20000, 0.02), (80000, 0.005)]``,
#        ordered on emission by the work they ask for (paths x steps). The seed is
#        held fixed across the sweep; moving the sample and the work together would
#        measure two things at once.
#
# Three rules keep the series honest, and each one is a test:
#   * It never moves a verdict. The curve is measurement taken after the grade is
#     decided. An extra level that crashes, times out, is ignored by the solver or
#     cannot be scored is recorded under ``skipped`` with its reason, and the verdict
#     stands exactly as the graded ladder set it.
#   * Sandbox only. An inline time and a sandboxed time are different measurement
#     conditions (step 2), so the inline path emits no series rather than one that
#     invites the comparison.
#   * Measured, never modelled. A point exists only where a real solve produced both
#     an error and a clock. Nothing is interpolated and no slope is fitted here; the
#     series is emitted raw, because a fitted exponent would be a claim and the
#     points are evidence.


def _wp_declared(problem):
    return problem.get("work_precision_levels") or []


def _wp_pde_levels(problem):
    """Declared extra resolutions, de-duplicated and ascending."""
    return sorted({int(n) for n in _wp_declared(problem) if int(n) > 0})


def _wp_sde_levels(problem):
    """Declared ``(num_paths, dt)`` pairs, ordered by the work each asks for --
    paths x steps, which is the sweep's own cost axis and so the order a reader
    expects the series in."""
    T = float(problem["T"])
    out = set()
    for item in _wp_declared(problem):
        n, dt = int(item[0]), float(item[1])
        if n > 0 and dt > 0:
            out.add((n, dt))
    return sorted(out, key=lambda nd: nd[0] * max(round(T / nd[1]), 1))


def _wp_point(level, rel_err, clock, graded):
    """One point on the curve: where it was taken, what it got wrong, what it cost."""
    pt = dict(level)
    pt["rel_err"] = float(rel_err)
    pt["solve_s"] = (clock or {}).get("solve_s")
    pt["wall_s"] = (clock or {}).get("wall_s")
    pt["graded"] = bool(graded)
    return pt


def _wp_series(axis, points, skipped):
    series = {"axis": axis, "path": "sandbox", "points": points}
    if skipped:
        series["skipped"] = skipped
    return series


def _wp_solve_pde(problem, plan_dir, N, timeout_s, mem_mb):
    """One extra PDE point: solve at ``N``, score it, return ``(point, None)`` or
    ``(None, reason)``.

    Every failure path returns a reason instead of raising, including the blanket
    ``except``. That is not defensive habit: this runs *after* the verdict is
    decided, so an exception escaping here would convert a graded PASS into an
    ``error`` status on the strength of a measurement the verdict does not use."""
    import runner
    try:
        run = runner.run_solver(problem=problem, plan_dir=plan_dir,
                                timeout_s=timeout_s, mem_mb=mem_mb, pde_N=N)
        if run["status"] != "ok":
            return None, run.get("reason", "crashed")
        if not run.get("accepted_resolution"):
            # A legacy ``solve_pde()`` ignores N, so every level would return the
            # same solve: one point drawn four times, which reads as a flat curve
            # rather than as the absent measurement it is.
            return None, "fixed_resolution"
        ev, err = _pde_eval_run(problem, run["result"])
        if err is not None:
            return None, err.get("status", "unscorable")
        if not np.isfinite(ev["primary_err"]):
            return None, "nonfinite"
        clock = {"solve_s": run.get("solve_s"), "wall_s": run.get("wall_s")}
        return _wp_point({"N": N}, ev["primary_err"], clock, False), None
    except Exception as exc:  # noqa: BLE001 -- measurement must not rewrite a verdict
        return None, type(exc).__name__


def _wp_sde_err(scored):
    """The SDE error axis: the worst relative error the verdict rests on.

    The gate is every component's mean and variance inside tolerance *and* every
    quantity of interest inside its own, so the curve is the max over all three and
    describes the solver that was graded rather than a differently-summarized one.
    The functional term matters more than it looks: a problem scored only on a price
    has no components at all, and without it the axis would be a constant zero and
    the curve would read as a perfect solver at every level. ``None`` where there is
    no error to plot: the stability route, or a run that did not score."""
    if scored.get("status") != "ok" or "worst_variance_rel_err" not in scored:
        return None
    v = max(float(scored["worst_variance_rel_err"]),
            float(scored.get("worst_mean_rel_err") or 0.0),
            float(scored.get("worst_functional_rel_err") or 0.0))
    return v if np.isfinite(v) else None


def _wp_solve_sde(problem, plan_dir, num_paths, dt, timeout_s, mem_mb, reference):
    """One extra SDE point, with the same never-raise contract as its PDE twin."""
    import runner
    try:
        run = runner.run_solver(problem=problem, plan_dir=plan_dir,
                                timeout_s=timeout_s, mem_mb=mem_mb,
                                sde_num_paths=num_paths, sde_dt=dt)
        if run["status"] != "ok":
            return None, run.get("reason", "crashed")
        # A solver with a hardcoded path count would otherwise contribute the same
        # error at every level and draw a flat curve -- the exact failure the ladder
        # exists to catch on the PDE side, so it is caught here too.
        got = np.asarray(run["result"].get("terminal_paths", []), dtype=float)
        if got.size and got.size // max(int(problem["state_dimension"]), 1) != num_paths:
            return None, "level_ignored"
        comps, meta, _res, err = _run_and_read_sde(
            problem, plan_dir, injected=(run["result"], run["run_meta"]))
        if err is not None:
            return None, err.get("status", "unscorable")
        scored = _score_sde_moments(problem, comps, _sde_moment_truth(problem),
                                    meta, reference, result=run["result"])
        rel = _wp_sde_err(scored)
        if rel is None:
            return None, "unscorable"
        clock = {"solve_s": run.get("solve_s"), "wall_s": run.get("wall_s")}
        return _wp_point({"num_paths": num_paths, "dt": dt}, rel, clock, False), None
    except Exception as exc:  # noqa: BLE001 -- measurement must not rewrite a verdict
        return None, type(exc).__name__


def _fold_work_precision_pde(out, problem, plan_dir, timings, *, extra_levels,
                             timeout_s, mem_mb):
    """Join the graded ladder's errors to its clocks, then add the declared levels.

    The graded points cost nothing: ``_score_pde_runs`` already recorded a
    per-resolution error in ``grid_errors`` and ``_produce_pde_runs`` already
    recorded a per-resolution clock, and until now the two sat in the record with no
    way to read one against the other. ``timings`` is empty on the inline path, which
    is what makes the whole series sandbox-only without a flag for it.

    The guards sit inside the ``try`` with everything else, so that nothing in the
    series -- not even deciding whether to build one -- can raise into a decided
    verdict."""
    try:
        if out.get("status") != "ok" or not timings:
            return out
        if not np.isfinite(out.get("rel_l2_err", np.inf)):
            return out
        by_N = {t["N"]: t for t in timings}
        points, skipped = [], []
        grid_errs = out.get("grid_errors")
        if grid_errs:
            for key, e in grid_errs.items():
                N = int(key)
                points.append(_wp_point({"N": N}, e, by_N.get(N), True))
        else:
            N = out.get("grid_N")
            points.append(_wp_point({"N": N}, out["rel_l2_err"], by_N.get(N), True))
        graded = {p["N"] for p in points}
        if extra_levels:
            for N in _wp_pde_levels(problem):
                if N in graded:
                    continue
                pt, why = _wp_solve_pde(problem, plan_dir, N, timeout_s, mem_mb)
                (points if pt else skipped).append(pt or {"N": N, "reason": why})
        points.sort(key=lambda p: -1 if p["N"] is None else p["N"])
        out["work_precision"] = _wp_series("N", points, skipped)
    except Exception as exc:  # noqa: BLE001 -- see _wp_solve_pde
        out["work_precision"] = {"axis": "N", "path": "sandbox", "points": [],
                                 "error": f"{type(exc).__name__}: {exc}"}
    return out


def _wp_sde_free_level(problem, out):
    """The second point the SDE curve has had since step 5 and was not plotting.

    A functional-scored problem is already re-solved at ``functional_level_paths``
    to prove its answer is an estimate rather than a constant -- same sandbox, same
    step, a quarter of the paths -- and the harness times that solve. That is a real
    (error, time) pair the grade has already paid for. Step 9 measured what skipping
    it cost: every problem in the externally anchored suite emitted a one-point
    "curve", and since none of them declares ``work_precision_levels``, a step-10
    sweep would have recorded no work-precision information at all.

    It is folded in only where the graded point's error is the same quantity. A
    problem with moment gates plots max(mean, variance, functional) and the coarse
    level scores the functional alone, so there the point is reported as skipped
    rather than drawn against a differently-summarized error.

    Returns ``(point, None)`` or ``(None, reason)``, and never raises -- the caller
    is measurement taken after the verdict."""
    lv = out.get("functional_levels") or {}
    if not problem.get("functional_truth") or lv.get("status") != "ok":
        return None, None  # nothing to say: this problem has no second level
    n_lo, dt = lv.get("num_paths_coarse"), lv.get("dt")
    if problem.get("moments") is not None:
        return None, "the coarse level scores only the functional"
    if lv.get("worst_functional_rel_err") is None or lv.get("solve_s") is None:
        return None, "coarse level not scored or not timed"
    return _wp_point({"num_paths": int(n_lo), "dt": float(dt)},
                     lv["worst_functional_rel_err"],
                     {"solve_s": lv.get("solve_s"), "wall_s": lv.get("wall_s")},
                     False), None


def _fold_work_precision_sde(out, problem, plan_dir, graded_run, *, extra_levels,
                             timeout_s, mem_mb, reference):
    """The SDE series: the graded (num_paths, dt) as its anchor, the coarse level of
    the two-level check where there is one, plus declared levels.

    ~~Unlike the PDE side there is no free second point~~ -- **[rev, step 9]** there
    is one wherever a problem is scored on a functional, and it is the same kind of
    reuse the PDE side makes of its order-check ladder. Beyond those two, a curve
    costs solves."""
    try:
        rel = _wp_sde_err(out)
        run_meta = out.get("run") or {}
        if rel is None or run_meta.get("solve_s") is None:
            return out
        clock = {"solve_s": run_meta.get("solve_s"),
                 "wall_s": (graded_run or {}).get("wall_s")}
        anchor = (int(run_meta["num_paths"]), float(run_meta["dt"]))
        points = [_wp_point({"num_paths": anchor[0], "dt": anchor[1]}, rel, clock, True)]
        skipped = []
        free, why = _wp_sde_free_level(problem, out)
        if free:
            points.append(free)
        elif why:
            skipped.append({"num_paths": (out.get("functional_levels") or {})
                            .get("num_paths_coarse"), "dt": anchor[1], "reason": why})
        if extra_levels:
            have = {(p["num_paths"], p["dt"]) for p in points}
            for n, dt in _wp_sde_levels(problem):
                if (n, dt) in have:
                    continue
                pt, why = _wp_solve_sde(problem, plan_dir, n, dt, timeout_s, mem_mb,
                                        reference)
                (points if pt else skipped).append(
                    pt or {"num_paths": n, "dt": dt, "reason": why})
        T = float(problem["T"])
        points.sort(key=lambda p: p["num_paths"] * max(round(T / p["dt"]), 1))
        out["work_precision"] = _wp_series("paths_dt", points, skipped)
    except Exception as exc:  # noqa: BLE001 -- see _wp_solve_pde
        out["work_precision"] = {"axis": "paths_dt", "path": "sandbox", "points": [],
                                 "error": f"{type(exc).__name__}: {exc}"}
    return out


# --- dispatch ---------------------------------------------------------------


def verify_problem(problem, workspace_dir, plan_dir=None, sandbox=False,
                   timeout_s=DEFAULT_SOLVER_TIMEOUT_S, mem_mb=DEFAULT_SOLVER_MEM_MB,
                   work_precision=True):
    """Independently verify one solved problem. Returns a result dict with a
    ``status`` field: no_ground_truth | ok | inconclusive | nonfinite | contract |
    error | crashed. ``contract`` is a solver that ran but did not return what the
    problem scores on (a declared functional); like ``nonfinite`` it is a failure
    of that solver, never an inconclusive.

    Dispatch is on ``ground_truth_kind`` (default ``exact``): ``reference`` compares
    against a discretization-free MC reference within an SE-aware tolerance (SDE) or
    a harness-owned high-resolution reference field within a tolerance widened by the
    reference's own error (PDE); ``stability`` checks only that a correct scheme stays
    finite / in-domain (and is checkable even though ``has_ground_truth`` is False).

    ``functional_truth`` is orthogonal to the kind rather than a kind of its own: any
    PDE problem may add scalar quantities of interest to its gate, and a problem whose
    *only* truth is a functional simply has no ``analytic`` and no ``reference``.

    ``sandbox=True`` runs ``solver.py`` in an isolated subprocess (wall-clock
    timeout, memory cap, import policy) via ``runner.py`` and scores the returned
    output. A sandbox failure -- timeout, out-of-memory, blocked import, wrong
    signature, or a return value that violates the contract -- yields
    ``status="crashed"`` with a ``reason``; it is a genuine failure of that solver,
    never an "unverified" of the benchmark.

    **Both callers now pass it.** ``oneshot.py`` always did; ``run.py`` does since
    2026-10-17, because a pipeline graded inline and a baseline graded sandboxed are
    measured under different conditions and neither their times nor their crash
    behaviour are comparable (plan-benchop-slv.md §6, step 2). The argument survives,
    defaulting to ``False``, for callers that drive scoring themselves --
    ``validate_ground_truth.py`` and the tests feed synthetic output and never
    execute a solver through here. Reaching parity took three fixes to what the
    boundary carries: ``functionals``, both result shapes at once, and an absolute
    solver path. All 34 solved workspaces now return the identical verdict either
    way.

    ``work_precision`` gates only the *extra* solves of the work-precision sweep --
    the levels a problem declares in ``work_precision_levels``. The graded ladder's
    own (error, time) points are emitted regardless, because they are already paid
    for. Turning it off makes a re-grade cheap; it cannot change a verdict either
    way, since nothing the sweep measures is an input to one."""
    kind = problem.get("ground_truth_kind", "exact")
    if kind != "stability" and not problem["has_ground_truth"]:
        return {"status": "no_ground_truth",
                "note": "no closed form; pipeline self-score is the only signal"}

    if plan_dir is None:
        plan_dir = best_plan_dir(workspace_dir)
    if plan_dir is None or not os.path.isdir(plan_dir):
        return {"status": "error", "error": f"no plan directory under {workspace_dir}"}

    # Findings F6: the kernel records the hash of the solver.py it scored in the
    # plan's metrics block. A file edited after scoring -- a refine cycle cut short
    # mid-edit -- is not the plan that earned the score, and grading it as if it
    # were is how a wrong 10 or a wrong FAIL ships with neither layer knowing.
    drift = solver_drift(plan_dir)
    if drift:
        return {"status": "error", "error": "solver.py changed after scoring: "
                                            f"scored {drift['scored'][:12]}, on disk "
                                            f"{drift['on_disk'][:12]}",
                "solver_drift": drift, "plan_dir": os.path.relpath(plan_dir, REPO_ROOT)}

    # SDE is a single run, so it is pre-executed here (sandboxed) and injected. PDE
    # may need two runs (the N/2N order check), so verify_pde orchestrates its own
    # sandboxed execution.
    injected = sde_run = None
    if sandbox and problem["type"] == "sde":
        import runner  # local import: keeps the pipeline path free of the dependency
        run = runner.run_solver(problem=problem, plan_dir=plan_dir,
                                timeout_s=timeout_s, mem_mb=mem_mb)
        if run["status"] != "ok":
            return _crashed(run, plan_dir)
        injected, sde_run = (run["result"], run["run_meta"]), run

    try:
        if problem["type"] == "sde":
            if kind == "stability":
                out = verify_sde_stability(problem, plan_dir, injected=injected)
            elif kind == "reference":
                out = verify_sde_reference(problem, plan_dir, injected=injected)
            else:
                out = verify_sde(problem, plan_dir, injected=injected)
        elif kind == "reference":
            out = verify_pde_reference(problem, plan_dir, sandbox=sandbox,
                                       timeout_s=timeout_s, mem_mb=mem_mb,
                                       work_precision=work_precision)
        else:
            out = verify_pde(problem, plan_dir, sandbox=sandbox,
                             timeout_s=timeout_s, mem_mb=mem_mb,
                             work_precision=work_precision)
        # Both SDE folds happen here rather than inside the three verify_sde* entry
        # points, for the same reason the graded SDE run is executed here: the
        # sandbox parameters live at this level, and the SDE side grades a single
        # pre-executed run while the PDE side orchestrates its own. The order is
        # not arbitrary -- the two-level check is a gate and runs first, the
        # work-precision series is measurement taken after the verdict is settled.
        if sde_run is not None:
            out = _fold_sde_functional_levels(
                out, problem, plan_dir, sde_run.get("result"),
                timeout_s=timeout_s, mem_mb=mem_mb)
        elif problem["type"] == "sde" and problem.get("functional_truth") \
                and isinstance(out, dict) and out.get("status") == "ok":
            # Inline, the harness cannot re-solve at a second path count, so the
            # anti-hardcoding defence is simply absent. It is recorded as absent:
            # a passing record with nothing in this field would read as a record
            # that had the defence and cleared it.
            out["functional_levels"] = {"status": "not_run", "reason": "inline"}
        if sde_run is not None:
            out = _fold_work_precision_sde(
                out, problem, plan_dir, sde_run,
                extra_levels=bool(work_precision), timeout_s=timeout_s,
                mem_mb=mem_mb, reference=(kind == "reference"))
    except OracleLeak as exc:
        return {"status": "error", "error": f"answer-key leakage: {exc}",
                "leakage": True, "plan_dir": os.path.relpath(plan_dir, REPO_ROOT)}
    except Exception as exc:  # noqa: BLE001 -- verification must never crash the runner
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                "plan_dir": os.path.relpath(plan_dir, REPO_ROOT)}
    out["plan_dir"] = os.path.relpath(plan_dir, REPO_ROOT)
    out.update(ledger_audit(workspace_dir, plan_dir, out))
    out.pop("_run_result", None)
    return out


def solver_sha256(plan_dir):
    import hashlib
    try:
        with open(os.path.join(plan_dir, "solver.py"), "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def scored_solver_sha256(plan_dir):
    """The ``solver_sha256`` the kernel wrote into the plan's ``<metrics>`` block,
    or ``None`` when the plan predates it or has no SOLUTION.md."""
    path = os.path.join(plan_dir, "SOLUTION.md")
    if not os.path.exists(path):
        return None
    try:
        from verifylib.review import parse_metrics
        with open(path, encoding="utf-8") as fh:
            value = parse_metrics(fh.read()).get("solver_sha256")
    except Exception:  # noqa: BLE001 -- an unreadable block is "not recorded"
        return None
    return str(value).strip() if isinstance(value, str) and value.strip() else None


def solver_drift(plan_dir):
    """``None`` when the file on disk is the one the kernel scored (or no score
    recorded a hash), else ``{"scored": ..., "on_disk": ...}``."""
    scored = scored_solver_sha256(plan_dir)
    if scored is None:
        return None
    on_disk = solver_sha256(plan_dir)
    if on_disk == scored:
        return None
    return {"scored": scored, "on_disk": on_disk}


def ledger_audit(workspace_dir, plan_dir, out):
    """Layer 2 at certification: did the solver obey the spec it was given?

    Output-derived findings are facts about the run and gate -- the score is capped
    and the review names the ledger entry by id and quote. Source-derived findings
    are read out of ``solver.py`` and only ever warn: they miss ``eps = 1e-3 * 10``
    and false-positive on a nondimensionalised solver, and a false hard-fail on a
    correct solver is worse than a missed relaxation because it is invisible as a
    false positive.
    """
    spec_path = os.path.join(workspace_dir, "problem_spec.json")
    if not os.path.exists(spec_path):
        return {}
    try:
        import json as _json

        from verifylib.audit import audit_run
        with open(spec_path) as fh:
            spec = _json.load(fh)
        findings = audit_run(spec, result=out.get("_run_result"),
                             run_meta=out.get("run"), plan_dir=plan_dir)
    except Exception as exc:  # noqa: BLE001 -- an audit must never fail a good run
        return {"ledger_audit_error": f"{type(exc).__name__}: {exc}"}

    gates = [f.as_dict() for f in findings if f.severity == "error"]
    warns = [f.as_dict() for f in findings if f.severity == "warning"]
    result = {}
    if warns:
        result["ledger_warnings"] = warns
    if gates:
        result["ledger_violations"] = gates
        result["score_cap"] = 3
    return result


def _wp_level_str(axis, p):
    return f"N={p['N']}" if axis == "N" else f"{p['num_paths']}p/dt={p['dt']:g}"


def _fmt_work_precision(out):
    """The error-vs-time series, one line, as ``level: error@time`` pairs. A level
    that was asked for and not measured is named with its reason rather than left
    out, so a short curve is never mistaken for a cheap one."""
    wp = out.get("work_precision") or {}
    lines = []
    if wp.get("error"):
        return [f"  work-precision: not recorded ({wp['error']})"]
    if wp.get("points"):
        axis = wp["axis"]
        pts = "  ".join(
            f"{_wp_level_str(axis, p)}: {p['rel_err']:.2e}"
            + ("" if p.get("solve_s") is None else f"@{p['solve_s']:.4g}s")
            + ("" if p.get("graded") else "*")
            for p in wp["points"])
        lines.append(f"  work-precision: {pts}   (* = swept, not graded)"
                     if any(not p.get("graded") for p in wp["points"])
                     else f"  work-precision: {pts}")
    if wp.get("skipped"):
        why = ", ".join(f"{_wp_level_str(wp['axis'], sk)}: {sk['reason']}"
                        for sk in wp["skipped"])
        lines.append(f"  not measured  : {why}")
    return lines


def _fmt_functionals(out, problem=None):
    """The quantity-of-interest lines, shared by the PDE and SDE branches of
    ``_fmt`` so one record cannot print what the other hides."""
    lines = []
    if out.get("functional_errs"):
        ftol = out.get("functional_tol", _functional_tol(problem or {}))
        used = out.get("functional_tol_used") or {}
        abs_errs = out.get("functional_abs_errs") or {}
        for name, err in out["functional_errs"].items():
            tol = used.get(name, ftol)
            ae = abs_errs.get(name)
            astr = f"  abs {ae:.3e}" if ae is not None else ""
            wide = f", widened from {ftol:g}" if tol > ftol else ""
            lines.append(f"  functional   : {name} rel {err:.3e}{astr}  "
                         f"(pass < {tol:g}{wide} -> {'ok' if err < tol else 'FAIL'})")
    lv = out.get("functional_levels")
    if lv and lv.get("status") != "ok":
        lines.append(f"  two-level    : not run ({lv.get('reason')}) -- no "
                     "anti-hardcoding check on this record")
    elif lv:
        worst = max(lv["values"].values(), key=lambda v: v["delta"] / max(v["band"], 1e-30))
        verdict = ("ok" if lv.get("passed")
                   else ("CONSTANT across levels" if lv.get("constant") else "INCONSISTENT"))
        lines.append(f"  two-level    : {lv['num_paths_graded']} vs "
                     f"{lv['num_paths_coarse']} paths, worst |delta| {worst['delta']:.3e} "
                     f"of band {worst['band']:.3e} -> {verdict}")
        if lv.get("level_ignored"):
            lines.append("  !! the coarse run ignored the path count it was given")
    return lines


def _fmt(out, problem=None):
    """Render one verification result. ``problem`` is display-only -- it supplies the
    problem's own tolerance, which the summary line previously printed as the global
    default whatever the problem had set."""
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
                     f"gate < {_pde_l2_tol(problem or {}):g})")
        lines.append(f"  verdict      : {'PASS' if out.get('passed') else 'FAIL'}")
        if out.get("error"):
            lines.append(f"  contract     : {out['error']}")
        if out.get("field_errs"):
            per = "  ".join(f"{k}={v:.2e}" for k, v in out["field_errs"].items())
            lines.append(f"  per-field    : {per}")
        lines.extend(_fmt_functionals(out, problem))
        if out.get("reference_error"):
            lines.append(f"  reference    : own error {out['reference_error']:.3e}"
                         + (f", tolerance widened to {out['tol_used']:.3e}"
                            if out.get("tol_widened") else ""))
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
        if (out.get("timing") or {}).get("solve_s") is not None:
            t = out["timing"]
            per = "  ".join(f"N={r['N']}: {r['solve_s']:.4g}s" for r in t["runs"]
                            if r.get("solve_s") is not None)
            lines.append(f"  solve time    : {t['solve_s']:.4g}s  ({per}; "
                         f"{t['wall_s']:.4g}s incl. {t['path']} overhead)")
        lines.extend(_fmt_work_precision(out))
        lines.append(f"  verified score: {out['verified_score']}/10")
        lines.append(f"  plan          : {out['plan_dir']}")
    else:  # SDE moments (exact or reference)
        for c in out["components"]:
            lines.append(
                f"  [{c['component']}] mean {c['empirical_mean']:.6g} vs {c['exact_mean']:.6g} "
                f"(rel {c['mean_rel_err']:.3%}{' skip' if c['near_zero_mean'] else ''}) | "
                f"var {c['empirical_variance']:.6g} vs {c['exact_variance']:.6g} "
                f"(rel {c['variance_rel_err']:.3%})")
        lines.extend(_fmt_functionals(out, problem))
        if (out.get("run") or {}).get("solve_s") is not None:
            lines.append(f"  solve time    : {out['run']['solve_s']:.4g}s")
        lines.extend(_fmt_work_precision(out))
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
    ap.add_argument("--sandbox", dest="sandbox", action="store_true", default=True,
                    help="run solver.py in an isolated subprocess (timeout, mem cap, "
                         "import policy). On by default, because that is how both "
                         "run.py and oneshot.py grade -- a spot-check that took the "
                         "other path could disagree with the recorded verdict")
    ap.add_argument("--no-sandbox", dest="sandbox", action="store_false",
                    help="import and call solver.py inline instead (debugging: a "
                         "traceback in this process, no budget)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_SOLVER_TIMEOUT_S,
                    help=f"sandbox wall-clock timeout per solve in seconds "
                         f"(default {DEFAULT_SOLVER_TIMEOUT_S})")
    ap.add_argument("--mem-mb", type=int, default=DEFAULT_SOLVER_MEM_MB,
                    help=f"sandbox address-space cap in MB, best-effort "
                         f"(default {DEFAULT_SOLVER_MEM_MB})")
    ap.add_argument("--no-work-precision", dest="work_precision", action="store_false",
                    default=True,
                    help="skip the extra solves of the work-precision sweep (the "
                         "levels the problem declares). The graded ladder's own "
                         "(error, time) points are free and are emitted either way; "
                         "no verdict depends on any of them")
    args = ap.parse_args(argv)

    problem = P.by_slug(args.slug)
    wdir = args.workspace_dir or os.path.join(args.workspace, args.slug)
    plan_dir = os.path.abspath(args.plan_dir) if args.plan_dir else None
    out = verify_problem(problem, wdir, plan_dir=plan_dir, sandbox=args.sandbox,
                         timeout_s=args.timeout, mem_mb=args.mem_mb,
                         work_precision=args.work_precision)
    print(f"=== {problem['id']} {args.slug} ({problem['type'].upper()}, tier {problem['tier']}) ===")
    print(_fmt(out, problem))


if __name__ == "__main__":
    main()
