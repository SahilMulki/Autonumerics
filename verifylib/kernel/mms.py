"""Tier B, route 1: the method of manufactured solutions.

The strongest evidence available without a closed form (manual §5). Pick a smooth
``u_mms`` unrelated to the real initial and boundary data, substitute it into the
operator to get ``f = L u_mms``, solve the *sourced* problem with the same code,
and check it recovers ``u_mms`` at the design order. It verifies the
**discretization** rather than the problem's solution -- but it is the same code
that then runs the real problem, which is what eliminates the "converged to the
wrong operator" failure tiers C--E cannot see.

Two rules from the manual are enforced here rather than left to a caller:

* **A faulty probe is skipped, not failed.** A wrong hand-derived source term makes
  a correct solver miss, and that is the formulator's defect. The probe is
  validated numerically first, and a probe that does not validate returns
  ``faulty_probe``.
* **The override must reach the same discretization.** A solver that silently falls
  back to its own initial condition when handed ``override["ic"]`` produces a
  beautiful MMS result that means nothing, so the probe error is required to be
  small *and* to fall.
"""

from __future__ import annotations

import numpy as np

from ..reference import TOL as FORMULA_TOL
from .ladder import axes_of, eval_on_axes, primary_of, rel_err
from .residual import _residual_of_formula, _system_with_source


def probe_exprs(spec):
    """``(exact, source)`` as ``{field: expression}``, or ``(None, None)``."""
    probe = (spec.get("verification") or {}).get("mms_probe") or {}
    exact, source = probe.get("exact"), probe.get("source")
    if isinstance(exact, str) and isinstance(source, str):
        return {"u": exact}, {"u": source}
    if isinstance(exact, dict) and isinstance(source, dict) and exact and source:
        return dict(exact), dict(source)
    return None, None


def validate_probe(spec, operator, tol=FORMULA_TOL):
    """Apply the operator to ``mms_probe.exact`` and require it to reproduce
    ``mms_probe.source``. Returns ``(ok, detail)``; ``ok`` is ``None`` when the
    probe cannot be evaluated at all."""
    exact, source = probe_exprs(spec)
    if exact is None or operator is None:
        return None, {"reason": "no MMS probe, or no operator to check it with"}
    sourced = ({**operator, "source": source["u"]} if operator["kind"] == "scalar"
               else _system_with_source(operator, source))
    return _residual_of_formula(spec, sourced, exact, tol)


def override_payload(spec, exprs, *, params=None, fixed_t=None):
    """One ``override`` field, in the form :mod:`verifylib.kernel.sandbox` rebuilds."""
    payload = {"kind": "expr", "exprs": dict(exprs),
               "params": dict(params or spec.get("parameters") or {}),
               "spatial_variables": list(spec.get("spatial_variables") or []),
               "multi": len(exprs) > 1}
    if fixed_t is not None:
        payload["fixed_t"] = float(fixed_t)
    return payload


def initial_time(spec):
    params = spec.get("parameters") or {}
    return float(params.get("t0", 0.0))


def run(spec, plan_dir, Ns, config, *, primary=None, metric="l2", run_solve=None):
    """Run the probe at two grids. ``(outcome, detail)``.

    ``outcome`` is True / False / ``"faulty_probe"`` / ``"unavailable"``.
    """
    from . import sandbox
    from .residual import operator_for

    exact, source = probe_exprs(spec)
    if exact is None:
        return "unavailable", {"reason": "no mms_probe with both exact and source"}
    operator, _ = operator_for(spec)
    ok, detail = validate_probe(spec, operator)
    if ok is False:
        # Manual §5: skip the test and report it as a spec defect. A broken probe
        # must not penalise a correct solver.
        return "faulty_probe", {"probe_validation": detail}
    probe_validated = bool(ok)

    axis_names = list(spec.get("spatial_variables") or [])
    t0 = initial_time(spec)
    override = {
        "ic": override_payload(spec, exact, fixed_t=t0),
        "source": override_payload(spec, source),
    }
    if str(((spec.get("boundary_conditions") or {}).get("type") or "")).startswith("dirichlet"):
        override["bc"] = override_payload(spec, exact)

    solve = run_solve or (lambda N, ov: sandbox.run(plan_dir, "pde", {"N": N}, override=ov))
    errors, grids = [], []
    for N in Ns[:2]:
        got = solve(N, override)
        if got["status"] != "ok":
            return "unavailable", {"reason": f"the sourced run crashed at N={N}: "
                                             f"{got.get('reason')}: {got.get('error')}",
                                   "probe_validated": probe_validated}
        result = got["result"]
        axes = axes_of(result, axis_names)
        want = eval_on_axes(exact, axes, axis_names, spec.get("parameters"),
                            result.get("t_final", (spec.get("time_interval") or {}).get("T", 0.0)))
        key = primary if primary in want else next(iter(want))
        errors.append(rel_err(primary_of(result, primary), want[key], metric))
        grids.append(N)

    if len(errors) < 2 or errors[0] <= 0.0 or errors[1] <= 0.0:
        order = float("inf")
    else:
        order = float(np.log2(errors[0] / errors[1]))
    tol = config.get("rel_l2_err_max", 0.01)
    floor = config.get("min_spatial_order", 1.0)
    passed = bool(errors[-1] < tol and (order >= floor or not np.isfinite(order)))
    return passed, {"errors": errors, "grids": grids, "mms_order": order,
                    "tol": tol, "order_floor": floor,
                    "probe_validated": probe_validated,
                    "probe_validation": detail}
