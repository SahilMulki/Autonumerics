"""Tier B, route 2: the degenerate limit.

Cheaper than MMS and needs no source derivation (manual §6). The spec names a
parameter setting under which the problem *does* have a closed form -- kill the
nonlinearity, freeze a variable coefficient, set the reaction rate to zero -- and
the solver runs with those parameters against that closed form.

It is Tier-B evidence and counts alongside MMS for certification. On
``pde_kuramoto_sivashinsky`` it is strictly the better of the two: gamma -> 0
leaves the linear KS equation, which is diagonal in Fourier space, and the real
initial condition is an exact two-mode field, so the check runs the same
discretization over the full t = 0..50 rather than on a smooth manufactured
problem at expected order 2.
"""

from __future__ import annotations

from .ladder import axes_of, eval_on_axes, primary_of, rel_err


def declared(spec):
    limit = (spec.get("verification") or {}).get("degenerate_limit") or {}
    exact = limit.get("exact")
    if isinstance(exact, str):
        return {**limit, "exact": {"u": exact}}
    if isinstance(exact, dict) and exact:
        return {**limit, "exact": dict(exact)}
    return None


def run(spec, plan_dir, N, config, *, primary=None, metric="l2", run_solve=None):
    """``(outcome, detail)`` -- True / False / ``"unavailable"``."""
    from . import sandbox

    limit = declared(spec)
    if limit is None:
        return "unavailable", {"reason": "no degenerate_limit with an exact solution"}

    params = {**(spec.get("parameters") or {}), **(limit.get("params") or {})}
    solve = run_solve or (lambda N_, ov: sandbox.run(plan_dir, "pde", {"N": N_}, override=ov))
    got = solve(N, {"params": limit.get("params") or {}})
    if got["status"] != "ok":
        return "unavailable", {"reason": f"the degenerate run crashed: {got.get('reason')}: "
                                         f"{got.get('error')}"}
    result = got["result"]
    axis_names = list(spec.get("spatial_variables") or [])
    axes = axes_of(result, axis_names)
    t_final = result.get("t_final", params.get("t_final",
                                               (spec.get("time_interval") or {}).get("T", 0.0)))
    want = eval_on_axes(limit["exact"], axes, axis_names, params, t_final)
    key = primary if primary in want else next(iter(want))
    err = rel_err(primary_of(result, primary), want[key], metric)
    tol = float(limit.get("tol", config.get("rel_l2_err_max", 0.01)))
    return bool(err < tol), {"error": err, "tol": tol, "N": N,
                             "params": limit.get("params"), "t_final": t_final}
