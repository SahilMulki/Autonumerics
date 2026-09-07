"""Temporal-error isolation (manual §10).

Nothing else checks the plan-creator's instruction to keep the temporal error
subdominant, and a plan whose spatial order looks low *because time-stepping error
dominates* will otherwise be told to fix its stencil and will chase that wrong fix
for all five refine cycles. It is the single most common misdiagnosis in this
pipeline, which is why the check earns its cost twice: as attribution when the
order is already low, and as confirmation that a passing plan is not passing by
accident.

The comparison run is ``Ns[1]``, which the ladder already computed -- so the caller
passes the cached result in rather than paying for a second full solve of it.
"""

from __future__ import annotations

from .ladder import primary_of, rms


def run(plan_dir, N, base_result, e_fine, tol, *, primary=None, run_solve=None,
        dt_factor=0.5):
    """``(outcome, detail)`` -- True / False / ``"unavailable"``.

    ``True`` when the temporal error is subdominant. A check that could not run
    reports ``unavailable`` and, per the manual's standing rule, never counts
    against the plan.
    """
    from . import sandbox
    solve = run_solve or (lambda N_, ov: sandbox.run(plan_dir, "pde", {"N": N_}, override=ov))
    got = solve(N, {"dt_factor": dt_factor})
    if got["status"] != "ok":
        return "unavailable", {"reason": f"the halved-dt run crashed: {got.get('reason')}: "
                                         f"{got.get('error')}"}
    base = primary_of(base_result, primary)
    half = primary_of(got["result"], primary)
    if base.shape != half.shape:
        return "unavailable", {"reason": f"the halved-dt run returned {half.shape}, not "
                                         f"{base.shape}; dt_factor changed the mesh"}
    share = rms(half - base) / (rms(base) + 1e-14)
    bar = 0.1 * max(float(e_fine), float(tol))
    return bool(share < bar), {"temporal_share": share, "bar": bar, "dt_factor": dt_factor,
                               "N": N}
