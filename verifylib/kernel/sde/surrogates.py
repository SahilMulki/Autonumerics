"""Tier A': the three deterministic references an SDE with no closed form can have.

``verification_manual.md`` §15-17. The formulator's ``null`` is not the end of the
inquiry: reaching the Path-B rubric at all means all three of these were attempted
and all three failed, and if a surrogate *was* available and the spec simply did
not declare it, that is a formulator defect to report rather than absorb.

Each carries its own trust guard, and a surrogate that fails its guard is demoted
rather than reported: an untrustworthy surrogate presented as ground truth is
worse than no surrogate at all.
"""

from __future__ import annotations

import numpy as np

from ...operator import evaluate


def moment_ode(spec):
    """§15. Exact for polynomial diffusions, and only when ``closes_exactly``.

    A moment-closure *approximation* is not ground truth; where the formulator
    marks ``closes_exactly: false`` the result is a sanity band (Tier D), never a
    reference.
    """
    from scipy.integrate import solve_ivp

    mo = (spec.get("verification") or {}).get("moment_ode") or {}
    if not mo.get("closes_exactly"):
        return None, {"reason": "no moment ODE, or it does not close exactly"}
    state, rhs, initial = mo.get("state") or [], mo.get("rhs") or [], mo.get("initial") or []
    if not state or len(state) != len(rhs) or len(initial) != len(state):
        return None, {"reason": f"the moment ODE shape is incomplete (state {state})"}

    params = dict(spec.get("parameters") or {})
    T = float((spec.get("time_interval") or {}).get("T", params.get("T", 1.0)))
    y0 = [float(evaluate(e, params)) for e in initial]

    def fn(t, y):
        ns = {**params, "t": t, **dict(zip(state, y, strict=True))}
        return [float(evaluate(r, ns)) for r in rhs]

    sol = solve_ivp(fn, (0.0, T), y0, rtol=1e-12, atol=1e-14, method="Radau",
                    dense_output=True)
    if not sol.success:
        return None, {"reason": f"the moment ODE integration failed: {sol.message}"}
    ns = {**params, "t": T, **dict(zip(state, sol.y[:, -1], strict=True))}
    out = {}
    for key, target in (("mean_from", "mean"), ("variance_from", "variance")):
        if isinstance(mo.get(key), str):
            out[target] = float(evaluate(mo[key], ns))
    if not out:
        return None, {"reason": "the moment ODE declares no mean_from or variance_from"}
    return out, {"route": "moment_ode", "T": T, "state": dict(zip(state, sol.y[:, -1].tolist(),
                                                                 strict=True))}


def kolmogorov(spec, *, widen=1.5):
    """§16. The moments of an SDE solve a deterministic PDE, so the PDE machinery
    produces a noise-free reference -- and the truncation guard is mandatory.

    The far-field boundary is artificial, so the answer has to be shown insensitive
    to it. Where it is not, the caller demotes the provenance to
    ``self_convergence`` and says so.
    """
    kol = (spec.get("verification") or {}).get("kolmogorov") or {}
    bounds = kol.get("bounds")
    if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
        return None, {"reason": "no kolmogorov.bounds declared"}
    if int(spec.get("state_dimension", 1) or 1) != 1:
        return None, {"reason": "the Kolmogorov route is 1-D here; the formulator should "
                                "not propose it above 2 state dimensions"}
    params = dict(spec.get("parameters") or {})
    T = float((spec.get("time_interval") or {}).get("T", 1.0))
    x0 = params.get("X_0", params.get("x0"))
    if x0 is None:
        return None, {"reason": "no X_0 to evaluate the solution at"}

    payoffs = kol.get("payoffs") or ["x", "x**2"]
    Nx, Nt = int(kol.get("Nx", 4001)), int(kol.get("Nt", 4000))
    got = {}
    for label, span in (("base", bounds), ("wide", _widen(bounds, widen))):
        got[label] = [_solve_backward(spec, params, T, span, Nx, Nt, phi, float(x0))
                      for phi in payoffs]
        if any(v is None for v in got[label]):
            return None, {"reason": "the backward Kolmogorov solve failed"}

    shifts = [abs(a - b) / (abs(a) + 1e-14) for a, b in zip(got["base"], got["wide"],
                                                            strict=True)]
    m1, m2 = got["base"][0], got["base"][1] if len(got["base"]) > 1 else None
    out = {"mean": m1}
    if m2 is not None:
        out["variance"] = m2 - m1 * m1
    return out, {"route": "kolmogorov", "truncation_shift": shifts,
                 "bounds": list(bounds), "widened": _widen(bounds, widen)}


def _widen(bounds, factor):
    lo, hi = float(bounds[0]), float(bounds[1])
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * float(factor)
    return [mid - half, mid + half]


def _solve_backward(spec, params, T, bounds, Nx, Nt, phi, x0):
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    a, b = float(bounds[0]), float(bounds[1])
    x = np.linspace(a, b, Nx)
    dx, dtau = x[1] - x[0], T / Nt

    def ev(expr):
        return np.asarray(evaluate(expr, {**params, "X": x}), dtype=float) * np.ones_like(x)

    try:
        f = ev(spec["drift_expression"])
        g2 = ev(spec["diffusion_expression"]) ** 2
        u = np.asarray(evaluate(phi, {**params, "x": x, "X": x}), dtype=float) * np.ones_like(x)
    except Exception:  # noqa: BLE001 -- a matrix-valued coefficient is prose, not a formula
        return None

    lo = 0.5 * g2 / dx ** 2 - f / (2 * dx)
    di = -g2 / dx ** 2
    up = 0.5 * g2 / dx ** 2 + f / (2 * dx)
    L = sp.diags([lo[1:], di, up[:-1]], [-1, 0, 1], format="lil")
    L[0, :] = 0
    L[-1, :] = 0
    identity = sp.identity(Nx, format="csc")
    lu = spla.splu((identity - 0.5 * dtau * L.tocsc()).tocsc())
    B = (identity + 0.5 * dtau * L.tocsc()).tocsc()
    for _ in range(Nt):
        u = lu.solve(B @ u)
        u[0] = 2 * u[1] - u[2]      # u_xx = 0 at the truncation boundary
        u[-1] = 2 * u[-2] - u[-3]
    return float(np.interp(x0, x, u))


def stationary(spec):
    """§17. For an ergodic scalar SDE the invariant density has a closed form
    essentially always, even when the transient dynamics have none."""
    stat = (spec.get("verification") or {}).get("stationary_density") or {}
    if not stat.get("ergodic"):
        return None, {"reason": "the spec does not declare the process ergodic"}
    support = stat.get("support")
    if not (isinstance(support, (list, tuple)) and len(support) == 2):
        return None, {"reason": "no normalizable support declared"}
    params = dict(spec.get("parameters") or {})
    x = np.linspace(float(support[0]), float(support[1]), int(stat.get("Nx", 20001)))
    try:
        f = np.asarray(evaluate(spec["drift_expression"], {**params, "X": x}),
                       dtype=float) * np.ones_like(x)
        g2 = (np.asarray(evaluate(spec["diffusion_expression"], {**params, "X": x}),
                         dtype=float) * np.ones_like(x)) ** 2
    except Exception as exc:  # noqa: BLE001
        return None, {"reason": f"the coefficients are not evaluable: {exc}"}
    with np.errstate(all="ignore"):
        phi = 2 * np.concatenate([[0.0], np.cumsum(
            0.5 * (f[1:] / g2[1:] + f[:-1] / g2[:-1]) * np.diff(x))])
        logp = phi - np.log(g2)
        p = np.exp(logp - np.nanmax(logp))
    if not np.all(np.isfinite(p)) or np.trapezoid(p, x) <= 0:
        return None, {"reason": "the stationary density is not normalizable on the "
                                "declared support"}
    p = p / np.trapezoid(p, x)
    mean = float(np.trapezoid(x * p, x))
    return ({"mean": mean, "variance": float(np.trapezoid(x * x * p, x) - mean * mean)},
            {"route": "stationary_density", "T_stat": stat.get("T_stat"),
             "support": [float(support[0]), float(support[1])]})


#: In the order §13 says to exhaust them.
ROUTES = (("moment_ode", moment_ode), ("kolmogorov", kolmogorov),
          ("stationary_density", stationary))


def best(spec, *, trust_tol=None):
    """The strongest surrogate available. ``(reference, detail)`` or ``(None, why)``."""
    tried = {}
    for name, fn in ROUTES:
        try:
            got, detail = fn(spec)
        except Exception as exc:  # noqa: BLE001 -- a failed surrogate is not a crash
            tried[name] = f"{type(exc).__name__}: {exc}"
            continue
        if got is None:
            tried[name] = detail.get("reason")
            continue
        if name == "kolmogorov" and trust_tol is not None:
            shifts = detail.get("truncation_shift") or [0.0]
            if max(shifts) >= 0.1 * float(trust_tol):
                tried[name] = (f"the truncation guard failed: widening the domain moved "
                               f"the answer by {max(shifts):.2e}")
                continue
        return got, {**detail, "attempted": tried}
    return None, {"reason": "no surrogate route produced a reference", "attempted": tried}
