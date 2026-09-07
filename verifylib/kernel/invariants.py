"""D2: the structural gate, and the **closed registry** that makes it mean anything.

Real gated invariants are catalogue entries, not expressions::

    {"name": "divergence_free", "gate": true, "tol": 1e-08, "fields": ["Bx", "By"]}
    {"name": "energy_bounded",  "gate": true, "bound": 100.0, "requires_trace": true}

``schema.py`` on its own validates only entries carrying an ``expr``, so
``{"name": "physically_reasonable", "gate": true}`` passed the schema, matched
nothing here, and -- unless this module were careful -- would render as a pass. A
gate nobody implements is worse than no gate, because it reads as evidence. Hence
three rules, which together are plan-no-closed-form §6:

1. :data:`INVARIANTS` is closed. It holds every name in ``verification_manual.md``
   §8 and §21, plus the three the workspace actually gates on that the manual does
   not name (``energy_bounded``, ``convexity``, ``shape``).
2. ``schema.check_invariant_names`` errors on a gated entry that neither names a
   registry entry nor carries an evaluable ``expr``.
3. A declared gate this module cannot evaluate reports ``not_reported`` -- never
   ``pass`` -- and caps the score at 9, which is already the manual's rule for a
   missing ``invariant_trace``, generalised.

numpy is optional at import, for the same reason it is in ``operator.py``: this
registry is imported by ``schema.py``, which must keep running on an interpreter
that has no numpy (guardrails E32).
"""

from __future__ import annotations

try:
    import numpy as np
except ImportError:  # pragma: no cover -- the registry itself needs no numpy
    np = None

from ..operator import _require_numpy, evaluate

#: What an invariant needs in order to be evaluated at all.
#:
#: ``final``    -- the final field, and usually the initial one.
#: ``trace``    -- ``invariant_trace`` from the solver; absent => ``not_reported``.
#: ``override`` -- a second solve through the ``override`` hook.
#: ``paths``    -- SDE ``terminal_paths``.
#: ``expr``     -- an evaluable predicate supplied by the spec entry itself.
NEEDS = ("final", "trace", "override", "paths", "expr")


class Invariant:
    """One registry entry: what it applies to, what it needs, what it accepts."""

    __slots__ = ("name", "side", "needs", "params", "summary")

    def __init__(self, name, side, needs, params, summary):
        self.name, self.side, self.needs = name, side, needs
        self.params, self.summary = frozenset(params), summary

    def __repr__(self):
        return f"Invariant({self.name!r}, {self.side!r}, needs={self.needs!r})"


#: Parameters every entry may carry, whatever its name.
COMMON_PARAMS = ("name", "gate", "tol", "note", "expr", "requires_trace", "check")


def _inv(name, side, needs, params, summary):
    return Invariant(name, side, needs, (*COMMON_PARAMS, *params), summary)


#: The closed set. Adding a name here is the *only* way to make it gateable, and
#: doing so means writing its evaluator in :func:`evaluate_invariant` below.
INVARIANTS = {
    # --- verification_manual.md §8, PDE -------------------------------------
    "mass_conservation": _inv(
        "mass_conservation", "pde", "final", ("absolute", "exact_value", "periodic"),
        "integral(u) is constant in time"),
    "mass_flux_balance": _inv(
        "mass_flux_balance", "pde", "trace", (),
        "integral(u(T)) - integral(u(0)) equals the integrated boundary flux"),
    "maximum_principle": _inv(
        "maximum_principle", "pde", "final", ("bc_values",),
        "u(T) stays within [min, max] of the initial and boundary data"),
    "energy_decay": _inv(
        "energy_decay", "pde", "trace", (),
        "integral(u**2) is non-increasing"),
    "energy_conservation": _inv(
        "energy_conservation", "pde", "trace", (),
        "the declared energy is constant in time"),
    "energy_bounded": _inv(
        "energy_bounded", "pde", "trace", ("bound",),
        "the declared energy stays finite and below `bound` at every step"),
    "total_variation": _inv(
        "total_variation", "pde", "final", ("axis",),
        "sum|du| is non-increasing (TVD)"),
    "positivity": _inv(
        "positivity", "both", "final", ("fields",),
        "min(u) >= -tol"),
    "symmetry": _inv(
        "symmetry", "pde", "final", ("axis", "parity", "fields"),
        "||u - P u|| / ||u|| ~ 0 for the declared reflection P"),
    "translation_invariance": _inv(
        "translation_invariance", "pde", "override", ("axis", "shift_cells"),
        "solve with the IC shifted, shift back, and recover the unshifted solution"),
    "divergence_free": _inv(
        "divergence_free", "pde", "final", ("fields",),
        "||div F|| ~ 0 over the declared field tuple"),
    "boundary_consistency": _inv(
        "boundary_consistency", "pde", "final", ("bc_values", "faces"),
        "the final field matches the declared Dirichlet data on each conforming face"),
    # --- gated in workspace/, not named by manual §8 -------------------------
    "convexity": _inv(
        "convexity", "pde", "final", (),
        "the discrete Hessian is positive semi-definite (selects the viscosity "
        "solution of a Monge-Ampere problem)"),
    # --- verification_manual.md §21, SDE -------------------------------------
    "finite": _inv(
        "finite", "sde", "paths", (),
        "no NaN/Inf in terminal_paths"),
    "support": _inv(
        "support", "sde", "paths", ("bounds", "lower", "upper"),
        "fraction of paths outside the declared support"),
    "martingale": _inv(
        "martingale", "sde", "paths", ("ci_mult", "reference_value_at_T"),
        "E[M_T] = M_0 within ci_mult standard errors"),
    "estimator_stability": _inv(
        "estimator_stability", "sde", "paths", ("n_batches",),
        "batch-means scatter confirms num_paths has settled"),
    "shape": _inv(
        "shape", "sde", "expr", (),
        "terminal_paths has the declared shape"),
}

#: Names the workspace uses without gating. Free-form by design (§6: "ungated
#: entries stay free-form notes"), listed only so a reader can see that leaving
#: them out of the registry was a decision rather than an oversight.
UNGATED_IN_USE = (
    "arbitrage_lower_bound", "arbitrage_upper_bound", "component_symmetry",
    "enthalpy_balance", "front_position", "gaussianity", "monotone_in_S",
)


def known(name) -> bool:
    return name in INVARIANTS


def gateable(entry) -> bool:
    """Whether a gated entry can be evaluated at all.

    Either it names a registry entry, or it carries its own ``expr`` -- which is
    how every gated SDE constraint in ``workspace/`` is written
    (``{"name": "finite", "expr": "np.isfinite(X)", "gate": true}``) and is
    generically evaluable whatever the name says.
    """
    if not isinstance(entry, dict):
        return False
    return known(entry.get("name")) or isinstance(entry.get("expr"), str)


# --- quadrature and norms ----------------------------------------------------

def rms(a, mask=None):
    _require_numpy("computing an invariant")
    a = a if mask is None else a[mask]
    return float(np.sqrt(np.mean(np.asarray(a, dtype=float) ** 2)))


def duplicated_endpoint(wraps, endpoint_exclusive, nd):
    """Per axis: does the returned grid repeat its left endpoint at the right one?

    Two different facts get conflated here and must not be. ``wraps`` is
    *boundary_conditions.type == "periodic"* -- a property of the problem.
    ``endpoint_exclusive`` is what ``ladder.detect_periodic`` measures -- a property
    of the array the solver chose to return, and what decides the refinement
    arithmetic. A grid duplicates its wrap node exactly when both are true:
    the physics wraps and the array still carries the right endpoint.

    ``pde_kuramoto_sivashinsky`` is that case and nothing else in ``workspace/``
    is: its solver returns ``linspace(0, L, N)`` with ``u[N-1] = u[0]``, so
    ``detect_periodic`` correctly reports *not* exclusive and the ladder is
    64/127/253 -- while the field still has the same physical point twice. Reading
    one flag for both facts rolls the full N-array by a fraction of a cell and
    reports a translation-invariance violation of 0.76 on three different correct
    spectral schemes, which is how this was found.
    """
    wraps = list(wraps or [False] * nd)
    excl = list(endpoint_exclusive or [False] * nd)
    return [bool(wraps[i]) and not bool(excl[i] if i < len(excl) else False)
            for i in range(nd)]


def _trim_duplicate_endpoint(u, axes, duplicated):
    """Drop the repeated wrap node so quadrature does not count it twice.

    ``np.trapezoid`` over such an array half-weights both copies *and* drops the
    wrap-around cell, which fakes a mass drift of order 1e-5 on a scheme that
    conserves mass exactly -- the failure the KS spec's own note warns about.
    """
    sl = [slice(None)] * u.ndim
    kept_axes = list(axes)
    for i, dup in enumerate(list(duplicated) + [False] * u.ndim):
        if i >= u.ndim:
            break
        if dup and len(axes[i]) > 1:
            sl[i] = slice(0, -1)
            kept_axes[i] = axes[i][:-1]
    return u[tuple(sl)], kept_axes


def integral(u, axes, periodic=None, duplicated=None):
    """Quadrature over d dimensions, periodicity-aware (manual §8).

    ``periodic`` selects the rectangle rule, which is spectrally accurate on a
    periodic grid; ``duplicated`` says which axes carry a repeated wrap node and
    must be trimmed first.
    """
    _require_numpy("computing an invariant")
    u = np.asarray(u, dtype=float)
    periodic = list(periodic or [False] * u.ndim)
    u, axes = _trim_duplicate_endpoint(u, axes,
                                       duplicated if duplicated is not None else periodic)
    out = u
    for i in range(u.ndim - 1, -1, -1):
        ax = axes[i]
        if periodic[i]:
            out = out.sum(axis=i) * float(ax[1] - ax[0])
        else:
            out = np.trapezoid(out, ax, axis=i)
    return float(out)


# --- the evaluators ----------------------------------------------------------
#
# Each returns ``(outcome, drift)``. ``outcome`` is True / False / a skip marker
# from metrics.SKIPS; ``drift`` is the number the review reports, or None.

def _tol(entry, default=1e-8):
    tol = entry.get("tol")
    return float(tol) if isinstance(tol, (int, float)) else default


#: The floor below which a declared tolerance is tighter than any real scheme can be.
#:
#: Real specs write ``tol: 0.0`` for a quantity that is exactly monotone or exactly
#: conserved -- ``pde_heat_1d``'s and ``pde_advection_1d``'s ``energy_decay`` both do.
#: The manual's reference snippets test ``drift < tol``, which on ``tol = 0.0``
#: reports a *violation* for a drift of exactly zero: a perfectly conserving scheme
#: failing its own conservation check.
#:
#: 1e-10 rather than machine epsilon, because the quantity being compared is not a
#: single arithmetic result. Measured on ``pde_advection_1d/3-crank-nicolson-fd4``:
#: a Crank-Nicolson advection solve, which is unitary and conserves the discrete L2
#: norm exactly, reports an energy drift of **2.0e-12** -- that is its sparse
#: solve's residual, accumulated over a thousand steps, and it is round-off class.
#:
#: Applied as ``max(tol, ROUNDOFF)``, never ``tol + ROUNDOFF``, so it only ever
#: takes effect where the declared tolerance is *below* the floor. Every tolerance a
#: formulator actually chose above it -- and the smallest in ``workspace/`` is
#: exactly 1e-10 -- is left at the value they wrote.
ROUNDOFF = 1e-10


def _within(value, tol):
    return bool(value <= max(float(tol), ROUNDOFF))


def _field(ctx, entry):
    names = entry.get("fields")
    if isinstance(names, list) and names:
        return [ctx.field(n) for n in names]
    return [ctx.primary()]


def _mass_conservation(entry, ctx):
    u_T, u_0 = ctx.primary(), ctx.initial()
    if u_0 is None:
        return "not_reported", None
    mT = integral(u_T, ctx.axes, ctx.periodic, ctx.duplicated)
    if entry.get("absolute"):
        # The relative form divides by |m0| + 1e-14, so a conserved value of
        # exactly zero (KS on a periodic domain) reports a "drift" of order 1e14 on
        # a perfect solver. The spec says so; honour it.
        exact = float(entry.get("exact_value", 0.0))
        scale = _domain_measure(ctx) * float(np.max(np.abs(u_T))) + 1e-300
        drift = abs(mT - exact) / scale
    else:
        m0 = integral(u_0, ctx.axes, ctx.periodic, ctx.duplicated)
        drift = abs(mT - m0) / (abs(m0) + 1e-14)
    return _within(drift, _tol(entry)), drift


def _domain_measure(ctx):
    out = 1.0
    for ax in ctx.axes:
        out *= abs(float(ax[-1] - ax[0])) or 1.0
    return out


def _maximum_principle(entry, ctx):
    u_T, u_0 = ctx.primary(), ctx.initial()
    if u_0 is None:
        return "not_reported", None
    bc = [float(v) for v in (entry.get("bc_values") or [])]
    tol = _tol(entry, 1e-10)
    lo = min([float(np.min(u_0)), *bc]) - tol
    hi = max([float(np.max(u_0)), *bc]) + tol
    over = max(0.0, float(np.max(u_T)) - hi, lo - float(np.min(u_T)))
    return over <= 0.0, over


def _positivity(entry, ctx):
    worst = min(float(np.min(f)) for f in _field(ctx, entry))
    tol = _tol(entry, 1e-10)
    return worst >= -tol, -min(worst, 0.0)


def _total_variation(entry, ctx):
    u_T, u_0 = ctx.primary(), ctx.initial()
    if u_0 is None:
        return "not_reported", None
    axis = int(entry.get("axis", 0))
    def tv(a):
        return float(np.sum(np.abs(np.diff(a, axis=axis))))
    growth = (tv(u_T) - tv(u_0)) / (tv(u_0) + 1e-14)
    return _within(growth, _tol(entry, 1e-8)), growth


def _symmetry(entry, ctx):
    axis, parity = int(entry.get("axis", 0)), entry.get("parity", "even")
    worst = 0.0
    for f in _field(ctx, entry):
        reflected = np.flip(f, axis=axis)
        if parity == "odd":
            reflected = -reflected
        worst = max(worst, rms(f - reflected) / (rms(f) + 1e-14))
    return _within(worst, _tol(entry)), worst


def _divergence_free(entry, ctx):
    names = entry.get("fields")
    if not (isinstance(names, list) and len(names) == len(ctx.axes)):
        return "not_reported", None
    div = None
    for i, name in enumerate(names):
        term = np.gradient(ctx.field(name), ctx.axes[i], axis=i, edge_order=2)
        div = term if div is None else div + term
    interior = tuple(slice(1, -1) for _ in ctx.axes)
    scale = max(rms(ctx.field(names[0])), 1e-14) / max(
        min(abs(float(a[1] - a[0])) for a in ctx.axes), 1e-300)
    return _within(rms(div[interior]) / scale, _tol(entry)), rms(div[interior])


def _boundary_consistency(entry, ctx):
    """A stencil needs ghost points, so the residual is interior-only and a boundary
    violation lives exactly where D1 cannot compute it (plan §5g). This is the check
    that covers that blind spot: compare the final field on each conforming Dirichlet
    face against the declared data.

    Faces come from the spec, through the entry, in the same ``"x=0"`` form
    ``reference._dirichlet_faces`` reads -- a moving or symbolic boundary is skipped
    rather than guessed at.
    """
    faces = entry.get("faces")
    values = entry.get("bc_values")
    if not faces and isinstance(values, list) and values:
        # The common scalar case: one declared level applied on every face.
        u = ctx.primary()
        worst = 0.0
        for axis in range(u.ndim):
            for index, want in ((0, values[0]), (-1, values[-1])):
                face = np.take(u, index, axis=axis)
                scale = max(float(np.max(np.abs(u))), 1e-300)
                worst = max(worst, float(np.max(np.abs(face - float(want)))) / scale)
        return _within(worst, _tol(entry, 1e-8)), worst
    if not isinstance(faces, dict) or not faces:
        return "not_reported", None
    u, worst = ctx.primary(), 0.0
    for key, want in faces.items():
        axis_name, _, coord = str(key).partition("=")
        try:
            index = 0 if float(coord) <= float(ctx.axes[0][0]) else -1
            axis = 0
        except ValueError:
            continue
        face = np.take(u, index, axis=axis)
        scale = max(float(np.max(np.abs(u))), 1e-300)
        worst = max(worst, float(np.max(np.abs(face - float(want)))) / scale)
    return _within(worst, _tol(entry, 1e-8)), worst


def _convexity(entry, ctx):
    """The discrete Hessian must be PSD: ``uxx >= -tol``, ``uyy >= -tol``,
    ``uxx*uyy - uxy**2 >= -tol``. This is the constraint that selects the
    viscosity solution of a Monge-Ampere problem, quoted from that spec's note."""
    from ..operator import d1, d2
    from .ladder import uniform_axes
    u = ctx.primary()
    if u.ndim != 2 or not uniform_axes(ctx.axes[:2]):
        # The discrete Hessian below is a fixed-h stencil; a graded mesh makes it
        # measure the mesh instead of the solution.
        return "not_reported", None
    hx, hy = (float(a[1] - a[0]) for a in ctx.axes[:2])
    uxx, uyy = d2(u, 0, hx), d2(u, 1, hy)
    uxy = d1(d1(u, 0, hx), 1, hy)
    s = tuple(slice(4, -4) for _ in range(2))
    tol = _tol(entry, 1e-6)
    worst = max(-float(np.min(uxx[s])), -float(np.min(uyy[s])),
                -float(np.min((uxx * uyy - uxy ** 2)[s])))
    return worst <= tol, max(worst, 0.0)


def _trace_of(entry, ctx, *candidates):
    trace = ctx.trace()
    if not trace:
        return None
    for key in (*candidates, entry.get("name")):
        if key in trace:
            series = np.asarray(trace[key], dtype=float)
            if series.size:
                return series
    return None


def _energy_decay(entry, ctx):
    series = _trace_of(entry, ctx, "energy")
    if series is None:
        return "not_reported", None
    growth = float(np.max(series) - series[0]) / (abs(float(series[0])) + 1e-14)
    return _within(growth, _tol(entry, 1e-8)), growth


def _energy_conservation(entry, ctx):
    series = _trace_of(entry, ctx, "energy")
    if series is None:
        return "not_reported", None
    drift = float(np.max(np.abs(series - series[0]))) / (abs(float(series[0])) + 1e-14)
    return _within(drift, _tol(entry, 1e-6)), drift


def _energy_bounded(entry, ctx):
    bound = float(entry.get("bound", np.inf))
    series = _trace_of(entry, ctx, "energy")
    if series is None:
        # No trace: the verdict is not_reported, but the *final* field still says
        # whether the run blew up, and that costs nothing. Recorded as the drift so
        # the review can report it without it ever reading as the gate's verdict.
        u = ctx.primary()
        final = float(np.mean(np.asarray(u, dtype=float) ** 2))
        return "not_reported", final
    worst = float(np.max(series))
    return bool(np.all(np.isfinite(series)) and worst < bound), worst


def _mass_flux_balance(entry, ctx):
    series = _trace_of(entry, ctx, "mass")
    if series is None:
        return "not_reported", None
    flux = _trace_of(entry, ctx, "boundary_flux", "flux")
    if flux is None:
        return "not_reported", None
    residual = abs(float(series[-1] - series[0]) - float(np.sum(flux)))
    return _within(residual, _tol(entry) * (abs(float(series[0])) + 1e-14)), residual


def _translation_invariance(entry, ctx):
    shift, axis = int(entry.get("shift_cells", 0)), int(entry.get("axis", 0))
    if not shift or ctx.solve is None:
        return "not_reported", None
    # The wrap node again: rolling the N-array of a duplicated-endpoint grid shifts
    # by a fraction of a cell and reports a spurious O(1) violation. Roll the M
    # independent values only.
    trimmed = bool(ctx.duplicated and ctx.duplicated[axis])
    def roll(a, k):
        if not trimmed:
            return np.roll(a, k, axis=axis)
        head = np.take(a, range(a.shape[axis] - 1), axis=axis)
        rolled = np.roll(head, k, axis=axis)
        return np.concatenate(
            [rolled, np.take(rolled, [0], axis=axis)], axis=axis)

    base = ctx.primary()
    ic = ctx.initial()
    if ic is None:
        return "not_reported", None
    # The IC crosses a process boundary as an array, not a closure: `ctx.solve`
    # takes it by keyword and the sandbox ships it through the same .npz the
    # results come back in.
    shifted = ctx.solve(ctx.N, ic_array=roll(ic, shift))
    if shifted is None:
        return "not_reported", None
    got = ctx.primary_of(shifted)
    drift = rms(roll(got, -shift) - base) / (rms(base) + 1e-14)
    return _within(drift, _tol(entry, 1e-4)), drift


# --- SDE ---------------------------------------------------------------------

def _finite(entry, ctx):
    X = ctx.paths()
    if X is None:
        return "not_reported", None
    bad = float(np.mean(~np.isfinite(X)))
    return bad == 0.0, bad


def _support(entry, ctx):
    X = ctx.paths()
    if X is None:
        return "not_reported", None
    lo = entry.get("lower", (entry.get("bounds") or [None, None])[0])
    hi = entry.get("upper", (entry.get("bounds") or [None, None])[1])
    if lo is None and hi is None:
        return "not_reported", None
    out = np.zeros(X.shape[:1], dtype=bool) if X.ndim else np.zeros((), dtype=bool)
    flat = X.reshape(X.shape[0], -1) if X.ndim > 1 else X.reshape(-1, 1)
    if lo is not None:
        out = out | np.any(flat < float(lo) - _tol(entry, 0.0), axis=1)
    if hi is not None:
        out = out | np.any(flat > float(hi) + _tol(entry, 0.0), axis=1)
    frac = float(np.mean(out))
    return _within(frac, _tol(entry, 0.0)), frac


def _martingale(entry, ctx):
    X = ctx.paths()
    m0 = entry.get("reference_value_at_T", ctx.initial_value())
    if X is None or m0 is None:
        return "not_reported", None
    flat = np.asarray(X, dtype=float).reshape(len(X), -1)[:, 0]
    m = float(np.mean(flat))
    se = float(np.std(flat, ddof=1) / np.sqrt(len(flat)))
    ci = float(entry.get("ci_mult", 2.0))
    return _within(abs(m - float(m0)), ci * se + _tol(entry, 0.0)), abs(m - float(m0))


def _estimator_stability(entry, ctx):
    X = ctx.paths()
    if X is None:
        return "not_reported", None
    flat = np.asarray(X, dtype=float).reshape(len(X), -1)[:, 0]
    n = int(entry.get("n_batches", 20))
    if len(flat) < 2 * n:
        return "not_reported", None
    means = np.array([float(np.mean(b)) for b in np.array_split(flat, n)])
    scatter = float(np.std(means, ddof=1) / np.sqrt(n) / (abs(float(np.mean(flat))) + 1e-14))
    return _within(scatter, _tol(entry, 1e-2)), scatter


_EVALUATORS = {
    "mass_conservation": _mass_conservation,
    "mass_flux_balance": _mass_flux_balance,
    "maximum_principle": _maximum_principle,
    "energy_decay": _energy_decay,
    "energy_conservation": _energy_conservation,
    "energy_bounded": _energy_bounded,
    "total_variation": _total_variation,
    "positivity": _positivity,
    "symmetry": _symmetry,
    "translation_invariance": _translation_invariance,
    "divergence_free": _divergence_free,
    "convexity": _convexity,
    "boundary_consistency": _boundary_consistency,
    "finite": _finite,
    "support": _support,
    "martingale": _martingale,
    "estimator_stability": _estimator_stability,
}

#: Names in the registry with no evaluator here. Empty by construction: a name is
#: only allowed into :data:`INVARIANTS` once it can be measured. ``shape`` is
#: absent because it is ``needs="expr"`` -- the spec entry supplies the predicate.
UNIMPLEMENTED = tuple(sorted(
    n for n, inv in INVARIANTS.items() if inv.needs != "expr" and n not in _EVALUATORS))


def _from_expr(entry, ctx):
    """A spec-supplied predicate. Every gated SDE constraint in ``workspace/``
    takes this route (``np.isfinite(X)``, ``X > 0``, ``X.shape == (num_paths, 2)``)."""
    try:
        value = evaluate(entry["expr"], ctx.expr_namespace())
    except Exception as exc:  # noqa: BLE001 -- an unevaluable predicate is not a failure
        return "not_reported", f"{type(exc).__name__}: {exc}"
    arr = np.asarray(value)
    if arr.dtype == bool or arr.dtype.kind == "b":
        frac = float(np.mean(~arr)) if arr.ndim else float(not bool(arr))
        return _within(frac, _tol(entry, 0.0)), frac
    return bool(value), None


def evaluate_invariant(entry, ctx):
    """One declared invariant. Returns ``(outcome, drift)``.

    An entry this module does not recognise reports ``not_reported``, never
    ``pass``: §6's third rule, and the reason the registry has to be closed.
    """
    _require_numpy("evaluating an invariant")
    name = entry.get("name")
    if isinstance(entry.get("expr"), str) and name not in _EVALUATORS:
        return _from_expr(entry, ctx)
    fn = _EVALUATORS.get(name)
    if fn is None:
        return "not_reported", None
    # No generic short-circuit for a missing trace: each trace evaluator reports
    # `not_reported` itself, and two of them have something useful to say anyway --
    # `energy_bounded` still checks the final field for finiteness and for the
    # bound, which needs no trace and which the KS spec explicitly asks for.
    try:
        return fn(entry, ctx)
    except Exception as exc:  # noqa: BLE001 -- never let a diagnostic kill the run
        return "not_reported", f"{type(exc).__name__}: {exc}"


# --- what an evaluator is handed ---------------------------------------------

class PdeContext:
    """Everything a PDE invariant can read, and nothing it can write.

    ``solve`` is the only door back to the solver, and only ``translation_invariance``
    uses it. Passing it explicitly rather than letting each evaluator reach for the
    sandbox keeps the cost of D2 visible: every other invariant here is arithmetic
    on arrays already in memory.
    """

    def __init__(self, fields, axes, *, periodic=None, endpoint_exclusive=None,
                 initial=None, trace=None, solve=None, N=None, primary=None, mask=None):
        # `solve(N, *, ic_array=None, params=None, dt_factor=None) -> result | None`
        # `periodic` is the *boundary condition*; `endpoint_exclusive` is the array
        # convention the solver chose. See :func:`duplicated_endpoint`.
        self.fields, self.axes = dict(fields), list(axes)
        self.periodic = list(periodic or [False] * len(self.axes))
        self.endpoint_exclusive = list(endpoint_exclusive or [False] * len(self.axes))
        self.duplicated = duplicated_endpoint(self.periodic, self.endpoint_exclusive,
                                              len(self.axes))
        self._initial, self._trace = initial, trace or {}
        self.solve, self.N, self._primary, self.mask = solve, N, primary, mask

    def field(self, name):
        return np.asarray(self.fields[name], dtype=float)

    def primary(self):
        if self._primary and self._primary in self.fields:
            return self.field(self._primary)
        return np.asarray(next(iter(self.fields.values())), dtype=float)

    def primary_of(self, result):
        fields = result.get("fields") or {}
        if self._primary and self._primary in fields:
            return np.asarray(fields[self._primary], dtype=float)
        return np.asarray(next(iter(fields.values())), dtype=float)

    def initial(self):
        return None if self._initial is None else np.asarray(self._initial, dtype=float)

    def trace(self):
        return self._trace

    def expr_namespace(self):
        return {"u": self.primary(), **{k: self.field(k) for k in self.fields}}


class SdeContext:
    """Everything an SDE constraint can read."""

    def __init__(self, terminal_paths, *, params=None, num_paths=None, x0=None,
                 path_integrals=None):
        self._paths = terminal_paths
        self.params = dict(params or {})
        self.num_paths = num_paths if num_paths is not None else (
            len(terminal_paths) if terminal_paths is not None else None)
        self.x0 = x0
        self.path_integrals = path_integrals or {}
        self.axes, self.periodic, self.mask = [], [], None
        self.duplicated, self.endpoint_exclusive = [], []
        self.solve, self.N = None, None

    def paths(self):
        return None if self._paths is None else np.asarray(self._paths, dtype=float)

    def initial_value(self):
        return self.x0

    def trace(self):
        return {}

    def primary(self):
        return self.paths()

    def initial(self):
        return None

    def field(self, name):
        raise KeyError(name)

    def expr_namespace(self):
        return {"X": self.paths(), "num_paths": self.num_paths, **self.params}


def run_declared(verification, ctx):
    """Evaluate every declared invariant and constraint. One implementation for
    both drivers, because they are the same operation on different contexts.

    The ``not_reported`` split matters for scoring: §6's rule caps the score at 9
    for a **gated** invariant the kernel could not evaluate, because a declared gate
    nothing implements reads as evidence. An ungated one is a diagnostic the spec
    asked for and did not get -- worth reporting, never worth a cap.
    """
    declared = [e for e in (verification.get("invariants") or []) if isinstance(e, dict)]
    declared += [e for e in (verification.get("constraints") or []) if isinstance(e, dict)]
    drifts, outcomes, gate_failed = {}, {}, []
    not_reported, not_reported_gated, non_gate_ok = [], [], True
    for entry in declared:
        outcome, drift = evaluate_invariant(entry, ctx)
        name = entry.get("name") or "?"
        drifts[name] = drift if not isinstance(drift, (str, type(None))) else drift
        outcomes[name] = outcome if isinstance(outcome, str) else bool(outcome)
        if isinstance(outcome, str):
            not_reported.append(name)
            if entry.get("gate"):
                not_reported_gated.append(name)
        elif outcome is False:
            if entry.get("gate"):
                gate_failed.append(name)
            else:
                non_gate_ok = False
    return {"drifts": drifts, "outcomes": outcomes, "gate_failed": gate_failed,
            "not_reported": not_reported, "not_reported_gated": not_reported_gated,
            "non_gate_ok": non_gate_ok,
            "invariants_ok": not gate_failed and non_gate_ok,
            "declared": len(declared)}
