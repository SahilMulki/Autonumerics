"""Operator expressions: term splitting, stencils, and a restricted namespace.

Shared by the reference check (this plan) and the D1 residual gate
(plan-no-closed-form) -- one implementation, so the two can never disagree about
what ``lap_u`` means. See §14 C1.

Three facts established by measurement, each of which the plan depends on:

* All 19 ``mms_probe.operator_check`` strings in ``workspace/`` split into signed
  top-level additive terms, which is what the term-balanced denominator needs.
* A single-term operator (``lap_u``) makes ``residual / max|term|`` identically 1,
  so it must be decomposed one level further.
* The namespace must be passed as **globals**, not locals: several real specs
  contain a lambda or comprehension, which cannot see a locals mapping.
"""

from __future__ import annotations

import ast
import math
from functools import partial

try:
    import numpy as np
except ImportError:  # pragma: no cover -- see _require_numpy
    np = None

# --- restricted evaluation ---------------------------------------------------
#
# numpy is optional *at import time* on purpose. This module ships inside a plugin
# whose hooks are executed by whatever `python3` the shell resolves -- not by the
# repo's own interpreter -- and there is no guarantee that one has numpy. Making
# the import hard would take the entire guard down, including the checks that need
# nothing but `ast`: expression-not-program, the leakage scans, the ledger shape.
# Those are the ones that catch the archived incidents, so they must survive an
# interpreter that cannot do arithmetic. See gate.check_spec_text.


def _require_numpy(what):
    if np is None:
        raise ModuleNotFoundError(
            f"{what} needs numpy, which is not available to this interpreter "
            f"({__import__('sys').executable}). The AST-only guards still run."
        )


def _erf(x):
    return np.vectorize(math.erf)(np.asarray(x, dtype=float))


def _safe_funcs():
    """Built on demand so importing this module never requires numpy."""
    _require_numpy("evaluating a spec expression")
    return {
        "np": np,
        "math": math,
        "erf": _erf,
        "erfc": lambda x: 1.0 - _erf(x),
        "ndtr": lambda x: 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / np.sqrt(2.0))),
    }


def evaluate(expr: str, names: dict):
    """Evaluate a spec-authored expression with no builtins.

    ``names`` is merged into the *globals* mapping, not passed as locals: an
    expression containing a lambda or a comprehension opens a new scope that
    cannot see a locals dict, and several real specs contain one.
    """
    safe = _safe_funcs()
    code = compile(ast.parse(expr, mode="eval"), "<spec-expression>", "eval")
    # errstate: a correct closed form legitimately divides by zero at a singular
    # boundary (Black-Scholes at t = T_maturity). numpy's warning path needs
    # __import__, which the emptied builtins do not have, so an unsuppressed
    # warning surfaces as a baffling KeyError('__import__') instead of a nan.
    with np.errstate(all="ignore"):
        return eval(code, {"__builtins__": {}, **safe, **names})  # noqa: S307


def assert_is_expression(expr: str, *, field: str) -> None:
    """An analytic solution is one evaluable expression, never a program.

    The archived ``pde_heston_2d`` spec put a 2063-character ``def`` with
    Gauss-Legendre quadrature in this field; this rejects it in one line.
    """
    try:
        ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"{field} must be a single evaluable expression, not a program "
            f"(got {len(expr)} chars; {exc.msg} at line {exc.lineno})"
        ) from exc


def is_program(text: str) -> bool:
    """True when ``text`` is valid Python but not a single expression.

    This is the exact shape of the archived Heston leak -- a ``def`` with
    Gauss-Legendre quadrature sitting in ``analytic_solution.expression``. Prose
    ("X ~ Normal(mu, sigma**2)") parses as neither and is not a program, which is
    what lets guidance fields keep the anti-program half of the check without
    hard-failing on legitimate descriptions of a matrix-valued coefficient.
    """
    try:
        ast.parse(text, mode="eval")
        return False  # a plain expression
    except SyntaxError:
        pass
    try:
        ast.parse(text, mode="exec")
        return True
    except SyntaxError:
        return False  # prose: neither expression nor program


# --- term splitting ----------------------------------------------------------

def split_terms(expr: str) -> list[tuple[int, str]]:
    """Split an operator expression into ``(sign, source)`` top-level terms.

    ``"u_t - alpha*lap_u"`` -> ``[(+1, "u_t"), (-1, "alpha*lap_u")]``.
    Measured: all 19 real operator strings split cleanly, 1 to 7 terms.
    """
    out: list[tuple[int, str]] = []

    def walk(node, sign):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            walk(node.left, sign)
            walk(node.right, sign * (1 if isinstance(node.op, ast.Add) else -1))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            walk(node.operand, -sign)
        else:
            out.append((sign, ast.unparse(node)))

    walk(ast.parse(expr, mode="eval").body, 1)
    return out


def normalize_helpers(expr: str) -> str:
    """``grad_u[0]`` -> ``grad_u_0`` so the component form is a plain name."""
    import re
    return re.sub(r"grad_u\[(\d)\]", r"grad_u_\1", expr)


# --- stencils ----------------------------------------------------------------
#
# Central-difference weights, keyed by order of accuracy. Index 0 of each tuple is
# the coefficient at offset -order/2, ascending to +order/2.
#
# The default stays **4** everywhere the reference check touches. reference.TOL,
# its PROBE_N escalation and the measured 16-of-20 Tier-A distribution are all
# calibrated against the 4th-order stencils, and silently sharpening them would
# move results the guardrails plan pins with tests. The 6th- and 8th-order
# variants are opt-in, and the only caller that opts in is the D1 residual, which
# must difference at higher order than the solver used (plan-no-closed-form §3b
# rule 1, §9b).

# Stored as **integer numerators over a common denominator**, not as floats, and
# summed before the single division. That is not tidiness: `1/12 - 2/3 + 2/3 -
# 1/12` does not cancel to zero in binary, it lands at 1.4e-17, and on
# `pde_burgers_inviscid` -- whose exact solution is a step function, so every
# derivative is identically zero away from the shock -- both the residual and its
# own term-balanced denominator become that 1.4e-17 and their ratio is 1. A
# formula that is exactly right then reports `failed`. The integer form cancels
# exactly, which is the property the original hard-coded 4th-order expressions had
# and the reference check's whole calibration rests on.
_D1_WEIGHTS = {
    2: ((-1, 0, 1), 2),
    4: ((1, -8, 0, 8, -1), 12),
    6: ((-1, 9, -45, 0, 45, -9, 1), 60),
    8: ((3, -32, 168, -672, 0, 672, -168, 32, -3), 840),
}

_D2_WEIGHTS = {
    2: ((1, -2, 1), 1),
    4: ((-1, 16, -30, 16, -1), 12),
    6: ((2, -27, 270, -490, 270, -27, 2), 180),
    8: ((-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9), 5040),
}

STENCIL_ORDERS = tuple(sorted(_D1_WEIGHTS))


def _apply(table, order, f, axis, scale):
    numerators, denominator = table[order]
    half = len(numerators) // 2
    out = None
    for k, w in enumerate(numerators):
        if w == 0:
            continue
        # np.roll(f, -s) shifts f[i+s] into position i, so offset s wants shift -s.
        term = w * (f if k == half else np.roll(f, half - k, axis))
        out = term if out is None else out + term
    return out / (denominator * scale)


def d1(f, axis, h, order=4):
    """Central first derivative, ``order``-th order accurate (default 4)."""
    _require_numpy("differencing a field")
    return _apply(_D1_WEIGHTS, order, f, axis, h)


def d2(f, axis, h, order=4):
    """Central second derivative, ``order``-th order accurate (default 4)."""
    _require_numpy("differencing a field")
    return _apply(_D2_WEIGHTS, order, f, axis, h * h)


STENCIL_HALO = 4  # points to exclude at each edge; np.roll wraps, so trim


def halo_for(order):
    """Edge points a stencil of this order corrupts by wrapping.

    ``STENCIL_HALO`` stays 4 because an 8th-order central stencil reaches exactly
    4 points, so the existing trim already covers every order offered here. This
    exists so a future 10th-order variant cannot silently reuse a halo that is one
    point too small.
    """
    if order == "spectral":
        return 0  # a spectral derivative is global; there is no corrupted edge
    return max(STENCIL_HALO, order // 2)


def spectral_d(f, axis, h, degree=1):
    """Fourier derivative along one axis of an endpoint-**exclusive** periodic grid.

    Exact to round-off for a band-limited field, which is what makes it the second
    member of the §5b insensitivity pair on a periodic problem: FD8 and a spectral
    derivative are as far apart in stencil family as this repo can get, so their
    agreeing is real evidence that the residual measures the solver rather than the
    check.

    The caller owns the endpoint convention. Passing an endpoint-**inclusive**
    array (whose last node duplicates the first) puts a full extra cell into the
    period and returns a derivative that is wrong everywhere, not just at the edge
    -- which is why :class:`Stencil` refuses to build a spectral member unless the
    grid was detected as exclusive.
    """
    _require_numpy("spectral differentiation")
    n = f.shape[axis]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=h)
    shape = [1] * f.ndim
    shape[axis] = n
    k = k.reshape(shape)
    if degree % 2 == 1 and n % 2 == 0:
        # The Nyquist mode has no meaningful odd derivative on a real field; leaving
        # it in makes d1 complex-valued in a way that shows up as noise at the grid
        # scale, which is exactly where the residual is being measured.
        k = np.where(np.abs(np.abs(k) - np.pi / h) < 1e-9 * np.pi / h, 0.0, k)
    return np.real(np.fft.ifft((1j * k) ** degree * np.fft.fft(f, axis=axis), axis=axis))


class Stencil:
    """One differencing family, so a caller can ask for the *same* derivative twice.

    ``order`` is 2, 4, 6, 8 or the string ``"spectral"``. ``periodic`` is a
    per-axis boolean sequence and is consulted only by the spectral family, which
    falls back to 8th-order FD on any axis that is not periodic-exclusive.
    """

    def __init__(self, order=4, periodic=None):
        if order != "spectral" and order not in _D1_WEIGHTS:
            raise ValueError(f"unknown stencil order {order!r}; expected one of "
                             f"{STENCIL_ORDERS} or 'spectral'")
        self.order = order
        self.periodic = tuple(periodic) if periodic is not None else None
        self.halo = halo_for(order if order != "spectral" or not self._all_periodic() else 8)

    def _all_periodic(self):
        return bool(self.periodic) and all(self.periodic)

    def _spectral_on(self, axis):
        return self.order == "spectral" and self.periodic is not None \
            and axis < len(self.periodic) and self.periodic[axis]

    def d1(self, f, axis, h):
        if self._spectral_on(axis):
            return spectral_d(f, axis, h, degree=1)
        return d1(f, axis, h, order=8 if self.order == "spectral" else self.order)

    def d2(self, f, axis, h):
        if self._spectral_on(axis):
            return spectral_d(f, axis, h, degree=2)
        return d2(f, axis, h, order=8 if self.order == "spectral" else self.order)

    def __repr__(self):
        return f"Stencil(order={self.order!r})"


#: The family every pre-existing caller gets. Constructing it needs no numpy.
_MODULE_STENCIL = Stencil(4)


def default_stencil():
    return _MODULE_STENCIL


def _derivative_helpers(name, a, spatial_vars, hs, lap, stencil=None):
    """Every spelling of a derivative of one field that real operator strings use."""
    nd = len(spatial_vars)
    st = stencil or _MODULE_STENCIL
    ns = {name: a}
    ns[f"lap_{name}"] = lap(a)
    ns[f"lap_lap_{name}"] = lap(lap(a))
    ns[f"lap_{name}3"] = lap(a ** 3)
    g = [st.d1(a, i, hs[i]) for i in range(nd)]
    ns[f"grad_{name}"] = g[0] if nd == 1 else np.array(g)
    for i in range(nd):
        ns[f"grad_{name}_{i}"] = g[i]
    for i, v in enumerate(spatial_vars):
        ns[f"{name}_{v}"] = g[i]
        ns[f"{name}_{v}{v}"] = st.d2(a, i, hs[i])
    # Mixed second derivatives, for every axis pair. pde_monge_ampere_2d needs u_xy;
    # Heston needs u_Sv, which is the same helper on non-Cartesian axis names.
    for i, vi in enumerate(spatial_vars):
        for j, vj in enumerate(spatial_vars):
            if i < j:
                mixed = st.d1(st.d1(a, i, hs[i]), j, hs[j])
                ns[f"{name}_{vi}{vj}"] = mixed
                ns[f"{name}_{vj}{vi}"] = mixed
    return ns


def coord_aliases(coords):
    """``coords`` plus the capitalised spelling of each axis.

    Several specs write the meshgrid capitalised -- ``"np.sin(X)*np.cos(Y)"`` --
    while ``spatial_variables`` stays lower-case. Both name the same array, and a
    claimed solution is evaluated outside :func:`build_namespace`, so the aliasing
    has to be available on its own.
    """
    out = dict(coords)
    for name, value in coords.items():
        out.setdefault(name.upper(), value)
    return out


def build_namespace(fields, coords, spatial_vars, hs, *, time_derivs=None, stencil=None):
    """Helper names for one or many fields, plus the callable spellings.

    ``fields`` maps a field name to its array. A scalar problem passes ``{"u": u}``;
    a system passes every component, because a system's momentum equation needs
    ``v`` and ``p`` to express the advection and pressure-gradient terms.

    Two vocabularies for the same object already coexist in ``workspace/`` --
    ``lap_lap_u`` (Cahn-Hilliard) and ``lap(lap_u)`` (Kuramoto-Sivashinsky) -- so
    both are supplied. Non-Cartesian axis names come free: the helpers are built
    from ``spatial_vars``, so ``["S", "v"]`` yields ``u_SS``, ``u_Sv``, ``u_vv``.
    """
    nd = len(spatial_vars)
    time_derivs = time_derivs or {}
    # `stencil` is how the D1 residual differences at *higher order than the solver
    # used* (§3b rule 1). Default None keeps every pre-existing caller -- the whole
    # reference check -- on the 4th-order stencils its tolerances are calibrated for.
    st = stencil or _MODULE_STENCIL

    def lap(a):
        return sum(st.d2(a, i, hs[i]) for i in range(nd))

    ns = coord_aliases(coords)
    ns["lap"] = lap
    ns["lap2"] = lambda a: lap(lap(a))
    for i, v in enumerate(spatial_vars):
        ns[f"d_{v}"] = partial(st.d1, axis=i, h=hs[i])
        ns[f"d_{v}{v}"] = partial(st.d2, axis=i, h=hs[i])
        ns[f"d{v}"] = ns[f"d_{v}"]

    # dt(u) is resolved by identity: the time derivative of an arbitrary expression
    # would need the whole expression re-evaluated at t +/- h, but nothing writes
    # dt(u*v) -- every real spelling is dt of a bare field.
    by_id = {}
    for name, a in fields.items():
        ns.update(_derivative_helpers(name, a, spatial_vars, hs, lap, st))
        dts = time_derivs.get(name) or (None, None)
        if dts[0] is not None:
            ns[f"{name}_t"] = dts[0]
            by_id[id(a)] = dts[0]
        if dts[1] is not None:
            ns[f"{name}_tt"] = dts[1]

    def dt(a):
        got = by_id.get(id(a))
        if got is None:
            raise ValueError(
                "dt() is defined on a declared field only (dt(u)), not on a general "
                "expression. Write the time derivative of a field, or name the term."
            )
        return got

    ns["dt"] = dt
    return ns


def helper_namespace(u, coords, spatial_vars, hs, *, u_t=None, u_tt=None, stencil=None):
    """Scalar convenience wrapper over :func:`build_namespace`."""
    return build_namespace({"u": u}, coords, spatial_vars, hs,
                           time_derivs={"u": (u_t, u_tt)}, stencil=stencil)
