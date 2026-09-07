"""Nested grid ladders: refine, detect the endpoint convention, restrict, norm.

Transcribed from ``verification_manual.md`` §3, which is already written as a
reference implementation. The one thing worth restating is *why* the ladder has to
nest: ``np.linspace(a, b, N)`` and ``np.linspace(a, b, 2N)`` share no interior
nodes, so differencing two numerical solutions across them needs interpolation,
whose error contaminates the very order estimate the difference exists to produce.
Halving ``dx`` instead makes ``fine[::2] == coarse`` exactly, and which arithmetic
does that depends on the endpoint convention -- which is detected from the grid the
solver returns, never assumed.
"""

from __future__ import annotations

import numpy as np


def refine(N, periodic):
    """The next resolution up the ladder. Exactly halves ``dx``."""
    return 2 * N if periodic else 2 * N - 1


def detect_periodic(axis, bounds):
    """True when the returned axis excludes its right endpoint."""
    right = float(bounds[1])
    return not np.isclose(float(axis[-1]), right, rtol=0.0, atol=1e-12 * max(1.0, abs(right)))


def build_ladder(N0, periodic, levels=3):
    out = [int(N0)]
    for _ in range(int(levels) - 1):
        out.append(refine(out[-1], periodic))
    return out


def restrict(u_fine, fine_axes, coarse_axes, atol=1e-10):
    """Sample ``u_fine`` at the coarse nodes. ``None`` when the grids do not nest."""
    index = []
    for coarse, fine in zip(coarse_axes, fine_axes, strict=True):
        j = np.abs(np.asarray(fine)[None, :] - np.asarray(coarse)[:, None]).argmin(axis=1)
        if not np.allclose(np.asarray(fine)[j], coarse, rtol=0.0,
                           atol=atol * max(1.0, float(np.ptp(coarse)))):
            return None
        index.append(j)
    return u_fine[np.ix_(*index)]


def interpolate_to(u_fine, fine_axes, coarse_axes):
    """The fallback when the grids do not nest: cubic, i.e. of order strictly above
    any scheme this repo tests. Sets ``interpolated`` in the metrics, and an order
    estimate carrying it may not certify a 10 (manual §3)."""
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(tuple(np.asarray(a) for a in fine_axes), u_fine,
                                     method="cubic", bounds_error=False, fill_value=None)
    mesh = np.meshgrid(*[np.asarray(a) for a in coarse_axes], indexing="ij")
    return interp(np.stack([m.ravel() for m in mesh], axis=-1)).reshape(mesh[0].shape)


def rms(a, mask=None):
    a = np.asarray(a, dtype=float)
    a = a if mask is None else a[mask]
    return float(np.sqrt(np.mean(a ** 2)))


def rel_err(numeric, exact, metric="l2", mask=None):
    d = np.asarray(numeric, dtype=float) - np.asarray(exact, dtype=float)
    ref = np.asarray(exact, dtype=float)
    if mask is not None:
        d, ref = d[mask], ref[mask]
    if metric == "l1":
        return float(np.mean(np.abs(d)) / (np.mean(np.abs(ref)) + 1e-14))
    return float(np.sqrt(np.mean(d ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-14))


def axes_of(result, axis_names):
    """The 1-D coordinate arrays a solve returned, in the spec's axis order."""
    grid = result.get("grid") or {}
    ordered = [grid[name] for name in axis_names if name in grid]
    if ordered:
        return ordered
    return [np.asarray(v) for v in grid.values() if np.ndim(v) == 1]


def coords_of(result, axis_names):
    return np.meshgrid(*axes_of(result, axis_names), indexing="ij")


def fields_of(result):
    return result.get("fields") or {}


def primary_of(result, primary=None):
    fields = fields_of(result)
    if primary and primary in fields:
        return np.asarray(fields[primary], dtype=float)
    return np.asarray(next(iter(fields.values())), dtype=float)


def eval_on_axes(exprs, axes, axis_names, params, t):
    """Evaluate ``{field: expression}`` on the mesh a solve returned.

    The mesh comes from the solver, never from the spec's bounds: a solver may
    return an endpoint-inclusive grid where the spec's convention is exclusive, and
    comparing a manufactured solution against the wrong nodes is a silent O(1)
    error that looks exactly like a failing scheme.
    """
    from ..operator import coord_aliases, evaluate
    mesh = np.meshgrid(*[np.asarray(a, dtype=float) for a in axes], indexing="ij")
    names = coord_aliases(dict(zip(axis_names, mesh, strict=False)))
    ones = np.ones_like(mesh[0])
    ns = {**(params or {}), **names, "t": float(t)}
    return {k: np.asarray(evaluate(e, ns), dtype=float) * ones for k, e in exprs.items()}


#: A grid whose spacing varies by more than this is not uniform enough for a
#: fixed-h stencil.
UNIFORM_RTOL = 1e-6


def is_uniform(axis, rtol=UNIFORM_RTOL):
    """Is this coordinate array equally spaced?

    Not a formality. ``pde_convection_diffusion_bl``'s Shishkin and Bakhvalov plans
    return graded meshes with spacing ratios of 115:1 and 485:1 -- exactly the right
    thing to do for a boundary layer, and exactly what makes ``h = axis[1] -
    axis[0]`` meaningless. Differencing those with a uniform-h stencil produced a
    residual that did not fall under refinement, so D1 reported ``stalled`` and the
    score dropped from 10 to 3 on two correct solvers. The stencils are the limit
    there, not the scheme, so the honest outcome is ``unavailable``.
    """
    h = np.diff(np.asarray(axis, dtype=float))
    if h.size == 0:
        return True
    scale = float(np.max(np.abs(h)))
    return bool(scale > 0 and float(np.max(np.abs(h - h[0]))) <= rtol * scale)


def uniform_axes(axes):
    return all(is_uniform(a) for a in axes)
