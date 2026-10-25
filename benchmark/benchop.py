"""Harness-owned derivations of the BENCHOP-SLV reference values.

BENCHOP-SLV is an external option-pricing benchmark (von Sydow et al., 2018; Uppsala
University) whose **24 reference values** were computed by a third party, before this
project existed, by methods with no relationship to anything here. That is the whole
point of adopting it: ``problems.py`` already holds an answer key the pipeline cannot
read, but that key and the pipeline share an author and a repo. A published number
does not.

Adopting a published number still requires work, because a transcribed constant is
only as good as the reading of the problem it came from. This file is that work: for
every one of the 24 values it carries an *independent* computation, so that what the
benchmark anchors on is a number this repo has reproduced rather than one it has
copied. Seven of the eight problems are derived by two different method *families* --
a quadrature and a grid, or a grid and a sampler -- so that a shared mistake has to be
made twice in two different languages. The eighth, ``hhw_eucall1``, has three
correlated factors and so admits neither a quadrature nor a tau-independent grid; it
rests on a sampler alone, and that is the one gap in this file.

Nothing here is loaded during a benchmark run, and ``pipeline-settings.json`` denies
the whole of ``benchmark/`` to the agents.

    uv run python benchmark/benchop.py --list
    uv run python benchmark/benchop.py --only hst_eucall
    uv run python benchmark/benchop.py --all              # writes the ledger; the
                                                          # stored one's per-problem
                                                          # wall times sum to 1532 s
    uv run python benchmark/benchop.py --all --check      # recompute, write nothing
    uv run python benchmark/benchop.py --all --fast       # coarse: a smoke test, not evidence
    uv run python benchmark/benchop.py --calibrate        # step 8: correct vs wrong,
                                                          # at each entry's own settings

The ledger it writes, ``benchmark/references/benchop_slv.json``, is what the problem
entries are built from: the published value, the absolute uncertainty that value
*itself* carries, and the measured agreement of every route that was run against it.

Sources
-------
Formulations (5 pp.)
    https://it.uu.se/research/scientific_computing/project/compfin/benchop/SLV/benchop_slv_0622.pdf
Reference values, as ``pr(i).refval``
    https://it.uu.se/research/scientific_computing/project/compfin/benchop/SLV/unitTestBenchopSLV.m
Paper
    https://ir.cwi.nl/pub/28249/28249.pdf  (DOI 10.1080/00207160.2018.1544368)

The values are in the MATLAB unit-test harness, not in the paper or the formulations:
the paper reports error-against-CPU-time curves only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.integrate import IntegrationWarning, quad
from scipy.interpolate import CubicSpline
from scipy.special import erf, ive

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "references")
LEDGER = os.path.join(REF_DIR, "benchop_slv.json")

# =============================================================================
# The published problem set
# =============================================================================
#
# Shared by both local-stochastic-volatility problems:
#   dS = r S dtau + sqrt(V) f(S) dW1,  dV = kappa (eta - V) dtau + sigma sqrt(V) dW2
#   f(s) = alpha s^2 / 2 + beta s + gamma;  Heston is f(s) = s, QLSV is f(s) = 0.01 s^2
# Feller is violated in every problem in the suite: 2 kappa eta = 0.221880 against
# sigma^2 = 1 here, and 2 kappa eta = 0.04 against sigma1^2 = 1 for Heston-Hull-White.

LSV = dict(T=1.0, r=0.0, kappa=2.58, eta=0.043, sigma=1.0, rho=-0.36, v0=0.114,
           K=100.0, spots=(75.0, 100.0, 125.0), barriers=(50.0, 150.0))

HHW = dict(T=10.0, kappa=0.5, eta=0.04, sigma1=1.0, sigma2=0.09, a=0.08, b=0.10,
           V0=0.04, R0=0.10, K=100.0, rho12=-0.9, spots=(75.0, 100.0, 125.0))

# SABR is a forward price with no drift: dS = sigma S^beta dW_S, dsigma = alpha sigma dW_sigma.
# The strikes are K_i = S0 exp(0.1 sqrt(T) delta_i) for delta = -1, 0, 1, which is how
# the transcription below was checked rather than trusted.
SABR = {
    "I": dict(T=2.0, S0=0.5, sig0=0.5, alpha=0.4, beta=0.5, rho=0.0),
    "II": dict(T=10.0, S0=0.07, sig0=0.4, alpha=0.8, beta=0.5, rho=-0.6),
}

PUBLISHED = {
    "sabr_set1": dict(
        title="SABR set I, European call", labels=("K_lo", "K_mid", "K_hi"),
        values=(0.221383196830866, 0.193836689413803, 0.166240814653231),
        abs_err=0.0),
    "sabr_set2": dict(
        title="SABR set II, European call", labels=("K_lo", "K_mid", "K_hi"),
        values=(0.052450313614407, 0.046585753491306, 0.039291470612989),
        abs_err=0.0),
    "hst_eucall": dict(
        title="Heston, European call K=100", labels=("S75", "S100", "S125"),
        values=(0.908502728459621, 9.046650119220969, 28.514786399298796),
        abs_err=0.0),
    "hst_dntouch": dict(
        title="Heston, double-no-touch L=50 U=150", labels=("S75", "S100", "S125"),
        values=(0.834539127387590, 0.899829293208866, 0.668399975738358),
        abs_err=0.0),
    "qlsv_eucall": dict(
        title="Quadratic LSV, European call K=100", labels=("S75", "S100", "S125"),
        values=(0.527472759419533, 8.902347915743665, 29.159828965633729),
        abs_err=0.0),
    "qlsv_dntouch": dict(
        title="Quadratic LSV, double-no-touch L=50 U=150", labels=("S75", "S100", "S125"),
        values=(0.933800903110254, 0.914799140676374, 0.592983062889906),
        abs_err=0.0),
    "hhw_eucall1": dict(
        title="Heston-Hull-White set I (full correlation), European call K=100",
        labels=("S75", "S100", "S125"),
        values=(37.52, 57.41, 78.58),
        # The only value in the suite not quoted to 15 digits: the source annotates it
        # %1E-02, so half a unit in the last place quoted is the floor on any gate
        # built from it, however accurate an independent derivation becomes.
        abs_err=5e-3),
    "hhw_eucall2": dict(
        title="Heston-Hull-White set II (zero rate correlation), European call K=100",
        labels=("S75", "S100", "S125"),
        values=(35.437896876285350, 54.728065308229503, 75.397596834993621),
        abs_err=0.0),
}


def sabr_strikes(which, p=None):
    p = p or SABR[which]
    return tuple(p["S0"] * np.exp(0.1 * np.sqrt(p["T"]) * d) for d in (-1.0, 0.0, 1.0))


# =============================================================================
# Route A -- semi-closed: the Heston characteristic function
# =============================================================================


def heston_cf(u, tau, kappa, eta, sigma, rho, v0):
    """CF of ln(S_T/S_0) - r*tau under Heston, in Albrecher's "little trap" branch.

    The branch matters: the textbook ``(b - rho sigma i u + d)`` form crosses the cut
    of the complex logarithm at long maturities and silently returns a different
    function. Every maturity here is 1 or 10 years, so it would."""
    rsu = rho * sigma * 1j * u
    d = np.sqrt((rsu - kappa) ** 2 + sigma ** 2 * (1j * u + u ** 2))
    num = kappa - rsu - d
    c = num / (kappa - rsu + d)
    e = np.exp(-d * tau)
    D = (num / sigma ** 2) * (1.0 - e) / (1.0 - c * e)
    C = (kappa * eta / sigma ** 2) * (num * tau
                                      - 2.0 * np.log((1.0 - c * e) / (1.0 - c)))
    return np.exp(C + D * v0)


def _lewis(k, phi, scale, upper=None, tol=1e-14):
    """Lewis' single-integral call:  S0 - scale/pi * int Re[e^{iuk} phi(u - i/2)]/(u^2+1/4).

    One integral rather than the two of the P1/P2 form, and far better conditioned:
    measured against the published Heston values, the two-integral form reaches
    4e-12 and this reaches 3e-14 -- the same quadrature tolerance in both."""
    def f(u):
        return (np.exp(1j * u * k) * phi(u)).real / (u * u + 0.25)
    edges = np.linspace(0.0, upper or 200.0, 9)
    tot = 0.0
    with warnings.catch_warnings():
        # The tolerance is deliberately below what the integrand can deliver in
        # double precision; QUADPACK says so on the last panel and is right to.
        warnings.simplefilter("ignore", IntegrationWarning)
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            val, _ = quad(f, lo, hi, limit=500, epsabs=tol, epsrel=tol)
            tot += val
    return -scale / np.pi * tot


def heston_call(S0, K, tau, kappa, eta, sigma, rho, v0, r=0.0):
    """European call under Heston. Exact up to quadrature."""
    k = np.log(S0 / K) + r * tau

    def phi(u):
        return heston_cf(u - 0.5j, tau, kappa, eta, sigma, rho, v0)
    return S0 + _lewis(k, phi, np.sqrt(S0 * K) * np.exp(-r * tau / 2.0))


def hw_integrated_rate(p):
    """M = int_0^T R dtau for Hull-White with a constant b: Gaussian, both moments closed."""
    a, s2, T, b, R0 = p["a"], p["sigma2"], p["T"], p["b"], p["R0"]
    mu = b * T + (R0 - b) * (1.0 - np.exp(-a * T)) / a
    var = (s2 ** 2 / a ** 2) * (T - 2.0 * (1.0 - np.exp(-a * T)) / a
                                + (1.0 - np.exp(-2.0 * a * T)) / (2.0 * a))
    return float(mu), float(var)


def hhw2_call(S0, p):
    """Heston-Hull-White with rho13 = rho23 = 0, in closed form up to one integral.

    With the rate independent of (S, V), write M = int R dtau. Then
    exp(-M)(S_T - K)^+ = (S0 E_T - K exp(-M))^+ with E_T the zero-rate Heston asset
    ratio, so the price is E_M[ HestonCall(S0, K exp(-M)) ] -- and the obvious route
    is to quadrature that expectation over M.

    Do not. The outer integration can be collapsed analytically instead: in Lewis'
    integrand the strike enters only through sqrt(K') and exp(iu ln(S0/K')), so
    replacing K' by K exp(-M) and taking the expectation inside turns the whole
    Hull-White factor into the Gaussian moment generating function E[exp(zM)] at
    z = -1/2 + iu. One integral, not two, and its integrand now carries
    exp(-var(M) u^2 / 2), which is why it terminates by u = 12.

    Measured: the outer-quadrature route plateaus at 3e-5 relative at 320
    Gauss-Hermite nodes -- it is the *inner* Heston call at strikes spanning
    K exp(-M) over nine orders of magnitude that fails, not the node count -- while
    this one reaches 6e-16."""
    mu, var = hw_integrated_rate(p)
    k0 = np.log(S0 / p["K"])

    def phi(u):
        z = -0.5 + 1j * u
        return (np.exp(mu * z + 0.5 * var * z * z)
                * heston_cf(u - 0.5j, p["T"], p["kappa"], p["eta"], p["sigma1"],
                            p["rho12"], p["V0"]))
    return S0 + _lewis(k0, phi, np.sqrt(S0 * p["K"]), upper=40.0)


# =============================================================================
# Route B -- a grid: two-dimensional finite differences
# =============================================================================
#
# Backward equation for the LSV family in time-to-maturity tau, r = 0:
#
#   u_tau = 1/2 V f(S)^2 u_SS + rho sigma V f(S) u_SV + 1/2 sigma^2 V u_VV
#           + kappa (eta - V) u_V


def fdw(xs, x0, m):
    """Fornberg weights for d^m/dx^m at x0 from the nodes xs, any spacing.

    Second-order one-sided stencils are the whole reason this is here. The first
    version of this solver used a one-sided *first-order* difference for u_V on the
    degenerate V = 0 row, which is exactly where a Feller-violating CIR keeps its
    density; replacing it took the Heston double-no-touch error at 400x160 from
    4.5e-4 to 1.4e-4."""
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    c = np.zeros((n, m + 1))
    c1, c4 = 1.0, xs[0] - x0
    c[0, 0] = 1.0
    for i in range(1, n):
        mn, c2, c5, c4 = min(i, m), 1.0, c4, xs[i] - x0
        for j in range(i):
            c3 = xs[i] - xs[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c[:, m]


def sinh_grid(lo, hi, n, focus, c=20.0):
    """Grid on [lo, hi] clustered at ``focus``, which is itself a node."""
    a, b = np.arcsinh((lo - focus) / c), np.arcsinh((hi - focus) / c)
    return focus + c * np.sinh(a + (b - a) * np.linspace(0.0, 1.0, n + 1))


def v_grid(vmax, nv, c=4.0):
    """Clustered at V = 0, where a Feller-violating CIR piles its density up."""
    return vmax * np.sinh(c * np.linspace(0.0, 1.0, nv + 1)) / np.sinh(c)


def _axis_stencils(X, drift, diff, left, right):
    """Per-node (first, second) derivative stencils, second order throughout.

    ``drift`` and ``diff`` are per-node coefficients, consulted only to decide where
    a central first difference would be unstable and an upwind one is needed.
    ``left``/``right`` name the end treatment: ``dirichlet`` (not an unknown),
    ``degenerate`` (no diffusion there, so no condition is needed and the row is
    one-sided advection) or ``linear`` (u_XX = 0)."""
    n = len(X) - 1
    out = []
    for j in range(n + 1):
        if (j == 0 and left == "dirichlet") or (j == n and right == "dirichlet"):
            out.append(None)
            continue
        if j == 0:
            s1 = [0, 1, 2]
        elif j == n:
            s1 = [n - 2, n - 1, n]
        else:
            h = max(X[j] - X[j - 1], X[j + 1] - X[j])
            if diff[j] > 0 and abs(drift[j]) * h <= 2.0 * diff[j]:
                s1 = [j - 1, j, j + 1]
            elif drift[j] >= 0:
                s1 = [j, j + 1, j + 2] if j + 2 <= n else [j - 1, j, j + 1]
            else:
                s1 = [j - 2, j - 1, j] if j >= 2 else [j - 1, j, j + 1]
        if 0 < j < n:
            s2, w2 = [j - 1, j, j + 1], fdw(X[j - 1:j + 2], X[j], 2)
        else:
            s2, w2 = [j], np.zeros(1)
        out.append((s1, fdw(X[s1], X[j], 1), s2, w2))
    return out


def fd2d_operator(X, Y, coef, x_ends=("dirichlet", "dirichlet"),
                  y_ends=("degenerate", "linear")):
    """Sparse ``A`` and the Dirichlet lift ``G`` for

        u_tau = a_xx u_xx + a_xy u_xy + a_yy u_yy + b_x u_x + b_y u_y

    on the tensor grid ``X x Y``. ``coef`` maps the meshed coordinates to those five
    arrays. End treatments are as in :func:`_axis_stencils`."""
    nx, ny = len(X) - 1, len(Y) - 1
    Xg, Yg = X[:, None] * np.ones((1, ny + 1)), np.ones((nx + 1, 1)) * Y[None, :]
    c = coef(Xg, Yg)
    axx, axy, ayy, bx, by = (np.broadcast_to(np.asarray(c[k], dtype=float), Xg.shape)
                             for k in ("a_xx", "a_xy", "a_yy", "b_x", "b_y"))
    # Upwinding is decided from the coefficients along the *same* axis, averaged
    # across the other one -- the stencil is per axis, so the decision has to be too.
    xst = _axis_stencils(X, bx.mean(axis=1), axx.mean(axis=1), *x_ends)
    yst = _axis_stencils(Y, by.mean(axis=0), ayy.mean(axis=0), *y_ends)
    live = [i for i in range(nx + 1) if xst[i] is not None]
    pos = {i: k for k, i in enumerate(live)}
    n = len(live) * (ny + 1)
    rows, cols, vals, grow, gcol, gval = [], [], [], [], [], []

    def emit(r, i, j, w):
        if w == 0.0:
            return
        if i in pos:
            rows.append(r)
            cols.append(pos[i] * (ny + 1) + j)
            vals.append(w)
        else:                                  # a Dirichlet column: move it to the rhs
            grow.append(r)
            gcol.append(i)
            gval.append(w)

    for i in live:
        s1x, w1x, s2x, w2x = xst[i]
        for j in range(ny + 1):
            r = pos[i] * (ny + 1) + j
            s1y, w1y, s2y, w2y = yst[j]
            for jj, w in zip(s1y, w1y, strict=True):
                emit(r, i, jj, by[i, j] * w)
            for jj, w in zip(s2y, w2y, strict=True):
                emit(r, i, jj, ayy[i, j] * w)
            for ii, w in zip(s1x, w1x, strict=True):
                emit(r, ii, j, bx[i, j] * w)
            for ii, w in zip(s2x, w2x, strict=True):
                emit(r, ii, j, axx[i, j] * w)
            if axy[i, j]:
                for ii, wx in zip(s1x, w1x, strict=True):
                    for jj, wy in zip(s1y, w1y, strict=True):
                        emit(r, ii, jj, axy[i, j] * wx * wy)
    A = sp.csc_matrix(sp.coo_matrix((vals, (rows, cols)), shape=(n, n)))
    G = sp.csc_matrix(sp.coo_matrix((gval, (grow, gcol)), shape=(n, nx + 1)))
    return A, G, np.array(live)


def lsv_operator(S, V, f, p, s_left="dirichlet", s_right="dirichlet"):
    """The LSV family's operator, r = 0."""
    kappa, eta, sig, rho = p["kappa"], p["eta"], p["sigma"], p["rho"]

    def coef(Sg, Vg):
        fS = f(Sg)
        return dict(a_xx=0.5 * Vg * fS ** 2, a_xy=rho * sig * Vg * fS,
                    a_yy=0.5 * sig ** 2 * Vg, b_x=np.zeros_like(Sg),
                    b_y=kappa * (eta - Vg))
    return fd2d_operator(S, V, coef, (s_left, s_right), ("degenerate", "linear"))


def cn_march(A, G, u0, boundary, T, nt, rannacher=4):
    """Crank-Nicolson with a Rannacher start -- every payoff here is non-smooth, and
    a plain CN start leaves the kink ringing in the solution for the whole run. The
    operator does not depend on tau, so one factorization serves every step."""
    g = G @ np.asarray(boundary, dtype=float)
    u = np.asarray(u0, dtype=float).reshape(-1)
    dt = T / nt
    identity = sp.identity(A.shape[0], format="csc")
    lu = spl.splu((identity - 0.5 * dt * A).tocsc(), permc_spec="MMD_AT_PLUS_A")
    B = (identity + 0.5 * dt * A).tocsc()
    for k in range(nt):
        if k < rannacher:
            for _ in range(2):               # two implicit-Euler halves
                u = lu.solve(u + 0.5 * dt * g)
        else:
            u = lu.solve(B @ u + dt * g)
    return u


def lsv_solve(payoff, f, S, V, nt, p, s_left="dirichlet", s_right="dirichlet",
              boundary=None, rannacher=4):
    A, G, live = lsv_operator(S, V, f, p, s_left, s_right)
    nV = len(V) - 1
    u0 = np.asarray(payoff(S[live][:, None] * np.ones((1, nV + 1))), float)
    bnd = np.zeros(len(S)) if boundary is None else boundary
    u = cn_march(A, G, u0, bnd, p["T"], nt, rannacher)
    return u.reshape(len(live), nV + 1), S[live], V


def sabr_operator(S, Y, p):
    """SABR in ``(S, y = ln sigma)``, which is the chart that makes the volatility
    axis a constant-coefficient one: ``dy = -alpha^2/2 dtau + alpha dW_sigma``. On
    the raw sigma axis the grid would have to span e^{-16} to e^{14} for set II."""
    alpha, beta, rho = p["alpha"], p["beta"], p["rho"]

    def coef(Sg, Yg):
        sig = np.exp(Yg)
        return dict(a_xx=0.5 * sig ** 2 * Sg ** (2.0 * beta),
                    a_xy=rho * alpha * sig * Sg ** beta,
                    a_yy=np.full_like(Sg, 0.5 * alpha ** 2),
                    b_x=np.zeros_like(Sg),
                    b_y=np.full_like(Sg, -0.5 * alpha ** 2))
    return fd2d_operator(S, Y, coef, ("dirichlet", "dirichlet"), ("linear", "linear"))


def sabr_pde(which, ns, ny, nt, smax=None, width=5.0, p=None):
    """European calls at all three strikes, by finite differences.

    S = 0 is Dirichlet because beta = 1/2 makes the origin attainable and absorbing,
    which is a property of the model rather than a convenience; S = smax is
    Dirichlet at the intrinsic value because a CEV process with beta < 1 *is* a
    true martingale, so the linear far field is the right one here -- the opposite
    of the QLSV call, where it is the thing that goes wrong."""
    p = p or SABR[which]
    smax = smax or 40.0 * p["S0"]
    y0 = np.log(p["sig0"])
    half = width * p["alpha"] * np.sqrt(p["T"])
    S = sinh_grid(0.0, smax, ns, p["S0"], c=2.0 * p["S0"])
    Y = np.linspace(y0 - half, y0 + half, ny + 1)
    A, G, live = sabr_operator(S, Y, p)
    out = []
    for K in sabr_strikes(which, p):
        u0 = np.maximum(S[live][:, None] - K, 0.0) * np.ones((1, len(Y)))
        bnd = np.zeros(len(S))
        bnd[-1] = S[-1] - K
        u = cn_march(A, G, u0, bnd, p["T"], nt).reshape(len(live), len(Y))
        out.append(value_at(u, S[live], Y, p["S0"], y0))
    return np.array(out)


def _lagrange(X, x0, k):
    j = int(np.clip(np.searchsorted(X, x0) - k // 2, 0, len(X) - k))
    idx = np.arange(j, j + k)
    return idx, fdw(X[idx], x0, 0)


def value_at(u, Sg, Vg, S0, v0, k=6):
    """Tensor Lagrange read-off, high enough order never to be the error term."""
    si, sw = _lagrange(Sg, S0, min(k, len(Sg)))
    vi, vw = _lagrange(Vg, v0, min(k, len(Vg)))
    return float(sw @ u[np.ix_(si, vi)] @ vw)


def f_heston(s):
    return s


def f_qlsv(s):
    return 0.01 * s * s


def lsv_pde(kind, model, ns, nv, nt, p=LSV, vmax=3.0, smax=1600.0):
    """One grid level of the LSV family; ``kind`` is ``eucall`` or ``dntouch``."""
    f = f_heston if model == "hst" else f_qlsv
    V = v_grid(vmax, nv)
    if kind == "dntouch":
        lo, hi = p["barriers"]
        S = np.linspace(lo, hi, ns + 1)
        u, Sg, Vg = lsv_solve(np.ones_like, f, S, V, nt, p)
    else:
        S = sinh_grid(0.0, smax, ns, p["K"])
        u, Sg, Vg = lsv_solve(lambda x: np.maximum(x - p["K"], 0.0), f, S, V, nt, p,
                              s_right="linear", boundary=np.zeros(len(S)))
    return np.array([value_at(u, Sg, Vg, s, p["v0"]) for s in p["spots"]])


# =============================================================================
# Shared: control variates
# =============================================================================


def control(y, controls, exact):
    """Regression control variate: subtract ``beta . (c - E[c])`` with ``E[c]``
    *known*, not sampled, so the estimator stays unbiased."""
    A = np.vstack([np.asarray(c) - e for c, e in zip(controls, exact, strict=True)]).T
    beta = np.linalg.lstsq(A - A.mean(0), y - y.mean(), rcond=None)[0]
    return y - A @ beta


# =============================================================================
# Route C -- a sampler: Monte Carlo
# =============================================================================


def qe_step(v, dt, kappa, eta, sigma, rng):
    """Andersen's quadratic-exponential CIR step, which is what makes a
    Feller-violating variance simulable at all: it matches the first two moments of
    the exact non-central chi-square and keeps the mass that piles up on zero."""
    e = np.exp(-kappa * dt)
    m = eta + (v - eta) * e
    if sigma == 0.0:
        return m
    s2 = ((v * sigma ** 2 * e * (1.0 - e) / kappa
           + eta * sigma ** 2 * (1.0 - e) ** 2 / (2.0 * kappa)) if kappa > 0
          else v * sigma ** 2 * dt)
    psi = s2 / np.maximum(m * m, 1e-300)
    out = np.empty_like(v)
    lo = psi <= 1.5
    if lo.any():
        q = psi[lo]
        b2 = 2.0 / q - 1.0 + np.sqrt(2.0 / q) * np.sqrt(2.0 / q - 1.0)
        out[lo] = (m[lo] / (1.0 + b2)) * (np.sqrt(b2)
                                          + rng.standard_normal(int(lo.sum()))) ** 2
    hi = ~lo
    if hi.any():
        q = (psi[hi] - 1.0) / (psi[hi] + 1.0)
        u = rng.random(int(hi.sum()))
        out[hi] = np.where(u <= q, 0.0,
                           np.log((1.0 - q) / (1.0 - u)) * m[hi] / (1.0 - q))
    return out


def _log_advance(x, iv, step):
    return x - 0.5 * iv + step


def _inv_advance(x, iv, step):
    """Drift-implicit: ``x_{n+1} = x_n + iv/x_{n+1} + step``, negative root.

    Not a refinement -- a correctness condition. In this chart ``-x`` is a Bessel
    process of dimension 3, whose drift blows up at the origin; an explicit step
    walks straight through it and hands back a *negative asset price*. The implicit
    one cannot, since the root it takes has the sign built in. Measured at
    S0 = 125, the two steps differ by 0.02 and the difference does not shrink with
    the step count; the implicit one is the one that reproduces the frozen-variance
    closed form in :func:`frozen_qlsv_call`."""
    c = x + step
    return 0.5 * (c - np.sqrt(c * c + 4.0 * iv))


# Lamperti charts: (to_x, from_x, advance). The point of the chart is that the
# diffusion coefficient becomes the constant sqrt(V), which is what lets a barrier be
# priced by a Brownian bridge rather than by discrete monitoring.
def _recip(s):
    return -100.0 / s


CHART_HESTON = (np.log, np.exp, _log_advance)
CHART_QLSV = (_recip, _recip, _inv_advance)


def bridge_survival(x0, x1, var, a, b, kmax=3):
    """P(a < x < b throughout the step | its endpoints), by the killed heat kernel.

    Discrete monitoring biases a barrier price high at the Broadie-Glasserman-Kou
    rate O(sqrt(dt)) -- measured on this problem at 1.0e-2 with 250 steps, halving
    only when the steps quadruple. This removes it."""
    d = b - a
    u, w = x0 - a, x1 - a
    base = (w - u) ** 2
    s = np.zeros_like(u)
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(-kmax, kmax + 1):
            s += np.exp(-((w - u - 2 * k * d) ** 2 - base) / (2.0 * var))
            s -= np.exp(-((w + u - 2 * k * d) ** 2 - base) / (2.0 * var))
    inside = (x0 > a) & (x0 < b) & (x1 > a) & (x1 < b)
    return np.where(inside, np.clip(s, 0.0, 1.0), 0.0)


def lsv_mc(charts, n, steps, seed, p=LSV, kind="eucall", antithetic=True,
           monitor="bridge"):
    """Per-path payoffs, shape ``(chart, spot, path)``.

    One variance-path set serves every spot and every chart, so a second model's
    price can be read as a paired difference from the first -- whose standard error
    is well below either price's own.

    ``monitor="discrete"`` replaces the Brownian bridge with the thing almost every
    first draft does instead -- test the barrier once per step -- and is here so
    that the step-8 calibration can measure the two against each other with one
    word changed and nothing else. It is never used by a derivation."""
    kappa, eta, sigma, rho, T = p["kappa"], p["eta"], p["sigma"], p["rho"], p["T"]
    rng = np.random.default_rng(seed)
    dt, m = T / steps, (2 * n if antithetic else n)
    nc, nspot = len(charts), len(p["spots"])
    v = np.full(m, float(p["v0"]))
    X = np.array([[np.full(m, to_x(s)) for s in p["spots"]] for to_x, _, _ in charts])
    alive = np.ones((nc, nspot, m)) if kind == "dntouch" else None
    bars = ([sorted((to_x(p["barriers"][0]), to_x(p["barriers"][1])))
             for to_x, _, _ in charts] if kind == "dntouch" else None)
    rt2 = 1.0 - rho ** 2
    for _ in range(steps):
        vn = qe_step(v, dt, kappa, eta, sigma, rng)
        iv = 0.5 * dt * (v + vn)
        # The exact identity sqrt(V) dW2 = (dV - kappa(eta-V)dtau)/sigma is what makes
        # the QE variance step and the asset step consistent with each other.
        iw = ((vn - v - kappa * eta * dt + kappa * iv) / sigma if sigma > 0
              else np.zeros_like(iv))
        z = rng.standard_normal(n)
        step = rho * iw + np.sqrt(np.maximum(rt2 * iv, 0.0)) * (
            np.concatenate([z, -z]) if antithetic else z)
        for r, (_, _, advance) in enumerate(charts):
            for q in range(nspot):
                xn = advance(X[r, q], iv, step)
                if kind == "dntouch":
                    lo, hi = bars[r]
                    alive[r, q] *= (bridge_survival(
                        X[r, q], xn, np.maximum(rt2 * iv, 1e-300), lo, hi)
                        if monitor == "bridge" else ((xn > lo) & (xn < hi)))
                X[r, q] = xn
        v = vn
    pay = np.empty((nc, nspot, n))
    for r, (_, from_x, _) in enumerate(charts):
        for q in range(nspot):
            y = (alive[r, q] if kind == "dntouch"
                 else np.maximum(from_x(X[r, q]) - p["K"], 0.0))
            pay[r, q] = 0.5 * (y[:n] + y[n:]) if antithetic else y
    return pay


def sabr_mc(which, n, steps, seed, antithetic=True, p=None):
    """SABR with beta = 1/2, in the Lamperti variable U = 2 sqrt(S):

        dU = (rho/alpha) dsigma + sqrt(1-rho^2) sigma dZ - (sigma^2 / 2U) dtau

    a Bessel process of dimension **0** in the clock int sigma^2 dtau, up to the
    known rho-term. Dimension 0 means the origin is attained and absorbing, which is
    the CEV boundary at S = 0 -- so the step is drift-implicit and absorbs exactly
    when the implicit quadratic loses its real root."""
    p = p or SABR[which]
    T, S0, sig0, alpha, rho = p["T"], p["S0"], p["sig0"], p["alpha"], p["rho"]
    rng = np.random.default_rng(seed)
    dt, m = T / steps, (2 * n if antithetic else n)
    rtdt, rt = np.sqrt(dt), np.sqrt(1.0 - rho ** 2)
    sig = np.full(m, sig0)
    U = np.full(m, 2.0 * np.sqrt(S0))
    for _ in range(steps):
        dw = rng.standard_normal(n) * rtdt
        dw = np.concatenate([dw, -dw]) if antithetic else dw
        sn = sig * np.exp(alpha * dw - 0.5 * alpha ** 2 * dt)
        ith = 0.5 * dt * (sig ** 2 + sn ** 2)
        z = rng.standard_normal(n)
        z = np.concatenate([z, -z]) if antithetic else z
        c = U + (rho / alpha) * (sn - sig) + rt * np.sqrt(ith) * z
        disc = c * c - 2.0 * ith
        U = np.where((U > 0.0) & (c > 0.0) & (disc >= 0.0),
                     0.5 * (c + np.sqrt(np.maximum(disc, 0.0))), 0.0)
        sig = sn
    return (U / 2.0) ** 2


HHW_CORR = {"I": (0.6, -0.7), "II": (0.0, 0.0)}      # (rho13, rho23)


def hhw_mc(which, n, steps, seed, p=HHW, antithetic=True, controls=True,
           variance="truncation"):
    """Heston-Hull-White, three correlated factors, full-truncation Euler.

    **Not** Andersen's QE step, although the Feller ratio of 0.04 is exactly where
    QE earns its keep, and it was tried. QE samples the variance without handing
    back the Brownian increment that produced it, and here a third factor is
    correlated with that same Brownian motion, so the increment has to be
    reconstructed from ``int sqrt(V) dW2 = (dV - kappa(eta-V) dtau) / sigma1``
    divided by ``sqrt(Vbar)``. Measured 2026-10-17, that reconstruction does not
    survive this problem:

    * With the variance on zero for a whole step -- 69% of steps here -- the
      numerator is the drift term alone and the denominator vanishes, so the
      quotient is of order 1000 and the asset overflows.
    * Guarding those steps with a fresh normal fixes the overflow and introduces a
      worse fault. The surviving reconstructions have sample mean -0.049 rather than
      zero, because keeping only the steps where the variance moved *is* a
      selection, and a mean-zero substitute for the rest does not cancel it. With
      rho23 = -0.7 that bias becomes a drift of order 1 in the short rate's Brownian
      motion over 500 steps: set I came out 4.9 high on a price of 78.58.
    * Set II cannot catch any of this. Its rho23 is 0, so the reconstruction is
      multiplied by zero and never used -- which is worth stating plainly, because
      "validated against the one set with a closed form" would have been false.

    Both control variates are exact martingale identities under Q: the discounted
    asset has expectation S0, and the discount factor itself has expectation P(0,T),
    which is Gaussian-closed for Hull-White."""
    T, K = p["T"], p["K"]
    kappa, eta, s1, s2, aa, bb = (p["kappa"], p["eta"], p["sigma1"], p["sigma2"],
                                  p["a"], p["b"])
    r13, r23 = HHW_CORR[which]
    L = np.linalg.cholesky(np.array([[1.0, p["rho12"], r13],
                                     [p["rho12"], 1.0, r23],
                                     [r13, r23, 1.0]]))
    rng = np.random.default_rng(seed)
    dt, m = T / steps, (2 * n if antithetic else n)
    rtdt = np.sqrt(dt)
    V, R = np.full(m, p["V0"]), np.full(m, p["R0"])
    X, M = np.zeros((len(p["spots"]), m)), np.zeros(m)
    for _ in range(steps):
        g = rng.standard_normal((3, n))
        dW = (L @ (np.concatenate([g, -g], axis=1) if antithetic else g)) * rtdt
        # Full truncation, or -- for the calibration's wrong half -- reflection,
        # which is the other named answer to "what does sqrt do at a negative
        # variance" and the one that is wrong here.
        vp = np.abs(V) if variance == "reflection" else np.maximum(V, 0.0)
        sq = np.sqrt(vp)
        Rn = R + aa * (bb - R) * dt + s2 * dW[2]
        Vn = V + kappa * (eta - vp) * dt + s1 * sq * dW[1]
        X += (0.5 * (R + Rn) - 0.5 * vp) * dt + sq * dW[0]
        M += 0.5 * (R + Rn) * dt
        V, R = (np.abs(Vn) if variance == "reflection" else Vn), Rn
    mu, var = hw_integrated_rate(p)
    p0T = np.exp(-mu + 0.5 * var)
    df = np.exp(-M)
    out = []
    for q, S0 in enumerate(p["spots"]):
        ST = S0 * np.exp(X[q])
        y, c1, c2 = df * np.maximum(ST - K, 0.0), df * ST, df
        if antithetic:
            y, c1, c2 = (0.5 * (a[:n] + a[n:]) for a in (y, c1, c2))
        # ``controls=False`` is how the calibration measures what a solver *without*
        # variance reduction reaches, which is the bar a problem's gate has to clear.
        out.append(control(y, [c1, c2], [S0, p0T]) if controls else y)
    return np.array(out)


# =============================================================================
# Route D -- conditioning: SABR set I is exact in the payoff
# =============================================================================


def besq0_call(x0, K, v):
    """``E[(X - K)^+]`` for ``X`` a squared Bessel process of dimension 0 at clock
    ``v/4`` from ``x0``. This is the beta = 1/2 CEV law: ``dS = sqrt(S) dB`` is
    BESQ(0) up to the factor 4, whose transition density is explicit,

        p(y) = (2t)^-1 sqrt(x0/y) exp(-(sqrt(y)-sqrt(x0))^2 / 2t) I_1(sqrt(x0 y)/t)

    (written in the exponentially scaled Bessel function, which is what keeps it
    finite at the clocks reached here). Dimension 0 is attained and absorbing, so
    the atom at zero is part of the law rather than a numerical accident."""
    t = v / 4.0
    if t <= 0:
        return max(x0 - K, 0.0)
    r = np.sqrt(x0)

    def f(w):                                  # substitute y = w^2
        return (w * w - K) * np.exp(-(w - r) ** 2 / (2.0 * t)) * ive(1, r * w / t)
    lo = np.sqrt(K)
    hi = max(r + 12.0 * np.sqrt(t) + 12.0 * t / max(r, 1e-12), lo + 1.0)
    edges = np.linspace(lo, hi, 9)
    tot = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            val, _ = quad(f, a, b, limit=200, epsabs=1e-15, epsrel=1e-13)
            tot += val
    return float(r / t * tot)


def sabr_clock_moments(which, p=None):
    """``E[v]`` and ``E[v^2]`` for the clock ``v = int_0^T sigma^2 dt``.

    Both are closed form because sigma is a driftless geometric Brownian motion:
    ``E[sigma_s^2 sigma_t^2] = sigma0^4 exp(alpha^2 (5s + t))`` for ``s < t``, and
    the double integral of that is elementary. They are what the conditional route
    uses as control variates -- the clock is the only thing it samples, so knowing
    two of its moments exactly removes most of the variance for free."""
    p = p or SABR[which]
    a2, T, s4 = p["alpha"] ** 2, p["T"], p["sig0"] ** 4
    mean = p["sig0"] ** 2 * (np.exp(a2 * T) - 1.0) / a2
    second = (2.0 * s4 / (5.0 * a2)) * ((np.exp(6.0 * a2 * T) - 1.0) / (6.0 * a2)
                                        - (np.exp(a2 * T) - 1.0) / a2)
    return float(mean), float(second)


def sabr1_conditional(n, steps, seed, which="I", p=None):
    """SABR with rho = 0, conditioned on the whole volatility path.

    With no correlation the asset's Brownian motion is independent of the
    volatility, so given the volatility path S is a CEV(1/2) process run on the
    clock ``v = int sigma^2 dt`` -- and :func:`besq0_call` prices it exactly. The
    only Monte Carlo left is over the scalar ``v``, which is why this reaches a
    standard error of 1e-4 on 4e5 paths where a two-factor sampler needs 1e7.

    Returns one array of ``v`` samples and the strike-wise prices."""
    p = p or SABR[which]
    if p["rho"] != 0.0:
        raise ValueError("the conditioning argument needs rho = 0")
    rng = np.random.default_rng(seed)
    T, sig0, alpha = p["T"], p["sig0"], p["alpha"]
    dt = T / steps
    w = np.zeros(n)
    s2 = np.full(n, sig0 ** 2)
    v = np.zeros(n)
    for k in range(steps):
        w += np.sqrt(dt) * rng.standard_normal(n)
        s2n = sig0 ** 2 * np.exp(2.0 * alpha * w - alpha ** 2 * (k + 1) * dt)
        v += 0.5 * dt * (s2 + s2n)
        s2 = s2n
    # The conditional price is a smooth function of v alone, so it is built once on
    # a grid and splined; evaluating the quadrature per path would cost 400000 of them.
    grid = np.unique(np.concatenate([
        np.linspace(v.min() * 0.95, np.quantile(v, 0.999), 300),
        np.geomspace(max(v.min() * 0.95, 1e-8), v.max() * 1.05, 120)]))
    mean, second = sabr_clock_moments(which, p)
    out = []
    for K in sabr_strikes(which, p):
        cs = CubicSpline(grid, [besq0_call(p["S0"], K, g) for g in grid])
        out.append(control(np.asarray(cs(v)), [v, v * v], [mean, second]))
    return v, np.array(out)


def sabr_call_payoffs(S, strikes, S0, n):
    """Per-strike payoffs from a path block of length ``2n`` (antithetic pairs),
    with the martingale identity ``E[S_T] = S0`` used as a control variate: a CEV
    process with beta < 1 is a true martingale, so whatever the absorbed scheme
    loses of that is measurable per path and can be taken back out."""
    out = []
    for K in strikes:
        y = np.maximum(S - K, 0.0)
        y, cv = 0.5 * (y[:n] + y[n:]), 0.5 * (S[:n] + S[n:])
        out.append(control(y, [cv], [S0]))
    return np.array(out)


def frozen_qlsv_call(S0, K, v, use_erf=False):
    """QLSV with the variance frozen at ``V = v0`` and rho = 0 -- the one case in
    this family with a closed form, and the arbiter of a disagreement the two
    routes could not otherwise settle.

    With V frozen, ``100/S`` is exactly a Bessel process of dimension 3 on the
    clock ``v = V T``, so ``S`` is a *strict local* martingale: ``E[S_T] =
    S0 erf(r0 / sqrt(2 theta))`` is strictly below ``S0``, and the European call is
    an integral against the explicit BES(3) density. ``use_erf`` returns that mean
    instead of the call."""
    theta, r0 = v, 100.0 / S0
    if use_erf:
        return float(S0 * erf(r0 / np.sqrt(2.0 * theta)))

    def pdf(r):
        return (r / (r0 * np.sqrt(2.0 * np.pi * theta))
                * (np.exp(-(r - r0) ** 2 / (2.0 * theta))
                   - np.exp(-(r + r0) ** 2 / (2.0 * theta))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        val, _ = quad(lambda r: (100.0 / r - K) * pdf(r), 1e-12, 100.0 / K,
                      limit=400, epsabs=1e-14, epsrel=1e-14)
    return float(val)


# =============================================================================
# Derivations: one entry per published problem function
# =============================================================================
#
# ``full`` is the setting the recorded ledger was built at. ``fast`` exists so the
# file can be exercised in seconds; its numbers are a smoke test, not evidence.

# =============================================================================
# The shifted twins -- section 5's memorization control
# =============================================================================
#
# BENCHOP's values have been on the internet since 2018, and 9.046650119220969 is
# exactly the kind of string a frontier model may carry. A twin at a parameter
# nobody has published converts that risk into a measurement: a solver that scores
# well on the published problem and badly on its twin was recalling, not solving.
#
# The twins are therefore *harness-owned* truth, and they follow make_references.py's
# rule rather than the published one -- two independent method families, and their
# disagreement stored as the reference's own error.
#
# What is shifted is the **initial** variance or volatility, and nothing else. That
# is the cleanest de-memorization available: the model, the payoff, the spot, the
# strikes, the maturity and -- for the Heston pair -- the Feller ratio 2*kappa*eta /
# sigma^2 = 0.2219 that makes the problem hard are all bit-identical to the
# published sibling, so a solver that genuinely solves one solves the other at the
# same cost and the same accuracy. Only the answer moves.
#
# How far it moves is the measurement that matters, and section 5's own suggestion
# (kappa 2.58 -> 2.43) does not survive it: measured 2026-10-20, that shift moves
# the Heston call by 2.4e-2, 2.2e-3 and 1.9e-3 relative at the three spots, which is
# *below* the gate a sampled price has to be given at two of them. A twin whose
# answer moves by less than the tolerance is not a control, it is a second copy of
# the same question. The shifts below were chosen by sweeping each parameter and
# keeping one that moves every value by at least ten times its gate: 10-21x for the
# double-no-touch, 24-50x for SABR, 15-32x for the call.

SHIFTED = {
    "hst_eucall_shift": dict(
        of="hst_eucall", title="Heston, European call K=100, V0 shifted",
        labels=("S75", "S100", "S125"), shift=dict(v0=[0.114, 0.164]),
        abs_err_rule="exact", params=dict(LSV, v0=0.164)),
    "hst_dntouch_shift": dict(
        of="hst_dntouch", title="Heston, double-no-touch L=50 U=150, V0 shifted",
        labels=("S75", "S100", "S125"), shift=dict(v0=[0.114, 0.164]),
        abs_err_rule="grid-self", params=dict(LSV, v0=0.164)),
    "sabr_set2_shift": dict(
        of="sabr_set2", title="SABR set II, European call, sigma0 shifted",
        labels=("K_lo", "K_mid", "K_hi"), shift=dict(sig0=[0.4, 0.34]),
        abs_err_rule="family-spread", params=dict(SABR["II"], sig0=0.34)),
}


SETTINGS = {
    "fast": dict(pde=(200, 80, 200), pde_c=(100, 40, 200), mc_n=50_000,
                 mc_steps=100, mc_fine=400, hhw_n=100_000, hhw_steps=250,
                 hhw_fine=1000, sabr_n=200_000, sabr_steps=200, sabr_fine=800,
                 sabr_pde=(200, 100, 100)),
    "full": dict(pde=(800, 320, 200), pde_c=(400, 160, 200), mc_n=400_000,
                 mc_steps=400, mc_fine=1600, hhw_n=1_000_000, hhw_steps=2000,
                 hhw_fine=8000, sabr_n=1_000_000, sabr_steps=1600, sabr_fine=6400,
                 sabr_pde=(800, 400, 200)),
}
SEED = 20261017

_MC_CACHE = {}


def _lsv_mc_cached(kind, n, steps):
    key = ("lsv", kind, n, steps)
    if key not in _MC_CACHE:
        _MC_CACHE[key] = lsv_mc([CHART_HESTON, CHART_QLSV], n, steps, SEED, kind=kind)
    return _MC_CACHE[key]


def _hhw_mc_cached(which, n, steps):
    key = ("hhw", which, n, steps)
    if key not in _MC_CACHE:
        _MC_CACHE[key] = hhw_mc(which, n, steps, SEED)
    return _MC_CACHE[key]


def _stats(pay):
    """(mean, standard error) per spot for a (spot, path) payoff block."""
    mu = pay.mean(axis=1)
    se = pay.std(axis=1, ddof=1) / np.sqrt(pay.shape[1])
    return np.asarray(mu), np.asarray(se)


def _richardson(fine, coarse, order=2.0, refine=2.0):
    """Extrapolate two levels. ``refine`` is the ratio between them -- 2 for the
    grids, which halve every spacing at once, and 4 for the step-count ladders."""
    return fine + (fine - coarse) / (refine ** order - 1.0)


def _route(name, family, values, note, stderr=None):
    return dict(name=name, family=family, values=[float(x) for x in values],
                stderr=None if stderr is None else [float(x) for x in stderr],
                note=note)


def derive(key, level="full"):
    """Every independent computation this repo has for one published problem."""
    c = SETTINGS[level]
    p = LSV
    routes = []
    if key == "hst_eucall":
        routes.append(_route(
            "heston-cf", "quadrature",
            [heston_call(s, p["K"], p["T"], p["kappa"], p["eta"], p["sigma"],
                         p["rho"], p["v0"]) for s in p["spots"]],
            "Lewis single integral of the little-trap characteristic function"))
        fine, coarse = lsv_pde("eucall", "hst", *c["pde"]), lsv_pde("eucall", "hst", *c["pde_c"])
        routes.append(_route("fd2d", "grid", fine, f"Crank-Nicolson {c['pde']}"))
        routes.append(_route("fd2d-richardson", "grid", _richardson(fine, coarse),
                             "two grid levels, extrapolated at order 2"))
        primary = "heston-cf"

    elif key == "hhw_eucall2":
        routes.append(_route("hhw-cf", "quadrature",
                             [hhw2_call(s, HHW) for s in HHW["spots"]],
                             "the Hull-White factor folded into Lewis' integrand"))
        mu, se = _stats(_hhw_mc_cached("II", c["hhw_n"], c["hhw_steps"]))
        routes.append(_route("mc", "sampler", mu,
                             f"full-truncation Euler, {c['hhw_n']} paths x "
                             f"{c['hhw_steps']} steps, two control variates", se))
        primary = "hhw-cf"

    elif key in ("hst_dntouch", "qlsv_dntouch"):
        model = key.split("_")[0]
        fine = lsv_pde("dntouch", model, *c["pde"])
        coarse = lsv_pde("dntouch", model, *c["pde_c"])
        routes.append(_route("fd2d", "grid", fine, f"Crank-Nicolson {c['pde']}"))
        routes.append(_route("fd2d-richardson", "grid", _richardson(fine, coarse),
                             "two grid levels, extrapolated at order 2"))
        pay = _lsv_mc_cached("dntouch", c["mc_n"], c["mc_steps"])
        mu, se = _stats(pay[0 if model == "hst" else 1])
        routes.append(_route("mc-bridge", "sampler", mu,
                             f"QE variance, Brownian-bridge barrier, {c['mc_n']} "
                             f"paths x {c['mc_steps']} steps", se))
        primary = "fd2d-richardson"

    elif key == "qlsv_eucall":
        fine = lsv_pde("eucall", "qlsv", *c["pde"])
        coarse = lsv_pde("eucall", "qlsv", *c["pde_c"])
        routes.append(_route("fd2d", "grid", fine, f"Crank-Nicolson {c['pde']}"))
        routes.append(_route("fd2d-richardson", "grid", _richardson(fine, coarse),
                             "two grid levels, extrapolated at order 2"))
        mu, se = _stats(_lsv_mc_cached("eucall", c["mc_n"], c["mc_steps"])[1])
        routes.append(_route(
            "mc", "sampler", mu,
            f"Bessel(3) chart, drift-implicit, {c['mc_n']} paths x {c['mc_steps']} "
            "steps -- DOES NOT AGREE; see the frozen-variance arbiter", se))
        primary = "fd2d-richardson"

    elif key == "sabr_set1":
        _v, cond = sabr1_conditional(c["sabr_n"], c["sabr_steps"], SEED)
        mu, se = _stats(cond)
        routes.append(_route("conditional-besq", "quadrature+sampler", mu,
                             f"exact CEV payoff given the volatility path, "
                             f"{c['sabr_n']} clock samples", se))
        S = sabr_mc("I", c["sabr_n"], c["sabr_steps"], SEED)
        mu2, se2 = _stats(sabr_call_payoffs(S, sabr_strikes("I"), SABR["I"]["S0"],
                                            c["sabr_n"]))
        routes.append(_route("mc-2f", "sampler", mu2,
                             f"two-factor absorbed CEV, {c['sabr_n']} paths x "
                             f"{c['sabr_steps']} steps, martingale control", se2))
        routes.append(_route("fd2d", "grid", sabr_pde("I", *c["sabr_pde"]),
                             f"Crank-Nicolson in (S, ln sigma), {c['sabr_pde']}"))
        primary = "conditional-besq"

    elif key == "sabr_set2":
        ks = sabr_strikes("II")
        out = {}
        for tag, steps in (("coarse", c["sabr_steps"]), ("fine", c["sabr_fine"])):
            S = sabr_mc("II", c["sabr_n"], steps, SEED)
            out[tag] = _stats(sabr_call_payoffs(S, ks, SABR["II"]["S0"], c["sabr_n"]))
        routes.append(_route("mc-2f", "sampler", out["fine"][0],
                             f"two-factor absorbed CEV, {c['sabr_n']} paths x "
                             f"{c['sabr_fine']} steps, martingale control",
                             out["fine"][1]))
        routes.append(_route(
            "mc-2f-coarse", "sampler", out["coarse"][0],
            f"the same at {c['sabr_steps']} steps, recorded so that the step effect is "
            "visible rather than assumed -- quartering the step divides the gap by "
            "about sixteen, which is second order and not the first order an Euler "
            "scheme would give",
            out["coarse"][1]))
        routes.append(_route(
            "mc-2f-richardson", "sampler",
            _richardson(out["fine"][0], out["coarse"][0], order=2.0, refine=4.0),
            "extrapolated at the order those two levels imply. Recorded, but not the "
            "primary: the order was inferred from the same two points it is applied "
            "to, so the confirmation this problem claims rests on a single measured "
            "level instead",
            out["fine"][1]))
        routes.append(_route("fd2d", "grid", sabr_pde("II", *c["sabr_pde"]),
                             f"Crank-Nicolson in (S, ln sigma), {c['sabr_pde']} -- the "
                             "independent second family, which the sampler alone would "
                             "not have been"))
        primary = "mc-2f"

    elif key == "hhw_eucall1":
        # The hardest problem in the suite, and the only one with no second family:
        # three correlated factors leave no quadrature and no two-dimensional grid.
        coarse = _stats(_hhw_mc_cached("I", c["hhw_n"], c["hhw_steps"]))
        fine = _stats(hhw_mc("I", c["hhw_n"], c["hhw_fine"], SEED))
        routes.append(_route("mc", "sampler", fine[0],
                             f"full-truncation Euler, {c['hhw_n']} paths x "
                             f"{c['hhw_fine']} steps, two control variates", fine[1]))
        routes.append(_route(
            "mc-coarse", "sampler", coarse[0],
            f"the same at {c['hhw_steps']} steps -- quadrupling the step count moves "
            "the answer by about its own standard error, so what is left is not the "
            "step", coarse[1]))
        # Set II is the same scheme on the same two factors with the rate decorrelated,
        # and it has a closed form -- so its offset is a direct measurement of this
        # sampler's bias at these settings. Subtracting it assumes the two
        # correlations do not change that bias, which is an assumption and is why all
        # three routes are kept.
        bias = _stats(_hhw_mc_cached("II", c["hhw_n"], c["hhw_steps"]))[0] - np.array(
            [hhw2_call(s0, HHW) for s0 in HHW["spots"]])
        routes.append(_route("mc-bias-controlled", "sampler", coarse[0] - bias,
                             "set II, run at the same settings, has a closed form; its "
                             "measured offset is subtracted here", coarse[1]))
        primary = "mc-bias-controlled"
    else:
        raise KeyError(key)
    return routes, primary


def derive_shifted(key, level="full"):
    """The twins' routes. Same machinery as the published problems, pointed at the
    shifted parameters -- which is the point: if the derivation is the same, a
    difference in a solver's score between a problem and its twin is about the
    solver."""
    c = SETTINGS[level]
    spec = SHIFTED[key]
    pr = spec["params"]
    routes = []
    if key == "hst_eucall_shift":
        routes.append(_route(
            "heston-cf", "quadrature",
            [heston_call(s0, pr["K"], pr["T"], pr["kappa"], pr["eta"], pr["sigma"],
                         pr["rho"], pr["v0"]) for s0 in pr["spots"]],
            "Lewis single integral, exact up to quadrature -- the same route that "
            "reproduced the published sibling to 5.3e-14"))
        routes.append(_route("fd2d", "grid", lsv_pde("eucall", "hst", *c["pde"], p=pr),
                             f"Crank-Nicolson {c['pde']}, as corroboration"))
        primary = "heston-cf"
    elif key == "hst_dntouch_shift":
        fine = lsv_pde("dntouch", "hst", *c["pde"], p=pr)
        coarse = lsv_pde("dntouch", "hst", *c["pde_c"], p=pr)
        rich = _richardson(fine, coarse)
        routes.append(_route("fd2d-richardson", "grid", rich,
                             "two grid levels, extrapolated at order 2"))
        routes.append(_route("fd2d", "grid", fine, f"Crank-Nicolson {c['pde']}"))
        mu, se = _stats(lsv_mc([CHART_HESTON], c["mc_n"], c["mc_steps"], SEED,
                               p=pr, kind="dntouch")[0])
        routes.append(_route("mc-bridge", "sampler", mu,
                             f"QE variance, Brownian-bridge barrier, {c['mc_n']} "
                             f"paths x {c['mc_steps']} steps", se))
        primary = "fd2d-richardson"
    elif key == "sabr_set2_shift":
        S = sabr_mc("II", c["sabr_n"], c["sabr_fine"], SEED, p=pr)
        mu, se = _stats(sabr_call_payoffs(S, sabr_strikes("II", pr), pr["S0"],
                                          c["sabr_n"]))
        routes.append(_route("mc-2f", "sampler", mu,
                             f"two-factor absorbed CEV, {c['sabr_n']} paths x "
                             f"{c['sabr_fine']} steps, martingale control", se))
        routes.append(_route("fd2d", "grid", sabr_pde("II", *c["sabr_pde"], p=pr),
                             f"Crank-Nicolson in (S, ln sigma), {c['sabr_pde']}"))
        primary = "mc-2f"
    else:
        raise KeyError(key)
    return routes, primary


def twin_abs_err(spec, routes, primary):
    """The twin's own absolute uncertainty, by the rule its spec names.

    ``exact``          the primary route is a quadrature of a closed form; the other
                       routes corroborate it and do not widen it.
    ``grid-self``      the gap between the finest grid and its extrapolation, which
                       is the standard error estimate for an extrapolated grid.
    ``family-spread``  the disagreement between two independent method families,
                       which is make_references.py's rule.

    Each rule is *validated on the published sibling*, where a third party
    adjudicates: :func:`twin_rule_validation` reads the stored ledger and reports
    what the rule would have predicted there against what the error turned out to
    be."""
    rule = spec["abs_err_rule"]
    if rule == "exact":
        return 0.0
    hit = next(r for r in routes if r["name"] == primary)
    if rule == "grid-self":
        other = next(r for r in routes if r["name"] == "fd2d")
    else:
        other = next(r for r in routes if r["family"] != hit["family"])
    return float(max(abs(a - b) for a, b in
                     zip(hit["values"], other["values"], strict=True)))


def twin_rule_validation(spec, stored):
    """What the twin's error rule would have claimed on the published sibling, and
    what the error there actually was. Costs nothing: both are already in the
    ledger."""
    sib = (stored or {}).get(spec["of"])
    if not sib:
        return None
    by = {r["name"]: r for r in sib["routes"]}
    rule = spec["abs_err_rule"]
    if rule == "exact":
        predicted = 0.0
    elif rule == "grid-self" and {"fd2d", "fd2d-richardson"} <= set(by):
        predicted = max(abs(a - b) for a, b in
                        zip(by["fd2d"]["values"], by["fd2d-richardson"]["values"],
                            strict=True))
    else:
        hit = by[sib["primary_route"]]
        other = next((r for r in sib["routes"] if r["family"] != hit["family"]), None)
        if other is None:
            return None
        predicted = max(abs(a - b) for a, b in
                        zip(hit["values"], other["values"], strict=True))
    return {"sibling": spec["of"], "rule": rule, "predicted": float(predicted),
            "actual": float(sib["confirmed_abs_err"])}


def build_shifted(key, level="full"):
    spec = SHIFTED[key]
    t0 = time.time()
    routes, primary = derive_shifted(key, level)
    hit = next(r for r in routes if r["name"] == primary)
    rec = {"title": spec["title"], "labels": list(spec["labels"]),
           "of": spec["of"], "shift": spec["shift"],
           "value": [float(x) for x in hit["values"]],
           "abs_err_rule": spec["abs_err_rule"],
           "reference_abs_err": twin_abs_err(spec, routes, primary),
           "primary_route": primary, "routes": routes,
           "wall_s": round(time.time() - t0, 1)}
    print(f"\n{key}  ({rec['wall_s']}s)  -- {spec['title']}")
    for r in routes:
        mark = "*" if r["name"] == primary else " "
        print(f" {mark}{r['name']:<20s} " + "  ".join(f"{v:.9g}" for v in r["values"]))
    print(f"   harness-owned truth; its own error {rec['reference_abs_err']:.2e} "
          f"by the {spec['abs_err_rule']} rule")
    return rec


def qlsv_convention_check(level="full"):
    """Why the QLSV European call has two different right answers.

    Freeze the variance (sigma = kappa = 0, rho = 0) and QLSV acquires a closed
    form, which the two routes can be held against. All three legs are computed
    here rather than quoted. Measured at S0 = 125:

        closed form (BES(3) density)   29.98063
        Monte Carlo, 1600 steps        29.86666 +- 0.0882   (1.3 standard errors)
        finite differences, u_SS = 0   32.20765            (25 standard errors)

    The grid is not converging slowly, it is converging to something else. With
    f(s) = 0.01 s^2 the asset is a *strict local* martingale, so the pricing PDE has
    more than one solution and the usual far-field condition -- value linear in S --
    selects the one that is not the expectation. The published BENCHOP values follow
    the grid: on the live problem the finite-difference route matches them to 7e-5
    while an honest sampler of the stated SDE sits 0.24 below at S0 = 125.

    That is a statement about the answer key, not about either route, and it is the
    reason this problem cannot be posed to both halves of the pipeline at once."""
    c = SETTINGS[level]
    v0, K = LSV["v0"], LSV["K"]
    frozen = dict(LSV, kappa=0.0, sigma=0.0, rho=0.0)
    out = {"frozen_clock": v0 * LSV["T"], "spots": list(LSV["spots"]),
           "closed_form": [frozen_qlsv_call(s, K, v0 * LSV["T"]) for s in LSV["spots"]],
           "local_mart_mean": [frozen_qlsv_call(s, K, v0 * LSV["T"], use_erf=True)
                               for s in LSV["spots"]]}
    S = sinh_grid(0.0, 1600.0, c["pde"][0], K)
    u, Sg, Vg = lsv_solve(lambda x: np.maximum(x - K, 0.0), f_qlsv, S,
                          v_grid(0.5, 60), 200, frozen, s_right="linear",
                          boundary=np.zeros(len(S)))
    out["fd2d_linear_far_field"] = [value_at(u, Sg, Vg, s, v0) for s in LSV["spots"]]
    pay = lsv_mc([CHART_QLSV], c["mc_n"] // 2, c["mc_fine"], SEED, p=frozen)[0]
    mu, se = _stats(pay)
    out["mc"] = [float(x) for x in mu]
    out["mc_stderr"] = [float(x) for x in se]
    return out


# =============================================================================
# Calibration -- the other half of the repo's rule
# =============================================================================
#
# A gate is only evidence if it does two things: let a correct scheme through at
# the resolution the problem ships with, and stop a plausible wrong one. Step 7
# measured the first half. This section measures the second, and it measures it as
# a *pair*: the two schemes differ by one decision, so whatever separates them is
# that decision and not the sampler, the seed, the path count or the step. SABR
# set II is the one place where the correct scheme and the wrong one are two
# decisions apart, and the scheme that sits between them is recorded with them.
#
# The wrong halves are not strawmen. Each is a named, defensible answer to a
# question the problem statement raises and then declines to answer:
#
#   * reflecting a CEV forward off zero instead of absorbing there -- the reflex
#     fix for "my prices went negative", and a different process;
#   * reflecting the variance instead of truncating it -- one of the two standard
#     answers to sqrt of a negative variance, and the wrong one when Feller fails;
#   * testing a barrier once per step instead of over the step -- what almost every
#     first draft does, biased high at the Broadie-Glasserman-Kou rate O(sqrt(dt)).
#
# None of them is reachable from the problem statement as *the* answer; each is
# reachable as *an* answer. That is what makes the measurement worth making.


def sabr_euler(which, n, steps, seed, boundary="absorb", antithetic=True, p=None):
    """Euler-Maruyama straight in S, which is what a solver writes before it
    reaches for a Lamperti transform: ``S += sigma sqrt(S) dW``, volatility
    exactly. ``boundary`` is the whole experiment -- "absorb" clamps a negative
    excursion to the absorbing state the CEV boundary actually is, "reflect" sends
    it back up, which is a different process with the same one-line diff."""
    p = p or SABR[which]
    T, S0, sig0, alpha, rho = p["T"], p["S0"], p["sig0"], p["alpha"], p["rho"]
    rng = np.random.default_rng(seed)
    dt, m = T / steps, (2 * n if antithetic else n)
    rtdt, rt = np.sqrt(dt), np.sqrt(1.0 - rho ** 2)
    sig, S = np.full(m, sig0), np.full(m, float(S0))
    for _ in range(steps):
        g = rng.standard_normal(n)
        g = np.concatenate([g, -g]) if antithetic else g
        h = rng.standard_normal(n)
        h = np.concatenate([h, -h]) if antithetic else h
        S = S + sig * np.sqrt(np.maximum(S, 0.0)) * (rho * g + rt * h) * rtdt
        S = np.maximum(S, 0.0) if boundary == "absorb" else np.abs(S)
        sig = sig * np.exp(alpha * g * rtdt - 0.5 * alpha ** 2 * dt)
    pay = np.array([np.maximum(S - K, 0.0) for K in sabr_strikes(which, p)])
    return 0.5 * (pay[:, :n] + pay[:, n:]) if antithetic else pay


def lsv_euler(n, steps, seed, p=LSV, model="hst", variance="truncation",
              antithetic=True):
    """Full-truncation Euler on both factors, no chart and no QE step -- the plain
    scheme, and the one an end-to-end sandbox run of this problem actually used.
    ``variance="reflection"`` changes one line: what the variance does when the
    step takes it below zero."""
    kappa, eta, sigma, rho, T, K = (p["kappa"], p["eta"], p["sigma"], p["rho"],
                                    p["T"], p["K"])
    f = (lambda s: s) if model == "hst" else (lambda s: 0.01 * s * s)
    rng = np.random.default_rng(seed)
    dt, m = T / steps, (2 * n if antithetic else n)
    rtdt, rt2 = np.sqrt(dt), np.sqrt(1.0 - rho ** 2)
    V = np.full(m, float(p["v0"]))
    S = np.array([np.full(m, float(s)) for s in p["spots"]])
    for _ in range(steps):
        g = rng.standard_normal(n)
        g = np.concatenate([g, -g]) if antithetic else g
        h = rng.standard_normal(n)
        h = np.concatenate([h, -h]) if antithetic else h
        vp = np.abs(V) if variance == "reflection" else np.maximum(V, 0.0)
        sq = np.sqrt(vp)
        S = np.maximum(S + sq * f(S) * (rho * g + rt2 * h) * rtdt, 0.0)
        V = V + kappa * (eta - vp) * dt + sigma * sq * g * rtdt
        if variance == "reflection":
            V = np.abs(V)
    pay = np.maximum(S - K, 0.0)
    return 0.5 * (pay[:, :n] + pay[:, n:]) if antithetic else pay


# key -> the decision the problem turns on, the scheme that decides it rightly,
# the one that decides it wrongly, and any further scheme worth recording beside
# them. The twins inherit their sibling's row, since a shifted initial variance
# changes the answer and not the scheme.
#
# The two SABR rows are not symmetric, and the asymmetry is the measurement. On
# set I the plain Euler in S is a correct scheme and the gate's job is to stop the
# reflecting one. On set II -- ten years at vol-of-vol 0.8, with the step capped
# at 0.004 by the verifier -- that same plain Euler carries a discretisation bias
# of about 1e-2, which is the gate, so what is correct there is the Lamperti
# scheme the problem is built around. Both are kept on both sets so the pair that
# differs by one decision is always on record.
CALIBRATION = {
    "sabr_set1": dict(decision="the CEV boundary at S = 0",
                      correct="euler-absorbing", wrong="euler-reflecting",
                      also=("lamperti-absorbing",)),
    "sabr_set2": dict(decision="the CEV boundary at S = 0",
                      correct="lamperti-absorbing", wrong="euler-reflecting",
                      also=("euler-absorbing",)),
    "hst_eucall": dict(decision="the variance below zero",
                       correct="euler-truncation", wrong="euler-reflection"),
    "hst_dntouch": dict(decision="barrier monitoring",
                        correct="qe-bridge", wrong="qe-discrete"),
    "qlsv_dntouch": dict(decision="barrier monitoring",
                         correct="qe-bridge", wrong="qe-discrete"),
    "hhw_eucall1": dict(decision="the variance below zero",
                        correct="euler-truncation", wrong="euler-reflection"),
    "hhw_eucall2": dict(decision="the variance below zero",
                        correct="euler-truncation", wrong="euler-reflection"),
}


def calibrate_run(key, scheme, n, steps, p=None):
    """Payoffs, shape ``(value, path)``, for one scheme on one problem."""
    if key.startswith("sabr"):
        which = "I" if key.endswith("1") else "II"
        if scheme == "lamperti-absorbing":
            S = sabr_mc(which, n, steps, SEED, p=p)
            return sabr_call_payoffs(S, sabr_strikes(which, p), (p or SABR[which])["S0"], n)
        return sabr_euler(which, n, steps, SEED,
                          boundary="absorb" if scheme.endswith("absorbing") else "reflect",
                          p=p)
    if key.startswith("hhw"):
        which = "I" if key.endswith("1") else "II"
        # controls=False: an agent's solver is unlikely to write two control
        # variates, so the bar the gate has to clear is the one without them.
        return hhw_mc(which, n, steps, SEED, controls=False,
                      variance=scheme.split("-")[1])
    model, kind = key.split("_")[0], key.split("_")[1]
    if kind == "dntouch":
        chart = CHART_HESTON if model == "hst" else CHART_QLSV
        return lsv_mc([chart], n, steps, SEED, p=p or LSV, kind="dntouch",
                      monitor=scheme.split("-")[1])[0]
    return lsv_euler(n, steps, SEED, p=p or LSV, model=model,
                     variance=scheme.split("-")[1])


def ship_settings():
    """What each entry actually ships with, read from ``problems.py`` rather than
    restated here: the path count, the step count the verifier's ``0.04/T`` cap
    leaves it at, and the per-value gate. A calibration measured at any other
    resolution would be measuring a different problem."""
    import sys
    sys.path.insert(0, HERE)
    import problems as P
    import verify as V
    out = {}
    for prob in P.benchop_problems():
        dt = V.sde_verify_dt(prob)
        tols = V._functional_tols(prob)
        out[prob["benchop_key"]] = dict(
            slug=prob["slug"], n=int(prob["num_paths"]),
            steps=int(round(prob["T"] / dt)),
            labels=list(prob["functional_truth"]),
            truth=[float(x) for x in prob["functional_truth"].values()],
            gate=[float(tols[k]) for k in prob["functional_truth"]])
    return out


def calibrate(keys=None, settings=None):
    """Run every scheme for every named problem at its shipped resolution."""
    ship = settings or ship_settings()
    shift_params = {k: v["params"] for k, v in SHIFTED.items()}
    out = {}
    for key, cfg in ship.items():
        base = SHIFTED[key]["of"] if key in SHIFTED else key
        spec = CALIBRATION[base]
        right, wrong = spec["correct"], spec["wrong"]
        ref, gate = np.asarray(cfg["truth"]), np.asarray(cfg["gate"])
        rec = dict(slug=cfg["slug"], decision=spec["decision"], labels=cfg["labels"],
                   paths=cfg["n"], steps=cfg["steps"], gate=[float(g) for g in gate],
                   schemes=[])
        for scheme in [right, wrong, *spec.get("also", ())]:
            t0 = time.time()
            pay = calibrate_run(base, scheme, cfg["n"], cfg["steps"],
                                p=shift_params.get(key))
            mu, se = _stats(pay)
            rel = np.abs(mu - ref) / np.abs(ref)
            rec["schemes"].append(dict(
                name=scheme, role=("correct" if scheme == right else
                                   "wrong" if scheme == wrong else "recorded"),
                values=[float(x) for x in mu],
                rel_err=[float(x) for x in rel],
                rel_stderr=[float(x) for x in se / np.abs(ref)],
                gate_ratio=[float(x) for x in rel / gate],
                wall_s=round(time.time() - t0, 1)))
            r = rec["schemes"][-1]
            print(f"  {cfg['slug']:<28s} {scheme:<20s} {r['wall_s']:6.1f}s  "
                  "relerr " + " ".join(f"{x:.2e}" for x in r["rel_err"])
                  + "  x gate " + " ".join(f"{x:6.2f}" for x in r["gate_ratio"]),
                  flush=True)
        c = next(x for x in rec["schemes"] if x["role"] == "correct")
        w = next(x for x in rec["schemes"] if x["role"] == "wrong")
        # The harness fails a problem when *any* one of the reported values misses
        # its gate, so that is the bar the wrong scheme has to clear. How many of
        # the three it misses is recorded beside it, because missing on one of
        # three is a weaker result than missing on all three and the difference
        # should not have to be reconstructed from the numbers later.
        rec["separates"] = bool(max(c["gate_ratio"]) < 1.0 <= max(w["gate_ratio"]))
        rec["wrong_values_missed"] = int(sum(1 for x in w["gate_ratio"] if x >= 1.0))
        out[key] = rec
        print(f"  {'':<28s} {'-> separates' if rec['separates'] else '-> DOES NOT SEPARATE'}"
              f": correct {max(c['gate_ratio']):.2f}x gate at worst, wrong "
              f"{max(w['gate_ratio']):.2f}x on {rec['wrong_values_missed']} of "
              f"{len(w['gate_ratio'])} values", flush=True)
    return out


# =============================================================================
# The ledger
# =============================================================================


def build(key, level="full", write=True):
    pub = PUBLISHED[key]
    ref = np.asarray(pub["values"], dtype=float)
    t0 = time.time()
    routes, primary = derive(key, level)
    rec = {
        "title": pub["title"],
        "labels": list(pub["labels"]),
        "published": [float(x) for x in pub["values"]],
        # The answer key's own uncertainty -- not ours. Only HHWeuCallI has one.
        "reference_abs_err": float(pub["abs_err"]),
        "primary_route": primary,
        "routes": [],
        "wall_s": None,
    }
    for r in routes:
        d = np.abs(np.asarray(r["values"]) - ref)
        rec["routes"].append(dict(r, abs_diff=[float(x) for x in d],
                                  rel_diff=[float(x) for x in d / np.abs(ref)]))
    hit = next(r for r in rec["routes"] if r["name"] == primary)
    rec["confirmed_abs_err"] = float(max(hit["abs_diff"]))
    rec["confirmed_rel_err"] = float(max(hit["rel_diff"]))
    rec["wall_s"] = round(time.time() - t0, 1)
    if key == "qlsv_eucall":
        rec["convention"] = qlsv_convention_check(level)

    print(f"\n{key}  ({rec['wall_s']}s)  -- {pub['title']}")
    for r in rec["routes"]:
        mark = "*" if r["name"] == primary else " "
        print(f" {mark}{r['name']:<20s} " + "  ".join(f"{v:.9g}" for v in r["values"]))
        print(f"   {'':<20s} " + "  ".join(f"|d|={v:.2e}" for v in r["abs_diff"]))
    print(f"   confirmed to {rec['confirmed_abs_err']:.2e} absolute "
          f"({rec['confirmed_rel_err']:.2e} relative); the published value carries "
          f"{rec['reference_abs_err']:.0e}")
    return rec


def _write(records, level, shifted=None, calibration=None):
    os.makedirs(REF_DIR, exist_ok=True)
    doc = {
        "generated": time.strftime("%Y-%m-%d"),
        "level": level,
        "source": {
            "formulations": "https://it.uu.se/research/scientific_computing/project/"
                            "compfin/benchop/SLV/benchop_slv_0622.pdf",
            "reference_values": "https://it.uu.se/research/scientific_computing/"
                                "project/compfin/benchop/SLV/unitTestBenchopSLV.m",
            "paper": "https://doi.org/10.1080/00207160.2018.1544368",
        },
        "note": "published = the external answer key, transcribed. reference_abs_err "
                "= the uncertainty that key itself carries. confirmed_abs_err = how "
                "closely this repo's own primary route reproduced it. The "
                "`shifted` entries are not published at all: they are section 5's "
                "memorization control, computed here, and their reference_abs_err "
                "is this repo's own uncertainty rather than anyone else's. "
                "`calibration` is step 8: per problem, at the resolution its entry "
                "ships with, what a correct scheme and a plausible wrong one each "
                "reach against the gate.",
        "problems": records,
        "shifted": shifted or {},
        "calibration": calibration or {},
    }
    with open(LEDGER, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"\nwrote {os.path.relpath(LEDGER, os.path.dirname(HERE))}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated problem keys")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="coarse settings: a smoke test, never evidence")
    ap.add_argument("--check", action="store_true",
                    help="recompute and compare against the stored ledger; write nothing")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure a correct and a plausible wrong scheme for every "
                         "entry, at the resolution that entry ships with, and store "
                         "the result under `calibration`")
    args = ap.parse_args(argv)
    level = "fast" if args.fast else "full"

    doc = {}
    if os.path.exists(LEDGER):
        with open(LEDGER) as fh:
            doc = json.load(fh)

    if args.list:
        stored = doc.get("problems", {})
        for k, v in PUBLISHED.items():
            st = stored.get(k)
            tag = (f"confirmed {st['confirmed_abs_err']:.1e}" if st else "MISSING")
            print(f"{k:<18} {len(v['values'])} values  published  {tag:<22} {v['title']}")
        twins = doc.get("shifted", {})
        for k, v in SHIFTED.items():
            st = twins.get(k)
            tag = (f"own err {st['reference_abs_err']:.1e}" if st else "MISSING")
            print(f"{k:<18} {len(v['labels'])} values  harness    {tag:<22} {v['title']}")
        return 0

    if args.calibrate:
        ship = ship_settings()
        if args.only:
            want = {t.strip() for t in args.only.split(",") if t.strip()}
            missing = want - set(ship)
            if missing:
                ap.error(f"not an entry in the suite: {', '.join(sorted(missing))}")
            ship = {k: v for k, v in ship.items() if k in want}
        print("=== calibration: correct vs plausible-wrong, at shipped resolution ===")
        cal = calibrate(settings=ship)
        if args.check:
            bad = [k for k, v in cal.items() if not v["separates"]]
            print("\nCHECK: " + (f"{len(bad)} entries do not separate: "
                                 + ", ".join(bad) if bad
                                 else f"all {len(cal)} entries separate"))
            return 1 if bad else 0
        merged = dict(doc.get("calibration", {}))
        merged.update(cal)
        _write(doc.get("problems", {}), doc.get("level", level),
               doc.get("shifted", {}), merged)
        return 0

    known = set(PUBLISHED) | set(SHIFTED)
    if args.only:
        keys = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [k for k in keys if k not in known]
        if unknown:
            ap.error(f"no derivation for: {', '.join(unknown)}")
    elif args.all:
        keys = list(PUBLISHED) + list(SHIFTED)
    else:
        ap.error("pass --only <keys>, --all, or --list")

    records = {k: build(k, level) for k in keys if k in PUBLISHED}
    twins = {}
    for k in (k for k in keys if k in SHIFTED):
        rec = build_shifted(k, level)
        rec["rule_validated_on_sibling"] = twin_rule_validation(
            SHIFTED[k], {**doc.get("problems", {}), **records})
        twins[k] = rec
    if args.check:
        if not os.path.exists(LEDGER):
            print("\nCHECK: no stored ledger to compare against")
            return 1
        stored = doc["problems"]
        ok = True
        for k, rec in records.items():
            was = stored.get(k)
            if was is None:
                print(f"CHECK {k}: not in the stored ledger")
                ok = False
                continue
            drift = max(abs(a - b) for a, b in
                        zip(rec["routes"][0]["values"], was["routes"][0]["values"],
                            strict=True))
            bar = max(10.0 * was["confirmed_abs_err"], 1e-12)
            print(f"CHECK {k}: primary-family drift {drift:.2e} vs stored "
                  f"{was['confirmed_abs_err']:.2e} -> {'ok' if drift <= bar else 'MISMATCH'}")
            ok = ok and drift <= bar
        return 0 if ok else 1
    if level == "fast":
        print("\n--fast: smoke numbers, ledger not written")
        return 0
    merged, merged_twins = dict(doc.get("problems", {})), dict(doc.get("shifted", {}))
    merged.update(records)
    merged_twins.update(twins)
    _write(merged, level, merged_twins, doc.get("calibration", {}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
