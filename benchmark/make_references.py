"""Compute the harness-owned reference fields for the no-closed-form problems.

A PDE without a closed form is still independently checkable, but only if somebody
computes the answer to an accuracy nobody in the benchmark can reach by accident.
That is this file: for each such problem, **two independent method families** are run
at high resolution, their disagreement is measured, and the result is stored under
``benchmark/references/{slug}.npz`` together with that number.

The measured disagreement is the reference's own error, and it is not decoration --
``verify.py`` widens the pass tolerance to ``REFERENCE_TOL_FACTOR`` times it, and
refuses to score against a reference that does not carry one. A reference whose
accuracy nobody established is an answer key, not ground truth.

Run offline, never during a benchmark run:

    uv run python benchmark/make_references.py --list
    uv run python benchmark/make_references.py --only pde_kuramoto_sivashinsky
    uv run python benchmark/make_references.py --all
    uv run python benchmark/make_references.py --all --check   # recompute, do not write

``--check`` recomputes and compares against what is stored, which is how a reference
is revalidated after any change to this file.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "references")
sys.path.insert(0, HERE)


# =============================================================================
# Shared spectral machinery
# =============================================================================


def _periodic_grid(L, M):
    """Endpoint-exclusive uniform grid -- the only convention on which the discrete
    Fourier transform is the interpolant of the periodic function."""
    return np.arange(M) * (L / M)


def rel_l2(a, b):
    """Relative L2 (RMS) difference, the metric verify.py scores in."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-300))


def _etdrk4_coefficients(Lop, h, n_contour=64, chunk=1 << 16):
    """Kassam-Trefethen contour-integral evaluation of the phi functions.

    Evaluating ``(e^{hL}-1)/hL`` directly loses all significant digits where ``hL`` is
    small, which on a stiff problem is exactly the modes that carry the solution. The
    contour average is the standard cure and is why ETDRK4 is usable here at all."""
    E = np.exp(h * Lop)
    E2 = np.exp(h * Lop / 2.0)
    shape = np.shape(Lop)
    flat = np.reshape(Lop, -1)                      # d-generic: the contour is per-mode
    # The phi functions are entire, so their value at hL is the mean over any circle
    # around it. For a *real* operator (KS, Cahn-Hilliard) the upper half-circle and
    # the real part suffice, since the lower half contributes the conjugate. For a
    # complex operator -- the dispersive i*k^3 of KdV, the i*k^2 of Schrodinger --
    # the phi values are complex and that shortcut is wrong by O(1): it was measured
    # to lose 4% of KdV's L2 invariant by t = 1. Those need the full circle and the
    # complex mean.
    is_complex = np.iscomplexobj(Lop)
    if is_complex:
        r = np.exp(2j * np.pi * (np.arange(1, 2 * n_contour + 1) - 0.5) / (2 * n_contour))
    else:
        r = np.exp(1j * np.pi * (np.arange(1, n_contour + 1) - 0.5) / n_contour)
    # The contour is evaluated in row blocks. Materializing it whole costs
    # (modes x n_contour) complex numbers, which is nothing for a 1-D reference and
    # half a gigabyte for a two-field 512^2 one -- the same code either way, so the
    # block size is the only thing standing between this and an out-of-memory build.
    out = [np.empty(flat.size, dtype=complex if is_complex else float) for _ in range(4)]
    for lo in range(0, flat.size, chunk):
        LR = h * flat[lo:lo + chunk, None] + r[None, :]
        eLR = np.exp(LR)
        blocks = ((np.exp(LR / 2.0) - 1.0) / LR,
                  (-4.0 - LR + eLR * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3,
                  (2.0 + LR + eLR * (-2.0 + LR)) / LR ** 3,
                  (-4.0 - 3.0 * LR - LR ** 2 + eLR * (4.0 - LR)) / LR ** 3)
        for dest, expr in zip(out, blocks, strict=True):
            m = np.mean(expr, axis=1)
            dest[lo:lo + chunk] = h * (m if is_complex else np.real(m))
    Q, f1, f2, f3 = (a.reshape(shape) for a in out)
    return E, E2, Q, f1, f2, f3


def etdrk4(v0, Lop, nonlinear, h, nsteps):
    """Exponential time differencing RK4 in Fourier space (Cox-Matthews/Kassam-
    Trefethen). ``nonlinear(v) -> N(v)`` works on spectral coefficients."""
    E, E2, Q, f1, f2, f3 = _etdrk4_coefficients(Lop, h)
    v = np.array(v0, dtype=complex)
    for _ in range(nsteps):
        Nv = nonlinear(v)
        a = E2 * v + Q * Nv
        Na = nonlinear(a)
        b = E2 * v + Q * Na
        Nb = nonlinear(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = nonlinear(c)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
    return v


def imex_sbdf3(v0, Lop, nonlinear, h, nsteps, startup_substeps=2000):
    """Third-order semi-implicit BDF: the stiff linear part implicit, the nonlinear
    part extrapolated. A different time-integration family from ETDRK4 -- it does not
    exponentiate the operator at all -- which is what makes the comparison mean
    something.

    The two startup values come from a **substepped** SBDF1/SBDF2 ladder rather than
    from one step of each. A single SBDF1 step leaves an O(h^2) error that no later
    third-order step can remove, and on a chaotic problem it is then amplified for the
    rest of the integration; taking the first two steps in ``startup_substeps`` pieces
    drops it to O(h^2/m) for a few thousand extra steps out of half a million. It also
    keeps the scheme independent of ETDRK4, which a high-order startup would not."""
    v = np.array(v0, dtype=complex)
    d = h / startup_substeps
    for _ in range(2 * startup_substeps):                            # SBDF1 to t = 2h
        v = (v + d * nonlinear(v)) / (1.0 - d * Lop)
        if _ == startup_substeps - 1:
            v1 = np.array(v)
    v2 = v
    N0 = nonlinear(v0)
    N1, N2 = nonlinear(v1), nonlinear(v2)
    vm2, vm1, v = np.array(v0, dtype=complex), v1, v2
    Nm2, Nm1, Nn = N0, N1, N2
    denom = 11.0 / 6.0 - h * Lop
    for _ in range(nsteps - 2):
        rhs = (3.0 * v - 1.5 * vm1 + (1.0 / 3.0) * vm2
               + h * (3.0 * Nn - 3.0 * Nm1 + Nm2))
        vnew = rhs / denom
        vm2, vm1, v = vm1, v, vnew
        Nm2, Nm1 = Nm1, Nn
        Nn = nonlinear(v)
    return v


def rk4_spectral(v0, Lop, nonlinear, h, nsteps):
    """Classical explicit RK4 on the full spectral ODE ``v' = L v + N(v)``. No
    exponential and no splitting, so on a dispersive problem it is the time family
    least related to ETDRK4 -- at the price of the operator's own stability limit
    (``h |L|_max < 2.8``), which is why it is run on the grid where that is
    affordable and compared on the nodes it shares with the finer one."""
    v = np.array(v0, dtype=complex)

    def rhs(v):
        return Lop * v + nonlinear(v)
    for _ in range(nsteps):
        k1 = rhs(v)
        k2 = rhs(v + 0.5 * h * k1)
        k3 = rhs(v + 0.5 * h * k2)
        k4 = rhs(v + h * k3)
        v = v + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return v


# =============================================================================
# Per-problem reference generators
# =============================================================================
#
# Each returns a dict:
#   axes        {name: 1-D grid}, in the problem's axis order
#   periodic    {name: bool} -- whether the axis may be trig-interpolated
#   fields      {name: array}
#   t_eval      float
#   agreement   {label: relative difference}, all of which must be small
#   schemes     (primary, secondary) description strings


def _ks_rhs_factory(k, gamma=1.0):
    ik = 1j * k

    def nonlinear(v):
        u = np.real(np.fft.ifft(v))
        return -0.5 * gamma * ik * np.fft.fft(u * u)
    return nonlinear


def reference_kuramoto_sivashinsky():
    """u_t + u u_x + u_xx + u_xxxx = 0 on [0, 32*pi], periodic, to t = 50.

    Chaos does not prevent a reference here, and the reason is quantitative: at
    L = 32*pi the largest Lyapunov exponent is ~0.05-0.1, so an error converged to
    1e-12 amplifies by e^2.5 to e^5 over t = 50 and is still ~1e-10 -- six orders
    below the problem's own 1% target. Two independent time integrators on the same
    spectral discretization, plus a spatial refinement check, measure that claim
    rather than assert it."""
    L = 32.0 * np.pi
    M, T = 1024, 50.0
    x = _periodic_grid(L, M)
    k = 2.0 * np.pi * np.fft.fftfreq(M, d=L / M)
    Lop = k ** 2 - k ** 4                       # u_t = -u u_x - u_xx - u_xxxx
    u0 = np.cos(x / 16.0) * (1.0 + np.sin(x / 16.0))
    v0 = np.fft.fft(u0)
    nonlinear = _ks_rhs_factory(k)

    h_a = 5e-4
    u_a = np.real(np.fft.ifft(etdrk4(v0, Lop, nonlinear, h_a, int(round(T / h_a)))))
    h_b = 1e-4
    u_b = np.real(np.fft.ifft(imex_sbdf3(v0, Lop, nonlinear, h_b, int(round(T / h_b)))))

    # Temporal self-convergence of the primary scheme.
    h_c = 2e-3
    u_c = np.real(np.fft.ifft(etdrk4(v0, Lop, nonlinear, h_c, int(round(T / h_c)))))

    # Spatial: half the modes, compared on the coarse nodes it shares with the fine grid.
    Mc = M // 2
    xc = _periodic_grid(L, Mc)
    kc = 2.0 * np.pi * np.fft.fftfreq(Mc, d=L / Mc)
    v0c = np.fft.fft(np.cos(xc / 16.0) * (1.0 + np.sin(xc / 16.0)))
    u_half = np.real(np.fft.ifft(etdrk4(v0c, kc ** 2 - kc ** 4, _ks_rhs_factory(kc),
                                        h_a, int(round(T / h_a)))))

    return {
        "axes": {"x": x},
        "periodic": {"x": True},
        "fields": {"u": u_a},
        "t_eval": T,
        "agreement": {
            "etdrk4_vs_imex_sbdf3": rel_l2(u_a, u_b),
            "etdrk4_dt_2e-3_vs_5e-4": rel_l2(u_c, u_a),
            "spatial_M512_vs_M1024": rel_l2(u_half, u_a[::2]),
        },
        "schemes": ("ETDRK4 spectral, M=1024, dt=5e-4",
                    "IMEX-SBDF3 spectral, M=1024, dt=1e-4"),
    }


# --- Cahn-Hilliard, free coarsening (the matched pair to the MMS problem) ----

CH_EPS = 0.03
CH_L = 1.0
CH_T = 0.1


def cahn_hilliard_ic(X, Y):
    """A deterministic, band-limited, zero-mean initial condition.

    Zero mean because Cahn-Hilliard conserves it exactly, which makes mass a usable
    structural gate; band-limited because the reference is spectral and a discontinuous
    or random field would put the reference's own accuracy in question rather than the
    solver's. Four modes is enough to phase-separate into several domains and then
    coarsen -- the behaviour the manufactured version of this problem does not have,
    since there the source supplies all the dynamics."""
    p = 2.0 * np.pi
    return (0.3 * np.sin(p * X) * np.sin(p * Y)
            + 0.2 * np.cos(2 * p * X) * np.cos(p * Y)
            - 0.15 * np.sin(3 * p * X) * np.cos(2 * p * Y)
            + 0.1 * np.cos(p * X) * np.sin(3 * p * Y))


def _ch_setup(M, S=2.0):
    """Spectral operator for u_t = Delta(-eps^2 Delta u + u^3 - u).

    ``S`` is the standard stabilization: adding and subtracting ``S Delta u`` moves the
    bounded part of the nonlinear term's stiffness into the exactly-integrated linear
    operator, which is what makes a workable step size possible on a fourth-order
    operator. It changes no solution -- the two occurrences cancel identically."""
    x = _periodic_grid(CH_L, M)
    KX, KY = np.meshgrid(2.0 * np.pi * np.fft.fftfreq(M, d=CH_L / M),
                         2.0 * np.pi * np.fft.fftfreq(M, d=CH_L / M), indexing="ij")
    k2 = KX ** 2 + KY ** 2
    Lop = -CH_EPS ** 2 * k2 ** 2 + (1.0 - S) * k2

    def nonlinear(v):
        u = np.real(np.fft.ifft2(v))
        return -k2 * (np.fft.fft2(u ** 3) - S * v)

    X, Y = np.meshgrid(x, x, indexing="ij")
    return x, np.fft.fft2(cahn_hilliard_ic(X, Y)), Lop, nonlinear


def reference_cahn_hilliard_2d_coarsening():
    """u_t = Delta(-eps^2 Delta u + u^3 - u) on the periodic unit square, to t = 0.1.

    The matched pair to ``pde_cahn_hilliard_2d``: same operator, same eps, same domain,
    with the manufactured source deleted. That one change removes the closed form and
    replaces smooth decay with spinodal separation followed by coarsening -- where an
    over-dissipative or too-coarsely-stepped scheme merges domains at the wrong time
    and still returns a perfectly smooth field."""
    M = 128
    x, v0, Lop, nonlinear = _ch_setup(M)

    h_a = 4e-6
    u_a = np.real(np.fft.ifft2(etdrk4(v0, Lop, nonlinear, h_a, int(round(CH_T / h_a)))))
    h_b = 2e-6
    u_b = np.real(np.fft.ifft2(imex_sbdf3(v0, Lop, nonlinear, h_b, int(round(CH_T / h_b)),
                                          startup_substeps=200)))
    h_c = 1.6e-5
    u_c = np.real(np.fft.ifft2(etdrk4(v0, Lop, nonlinear, h_c, int(round(CH_T / h_c)))))

    Mc = M // 2
    _xc, v0c, Lopc, nlc = _ch_setup(Mc)
    u_half = np.real(np.fft.ifft2(etdrk4(v0c, Lopc, nlc, h_a, int(round(CH_T / h_a)))))

    return {
        "axes": {"x": x, "y": x},
        "periodic": {"x": True, "y": True},
        "fields": {"u": u_a},
        "t_eval": CH_T,
        "agreement": {
            "etdrk4_vs_imex_sbdf3": rel_l2(u_a, u_b),
            "etdrk4_dt_1.6e-5_vs_4e-6": rel_l2(u_c, u_a),
            "spatial_M64_vs_M128": rel_l2(u_half, u_a[::2, ::2]),
        },
        "schemes": ("ETDRK4 spectral (stabilized split), M=128^2, dt=4e-6",
                    "IMEX-SBDF3 spectral (stabilized split), M=128^2, dt=2e-6"),
    }


# --- KdV, Zabusky-Kruskal: dispersive soliton fission ------------------------

KDV_DELTA = 0.022
KDV_L = 2.0
KDV_T = 1.0


def _kdv_setup(M):
    x = _periodic_grid(KDV_L, M)
    k = 2.0 * np.pi * np.fft.fftfreq(M, d=KDV_L / M)
    Lop = 1j * KDV_DELTA ** 2 * k ** 3           # u_t = -u u_x - d^2 u_xxx
    ik = 1j * k
    dealias = np.abs(k) < (2.0 / 3.0) * np.max(np.abs(k))

    def nonlinear(v):
        u = np.real(np.fft.ifft(v))
        return -0.5 * ik * np.fft.fft(u * u) * dealias
    return x, k, Lop, nonlinear, np.fft.fft(np.cos(np.pi * x))


def reference_kdv_zabusky_kruskal():
    """u_t + u u_x + d^2 u_xxx = 0, d = 0.022, u(x,0) = cos(pi x) on [0, 2), to t = 1.

    The Zabusky-Kruskal experiment: the cosine steepens, breaks at t = 1/pi, and by
    t = 1 has fissioned into a train of solitons of width ~ d. The linear operator is
    purely dispersive, so the phi functions must be evaluated with the full complex
    contour (``_etdrk4_coefficients``), and the second family is a fully explicit RK4
    on the spectral ODE, run on the M = 512 grid where its dispersive CFL
    (h < 2.8 / (d^2 k_max^3)) is affordable and compared on the shared nodes -- the
    M = 512 solve is itself resolved to round-off (spatial check below)."""
    M = 1024
    x, k, Lop, nonlinear, v0 = _kdv_setup(M)
    h_a = 1e-4
    u_a = np.real(np.fft.ifft(etdrk4(v0, Lop, nonlinear, h_a, int(round(KDV_T / h_a)))))
    h_c = 2e-4
    u_c = np.real(np.fft.ifft(etdrk4(v0, Lop, nonlinear, h_c, int(round(KDV_T / h_c)))))

    Mb = 512
    _, kb, Lb, nlb, v0b = _kdv_setup(Mb)
    h_b = 0.5 * 2.8 / (KDV_DELTA ** 2 * float(np.max(np.abs(kb))) ** 3)
    nb = int(np.ceil(KDV_T / h_b))
    u_b = np.real(np.fft.ifft(rk4_spectral(v0b, Lb, nlb, KDV_T / nb, nb)))
    u_half = np.real(np.fft.ifft(etdrk4(v0b, Lb, nlb, h_a, int(round(KDV_T / h_a)))))

    # The equation conserves int u, int u^2 and int (u^3/3 - d^2 u_x^2) exactly; the
    # first two are computed spectrally on the reference and must be at round-off.
    dx = KDV_L / M
    ux = np.real(np.fft.ifft(1j * k * np.fft.fft(u_a)))
    l2_drift = abs(np.sum(u_a ** 2) * dx - 1.0)
    e0 = np.sum(np.cos(np.pi * x) ** 3 / 3.0 - KDV_DELTA ** 2 * (np.pi * np.sin(np.pi * x)) ** 2) * dx
    e1 = np.sum(u_a ** 3 / 3.0 - KDV_DELTA ** 2 * ux ** 2) * dx
    return {
        "axes": {"x": x},
        "periodic": {"x": True},
        "fields": {"u": u_a},
        "t_eval": KDV_T,
        "agreement": {
            "etdrk4_vs_rk4_M512": rel_l2(u_b, u_a[::2]),
            "etdrk4_dt_2e-4_vs_1e-4": rel_l2(u_c, u_a),
            "spatial_M512_vs_M1024": rel_l2(u_half, u_a[::2]),
            "l2_invariant_drift": l2_drift,
            "energy_invariant_drift": abs(e1 - e0) / abs(e0),
        },
        "schemes": ("ETDRK4 spectral (2/3 dealiased), M=1024, dt=1e-4",
                    f"explicit RK4 spectral, M=512, dt={KDV_T / nb:.3g}"),
    }


# --- Gross-Pitaevskii: a trapped, non-integrable NLS as a two-field system ----

GP_L = 10.0
GP_G = 5.0
GP_XC = 1.0
GP_T = 2.0
#: Interior sine modes of the solve; the endpoint-inclusive grid it lives on has
#: GP_N_SOLVE + 1 intervals.
GP_N_SOLVE = 1023
#: The stored grid has lcm(64, 129) = 8256 intervals, so both the harness's N = 65
#: and 2N = 130 nodes are exact subsets (the nesting rule in problems.py); the field
#: is put there by zero-padding its sine coefficients, which is exact.
GP_M_OUT = 8257


def gp_psi0(x):
    return np.pi ** -0.25 * np.exp(-0.5 * (x - GP_XC) ** 2) + 0j


def _sine_grid(n):
    """``n`` interior nodes of the endpoint-inclusive grid on [-L, L], and the sine
    wavenumbers k_m = m pi / 2L of the Dirichlet eigenbasis they carry."""
    x = np.linspace(-GP_L, GP_L, n + 2)[1:-1]
    k = np.arange(1, n + 1) * np.pi / (2.0 * GP_L)
    return x, k


def _dst(p):
    """Orthonormal DST-I of a complex interior field (sine coefficients)."""
    from scipy.fft import dst
    return dst(p.real, type=1, norm="ortho") + 1j * dst(p.imag, type=1, norm="ortho")


def _idst(a):
    from scipy.fft import idst
    return idst(a.real, type=1, norm="ortho") + 1j * idst(a.imag, type=1, norm="ortho")


def strang_gp(psi, k, potential, h, nsteps):
    """Strang split-step in the Dirichlet sine basis for
    ``i psi_t = -1/2 psi_xx + V psi + g |psi|^2 psi``, psi(+-L) = 0.

    Both substeps are *exact*: the kinetic flow is a phase rotation of the sine
    coefficients and the potential-plus-nonlinear flow is a pointwise phase rotation
    whose rate ``V + g |psi|^2`` is itself invariant under that rotation. The only
    error is the splitting, which makes this a different family from an exponential
    integrator on the kinetic operator with the potential treated as a nonlinearity."""
    ek = np.exp(-0.5j * k ** 2 * h)
    for _ in range(nsteps):
        psi = psi * np.exp(-0.5j * h * (potential + GP_G * np.abs(psi) ** 2))
        psi = _idst(ek * _dst(psi))
        psi = psi * np.exp(-0.5j * h * (potential + GP_G * np.abs(psi) ** 2))
    return psi


def yoshida4_gp(psi, k, potential, h, nsteps):
    """Fourth-order triple-jump composition of Strang steps (Yoshida 1990)."""
    w1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    w0 = 1.0 - 2.0 * w1
    for _ in range(nsteps):
        for w in (w1, w0, w1):
            psi = strang_gp(psi, k, potential, w * h, 1)
    return psi


def _gp_pair(p):
    return np.concatenate([p.real, p.imag])


def reference_gross_pitaevskii_1d():
    """i psi_t = -1/2 psi_xx + x^2/2 psi + g |psi|^2 psi, g = 5, psi(+-10) = 0, from a
    displaced Gaussian, to t = 2, as the real/imaginary pair.

    The wall matters at the 1e-7 level: by t = 2 the nonlinear dynamics have radiated
    fast low-amplitude ripples that reach x = +-10 and reflect, so a periodic box is
    *not* this problem (measured: a Fourier solve on the same box differs by ~1e-6 and
    has an algebraic spectral tail from the trap's kink at the seam). The sine basis
    carries the Dirichlet condition exactly. Three time families -- a fourth-order
    split-step composition (both substeps exact, only the splitting is approximate),
    ETDRK4 with the trap treated as part of the nonlinearity, and a fully explicit
    RK4 -- plus the primary's own step-halving and a spatial check."""
    n = GP_N_SOLVE
    x, k = _sine_grid(n)
    V = 0.5 * x ** 2
    Lop = -0.5j * k ** 2

    def nonlinear(a):
        p = _idst(a)
        return _dst(-1j * (V + GP_G * np.abs(p) ** 2) * p)
    p0 = gp_psi0(x)
    h = 2e-4
    p_a = yoshida4_gp(p0, k, V, h, int(round(GP_T / h)))
    p_c = yoshida4_gp(p0, k, V, 2.0 * h, int(round(GP_T / (2.0 * h))))
    p_b = _idst(etdrk4(_dst(p0), Lop, nonlinear, h, int(round(GP_T / h))))
    h_r = 0.5 * 2.8 / (0.5 * float(k[-1]) ** 2)
    n_r = int(np.ceil(GP_T / h_r))
    p_r = _idst(rk4_spectral(_dst(p0), Lop, nonlinear, GP_T / n_r, n_r))

    nc = (n + 1) // 2 - 1                          # every other node of the fine grid
    xc, kc = _sine_grid(nc)
    p_half = yoshida4_gp(gp_psi0(xc), kc, 0.5 * xc ** 2, h, int(round(GP_T / h)))

    dx = 2.0 * GP_L / (n + 1)
    mass_drift = abs(np.sum(np.abs(p_a) ** 2) * dx - 1.0)

    # Onto the nesting, endpoint-inclusive grid: zero-pad the sine coefficients.
    # Orthonormal DST-I scales as sqrt(2/(n+1)), so the padded coefficients are
    # rescaled by the ratio of the two normalizations.
    a = _dst(p_a)
    n_out = GP_M_OUT - 2
    a_out = np.zeros(n_out, dtype=complex)
    a_out[:n] = a * np.sqrt((n_out + 1) / (n + 1))
    p_out = np.zeros(GP_M_OUT, dtype=complex)
    p_out[1:-1] = _idst(a_out)
    x_out = np.linspace(-GP_L, GP_L, GP_M_OUT)
    # The solve grid (1024 intervals) and the stored grid (8256) share every 64th
    # unit of length: node 16j of the solve grid is node 129j of the stored one.
    g = np.gcd(n + 1, GP_M_OUT - 1)
    p_full = np.zeros(n + 2, dtype=complex)
    p_full[1:-1] = p_a
    shared_solve = p_full[::(n + 1) // g]
    shared_out = p_out[::(GP_M_OUT - 1) // g]
    return {
        "axes": {"x": x_out},
        "periodic": {"x": False},
        "fields": {"re": p_out.real, "im": p_out.imag},
        "t_eval": GP_T,
        "agreement": {
            "yoshida4_vs_etdrk4": rel_l2(_gp_pair(p_b), _gp_pair(p_a)),
            "yoshida4_vs_rk4": rel_l2(_gp_pair(p_r), _gp_pair(p_a)),
            "yoshida4_dt_4e-4_vs_2e-4": rel_l2(_gp_pair(p_c), _gp_pair(p_a)),
            "spatial_n511_vs_n1023": rel_l2(_gp_pair(p_half), _gp_pair(p_a[1::2])),
            "mass_invariant_drift": mass_drift,
            "resample_on_shared_nodes": rel_l2(_gp_pair(shared_out), _gp_pair(shared_solve)),
        },
        "schemes": ("Yoshida-4 split-step, sine basis, N=1025, dt=2e-4",
                    f"ETDRK4 sine-spectral (trap in N), N=1025, dt=2e-4; explicit RK4, dt={GP_T / n_r:.3g}"),
    }


# --- Gray-Scott: the trivial attractor, made concrete -----------------------

GS_DU, GS_DV, GS_F, GS_K = 3.2e-4, 1.6e-4, 0.04, 0.06
GS_L, GS_T, GS_W = 1.0, 300.0, 0.24


def gray_scott_ic(X, Y):
    """Two overlapping Gaussian seeds on the ``u = 1, v = 0`` background.

    Deterministic and smooth -- the literature's usual square patch seeded with random
    noise would make the reference's own accuracy the question rather than the
    solver's. The seed width is tied to the diffusion length: at these diffusivities a
    seed of the classic 0.06 simply dissolves, and the solution converges to the
    trivial state for real rather than through anybody's numerical error."""
    b = (np.exp(-((X - 0.42) ** 2 + (Y - 0.5) ** 2) / (2.0 * GS_W ** 2))
         + 0.7 * np.exp(-((X - 0.62) ** 2 + (Y - 0.56) ** 2) / (2.0 * (0.75 * GS_W) ** 2)))
    b = np.minimum(b, 1.0)
    return 1.0 - 0.5 * b, 0.25 * b


def _gs_setup(M):
    """Both species carried in one stacked array, so the shared integrators apply
    unchanged: every operation in them is elementwise, and ``fft2`` acts on the last
    two axes."""
    x = _periodic_grid(GS_L, M)
    kk = 2.0 * np.pi * np.fft.fftfreq(M, d=GS_L / M)
    KX, KY = np.meshgrid(kk, kk, indexing="ij")
    k2 = KX ** 2 + KY ** 2
    Lop = np.stack([-GS_DU * k2, -GS_DV * k2])
    X, Y = np.meshgrid(x, x, indexing="ij")
    u0, v0 = gray_scott_ic(X, Y)

    def nonlinear(V):
        u, v = np.real(np.fft.ifft2(V[0])), np.real(np.fft.ifft2(V[1]))
        uvv = u * v * v
        return np.stack([np.fft.fft2(-uvv + GS_F * (1.0 - u)),
                         np.fft.fft2(uvv - (GS_F + GS_K) * v)])

    return x, np.stack([np.fft.fft2(u0), np.fft.fft2(v0)]), Lop, nonlinear


def _gs_fields(V):
    return {"u": np.real(np.fft.ifft2(V[0])), "v": np.real(np.fft.ifft2(V[1]))}


def reference_gray_scott_2d():
    """Gray-Scott reaction-diffusion on the periodic unit square. **Parked -- see below.**

    The intent was the trivial-attractor trap of plan-no-closed-form.md §5f as a
    benchmark problem: ``u = 1, v = 0`` is an exact homogeneous steady state, so a
    scheme that over-diffuses and destroys the pattern satisfies every term of the
    operator to machine precision, having eliminated the only phenomenon the problem
    is about. A residual check cannot see that; a reference field and a non-triviality
    gate both can.

    **It does not ship, and the reason is worth more than the problem would have been.**
    Certifying the reference and discriminating between solvers turn out to pull in
    opposite directions here:

    * At ``T = 1000`` the pattern dynamics amplify spatial truncation error, so
      doubling the reference grid only halved the M/2-vs-M disagreement
      (3.2e-04 -> 1.5e-04) instead of collapsing it the way a spectral method should.
      No affordable resolution certifies that reference.
    * At ``T = 300``, where it *is* certifiable (spatial check 4.0e-05, schemes
      agreeing to 2.0e-07), a correct solve is already **7.1e-04 at M = 32** -- two
      points per feature -- and improves only to 6.9e-05 at M = 192. Every resolution
      clears the 1% gate, the observed order is ~1, and both grids pass, so the order
      check is waived too. Nothing discriminates.

    The cause is the metric, not the physics: the pattern occupies a small fraction of
    the domain while ``u ~ 1`` fills the rest, so relative L2 over the whole field
    dilutes exactly the error that matters. A version of this problem would need a
    pattern-weighted metric, or a regime whose features fill the domain -- neither of
    which is a change to make while calling the result the same problem. Retained
    because the next attempt should start from these numbers rather than rediscover
    them."""
    # Two measured constraints set M and T, and neither was guessable up front.
    #
    # M = 256 is not enough: its M/2-vs-M spatial check lands at 3.2e-04, only ~30x
    # below the problem's own 1% gate, so the grid doubles and the check moves to
    # 256-vs-512.
    #
    # T = 1000 is not reachable: the M/2-vs-M difference *grows* with the horizon
    # (5.0e-05 at t=100, 9.3e-05 at t=500, 2.0e-04 at t=1000) because the pattern
    # dynamics amplify spatial truncation error, so doubling the grid at t = 1000 only
    # halved the disagreement (3.2e-04 -> 1.5e-04) instead of collapsing it the way a
    # spectral method should. That is sensitivity, not discretization, and no
    # affordable resolution fixes it. The horizon is therefore inside the window where
    # the reference is certifiable -- the pattern has formed by t ~ 100 and the
    # amplification has not yet taken over.
    M = 512
    x, V0, Lop, nonlinear = _gs_setup(M)

    h_a = 0.25
    F_a = _gs_fields(etdrk4(V0, Lop, nonlinear, h_a, int(round(GS_T / h_a))))
    h_b = 0.125
    F_b = _gs_fields(imex_sbdf3(V0, Lop, nonlinear, h_b, int(round(GS_T / h_b)),
                                startup_substeps=100))
    h_c = 1.0
    F_c = _gs_fields(etdrk4(V0, Lop, nonlinear, h_c, int(round(GS_T / h_c))))

    Mc = M // 2
    _xc, V0c, Lopc, nlc = _gs_setup(Mc)
    F_half = _gs_fields(etdrk4(V0c, Lopc, nlc, h_a, int(round(GS_T / h_a))))

    worst = lambda A, B, sl=slice(None): max(  # noqa: E731
        rel_l2(A[n][sl], B[n][sl]) for n in ("u", "v"))
    return {
        "axes": {"x": x, "y": x},
        "periodic": {"x": True, "y": True},
        "fields": F_a,
        "t_eval": GS_T,
        "agreement": {
            "etdrk4_vs_imex_sbdf3": worst(F_a, F_b),
            "etdrk4_dt_1.0_vs_0.25": worst(F_c, F_a),
            "spatial_M256_vs_M512": max(rel_l2(F_half[n], F_a[n][::2, ::2])
                                        for n in ("u", "v")),
        },
        "schemes": ("ETDRK4 spectral, M=512^2, dt=0.25",
                    "IMEX-SBDF3 spectral, M=512^2, dt=0.125"),
    }


GENERATORS = {
    "pde_kuramoto_sivashinsky": reference_kuramoto_sivashinsky,
    "pde_cahn_hilliard_2d_coarsening": reference_cahn_hilliard_2d_coarsening,
    "pde_kdv_zabusky_kruskal": reference_kdv_zabusky_kruskal,
    "pde_gross_pitaevskii_1d": reference_gross_pitaevskii_1d,
}

#: Built on request, never by ``--all``: a generator whose problem does not ship, kept
#: so the next attempt can rebuild and re-measure it rather than start over.
PARKED = {
    "pde_gray_scott_2d": reference_gray_scott_2d,
}
ALL_GENERATORS = {**GENERATORS, **PARKED}


# =============================================================================
# Artifact I/O
# =============================================================================


def _pack(slug, ref):
    names = list(ref["axes"])
    err = max(ref["agreement"].values())
    data = {
        "axis_names": np.array(names, dtype="U32"),
        "field_names": np.array(list(ref["fields"]), dtype="U32"),
        "periodic": np.array([ref["periodic"][n] for n in names], dtype=bool),
        "t_eval": np.array(float(ref["t_eval"])),
        "reference_error": np.array(float(err)),
        "schemes": np.array(list(ref["schemes"]), dtype="U128"),
    }
    for n in names:
        data[f"axis_{n}"] = np.asarray(ref["axes"][n], dtype=float)
    for n, v in ref["fields"].items():
        data[f"field_{n}"] = np.asarray(v, dtype=float)
    for label, value in ref["agreement"].items():
        data[f"agreement_{label}"] = np.array(float(value))
    return data, err


def build(slug, write=True):
    t0 = time.time()
    ref = ALL_GENERATORS[slug]()
    data, err = _pack(slug, ref)
    wall = time.time() - t0
    print(f"\n{slug}  ({wall:.1f}s)")
    print(f"  schemes : {ref['schemes'][0]}\n            {ref['schemes'][1]}")
    for label, value in ref["agreement"].items():
        print(f"  {label:<32} {value:.3e}")
    print(f"  reference_error (max)            {err:.3e}")

    path = os.path.join(REF_DIR, f"{slug}.npz")
    if write:
        os.makedirs(REF_DIR, exist_ok=True)
        np.savez_compressed(path, **data)
        print(f"  wrote {os.path.relpath(path, os.path.dirname(HERE))}")
        return err, True

    if not os.path.exists(path):
        print("  CHECK: no stored artifact to compare against")
        return err, False
    with np.load(path, allow_pickle=False) as z:
        drift = max(rel_l2(data[f"field_{n}"], z[f"field_{n}"])
                    for n in ref["fields"])
        stored = float(z["reference_error"])
    ok = drift <= max(10.0 * stored, 1e-12)
    print(f"  CHECK: drift vs stored {drift:.3e} (stored reference_error {stored:.3e}) "
          f"-> {'ok' if ok else 'MISMATCH'}")
    return err, ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated slugs")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="recompute and compare against the stored artifact; write nothing")
    args = ap.parse_args(argv)

    if args.list:
        for slug in ALL_GENERATORS:
            path = os.path.join(REF_DIR, f"{slug}.npz")
            tag = "parked" if slug in PARKED else (
                "present" if os.path.exists(path) else "MISSING")
            print(f"{slug:<40} {tag}")
        return 0

    if args.only:
        slugs = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [s for s in slugs if s not in ALL_GENERATORS]
        if unknown:
            ap.error(f"no reference generator for: {', '.join(unknown)}")
    elif args.all:
        slugs = list(GENERATORS)
    else:
        ap.error("pass --only <slugs>, --all, or --list")

    ok = True
    for slug in slugs:
        _err, good = build(slug, write=not args.check)
        ok = ok and good
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
