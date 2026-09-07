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
    r = np.exp(1j * np.pi * (np.arange(1, n_contour + 1) - 0.5) / n_contour)
    # The contour is evaluated in row blocks. Materializing it whole costs
    # (modes x n_contour) complex numbers, which is nothing for a 1-D reference and
    # half a gigabyte for a two-field 512^2 one -- the same code either way, so the
    # block size is the only thing standing between this and an out-of-memory build.
    out = [np.empty(flat.size) for _ in range(4)]
    for lo in range(0, flat.size, chunk):
        LR = h * flat[lo:lo + chunk, None] + r[None, :]
        eLR = np.exp(LR)
        blocks = ((np.exp(LR / 2.0) - 1.0) / LR,
                  (-4.0 - LR + eLR * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3,
                  (2.0 + LR + eLR * (-2.0 + LR)) / LR ** 3,
                  (-4.0 - 3.0 * LR - LR ** 2 + eLR * (4.0 - LR)) / LR ** 3)
        for dest, expr in zip(out, blocks, strict=True):
            dest[lo:lo + chunk] = h * np.real(np.mean(expr, axis=1))
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
