"""Autonumerics benchmark problem set.

37 curated problems: 15 PDE (5 per tier) + 22 SDE (5 tier-1, 5 tier-2, 12 tier-3).
The SDE tier 3 is weighted toward hard cases from a numerical-SDE literature review
(Feller-violated log-Heston, many-channel stiffness, superlinear/tamed drift, and
blow-up/domain stability), on top of the five classical tier-3 stress regimes.

Each problem carries:
  - metadata (id, slug, type, tier, family, title, challenge)
  - the ``description`` that becomes ``workspace/{slug}/problem.md``
  - a *ground-truth* callable that the independent verifier uses to check the
    pipeline's final numerical output. This ground truth is authored and
    validated HERE, independently of the pipeline's own formulator, so a
    formulator mistake cannot corrupt the benchmark verdict.

Ground-truth conventions
-------------------------
Every problem carries a ``ground_truth_kind`` (default ``"exact"``) that tells the
verifier how to check it:

* ``exact`` -- an exact closed-form / machine-precision ground truth. SDE problems
  expose ``moments(T) -> dict`` returning the exact terminal moments; keys are
  ``mean``/``variance`` for scalar state, or ``mean_X``/``variance_X``/``mean_Y``/
  ``variance_Y`` (and ``_Z`` for 3-D) for vector state. PDE problems expose
  ``analytic(t, *coords) -> ndarray`` giving the exact solution at ``t_eval``.
* ``reference`` -- no elementary moment formula, but the *exact solution path* is
  known, so ``moments(T)`` returns ``{key: (value, standard_error)}`` from a
  discretization-error-free Monte-Carlo of that exact solution. The verifier
  compares within an SE-aware tolerance.
* ``stability`` -- no ground truth at all; the independent check is only that a
  correct scheme stays finite / in its domain (a ``stability_check`` field says
  which). ``has_ground_truth`` is ``False`` but the problem is still independently
  checkable for blow-up.

PDE ``coords`` is ``(x,)`` for 1-D or the meshgrid ``(X, Y)`` for 2-D; steady
problems ignore ``t``. A problem with no closed form *and* no stability check sets
``analytic=None``/``has_ground_truth=False`` with the default ``exact`` kind; the
verifier reports "pipeline self-score only" for those (e.g. Kuramoto-Sivashinsky).

Tiers
-----
1 = textbook, covered by the pipeline's reference manuals (expected to pass)
2 = moderate: correct but requires care (boundary layers, indefinite operators,
    multi-D, scale, positivity/log guards)
3 = hard: beyond the reference manuals (shocks, fractional operators, stiffness,
    heavy-tailed sampling, chaos) -- where the benchmark exposes limits
"""

from __future__ import annotations

from functools import lru_cache, partial

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import erf, erfcx, ndtr  # ndtr = standard-normal CDF

# =============================================================================
# SDE ground-truth moment formulas (validated in validate_ground_truth.py)
# =============================================================================


def gbm_moments(T, X0, mu, sigma):
    """Geometric Brownian motion / Black-Scholes asset (lognormal)."""
    mean = X0 * np.exp(mu * T)
    var = X0**2 * np.exp(2 * mu * T) * (np.exp(sigma**2 * T) - 1.0)
    return {"mean": mean, "variance": var}


def ou_moments(T, X0, theta, mu, sigma):
    """Ornstein-Uhlenbeck (Vasicek)."""
    mean = mu + (X0 - mu) * np.exp(-theta * T)
    var = sigma**2 / (2 * theta) * (1.0 - np.exp(-2 * theta * T))
    return {"mean": mean, "variance": var}


def bm_drift_moments(T, X0, mu, sigma):
    """Brownian motion with constant drift (additive noise)."""
    return {"mean": X0 + mu * T, "variance": sigma**2 * T}


def bm_standard_moments(T):
    """Standard Brownian motion started at 0."""
    return {"mean": 0.0, "variance": float(T)}


def linear_additive_moments(T, X0, a, b, c):
    """Linear SDE with additive noise: dX = (a + b X) dt + c dW."""
    mean = np.exp(b * T) * (X0 + a / b) - a / b
    var = c**2 / (2 * b) * (np.exp(2 * b * T) - 1.0)
    return {"mean": mean, "variance": var}


def cir_moments(T, X0, kappa, theta, sigma):
    """Cox-Ingersoll-Ross square-root process.

    Moment formulas hold regardless of the Feller condition 2*kappa*theta >= sigma**2;
    the Feller condition only governs whether the path can reach 0, not the moments.
    """
    mean = theta + (X0 - theta) * np.exp(-kappa * T)
    var = (sigma**2 / kappa) * (
        X0 * (np.exp(-kappa * T) - np.exp(-2 * kappa * T))
        + 0.5 * theta * (1.0 - np.exp(-kappa * T)) ** 2
    )
    return {"mean": mean, "variance": var}


def exp_ou_moments(T, X0, theta, sigma):
    """Exponential Ornstein-Uhlenbeck (lognormal, mean-reverting log)."""
    v = sigma**2 / (2 * theta) * (1.0 - np.exp(-2 * theta * T))
    mean = np.exp(v / 2)
    var = np.exp(v) * (np.exp(v) - 1.0)
    return {"mean": mean, "variance": var}


def gbm2d_moments(T, X0, Y0, mu1, s1, mu2, s2):
    """Two correlated GBMs. Correlation does not affect the marginal moments."""
    return {
        "mean_X": X0 * np.exp(mu1 * T),
        "variance_X": X0**2 * np.exp(2 * mu1 * T) * (np.exp(s1**2 * T) - 1.0),
        "mean_Y": Y0 * np.exp(mu2 * T),
        "variance_Y": Y0**2 * np.exp(2 * mu2 * T) * (np.exp(s2**2 * T) - 1.0),
    }


def oscillator_moments(T, X0, Y0, sigma):
    """Stochastic harmonic oscillator: dX = Y dt, dY = -X dt + sigma dW."""
    return {
        "mean_X": X0 * np.cos(T) + Y0 * np.sin(T),
        "variance_X": sigma**2 / 2 * (T - np.sin(T) * np.cos(T)),
        "mean_Y": -X0 * np.sin(T) + Y0 * np.cos(T),
        "variance_Y": sigma**2 / 2 * (T + np.sin(T) * np.cos(T)),
    }


# =============================================================================
# Tier-3 "hard SDE" ground truth (from benchmark/hard_sde_candidates_from_lit.md)
# =============================================================================


def log_heston_moments(T, r, a, b, sigma, rho, X0, Y0):
    """Log-Heston terminal moments of (X=log-price, Y=variance), exact.

        dX = (r - Y/2) dt + sqrt(Y) (rho dW1 + sqrt(1-rho^2) dW2)
        dY = (a - b Y) dt + sigma sqrt(Y) dW1          (CIR variance)

    (X, Y) is an affine process, so the moment system

        s = [E X, E Y, E X^2, E XY, E Y^2],   ds/dt = A s + c

    *closes* (all right-hand sides are linear in the moments). It is solved
    exactly by the matrix exponential of the 6x6 augmentation [[A, c], [0, 0]].
    Independent of the Feller condition; valid for any sigma^2 vs 2a (or 4a).
    """
    A = np.array([
        [0.0, -0.5,            0.0,  0.0,  0.0],   # d E[X]
        [0.0, -b,              0.0,  0.0,  0.0],   # d E[Y]
        [2 * r, 1.0,           0.0, -1.0,  0.0],   # d E[X^2]
        [a,   r + rho * sigma, 0.0, -b,  -0.5],    # d E[XY]
        [0.0, 2 * a + sigma**2, 0.0, 0.0, -2 * b],  # d E[Y^2]
    ])
    c = np.array([r, a, 0.0, 0.0, 0.0])
    s0 = np.array([X0, Y0, X0**2, X0 * Y0, Y0**2])
    aug = np.zeros((6, 6))
    aug[:5, :5] = A
    aug[:5, 5] = c
    sT = (expm(aug * T) @ np.concatenate([s0, [1.0]]))[:5]
    EX, EY, EX2, EXY, EY2 = sT
    return {
        "mean_X": float(EX), "variance_X": float(EX2 - EX**2),
        "mean_Y": float(EY), "variance_Y": float(EY2 - EY**2),
    }


# --- multi-channel stiff linear system --------------------------------------
# dX = F X dt + sum_{r=1}^m G_r X dW_r,  X in R^2, m non-commuting noise channels.
# Ground truth is exact:
#   mean     : E[X_T] = expm(F T) X0
#   2nd mom. : P = E[X X^T] solves dP/dt = F P + P F^T + sum_r G_r P G_r^T, i.e.
#              vec(P_T) = expm((I(x)F + F(x)I + sum_r G_r(x)G_r) T) vec(P0).
# The two shear generators do not commute ([M1, M2] = diag(1, -1) != 0). Explicit
# Euler-Maruyama is mean-square unstable as the channel count m grows; a split-step
# / drift-implicit scheme is required. Parameters chosen (and validated by direct
# simulation) so the terminal variance is O(0.1-1) -- comfortably MC-estimable --
# while the mean has decayed below the near-zero-mean skip threshold.
_MC_M1 = np.array([[0.0, 1.0], [0.0, 0.0]])
_MC_M2 = np.array([[0.0, 0.0], [1.0, 0.0]])
_MC_F = np.array([[-2.0, 3.0], [-3.0, -2.0]])   # spiral drift: Re(eig) = -2, Im = +-3
_MC_M = 13                                       # noise-channel count
_MC_GAMMA = 0.6                                  # per-channel noise scale
_MC_GS = [_MC_GAMMA * (_MC_M1 if r % 2 == 0 else _MC_M2) for r in range(_MC_M)]
_MC_X0 = np.array([1.0, 1.0])
# The parameters are chosen so the terminal moments are cleanly MC-estimable AND a
# competent Euler-Maruyama at the harness step resolves the variance within the pass
# gate (strong multiplicative noise makes the terminal variance step-size sensitive,
# so a runaway-stiff regime would false-fail a correct-but-explicit solver when the
# verifier re-runs it). The exact moments are cross-checked three ways in
# validate_ground_truth.py (matrix Lyapunov ODE, vec/expm, direct simulation);
# retune _MC_F / _MC_GAMMA there if any drifts.


def multichannel_stiff_moments(T, F=_MC_F, Gs=_MC_GS, X0=_MC_X0):
    """Exact terminal mean/variance of the multi-channel stiff linear system."""
    F = np.asarray(F, dtype=float)
    X0 = np.asarray(X0, dtype=float)
    d = F.shape[0]
    mean = expm(F * T) @ X0
    Ii = np.eye(d)
    M = np.kron(Ii, F) + np.kron(F, Ii) + sum(np.kron(G, G) for G in Gs)
    vecP0 = np.outer(X0, X0).reshape(-1, order="F")     # column-major vec
    PT = (expm(M * T) @ vecP0).reshape(d, d, order="F")
    var = np.diag(PT) - mean**2
    out = {}
    for i, lab in enumerate(["X", "Y", "Z"][:d]):
        out[f"mean_{lab}"] = float(mean[i])
        out[f"variance_{lab}"] = float(var[i])
    return out


def quintic_ic_moments(T, sigma_bar, n_gh=160):
    """Noise-free quintic dX = -X^5 dt with random IC X0 ~ N(0, sigma_bar^2).

    Exact per-sample solution X_t = xi / (1 + 4 t xi^4)^{1/4}; the terminal law is
    a deterministic transform of the Gaussian IC, so the moments are a 1-D Gaussian
    integral evaluated to machine precision by Gauss-Hermite quadrature. The mean is
    0 by symmetry (X_t is odd in xi); the variance is E[X_T^2].
    """
    u, w = np.polynomial.hermite.hermgauss(n_gh)        # weight exp(-u^2)
    xi = np.sqrt(2.0) * sigma_bar * u                   # xi ~ N(0, sigma_bar^2)
    second_moment = (w @ (xi**2 / np.sqrt(1.0 + 4.0 * T * xi**4))) / np.sqrt(np.pi)
    return {"mean": 0.0, "variance": float(second_moment)}


# --- stochastic Ginzburg-Landau: REFERENCE moments (MC of the exact solution) ---
# dX = ((sigma^2/2) X - X^3) dt + sigma X dW,  X0 = 1, T = 1.
# Exact solution: X_t = exp(sigma W_t) / sqrt(1 + 2 * int_0^t exp(2 sigma W_u) du).
# There is no elementary moment formula, but simulating the *exact solution* (only
# the inner integral is discretized, finely) gives a reference with zero scheme bias
# -- so it cleanly exposes a wrong (e.g. untamed-Euler) scheme. Anchored to the
# literature value E[X_1^2] = 0.8114 at sigma = 2 in validate_ground_truth.py.
_GL_REF_N = 400_000
_GL_REF_DT = 0.001
_GL_REF_SEED = 20260715


def _gl_exact_mc(sigma, T=1.0, n_paths=_GL_REF_N, dt_ref=_GL_REF_DT,
                 seed=_GL_REF_SEED, chunk=25_000):
    """MC of the exact GL solution. Returns (mean, se_mean, var, se_var, second_moment)."""
    rng = np.random.default_rng(seed)
    n_steps = int(round(T / dt_ref))
    dt = T / n_steps
    xt = np.empty(n_paths)
    done = 0
    while done < n_paths:
        b = min(chunk, n_paths - done)
        dW = rng.standard_normal((b, n_steps)) * np.sqrt(dt)
        W = np.concatenate([np.zeros((b, 1)), np.cumsum(dW, axis=1)], axis=1)
        I_T = np.trapezoid(np.exp(2.0 * sigma * W), dx=dt, axis=1)
        xt[done:done + b] = np.exp(sigma * W[:, -1]) / np.sqrt(1.0 + 2.0 * I_T)
        done += b
    n = n_paths
    mean = float(xt.mean())
    var = float(xt.var(ddof=1))
    se_mean = float(xt.std(ddof=1) / np.sqrt(n))
    m4 = float(np.mean((xt - mean) ** 4))
    se_var = float(np.sqrt(max(m4 - (n - 3) / (n - 1) * var**2, 0.0) / n))
    return mean, se_mean, var, se_var, float(np.mean(xt**2))


@lru_cache(maxsize=None)
def _gl_ref_cached(sigma):
    mean, se_mean, var, se_var, _sm = _gl_exact_mc(sigma)
    return {"mean": (mean, se_mean), "variance": (var, se_var)}


def gl_reference_moments(T, sigma):
    """Reference terminal moments (value, standard_error) for Ginzburg-Landau."""
    if abs(T - 1.0) > 1e-12:
        raise ValueError("Ginzburg-Landau reference moments are frozen at T = 1")
    return _gl_ref_cached(float(sigma))


# =============================================================================
# PDE ground-truth analytic solutions
# =============================================================================


def _heat_1d(t, x, *, alpha):
    return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)


def _heat_2d(t, X, Y, *, alpha):
    return np.exp(-2.0 * alpha * np.pi**2 * t) * np.sin(np.pi * X) * np.sin(np.pi * Y)


def _wave_1d(t, x, *, c):
    return np.cos(np.pi * c * t) * np.sin(np.pi * x)


def _wave_2d(t, X, Y, *, c):
    return np.cos(np.sqrt(2.0) * np.pi * c * t) * np.sin(np.pi * X) * np.sin(np.pi * Y)


def _advection_1d(t, x, *, a):
    # Periodic transport on [0,1] of a narrow Gaussian initially centered at 0.3.
    # Signed periodic offset of x from the moving pulse center (0.3 + a t).
    center = (0.3 + a * t) % 1.0
    d = (x - center + 0.5) % 1.0 - 0.5  # nearest periodic image, in [-0.5, 0.5)
    return np.exp(-100.0 * d**2)


def _poisson_2d(_t, X, Y):
    return np.sin(np.pi * X) * np.sin(np.pi * Y)


def _laplace_2d(_t, X, Y):
    return np.sinh(np.pi * Y) * np.sin(np.pi * X) / np.sinh(np.pi)


def _helmholtz_2d(_t, X, Y):
    return np.sin(np.pi * X) * np.sin(np.pi * Y)


def _anisotropic_2d(_t, X, Y):
    return np.sin(np.pi * X) * np.sin(np.pi * Y)


def _convection_diffusion_bl(_t, x, *, eps):
    # -eps u'' + u' = 1, u(0)=u(1)=0. With eps=1e-3, exp(-1/eps) underflows to 0,
    # and every exponent below is <= 0, so there is no overflow.
    return x - (np.exp(-(1.0 - x) / eps) - np.exp(-1.0 / eps)) / (1.0 - np.exp(-1.0 / eps))


def _burgers_inviscid(t, x):
    # Riemann data u=1 (x<0), u=0 (x>0); entropy shock travels at speed 1/2.
    return np.where(x < 0.5 * t, 1.0, 0.0)


def _fokker_planck_ou(t, x, *, mu0, var0):
    mu = mu0 * np.exp(-t)
    var = 1.0 + (var0 - 1.0) * np.exp(-2.0 * t)
    return np.exp(-((x - mu) ** 2) / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)


def _fractional_diffusion(t, x, *, D, alpha):
    # d_t^alpha u = D u_xx, u(x,0)=sin(pi x), Dirichlet 0.
    # Mode solution u = E_alpha(-D pi^2 t^alpha) sin(pi x).
    # For alpha=1/2, E_{1/2}(-y) = exp(y^2) erfc(y) = erfcx(y), computed stably.
    y = D * np.pi**2 * t**alpha
    return erfcx(y) * np.sin(np.pi * x)


def _black_scholes_call(_t, S, *, K, r, sigma, T):
    # Present value V(S, t=0) of a European call; time-to-maturity tau = T.
    S = np.asarray(S, dtype=float)
    out = np.zeros_like(S)
    pos = S > 0
    Sp = S[pos]
    d1 = (np.log(Sp / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    out[pos] = Sp * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)
    return out


# =============================================================================
# PDE ground truth -- HardNumerics-LowDim imports, Batch 1 (scalar)
# (validated independently in validate_ground_truth.py)
# =============================================================================

# --- Stefan one-phase melting: similarity solution (moving free boundary) ----
# lambda solves  lambda e^{lambda^2} erf(lambda) = 1/sqrt(pi); front s(t)=2 lambda sqrt(t).
_STEFAN_LAMBDA = brentq(lambda L: L * np.exp(L**2) * erf(L) - 1.0 / np.sqrt(np.pi),
                        1e-6, 5.0)


def _stefan_1d(t, x):
    """Temperature of the one-phase Stefan problem: melt for x<s(t), solid (u=0)
    ahead of the front. The kink at the moving front s(t) is what limits the L2
    convergence order."""
    x = np.asarray(x, dtype=float)
    s = 2.0 * _STEFAN_LAMBDA * np.sqrt(t)
    u = 1.0 - erf(x / (2.0 * np.sqrt(t))) / erf(_STEFAN_LAMBDA)
    return np.where(x < s, u, 0.0)


def _monge_ampere_2d(_t, X, Y):
    """Exact convex solution of det(D^2 u)=exp(x^2+y^2)(1+x^2+y^2) on [0,1]^2."""
    return np.exp((X**2 + Y**2) / 2.0)


# --- porous medium (degenerate diffusion): Barenblatt-Pattle profile ---------
# u_t = Delta(u^m), m=2, d=2. The self-similar solution has exponent 1/(m-1)=1;
# alpha=1/2, beta=1/4, and k = alpha(m-1)/(2 d m) = 1/16 (the HardNumerics spec's
# k=1/8 is inconsistent with its own formula -- 1/16 is the value that makes the
# PME residual vanish, confirmed symbolically in validate_ground_truth.py).
_BARENBLATT_C = 0.5
_BARENBLATT_K = 1.0 / 16.0


def _barenblatt_2d(t, X, Y, *, C=_BARENBLATT_C, k=_BARENBLATT_K):
    r2 = X**2 + Y**2
    inner = C - k * r2 * t**(-0.5)
    return t**(-0.5) * np.maximum(inner, 0.0)


# --- Poisson on an L-shaped (re-entrant corner) domain -----------------------
# Harmonic singular solution u = r^{2/3} sin(2 theta/3), theta in (0, 3pi/2), on
# Omega = (-1,1)^2 \ [0,1]x[-1,0]. u is not in H^2 at the corner, so uniform
# low-order schemes lose convergence rate. Scored over the in-domain nodes only.
def _reentrant_corner_2d(_t, X, Y):
    r = np.sqrt(X**2 + Y**2)
    th = np.arctan2(Y, X)
    th = np.where(th < 0.0, th + 2.0 * np.pi, th)   # branch in (0, 2pi); domain uses (0, 3pi/2)
    with np.errstate(invalid="ignore"):
        u = np.power(r, 2.0 / 3.0) * np.sin(2.0 / 3.0 * th)
    return np.where(r > 0.0, u, 0.0)


def _lshape_mask(X, Y):
    """In-domain mask for the L-shape (-1,1)^2 minus the [0,1]x[-1,0] quadrant."""
    return ~((X > 0.0) & (Y < 0.0))


# --- Cahn-Hilliard (4th-order, stiff): manufactured solution + source --------
# u_t = Delta mu + s,  mu = -eps^2 Delta u + u^3 - u,  periodic on [0,1]^2.
# u* = 0.2 sin(2pi x) sin(2pi y) cos(t) is hidden; the solver is handed the source
# s(x,y,t) below (derived from u* and cross-checked against a sympy derivation and
# the PDE residual in validate_ground_truth.py).
_CH_EPS = 0.1


def _cahn_hilliard_2d(t, X, Y):
    return 0.2 * np.sin(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y) * np.cos(t)


def cahn_hilliard_source(x, y, t, eps=_CH_EPS):
    """MMS source s = u_t - Delta(mu) for the manufactured Cahn-Hilliard solution.
    p = sin(2pi x) sin(2pi y); uses Delta u = -8 pi^2 u."""
    pi = np.pi
    sx, sy = np.sin(2 * pi * x), np.sin(2 * pi * y)
    cx, cy = np.cos(2 * pi * x), np.cos(2 * pi * y)
    p = sx * sy
    grad_factor = cx**2 * sy**2 + sx**2 * cy**2      # |grad u|^2 / (0.16 pi^2 cos^2 t)
    return (-0.2 * p * np.sin(t)
            + 1.6 * pi**2 * (8 * pi**2 * eps**2 - 1.0) * p * np.cos(t)
            + 0.192 * pi**2 * np.cos(t)**3 * p * (p**2 - grad_factor))


# --- Heston European call: semi-closed characteristic-function reference ------
# The "answer" is a deterministic Gauss-Legendre quadrature of Heston's (1993)
# formula (Albrecher et al. "Little Trap" form, numerically stable) -- there is no
# elementary closed form to transcribe, which is exactly why it resists leakage.
_HESTON = dict(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, r=0.03, K=1.0, T=1.0)
_HESTON_GL_N = 160
_HESTON_PHI_MAX = 200.0


def _heston_call(_t, S, v, *, params=_HESTON):
    kappa, theta = params["kappa"], params["theta"]
    sigma, rho = params["sigma"], params["rho"]
    r, K, T = params["r"], params["K"], params["T"]
    S = np.asarray(S, dtype=float)
    v = np.asarray(v, dtype=float)
    x = np.log(np.maximum(S, 1e-300))
    nodes, weights = np.polynomial.legendre.leggauss(_HESTON_GL_N)
    phi = 0.5 * _HESTON_PHI_MAX * (nodes + 1.0)
    w = 0.5 * _HESTON_PHI_MAX * weights
    phi_b = phi.reshape((1,) * x.ndim + (-1,))
    xb, vb = x[..., None], v[..., None]
    i = 1j

    def prob(j):
        u = 0.5 if j == 1 else -0.5
        b = kappa - rho * sigma if j == 1 else kappa
        rs = rho * sigma * i * phi_b
        d = np.sqrt((rs - b)**2 - sigma**2 * (2 * u * i * phi_b - phi_b**2))
        g = (b - rs - d) / (b - rs + d)          # little-trap factor (= 1/g_original)
        edt = np.exp(-d * T)
        D = ((b - rs - d) / sigma**2) * ((1.0 - edt) / (1.0 - g * edt))
        C = (r * i * phi_b * T
             + (kappa * theta / sigma**2)
             * ((b - rs - d) * T - 2.0 * np.log((1.0 - g * edt) / (1.0 - g))))
        f = np.exp(C + D * vb + i * phi_b * xb)
        integrand = np.real(np.exp(-i * phi_b * np.log(K)) * f / (i * phi_b))
        return 0.5 + (1.0 / np.pi) * np.sum(integrand * w, axis=-1)

    price = S * prob(1) - K * np.exp(-r * T) * prob(2)
    return np.maximum(price, 0.0)


# =============================================================================
# PDE ground truth -- HardNumerics-LowDim imports, Batch 2 (systems / 3-D)
# Multi-field problems return an {field_name: ndarray} dict from ``analytic``; the
# manufactured / exact fields are hidden here and validated in
# validate_ground_truth.py (against the PDE residual and the div-free constraints).
# =============================================================================

# --- Fichera 3-D corner singularity (scalar, MMS, cube-minus-octant) ---------
_FICHERA_EPS = 1e-4
_FICHERA_Q = 0.5


def _fichera_3d(_t, X, Y, Z):
    return (X**2 + Y**2 + Z**2 + _FICHERA_EPS**2) ** (_FICHERA_Q / 2.0)


def fichera_source(X, Y, Z):
    """f = -Delta u for the regularized radial singular solution."""
    q, eps2 = _FICHERA_Q, _FICHERA_EPS**2
    r2 = X**2 + Y**2 + Z**2
    lap = q * (3.0 * eps2 + (q + 1.0) * r2) * (r2 + eps2) ** (q / 2.0 - 2.0)
    return -lap


def _fichera_mask(X, Y, Z):
    """(-1,1)^3 with the positive octant [0,1]^3 removed (the re-entrant vertex)."""
    return ~((X > 0.0) & (Y > 0.0) & (Z > 0.0))


# --- 3-D acoustic wave in a layered (discontinuous-c) medium (scalar, MMS) ----
# Wavenumber 2*pi per axis (one wavelength across the unit cube): a genuine 3-D
# layered-media wave (C2 + E1 axes) that a competent 2nd-order FD scheme can still
# resolve at the harness's 3-D grid_N -- 4*pi (the HardNumerics figure) needs an
# infeasibly fine 3-D grid to clear the accuracy gate (validated with a reference
# leapfrog solver: 4*pi is ~10% error at N=64, 2*pi is ~0.2%).
_ACOUSTIC_K = 2.0 * np.pi                 # spatial wavenumber per axis
_ACOUSTIC_W = np.sqrt(3.0) * _ACOUSTIC_K  # omega = |k| (a free wave where c = 1)


def _acoustic_cspeed(Z):
    return 1.0 + 0.5 * (np.asarray(Z) > 0.5)


def _acoustic_3d(t, X, Y, Z):
    k = _ACOUSTIC_K
    return np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z) * np.cos(_ACOUSTIC_W * t)


def acoustic_source(X, Y, Z, t):
    """s = p_tt - c^2 Delta p = (c^2 |k|^2 - omega^2) p; |k|^2 = 3 k^2."""
    p = _acoustic_3d(t, X, Y, Z)
    c = _acoustic_cspeed(Z)
    return (c**2 * (3.0 * _ACOUSTIC_K**2) - _ACOUSTIC_W**2) * p


# --- 2-D incompressible Navier-Stokes: Taylor-Green vortex (exact) -----------
_TG_NU = 0.1


def _taylor_green_2d(t, X, Y):
    e = np.exp(-2.0 * _TG_NU * t)
    return {
        "u": np.sin(X) * np.cos(Y) * e,
        "v": -np.cos(X) * np.sin(Y) * e,
        # pressure sign fixed vs the HardNumerics prose so it is an *exact* NS
        # solution: +1/4 makes the convective term and grad p cancel.
        "p": 0.25 * (np.cos(2.0 * X) + np.cos(2.0 * Y)) * e**2,
    }


# --- 2-D resistive MHD, manufactured divergence-free fields (system, MMS) -----
_MHD_NU = 0.1
_MHD_ETA = 0.1


def _mhd_2d(t, X, Y):
    sx, sy = np.sin(np.pi * X), np.sin(np.pi * Y)
    cx, cy = np.cos(np.pi * X), np.cos(np.pi * Y)
    ct, st = np.cos(t), np.sin(t)
    return {
        "u": np.pi * sx**2 * np.sin(2 * np.pi * Y) * ct,
        "v": -np.pi * np.sin(2 * np.pi * X) * sy**2 * ct,
        "Bx": -np.pi * cx**2 * np.sin(2 * np.pi * Y) * st,
        "By": np.pi * np.sin(2 * np.pi * X) * cy**2 * st,
        "p": sx * sy * ct,
    }


def mhd_source(x, y, t):
    """(f_u1, f_u2, f_B1, f_B2): the MMS momentum + induction sources. Generated
    from the manufactured fields with SymPy and validated against the PDE residual
    (and this exact test triple) in validate_ground_truth.py."""
    pi = np.pi
    sx, sy = np.sin(pi * x), np.sin(pi * y)
    cx, cy = np.cos(pi * x), np.cos(pi * y)
    ct, st = np.cos(t), np.sin(t)
    c2x, c2y = np.cos(2 * pi * x), np.cos(2 * pi * y)
    fu1 = (1 / 5) * pi * (-20 * pi**2 * (-4 * cx**2 * cy**2 + cx**2 + cy**2) * st**2 * sx * cx * cy**2
                          - 10 * st * sx**2 * sy * cy + 40 * pi**2 * sx**3 * sy**2 * ct**2 * cx * cy**2
                          - 20 * pi**2 * sx**3 * sy**2 * ct**2 * cx * c2y + 4 * pi**2 * sx**2 * sy * ct * cy
                          + 5 * sy * ct * cx - 2 * pi**2 * sy * ct * c2x * cy)
    fu2 = (1 / 5) * pi * (-20 * pi**2 * (-4 * cx**2 * cy**2 + cx**2 + cy**2) * st**2 * sy * cx**2 * cy
                          + 10 * st * sx * sy**2 * cx + 40 * pi**2 * sx**2 * sy**3 * ct**2 * cx**2 * cy
                          - 20 * pi**2 * sx**2 * sy**3 * ct**2 * c2x * cy - 4 * pi**2 * sx * sy**2 * ct * cx
                          + 2 * pi**2 * sx * ct * cx * c2y + 5 * sx * ct * cy)
    fB1 = (2 / 5) * pi * (20 * pi**2 * st * sx**3 * sy**2 * ct * cx - 10 * pi**2 * st * sx**3 * ct * cx
                          + 4 * pi**2 * st * sx**2 * sy * cy - 40 * pi**2 * st * sx * sy**4 * ct * cx
                          + 30 * pi**2 * st * sx * sy**2 * ct * cx - 3 * pi**2 * st * sy * cy
                          + 5 * sx**2 * sy * ct * cy - 5 * sy * ct * cy)
    fB2 = (2 / 5) * pi * (-40 * pi**2 * st * sx**4 * sy * ct * cy + 20 * pi**2 * st * sx**2 * sy**3 * ct * cy
                          + 30 * pi**2 * st * sx**2 * sy * ct * cy - 4 * pi**2 * st * sx * sy**2 * cx
                          + 3 * pi**2 * st * sx * cx - 10 * pi**2 * st * sy**3 * ct * cy
                          - 5 * sx * sy**2 * ct * cx + 5 * sx * ct * cx)
    return fu1, fu2, fB1, fB2


# --- 2-D Keller-Segel chemotaxis, manufactured positive fields (system, MMS) --
_KS = dict(Dr=1.0, Dc=1.0, chi=1.0, alpha=1.0, beta=1.0)


def _keller_segel_2d(t, X, Y):
    return {
        "rho": 1.0 + 0.2 * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-t),
        "c": 1.0 + 0.1 * np.cos(np.pi * X) * np.cos(np.pi * Y) * np.exp(-2.0 * t),
    }


def keller_segel_source(x, y, t):
    """(s_rho, s_c) MMS sources for the chemotaxis system (D_r=D_c=chi=alpha=beta=1)."""
    pi = np.pi
    ss = np.sin(pi * x) * np.sin(pi * y)
    cc = np.cos(pi * x) * np.cos(pi * y)
    rho = 1.0 + 0.2 * ss * np.exp(-t)
    grad_dot = -0.04 * pi**2 * np.sin(pi * x) * np.cos(pi * x) * np.sin(pi * y) * np.cos(pi * y) * np.exp(-3 * t)
    rho_lapc = rho * (-0.2 * pi**2 * cc * np.exp(-2 * t))
    s_rho = (-0.2 * ss * np.exp(-t)) - (-0.4 * pi**2 * ss * np.exp(-t)) + (grad_dot + rho_lapc)
    s_c = (0.2 * pi**2 - 0.1) * cc * np.exp(-2 * t) - 0.2 * ss * np.exp(-t)
    return s_rho, s_c


# --- 2-D nearly-incompressible linear elasticity (system, MMS, locking) -------
_ELAST_MU = 1.0
_ELAST_LAMBDA = 1.0e4      # lambda/mu = 1e4 -> volumetric locking stress test


def _elasticity_2d(_t, X, Y):
    # The manufactured displacement is divergence-free (div u == 0), so the
    # lambda*(div u) term vanishes for the exact field: a locking-free scheme must
    # reproduce that while a P1 displacement scheme locks as lambda/mu -> 1e4.
    return {
        "u": np.sin(np.pi * X)**2 * np.sin(2.0 * np.pi * Y),
        "v": -np.sin(2.0 * np.pi * X) * np.sin(np.pi * Y)**2,
    }


def elasticity_source(x, y):
    """f = -div(sigma) = -mu*Delta u (lambda drops out since div u == 0 exactly)."""
    pi = np.pi
    lap_u = 2 * pi**2 * np.cos(2 * pi * x) * np.sin(2 * pi * y) \
        - 4 * pi**2 * np.sin(pi * x)**2 * np.sin(2 * pi * y)
    lap_v = 4 * pi**2 * np.sin(2 * pi * x) * np.sin(pi * y)**2 \
        - 2 * pi**2 * np.sin(2 * pi * x) * np.cos(2 * pi * y)
    return -_ELAST_MU * lap_u, -_ELAST_MU * lap_v


# --- 3-D Maxwell periodic plane wave (system, exact) -------------------------
_MAXWELL_K = np.array([1.0, 1.0, 1.0])
_MAXWELL_KMAG = np.sqrt(3.0)
_MAXWELL_A = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)          # polarization, a.k=0
_MAXWELL_B = np.cross(_MAXWELL_K, _MAXWELL_A) / _MAXWELL_KMAG   # (1,1,-2)/sqrt(6)


def _maxwell_3d(t, X, Y, Z):
    phase = _MAXWELL_K[0] * X + _MAXWELL_K[1] * Y + _MAXWELL_K[2] * Z - _MAXWELL_KMAG * t
    s = np.sin(phase)
    a, b = _MAXWELL_A, _MAXWELL_B
    return {
        "Ex": a[0] * s, "Ey": a[1] * s, "Ez": a[2] * s,
        "Hx": b[0] * s, "Hy": b[1] * s, "Hz": b[2] * s,
    }


# --- reusable structure / secondary diagnostics (authored here, not solver-reported)
# Each is called ``fn(num, exact, axes, mask) -> (value, passed)`` where ``num`` and
# ``exact`` are {field_name: ndarray} dicts and ``axes`` the 1-D coordinate arrays.
def _diag_min_value(field="u", floor=-1e-6):
    """min field value over the (masked) domain -- a positivity / nonnegativity check."""
    def fn(num, exact, axes, mask):
        u = num[field]
        vals = u if mask is None else u[mask]
        mn = float(np.min(vals))
        return mn, bool(mn >= floor)
    return fn


def _diag_positivity(fields, floor=0.0, tol=1e-6):
    """min value across several fields -- the Keller-Segel positivity gate."""
    def fn(num, exact, axes, mask):
        mn = min(float(np.min(num[f] if mask is None else num[f][mask])) for f in fields)
        return mn, bool(mn >= floor - tol)
    return fn


def _interior(a, w=2):
    """Strip ``w`` boundary layers along every axis (finite-difference gradients are
    one-sided and unreliable at the edges, and wrong at a periodic wrap)."""
    if a.ndim == 0:
        return a
    sl = tuple(slice(w, -w) if s > 2 * w + 1 else slice(None) for s in a.shape)
    return a[sl]


def _divergence(num, comps, axes):
    """Discrete divergence sum_i d(comp_i)/d(axis_i) of a vector field."""
    div = np.zeros_like(num[comps[0]])
    for i, cn in enumerate(comps):
        div = div + np.gradient(num[cn], axes[i], axis=i, edge_order=2)
    return div


def _diag_divergence_free(comps, tol=0.05):
    """Relative divergence ||div F|| / ||grad F|| over the interior -- the
    structure-preservation gate for incompressible velocity / solenoidal B, E, H.
    Dimensionless and ~O(h^2) for a genuinely divergence-free field, ~O(1) when the
    solver injects spurious divergence."""
    def fn(num, exact, axes, mask):
        div = _divergence(num, comps, axes)
        grad_sq = np.zeros_like(div)
        for cn in comps:
            for i in range(len(axes)):
                grad_sq = grad_sq + np.gradient(num[cn], axes[i], axis=i, edge_order=2) ** 2
        d_int, g_int = _interior(div), _interior(grad_sq)
        num_norm = float(np.sqrt(np.mean(d_int ** 2)))
        den = float(np.sqrt(np.mean(g_int))) + 1e-14
        rel = num_norm / den
        return rel, bool(rel < tol)
    return fn


def _diag_div_error(comps, tol=None):
    """Relative error of the numerical divergence against the exact divergence --
    the volumetric-locking indicator for near-incompressible elasticity (a locked
    scheme drives div u to ~0, far from the true div u)."""
    def fn(num, exact, axes, mask):
        dn = _interior(_divergence(num, comps, axes))
        de = _interior(_divergence(exact, comps, axes))
        err = float(np.sqrt(np.mean((dn - de) ** 2)))
        ref = float(np.sqrt(np.mean(de ** 2))) + 1e-14
        rel = err / ref
        return rel, (True if tol is None else bool(rel < tol))
    return fn


def _diag_energy_error(comps, tol=None):
    """Relative error in the field 'energy' 0.5 sum |comp|^2 (kinetic / magnetic /
    field energy), integrated over the interior -- a reported conservation drift."""
    def fn(num, exact, axes, mask):
        en_n = _interior(sum(num[c] ** 2 for c in comps))
        en_e = _interior(sum(exact[c] ** 2 for c in comps))
        err = abs(float(np.mean(en_n)) - float(np.mean(en_e)))
        ref = abs(float(np.mean(en_e))) + 1e-14
        rel = err / ref
        return rel, (True if tol is None else bool(rel < tol))
    return fn


def _diag_convexity_min_eig(field="u", floor=-1e-2):
    """min eigenvalue of the discrete Hessian over the interior -- a convexity check."""
    def fn(num, exact, axes, mask):
        u = num[field]
        x, y = axes[0], axes[1]
        ux = np.gradient(u, x, axis=0)
        uy = np.gradient(u, y, axis=1)
        uxx = np.gradient(ux, x, axis=0)
        uyy = np.gradient(uy, y, axis=1)
        uxy = np.gradient(ux, y, axis=1)
        tr = uxx + uyy
        disc = np.maximum(tr**2 - 4.0 * (uxx * uyy - uxy**2), 0.0)
        lam = 0.5 * (tr - np.sqrt(disc))
        core = lam[2:-2, 2:-2] if lam.ndim == 2 and min(lam.shape) > 4 else lam
        mn = float(np.min(core))
        return mn, bool(mn >= floor)
    return fn


# =============================================================================
# Problem set
# =============================================================================

PROBLEMS = [
    # -------------------------------------------------------------------------
    # PDE -- Tier 1 (textbook; covered by pde_manual.md)
    # -------------------------------------------------------------------------
    {
        "id": "P01",
        "slug": "pde_heat_1d",
        "type": "pde",
        "tier": 1,
        "family": "heat",
        "title": "1D Heat Equation",
        "challenge": "Textbook parabolic IVBP; explicit FTCS and implicit Crank-Nicolson both apply.",
        "dims": 1,
        "domain": {"x": [0.0, 1.0]},
        "t_eval": 0.5,
        "has_ground_truth": True,
        "analytic": partial(_heat_1d, alpha=0.1),
        "description": """# 1D Heat Equation

Solve the one-dimensional heat equation on [0, 1]:

    u_t = 0.1 * u_xx,   x in [0, 1],   t in [0, 0.5]

**Initial condition**:  u(x, 0) = sin(pi x)

**Boundary conditions** (homogeneous Dirichlet):  u(0, t) = 0  and  u(1, t) = 0

Report the numerical solution at t = 0.5. It will be scored against a hidden
reference solution (relative L2 error, plus an observed spatial-convergence-order
check across two resolutions).
""",
    },
    {
        "id": "P02",
        "slug": "pde_heat_2d",
        "type": "pde",
        "tier": 1,
        "family": "heat",
        "title": "2D Heat Equation",
        "challenge": "Separable 2D diffusion; tests 2D grid assembly and CFL in two dimensions.",
        "dims": 2,
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "t_eval": 0.1,
        "has_ground_truth": True,
        "analytic": partial(_heat_2d, alpha=1.0),
        "description": """# 2D Heat Equation

Solve the two-dimensional heat equation on the unit square:

    u_t = alpha * (u_xx + u_yy),   (x, y) in (0, 1) x (0, 1),   t in [0, 0.1]

**Parameters**:  alpha = 1

**Initial condition**:  u(x, y, 0) = sin(pi x) sin(pi y)

**Boundary conditions** (homogeneous Dirichlet):  u = 0 on all four edges.

Report the numerical solution at t = 0.1. It will be scored against a hidden
reference solution (relative L2 error, plus an observed spatial-convergence-order
check across two resolutions).
""",
    },
    {
        "id": "P03",
        "slug": "pde_wave_1d",
        "type": "pde",
        "tier": 1,
        "family": "wave",
        "title": "1D Wave Equation",
        "challenge": "Second-order-in-time hyperbolic PDE; needs a stable leap-frog / Newmark step.",
        "dims": 1,
        "domain": {"x": [0.0, 1.0]},
        "t_eval": 1.0,
        "has_ground_truth": True,
        "analytic": partial(_wave_1d, c=1.0),
        "description": """# 1D Wave Equation

Solve the one-dimensional wave equation on [0, 1]:

    u_tt = c^2 * u_xx,   x in [0, 1],   t in [0, 1]

**Parameters**:  c = 1

**Initial conditions**:  u(x, 0) = sin(pi x),   u_t(x, 0) = 0

**Boundary conditions** (fixed ends):  u(0, t) = 0  and  u(1, t) = 0

Report the numerical solution at t = 1. It will be scored against a hidden
reference solution (relative L2 error, plus an observed spatial-convergence-order
check across two resolutions).
""",
    },
    {
        "id": "P04",
        "slug": "pde_advection_1d",
        "type": "pde",
        "tier": 1,
        "family": "advection",
        "title": "1D Linear Advection (periodic)",
        "challenge": "Transports a narrow Gaussian; first-order upwind smears it, so it stresses numerical diffusion.",
        "dims": 1,
        "domain": {"x": [0.0, 1.0]},
        "t_eval": 0.5,
        "has_ground_truth": True,
        "analytic": partial(_advection_1d, a=1.0),
        "description": """# 1D Linear Advection (periodic)

Solve the periodic transport equation on [0, 1]:

    u_t + a * u_x = 0,   x in [0, 1],   t in [0, 0.5]

**Parameters**:  a = 1

**Initial condition**:  u(x, 0) = exp(-100 * (x - 0.3)^2)

**Boundary conditions**:  periodic, u(0, t) = u(1, t)

Report the numerical solution at t = 0.5. It will be scored against a hidden
reference solution (relative L2 error, plus an observed spatial-convergence-order
check across two resolutions). A low-order upwind scheme smears the pulse, so a
higher-order, low-dissipation method is needed to converge.
""",
    },
    {
        "id": "P05",
        "slug": "pde_poisson_2d",
        "type": "pde",
        "tier": 1,
        "family": "poisson",
        "title": "2D Poisson Equation (manufactured)",
        "challenge": "Steady elliptic solve; 2D sparse assembly and a direct/iterative linear solve.",
        "dims": 2,
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "t_eval": None,
        "has_ground_truth": True,
        "analytic": _poisson_2d,
        "description": """# 2D Poisson Equation (manufactured solution)

Solve the Poisson equation on the unit square:

    -(u_xx + u_yy) = f(x, y),   (x, y) in (0, 1) x (0, 1)

with source term f(x, y) = 2 * pi^2 * sin(pi x) sin(pi y).

**Boundary conditions** (homogeneous Dirichlet):  u = 0 on the entire boundary.

This is a steady problem (no time dependence). Report the numerical solution; it
will be scored against a hidden reference solution (relative L2 error, plus an
observed spatial-convergence-order check across two resolutions).
""",
    },
    # -------------------------------------------------------------------------
    # PDE -- Tier 2 (moderate; partly beyond the manual)
    # -------------------------------------------------------------------------
    {
        "id": "P06",
        "slug": "pde_laplace_2d",
        "type": "pde",
        "tier": 2,
        "family": "laplace",
        "title": "2D Laplace Equation (mixed Dirichlet)",
        "challenge": "Non-separable-looking BC data; solution grows like sinh in y, so it is not symmetric in x/y.",
        "dims": 2,
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "t_eval": None,
        "has_ground_truth": True,
        "analytic": _laplace_2d,
        "description": """# 2D Laplace Equation

Solve Laplace's equation on the unit square:

    u_xx + u_yy = 0,   (x, y) in (0, 1) x (0, 1)

**Boundary conditions** (Dirichlet):
    u(x, 0) = 0,   u(x, 1) = sin(pi x),   u(0, y) = 0,   u(1, y) = 0

This is a steady problem. Report the numerical solution; it will be scored against
a hidden reference solution (relative L2 error, plus an observed
spatial-convergence-order check across two resolutions).
""",
    },
    {
        "id": "P07",
        "slug": "pde_convection_diffusion_bl",
        "type": "pde",
        "tier": 2,
        "family": "convection-diffusion",
        "title": "Singularly Perturbed Convection-Diffusion",
        "challenge": "eps=1e-3 boundary layer near x=1; centered differences oscillate, so it needs upwinding or a fine/graded mesh.",
        "dims": 1,
        "domain": {"x": [0.0, 1.0]},
        "t_eval": None,
        "has_ground_truth": True,
        "analytic": partial(_convection_diffusion_bl, eps=1e-3),
        "description": """# Singularly Perturbed Convection-Diffusion (boundary-layer)

Solve the steady singularly-perturbed boundary-value problem on [0, 1]:

    -eps * u''(x) + u'(x) = 1,   0 < x < 1

**Parameters**:  eps = 1e-3

**Boundary conditions** (Dirichlet):  u(0) = 0  and  u(1) = 0

This is a steady problem with a thin boundary layer near x = 1. A stable scheme
must resolve or upwind the boundary layer. Report the numerical solution; it will
be scored against a hidden reference solution (relative L2 error, plus an observed
spatial-convergence-order check across two resolutions).
""",
    },
    {
        "id": "P08",
        "slug": "pde_helmholtz_2d",
        "type": "pde",
        "tier": 2,
        "family": "helmholtz",
        "title": "2D Helmholtz Equation (indefinite)",
        "challenge": "k=10 makes the discrete operator indefinite; naive iterative solvers stall.",
        "dims": 2,
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "t_eval": None,
        "has_ground_truth": True,
        "analytic": _helmholtz_2d,
        "description": """# 2D Helmholtz Equation (manufactured solution)

Solve the Helmholtz equation on the unit square:

    -(u_xx + u_yy) - k^2 * u = f(x, y),   (x, y) in (0, 1) x (0, 1)

**Parameters**:  k = 10

with source term f(x, y) = (2 * pi^2 - k^2) * sin(pi x) sin(pi y).

**Boundary conditions** (homogeneous Dirichlet):  u = 0 on the entire boundary.

This is a steady problem. Note that the operator is indefinite because
k^2 = 100 > 2 pi^2. Report the numerical solution; it will be scored against a
hidden reference solution (relative L2 error, plus an observed
spatial-convergence-order check across two resolutions).
""",
    },
    {
        "id": "P09",
        "slug": "pde_anisotropic_diffusion",
        "type": "pde",
        "tier": 2,
        "family": "anisotropic-diffusion",
        "title": "Anisotropic Diffusion (manufactured)",
        "challenge": "100:1 diffusion-tensor anisotropy stresses conditioning and mesh aspect ratio.",
        "dims": 2,
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "t_eval": None,
        "has_ground_truth": True,
        "analytic": _anisotropic_2d,
        "description": """# Anisotropic Diffusion (manufactured solution)

Solve the anisotropic steady diffusion equation on the unit square:

    -(A_xx * u_xx + A_yy * u_yy) = f(x, y),   (x, y) in (0, 1) x (0, 1)

with diffusion tensor A = diag(1, 100) and source term
f(x, y) = 101 * pi^2 * sin(pi x) sin(pi y).

**Boundary conditions** (homogeneous Dirichlet):  u = 0 on the entire boundary.

This is a steady problem. Report the numerical solution; it will be scored against
a hidden reference solution (relative L2 error, plus an observed
spatial-convergence-order check across two resolutions).
""",
    },
    {
        "id": "P10",
        "slug": "pde_wave_2d",
        "type": "pde",
        "tier": 2,
        "family": "wave",
        "title": "2D Wave Equation",
        "challenge": "2D hyperbolic with a sqrt(2) modal frequency; CFL couples both spatial directions.",
        "dims": 2,
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "t_eval": 0.5,
        "has_ground_truth": True,
        "analytic": partial(_wave_2d, c=1.0),
        "description": """# 2D Wave Equation

Solve the two-dimensional wave equation on the unit square:

    u_tt = c^2 * (u_xx + u_yy),   (x, y) in (0, 1) x (0, 1),   t in [0, 0.5]

**Parameters**:  c = 1

**Initial conditions**:  u(x, y, 0) = sin(pi x) sin(pi y),   u_t(x, y, 0) = 0

**Boundary conditions** (fixed):  u = 0 on the entire boundary.

Report the numerical solution at t = 0.5. It will be scored against a hidden
reference solution (relative L2 error, plus an observed spatial-convergence-order
check across two resolutions).
""",
    },
    # -------------------------------------------------------------------------
    # PDE -- Tier 3 (hard; beyond the manual)
    # -------------------------------------------------------------------------
    {
        "id": "P11",
        "slug": "pde_burgers_inviscid",
        "type": "pde",
        "tier": 3,
        "family": "burgers-inviscid",
        "title": "Inviscid Burgers (Riemann shock)",
        "challenge": "Entropy shock; a conservative/Godunov scheme captures the speed but smears the front, so <1% L2 is very hard.",
        "dims": 1,
        "domain": {"x": [-1.0, 1.0]},
        "t_eval": 1.0,
        "has_ground_truth": True,
        # A shock makes the L2 convergence order fractional (~1/2), so the 2-grid
        # order gate is disabled here; scored by single-grid relative L2 only.
        "pde_order_check": False,
        "analytic": _burgers_inviscid,
        "description": """# Inviscid Burgers Equation (Riemann shock)

Solve the inviscid Burgers equation in conservation form on [-1, 1]:

    u_t + (u^2 / 2)_x = 0,   x in [-1, 1],   t in [0, 1]

**Initial condition** (Riemann data):  u(x, 0) = 1 for x < 0,  u(x, 0) = 0 for x >= 0

**Boundary conditions**:  inflow u(-1, t) = 1 at the left, outflow at the right.

Use a conservative shock-capturing scheme that selects the entropy solution.
Report the numerical solution at t = 1; it will be scored against a hidden
reference (entropy) solution by relative L2 error.
""",
    },
    {
        "id": "P12",
        "slug": "pde_fokker_planck_ou",
        "type": "pde",
        "tier": 3,
        "family": "fokker-planck",
        "title": "Fokker-Planck (Ornstein-Uhlenbeck)",
        "challenge": "Drift-diffusion in conservation form; must conserve mass and track a moving, narrowing Gaussian.",
        "dims": 1,
        "domain": {"x": [-8.0, 8.0]},
        "t_eval": 1.0,
        "has_ground_truth": True,
        "analytic": partial(_fokker_planck_ou, mu0=1.0, var0=0.25),
        "description": """# Fokker-Planck Equation (Ornstein-Uhlenbeck)

Solve the Fokker-Planck equation for the OU process on [-8, 8]:

    p_t = p_xx + (x p)_x,   x in [-8, 8],   t in [0, 1]

**Initial condition**:  p(x, 0) = Gaussian with mean mu0 = 1 and variance var0 = 0.25,
i.e. p(x, 0) = (2 pi * 0.25)^(-1/2) exp(-(x - 1)^2 / (2 * 0.25)).

**Boundary conditions**:  p -> 0 at x = -8 and x = 8 (Dirichlet, effectively zero).

Use a conservative (mass-preserving) discretization of the drift-diffusion
operator. Report the numerical solution at t = 1; it will be scored against a
hidden reference solution (relative L2 error, plus an observed
spatial-convergence-order check across two resolutions).
""",
    },
    {
        "id": "P13",
        "slug": "pde_fractional_diffusion",
        "type": "pde",
        "tier": 3,
        "family": "fractional-diffusion",
        "title": "Time-Fractional Subdiffusion",
        "challenge": "Caputo derivative of order 1/2 needs a history-aware (L1) scheme and the Mittag-Leffler solution; far outside the manual.",
        "dims": 1,
        "domain": {"x": [0.0, 1.0]},
        "t_eval": 0.5,
        "has_ground_truth": True,
        "analytic": partial(_fractional_diffusion, D=1.0, alpha=0.5),
        "description": """# Time-Fractional Subdiffusion Equation

Solve the time-fractional diffusion equation on [0, 1]:

    D_t^alpha u = D * u_xx,   x in [0, 1],   t in [0, 0.5]

where D_t^alpha is the Caputo fractional time derivative of order alpha.

**Parameters**:  alpha = 0.5,  D = 1

**Initial condition**:  u(x, 0) = sin(pi x)

**Boundary conditions** (homogeneous Dirichlet):  u(0, t) = 0  and  u(1, t) = 0

The Caputo derivative is non-local in time, so a correct scheme must be
history-aware (e.g. an L1 discretization of the fractional time derivative).
Report the numerical solution at t = 0.5; it will be scored against a hidden
reference solution by relative L2 error.
""",
    },
    {
        "id": "P14",
        "slug": "pde_black_scholes_call",
        "type": "pde",
        "tier": 3,
        "family": "black-scholes",
        "title": "Black-Scholes European Call",
        "challenge": "Backward-in-time terminal-value problem with a non-smooth payoff kink; convection-dominated near S=Smax.",
        "dims": 1,
        "domain": {"S": [0.0, 4.0]},
        "t_eval": 0.0,
        "has_ground_truth": True,
        "analytic": partial(_black_scholes_call, K=1.0, r=0.05, sigma=0.2, T=1.0),
        "description": """# Black-Scholes European Call Option

Solve the Black-Scholes PDE for a European call on 0 < S < 4, 0 < t < 1:

    V_t + 0.5 * sigma^2 * S^2 * V_SS + r * S * V_S - r * V = 0

**Parameters**:  sigma = 0.2,  r = 0.05,  strike K = 1,  maturity T = 1,  S_max = 4

**Terminal condition** (payoff at maturity):  V(S, T) = max(S - K, 0)

**Boundary conditions**:  V(0, t) = 0  and  V(4, t) = 4 - K exp(-r (T - t)).

This is a backward-in-time problem; integrate from t = T back to t = 0. Report the
present value V(S, 0) across S; it will be scored against a hidden reference price
by relative L2 error.
""",
    },
    {
        "id": "P15",
        "slug": "pde_kuramoto_sivashinsky",
        "type": "pde",
        "tier": 3,
        "family": "kuramoto-sivashinsky",
        "title": "Kuramoto-Sivashinsky (chaotic)",
        "challenge": "Fourth-order, chaotic, no closed form; stiff and requires an ETDRK/IMEX spectral integrator. No analytic ground truth.",
        "dims": 1,
        "domain": {"x": [0.0, 100.53096491487338]},  # 32*pi
        "t_eval": 50.0,
        "has_ground_truth": False,
        "analytic": None,
        "description": """# Kuramoto-Sivashinsky Equation (chaotic)

Solve the Kuramoto-Sivashinsky equation on the periodic interval [0, 32 pi]:

    u_t + u u_x + u_xx + u_xxxx = 0,   x in [0, 32 pi],   t in [0, 50]

**Initial condition**:  u(x, 0) = cos(x / 16) * (1 + sin(x / 16))

**Boundary conditions**:  periodic.

This equation is chaotic and has no closed-form solution. The stabilizing fourth-order
term and destabilizing second-order term make it stiff; an exponential-integrator or
IMEX spectral method is appropriate. Report the numerical solution at t = 50.
""",
    },
    # -------------------------------------------------------------------------
    # PDE -- Tier 3 HardNumerics imports, Batch 1 (scalar; hard, beyond the manual)
    # -------------------------------------------------------------------------
    {
        "id": "P16",
        "slug": "pde_stefan_1d_similarity",
        "type": "pde",
        "tier": 3,
        "discriminator": True,
        "family": "stefan-free-boundary",
        "title": "One-Phase Stefan Problem (moving front)",
        "challenge": "Moving free boundary (phase change); the front's kink limits the L2 rate and a naive fixed-grid scheme mislocates it.",
        "dims": 1,
        "domain": {"x": [0.0, 2.0]},
        "t_eval": 1.0,
        "has_ground_truth": True,
        "analytic": _stefan_1d,
        "min_order": 0.75,
        "description": """# One-Phase Stefan Problem (melting with a moving boundary)

Solve the one-phase melting problem for the temperature u(x, t) on the half-line
x >= 0, over t in [0.05, 1.0]:

    u_t = u_xx,          0 < x < s(t)   (liquid; ahead of the front u = 0)
    u(0, t) = 1,
    u(s(t), t) = 0,
    s'(t) = -u_x(s(t), t),              (Stefan condition; front velocity)

with a moving phase boundary x = s(t) separating the melt (u > 0) from the solid
(u = 0). Take the initial state from the exact similarity profile at t0 = 0.05
(front s(t0) = 2 * lambda * sqrt(t0), where lambda solves
lambda * exp(lambda^2) * erf(lambda) = 1/sqrt(pi)); the temperature is 0 for
x >= s(t).

Report the temperature field u(x, 1) on [0, 2] (set solid-region values to 0). It
will be scored against a hidden reference solution (relative L2 error, plus an
observed spatial-convergence-order check across two resolutions). The kink at the
front makes the convergence order fractional, so a scheme that resolves the moving
interface is needed.
""",
    },
    {
        "id": "P17",
        "slug": "pde_monge_ampere_2d",
        "type": "pde",
        "tier": 3,
        "discriminator": True,
        "family": "monge-ampere",
        "title": "Monge-Ampere (fully nonlinear, convex)",
        "challenge": "Fully nonlinear elliptic det(D^2 u)=f; needs a convexity-preserving (monotone wide-stencil) scheme, not a linear solve.",
        "dims": 2,
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "t_eval": None,
        "has_ground_truth": True,
        "analytic": _monge_ampere_2d,
        "min_order": 0.75,
        "diagnostics": [
            {"name": "min_hessian_eig", "fn": _diag_convexity_min_eig(), "gate": False},
        ],
        "description": """# Monge-Ampere Equation (fully nonlinear, convex solution)

Solve the fully nonlinear elliptic Monge-Ampere equation on the unit square for a
convex function u(x, y):

    det(D^2 u) = f(x, y),   (x, y) in (0, 1) x (0, 1),   u convex,

where D^2 u is the Hessian and the right-hand side is

    f(x, y) = exp(x^2 + y^2) * (1 + x^2 + y^2).

**Boundary conditions** (Dirichlet): u = g(x, y) on the boundary, with the boundary
data g(x, y) = exp((x^2 + y^2) / 2).

This is a steady problem. The equation is degenerate-elliptic and admits multiple
solutions unless the convexity constraint is enforced, so a monotone / wide-stencil
scheme that selects the convex (viscosity) solution is required. Report the
numerical solution; it is scored against a hidden reference solution (relative L2
error, plus an observed spatial-convergence-order check across two resolutions).
""",
    },
    {
        "id": "P18",
        "slug": "pde_porous_medium_2d",
        "type": "pde",
        "tier": 3,
        "discriminator": True,
        "family": "porous-medium",
        "title": "Porous-Medium (degenerate, compact support)",
        "challenge": "Degenerate diffusion u_t=Delta(u^2) with a compact-support free boundary; scored in the L1 norm because the moving edge pollutes the L2 rate.",
        "dims": 2,
        "domain": {"x": [-4.0, 4.0], "y": [-4.0, 4.0]},
        "t_eval": 1.0,
        "has_ground_truth": True,
        "analytic": _barenblatt_2d,
        "pde_metric": "l1",
        "min_order": 0.6,
        "pde_l2_tol": 0.02,
        "diagnostics": [
            {"name": "min_value", "fn": _diag_min_value(floor=-1e-3), "gate": False},
        ],
        "description": """# Porous-Medium Equation (degenerate diffusion, Barenblatt profile)

Solve the degenerate porous-medium equation on the square [-4, 4]^2 (large enough
that the solution's support stays away from the boundary), over t in [0.25, 1.0]:

    u_t = Delta(u^2),   (the diffusivity 2u vanishes where u -> 0)

**Initial condition** (a compact-support paraboloid cap at t0 = 0.25):

    u(x, y, 0.25) = max(1 - (x^2 + y^2) / 4,  0).

**Boundary conditions**:  u = 0 on the domain boundary (the support never reaches it).

The diffusion degenerates at the edge of the support, producing a moving free
boundary where u is only Lipschitz, so a conservative scheme that preserves
nonnegativity and compact support is needed. Report the numerical solution u(x, y, 1);
it is scored against a hidden reference solution in the **relative L1 norm** (which
is the appropriate metric for a compact-support solution), plus an observed
convergence-order check across two resolutions.
""",
    },
    {
        "id": "P19",
        "slug": "pde_poisson_lshape",
        "type": "pde",
        "tier": 3,
        "discriminator": True,
        "family": "poisson-reentrant-corner",
        "title": "Poisson on an L-shaped domain (corner singularity)",
        "challenge": "Re-entrant corner: u ~ r^{2/3} is not in H^2, so uniform low-order methods lose their rate; the order gate is the whole point.",
        "dims": 2,
        "domain": {"x": [-1.0, 1.0], "y": [-1.0, 1.0]},
        "t_eval": None,
        "has_ground_truth": True,
        "analytic": _reentrant_corner_2d,
        "domain_mask": _lshape_mask,
        "min_order": 1.25,
        "axes": ["x", "y"],
        "description": """# Poisson (Laplace) on an L-shaped Domain (re-entrant corner)

Solve Laplace's equation on the L-shaped domain
Omega = (-1, 1)^2 \\ [0, 1] x [-1, 0] (the unit square with its fourth quadrant
removed, leaving a 270-degree re-entrant corner at the origin):

    -Delta u = 0   in Omega.

**Boundary conditions** (Dirichlet): on the outer boundary of Omega, u is given by

    g(r, theta) = r^(2/3) * sin(2*theta/3),

where (r, theta) are polar coordinates about the origin with theta in (0, 3*pi/2)
measured counter-clockwise from the positive x-axis; on the two faces of the slit
(theta = 0 and theta = 3*pi/2) the data is 0.

This is a steady problem. Return the solution on a rectangular grid covering
[-1, 1]^2; **only the in-domain nodes are scored** (values in the removed quadrant
are ignored). The corner makes u fall short of H^2, so a uniform low-order scheme
converges below second order -- scoring is the relative L2 error over the domain
plus an observed convergence-order check (a graded / adaptive mesh recovers the
rate that a naive uniform mesh loses).
""",
    },
    {
        "id": "P20",
        "slug": "pde_cahn_hilliard_2d",
        "type": "pde",
        "tier": 3,
        "discriminator": True,
        "family": "cahn-hilliard",
        "title": "Cahn-Hilliard (4th-order, stiff, MMS)",
        "challenge": "Fourth-order stiff phase-field operator; explicit stepping needs dt~h^4, so an energy-stable / IMEX scheme is required.",
        "dims": 2,
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "t_eval": 0.5,
        "has_ground_truth": True,
        "analytic": _cahn_hilliard_2d,
        "min_order": 1.4,
        "description": """# Cahn-Hilliard Equation (manufactured solution)

Solve the Cahn-Hilliard equation on the periodic unit square [0, 1]^2, over
t in [0, 0.5]:

    u_t = Delta(mu) + s(x, y, t),
    mu  = -eps^2 * Delta u + u^3 - u,

**Parameters**:  eps = 0.1.

**Initial condition**:  u(x, y, 0) = 0.2 * sin(2*pi*x) * sin(2*pi*y).

**Boundary conditions**:  periodic in both directions.

The manufactured source term s(x, y, t) (which makes a known smooth field the exact
solution) is, with p = sin(2*pi*x) * sin(2*pi*y):

    s = -0.2 * p * sin(t)
        + 1.6*pi^2 * (8*pi^2*eps^2 - 1) * p * cos(t)
        + 0.192*pi^2 * cos(t)^3 * p * ( p^2
              - cos(2*pi*x)^2 * sin(2*pi*y)^2 - sin(2*pi*x)^2 * cos(2*pi*y)^2 ).

The fourth-order operator is stiff (a fully explicit scheme needs dt ~ h^4), so an
energy-stable or IMEX/spectral integrator is appropriate. Report the numerical
solution u(x, y, 0.5); it is scored against a hidden reference solution (relative L2
error, plus an observed spatial-convergence-order check across two resolutions).
""",
    },
    {
        "id": "P21",
        "slug": "pde_heston_2d",
        "type": "pde",
        "tier": 3,
        "discriminator": True,
        "family": "heston",
        "title": "Heston European Call (2D, degenerate)",
        "challenge": "2D convection-diffusion with a cross-derivative and a non-smooth payoff kink; degenerate as v->0. Reference is a semi-closed characteristic-function integral.",
        "dims": 2,
        "domain": {"S": [0.0, 4.0], "v": [0.0, 1.0]},
        "t_eval": 0.0,
        "has_ground_truth": True,
        "analytic": _heston_call,
        "axes": ["S", "v"],
        "min_order": 1.0,
        "diagnostics": [
            {"name": "min_price", "fn": _diag_min_value(floor=-1e-4), "gate": False},
        ],
        "description": """# Heston Stochastic-Volatility European Call

Price a European call V(S, v) under the Heston model on 0 <= S <= 4, 0 <= v <= 1,
with time to maturity tau running from 0 (payoff) to T = 1:

    V_tau = 0.5*v*S^2*V_SS + rho*sigma*v*S*V_Sv + 0.5*sigma^2*v*V_vv
            + r*S*V_S + kappa*(theta - v)*V_v - r*V.

**Parameters**:  kappa = 2.0,  theta = 0.04,  sigma = 0.3,  rho = -0.7,  r = 0.03,
strike K = 1,  maturity T = 1.

**Payoff (initial condition at tau = 0)**:  V(S, v, 0) = max(S - K, 0).

**Boundary behaviour**:  V(0, v) = 0; at S = 4 the call is deep in the money
(V ~ S - K*exp(-r*tau)); the v = 0 boundary is degenerate (the diffusion in v
vanishes) and the equation there reduces to the convection-reaction part.

Integrate to tau = T and report the price surface V(S, v) on a tensor grid in
(S, v). The mixed second derivative V_Sv and the non-smooth payoff kink at S = K
make this hard. Scoring is against a hidden reference price computed from Heston's
semi-closed characteristic-function formula (deterministic quadrature): relative L2
error, plus an observed spatial-convergence-order check across two resolutions.
""",
    },
    # -------------------------------------------------------------------------
    # PDE -- Tier 3 HardNumerics imports, Batch 2 (systems / 3-D)
    # Phase 1: higher-d scalar (Fichera, acoustic); Phase 2: 2-D systems
    # (Navier-Stokes, MHD, Keller-Segel, elasticity); Phase 3: 3-D system (Maxwell).
    # -------------------------------------------------------------------------
    {
        "id": "P22",
        "slug": "pde_fichera_3d",
        "type": "pde",
        "tier": 3,
        "discriminator": True,
        "family": "fichera-corner",
        "title": "Fichera 3D Corner Singularity (MMS)",
        "challenge": "3-D re-entrant vertex (cube minus an octant); the regularized singular solution has a sharp near-corner layer, and only in-domain nodes are scored.",
        "dims": 3,
        "domain": {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [-1.0, 1.0]},
        "t_eval": None,
        "has_ground_truth": True,
        "analytic": _fichera_3d,
        "domain_mask": _fichera_mask,
        "axes": ["x", "y", "z"],
        "grid_N": 24,
        "min_order": 1.2,
        "description": """# Fichera Corner Problem (3-D singularity, manufactured solution)

Solve the Poisson equation on the 3-D Fichera domain
Omega = (-1, 1)^3 \\ [0, 1]^3 (the cube with one octant removed, leaving a 3-D
re-entrant vertex at the origin):

    -Delta u = f(x, y, z)   in Omega,   u = u_exact on the boundary.

With r^2 = x^2 + y^2 + z^2 and a regularization eps = 1e-4, the manufactured source
is (from f = -Delta of the hidden singular solution (r^2 + eps^2)^(1/4)):

    f(x, y, z) = -0.5 * (3*eps^2 + 1.5*r^2) * (r^2 + eps^2)^(-7/4).

The Dirichlet boundary data u_exact equals (r^2 + eps^2)^(1/4) on the boundary of
Omega. This is a steady problem. Return the solution on a rectangular grid covering
[-1, 1]^3; **only the in-domain nodes are scored** (values in the removed octant are
ignored). The near-vertex layer degrades uniform low-order convergence, so scoring
is the relative L2 error over the domain plus an observed convergence-order check.
""",
    },
    {
        "id": "P23",
        "slug": "pde_acoustic_3d_layered",
        "type": "pde",
        "tier": 3,
        "discriminator": True,
        "family": "acoustic-layered",
        "title": "3D Acoustic Wave in a Layered Medium (MMS)",
        "challenge": "3-D wave with a discontinuous (layered) wave speed and high spatial frequency; numerical dispersion/phase pollution unless enough points per wavelength are used.",
        "dims": 3,
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]},
        "t_eval": 0.5,
        "has_ground_truth": True,
        "analytic": _acoustic_3d,
        "axes": ["x", "y", "z"],
        "grid_N": 32,
        "min_order": 1.5,
        "description": """# 3-D Acoustic Wave in a Layered Medium (manufactured solution)

Solve the acoustic wave equation for the pressure p(x, y, z, t) on the unit cube
[0, 1]^3, over t in [0, 0.5]:

    p_tt = c(x, y, z)^2 * Delta p + s(x, y, z, t),

with a **layered (discontinuous) wave speed**

    c(x, y, z) = 1 + 0.5 * [z > 0.5]        (1 in the lower layer, 1.5 in the upper).

Writing kv = 2*pi and omega = sqrt(3)*kv, the manufactured source is

    s = 12*pi^2 * (c(x,y,z)^2 - 1) * sin(kv*x) * sin(kv*y) * sin(kv*z) * cos(omega*t)

(so s = 0 in the lower layer, where the field is a free wave). The initial and
boundary data are taken from the hidden manufactured pressure at the appropriate
times; the field vanishes on all six faces of the cube. Report p(x, y, z, 0.5); it
is scored against the hidden reference (relative L2 error, plus an observed
spatial-convergence-order check). The high frequency across the layer interface
stresses phase accuracy and points-per-wavelength.
""",
    },
    {
        "id": "P24",
        "slug": "pde_navier_stokes_2d",
        "type": "pde",
        "tier": 3,
        "discriminator": True,
        "family": "navier-stokes",
        "title": "2D Navier-Stokes (Taylor-Green vortex)",
        "challenge": "Incompressible NS system (u, v, p); a scheme that fails to enforce div u = 0 pollutes the solution -- the divergence constraint is a hard gate, separate from field accuracy.",
        "dims": 2,
        "domain": {"x": [0.0, 6.283185307179586], "y": [0.0, 6.283185307179586]},
        "t_eval": 0.5,
        "has_ground_truth": True,
        "analytic": _taylor_green_2d,
        "axes": ["x", "y"],
        "fields": ["u", "v", "p"],
        "primary_field": "u",
        "required_fields": ["u", "v"],
        "gauge_fields": ["p"],
        "grid_N": 48,
        "min_order": 1.5,
        "diagnostics": [
            {"name": "divergence_norm", "fn": _diag_divergence_free(["u", "v"]), "gate": True},
            {"name": "energy_decay_error", "fn": _diag_energy_error(["u", "v"]), "gate": False},
        ],
        "description": """# 2-D Incompressible Navier-Stokes (Taylor-Green vortex)

Solve the incompressible Navier-Stokes equations on the periodic square
[0, 2*pi]^2, over t in [0, 0.5], for the velocity (u, v) and pressure p:

    u_t + (u . grad) u + grad p = nu * Delta u,
    div u = 0.

**Parameters**:  nu = 0.1.

**Initial condition** (the Taylor-Green vortex at t = 0):

    u(x, y, 0) =  sin(x) cos(y),
    v(x, y, 0) = -cos(x) sin(y),
    p(x, y, 0) =  0.25 (cos(2x) + cos(2y)).

**Boundary conditions**:  periodic in both directions.

Report the fields at t = 0.5 as a `fields` dict with keys `u`, `v`, `p` (each an
(N, N) array). Scoring, against a hidden reference: relative L2 error of the
velocity (u, v) plus an observed spatial-convergence-order check; the pressure is
compared after removing its mean (it is defined up to a constant). The **discrete
divergence of the velocity must stay near zero** -- this is a hard structural gate
(a solution that is accurate but not divergence-free does not pass).
""",
    },
    {
        "id": "P25",
        "slug": "pde_mhd_2d",
        "type": "pde",
        "tier": 3,
        "discriminator": True,
        "family": "mhd",
        "title": "2D Resistive MHD (divergence-free, MMS)",
        "challenge": "Coupled velocity + magnetic field; the solenoidal constraint div B = 0 is a distinct hard gate from div u = 0, and the Lorentz/induction coupling is stiff.",
        "dims": 2,
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "t_eval": 0.5,
        "has_ground_truth": True,
        "analytic": _mhd_2d,
        "axes": ["x", "y"],
        "fields": ["u", "v", "Bx", "By", "p"],
        "primary_field": "u",
        "required_fields": ["u", "v", "Bx", "By"],
        "gauge_fields": ["p"],
        "grid_N": 48,
        "min_order": 1.4,
        "diagnostics": [
            {"name": "div_B_norm", "fn": _diag_divergence_free(["Bx", "By"]), "gate": True},
            {"name": "div_u_norm", "fn": _diag_divergence_free(["u", "v"]), "gate": False},
            {"name": "magnetic_energy_error", "fn": _diag_energy_error(["Bx", "By"]), "gate": False},
        ],
        "description": """# 2-D Resistive MHD (manufactured divergence-free fields)

Solve the incompressible resistive MHD system on [0, 1]^2, over t in [0, 0.5], for
the velocity (u, v), magnetic field (Bx, By), and pressure p:

    u_t + (u . grad) u + grad p - nu Delta u - (curl B) x B = f_u,
    B_t - curl(u x B) - eta Delta B = f_B,
    div u = 0,   div B = 0.

**Parameters**:  nu = 0.1,  eta = 0.1.

The manufactured sources (making a known smooth divergence-free (u, B) the exact
solution), with sx = sin(pi x), sy = sin(pi y), cx = cos(pi x), cy = cos(pi y):

    f_u1 = (pi/5) [ -20 pi^2 (-4 cx^2 cy^2 + cx^2 + cy^2) sin(t)^2 sx cx cy^2
                    - 10 sin(t) sx^2 sy cy + 40 pi^2 sx^3 sy^2 cos(t)^2 cx cy^2
                    - 20 pi^2 sx^3 sy^2 cos(t)^2 cx cos(2 pi y)
                    + 4 pi^2 sx^2 sy cos(t) cy + 5 sy cos(t) cx
                    - 2 pi^2 sy cos(t) cos(2 pi x) cy ],
    f_u2 = (pi/5) [ -20 pi^2 (-4 cx^2 cy^2 + cx^2 + cy^2) sin(t)^2 sy cx^2 cy
                    + 10 sin(t) sx sy^2 cx + 40 pi^2 sx^2 sy^3 cos(t)^2 cx^2 cy
                    - 20 pi^2 sx^2 sy^3 cos(t)^2 cos(2 pi x) cy
                    - 4 pi^2 sx sy^2 cos(t) cx + 2 pi^2 sx cos(t) cx cos(2 pi y)
                    + 5 sx cos(t) cy ],
    f_B1 = (2 pi/5) [ 20 pi^2 sin(t) sx^3 sy^2 cos(t) cx - 10 pi^2 sin(t) sx^3 cos(t) cx
                      + 4 pi^2 sin(t) sx^2 sy cy - 40 pi^2 sin(t) sx sy^4 cos(t) cx
                      + 30 pi^2 sin(t) sx sy^2 cos(t) cx - 3 pi^2 sin(t) sy cy
                      + 5 sx^2 sy cos(t) cy - 5 sy cos(t) cy ],
    f_B2 = (2 pi/5) [ -40 pi^2 sin(t) sx^4 sy cos(t) cy + 20 pi^2 sin(t) sx^2 sy^3 cos(t) cy
                      + 30 pi^2 sin(t) sx^2 sy cos(t) cy - 4 pi^2 sin(t) sx sy^2 cx
                      + 3 pi^2 sin(t) sx cx - 10 pi^2 sin(t) sy^3 cos(t) cy
                      - 5 sx sy^2 cos(t) cx + 5 sx cos(t) cx ].

Initial and Dirichlet boundary data come from the hidden manufactured fields.
Report the fields at t = 0.5 as a `fields` dict with keys `u`, `v`, `Bx`, `By`, `p`.
Scoring: relative L2 error of (u, v, Bx, By) plus an observed convergence-order
check (pressure mean-removed). The **discrete div B must stay near zero** -- a hard
structural gate distinct from field accuracy; div u is reported.
""",
    },
    {
        "id": "P26",
        "slug": "pde_keller_segel_2d",
        "type": "pde",
        "tier": 3,
        "discriminator": True,
        "family": "keller-segel",
        "title": "2D Keller-Segel Chemotaxis (positivity, MMS)",
        "challenge": "Chemotaxis system (density rho, chemoattractant c); an aggregating drift makes naive schemes produce negative density -- positivity is a hard gate.",
        "dims": 2,
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "t_eval": 0.5,
        "has_ground_truth": True,
        "analytic": _keller_segel_2d,
        "axes": ["x", "y"],
        "fields": ["rho", "c"],
        "primary_field": "rho",
        "required_fields": ["rho", "c"],
        "grid_N": 48,
        "min_order": 1.4,
        "diagnostics": [
            {"name": "min_density", "fn": _diag_positivity(["rho", "c"], floor=0.0), "gate": True},
        ],
        "description": """# 2-D Keller-Segel Chemotaxis (manufactured positive fields)

Solve the Keller-Segel chemotaxis system on [0, 1]^2, over t in [0, 0.5], for the
cell density rho and the chemoattractant c:

    rho_t = D_r Delta rho - chi div(rho grad c) + s_rho,
    c_t   = D_c Delta c + alpha rho - beta c + s_c.

**Parameters**:  D_r = D_c = chi = alpha = beta = 1.

The manufactured sources (making known positive fields the exact solution), with
ss = sin(pi x) sin(pi y) and cc = cos(pi x) cos(pi y):

    s_rho = -0.2 ss e^(-t) + 0.4 pi^2 ss e^(-t)
            - 0.04 pi^2 sin(pi x) cos(pi x) sin(pi y) cos(pi y) e^(-3t)
            - 0.2 pi^2 (1 + 0.2 ss e^(-t)) cc e^(-2t),
    s_c   = (0.2 pi^2 - 0.1) cc e^(-2t) - 0.2 ss e^(-t).

Initial and Dirichlet boundary data come from the hidden manufactured fields.
Report the fields at t = 0.5 as a `fields` dict with keys `rho`, `c`. Scoring:
relative L2 error of rho and c plus an observed convergence-order check. **Both rho
and c must stay nonnegative everywhere** -- positivity is a hard structural gate
(the failure mode a naive chemotaxis discretization exhibits).
""",
    },
    {
        "id": "P27",
        "slug": "pde_elasticity_2d",
        "type": "pde",
        "tier": 3,
        "discriminator": True,
        "family": "elasticity-locking",
        "title": "2D Nearly-Incompressible Elasticity (locking, MMS)",
        "challenge": "lambda/mu = 1e4: a standard displacement scheme locks (volumetric over-stiffening), so the displacement error plateaus under refinement instead of converging.",
        "dims": 2,
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "t_eval": None,
        "has_ground_truth": True,
        "analytic": _elasticity_2d,
        "axes": ["x", "y"],
        "fields": ["u", "v"],
        "primary_field": "u",
        "required_fields": ["u", "v"],
        "grid_N": 48,
        "min_order": 1.3,
        "diagnostics": [
            {"name": "div_u_norm", "fn": _diag_divergence_free(["u", "v"]), "gate": False},
        ],
        "description": """# 2-D Nearly-Incompressible Linear Elasticity (manufactured solution)

Solve the linear elasticity equations on [0, 1]^2 for the displacement (u, v):

    -div(sigma(u)) = f,
    sigma = 2 mu epsilon(u) + lambda (div u) I,   epsilon = (grad u + grad u^T)/2.

**Parameters**:  mu = 1,  lambda = 1e4  (nearly incompressible; lambda/mu = 1e4).

The manufactured body force (the hidden displacement is divergence-free, so the
lambda term does not appear in f):

    f1 = -( 2 pi^2 cos(2 pi x) sin(2 pi y) - 4 pi^2 sin(pi x)^2 sin(2 pi y) ),
    f2 = -( 4 pi^2 sin(2 pi x) sin(pi y)^2 - 2 pi^2 sin(2 pi x) cos(2 pi y) ).

**Boundary conditions** (Dirichlet): u = v = 0 on the boundary of [0, 1]^2.

This is a steady problem. Report the displacement as a `fields` dict with keys
`u`, `v`. Scoring: relative L2 displacement error plus an observed
convergence-order check. The large lambda/mu ratio induces **volumetric locking**
in a naive displacement discretization -- the displacement error stops decreasing
under refinement (failing the order gate) unless a locking-free formulation is
used; the discrete div u is reported.
""",
    },
    {
        "id": "P28",
        "slug": "pde_maxwell_3d",
        "type": "pde",
        "tier": 3,
        "discriminator": True,
        "family": "maxwell",
        "title": "3D Maxwell Plane Wave (divergence-free, exact)",
        "challenge": "Full 3-D Maxwell system (E, H each 3-vectors); a scheme that does not preserve div E = div H = 0 accumulates spurious charge -- two hard 3-D structural gates plus phase accuracy.",
        "dims": 3,
        "domain": {"x": [0.0, 6.283185307179586], "y": [0.0, 6.283185307179586],
                   "z": [0.0, 6.283185307179586]},
        "t_eval": 0.5,
        "has_ground_truth": True,
        "analytic": _maxwell_3d,
        "axes": ["x", "y", "z"],
        "fields": ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
        "primary_field": "Ex",
        "required_fields": ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
        "grid_N": 24,
        "min_order": 1.6,
        "diagnostics": [
            {"name": "divE_norm", "fn": _diag_divergence_free(["Ex", "Ey", "Ez"]), "gate": True},
            {"name": "divH_norm", "fn": _diag_divergence_free(["Hx", "Hy", "Hz"]), "gate": True},
            {"name": "field_energy_error",
             "fn": _diag_energy_error(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]), "gate": False},
        ],
        "description": """# 3-D Maxwell Equations (periodic plane wave, exact solution)

Solve Maxwell's equations on the periodic cube [0, 2*pi]^3, over t in [0, 0.5], for
the electric field E = (Ex, Ey, Ez) and magnetic field H = (Hx, Hy, Hz):

    E_t = curl H,
    H_t = -curl E,
    div E = 0,   div H = 0.

**Initial condition** (a plane wave with wave vector k = (1, 1, 1), polarization
a = (1, -1, 0)/sqrt(2), and b = (k x a)/|k| = (1, 1, -2)/sqrt(6)):

    E(x, 0) = a * sin(x + y + z),
    H(x, 0) = b * sin(x + y + z).

**Boundary conditions**:  periodic in all three directions.

Report the fields at t = 0.5 as a `fields` dict with keys `Ex, Ey, Ez, Hx, Hy, Hz`
(each an (N, N, N) array). Scoring: relative L2 error of E and H plus an observed
spatial-convergence-order check. **Both div E and div H must stay near zero** --
two hard structural gates (a Yee-type staggered scheme preserves them; a naive
collocated scheme accumulates spurious divergence).
""",
    },
    # -------------------------------------------------------------------------
    # SDE -- Tier 1 (gentle parameter regimes; families in sde_manual.md)
    # -------------------------------------------------------------------------
    {
        "id": "S01",
        "slug": "sde_gbm",
        "type": "sde",
        "tier": 1,
        "family": "geometric_brownian_motion",
        "title": "Geometric Brownian Motion",
        "challenge": "Canonical multiplicative-noise SDE; Euler-Maruyama and Milstein both apply.",
        "state_dimension": 1,
        "T": 1.0,
        "dt": 0.01,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "moments": partial(gbm_moments, X0=1.0, mu=0.1, sigma=0.2),
        "description": """# Geometric Brownian Motion

Simulate the following stochastic differential equation (Ito):

    dX = 0.1 X dt + 0.2 X dW,   X(0) = 1.0

over the time interval [0, 1].

Compute the empirical mean E[X(T)] and variance Var[X(T)] at T = 1 using Monte
Carlo simulation and compare to the known analytic moments.
""",
    },
    {
        "id": "S02",
        "slug": "sde_ornstein_uhlenbeck",
        "type": "sde",
        "tier": 1,
        "family": "ornstein_uhlenbeck",
        "title": "Ornstein-Uhlenbeck Process",
        "challenge": "Additive noise, linear mean-reverting drift; EM suffices at a modest step.",
        "state_dimension": 1,
        "T": 1.0,
        "dt": 0.01,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "moments": partial(ou_moments, X0=2.0, theta=1.5, mu=0.0, sigma=0.5),
        "description": """# Ornstein-Uhlenbeck Process

Simulate the following stochastic differential equation (Ito):

    dX = 1.5 * (0.0 - X) dt + 0.5 dW,   X(0) = 2.0

over the time interval [0, 1].

Compute the empirical mean E[X(T)] and variance Var[X(T)] at T = 1 using Monte
Carlo simulation and compare to the known analytic moments.
""",
    },
    {
        "id": "S03",
        "slug": "sde_bm_with_drift",
        "type": "sde",
        "tier": 1,
        "family": "bm_with_drift",
        "title": "Brownian Motion with Drift",
        "challenge": "Constant drift and additive noise; exactly integrable, so EM should be essentially exact in the mean.",
        "state_dimension": 1,
        "T": 1.0,
        "dt": 0.01,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "moments": partial(bm_drift_moments, X0=1.0, mu=0.5, sigma=0.3),
        "description": """# Brownian Motion with Drift

Simulate the following stochastic differential equation (Ito):

    dX = 0.5 dt + 0.3 dW,   X(0) = 1.0

over the time interval [0, 1].

Compute the empirical mean E[X(T)] and variance Var[X(T)] at T = 1 using Monte
Carlo simulation and compare to the known analytic moments.
""",
    },
    {
        "id": "S04",
        "slug": "sde_linear_additive",
        "type": "sde",
        "tier": 1,
        "family": "linear_sde_additive",
        "title": "Linear SDE with Additive Noise",
        "challenge": "Affine drift with additive noise; closed-form moments from the linear moment ODEs.",
        "state_dimension": 1,
        "T": 1.0,
        "dt": 0.01,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "moments": partial(linear_additive_moments, X0=0.0, a=2.0, b=-1.0, c=0.5),
        "description": """# Linear SDE with Additive Noise

Simulate the following linear stochastic differential equation (Ito):

    dX = (2.0 - 1.0 * X) dt + 0.5 dW,   X(0) = 0.0

over the time interval [0, 1].

Compute the empirical mean E[X(T)] and variance Var[X(T)] at T = 1 using Monte
Carlo simulation and compare to the known analytic moments.
""",
    },
    {
        "id": "S05",
        "slug": "sde_bm_standard",
        "type": "sde",
        "tier": 1,
        "family": "bm_standard",
        "title": "Standard Brownian Motion",
        "challenge": "Zero mean by symmetry; exercises the near-zero-mean skip in the scoring rubric.",
        "state_dimension": 1,
        "T": 1.0,
        "dt": 0.01,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "moments": lambda T: bm_standard_moments(T),
        "description": """# Standard Brownian Motion

Simulate standard Brownian motion (Ito):

    dX = dW,   X(0) = 0.0

over the time interval [0, 1].

Compute the empirical mean E[X(T)] and variance Var[X(T)] at T = 1 using Monte
Carlo simulation and compare to the known analytic moments (mean 0, variance T).
""",
    },
    # -------------------------------------------------------------------------
    # SDE -- Tier 2 (moderate: positivity/log guards, scale, multi-D)
    # -------------------------------------------------------------------------
    {
        "id": "S06",
        "slug": "sde_cir",
        "type": "sde",
        "tier": 2,
        "family": "cox_ingersoll_ross",
        "title": "Cox-Ingersoll-Ross (Feller satisfied)",
        "challenge": "Square-root diffusion needs a positivity guard before sqrt; Feller condition holds here.",
        "state_dimension": 1,
        "T": 1.0,
        "dt": 0.01,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "moments": partial(cir_moments, X0=0.5, kappa=2.0, theta=1.0, sigma=0.5),
        "description": """# Cox-Ingersoll-Ross Process

Simulate the following stochastic differential equation (Ito):

    dX = 2.0 * (1.0 - X) dt + 0.5 * sqrt(X) dW,   X(0) = 0.5

over the time interval [0, 1]. The Feller condition 2 kappa theta >= sigma^2 holds
(4 >= 0.25), so the process stays positive; still, apply a positivity guard before
the square root to protect against floating-point undershoot.

Compute the empirical mean E[X(T)] and variance Var[X(T)] at T = 1 using Monte
Carlo simulation and compare to the known analytic moments.
""",
    },
    {
        "id": "S07",
        "slug": "sde_exponential_ou",
        "type": "sde",
        "tier": 2,
        "family": "exponential_ornstein_uhlenbeck",
        "title": "Exponential Ornstein-Uhlenbeck",
        "challenge": "Drift contains log(X); needs a guard to keep X strictly positive.",
        "state_dimension": 1,
        "T": 1.0,
        "dt": 0.01,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "moments": partial(exp_ou_moments, X0=1.0, theta=1.0, sigma=0.4),
        "description": """# Exponential Ornstein-Uhlenbeck Process

Simulate the following stochastic differential equation (Ito):

    dX = X * (-1.0 * log(X) + 0.4^2 / 2) dt + 0.4 * X dW,   X(0) = 1.0

over the time interval [0, 1]. The log(X) term requires X > 0; guard the state to
stay strictly positive.

Compute the empirical mean E[X(T)] and variance Var[X(T)] at T = 1 using Monte
Carlo simulation and compare to the known analytic moments.
""",
    },
    {
        "id": "S08",
        "slug": "sde_black_scholes",
        "type": "sde",
        "tier": 2,
        "family": "black_scholes",
        "title": "Black-Scholes Asset Price",
        "challenge": "GBM at financial scale (X0=100); large magnitudes stress relative-error bookkeeping.",
        "state_dimension": 1,
        "T": 1.0,
        "dt": 0.01,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "moments": partial(gbm_moments, X0=100.0, mu=0.05, sigma=0.2),
        "description": """# Black-Scholes Asset Price

Simulate the risk-neutral asset price (Ito):

    dX = 0.05 X dt + 0.20 X dW,   X(0) = 100.0

over the time interval [0, 1].

Compute the empirical mean E[X(T)] and variance Var[X(T)] at T = 1 using Monte
Carlo simulation and compare to the known analytic moments.
""",
    },
    {
        "id": "S09",
        "slug": "sde_gbm_2d_correlated",
        "type": "sde",
        "tier": 2,
        "family": "gbm_2d_correlated",
        "title": "2D Correlated Geometric Brownian Motion",
        "challenge": "Vector state with correlated noise; requires a Cholesky factor and per-component scoring.",
        "state_dimension": 2,
        "T": 1.0,
        "dt": 0.01,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "moments": partial(gbm2d_moments, X0=1.0, Y0=1.0, mu1=0.10, s1=0.20, mu2=0.15, s2=0.25),
        "description": """# 2D Correlated Geometric Brownian Motion

Simulate the coupled system (Ito) of two correlated GBMs:

    dX = 0.10 X dt + 0.20 X dW1,   X(0) = 1.0
    dY = 0.15 Y dt + 0.25 Y dW2,   Y(0) = 1.0

where dW1 and dW2 are correlated Brownian increments with correlation rho = 0.60.
Introduce the correlation with a Cholesky factor.

Over the time interval [0, 1], compute the empirical means and variances of X(T)
and Y(T) at T = 1 using Monte Carlo simulation and compare to the known analytic
per-component moments.
""",
    },
    {
        "id": "S10",
        "slug": "sde_stochastic_oscillator",
        "type": "sde",
        "tier": 2,
        "family": "stochastic_oscillator",
        "title": "Stochastic Harmonic Oscillator",
        "challenge": "Vector state, noise only in the velocity component; the X-mean returns near zero at T=2pi.",
        "state_dimension": 2,
        "T": 6.283185307179586,  # 2*pi
        "dt": 0.01,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "moments": partial(oscillator_moments, X0=1.0, Y0=0.0, sigma=0.3),
        "description": """# Stochastic Harmonic Oscillator

Simulate the coupled system (Ito):

    dX = Y dt,                 X(0) = 1.0
    dY = -X dt + 0.3 dW,       Y(0) = 0.0

over the time interval [0, 2 pi]. Noise enters only the Y (velocity) component.

Compute the empirical means and variances of X(T) and Y(T) at T = 2 pi using Monte
Carlo simulation and compare to the known analytic per-component moments. (The mean
of Y returns near zero at T = 2 pi.)
""",
    },
    # -------------------------------------------------------------------------
    # SDE -- Tier 3 (hard stress regimes; exact moments still available)
    # -------------------------------------------------------------------------
    {
        "id": "S11",
        "slug": "sde_cir_feller_violated",
        "type": "sde",
        "tier": 3,
        "family": "cox_ingersoll_ross",
        "title": "CIR with Feller Condition Violated",
        "challenge": "2 kappa theta = 1 < sigma^2 = 4, so paths hit zero frequently; naive Euler biases the variance badly.",
        "state_dimension": 1,
        "T": 1.0,
        "dt": 0.005,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "moments": partial(cir_moments, X0=0.5, kappa=1.0, theta=0.5, sigma=2.0),
        "description": """# Cox-Ingersoll-Ross with Feller Condition Violated

Simulate the following stochastic differential equation (Ito):

    dX = 1.0 * (0.5 - X) dt + 2.0 * sqrt(X) dW,   X(0) = 0.5

over the time interval [0, 1]. Here 2 kappa theta = 1 < sigma^2 = 4, so the Feller
condition is VIOLATED and the process reaches zero frequently. A robust boundary
treatment (full truncation / reflection) is needed; a positivity guard is mandatory.

Compute the empirical mean E[X(T)] and variance Var[X(T)] at T = 1 using Monte
Carlo simulation and compare to the known analytic moments. (The moment formulas
remain valid regardless of the Feller condition.)
""",
    },
    {
        "id": "S12",
        "slug": "sde_gbm_high_vol",
        "type": "sde",
        "tier": 3,
        "family": "geometric_brownian_motion",
        "title": "High-Volatility GBM",
        "challenge": "sigma=1.0 gives a heavy-tailed lognormal; the Monte Carlo variance estimator itself is high-variance.",
        "state_dimension": 1,
        "T": 1.0,
        "dt": 0.005,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "moments": partial(gbm_moments, X0=1.0, mu=0.05, sigma=1.0),
        "description": """# High-Volatility Geometric Brownian Motion

Simulate the following stochastic differential equation (Ito):

    dX = 0.05 X dt + 1.0 X dW,   X(0) = 1.0

over the time interval [0, 1]. The large volatility makes X(T) strongly
heavy-tailed (lognormal with log-variance 1), so the empirical variance has large
sampling error and demands many paths and small steps.

Compute the empirical mean E[X(T)] and variance Var[X(T)] at T = 1 using Monte
Carlo simulation and compare to the known analytic moments.
""",
    },
    {
        "id": "S13",
        "slug": "sde_ou_stiff",
        "type": "sde",
        "tier": 3,
        "family": "ornstein_uhlenbeck",
        "title": "Stiff Ornstein-Uhlenbeck (fast mean reversion)",
        "challenge": "theta=50 makes explicit EM unstable unless dt < 2/theta = 0.04; tests time-step selection / stability.",
        "state_dimension": 1,
        "T": 1.0,
        "dt": 0.002,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "moments": partial(ou_moments, X0=2.0, theta=50.0, mu=0.0, sigma=2.0),
        "description": """# Stiff Ornstein-Uhlenbeck Process (fast mean reversion)

Simulate the following stochastic differential equation (Ito):

    dX = 50.0 * (0.0 - X) dt + 2.0 dW,   X(0) = 2.0

over the time interval [0, 1]. The fast mean reversion (theta = 50) makes the
problem stiff: explicit Euler-Maruyama is unstable unless dt < 2/theta = 0.04, so
the step size must be chosen carefully (or an implicit scheme used).

Compute the empirical mean E[X(T)] and variance Var[X(T)] at T = 1 using Monte
Carlo simulation and compare to the known analytic moments. (The mean decays to
near zero at T = 1.)
""",
    },
    {
        "id": "S14",
        "slug": "sde_gbm_2d_high_corr",
        "type": "sde",
        "tier": 3,
        "family": "gbm_2d_correlated",
        "title": "2D GBM with Near-Singular Correlation",
        "challenge": "rho=0.95 makes the correlation matrix nearly singular; the Cholesky factor is ill-conditioned and volatility is higher.",
        "state_dimension": 2,
        "T": 1.0,
        "dt": 0.005,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "moments": partial(gbm2d_moments, X0=1.0, Y0=1.0, mu1=0.10, s1=0.30, mu2=0.10, s2=0.30),
        "description": """# 2D Geometric Brownian Motion with Near-Singular Correlation

Simulate the coupled system (Ito) of two strongly correlated GBMs:

    dX = 0.10 X dt + 0.30 X dW1,   X(0) = 1.0
    dY = 0.10 Y dt + 0.30 Y dW2,   Y(0) = 1.0

where dW1 and dW2 have correlation rho = 0.95. The near-singular correlation matrix
makes the Cholesky factor ill-conditioned; the higher volatility (0.30) also widens
the terminal distribution.

Over the time interval [0, 1], compute the empirical means and variances of X(T)
and Y(T) at T = 1 using Monte Carlo simulation and compare to the known analytic
per-component moments.
""",
    },
    {
        "id": "S15",
        "slug": "sde_oscillator_long_horizon",
        "type": "sde",
        "title": "Long-Horizon Stochastic Oscillator",
        "tier": 3,
        "family": "stochastic_oscillator",
        "challenge": "Integrated over five periods (T=10pi); phase error and variance accumulate over ~15000 steps.",
        "state_dimension": 2,
        "T": 31.41592653589793,  # 10*pi
        "dt": 0.005,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "moments": partial(oscillator_moments, X0=1.0, Y0=0.0, sigma=0.3),
        "description": """# Long-Horizon Stochastic Harmonic Oscillator

Simulate the coupled system (Ito):

    dX = Y dt,                 X(0) = 1.0
    dY = -X dt + 0.3 dW,       Y(0) = 0.0

over the long time interval [0, 10 pi] (five full periods). Long integration
accumulates phase error in the mean and growth in the variance, stressing the
step size and the stability of the scheme.

Compute the empirical means and variances of X(T) and Y(T) at T = 10 pi using
Monte Carlo simulation and compare to the known analytic per-component moments.
""",
    },
    # -------------------------------------------------------------------------
    # SDE -- Tier 3 "hard SDE" additions (from the deep-literature review;
    # see benchmark/hard_sde_candidates_from_lit.md). These push past the classical
    # families above: a Feller-violated coupled 2-D affine system, a stiff
    # many-channel linear system, a superlinear noise-free quintic, two
    # superlinear problems scored against a discretization-free reference, and two
    # blow-up / domain stability stress tests.
    # -------------------------------------------------------------------------
    {
        "id": "S16",
        "slug": "sde_log_heston_feller_violated",
        "type": "sde",
        "tier": 3,
        # Part of the hard-SDE discriminator subset: the problems where the
        # pipeline-vs-one-shot comparison is expected to separate (see
        # discriminator_problems()). Set this on hard PDEs as they are added.
        "discriminator": True,
        "family": "log_heston",
        "title": "Log-Heston with Feller Condition Violated",
        "challenge": "Coupled 2-D affine (log-price + CIR variance); 2a=0.2 < sigma^2=1.0 so the variance hits zero and naive sqrt(Y) Euler goes negative -- needs positivity-preserving/exact CIR sampling.",
        "state_dimension": 2,
        "T": 1.0,
        "dt": 0.005,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "ground_truth_kind": "exact",
        "moments": partial(log_heston_moments, r=0.05, a=0.1, b=1.0, sigma=1.0,
                           rho=-0.9, X0=0.0, Y0=0.1),
        "description": """# Log-Heston Model with Feller Condition Violated

Simulate the two-dimensional log-Heston system (Ito), state (X, Y) with X the
log-price and Y the instantaneous variance:

    dX = (r - Y/2) dt + sqrt(Y) * ( rho dW1 + sqrt(1 - rho^2) dW2 ),   X(0) = 0.0
    dY = (a - b Y) dt + sigma sqrt(Y) dW1,                              Y(0) = 0.1

over [0, 1]. W1 drives the variance and is correlated (through rho) with the
log-price; W2 is the independent second factor of the log-price.

**Parameters**:  r = 0.05,  a = 0.1,  b = 1.0,  sigma = 1.0,  rho = -0.9

The variance Y is a CIR process with 2 a = 0.2 < sigma^2 = 1.0, so the **Feller
condition is VIOLATED**: Y reaches 0 frequently. Fast/naive schemes fail here -- a
plain Euler step on sqrt(Y) drives Y negative and produces NaNs, and the
Ninomiya-Victoir splitting is only well-defined for sigma^2 <= 4a (here
1.0 > 0.4). A positivity-preserving treatment (exact noncentral chi-square CIR
sampling, or a full-truncation / implicit scheme with a guard before the square
root) is required.

Return the terminal states of both components as `terminal_paths` (shape
(num_paths, 2): column 0 = X = log-price, column 1 = Y = variance). Compute the
empirical means and variances of X(T) and Y(T) at T = 1 and compare to the known
per-component moments. (Because (X, Y) is affine, the terminal moments are exact:
they solve a closed linear ODE system.)
""",
    },
    {
        "id": "S17",
        "slug": "sde_multichannel_stiff_m13",
        "type": "sde",
        "tier": 3,
        "discriminator": True,
        "family": "multichannel_stiff_linear",
        "title": "Multi-Channel Stiff Linear SDE (13 noise channels)",
        "challenge": "13 non-commuting multiplicative noise channels on a spiral drift; a correct solver must assemble all 13 channels and resolve the step-size-sensitive terminal variance, well beyond the 1-2 channel baseline.",
        "state_dimension": 2,
        "T": 1.0,
        # The verifier re-runs at exactly this step, and the *mean* gate (5%) is
        # tighter than the variance gate (10%): explicit Euler's O(dt) drift error
        # puts the X-mean 5.31% off at dt=5e-3 -- over the gate before any MC noise,
        # making the verdict a coin flip. At 5e-4 the bias is 0.53%, leaving only the
        # ~1.6% MC noise floor. Pinned by the mean-bias guard in
        # validate_ground_truth.py.
        "dt": 0.0005,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "ground_truth_kind": "exact",
        "moments": multichannel_stiff_moments,
        "description": """# Multi-Channel Stiff Linear SDE (13 noise channels)

Simulate the 2-dimensional linear Ito system driven by m = 13 independent Brownian
motions:

    dX = F X dt + sum_{r=1}^{13} G_r X dW_r,   X(0) = (1, 1)^T

with the spiral drift matrix

    F = [[-2,  3],
         [-3, -2]]

and non-commuting multiplicative diffusion matrices that alternate between two shear
generators:

    G_r = 0.6 * M1  for r odd,     G_r = 0.6 * M2  for r even,
    M1 = [[0, 1], [0, 0]],         M2 = [[0, 0], [1, 0]]      ([M1, M2] != 0).

over [0, 1]. This is well beyond the 1-2 channel classical baseline: all 13
multiplicative, non-commuting channels must be assembled correctly, and the strong
multiplicative noise makes the terminal variance sensitive to the step size, so the
scheme and step must be chosen with care.

Return the terminal states as `terminal_paths` (shape (num_paths, 2): column 0 = X,
column 1 = Y). Compute the empirical means and variances of each component at T = 1
and compare to the exact moments. (Being linear, the system has exact moments:
mean = expm(F T) X0, and the second-moment matrix solves
dP/dt = F P + P F^T + sum_r G_r P G_r^T.)
""",
    },
    {
        "id": "S18",
        "slug": "sde_quintic_random_ic",
        "type": "sde",
        "tier": 3,
        "discriminator": True,
        "family": "quintic_random_ic",
        "title": "Noise-Free Quintic with Random Initial Condition",
        "challenge": "Superlinear drift dX=-X^5 dt with a Gaussian initial condition; plain Euler-Maruyama overshoots and diverges on the large-|X0| tail, so it needs a tamed/adaptive step.",
        "state_dimension": 1,
        "T": 1.0,
        "dt": 0.005,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "ground_truth_kind": "exact",
        "moments": partial(quintic_ic_moments, sigma_bar=1.0 / 3.0),
        "description": """# Noise-Free Quintic SDE with Random Initial Condition

Simulate the superlinear, noise-free Ito SDE

    dX = -X^5 dt,     X(0) = xi ~ Normal(0, sigma_bar^2),   sigma_bar = 1/3

over [0, 1]. The randomness enters only through the Gaussian initial condition;
each path is then a deterministic (but stiff) decay. The exact per-path solution
is X_t = xi / (1 + 4 t xi^4)^{1/4}.

Naive Euler-Maruyama (y_{n+1} = y_n - y_n^5 * dt) overshoots and blows up on paths
with large |xi| -- the quintic drift compounds double-exponentially once |X| is
large. A tamed step (drift divided by 1 + dt*|drift|) or an adaptive/implicit
integrator is the correct choice.

Compute the empirical mean E[X(T)] and variance Var[X(T)] at T = 1 by Monte Carlo
over the random initial condition, and compare to the known moments. (The mean is 0
by symmetry; the variance E[X(T)^2] is an exact Gaussian integral.)
""",
    },
    {
        "id": "S19",
        "slug": "sde_ginzburg_landau_s4",
        "type": "sde",
        "tier": 3,
        "discriminator": True,
        "family": "ginzburg_landau",
        "title": "Stochastic Ginzburg-Landau (sigma=4)",
        "challenge": "Superlinear cubic drift with multiplicative noise; untamed Euler biases E[X^2] low. Scored against a discretization-free Monte-Carlo of the exact solution.",
        "state_dimension": 1,
        "T": 1.0,
        # The verifier re-runs the solver at exactly this step, so it must be fine
        # enough for a *correct* (tamed) scheme to resolve the terminal variance:
        # at dt=5e-3 even a correct tamed Euler is 236% off, at 2e-4 it is <1%.
        # See the solvable-at-problem-dt check in validate_ground_truth.py.
        "dt": 0.0002,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "ground_truth_kind": "reference",
        "moments": partial(gl_reference_moments, sigma=4.0),
        "description": """# Stochastic Ginzburg-Landau Equation (sigma = 4)

Simulate the stochastic Ginzburg-Landau SDE (Ito)

    dX = ( (sigma^2/2) X - X^3 ) dt + sigma X dW,     X(0) = 1.0

over [0, 1], with **sigma = 4**. The drift is superlinear (cubic), so plain
Euler-Maruyama systematically underestimates E[X(T)^2] (and at larger sigma will
produce NaNs). A tamed / truncated Euler scheme is the correct choice.

Compute the empirical mean E[X(T)] and variance Var[X(T)] at T = 1 by Monte Carlo,
and compare to the reference moments. (No elementary moment formula exists; the
reference is a discretization-free Monte-Carlo of the exact solution
X_t = exp(sigma W_t) / sqrt(1 + 2 * integral_0^t exp(2 sigma W_u) du).)
""",
    },
    {
        "id": "S20",
        "slug": "sde_ginzburg_landau_s6",
        "type": "sde",
        "tier": 3,
        "discriminator": True,
        "family": "ginzburg_landau",
        "title": "Stochastic Ginzburg-Landau (sigma=6)",
        "challenge": "Same superlinear family at higher volatility; naive Euler produces partial NaNs and a badly biased variance. Scored against a discretization-free reference.",
        "state_dimension": 1,
        "T": 1.0,
        # See the sigma=4 note: at dt=5e-3 even a correct tamed Euler is 4751% off
        # (the terminal variance is extremely step-sensitive at this volatility);
        # 2e-4 brings a correct scheme to a few percent.
        "dt": 0.0002,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": True,
        "ground_truth_kind": "reference",
        "moments": partial(gl_reference_moments, sigma=6.0),
        "description": """# Stochastic Ginzburg-Landau Equation (sigma = 6)

Simulate the stochastic Ginzburg-Landau SDE (Ito)

    dX = ( (sigma^2/2) X - X^3 ) dt + sigma X dW,     X(0) = 1.0

over [0, 1], with **sigma = 6**. At this volatility plain Euler-Maruyama produces
partial NaNs and a severely biased variance; a tamed / truncated Euler scheme is
required to keep the paths finite and the moments accurate.

Compute the empirical mean E[X(T)] and variance Var[X(T)] at T = 1 by Monte Carlo,
and compare to the reference moments. (No elementary moment formula exists; the
reference is a discretization-free Monte-Carlo of the exact solution
X_t = exp(sigma W_t) / sqrt(1 + 2 * integral_0^t exp(2 sigma W_u) du).)
""",
    },
    {
        "id": "S21",
        "slug": "sde_quintic_drift_noise",
        "type": "sde",
        "tier": 3,
        "discriminator": True,
        "family": "quintic_superlinear",
        "title": "Quintic Superlinear Drift with Multiplicative Noise",
        "challenge": "dX=-X^5 dt + X dW: the superlinear drift makes plain Euler-Maruyama diverge almost surely (Hutzenthaler-Jentzen-Kloeden); a tamed scheme stays finite. No closed form -- checked only for non-divergence.",
        "state_dimension": 1,
        "T": 1.0,
        "dt": 0.005,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": False,
        "ground_truth_kind": "stability",
        "stability_check": {"type": "finite"},
        "moments": None,
        "description": """# Quintic Superlinear Drift with Multiplicative Noise

Simulate the Ito SDE

    dX = -X^5 dt + X dW,     X(0) = 1.0

over [0, 1]. The drift grows superlinearly (degree 5). By the Hutzenthaler-Jentzen-
Kloeden divergence theorem, plain Euler-Maruyama diverges (its moments blow up to
infinity, and in floating point it returns Inf/NaN). A tamed Euler scheme -- drift
step divided by 1 + dt*|drift| -- or an implicit/adaptive integrator stays finite
and converges.

There is no closed-form solution, so this problem is a **stability** benchmark: the
independent check confirms only that the terminal states stay finite (no blow-up),
which is exactly what a naive scheme gets wrong. Return the terminal states as
`terminal_paths` (shape (num_paths,)) and report the empirical mean and variance of
X(T) at T = 1.
""",
    },
    {
        "id": "S22",
        "slug": "sde_fene_blowup",
        "type": "sde",
        "tier": 3,
        "discriminator": True,
        "family": "fene",
        "title": "FENE Nonlinear Spring (Domain Blow-up)",
        "challenge": "dX=-X/(1-X^2) dt + dW confined to (-1, 1) by a singular spring force; a fixed-step Euler overshoots the boundary and escapes to NaN. Checked for staying in-domain.",
        "state_dimension": 1,
        "T": 1.0,
        "dt": 0.005,
        "num_paths": 50000,
        "seed": 42,
        "has_ground_truth": False,
        "ground_truth_kind": "stability",
        "stability_check": {"type": "domain", "abs_bound": 1.0},
        "moments": None,
        "description": """# FENE Nonlinear Spring (Finitely Extensible, Domain Blow-up)

Simulate the finitely-extensible (FENE) Ito SDE on the open interval (-1, 1):

    dX = -X / (1 - X^2) dt + dW,     X(0) = 0.0

over [0, 1]. The spring force -X/(1 - X^2) diverges as |X| -> 1, confining the true
process to (-1, 1) for all time. But a naive fixed-step Euler-Maruyama near the
boundary takes a huge step, overshoots past +-1, and then the force flips sign and
the path runs away to +-Inf / NaN. A boundary-aware adaptive step (or a suitable
implicit / rejection scheme) keeps every path inside (-1, 1).

There is no closed-form solution, so this problem is a **stability** benchmark: the
independent check confirms only that every terminal state stays inside the domain
(|X(T)| < 1 and finite), which is what a naive scheme gets wrong. Return the
terminal states as `terminal_paths` (shape (num_paths,)) and report the empirical
mean and variance of X(T) at T = 1.
""",
    },
]


# --- convenience accessors -------------------------------------------------


def by_slug(slug):
    for p in PROBLEMS:
        if p["slug"] == slug:
            return p
    raise KeyError(f"no benchmark problem with slug {slug!r}")


def by_id(pid):
    for p in PROBLEMS:
        if p["id"] == pid:
            return p
    raise KeyError(f"no benchmark problem with id {pid!r}")


def pde_problems():
    return [p for p in PROBLEMS if p["type"] == "pde"]


def sde_problems():
    return [p for p in PROBLEMS if p["type"] == "sde"]


def discriminator_problems():
    """The subset where the pipeline-vs-one-shot comparison is expected to separate
    -- run at higher repetition counts. Currently the 7 hard-literature SDEs; flag
    incoming hard PDEs with ``"discriminator": True`` to fold them in (a one-line
    change per problem, so the subset is defined in exactly one place)."""
    return [p for p in PROBLEMS if p.get("discriminator")]


if __name__ == "__main__":
    # Quick inventory.
    n_pde = len(pde_problems())
    n_sde = len(sde_problems())
    print(f"Total problems: {len(PROBLEMS)}  ({n_pde} PDE, {n_sde} SDE)")
    for typ in ("pde", "sde"):
        for tier in (1, 2, 3):
            items = [p for p in PROBLEMS if p["type"] == typ and p["tier"] == tier]
            gt = sum(1 for p in items if p["has_ground_truth"])
            print(f"  {typ.upper()} tier {tier}: {len(items)} problems ({gt} with ground truth)")
            for p in items:
                print(f"      {p['id']}  {p['slug']:<32} {p['family']}")
