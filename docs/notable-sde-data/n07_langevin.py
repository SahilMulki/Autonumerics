"""N07 truth derivation: underdamped Langevin at moderate friction (BAOAB).

    dq = p dt
    dp = (-omega^2 q - gamma p) dt + sqrt(2 gamma / beta) dW

A 2-D linear SDE with additive noise, so the mean is `e^{At} X0` and the covariance
is the solution of the matrix Lyapunov ODE -- **exact**, by the same matrix-exponential
machinery `sde_oscillator_long_horizon` and `sde_multichannel_stiff_m13` already use.

Three things this script establishes, in order of what the plan needs:

  1. the truth, by **two independent routes** (Gauss-Legendre quadrature of the
     stochastic-convolution integral, and `solve_ivp` on the covariance ODE), plus
     the closed-form stationary limit as a third anchor;
  2. the **(gamma, omega) choice**, which plan-notable-sde.md section 2 (N07) says is
     set by this sweep and not by taste: the pair must make Euler-Maruyama's bias on
     Var(q) exceed the 10% variance gate at the step the verifier re-runs at, while
     BAOAB clears it;
  3. the **vacuity check** (section 1 rule 6): Var(q) at T must sit far from both the
     initial condition and the stationary value, or the entry is answerable without
     integrating anything.

Seeded, numpy + scipy only. Run:
`uv run python docs/notable-sde-data/n07_langevin.py`
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

BETA = 1.0                      # inverse temperature; kT = 1
X0 = np.array([1.0, 0.0])       # (q, p)
T = 1.0
DT = 0.04                       # verify.sde_verify_dt: min(dt, 0.04/T)
SEED = 42


def drift_matrix(gamma, omega):
    return np.array([[0.0, 1.0], [-omega ** 2, -gamma]])


def noise_cov(gamma):
    """Q = b b^T for b = (0, sqrt(2 gamma / beta))."""
    s = 2.0 * gamma / BETA
    return np.array([[0.0, 0.0], [0.0, s]])


# --- route 1: Gauss-Legendre quadrature of the stochastic convolution -------

def exact_quadrature(gamma, omega, t=T, nodes=200):
    """mean = e^{At} X0 and Sigma(t) = int_0^t e^{As} Q e^{A^T s} ds.

    The integrand is analytic in `s`, so Gauss-Legendre converges geometrically and
    200 nodes is machine precision -- checked against `nodes=400` below.
    """
    A, Q = drift_matrix(gamma, omega), noise_cov(gamma)
    xi, w = np.polynomial.legendre.leggauss(nodes)
    s = 0.5 * t * (xi + 1.0)
    jac = 0.5 * t
    cov = np.zeros((2, 2))
    for sk, wk in zip(s, w, strict=True):
        E = expm(A * sk)
        cov += jac * wk * (E @ Q @ E.T)
    return expm(A * t) @ X0, cov


# --- route 2: the covariance ODE, integrated ---------------------------------

def exact_ode(gamma, omega, t=T):
    """Sigma' = A Sigma + Sigma A^T + Q, Sigma(0) = 0, at rtol 1e-13."""
    A, Q = drift_matrix(gamma, omega), noise_cov(gamma)

    def rhs(_t, y):
        S = y.reshape(2, 2)
        return (A @ S + S @ A.T + Q).ravel()

    sol = solve_ivp(rhs, [0.0, t], np.zeros(4), rtol=1e-13, atol=1e-15,
                    dense_output=True)
    return expm(A * t) @ X0, sol.y[:, -1].reshape(2, 2)


def stationary(omega):
    """The Gibbs covariance: Var(q) = 1/(beta omega^2), Var(p) = 1/beta, no
    correlation. Independent of gamma, which is the fluctuation-dissipation
    statement and a third check on the two routes above."""
    return np.array([[1.0 / (BETA * omega ** 2), 0.0], [0.0, 1.0 / BETA]])


# --- the two schemes ---------------------------------------------------------

def euler_maruyama(gamma, omega, dt, n_paths, nt, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.tile(X0, (n_paths, 1))
    c = np.sqrt(2.0 * gamma / BETA * dt)
    for _ in range(nt):
        xi = rng.standard_normal(n_paths)
        q, p = x[:, 0], x[:, 1]
        x = np.stack([q + p * dt,
                      p + (-omega ** 2 * q - gamma * p) * dt + c * xi], axis=1)
    return x


def baoab(gamma, omega, dt, n_paths, nt, seed=SEED):
    """B-A-O-A-B, the Leimkuhler-Matthews ordering: half kick, half drift, exact
    Ornstein-Uhlenbeck on the momentum, half drift, half kick."""
    rng = np.random.default_rng(seed)
    q = np.full(n_paths, X0[0])
    p = np.full(n_paths, X0[1])
    d = np.exp(-gamma * dt)
    c = np.sqrt((1.0 - d ** 2) / BETA)
    for _ in range(nt):
        p = p - 0.5 * dt * omega ** 2 * q
        q = q + 0.5 * dt * p
        p = d * p + c * rng.standard_normal(n_paths)
        q = q + 0.5 * dt * p
        p = p - 0.5 * dt * omega ** 2 * q
    return np.stack([q, p], axis=1)


def var_err(x, cov):
    """Relative error on Var(q), which is the component the gate reads."""
    return abs(float(x[:, 0].var(ddof=1)) / cov[0, 0] - 1.0)


SDE_VAR_TOL = 0.10          # verify.SDE_VAR_TOL


def em_spectral_radius(gamma, omega, dt=DT):
    """Spectral radius of Euler-Maruyama's deterministic amplification `I + A dt`.

    The distinction this exists to draw: above 1 the scheme is *linearly unstable*
    and its variance error is a divergence rather than a bias. Section 5 of the plan
    records that a loud control calibrates badly -- it separates everything from
    everything and sets no scale -- so a pair whose separation comes from EM blowing
    up is a worse control than one where EM is stable and merely wrong.
    """
    M = np.eye(2) + drift_matrix(gamma, omega) * dt
    return float(max(abs(np.linalg.eigvals(M))))


if __name__ == "__main__":
    print(f"Langevin, beta={BETA}, X0={tuple(X0)}, T={T}, dt={DT} "
          f"({int(round(T / DT))} steps)\n")

    # --- 1. the truth, two routes + the stationary anchor -------------------
    print("--- truth: two independent routes ---")
    for gamma, omega in [(2.0, 5.0), (1.0, 1.0), (10.0, 3.0)]:
        m_q, c_q = exact_quadrature(gamma, omega)
        m_o, c_o = exact_ode(gamma, omega)
        _m2, c_q2 = exact_quadrature(gamma, omega, nodes=400)
        print(f"  gamma={gamma:<5} omega={omega:<4} Var(q)={c_q[0, 0]:.12f}  "
              f"|quad-ode|={abs(c_q[0, 0] - c_o[0, 0]):.2e}  "
              f"|200-400 nodes|={abs(c_q[0, 0] - c_q2[0, 0]):.2e}  "
              f"|mean diff|={np.max(np.abs(m_q - m_o)):.2e}")

    print("\n--- the stationary limit (third anchor, t = 200) ---")
    for gamma, omega in [(2.0, 5.0), (10.0, 3.0)]:
        _m, c = exact_quadrature(gamma, omega, t=200.0, nodes=600)
        st = stationary(omega)
        print(f"  gamma={gamma:<5} omega={omega:<4} Var(q)={c[0, 0]:.10f} vs "
              f"Gibbs 1/(beta w^2)={st[0, 0]:.10f}  rel={abs(c[0, 0] / st[0, 0] - 1):.2e}")

    # --- 2. the (gamma, omega) sweep, under all three constraints ----------
    #
    # Three things have to hold at once and they pull against each other:
    #   (a) EM linearly STABLE (rho < 1), or its error is a divergence and the
    #       control is loud -- rho < 1 requires gamma > omega^2 dt;
    #   (b) EM's variance error ABOVE the gate, or the wrong scheme passes;
    #   (c) Var(q, T) FAR FROM STATIONARY, or the entry is answerable by returning
    #       the Gibbs variance without integrating anything (section 1 rule 6).
    # (b) wants fast relaxation and strong restoring force; (c) wants slow.
    print(f"\n--- the (gamma, omega) sweep at dt={DT}, 200 000 paths; gate "
          f"SDE_VAR_TOL={SDE_VAR_TOL} ---")
    print("    'x gate' columns are multiples of the gate: EM must exceed 1, BAOAB "
          "must be under 1,\n    and 'from stat' is how far Var(q,T) sits from the "
          "stationary value in gate units.")
    print(f"\n{'gamma':>6} {'omega':>6} {'rho(EM)':>8} {'Var(q,T)':>11} "
          f"{'from stat':>10} {'EM x gate':>10} {'BAOAB x gate':>13}  verdict")
    nt = int(round(T / DT))
    rows = []
    for gamma in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        for omega in (2.0, 3.0, 4.0, 5.0, 8.0):
            _m, cov = exact_quadrature(gamma, omega)
            vq = cov[0, 0]
            e_em = var_err(euler_maruyama(gamma, omega, DT, 200_000, nt), cov)
            e_ba = var_err(baoab(gamma, omega, DT, 200_000, nt), cov)
            rho = em_spectral_radius(gamma, omega)
            from_stat = abs(vq / stationary(omega)[0, 0] - 1.0)
            ok = (rho < 1.0 and e_em > SDE_VAR_TOL and e_ba < SDE_VAR_TOL
                  and from_stat > 3.0 * SDE_VAR_TOL)
            rows.append(dict(gamma=gamma, omega=omega, rho=rho, vq=vq, e_em=e_em,
                             e_ba=e_ba, from_stat=from_stat, ok=ok))
            why = []
            if rho >= 1.0:
                why.append("EM unstable")
            if e_em <= SDE_VAR_TOL:
                why.append("EM passes")
            if from_stat <= 3.0 * SDE_VAR_TOL:
                why.append("near-stationary")
            print(f"{gamma:6.1f} {omega:6.1f} {rho:8.4f} {vq:11.7f} "
                  f"{from_stat / SDE_VAR_TOL:10.2f} {e_em / SDE_VAR_TOL:10.2f} "
                  f"{e_ba / SDE_VAR_TOL:13.4f}  "
                  f"{'ADMISSIBLE' if ok else ', '.join(why)}")

    good = [r for r in rows if r["ok"]]
    best = max(good, key=lambda r: r["e_em"]) if good else None
    print(f"\n{len(good)} of {len(rows)} pairs satisfy all three constraints.")
    if best is not None:
        print(f"CHOSEN -- largest EM error among the admissible: "
              f"gamma={best['gamma']}, omega={best['omega']}")
        print(f"  rho(EM) = {best['rho']:.4f} (stable, so EM is biased and not "
              f"diverging)")
        print(f"  Var(q,T) = {best['vq']:.8f}, {best['from_stat']:.1%} from the "
              f"stationary {stationary(best['omega'])[0, 0]:.8f}")
        print(f"  EM    {best['e_em']:.4f} = {best['e_em'] / SDE_VAR_TOL:.1f}x the gate "
              f"-> FAILS")
        print(f"  BAOAB {best['e_ba']:.5f} = gate/{SDE_VAR_TOL / best['e_ba']:.0f}"
              f" -> CLEARS")
        print(f"  separation {best['e_em'] / best['e_ba']:.0f}x")
    else:
        print("NONE -- the entry cannot be calibrated at this horizon; section 2's "
              "fallback (grade\nthe stationary measure at long T) applies.")

    # --- 3. vacuity --------------------------------------------------------
    print("\n--- vacuity (section 1 rule 6): is Var(q) at T its own number? ---")
    print("    Folded into the sweep above as the 'from stat' column, which is why "
          "the admissible\n    set is so much smaller than the separating set. "
          "Spelled out for the chosen pair:")
    for r in ([best] if best is not None else rows[:1]):
        st = stationary(r["omega"])[0, 0]
        print(f"      gamma={r['gamma']}, omega={r['omega']}: Var(q,0)=0, "
              f"Var(q,T)={r['vq']:.7f}, Var(q,inf)={st:.7f}")
        print(f"      T is {r['from_stat'] / SDE_VAR_TOL:.1f} gate-widths from the "
              f"stationary value, so returning the Gibbs\n      variance fails -- "
              f"which is what rule 6 asks of the entry.")
