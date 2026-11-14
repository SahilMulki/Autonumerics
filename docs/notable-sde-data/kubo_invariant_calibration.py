"""H-N3 calibration: the conserved-amplitude gate on the Kubo oscillator (N02).

Measures the quantity `problems._sde_diag_conserved_norm` gates -- the RMS relative
drift of |X(T)| across paths -- for every scheme a plan might plausibly propose, at
**exactly** the configuration N02 ships with and the step the verifier re-runs at
(`verify.sde_verify_dt` = min(dt, 0.04/T) = 0.04).

The repo's rule (benchmark/README.md): a correct scheme must clear the gate at the
resolution the entry ships with, and a plausible wrong one must fail it, the two
differing by exactly one decision. Here that decision is the Ito-Stratonovich
conversion. The gate is set from this output and then not moved.

Stratonovich system, a = sigma = 1, X(0) = (1, 0), T = 1:

    dX1 = -a X2 dt - sigma X2 o dW
    dX2 =  a X1 dt + sigma X1 o dW

Exact flow: a rotation by theta(t) = a t + sigma W(t), so |X(t)| = |X(0)| on every
path at every time, exactly.

Run: `uv run python docs/notable-sde-data/kubo_invariant_calibration.py`
"""
import numpy as np

A = SIG = 1.0
T = 1.0
DT = 0.04            # verify.sde_verify_dt(N02) = min(0.04, 0.04/1.0)
X0 = np.array([1.0, 0.0])
N_PATHS = 50_000
SEED = 42


def increments(n_paths, nt, seed=SEED):
    return np.random.default_rng(seed).standard_normal((n_paths, nt)) * np.sqrt(DT)


def _rot(theta, x0):
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([x0[0] * c - x0[1] * s, x0[0] * s + x0[1] * c], axis=1)


def exact_rotation(dW):
    """Exact sampling: X(T) = R(aT + sigma W(T)) X(0). No discretization error."""
    return _rot(A * T + SIG * dW.sum(axis=1), X0)


def implicit_midpoint(dW):
    """Stratonovich implicit midpoint -- a Cayley rotation, orthogonal by
    construction, so it conserves |X| to round-off at any step."""
    x = np.tile(X0, (dW.shape[0], 1))
    for k in range(dW.shape[1]):
        h = A * DT + SIG * dW[:, k]          # the rotation angle increment
        # (I - K/2)^{-1} (I + K/2) with K = [[0,-h],[h,0]]: the Cayley transform.
        den = 1.0 + h * h / 4.0
        c = (1.0 - h * h / 4.0) / den
        s = h / den
        x = np.stack([x[:, 0] * c - x[:, 1] * s, x[:, 0] * s + x[:, 1] * c], axis=1)
    return x


def stratonovich_heun(dW):
    """Explicit Stratonovich Heun (predictor-corrector) -- converges to the right
    equation, conserves |X| only to O(dt)."""
    x = np.tile(X0, (dW.shape[0], 1))
    for k in range(dW.shape[1]):
        h = A * DT + SIG * dW[:, k]
        p0 = np.stack([-x[:, 1] * h, x[:, 0] * h], axis=1)          # drift+noise, Strat form
        xp = x + p0
        p1 = np.stack([-xp[:, 1] * h, xp[:, 0] * h], axis=1)
        x = x + 0.5 * (p0 + p1)
    return x


def ito_euler_converted(dW):
    """CORRECT by one reading: convert to Ito (the -sigma^2/2 X term) and step with
    Euler-Maruyama. Right equation, first-order weak -- but no conservation."""
    x = np.tile(X0, (dW.shape[0], 1))
    for k in range(dW.shape[1]):
        dw = dW[:, k]
        x1, x2 = x[:, 0], x[:, 1]
        nx1 = x1 + (-A * x2 - 0.5 * SIG ** 2 * x1) * DT - SIG * x2 * dw
        nx2 = x2 + (A * x1 - 0.5 * SIG ** 2 * x2) * DT + SIG * x1 * dw
        x = np.stack([nx1, nx2], axis=1)
    return x


def ito_euler_misread(dW):
    """WRONG by exactly one decision: the same Euler step with the conversion term
    omitted -- i.e. the Stratonovich coefficients read as Ito ones. Solves a
    different equation, whose second moment grows as e^{sigma^2 t}."""
    x = np.tile(X0, (dW.shape[0], 1))
    for k in range(dW.shape[1]):
        dw = dW[:, k]
        x1, x2 = x[:, 0], x[:, 1]
        nx1 = x1 + (-A * x2) * DT - SIG * x2 * dw
        nx2 = x2 + (A * x1) * DT + SIG * x1 * dw
        x = np.stack([nx1, nx2], axis=1)
    return x


def mean_sq_drift(x):
    """**The gated quantity**: | E|X(T)|^2 / |X(0)|^2 - 1 |.

    The exact flow conserves |X| pathwise, so it conserves E|X|^2 too, and this
    statistic isolates the *systematic* part of a violation. That is what makes it
    the sharp one: a correct but non-conservative scheme (Heun, Euler) has a genuine
    O(dt) per-path wobble whose RMS is large and whose mean is small, so gating the
    per-path RMS charges a correct scheme for its own variance. Measured below: 4.7x
    separation on the RMS against 34.5x on this.
    """
    r2 = (x ** 2).sum(axis=1) / float(np.dot(X0, X0))
    return float(abs(np.mean(r2) - 1.0))


def drift_rms(x):
    """Reported beside the gate, never gated: RMS over paths of | |X(T)|/|X(0)| - 1 |.

    Per-path rather than in the mean, so it sees a scheme whose amplitude wanders
    symmetrically -- which the gated statistic cannot. RMS rather than the max
    because the max over `n` paths of an O(dt) random drift grows with `n`, making
    the number depend on the path count instead of the scheme.
    """
    r = np.linalg.norm(x, axis=1) / np.linalg.norm(X0)
    return float(np.sqrt(np.mean((r - 1.0) ** 2)))


SCHEMES = [
    ("exact rotation           [correct]", exact_rotation),
    ("implicit midpoint        [correct]", implicit_midpoint),
    ("Stratonovich Heun        [correct]", stratonovich_heun),
    ("Ito-Euler + conversion   [correct]", ito_euler_converted),
    ("Ito-Euler, misread       [WRONG  ]", ito_euler_misread),
]

if __name__ == "__main__":
    nt = int(round(T / DT))
    dW = increments(N_PATHS, nt)
    print(f"Kubo oscillator, a=sigma=1, X0=(1,0), T={T}, dt={DT} ({nt} steps), "
          f"{N_PATHS} paths, seed {SEED}")
    print(f"{'scheme':<36} {'GATED |E|X|^2-1|':>17} {'(RMS per-path)':>16} "
          f"{'E|X(T)|^2':>11}")
    gated, rms = {}, {}
    for name, fn in SCHEMES:
        x = fn(dW)
        gated[name], rms[name] = mean_sq_drift(x), drift_rms(x)
        print(f"{name:<36} {gated[name]:17.3e} {rms[name]:16.3e} "
              f"{float((x ** 2).sum(1).mean()):11.4f}")

    for label, rows in (("GATED  |E|X|^2/|X0|^2 - 1|", gated),
                        ("report  RMS per-path drift", rms)):
        worst_correct = max(v for k, v in rows.items() if "correct" in k)
        wrong = [v for k, v in rows.items() if "WRONG" in k][0]
        print(f"\n{label}")
        print(f"  worst correct scheme : {worst_correct:.4e}")
        print(f"  wrong scheme         : {wrong:.4e}")
        print(f"  separation           : {wrong / worst_correct:.1f}x")
        print(f"  geometric-mean gate  : {np.sqrt(worst_correct * wrong):.4e}")
