"""Seed resolution of the notable suite's accuracy axis  [plan-notable-sde.md §8g].

The suite grades every solver at one fixed seed (42). This script measures what that
costs. Two independent measurements:

  Part 1 -- self-contained, no workspace needed.  The Monte-Carlo noise floor of each
  moment-graded entry at its own path count, and, for N04, the full seed distribution
  of a scheme that is *exact in law*. Both N04 competitors are reimplemented here from
  their graded sources so the headline finding survives `workspace/` being gitignored.

  Part 2 -- re-grades the graded solvers at k seeds, if they are still on disk.
  `python3 -I n03_n04_n06_n07_seed_resolution.py --sweep 16`

Measured 2026-11-20. numpy only. Every figure in §8g comes from here.
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/docs/", 1)[0] + "/benchmark")
import problems as P  # noqa: E402

np.seterr(all="ignore")
PS = {p["id"]: p for p in P.notable_problems()}
MEAN_TOL, VAR_TOL = 0.05, 0.10


def truth(p):
    """[(exact_mean, exact_variance)] per component; drops the truths' own error bars."""
    m = p["moments"](p["T"])
    g = lambda z: z[0] if isinstance(z, tuple) else z  # noqa: E731
    if "mean" in m:
        return [(g(m["mean"]), g(m["variance"]))]
    keys = sorted({k.split("_", 1)[1] for k in m})
    return [(g(m["mean_" + k]), g(m["variance_" + k])) for k in keys]


def frac_of_gate(X, tr):
    """The suite's comparable metric: worst moment as a fraction of *its own* gate.

    A near-zero exact mean is skipped, as `verify.py` skips it.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[1] != len(tr) and X.shape[0] == len(tr):
        X = X.T
    worst = 0.0
    for i, (em, ev) in enumerate(tr):
        c = X[:, i]
        if abs(em) >= 0.01:
            worst = max(worst, abs(c.mean() - em) / max(abs(em), 1e-10) / MEAN_TOL)
        worst = max(worst, abs(c.var(ddof=1) - ev) / max(abs(ev), 1e-10) / VAR_TOL)
    return worst


# --- Part 1a: the noise floor ------------------------------------------------
#
# The relative standard error of a *sample variance* is sqrt((kurtosis - 1) / n): it is
# set by the fourth moment, which is why the variance gate is the loosest of the three.
# The reference sample is a fine-step tamed Milstein, which only has to get the shape
# of the terminal law right, not its moments.

def _n03():
    k = PS["N03"]["moments"].keywords
    pos = lambda x: np.maximum(x, 1e-10)  # noqa: E731
    return (lambda x: k["a_m1"] / pos(x) - k["a_0"] + k["a_1"] * x - k["a_2"] * pos(x) ** k["rho"],
            lambda x: k["sigma"] * pos(x) ** k["theta"], k["X0"], (1e-10, np.inf))


def _n04():
    k = PS["N04"]["moments"].keywords
    return (lambda x: k["a"] - (k["a"] + k["b"]) * x,
            lambda x: k["sigma"] * np.sqrt(np.clip(x * (1 - x), 0, None)), k["X0"], (0.0, 1.0))


def _n06():
    k = PS["N06"]["moments"].keywords
    N = k["N_pop"]
    return (lambda x: x * ((k["beta"] * N - k["mu"] - k["gamma"]) - k["beta"] * x),
            lambda x: k["sigma"] * x * (N - x), k["I0"], (0.0, N))


SCALAR = {"N03": _n03, "N04": _n04, "N06": _n06}


def _tamed_milstein(b, s, X0, T, dt, n, seed, sub, dom):
    rng = np.random.default_rng(seed)
    h, lo, hi = dt / sub, *dom
    x = np.full(n, float(X0))
    e = 1e-7 * max(1.0, abs(hi) if np.isfinite(hi) else abs(X0))
    for _ in range(int(round(T / h))):
        dW = rng.standard_normal(n) * np.sqrt(h)
        bx, sx = b(x), s(x)
        sp = (s(x + e) - s(x - e)) / (2 * e)
        x = np.clip(x + bx * h / (1.0 + h * np.abs(bx)) + sx * dW + 0.5 * sx * sp * (dW**2 - h), lo, hi)
    return x


def noise_floor():
    print("Part 1a -- the Monte-Carlo noise floor at each entry's own path count")
    print(f"  {'':5s} {'paths':>7s} {'kurt':>6s} {'relSE(mean)':>12s} {'relSE(var)':>11s} "
          f"{'key err (var)':>14s}")
    for pid in ("N03", "N04", "N06"):
        p = PS[pid]
        n = int(p["num_paths"])
        b, s, X0, dom = SCALAR[pid]()
        ref = _tamed_milstein(b, s, X0, p["T"], p["dt"], 80000, 7, 32, dom)
        c = ref - ref.mean()
        kurt = float((c**4).mean() / (c**2).mean() ** 2)
        m = p["moments"](p["T"])
        ev = m["variance"][0] if isinstance(m["variance"], tuple) else m["variance"]
        key = (m["variance"][1] / abs(ev)) if isinstance(m["variance"], tuple) else 0.0
        print(f"  {pid:5s} {n:7d} {kurt:6.2f} {ref.std(ddof=1)/abs(ref.mean())/np.sqrt(n)*100:11.4f}% "
              f"{np.sqrt(max(kurt-1,0)/n)*100:10.4f}% {key*100:13.4f}%")
    print("  (the variance's own estimator is the noisier one; its gate is 10%, the mean's 5%)\n")


# --- Part 1b: N04, where both competitors are bias-free ----------------------
#
# At a = b = sigma^2/4 the Lamperti transform Y = 2 arcsin(sqrt X) makes the drift vanish
# *identically*, so Y is a reflected Brownian motion on [0, pi] and the terminal law is the
# triangle fold of a Gaussian. Both graded solvers are reimplemented below.

_FOLD = lambda z: np.abs(np.mod(z + np.pi, 2.0 * np.pi) - np.pi)  # noqa: E731


def n04_autonumerics(n, dt, T, seed):
    """`1-lamperti-reflected-euler`: EM in Y with unit additive noise, folded each step."""
    Nt = max(1, round(T / dt))
    dW = np.sqrt(T / Nt) * np.random.default_rng(seed).standard_normal((n, Nt))
    Y = np.full(n, 2.0 * np.arcsin(np.sqrt(0.05)))
    for k in range(Nt):                      # the graded solver also adds a drift term that
        Y = _FOLD(Y + dW[:, k])              # is analytically zero (mu_y ~ 1e-17); omitted
    return np.sin(0.5 * Y) ** 2              # here because forcing it to zero changes nothing


def n04_exact_in_law(n, dt, T, seed):
    """The same scheme folded ONCE: exact in law, since R_t = fold(B_t)."""
    Nt = max(1, round(T / dt))
    dW = np.sqrt(T / Nt) * np.random.default_rng(seed).standard_normal((n, Nt))
    return np.sin(0.5 * _FOLD(2.0 * np.arcsin(np.sqrt(0.05)) + dW.sum(1))) ** 2


def n04_opus(n, dt, T, seed):
    """C0's moment-matched Beta step: each step is drawn from a Beta carrying the EXACT
    conditional mean and variance over dt. The generator maps polynomials to polynomials,
    so those two conditional moments are exactly affine and quadratic in X_n; by the tower
    property the two graded moments then carry no discretization bias at any dt."""
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    h = T / Nt
    X = np.full(n, 0.05)
    e1, e2 = np.exp(-0.5 * h), np.exp(-2.0 * h)
    for _ in range(Nt):
        mu = 0.5 + (X - 0.5) * e1
        B = X - 0.5
        v = (0.375 + B * e1 + (X * X - 0.375 - B) * e2) - mu * mu
        mu = np.clip(mu, 1e-14, 1.0 - 1e-14)
        vmax = mu * (1.0 - mu)
        nu = vmax / np.clip(v, 1e-300, vmax * (1.0 - 1e-12)) - 1.0
        Xn = rng.beta(np.maximum(mu * nu, 1e-300), np.maximum((1.0 - mu) * nu, 1e-300))
        X = np.clip(np.where(np.isfinite(Xn), Xn, mu), 0.0, 1.0)
    return X


def n04_bias(K=20):
    p = PS["N04"]
    T, dt, n, tr = p["T"], p["dt"], int(p["num_paths"]), truth(p)
    em, ev = tr[0]
    print(f"Part 1b -- N04: is either side actually biased?  (n={n}, dt={dt}, {K} seeds)")
    for lab, f in (("AutoNumerics (Lamperti fold)", n04_autonumerics),
                   ("Opus C0 (moment-matched Beta)", n04_opus),
                   ("fold once (exact in law)", n04_exact_in_law)):
        bm, bv, fr = [], [], []
        for s in range(K):
            X = f(n, dt, T, s)
            bm.append((X.mean() - em) / em)
            bv.append((X.var(ddof=1) - ev) / ev)
            fr.append(frac_of_gate(X, tr))
        bm, bv, fr = np.array(bm), np.array(bv), np.array(fr)
        se = lambda a: a.std(ddof=1) / np.sqrt(len(a)) * 100  # noqa: E731
        print(f"  {lab:30s} mean bias {bm.mean()*100:+7.4f}% +/- {se(bm):.4f}   "
              f"var bias {bv.mean()*100:+7.4f}% +/- {se(bv):.4f}   "
              f"frac median {np.median(fr):.4f} range [{fr.min():.4f},{fr.max():.4f}]")
    print("  no bias here is distinguishable from zero: on N04 BOTH sides are exact in the")
    print("  two graded moments, and the suite's 7.5x reported gap is the seed-42 draw.\n")


def n04_exact_distribution(K=400):
    """Where the three recorded N04 points sit in an exact scheme's own seed distribution."""
    p = PS["N04"]
    T, n, tr = p["T"], int(p["num_paths"]), truth(p)
    rng = np.random.default_rng(20261120)
    Y0 = 2 * np.arcsin(np.sqrt(0.05))
    fr = np.array([frac_of_gate(np.sin(0.5 * _FOLD(Y0 + rng.standard_normal(n) * np.sqrt(T))) ** 2, tr)
                   for _ in range(K)])
    print(f"Part 1c -- an exact-in-law scheme's own seed distribution ({K} seeds, n={n})")
    qs = [5, 25, 50, 75, 95]
    print("  percentiles " + "  ".join(f"{q}%={np.percentile(fr,q):.4f}" for q in qs)
          + f"  max={fr.max():.4f}")
    for lab, v in (("Opus C0 (recorded)", 0.0169), ("AutoNumerics rep1=rep2", 0.1262),
                   ("AutoNumerics pre-fix", 0.3603)):
        print(f"    {lab:24s} {v:.4f} -> {(fr > v).mean()*100:5.1f}% of exact-scheme seeds do worse")
    print("  the pre-fix winner is outside 400/400: that bias was real, and the fix removed it.\n")


# --- Part 2: re-grade the graded solvers at k seeds --------------------------

WINNERS = {"N03": ("sde_aitsahalia_rate", "1-lamperti-trapezoidal"),
           "N04": ("sde_wright_fisher_mutation", "1-lamperti-reflected-euler"),
           "N06": ("sde_sis_epidemic", "2-split-step-implicit-milstein"),
           "N07": ("sde_langevin_baoab", "2-stochastic-heun")}
RECORDED = {"N03": (0.0442, 0.0124), "N04": (0.1262, 0.0169),
            "N06": (0.1306, 0.0550), "N07": (0.1622, 0.1086)}


def sweep(K=16):
    import importlib.util
    import os
    root = __file__.rsplit("/docs/", 1)[0]

    def load(path, name):
        if not os.path.exists(path):
            return None
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    print(f"Part 2 -- the graded solvers re-graded at {K} seeds")
    print(f"  {'':4s} | {'seed 42 (the suite)':>26s} | {'median over seeds':>26s} | "
          f"{'AN avg|bias| mean/var':>22s}")
    for pid, (slug, plan) in WINNERS.items():
        p = PS[pid]
        T, dt, n, tr = p["T"], p["dt"], int(p["num_paths"]), truth(p)
        mods = {"AN": load(f"{root}/workspace/{slug}/plans/{plan}/solver.py", f"an_{pid}"),
                "OP": load(f"{root}/workspace_oneshot/opus/{slug}/c0-naked/rep1/solver.py", f"op_{pid}")}
        if not all(mods.values()):
            print(f"  {pid}: solver sources not on disk (workspace/ is gitignored) -- "
                  f"recorded seed 42: {RECORDED[pid][0]:.4f} vs {RECORDED[pid][1]:.4f}")
            continue
        res = {}
        for side, mod in mods.items():
            fr, bm, bv = [], [], []
            for s in list(range(K)) + [42]:
                try:
                    r = mod.solve_sde(n, dt, T, seed=s)
                except TypeError:
                    r = mod.solve_sde(n, dt, T)
                X = np.asarray(r["terminal_paths"], dtype=float)
                if s == 42:
                    res[side + "42"] = frac_of_gate(X, tr)
                    continue
                fr.append(frac_of_gate(X, tr))
                Xc = X[:, None] if X.ndim == 1 else (X.T if X.shape[1] != len(tr) else X)
                bm.append(max(abs((Xc[:, i].mean() - em) / max(abs(em), 1e-10)) for i, (em, _) in enumerate(tr)))
                bv.append(max(abs((Xc[:, i].var(ddof=1) - ev) / max(abs(ev), 1e-10)) for i, (_, ev) in enumerate(tr)))
            res[side] = (np.array(fr), np.mean(bm), np.mean(bv))
        a, o = res["AN"], res["OP"]
        ok = ("ok" if abs(res["AN42"] - RECORDED[pid][0]) < 5e-4
              and abs(res["OP42"] - RECORDED[pid][1]) < 5e-4 else "MISMATCH")
        # p90/p10, not max/min: a range statistic grows with k and is not comparable
        # across sweep sizes, which is how a k=8 sweep misread N03's reversal as a tie.
        sp = lambda f: np.percentile(f, 90) / np.percentile(f, 10)  # noqa: E731
        print(f"  {pid:4s} | AN {res['AN42']:.4f}  Opus {res['OP42']:.4f}  "
              f"{res['AN42']/res['OP42']:4.1f}x | AN {np.median(a[0]):.4f}  Opus {np.median(o[0]):.4f}  "
              f"{np.median(a[0])/np.median(o[0]):4.2f}x | {a[1]*100:6.3f}% {a[2]*100:6.3f}%  "
              f"p90/p10 {sp(a[0]):5.1f}x vs {sp(o[0]):5.1f}x  [{ok}]")
    print("  a wide p90/p10 is a noise-limited solver; near 2x is a bias-limited one.")
    print("  medians are stable from about k=16; k=8 is too few (it misreads N03 as a tie).")


if __name__ == "__main__":
    noise_floor()
    n04_bias()
    n04_exact_distribution()
    if "--sweep" in sys.argv:
        sweep(int(sys.argv[sys.argv.index("--sweep") + 1]))
