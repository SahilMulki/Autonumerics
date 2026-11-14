"""N05 truth derivation: Merton jump diffusion.

    dX = (mu - lambda*kappa) X dt + sigma X dW + X dJ

`J` compound Poisson, intensity `lambda`, multiplicative jumps `Y = e^Z` with
`Z ~ N(m, delta^2)`, and `kappa = E[Y] - 1 = e^{m + delta^2/2} - 1` the compensator.

Two truths, both exact, and the second by three independent routes:

  * **terminal moments**, in closed form from
    `X(T) = X0 exp((mu - lambda kappa - sigma^2/2)T + sigma W(T)) * prod Y_i`;
  * **the European call**, by Merton's Poisson-weighted Black-Scholes series, checked
    against (i) a direct quadrature of the lognormal-mixture density -- which uses no
    Black-Scholes formula at all -- and (ii) Monte Carlo of the exact solution. Two
    method families plus a sampler, which is the discipline `benchop.py` applies to a
    published value.

Also measured here: the **one-decision-wrong control** (the compensator dropped), and
the **vacuity finding** that makes this entry's gate the second moment and the price
rather than the mean.

Seeded, numpy + scipy only. Run:
`uv run python docs/notable-sde-data/n05_merton.py`
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

# A parameter set in wide use in the jump-diffusion numerics literature (the
# d'Halluin-Forsyth-Vetzal test case). `[to confirm against the source in step 6]`:
# this script DERIVES the price rather than quoting one, so a mis-transcribed
# parameter would show up as a disagreement with no published number, not as a wrong
# answer that agrees with itself.
S0 = 100.0
K = 100.0
R = 0.05
SIGMA = 0.15
T = 0.25
LAM = 0.1          # jump intensity
M_J = -0.90        # mean of the LOG jump size
D_J = 0.45         # sd of the LOG jump size
SEED = 42

KAPPA = np.exp(M_J + 0.5 * D_J ** 2) - 1.0


# --- exact terminal moments -------------------------------------------------

def exact_moments(mu=R, T=T):
    """E[X_T] and Var[X_T], exact.

    E[prod Y_i] = exp(lambda T kappa) by conditioning on the Poisson count, so the
    compensator cancels exactly and E[X_T] = X0 e^{mu T} -- the fact that makes the
    mean vacuous below. The second moment keeps E[Y^2] = e^{2m + 2 delta^2}.
    """
    m1 = S0 * np.exp(mu * T)
    e_y2 = np.exp(2.0 * M_J + 2.0 * D_J ** 2)
    m2 = (S0 ** 2 * np.exp(2.0 * (mu - LAM * KAPPA) * T + SIGMA ** 2 * T)
          * np.exp(LAM * T * (e_y2 - 1.0)))
    return m1, m2 - m1 ** 2


# --- route 1: Merton's series -----------------------------------------------

def bs_call(s, k, t, r, vol):
    if vol <= 0 or t <= 0:
        return max(s - k * np.exp(-r * t), 0.0)
    v = vol * np.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * vol ** 2) * t) / v
    return s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d1 - v)


def poisson_weights(n_terms, lam=None, t=None):
    """`e^{-lam t} (lam t)^n / n!`, built by a log recursion so no factorial
    overflows and the weights stay exact at large `n`."""
    lam = LAM if lam is None else lam
    t = T if t is None else t
    out, logw = [], -lam * t
    for n in range(n_terms):
        if n:
            logw += np.log(lam * t) - np.log(n)
        out.append(np.exp(logw))
    return np.array(out)


def merton_series(n_terms=80, lam=None, t=None, compensate=True):
    """Poisson-weighted Black-Scholes: conditional on `n` jumps the terminal law is
    lognormal again, with variance `sigma^2 + n delta^2 / T` and a drift shifted by
    the jumps' own mean.

    `compensate=False` is the one-decision-wrong control -- the `- lambda kappa`
    term omitted from the drift.
    """
    lam = LAM if lam is None else lam
    t = T if t is None else t
    kappa = np.exp(M_J + 0.5 * D_J ** 2) - 1.0
    drift_adj = -lam * kappa if compensate else 0.0
    total = 0.0
    for n, w in enumerate(poisson_weights(n_terms, lam, t)):
        vol_n = np.sqrt(SIGMA ** 2 + n * D_J ** 2 / t)
        r_n = R + drift_adj + n * (M_J + 0.5 * D_J ** 2) / t
        total += w * bs_call(S0 * np.exp((r_n - R) * t), K, t, R, vol_n)
    return total


# --- route 2: quadrature of the lognormal mixture (no BS formula) -----------

def mixture_quadrature(n_terms=60):
    """E[(X_T - K)^+] integrated directly against the mixture density.

    Conditional on `n` jumps, `ln X_T` is normal with mean `a_n` and sd `b_n`, so the
    payoff integral is a 1-D quadrature over the log-price. Nothing here evaluates a
    Black-Scholes formula, so a sign error in `bs_call` or in the drift adjustment
    cannot be reproduced.
    """
    total = 0.0
    log_w = -LAM * T
    for n in range(n_terms):
        if n:
            log_w += np.log(LAM * T) - np.log(n)
        w = np.exp(log_w)
        if w < 1e-18 and n > 5:
            continue
        a = (np.log(S0) + (R - LAM * KAPPA - 0.5 * SIGMA ** 2) * T + n * M_J)
        b = np.sqrt(SIGMA ** 2 * T + n * D_J ** 2)
        lo, hi = a - 12.0 * b, a + 12.0 * b
        val, _err = quad(lambda y, a=a, b=b: (np.exp(y) - K)
                         * np.exp(-0.5 * ((y - a) / b) ** 2) / (b * np.sqrt(2 * np.pi)),
                         max(np.log(K), lo), hi, limit=400)
        total += w * val
    return np.exp(-R * T) * total


# --- route 3: Monte Carlo of the exact solution -----------------------------

def mc_exact(n=4_000_000, seed=SEED):
    """No scheme: the exact terminal law, sampled. Poisson count, then the sum of
    that many normal log-jumps."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n) * np.sqrt(T)
    nj = rng.poisson(LAM * T, n)
    jump = rng.standard_normal(n) * (D_J * np.sqrt(np.maximum(nj, 0))) + nj * M_J
    x = S0 * np.exp((R - LAM * KAPPA - 0.5 * SIGMA ** 2) * T + SIGMA * w + jump)
    pay = np.maximum(x - K, 0.0)
    price = float(np.exp(-R * T) * pay.mean())
    se = float(np.exp(-R * T) * pay.std(ddof=1) / np.sqrt(n))
    return price, se, float(x.mean()), float(x.var(ddof=1))


if __name__ == "__main__":
    print(f"Merton jump diffusion: S0={S0}, K={K}, r={R}, sigma={SIGMA}, T={T}, "
          f"lambda={LAM}, m={M_J}, delta={D_J}")
    print(f"compensator kappa = e^(m + d^2/2) - 1 = {KAPPA:.10f}\n")

    m1, v1 = exact_moments()
    print("--- terminal moments (exact) ---")
    print(f"  E[X_T]   = {m1:.10f}   (= S0 e^{{rT}}, independent of lambda, m, "
          f"delta and sigma)")
    print(f"  Var[X_T] = {v1:.10f}")

    print("\n--- the European call, three routes ---")
    p_series = merton_series()
    p_quad = mixture_quadrature()
    p_mc, se_mc, mc_m1, mc_v1 = mc_exact()
    print(f"  Merton series        {p_series:.10f}")
    print(f"  mixture quadrature   {p_quad:.10f}   |diff| = {abs(p_series - p_quad):.2e}")
    print(f"  Monte Carlo (exact)  {p_mc:.6f} +- {se_mc:.6f}   "
          f"({abs(p_series - p_mc) / se_mc:.2f} sigma from the series)")
    print(f"  moments by MC: mean {mc_m1:.6f} vs {m1:.6f}, "
          f"var {mc_v1:.6f} vs {v1:.6f}")

    print("\n--- the controls, measured on every gate ---")
    print("  The plan proposed ONE control (the compensator dropped) and proposed to "
          "judge it on\n  the MEAN. Both halves of that turn out wrong: the mean is "
          "vacuous (below), and at this\n  parameter set the control is far too small "
          "to fail a 5% gate. Measured:")
    p_nocomp = merton_series(compensate=False)
    p_nojump = bs_call(S0, K, T, R, SIGMA)          # jumps omitted entirely
    factor = np.exp(-LAM * KAPPA * T)
    print(f"\n  {'control':<34} {'mean':>10} {'price':>12} {'price err':>11}")
    print(f"  {'compensator dropped':<34} {abs(factor - 1.0):10.3%} "
          f"{p_nocomp:12.6f} {abs(p_nocomp / p_series - 1):11.3%}")
    print(f"  {'jump channel omitted (pure BS)':<34} {'0.000%':>10} "
          f"{p_nojump:12.6f} {abs(p_nojump / p_series - 1):11.3%}")
    print(f"\n  the 5% mean gate: compensator control is "
          f"{abs(factor - 1.0) / 0.05:.2f}x -> PASSES, i.e. NOT A CONTROL")
    print(f"  a 1% functional gate: {abs(p_nocomp / p_series - 1) / 0.01:.1f}x and "
          f"{abs(p_nojump / p_series - 1) / 0.01:.1f}x -> both FAIL")

    print("\n  the same two controls at a longer horizon and a busier jump channel")
    print("  (lambda=1, T=1), for comparison -- jumps are rare here: lambda*T = "
          f"{LAM * T:.3f}, so only\n  {100 * (1 - np.exp(-LAM * T)):.1f}% of paths "
          "jump at all:")
    for lam_x, t_x in ((LAM, T), (1.0, 1.0)):
        base = merton_series(lam=lam_x, t=t_x)
        nc = merton_series(lam=lam_x, t=t_x, compensate=False)
        nj = bs_call(S0, K, t_x, R, SIGMA)
        kap = np.exp(M_J + 0.5 * D_J ** 2) - 1.0
        mean_err = abs(np.exp(-lam_x * kap * t_x) - 1.0)
        print(f"    lambda={lam_x:<4} T={t_x:<4} price {base:8.5f} | "
              f"mean err {mean_err:7.2%} ({mean_err / 0.05:5.2f}x gate) | "
              f"no-comp {abs(nc / base - 1):7.2%} | no-jump {abs(nj / base - 1):7.2%}")

    print("\n--- vacuity (section 1 rule 6) ---")
    print("  E[X_T] = S0 e^{rT} exactly, for every lambda, m, delta and sigma, so the")
    print("  MEAN ALONE IS ANSWERABLE WITHOUT SIMULATING A SINGLE JUMP. Perturbing "
          "each\n  parameter by 10% and recomputing:")
    base_m1, base_v1 = exact_moments()
    for name in ("LAM", "M_J", "D_J", "SIGMA"):
        old = globals()[name]
        globals()[name] = old * 1.1
        KAPPA_new = np.exp(globals()["M_J"] + 0.5 * globals()["D_J"] ** 2) - 1.0
        g = globals()
        g["KAPPA"] = KAPPA_new
        m, v = exact_moments()
        print(f"    {name:>6} +10%:  mean {abs(m / base_m1 - 1):.2e}   "
              f"variance {abs(v / base_v1 - 1):.4f}")
        globals()[name] = old
        g["KAPPA"] = np.exp(M_J + 0.5 * D_J ** 2) - 1.0
    print("  => the mean is recorded and the gates that decide this entry are the")
    print("     second moment and the price.")
