import numpy as np

KAPPA = 0.5
THETA = 0.04
SIGMA_V = 1.0
V0 = 0.04
A_R = 0.08
B_R = 0.10
SIGMA_R = 0.09
R0 = 0.10
RHO = -0.9
K_STRIKE = 100.0
SPOTS = (75.0, 100.0, 125.0)


def _ncdf(x):
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * ax)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
               + t * (-1.821255978 + t * 1.330274429))))
    pdf = np.exp(-0.5 * ax * ax) * 0.3989422804014327
    c = 1.0 - pdf * poly
    return np.where(x >= 0, c, 1.0 - c)


def _hw_moments(T):
    a, s, th = A_R, SIGMA_R, B_R
    e1 = np.exp(-a * T)
    e2 = np.exp(-2 * a * T)
    mR = th + (R0 - th) * e1
    mI = th * T + (R0 - th) * (1 - e1) / a
    vR = s * s * (1 - e2) / (2 * a)
    vI = s * s / (a * a) * (T - 2 * (1 - e1) / a + (1 - e2) / (2 * a))
    cRI = s * s / a * ((1 - e1) / a - (1 - e2) / (2 * a))
    return mR, mI, vR, vI, cRI


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, int(round(T / dt)))
    dt = T / Nt

    ek = np.exp(-KAPPA * dt)
    c1 = SIGMA_V ** 2 * ek * (1 - ek) / KAPPA
    c2 = THETA * SIGMA_V ** 2 * (1 - ek) ** 2 / (2 * KAPPA)
    psi_c = 1.5

    mR, mI, vR, vI, cRI = _hw_moments(T)
    sR = np.sqrt(vR)
    cor = cRI / (sR * np.sqrt(vI))
    sI = np.sqrt(vI)

    chunk = 200000
    out = np.empty((num_paths, 3))
    sums = {s0: 0.0 for s0 in SPOTS}
    sq = np.sqrt(1 - RHO ** 2)

    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        V = np.full(n, V0)
        IV = np.zeros(n)
        Vn = np.empty(n)
        for _ in range(Nt):
            m = THETA + (V - THETA) * ek
            s2 = V * c1 + c2
            psi = s2 / (m * m)
            Z = rng.standard_normal(n)
            quad = psi <= psi_c
            iq = np.nonzero(quad)[0]
            ie = np.nonzero(~quad)[0]
            if iq.size:
                pq = psi[iq]
                inv = 2.0 / pq
                b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
                aq = m[iq] / (1.0 + b2)
                Vn[iq] = aq * (np.sqrt(b2) + Z[iq]) ** 2
            if ie.size:
                pe = psi[ie]
                p = (pe - 1.0) / (pe + 1.0)
                beta = (1.0 - p) / m[ie]
                U = rng.random(ie.size)
                val = np.where(U <= p, 0.0,
                               np.log((1.0 - p) / np.maximum(1.0 - U, 1e-300)) / beta)
                Vn[ie] = np.maximum(val, 0.0)
            IV += 0.5 * (V + Vn) * dt
            V, Vn = Vn, V

        # Hull-White: exact joint (R_T, integral R)
        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)
        RT = mR + sR * z1
        IR = mI + sI * (cor * z1 + np.sqrt(1 - cor ** 2) * z2)

        M = (V - V0 - KAPPA * THETA * T + KAPPA * IV) / SIGMA_V
        sd = sq * np.sqrt(np.maximum(IV, 0.0))
        sdc = np.maximum(sd, 1e-12)
        g = np.exp(RHO * M - 0.5 * RHO ** 2 * IV)
        disc = np.exp(-IR)
        for s0 in SPOTS:
            F = s0 * np.exp(IR) * g
            d1 = (np.log(F / K_STRIKE) + 0.5 * sdc * sdc) / sdc
            d2 = d1 - sdc
            val = s0 * g * _ncdf(d1) - K_STRIKE * disc * _ncdf(d2)
            sums[s0] += float(np.sum(val))

        Zs = rng.standard_normal(n)
        ST = 100.0 * np.exp(IR - 0.5 * IV + RHO * M + sd * Zs)
        out[start:start + n, 0] = ST
        out[start:start + n, 1] = V
        out[start:start + n, 2] = RT
        start += n

    functionals = {
        "price_S75": sums[75.0] / num_paths,
        "price_S100": sums[100.0] / num_paths,
        "price_S125": sums[125.0] / num_paths,
    }
    return {
        "terminal_paths": out,
        "empirical_mean": [float(x) for x in out.mean(axis=0)],
        "empirical_variance": [float(x) for x in out.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    r = solve_sde(100000, 0.004, 10, 42)
    print(r["functionals"], r["empirical_mean"])
