# Quadratic Local-Stochastic-Volatility Double-No-Touch

Simulate the system (Ito) with a zero interest rate:

    dS = sqrt(V) * f(S) dW1,                     S(0) = S0,   f(s) = 0.01 * s^2
    dV = 2.58 * (0.043 - V) dt + sqrt(V) dW2,    V(0) = 0.114

with corr(dW1, dW2) = -0.36, over the interval [0, 1].

This is the Heston variance driving a **local** volatility that is quadratic in
the asset: f(100) = 100, so the model is calibrated to agree with
f(s) = s at S = 100 and to differ increasingly away from it. Two consequences:

- The diffusion coefficient grows faster than linearly in S. A scheme that
  works in log space for a linear local volatility is not doing the same thing
  here, and the obvious explicit step in the natural transformed variable can
  overshoot that variable's boundary and return a negative asset price.
- **The Feller condition fails** for the variance: 2*kappa*eta = 0.221880
  against sigma^2 = 1.

The contract pays **1 at T if S has stayed strictly inside the corridor
(50, 150) for the whole interval**, and 0 if it ever touched or crossed either
barrier. Monitoring is **continuous in time, not once per step**: a path that
leaves the corridor between two time points and returns has still knocked out.
Deciding the barrier only at grid points overstates survival, and that error
shrinks only as sqrt(dt).

The rate is zero, so no discounting is applied and the value is exactly the
survival probability.

Report that probability from each of the three initial spots

    S0 = 75,   S0 = 100,   S0 = 125

as `no_touch_S75`, `no_touch_S100` and `no_touch_S125` in the `functionals`
dict. The variance process does not depend on S, so one set of variance paths
serves all three spots.

Return the terminal states (S, V) for **S0 = 100** in `terminal_paths`, using
the simulated value of S at T whether or not that path knocked out.


---

## Solver contract (how this problem is run and scored)

Implement, in `solver.py`, exactly this function:

    def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict

It is imported and called **twice**, with these exact arguments:

    solve_sde(num_paths=400000, dt=0.002, T=1, seed=42)

and then again with `num_paths=100000`, everything else unchanged (see **Two path counts** below).

Use the `dt` you are given (do not hardcode a different step). It must return a dict with:

- `terminal_paths`: 2-D numpy array of shape `(num_paths, 2)` — the 2 state components (X, Y) at time T, in that column order
- `empirical_mean`: list of floats, one per component
- `empirical_variance`: list of floats, one per component (use `ddof=1`)
- `functionals`: a dict of scalar results, containing `"no_touch_S75"`, `"no_touch_S100"`, `"no_touch_S125"` — each a float

Requirements:
- Seed all randomness with `np.random.default_rng(seed)` (not `np.random.seed`).
- Round the step count: `Nt = max(1, round(T / dt))`, then `dt = T / Nt`.
- Keep `solve_sde` importable — no work at module load outside `if __name__ == "__main__"`.
- Use the Python standard library and numpy only.

Scoring is independent of any self-report: this problem has no terminal-moment formula, so the quantity of interest below is the whole grade.

**Quantity of interest.** `no_touch_S75`, `no_touch_S100`, `no_touch_S125` must be within 0.8% (relative) of the reference value. It is read from the `functionals` dict you return, and it is gated independently of the moments: reporting a number without a simulation behind it, or a simulation without the number, fails.

**Two path counts.** `solve_sde` is called a second time with `num_paths=100000` and the same `dt`, `T` and `seed`, and the two reported values of each quantity are compared. They must differ — two Monte-Carlo estimates taken at different sample sizes do — and they must agree to within the sampling noise the two counts imply. A value that is identical at both counts is a constant, not an estimate, and fails. Nothing here requires the finer run to be the more accurate one; the check is symmetric.
