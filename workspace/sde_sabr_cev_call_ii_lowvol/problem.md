# SABR Forward Price, CEV Exponent 1/2, Ten-Year Horizon

Simulate the two-factor system (Ito) for a forward price with stochastic
volatility:

    dS     = sigma * S^(1/2) dW_S,      S(0)     = 0.07
    dsigma = 0.8 * sigma dW_sigma,      sigma(0) = 0.34

with corr(dW_S, dW_sigma) = -0.6, over the interval [0, 10]. S is a forward
price, so it has no drift and no discounting is applied anywhere.

This is the long-horizon, high-vol-of-vol parameter set, and it is harder than
it looks:

- Over ten years at vol-of-vol 0.8 the volatility is lognormal with a log
  standard deviation of about 2.5, so sigma(T) ranges over several orders of
  magnitude between paths. The step has to be fine enough for the paths where
  it is large.
- The exponent 1/2 makes S = 0 **attainable and absorbing**, and at these
  parameters a substantial fraction of paths reach it. A path absorbed at zero
  stays there and contributes nothing to the payoff; a scheme that lets S go
  negative, or that reflects it, is simulating a different process.
- S is a true martingale, so E[S(T)] = S(0) = 0.07 exactly. A time-discretised
  scheme does not inherit that.

Report the **undiscounted** European call values E[(S(T) - K)^+] at the three
strikes

    K1 = 0.051022538988
    K2 = 0.07
    K3 = 0.096035989138

as `price_K1`, `price_K2` and `price_K3` in the `functionals` dict, and the
terminal states (S, sigma) in `terminal_paths`.


---

## Solver contract (how this problem is run and scored)

Implement, in `solver.py`, exactly this function:

    def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict

It is imported and called **twice**, with these exact arguments:

    solve_sde(num_paths=1000000, dt=0.004, T=10, seed=42)

and then again with `num_paths=250000`, everything else unchanged (see **Two path counts** below).

Use the `dt` you are given (do not hardcode a different step). It must return a dict with:

- `terminal_paths`: 2-D numpy array of shape `(num_paths, 2)` — the 2 state components (X, Y) at time T, in that column order
- `empirical_mean`: list of floats, one per component
- `empirical_variance`: list of floats, one per component (use `ddof=1`)
- `functionals`: a dict of scalar results, containing `"price_K1"`, `"price_K2"`, `"price_K3"` — each a float

Requirements:
- Seed all randomness with `np.random.default_rng(seed)` (not `np.random.seed`).
- Round the step count: `Nt = max(1, round(T / dt))`, then `dt = T / Nt`.
- Keep `solve_sde` importable — no work at module load outside `if __name__ == "__main__"`.
- Use the Python standard library and numpy only.

Scoring is independent of any self-report: this problem has no terminal-moment formula, so the quantity of interest below is the whole grade.

**Quantity of interest.** `price_K1`, `price_K2`, `price_K3` must be within 1% (relative) of the reference value. It is read from the `functionals` dict you return, and it is gated independently of the moments: reporting a number without a simulation behind it, or a simulation without the number, fails.

**Two path counts.** `solve_sde` is called a second time with `num_paths=250000` and the same `dt`, `T` and `seed`, and the two reported values of each quantity are compared. They must differ — two Monte-Carlo estimates taken at different sample sizes do — and they must agree to within the sampling noise the two counts imply. A value that is identical at both counts is a constant, not an estimate, and fails. Nothing here requires the finer run to be the more accurate one; the check is symmetric.
