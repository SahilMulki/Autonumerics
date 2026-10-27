# Heston-Hull-White Call, All Three Factors Correlated

Simulate the three-factor system (Ito) in which the short rate is stochastic:

    dS = R * S dt + sqrt(V) * S dW1,             S(0) = S0
    dV = 0.5 * (0.04 - V) dt + sqrt(V) dW2,      V(0) = 0.04
    dR = 0.08 * (0.10 - R) dt + 0.09 dW3,        R(0) = 0.10

over [0, 10], with the correlation matrix

    corr(dW1, dW2) = -0.9
    corr(dW1, dW3) =  0.6
    corr(dW2, dW3) = -0.7

Three things make this the hardest problem in this group:

- **The Feller condition fails badly.** 2*kappa*eta = 0.04 against
  sigma^2 = 1, a ratio of 0.04, so the variance sits on zero for a large
  fraction of the time and every square root has to say what it does there.
- **The discount factor is stochastic and correlated with the asset.** The
  payoff is discounted by exp(-integral of R over [0, T]), which must be
  accumulated along each path rather than applied at the end with a constant
  rate. Over ten years at these parameters the discount factor varies widely
  between paths.
- **All three Brownian motions are correlated**, so the increments must be
  drawn from the full 3x3 correlation structure. Note that a scheme which
  samples the variance from its exact conditional law rather than stepping it
  gains accuracy on V and then has no Brownian increment left to correlate W3
  with; that trade has to be resolved, not ignored.

Report the **discounted** European call value
E[exp(-integral of R dt) * (S(T) - 100)^+] from each of the three initial spots

    S0 = 75,   S0 = 100,   S0 = 125

as `price_S75`, `price_S100` and `price_S125` in the `functionals` dict. The
variance and the rate do not depend on S, so one set of (V, R) paths serves all
three spots.

Return the terminal states (S, V, R) for **S0 = 100** in `terminal_paths`.


---

## Solver contract (how this problem is run and scored)

Implement, in `solver.py`, exactly this function:

    def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict

It is imported and called **twice**, with these exact arguments:

    solve_sde(num_paths=1000000, dt=0.004, T=10, seed=42)

and then again with `num_paths=250000`, everything else unchanged (see **Two path counts** below).

Use the `dt` you are given (do not hardcode a different step). It must return a dict with:

- `terminal_paths`: 2-D numpy array of shape `(num_paths, 3)` — the 3 state components (X, Y, Z) at time T, in that column order
- `empirical_mean`: list of floats, one per component
- `empirical_variance`: list of floats, one per component (use `ddof=1`)
- `functionals`: a dict of scalar results, containing `"price_S75"`, `"price_S100"`, `"price_S125"` — each a float

Requirements:
- Seed all randomness with `np.random.default_rng(seed)` (not `np.random.seed`).
- Round the step count: `Nt = max(1, round(T / dt))`, then `dt = T / Nt`.
- Keep `solve_sde` importable — no work at module load outside `if __name__ == "__main__"`.
- Use the Python standard library and numpy only.

Scoring is independent of any self-report: this problem has no terminal-moment formula, so the quantity of interest below is the whole grade.

**Quantity of interest.** `price_S75`, `price_S100`, `price_S125` must be within 0.3% (relative) of the reference value. It is read from the `functionals` dict you return, and it is gated independently of the moments: reporting a number without a simulation behind it, or a simulation without the number, fails.

**Two path counts.** `solve_sde` is called a second time with `num_paths=250000` and the same `dt`, `T` and `seed`, and the two reported values of each quantity are compared. They must differ — two Monte-Carlo estimates taken at different sample sizes do — and they must agree to within the sampling noise the two counts imply. A value that is identical at both counts is a constant, not an estimate, and fails. Nothing here requires the finer run to be the more accurate one; the check is symmetric.
