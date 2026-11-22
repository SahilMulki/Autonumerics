# Merton jump diffusion: a European call

Simulate the scalar Ito SDE with jumps

    dX = ( r - lambda kappa ) X dt + sigma X dW + X dJ,     X(0) = 100

where J is a compound Poisson process with intensity lambda whose jumps multiply X
by Y = e^Z with Z ~ N(m, delta^2), and kappa = E[Y] - 1 = e^{m + delta^2/2} - 1 is
the mean relative jump size, so that the drift is **compensated** and E[X(t)] grows
at the rate r. Parameters: r = 0.05, sigma = 0.15, lambda = 0.1, m = -0.90,
delta = 0.45, over the time interval [0, 0.25].

You draw the jump times and sizes yourself (seeded, like the Brownian increments).
A path with no jumps is geometric Brownian motion.

Report, in the `functionals` dict, the discounted European call price at T = 0.25,

    call_price = e^{-rT} E[ (X(T) - K)^+ ],      K = 100

estimated from your terminal paths, which you return as `terminal_paths`.


---

## Solver contract (how this problem is run and scored)

Implement, in `solver.py`, exactly this function:

    def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict

It is imported and called **twice**, with these exact arguments:

    solve_sde(num_paths=400000, dt=0.01, T=0.25, seed=42)

and then again with `num_paths=100000`, everything else unchanged (see **Two path counts** below).

Use the `dt` you are given (do not hardcode a different step). It must return a dict with:

- `terminal_paths`: 1-D numpy array of shape `(num_paths,)` — the state at time T
- `empirical_mean`: float — mean of the state at T
- `empirical_variance`: float — variance of the state at T (use `ddof=1`)
- `functionals`: a dict of scalar results, containing `"call_price"` — each a float

Requirements:
- Seed all randomness with `np.random.default_rng(seed)` (not `np.random.seed`).
- Round the step count: `Nt = max(1, round(T / dt))`, then `dt = T / Nt`.
- Keep `solve_sde` importable — no work at module load outside `if __name__ == "__main__"`.
- Use the Python standard library and numpy only.

Scoring is independent of any self-report: this problem has no terminal-moment formula, so the quantity of interest below is the whole grade.

**A discretised time-stepping scheme is required.** Advance the state step by step at the `dt` you are given. Where this equation has an exactly samplable terminal law or an exact transformation that makes one available, sampling it directly is not what this problem measures: declare the scheme you step with, and step with it.

**Quantity of interest.** `call_price` must be within 1% (relative) of the reference value. It is read from the `functionals` dict you return, and it is gated independently of the moments: reporting a number without a simulation behind it, or a simulation without the number, fails.

**Two path counts.** `solve_sde` is called a second time with `num_paths=100000` and the same `dt`, `T` and `seed`, and the two reported values of each quantity are compared. They must differ — two Monte-Carlo estimates taken at different sample sizes do — and they must agree to within the sampling noise the two counts imply. A value that is identical at both counts is a constant, not an estimate, and fails. Nothing here requires the finer run to be the more accurate one; the check is symmetric.
