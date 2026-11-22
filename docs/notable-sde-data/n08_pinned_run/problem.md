# The 3/2 stochastic-volatility model

Simulate the scalar Ito SDE for an instantaneous variance V > 0

    dV = kappa V ( theta - V ) dt + epsilon V^{3/2} dW,     V(0) = 0.15

with kappa = 22.84, theta = 0.218 and epsilon = 8.56, over the time interval
[0, 0.1].

The diffusion coefficient grows like V^{3/2}, so a large excursion is amplified
rather than damped, and the drift is quadratic. The exact process stays strictly
positive for all time; the harness checks that every returned terminal state is
finite and strictly positive.

The quantities graded are the moments of the **reciprocal** variance at T = 0.1.
Report, in the `functionals` dict, estimated from your terminal paths:

    inv_var_mean       E[ 1 / V(T) ]
    inv_var_variance   Var[ 1 / V(T) ]   (use ddof=1)

Both are known exactly. Return the terminal values of V itself as `terminal_paths`.


---

## Solver contract (how this problem is run and scored)

Implement, in `solver.py`, exactly this function:

    def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict

It is imported and called **twice**, with these exact arguments:

    solve_sde(num_paths=200000, dt=0.004, T=0.1, seed=42)

and then again with `num_paths=50000`, everything else unchanged (see **Two path counts** below).

Use the `dt` you are given (do not hardcode a different step). It must return a dict with:

- `terminal_paths`: 1-D numpy array of shape `(num_paths,)` — the state at time T
- `empirical_mean`: float — mean of the state at T
- `empirical_variance`: float — variance of the state at T (use `ddof=1`)
- `functionals`: a dict of scalar results, containing `"inv_var_mean"`, `"inv_var_variance"` — each a float

Requirements:
- Seed all randomness with `np.random.default_rng(seed)` (not `np.random.seed`).
- Round the step count: `Nt = max(1, round(T / dt))`, then `dt = T / Nt`.
- Keep `solve_sde` importable — no work at module load outside `if __name__ == "__main__"`.
- Use the Python standard library and numpy only.

Scoring is independent of any self-report: this problem has no terminal-moment formula, so the quantity of interest below is the whole grade.

**A discretised time-stepping scheme is required.** Advance the state step by step at the `dt` you are given. Where this equation has an exactly samplable terminal law or an exact transformation that makes one available, sampling it directly is not what this problem measures: declare the scheme you step with, and step with it.

**Quantity of interest.** `inv_var_mean`, `inv_var_variance` must each be within their own relative tolerance of the reference value — `inv_var_mean` within 5%; `inv_var_variance` within 10%. It is read from the `functionals` dict you return, and it is gated independently of the moments: reporting a number without a simulation behind it, or a simulation without the number, fails.

**Two path counts.** `solve_sde` is called a second time with `num_paths=50000` and the same `dt`, `T` and `seed`, and the two reported values of each quantity are compared. They must differ — two Monte-Carlo estimates taken at different sample sizes do — and they must agree to within the sampling noise the two counts imply. A value that is identical at both counts is a constant, not an estimate, and fails. Nothing here requires the finer run to be the more accurate one; the check is symmetric.
