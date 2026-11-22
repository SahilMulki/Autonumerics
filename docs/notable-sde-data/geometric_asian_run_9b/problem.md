# Geometric-average Asian option

Simulate the scalar Ito SDE (Black-Scholes asset)

    dS = r S dt + sigma S dW,     S(0) = 100

with r = 0.05 and sigma = 0.2, over the time interval [0, 1], and price the
fixed-strike **geometric-average** Asian call with strike K = 100: the payoff is
(A - K)^+ where

    A = exp( (1/T) int_0^T ln S(t) dt )

is the continuous geometric average of the path.

The average is a time integral along each path, and how you discretise that
integral decides the answer: at 25 steps the choice of quadrature rule moves the
price by several percent.

Return the terminal asset values S(T) as `terminal_paths`, and report, in the
`functionals` dict, estimated from the same paths:

    call_price             e^{-rT} E[ (A - K)^+ ]
    geo_average_variance   Var[ A ]   (use ddof=1)

Both are known in closed form.


---

## Solver contract (how this problem is run and scored)

Implement, in `solver.py`, exactly this function:

    def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict

It is imported and called **twice**, with these exact arguments:

    solve_sde(num_paths=400000, dt=0.04, T=1, seed=42)

and then again with `num_paths=100000`, everything else unchanged (see **Two path counts** below).

Use the `dt` you are given (do not hardcode a different step). It must return a dict with:

- `terminal_paths`: 1-D numpy array of shape `(num_paths,)` — the state at time T
- `empirical_mean`: float — mean of the state at T
- `empirical_variance`: float — variance of the state at T (use `ddof=1`)
- `functionals`: a dict of scalar results, containing `"call_price"`, `"geo_average_variance"` — each a float

Requirements:
- Seed all randomness with `np.random.default_rng(seed)` (not `np.random.seed`).
- Round the step count: `Nt = max(1, round(T / dt))`, then `dt = T / Nt`.
- Keep `solve_sde` importable — no work at module load outside `if __name__ == "__main__"`.
- Use the Python standard library and numpy only.

Scoring is independent of any self-report: this problem has no terminal-moment formula, so the quantity of interest below is the whole grade.

**A discretised time-stepping scheme is required.** Advance the state step by step at the `dt` you are given. Where this equation has an exactly samplable terminal law or an exact transformation that makes one available, sampling it directly is not what this problem measures: declare the scheme you step with, and step with it.

**Quantity of interest.** `call_price`, `geo_average_variance` must each be within their own relative tolerance of the reference value — `call_price` within 1%; `geo_average_variance` within 3%. It is read from the `functionals` dict you return, and it is gated independently of the moments: reporting a number without a simulation behind it, or a simulation without the number, fails.

**Two path counts.** `solve_sde` is called a second time with `num_paths=100000` and the same `dt`, `T` and `seed`, and the two reported values of each quantity are compared. They must differ — two Monte-Carlo estimates taken at different sample sizes do — and they must agree to within the sampling noise the two counts imply. A value that is identical at both counts is a constant, not an estimate, and fails. Nothing here requires the finer run to be the more accurate one; the check is symmetric.
