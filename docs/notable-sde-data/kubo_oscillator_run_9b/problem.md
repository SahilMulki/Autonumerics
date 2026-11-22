# Kubo oscillator

Simulate the two-dimensional **Stratonovich** system

    dX1 = -a X2 dt - sigma X2 o dW
    dX2 =  a X1 dt + sigma X1 o dW,        X(0) = (1, 0)

with a = 1 and sigma = 1, driven by a single Brownian motion W, over the time
interval [0, 1]. The symbol `o dW` denotes the Stratonovich integral: the
coefficients above are Stratonovich coefficients, not Ito ones.

The exact flow of this system is a rotation of X(0), so the amplitude |X(t)| is
conserved on every path at every time. The harness checks that the sample you
return has conserved it; a scheme that reads the coefficients as Ito coefficients
solves a different equation, whose amplitude grows.

This problem is graded **path by path**, not in distribution. You are handed the
Brownian increments and your job is to advance the state through them with a
discretised scheme; the harness compares your terminal states to the exact solution
driven by those same increments.


---

## Solver contract (how this problem is run and scored)

Implement, in `solver.py`, exactly this function:

    def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None) -> dict

It is imported and called **once**, with these exact arguments:

    solve_sde(num_paths=2000, dt=0.04, T=1, seed=42)

Use the `dt` you are given (do not hardcode a different step).

`dW` is always supplied: a numpy array of shape `(2000, 25)` holding the Brownian increments, **already scaled by sqrt(dt)**. Use exactly these — take `Nt = dW.shape[1]` and `dt = T / Nt` — and do not draw your own randomness for the Brownian path.

It must return a dict with:

- `terminal_paths`: 2-D numpy array of shape `(num_paths, 2)` — the 2 state components (X, Y) at time T, in that column order
- `empirical_mean`: list of floats, one per component
- `empirical_variance`: list of floats, one per component (use `ddof=1`)

Requirements:
- Seed all randomness with `np.random.default_rng(seed)` (not `np.random.seed`).
- Round the step count: `Nt = max(1, round(T / dt))`, then `dt = T / Nt`.
- Keep `solve_sde` importable — no work at module load outside `if __name__ == "__main__"`.
- Use the Python standard library and numpy only.

This problem has an exact solution that is a function of the Brownian path, so it is graded **path by path** rather than in distribution. Scoring is independent of any self-report: the harness evaluates its own exact solution on the very increments it handed you and gates the relative strong error `sqrt(mean((X_num - X_exact)^2)) / sqrt(mean(X_exact^2))` over paths at < 20%.

**A discretised scheme is required.** The exact solution of this equation is known in closed form, and a solver that evaluates it directly is exact at every `dt`. That is not what this problem measures, and it is detected rather than forbidden by politeness: the harness re-solves at `dt`, `dt/2` and `dt/4` driven by coarsenings of one Brownian path, and a solver carrying no discretization error returns terminal states identical to rounding at all three. Such a run is **excluded** from the comparison rather than failed — it is recorded as having transcribed the solution instead of integrating the equation. Advance the state step by step from the increments you are given.
