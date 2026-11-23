# Clark-Cameron system

Simulate the two-dimensional Ito system driven by **two independent** Brownian
motions W1 and W2:

    dS = mu dt + dW2,        S(0) = 1
    dU = S dW1,              U(0) = 0

with mu = 1, over the time interval [0, 1]. The state is (S, U), in that order.

This problem is graded **path by path**, not in distribution: the harness compares
your terminal states to a reference solution driven by the very same Brownian
path. The reference is built on a finer grid than the one you step on, and you are
handed that finer grid too: `dW_sub[:, k, :, :]` holds the 256 sub-increments that
sum to `dW[:, k, :]` for step k. The strong error of any scheme that uses only the
step increments `dW` is bounded below by a constant times sqrt(dt), whatever the
scheme (Clark and Cameron, 1980); the quantity that bound depends on is the Levy
area of (W1, W2) over each step, which the sub-increments carry and the step
increments do not. Advance the state step by step at the step `dt` you are given.


---

## Solver contract (how this problem is run and scored)

Implement, in `solver.py`, exactly this function:

    def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              dW_sub: np.ndarray | None = None) -> dict

It is imported and called **once**, with these exact arguments:

    solve_sde(num_paths=1000, dt=0.04, T=1, seed=42)

Use the `dt` you are given (do not hardcode a different step).

`dW` is always supplied: a numpy array of shape `(1000, 25, 2)` holding the Brownian increments, **already scaled by sqrt(dt)**, whose last axis indexes the 2 independent noise sources. Use exactly these — take `Nt = dW.shape[1]` and `dt = T / Nt` — and do not draw your own randomness for the Brownian path.

`dW_sub` is always supplied too: a numpy array of shape `(1000, 25, 256, 2)` holding, for each step, the 256 finer increments that sum to it (`dW_sub.sum(axis=2) == dW` to rounding), each already scaled by the square root of its own sub-step `dt / 256`. They are the same Brownian path at a finer resolution, not independent noise.

It must return a dict with:

- `terminal_paths`: 2-D numpy array of shape `(num_paths, 2)` — the 2 state components (X, Y) at time T, in that column order
- `empirical_mean`: list of floats, one per component
- `empirical_variance`: list of floats, one per component (use `ddof=1`)

Requirements:
- Seed all randomness with `np.random.default_rng(seed)` (not `np.random.seed`).
- Round the step count: `Nt = max(1, round(T / dt))`, then `dt = T / Nt`.
- Keep `solve_sde` importable — no work at module load outside `if __name__ == "__main__"`.
- Use the Python standard library and numpy only.

This problem is graded **path by path** rather than in distribution, against a reference solution the harness computes on the finer grid from the very same Brownian path it handed you (the reference's own error, measured, is 0.2% relative, far below the gate). Scoring is independent of any self-report: the harness gates the relative strong error `sqrt(mean((X_num - X_exact)^2)) / sqrt(mean(X_exact^2))` over paths at < 2%.

**A discretised scheme at the step `dt` is required.** The harness re-solves at `dt`, `dt/2` and `dt/4` driven by coarsenings of one Brownian path, and a solver that simply integrates on the sub-grid whatever `dt` it is given returns terminal states identical to rounding at all three. Such a run is **excluded** from the comparison rather than failed — it is recorded as having bypassed the step rather than integrating at it. Use `dW_sub` to inform the step (for instance to assemble the iterated integrals a higher-order scheme needs), and advance the state once per step of `dW`.
