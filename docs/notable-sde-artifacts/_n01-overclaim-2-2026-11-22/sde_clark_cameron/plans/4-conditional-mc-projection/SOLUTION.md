---
id: 4
plan_slug: conditional-mc-projection
scheme: conditional-expectation-projection
scheme_family: euler-maruyama
conditioning: component-1-integrand-conditioned-on-component-0-path
strategy: Rao-Blackwellise component 1 as far as the pathwise contract allows — replace U's step increment by its exact conditional expectation given the fully resolved S (W2) path and the supplied W1 step increment, which removes the within-step W1 sampling variance entirely and is the provably L2-optimal scheme that never reads the W1 sub-increments.
---

## SDE Reference

Two-dimensional Ito system driven by **two independent** Brownian motions W1, W2:

```
dS = mu dt + dW2,     S(0) = S_0 = 1
dU = S dW1,           U(0) = U_0 = 0
```

State is `(S, U)` in that order, so `terminal_paths[:, 0] = S(T)`, `terminal_paths[:, 1] = U(T)`.

Parameters: `S_0 = 1.0`, `U_0 = 0.0`, `mu = 1.0`. Time interval `[0, T]` with `T = 1.0`.

Drift / diffusion:

```
f(S, U) = [mu, 0.0]
G(S, U) = [[0.0, 1.0],
           [S,   0.0]]          against the channel pair (dW1, dW2)
```

**Channel convention (load-bearing for a pathwise grade):** the last axis of `dW` is
`channel 0 = W1` (drives U) and `channel 1 = W2` (drives S). This is the formulator's
requirement `R12`, recorded as `ambiguous` because `problem.md` fixes the numbering of W1/W2 in
the equation display but never spells the axis mapping out. Do not swap them — against a fixed
reference path a swap is a different terminal state, not a distributionally equivalent one.

Exact moments at `T = 1` (from `analytic_moments`): `mean_S = 2`, `var_S = 1`, `mean_U = 0`,
`var_U = 17/6 = 2.8333333333333335`, `cov(S, U) = 0`. Closed forms:
`mean_X = S_0 + mu*t`, `variance_X = t`, `mean_Y = 0`,
`variance_Y = ((S_0 + mu*t)**3 - S_0**3)/(3*mu) + t**2/2`.

## Numerical Scheme

- **Method**: conditional-expectation (Rao-Blackwellised) step for `U`, one update per step of
  `dW`. Exact step for `S`. The **W2 channel of `dW_sub` is read; the W1 channel is never read.**
- **dt**: 0.04 (the contract's step; `Nt = max(1, round(T/dt)) = 25`, then `dt = T/Nt`)
- **num_paths**: 16000 (`evaluation_thresholds.num_paths`; the contract call itself is
  `solve_sde(num_paths=1000, dt=0.04, T=1, seed=42)` and the kernel uses
  `graded_num_paths = 1000`, `conv_num_paths = 2000` on the CRN ladder)
- **T**: 1.0
- **seed**: 42
- **Milstein correction**: **none taken.** `milstein_eligible` is `false` with
  `milstein_ineligible_reason: levy_area`, and this plan deliberately does not assemble the Levy
  area `I_{1,2}`. It is therefore graded against the `dW`-only floor, not against
  `expected_strong_order_milstein`. Do **not** declare `levy_area_treatment` for this plan.
- **Expected strong order**: 0.5 (`verification.expected_strong_order`). This is a hard
  theoretical ceiling, not a shortfall: Clark & Cameron (1980) prove `c*sqrt(dt)` is a lower
  bound for *every* scheme whose `U` update sees only the step increments of W1, and this plan is
  the member of that class that attains the smallest possible `c` (see below).
- **Expected weak order**: 1.0
- **Reference for scoring**: analytic moments (`has_analytic_solution: true`), backed by the
  exactly-closing moment ODE on component 0 and the four `dynkin_test_functions`. The pathwise
  2% gate is scored by the harness against its own fine-grid reference, not by this plan.

### The conditioning argument (what is integrated out, and why it is legitimate here)

Component 1 is the conditionally tractable one. `U`'s own drift is zero and its diffusion row
`[S, 0]` has degree **zero** in `U`; nothing in `S`'s coefficients reads `U`. So the `S` path can
be generated first and, **conditionally on the whole W2 path**, the integrand `phi(s) = S_s` is a
*deterministic* function of time and

```
U(T) | sigma(W2)   ~   Normal( 0,  int_0^T S_s^2 ds )
```

in closed form — exactly the structure `verifylib/pool.py`'s `conditionally_tractable` detects.
Over one step, with `u = s - t_n`, `B(u) = W2(t_n + u) - W2(t_n)`:

```
phi(t_n + u) = S_n + mu*u + B(u)                    # known exactly given the W2 sub-path
Delta U      = int_0^dt phi dW1
```

and the conditional expectation of that increment given the W2 path **and** the one piece of W1
information the contract hands over at the step, `dW1_n`, is the orthogonal projection onto
`dW1_n`:

```
E[ Delta U | sigma(W2), dW1_n ]  =  phi_bar_n * dW1_n ,
    phi_bar_n = (1/dt) * int_0^dt phi(t_n + u) du = S_n + mu*dt/2 + B_bar_n
    B_bar_n   = (1/dt) * int_0^dt B(u) du
```

**That is this plan's step.** It is Rao-Blackwellisation of component 1 applied to the one
direction in which the solver has residual uncertainty: the fine structure of W1 inside the step.
That direction's variance is removed *entirely*, not reduced — the step is the conditional mean,
and by orthogonality the scheme's variance plus the residual variance reconstruct `Var[U(T)]`
exactly, with no cross term.

```
S_{n+1} = S_n + mu*dt + dW[:, n, 1]                         # exact
U_{n+1} = U_n + (S_n + mu*dt/2 + B_bar_n) * dW[:, n, 0]     # conditional mean of the increment
```

`B_bar_n` from the **W2 channel only** of `dW_sub` (stochastic Fubini gives
`int_0^dt B du = int_0^dt (dt - s) dW2(s)` exactly; the sub-grid trapezoid below is exact for the
piecewise-linear interpolant of the supplied sub-path). Let `M = dW_sub.shape[2]` — do **not**
hardcode 256 — and `h = dt / M`, `s_j = j*h`:

```python
d2 = dW_sub[:, :, :, 1]                             # W2 sub-increments ONLY
h  = dt / d2.shape[2]
w  = (dt - (np.arange(d2.shape[2]) * h) - 0.5 * h) / dt     # shape (M,)
B_bar = d2 @ w                                      # shape (num_paths, Nt)
```

### Why this is a distinct scheme, and what it measures

Per step the residual is `R_n = int (phi - phi_bar_n) dW1`, whose variance works out to

```
Var[R_n] = dt^2/6 + mu^2 * dt^3/12          (Euler-Maruyama's is dt^2/2 + mu^2*dt^3/3)
```

so over `Nt = T/dt` steps the strong error is `sqrt(T*(dt/6 + mu^2*dt^2/12))` against
Euler-Maruyama's `sqrt(T*(dt/2 + mu^2*dt^2/3))` — the same `sqrt(dt)` rate with the constant
reduced by the asymptotic factor `sqrt(3)`.

| scheme | residual RMS at dt = 0.04 | vs `sqrt(E[U_T^2]) = 1.683` | vs full state `2.799` |
|---|---|---|---|
| Euler-Maruyama (`dW` only) | 0.1433 | 8.5% | 5.1% |
| **this plan** (`dW` + W2 sub-path) | **0.0825** | **4.9%** | **2.95%** |
| Levy-area Milstein (`dW` + both channels) | 0.0231 | 1.37% | 0.82% |

**This plan is expected to miss the 2% pathwise gate, by construction and not by defect**, which
is why it is numbered last. Its value is threefold and none of it is the gate:

1. It **measures the Clark-Cameron lower bound rather than asserting it.** The bound is a
   statement about the best possible `dW`-only scheme; Euler-Maruyama is a convenient member of
   that class, not the optimal one. A measured pair of 8.5% and 4.9% at the same observed order
   ~0.5, with a ratio near `sqrt(3)`, is direct evidence that the gap the Levy-area plans close is
   genuinely the Levy area and not slack in the baseline.
2. Its **moments are near-exact and cheap to check.** `E[U(T)] = 0` holds exactly (every factor
   multiplying `dW1_n` is W2-measurable, hence independent of `dW1_n`), and the variance deficit
   is the residual variance above: `Var[U(T)] = 17/6 - T*(dt/6 + mu^2*dt^2/12) = 2.8265` at
   `dt = 0.04`, a **0.24%** relative error against `variance_rel_err_max = 0.1`.
3. It is a **cross-family-adjacent cross-check** on the channel mapping `R12`. It reads only the
   W2 channel of `dW_sub` and the W1 channel of `dW`, so a plan that has the two channels swapped
   produces a visibly different error here than the Levy-area plans do.

### The conditional law, and exactly which quantity it may be used for

The conditional law `U(T) | sigma(W2) ~ Normal(0, int_0^T S^2 ds)` is a **distributional**
reference. It may be used for one quantity and must not touch another:

- **It may be used for the `Var[U(T)]` reference**, as the Rao-Blackwellised estimator
  `Vhat = mean_paths( sub-grid Riemann sum of S_s^2 ds )`, which has **zero W1 sampling
  variance** and converges to `17/6` with a standard error far below that of the sample variance
  of `U(T)` at the same path count. Report it in the **Results** section as a diagnostic, next to
  `17/6 = 2.8333333333333335` and next to the realised sample variance.
- **It must not be substituted into `empirical_variance`, and no part of `terminal_paths` may be
  drawn from it.** `empirical_variance` is contractually the `ddof=1` variance of the returned
  `terminal_paths` (`R16`), the kernel recomputes the graded moments from `terminal_paths` itself
  and separately checks the reported pair against them (`reported_moments_agree`), so a
  substituted value is both ungraded and a misreport.

## Implementation Notes

- **`exact_sampling_components` is deliberately absent, and the pool-level warning this plan
  answers is only partly dischargeable.** `verifylib/pool.py`'s rule asks for component 1 to be
  *integrated out*; that field's test is **non-consumption of the supplied increments**. This
  plan's `U` step consumes `dW[:, n, 0]` in full and draws no auxiliary randomness, so declaring
  the field would be false, and it would make the kernel abstain on a strong order that is real,
  measurable and informative (0.5). `sde_manual.md` rule 0 is explicit: never declare it for a
  component actually stepped with the supplied increments.
  The *full* Rao-Blackwellisation — replacing `U(T)` by `E[U(T) | sigma(W2)] = 0`, or drawing it
  from `Normal(0, int S^2 ds)` with fresh randomness — is **incompatible with this problem's
  contract**, three ways over: it is graded path by path against a reference driven by the very
  same W1 path (`R6`, `R21`), so the first choice gives a 100% relative strong error and a zero
  variance, and the second decorrelates from the reference path entirely while violating "do not
  draw your own randomness for the Brownian path" (`R13`). The sampling variance Rao-
  Blackwellisation removes is the variance of a path the solver *draws*; here the solver draws
  nothing. What remains genuinely removable is the within-step W1 fine structure, and this plan
  removes all of it.
- **`R22` safety: this plan does not bypass the step.** The W1 channel of `dW_sub` is never read,
  so the update is not algebraically a sub-grid Euler sum — `sum_j phi(s_j) dW1_j` needs the W1
  sub-increments, which this solver never touches. Terminal states at `dt`, `dt/2`, `dt/4` are
  genuinely different and the order study has a real `dt`-dependence to measure. Advance the
  state **once per step of `dW`** (`R9`/`R22`/`R23`): reduce `dW_sub[..., 1]` to the per-step
  scalar `B_bar` first (the matrix-vector product above contracts the sub-axis away), then step.
- **Left endpoint in the right sense.** `phi_bar_n` is a *time average over the step of the
  integrand*, multiplying the step's own `dW1_n`; it is **not** a midpoint or right-endpoint
  evaluation of `S`. It stays an Ito step because `phi_bar_n` is `sigma(W2)`-measurable and W1 is
  independent of W2, so `E[phi_bar_n * dW1_n] = 0` and the martingale invariant holds exactly. Do
  not reach for `S_{n+1}` or `(S_n + S_{n+1})/2` as a shortcut: `(S_n + S_{n+1})/2` is a
  Stratonovich-flavoured step whose W2-measurability saves the mean but whose `B` average is the
  *endpoint* average `B(dt)/2` rather than `B_bar`, which leaves residual variance
  `dt^2/4` instead of `dt^2/6` and is not the conditional expectation.
- **`dW_sub` may be withheld.** `implementation_notes` records that the kernel may omit it at the
  16000-path moment call (1.6 GB there). Handle `dW_sub is None` by setting `B_bar = 0`, i.e.
  falling back to `U_{n+1} = U_n + (S_n + mu*dt/2) * dW1_n` — still the conditional expectation
  given `sigma(S_n)` and `dW1_n`, still exact in the mean, residual variance
  `dt^2/2 + mu^2*dt^3/12`. Do **not** silently substitute a self-drawn sub-path.
- **Memory.** `dW_sub` is `num_paths * Nt * M * 2` float64 = `num_paths * 102400` bytes,
  independent of ladder level. Only the W2 channel is needed; slice it and chunk over paths rather
  than materialising a second full-size temporary.
- **Never draw the Brownian path when `dW` is given** (`R13`). Keep the `dW is None` branch for
  standalone runs only, seeded with `np.random.default_rng(seed)` — not `np.random.seed` (`R17`).
- **Positivity / clipping:** none. The system is linear with affine coefficients; `S` and `U` are
  both signed and unbounded. The only gated constraint is `finite` (`np.isfinite(X)`).
- **No variance reduction wrapper, on purpose** (manual rule 4 / step 1c). The problem declares no
  payoff and no `functionals` entry, and is graded path by path against a fixed supplied path, so
  an antithetic pair or control variate would require drawing or negating that path (`R13`) and
  would break the pathwise comparison. There is no `functional_se` to report. The conditioning
  above is a change of *scheme*, not a wrapper: it changes the pathwise dynamics, so the CRN
  ladder sees a different scheme than Euler-Maruyama — which is the whole point of adding it as a
  plan rather than as a second estimator.
- **Mean check on component 1 is skipped**: `|mean_U| = 0 < near_zero_mean_threshold = 0.01`, so
  `U` is graded on variance (`17/6`) and on the martingale invariant, not on a relative mean error.
- **`U(T)` is not Gaussian** unconditionally (measured excess kurtosis 1.32) even though it is
  Gaussian *given* the W2 path — the mixture over `int S^2 ds` is not. `verification.terminal_law`
  is `null`; expect no KS-style test and assume normality nowhere in the solver.
- **Path-count resolution.** `variance_rel_err_max = 0.1` with `ci_mult = 2.0` and kurtosis 1.32
  puts the moment-resolution floor near 10600 paths, which is why `num_paths = 16000` and not the
  contract's 1000.
- **Return contract**: `terminal_paths` shape `(num_paths, 2)`, `empirical_mean` and
  `empirical_variance` as lists of floats, one per component, variance with `ddof=1` (`R15`,
  `R16`). `solve_sde` must stay importable — no module-load work outside
  `if __name__ == "__main__"` (`R19`). numpy and the standard library only (`R20`).

## Results

(Filled by solver after running)

## Evaluation

(Filled by evaluator)

<review score=0>

Awaiting solver.

</review>
