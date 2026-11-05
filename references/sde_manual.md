# SDE Solver Manual

## Numerical Schemes

### Euler-Maruyama (EM)

Strong order 0.5, weak order 1.0. Use for any SDE.

**Scalar update** (state X ∈ ℝ):
```
dW_n  = sqrt(dt) * Z_n,   Z_n ~ N(0,1)
X_{n+1} = X_n + f(X_n, t_n) * dt + g(X_n, t_n) * dW_n
```

**Vector update** (state X ∈ ℝ^d, noise W ∈ ℝ^m):
```
dW_n  = sqrt(dt) * Z_n,   Z_n ~ N(0, I_m)
X_{n+1} = X_n + f(X_n, t_n) * dt + G(X_n, t_n) @ dW_n
```
where G is the d×m diffusion matrix.

**Monte Carlo layout**: `X` has shape `(num_paths, d)`. All paths are stepped simultaneously using NumPy broadcasting.

---

### Milstein (scalar only)

Strong order 1.0, weak order 1.0. Use only when `state_dimension == 1` AND `noise_structure == "multiplicative"` AND `milstein_eligible == true` in sde_spec.json.

**Scalar update**:
```
dW_n  = sqrt(dt) * Z_n,   Z_n ~ N(0,1)
X_{n+1} = X_n + f(X_n, t_n) * dt + g(X_n, t_n) * dW_n
         + 0.5 * g(X_n, t_n) * g'(X_n, t_n) * (dW_n**2 - dt)
```
where `g'(X, t) = dg/dX`. This correction term is exactly what improves the strong order.

**Why not multi-D Milstein?** Multi-dimensional Milstein requires the Lévy area (the iterated stochastic integral ∫∫dW_i dW_j), which is not computable without approximation. Use EM for all multi-D problems.

---

## Milstein Corrections by SDE Family

| SDE | g(X) | dg/dX | Milstein correction |
|---|---|---|---|
| GBM / Black-Scholes | σX | σ | 0.5 σ²X (dW²−dt) |
| CIR | σ√X | σ/(2√X) | (σ²/4)(dW²−dt) |
| Exp-OU | σX | σ | 0.5 σ²X (dW²−dt) |
| BM with drift | σ (const) | 0 | 0 (Milstein == EM) |
| OU | σ (const) | 0 | 0 (Milstein == EM) |
| Linear additive | c (const) | 0 | 0 (Milstein == EM) |

---

## Implementation Notes by SDE Family

### CIR (Cox-Ingersoll-Ross)
- Diffusion g(X) = σ√X is undefined for X < 0.
- Apply a positivity guard before computing sqrt: `X_pos = np.maximum(X_n, 0.0)`
- The Feller condition (2κθ ≥ σ²) guarantees X stays positive mathematically, but floating-point rounding can produce tiny negative values near zero. The guard is mandatory.

### Exp-OU (Exponential Ornstein-Uhlenbeck)
- The drift is f(X) = X·(−θ·log(X) + σ²/2). Requires X > 0 for log(X) to be defined.
- With X_0 = 1.0 and typical parameters, X stays positive. Guard with `np.maximum(X_n, 1e-12)` if needed.

### 2D Correlated GBM
- Introduce correlation via Cholesky: `dW_corr = rho * dW1 + sqrt(1 - rho**2) * dW2`
- X and Y are updated independently after generating correlated increments.

### Stochastic Oscillator
- Noise enters only the Y component: `G = [[0], [sigma]]`
- X update has no stochastic term: `X_{n+1} = X_n + Y_n * dt`
- Y update: `Y_{n+1} = Y_n - X_n * dt + sigma * dW_n`

---

## Analytic Moments Reference

All expressions below use NumPy syntax. `t` is the evaluation time (use terminal T).

### 1. Standard Brownian Motion (`bm_standard`)
```python
X_0 = 0.0
exact_mean     = 0.0
exact_variance = t
```

### 2. BM with Drift (`bm_with_drift`)
```python
X_0, mu, sigma = 1.0, 0.5, 0.3
exact_mean     = X_0 + mu * t
exact_variance = sigma**2 * t
```

### 3. Geometric Brownian Motion (`geometric_brownian_motion`)
```python
X_0, mu, sigma = 1.0, 0.1, 0.2
exact_mean     = X_0 * np.exp(mu * t)
exact_variance = X_0**2 * np.exp(2*mu*t) * (np.exp(sigma**2 * t) - 1)
```

### 4. Ornstein-Uhlenbeck (`ornstein_uhlenbeck`)
```python
X_0, theta, mu, sigma = 2.0, 1.5, 0.0, 0.5
exact_mean     = mu + (X_0 - mu) * np.exp(-theta * t)
exact_variance = sigma**2 / (2*theta) * (1 - np.exp(-2*theta*t))
```

### 5. Linear SDE Additive (`linear_sde_additive`)
```python
X_0, a, b, c = 0.0, 2.0, -1.0, 0.5
exact_mean     = np.exp(b*t) * (X_0 + a/b) - a/b
exact_variance = c**2 / (2*b) * (np.exp(2*b*t) - 1)
# With b=-1: variance = 0.125*(1 - exp(-2t)) > 0
```

### 6. Cox-Ingersoll-Ross (`cox_ingersoll_ross`)
```python
X_0, kappa, theta, sigma = 0.5, 2.0, 1.0, 0.5
exact_mean     = theta + (X_0 - theta) * np.exp(-kappa * t)
exact_variance = (sigma**2 / kappa) * (
    X_0 * (np.exp(-kappa*t) - np.exp(-2*kappa*t))
    + 0.5 * theta * (1 - np.exp(-kappa*t))**2
)
```

### 7. Exponential Ornstein-Uhlenbeck (`exponential_ornstein_uhlenbeck`)
```python
X_0, theta, sigma = 1.0, 1.0, 0.4
# v_t = sigma^2/(2*theta) * (1 - exp(-2*theta*t))
v_t = sigma**2 / (2*theta) * (1 - np.exp(-2*theta*t))
exact_mean     = np.exp(v_t / 2)
exact_variance = np.exp(v_t) * (np.exp(v_t) - 1)
```

### 8. Black-Scholes (`black_scholes`)
```python
X_0, r, sigma = 100.0, 0.05, 0.20
exact_mean     = X_0 * np.exp(r * t)
exact_variance = X_0**2 * np.exp(2*r*t) * (np.exp(sigma**2 * t) - 1)
```

### 9. 2D Correlated GBM (`gbm_2d_correlated`)
```python
X_0, Y_0 = 1.0, 1.0
mu1, sigma1, mu2, sigma2, rho = 0.10, 0.20, 0.15, 0.25, 0.60
exact_mean_X     = X_0 * np.exp(mu1 * t)
exact_variance_X = X_0**2 * np.exp(2*mu1*t) * (np.exp(sigma1**2 * t) - 1)
exact_mean_Y     = Y_0 * np.exp(mu2 * t)
exact_variance_Y = Y_0**2 * np.exp(2*mu2*t) * (np.exp(sigma2**2 * t) - 1)
```

### 10. Stochastic Oscillator (`stochastic_oscillator`)
```python
X_0, Y_0, sigma = 1.0, 0.0, 0.30
T = 2 * np.pi
exact_mean_X     = X_0 * np.cos(t) + Y_0 * np.sin(t)
exact_mean_Y     = -X_0 * np.sin(t) + Y_0 * np.cos(t)
exact_variance_X = sigma**2 / 2 * (t - np.sin(t) * np.cos(t))
exact_variance_Y = sigma**2 / 2 * (t + np.sin(t) * np.cos(t))
# At t=2pi: exact_mean_X ≈ 1.0, exact_mean_Y ≈ 0.0 (near zero — skip mean check)
```

---

## Scheme Selection Rules

**Read rule 0 first.** Rules 1–3 are about Euler–Maruyama and Milstein because those were the only
two schemes in this manual, and rule 1 sends every multi-dimensional problem to EM. That rule is
correct about *Milstein* and was being read as a rule about everything else: on a ten-problem
option-pricing suite where all ten problems are multi-dimensional, it says "EM only" ten times, and
a single untooled model completion chose an exact transition or Andersen QE on nine of them and was
more accurate on seven.

0. **Where the transition law is known in closed form, sample it.** A scheme that draws
   `X_{t+Δ} | X_t` from its exact conditional law carries **no discretization error at all**, at any
   `dt`, and no amount of step refinement makes a discretised scheme competitive with it. This is
   not an exotic option — it covers a large share of the SDEs anyone prices:

   | SDE | exact transition |
   |---|---|
   | CIR / Heston variance | non-central χ², or the Poisson-mixed central χ² form |
   | squared Bessel, BESQ(0) | `½·v·Γ(N)` with `N ~ Poisson(2S/v)` |
   | GBM, Black–Scholes | lognormal — exact in one step |
   | OU, Vasicek | Gaussian with the exact conditional mean and variance |
   | Heston as a pair | Broadie–Kaya, or Andersen's QE as the practical approximation |

   Key the choice on the spec, not on recall: `analytic_moments.has_analytic_solution`,
   `verification.terminal_law`, and the problem's own `implementation_notes` say when a law is
   available. Where it is, propose at least one plan that uses it, and keep a discretised plan in
   the pool as the cross-check.

   **Declare it.** The test is **whether the step consumes the Brownian increments the evaluator
   supplies**, not whether its draw is exact. A step that generates its own auxiliary randomness
   cannot consume them, so the common-random-number order study reads a strong order near zero on a
   scheme that has no order to measure. Put the affected state components in the frontmatter:

   ```yaml
   scheme_family: other
   exact_sampling_components: 0        # S draws its own randomness; the vol leg is stepped
   ```

   The kernel then **abstains** on those components rather than failing them. An abstention is not
   a pass: it costs the plan the order check's evidence instead of charging it a penalty it did not
   earn. Declaring it on a component you discretised buys nothing and loses the check.

   The *graded functional's* dt-bias term abstains too, but only where **every** component is
   declared exact — there the whole state is drawn from its exact law and there is no discretization
   anywhere. A partially exact plan is charged its bias in full, because the graded quantity depends
   on the whole state and one discretised leg biases it however exactly the others are drawn.
   Measured on `sde_sabr_cev_call_i` plan 1 (forward exact, volatility stepped): its ladder reads
   observed orders 1.90–2.29 with clean factor-of-four difference ratios, a real second-order
   convergence, and abstaining on it promoted the plan over one 2.3× more accurate.

   **This covers moment-matched schemes, not only literally exact ones — Andersen's QE included.**
   The field is named for the case it was written for and the name reads narrower than the rule.
   QE's variance leg draws `Z_V` and `U` fresh each step as a moment-matched approximation to the
   non-central chi-squared transition; it is therefore *not* exact sampling, and it is *equally*
   unable to consume the supplied increments. Declare it. Broadie–Kaya, a Poisson-mixed BESQ(0)
   draw and an exact log-normal step all qualify for the same reason.

   Measured on `sde_heston_call_feller` (2026-10-31): a correct Andersen-QE plan read this section,
   reasoned that QE is "a moment-matched approximation, not literal inversion sampling", and wrote
   in its own SOLUTION.md "do **not** declare `exact_sampling_components` for this plan". The
   reasoning was right about exactness and wrong about the rule. Its strong order came back
   -0.0021, the order study **failed** it, and the plan scored 3 with all three of its prices
   inside their gates — then lost the selection to a plain Euler plan that could not allocate at
   grading scale. With the declaration the same plan scores 7 and wins. The one thing never to do
   is declare it for a component you actually stepped with the supplied increments: there a
   near-zero order is a real defect and the declaration hides it.

0b. **Where one component is tractable given the others, integrate it out instead of sampling it.**
   This is conditional Monte Carlo, also called Rao–Blackwellisation, and it is a different
   technique from a control variate: a control variate *reduces* a component's sampling variance,
   conditioning removes it **entirely**, replacing the sampled leg with its conditional
   expectation.

   The triggering condition is structural and readable off `verification.generator`, which states
   the drift and the diffusion matrix as expressions in `X[:, i]`. Component *i* qualifies when
   both hold:

   - its own drift and every entry of its diffusion row are **of degree at most one in `X[:, i]`**,
     so with the other components' paths fixed its equation integrates in closed form — a
     time-changed geometric Brownian motion, or an Ornstein–Uhlenbeck bridge;
   - **no other component's coefficients read `X[:, i]`**, so those paths can be generated first.

   Heston is the standard instance: the asset's diffusion is `sqrt(max(V, 0)) · S`, degree one in
   `S`, and the variance leg never reads `S`. So given the variance path, `S_T` is lognormal with
   known integrated variance, and the call price is a Black–Scholes formula averaged over variance
   paths — the asset leg's sampling error is gone rather than halved. The same shape covers
   Heston–Hull–White, SABR with a tractable forward leg, and any stochastic-volatility model whose
   volatility process is autonomous.

   Where it holds, propose at least one plan that conditions, and declare the integrated-out
   component in `exact_sampling_components` — it is not stepped with the supplied increments at all,
   which is what that field tests. `verifylib/pool.py` detects the condition from the declared
   generator and reports a pool where no plan takes it, so this is checked rather than suggested.

   Measured: conditional Monte Carlo appeared in **zero of the thirty** plans on the BENCHOP suite,
   and in the best unaided model draw on `sde_heston_call_feller`, where it is responsible for a 37×
   accuracy gap on the out-of-the-money strike. No path count reaches it: that pool's own
   work-precision study went 500k → 2M paths and the error got *worse*, 0.003647 → 0.004019, because
   the estimator is bias-dominated and more paths tighten the term that is not binding.

1. **`noise_structure == "additive"` OR `state_dimension > 1`** → **Milstein** adds no value for
   additive noise (dg/dX = 0) and is not well-defined for multi-D without Lévy areas. This rule
   rules Milstein out; it does **not** rule everything but EM in — see rule 0, and consider an
   implicit or split-step treatment of a stiff component.

2. **`noise_structure == "multiplicative"` AND `state_dimension == 1`** → propose both EM and Milstein. Milstein achieves strong order 1.0 vs EM's 0.5.

3. **`stiff == true`** → implicit or semi-implicit scheme required. Not yet supported; document the limitation.

4. **Variance reduction is a scheme choice too.** An antithetic pair or a control variate on a
   correlated analytically-priced instrument cuts the standard error at the same path count, and
   `functional_se` is where that shows up. It is the one axis on which two plans running the same
   scheme at the same `dt` are distinguishable without an answer key.

   **But a variance-reduced copy is not a third plan.** A wrapper does not change the pathwise
   dynamics, so the common-random-number ladder measures the same strong order on *scheme X* and on
   *scheme X plus a control variate* — identically, to the last digit — and every other statistic
   the kernel computes on the pair agrees too. Measured: **four of the ten** BENCHOP pools contained
   such a pair (orders 0.4862, 1.0078, 1.0036, 0.5445, each shared by two plans of one family), so a
   three-plan pool was a two-plan pool and nothing detected it until `verifylib/pool.py` did. Where
   a third *scheme* is available — rule 0, rule 0b, a Milstein where it is eligible, an implicit
   treatment of a stiff leg — propose that, and apply the variance reduction to one of the schemes
   already in the pool rather than spending a plan slot on it.

5. **Where the spec declares a boundary the answer is sensitive to, treat the crossing explicitly.**
   `verification.structural_facts.barrier` and `.absorbing_boundary` are declarations that the
   problem has one: a continuous barrier the payoff is monitored against, or a level the state
   reaches and stays at.

   Discrete monitoring of a continuous crossing is wrong at **O(√dt)** whatever the step scheme's
   order, because the path can cross and return between two grid points. That error does not respond
   to a better step scheme and is not what `dt` refinement buys — a second-order scheme monitored
   discretely still carries it. The treatments are a Brownian-bridge crossing probability applied
   per step, an exact absorption rule for a level the state cannot leave, or a reflection argument.

   Declare which, in the frontmatter:

   ```yaml
   boundary_treatment: brownian-bridge    # or: absorption | reflection | none
   ```

   `boundary_treatment: none` is a legitimate declaration where the plan has the argument that the
   payoff is insensitive to the crossing — say the argument in the Implementation Notes. What is not
   legitimate is silence, which is what `verifylib/pool.py` reports: a pool where a declared barrier
   or absorbing boundary is named by the spec and by no plan.

---

## Solver Code Template

```python
import numpy as np

def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    # `dW`: pre-generated Brownian INCREMENTS (already scaled by sqrt(dt)), shape
    #       (num_paths, Nt). The evaluator supplies these so that dt, dt/2 and dt/4 are
    #       all driven by ONE Brownian path — without that, MC noise swamps the
    #       discretization bias and no convergence order is measurable. When given, use
    #       them exactly and take Nt from their shape; draw no randomness of your own.
    # `observables`: {name: phi(X)} — accumulate the path integral of phi for the
    #       Dynkin identity check, which is available for every SDE with no closed form.
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))   # one block, path-major
    else:
        Nt = dW.shape[1]
        dt = T / Nt

    # Initialize: shape (num_paths,) for scalar, (num_paths, d) for vector
    X = np.full(num_paths, X_0, dtype=float)
    acc = {k: np.zeros(num_paths) for k in (observables or {})}

    for n in range(Nt):
        X_prev = X
        # --- EM step ---
        X = X + f(X) * dt + g(X) * dW[:, n]
        # --- Milstein correction (if applicable) — note it uses X_prev, not X ---
        # X = X + 0.5 * g(X_prev) * dg_dX(X_prev) * (dW[:, n]**2 - dt)
        for k, phi in (observables or {}).items():
            acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt      # trapezoid path integral

    out = {
        "terminal_paths": X,           # shape (num_paths,) or (num_paths, d)
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if observables:
        out["path_integrals"] = acc
    # The quantity of interest, wherever the spec declares one, and its own error
    # bar. Two lines, and they are the only thing that tells a kernel with no answer
    # key that your control variate was worth writing.
    payoff = np.exp(-r * T) * np.maximum(X - K, 0.0)        # the problem's own payoff
    out["functionals"] = {"price_K1": float(np.mean(payoff))}
    out["functional_se"] = {
        "price_K1": float(np.std(payoff, ddof=1) / np.sqrt(num_paths))}
    return out

if __name__ == "__main__":
    result = solve_sde(num_paths=50000, dt=0.01, T=1.0)
    print(f"Empirical mean:     {result['empirical_mean']:.6f}")
    print(f"Empirical variance: {result['empirical_variance']:.6f}")
    # confirm both hooks work — the evaluator depends on them
    dW = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 100))
    print("dW hook:", solve_sde(1000, 0.01, 1.0, dW=dW)["dt"])
    print("obs hook:", solve_sde(1000, 0.01, 1.0,
                                 observables={"x": lambda a: a})["path_integrals"]["x"].shape)
```

The evaluator will import `solve_sde` from `solver.py` and call it directly.

## Measured Convergence Orders

The evaluator measures these directly, with common random numbers, and compares them against what the
plan claimed. Reference values measured on GBM at 20 000 paths:

| Scheme | Strong order | Weak order |
|---|---|---|
| Euler–Maruyama | 0.5 (measured 0.488) | 1.0 |
| Milstein | 1.0 (measured 1.006) | 1.0 |

**The strong order is the reliable discriminator** — it is a pathwise difference, so CRN removes
almost all its variance. The weak order is a difference *of means*, so MC error does not cancel, and
for some `(scheme, φ)` pairs the leading term vanishes and the estimate returns nonsense (GBM with
Euler–Maruyama and `φ = tanh(x)` yields a spurious 4.18). See `verification_manual.md` §19 for the
noise-floor filter that removes those.

A Milstein plan measuring strong order ≈ 0.5 has a broken correction term — most often a sign error,
or applying it with `X_{n+1}` instead of `X_n`.
