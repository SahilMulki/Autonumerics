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

1. **`noise_structure == "additive"` OR `state_dimension > 1`** → EM only. Milstein adds no value for additive noise (dg/dX = 0) and is not well-defined for multi-D without Lévy areas.

2. **`noise_structure == "multiplicative"` AND `state_dimension == 1`** → propose both EM and Milstein. Milstein achieves strong order 1.0 vs EM's 0.5.

3. **`stiff == true`** → implicit or semi-implicit scheme required. Not yet supported; document the limitation.

---

## Solver Code Template

```python
import numpy as np

def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    # Initialize: shape (num_paths,) for scalar, (num_paths, d) for vector
    X = np.full(num_paths, X_0, dtype=float)

    for _ in range(Nt):
        Z  = rng.standard_normal(num_paths)
        dW = np.sqrt(dt) * Z
        # --- EM step ---
        X = X + f(X) * dt + g(X) * dW
        # --- Milstein correction (if applicable) ---
        # X = X + 0.5 * g(X) * dg_dX(X) * (dW**2 - dt)

    return {
        "terminal_paths": X,           # shape (num_paths,) or (num_paths, d)
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }

if __name__ == "__main__":
    result = solve_sde(num_paths=50000, dt=0.01, T=1.0)
    print(f"Empirical mean:     {result['empirical_mean']:.6f}")
    print(f"Empirical variance: {result['empirical_variance']:.6f}")
```

The evaluator will import `solve_sde` from `solver.py` and call it directly.
