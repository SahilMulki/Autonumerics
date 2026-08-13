import numpy as np
from scipy.integrate import quad

from solver import solve_sde

# --- Hyperparameters (from SOLUTION.md) ---
NUM_PATHS = 50000
DT = 0.001
T = 1.0
SEED = 42

# --- Problem parameters (from problem_spec.json) ---
SIGMA_BAR = 1.0 / 3.0

VAR_THRESHOLD = 0.10
MEAN_THRESHOLD = 0.05
NEAR_ZERO_MEAN_THRESHOLD = 0.01


def exact_moments(t: float, sigma_bar: float) -> tuple[float, float]:
    """Exact mean (=0 by symmetry) and variance (Gaussian quadrature) at time t."""
    exact_mean = 0.0

    def integrand(xi):
        return (xi ** 2 / np.sqrt(1.0 + 4.0 * t * xi ** 4)) * np.exp(
            -xi ** 2 / (2.0 * sigma_bar ** 2)
        ) / (np.sqrt(2.0 * np.pi) * sigma_bar)

    exact_variance, _ = quad(integrand, -np.inf, np.inf)
    return exact_mean, exact_variance


def main():
    result = solve_sde(num_paths=NUM_PATHS, dt=DT, T=T, seed=SEED)

    empirical_mean = result["empirical_mean"]
    empirical_var = result["empirical_variance"]
    dt_used = result["dt"]

    finite_ok = np.isfinite(empirical_mean) and np.isfinite(empirical_var)
    if not finite_ok:
        print("=== Evaluation Results ===")
        print("Scheme:              tamed euler-maruyama")
        print(f"dt:                  {dt_used}")
        print(f"num_paths:           {NUM_PATHS}")
        print(f"T:                   {T}")
        print()
        print("Non-finite output detected (NaN/Inf). FAIL.")
        return

    exact_mean, exact_var = exact_moments(T, SIGMA_BAR)

    mean_rel_err = abs(empirical_mean - exact_mean) / max(abs(exact_mean), 1e-10)
    var_rel_err = abs(empirical_var - exact_var) / max(abs(exact_var), 1e-10)

    variance_passes = var_rel_err < VAR_THRESHOLD

    near_zero_mean = abs(exact_mean) < NEAR_ZERO_MEAN_THRESHOLD
    mean_passes = near_zero_mean or (mean_rel_err < MEAN_THRESHOLD)

    overall_pass = variance_passes and mean_passes

    print("=== Evaluation Results ===")
    print("Scheme:              tamed euler-maruyama")
    print(f"dt:                  {dt_used}")
    print(f"num_paths:           {NUM_PATHS}")
    print(f"T:                   {T}")
    print()
    print(f"Empirical mean:      {empirical_mean:.6f}")
    print(f"Exact mean:          {exact_mean:.6f}")
    mean_note = "skipped (near-zero)" if near_zero_mean else ""
    print(
        f"Mean rel. error:     {mean_rel_err:.4f}  "
        f"({'PASS' if mean_passes else 'FAIL'} | {mean_note})"
    )
    print()
    print(f"Empirical variance:  {empirical_var:.6f}")
    print(f"Exact variance:      {exact_var:.6f}")
    print(
        f"Variance rel. error: {var_rel_err:.4f}  "
        f"({'PASS' if variance_passes else 'FAIL'})"
    )
    print()
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
