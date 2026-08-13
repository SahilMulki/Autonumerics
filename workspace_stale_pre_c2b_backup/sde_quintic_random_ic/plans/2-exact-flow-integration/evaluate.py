import numpy as np
from scipy import integrate

from solver import solve_sde

# --- Hyperparameters (from SOLUTION.md / problem_spec.json defaults) ---
NUM_PATHS = 50000
DT = 0.001
T = 1.0
SEED = 42
SCHEME = "euler-maruyama (exact flow-map closed form)"

SIGMA_BAR = 1.0 / 3.0

# --- Thresholds (from problem_spec.json -> evaluation_thresholds) ---
VAR_THRESHOLD = 0.10
MEAN_THRESHOLD = 0.05
NEAR_ZERO_MEAN_THRESHOLD = 0.01


def exact_variance_at_T(t: float, sigma_bar: float) -> float:
    integrand = lambda xi: (
        (xi**2 / np.sqrt(1.0 + 4.0 * t * xi**4))
        * np.exp(-(xi**2) / (2.0 * sigma_bar**2))
        / (np.sqrt(2.0 * np.pi) * sigma_bar)
    )
    value, _ = integrate.quad(integrand, -np.inf, np.inf)
    return value


def main():
    result = solve_sde(num_paths=NUM_PATHS, dt=DT, T=T, seed=SEED)

    empirical_mean = result["empirical_mean"]
    empirical_var = result["empirical_variance"]

    exact_mean = 0.0
    exact_var = exact_variance_at_T(T, SIGMA_BAR)

    finite = np.isfinite(empirical_mean) and np.isfinite(empirical_var)
    if not finite:
        print("=== Evaluation Results ===")
        print("Non-finite output detected (NaN/Inf). FAIL.")
        return

    mean_rel_err = abs(empirical_mean - exact_mean) / max(abs(exact_mean), 1e-10)
    var_rel_err = abs(empirical_var - exact_var) / max(abs(exact_var), 1e-10)

    variance_passes = var_rel_err < VAR_THRESHOLD

    near_zero_mean = abs(exact_mean) < NEAR_ZERO_MEAN_THRESHOLD
    mean_passes = near_zero_mean or (mean_rel_err < MEAN_THRESHOLD)

    overall_pass = variance_passes and mean_passes

    print("=== Evaluation Results ===")
    print(f"Scheme:              {SCHEME}")
    print(f"dt:                  {result['dt']}")
    print(f"num_paths:           {NUM_PATHS}")
    print(f"T:                   {T}")
    print()
    print(f"Empirical mean:      {empirical_mean:.6f}")
    print(f"Exact mean:          {exact_mean:.6f}")
    print(
        f"Mean rel. error:     {mean_rel_err:.4f}  "
        f"({'PASS' if mean_passes else 'FAIL'} | "
        f"{'skipped (near-zero)' if near_zero_mean else ''})"
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
