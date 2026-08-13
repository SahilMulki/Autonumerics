import numpy as np
from scipy.linalg import expm

from solver import solve_sde

# --- Hyperparameters (from SOLUTION.md Results section / problem_spec.json thresholds) ---
NUM_PATHS = 50000
DT = 0.001
T = 1.0
SEED = 42

VAR_THRESHOLD = 0.10
MEAN_THRESHOLD = 0.05
NEAR_ZERO_MEAN_THRESHOLD = 0.01


def exact_moments(t: float):
    F = np.array([[-2.0, 3.0], [-3.0, -2.0]])
    X0 = np.array([1.0, 1.0])

    mean = expm(F * t) @ X0
    mean_X, mean_Y = mean[0], mean[1]

    A = np.array([
        [-4.0, 6.0, 2.52],
        [-3.0, -4.0, 3.0],
        [2.16, -6.0, -4.0],
    ])
    p0 = np.array([1.0, 1.0, 1.0])
    P11, P12, P22 = expm(A * t) @ p0

    var_X = P11 - mean_X ** 2
    var_Y = P22 - mean_Y ** 2

    return mean_X, mean_Y, var_X, var_Y


def main():
    result = solve_sde(num_paths=NUM_PATHS, dt=DT, T=T, seed=SEED)

    emp_mean = np.array(result["empirical_mean"])
    emp_var = np.array(result["empirical_variance"])
    dt_used = result["dt"]

    exact_mean_X, exact_mean_Y, exact_var_X, exact_var_Y = exact_moments(T)

    exact_means = [exact_mean_X, exact_mean_Y]
    exact_vars = [exact_var_X, exact_var_Y]
    labels = ["X", "Y"]

    print("=== Evaluation Results ===")
    print("Scheme:              euler-maruyama")
    print(f"dt:                  {dt_used}")
    print(f"num_paths:           {NUM_PATHS}")
    print(f"T:                   {T}")
    print()

    overall_pass = True
    per_component = {}

    for i, label in enumerate(labels):
        empirical_mean = emp_mean[i]
        exact_mean = exact_means[i]
        empirical_var = emp_var[i]
        exact_var = exact_vars[i]

        mean_rel_err = abs(empirical_mean - exact_mean) / max(abs(exact_mean), 1e-10)
        var_rel_err = abs(empirical_var - exact_var) / max(abs(exact_var), 1e-10)

        variance_passes = var_rel_err < VAR_THRESHOLD
        near_zero_mean = abs(exact_mean) < NEAR_ZERO_MEAN_THRESHOLD
        mean_passes = near_zero_mean or (mean_rel_err < MEAN_THRESHOLD)

        component_pass = variance_passes and mean_passes
        overall_pass = overall_pass and component_pass

        per_component[label] = dict(
            empirical_mean=empirical_mean,
            exact_mean=exact_mean,
            mean_rel_err=mean_rel_err,
            mean_passes=mean_passes,
            near_zero_mean=near_zero_mean,
            empirical_var=empirical_var,
            exact_var=exact_var,
            var_rel_err=var_rel_err,
            variance_passes=variance_passes,
        )

        print(f"--- Component {label} ---")
        print(f"Empirical mean:      {empirical_mean:.6f}")
        print(f"Exact mean:          {exact_mean:.6f}")
        print(f"Mean rel. error:     {mean_rel_err:.4f}  "
              f"({'PASS' if mean_passes else 'FAIL'} | {'skipped (near-zero)' if near_zero_mean else ''})")
        print()
        print(f"Empirical variance:  {empirical_var:.6f}")
        print(f"Exact variance:      {exact_var:.6f}")
        print(f"Variance rel. error: {var_rel_err:.4f}  ({'PASS' if variance_passes else 'FAIL'})")
        print()

    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")

    return per_component, overall_pass


if __name__ == "__main__":
    main()
