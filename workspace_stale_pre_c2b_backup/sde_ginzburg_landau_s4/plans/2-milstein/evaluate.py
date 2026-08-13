import numpy as np

from solver import solve_sde

# --- Hyperparameters (from SOLUTION.md "Results" section) ---
NUM_PATHS = 50000
DT = 0.0005
T = 1.0
SEED = 42

SCHEME = "milstein (tamed-drift Milstein)"

# --- Analytic (reference) moments from problem_spec.json -> analytic_moments ---
# mean_expression = "0.659", variance_expression = "1.117"
# (Reference-quality Monte-Carlo estimates of the exact pathwise solution; no closed form.)
exact_mean = 0.659
exact_var = 1.117

# --- Evaluation thresholds from problem_spec.json -> evaluation_thresholds ---
var_threshold = 0.10
mean_threshold = 0.05
near_zero_mean_threshold = 0.01


def main():
    result = solve_sde(num_paths=NUM_PATHS, dt=DT, T=T, seed=SEED)

    terminal_paths = result["terminal_paths"]
    all_finite = bool(np.isfinite(terminal_paths).all())

    empirical_mean = result["empirical_mean"]
    empirical_var = result["empirical_variance"]
    dt_used = result["dt"]

    mean_rel_err = abs(empirical_mean - exact_mean) / max(abs(exact_mean), 1e-10)
    var_rel_err = abs(empirical_var - exact_var) / max(abs(exact_var), 1e-10)

    variance_passes = var_rel_err < var_threshold

    near_zero_mean = abs(exact_mean) < near_zero_mean_threshold
    mean_passes = near_zero_mean or (mean_rel_err < mean_threshold)

    overall_pass = variance_passes and mean_passes and all_finite

    print("=== Evaluation Results ===")
    print(f"Scheme:              {SCHEME}")
    print(f"dt:                  {dt_used}")
    print(f"num_paths:           {NUM_PATHS}")
    print(f"T:                   {T}")
    print(f"All finite:          {all_finite}")
    print()
    print(f"Empirical mean:      {empirical_mean:.6f}")
    print(f"Exact mean:          {exact_mean:.6f}")
    mean_skip_note = "skipped (near-zero)" if near_zero_mean else ""
    print(f"Mean rel. error:     {mean_rel_err:.4f}  ({'PASS' if mean_passes else 'FAIL'} | {mean_skip_note})")
    print()
    print(f"Empirical variance:  {empirical_var:.6f}")
    print(f"Exact variance:      {exact_var:.6f}")
    print(f"Variance rel. error: {var_rel_err:.4f}  ({'PASS' if variance_passes else 'FAIL'})")
    print()
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")

    return {
        "mean_rel_err": mean_rel_err,
        "var_rel_err": var_rel_err,
        "mean_passes": mean_passes,
        "variance_passes": variance_passes,
        "overall_pass": overall_pass,
        "all_finite": all_finite,
    }


if __name__ == "__main__":
    main()
