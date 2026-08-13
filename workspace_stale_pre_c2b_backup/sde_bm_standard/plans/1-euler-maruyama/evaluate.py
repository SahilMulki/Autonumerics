import sys
import os
import numpy as np

# Add the plan directory to path so we can import solver
plan_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, plan_dir)

from solver import solve_sde

# Hyperparameters from problem_spec.json / SOLUTION.md
num_paths = 50000
dt = 0.01
T = 1.0
seed = 42
scheme = "euler-maruyama"

# Run solver
result = solve_sde(num_paths=num_paths, dt=dt, T=T, seed=seed)

empirical_mean = result["empirical_mean"]
empirical_var = result["empirical_variance"]

# Exact moments at T (from problem_spec.json analytic_moments)
# mean_expression: "0.0"
# variance_expression: "t"
exact_mean = 0.0
exact_var = T  # = 1.0

# Relative errors
mean_rel_err = abs(empirical_mean - exact_mean) / max(abs(exact_mean), 1e-10)
var_rel_err = abs(empirical_var - exact_var) / max(abs(exact_var), 1e-10)

# Pass/fail thresholds
var_threshold = 0.10
mean_threshold = 0.05
near_zero_mean_threshold = 0.01

near_zero_mean = abs(exact_mean) < near_zero_mean_threshold
mean_passes = near_zero_mean or (mean_rel_err < mean_threshold)
variance_passes = var_rel_err < var_threshold
overall_pass = variance_passes and mean_passes

print("=== Evaluation Results ===")
print(f"Scheme:              {scheme}")
print(f"dt:                  {dt}")
print(f"num_paths:           {num_paths}")
print(f"T:                   {T}")
print()
print(f"Empirical mean:      {empirical_mean:.6f}")
print(f"Exact mean:          {exact_mean:.6f}")
mean_status = "PASS" if mean_passes else "FAIL"
skip_note = " (skipped (near-zero))" if near_zero_mean else ""
print(f"Mean rel. error:     {mean_rel_err:.4f}  ({mean_status}{skip_note})")
print()
print(f"Empirical variance:  {empirical_var:.6f}")
print(f"Exact variance:      {exact_var:.6f}")
var_status = "PASS" if variance_passes else "FAIL"
print(f"Variance rel. error: {var_rel_err:.4f}  ({var_status})")
print()
print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
