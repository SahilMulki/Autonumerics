import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from solver import solve_sde

# --- Hyperparameters (from SOLUTION.md) ---
scheme = "milstein (tamed drift)"
num_paths = 50000
dt = 0.0001
T = 1.0
seed = 42

# --- Analytic moments (from problem_spec.json) ---
exact_mean = 0.676
exact_var = 1.88

# --- Thresholds (from problem_spec.json evaluation_thresholds) ---
var_threshold = 0.10
mean_threshold = 0.05
near_zero_mean_threshold = 0.01

# --- Run solver ---
result = solve_sde(num_paths=num_paths, dt=dt, T=T, seed=seed)

terminal_paths = result["terminal_paths"]
empirical_mean = result["empirical_mean"]
empirical_var = result["empirical_variance"]
actual_dt = result["dt"]

all_finite = bool(np.all(np.isfinite(terminal_paths)))

if not all_finite:
    print("=== Evaluation Results ===")
    print(f"Scheme:              {scheme}")
    print(f"dt:                  {actual_dt}")
    print(f"num_paths:           {num_paths}")
    print(f"T:                   {T}")
    print()
    print("Non-finite values detected in terminal_paths. Evaluation aborted.")
    print("Overall: FAIL")
    sys.exit(0)

mean_rel_err = abs(empirical_mean - exact_mean) / max(abs(exact_mean), 1e-10)
var_rel_err = abs(empirical_var - exact_var) / max(abs(exact_var), 1e-10)

variance_passes = var_rel_err < var_threshold

near_zero_mean = abs(exact_mean) < near_zero_mean_threshold
mean_passes = near_zero_mean or (mean_rel_err < mean_threshold)

overall_pass = variance_passes and mean_passes

print("=== Evaluation Results ===")
print(f"Scheme:              {scheme}")
print(f"dt:                  {actual_dt}")
print(f"num_paths:           {num_paths}")
print(f"T:                   {T}")
print()
print(f"Empirical mean:      {empirical_mean:.6f}")
print(f"Exact mean:          {exact_mean:.6f}")
print(f"Mean rel. error:     {mean_rel_err:.4f}  ({'PASS' if mean_passes else 'FAIL'} | {'skipped (near-zero)' if near_zero_mean else ''})")
print()
print(f"Empirical variance:  {empirical_var:.6f}")
print(f"Exact variance:      {exact_var:.6f}")
print(f"Variance rel. error: {var_rel_err:.4f}  ({'PASS' if variance_passes else 'FAIL'})")
print()
print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
