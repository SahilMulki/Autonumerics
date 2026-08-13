import sys
import os
import numpy as np

# Allow importing solver from same directory
sys.path.insert(0, os.path.dirname(__file__))
from solver import solve_sde

# Hyperparameters
num_paths = 50000
dt = 0.01
T = 1.0
seed = 42
scheme = "Milstein"

# Parameters
theta = 1.0
sigma = 0.4

# Run solver
result = solve_sde(num_paths=num_paths, dt=dt, T=T, seed=seed)
empirical_mean = result["empirical_mean"]
empirical_var = result["empirical_variance"]

# Analytic moments at T
v_t = sigma**2 / (2 * theta) * (1 - np.exp(-2 * theta * T))
exact_mean = np.exp(v_t / 2)
exact_var = np.exp(v_t) * (np.exp(v_t) - 1)

# Relative errors
mean_rel_err = abs(empirical_mean - exact_mean) / max(abs(exact_mean), 1e-10)
var_rel_err = abs(empirical_var - exact_var) / max(abs(exact_var), 1e-10)

# Thresholds
var_threshold = 0.10
mean_threshold = 0.05
near_zero_mean_threshold = 0.01

# Pass/fail
variance_passes = var_rel_err < var_threshold
near_zero_mean = abs(exact_mean) < near_zero_mean_threshold
mean_passes = near_zero_mean or (mean_rel_err < mean_threshold)
overall_pass = variance_passes and mean_passes

print("=== Evaluation Results ===")
print(f"Scheme:              {scheme}")
print(f"dt:                  {result['dt']}")
print(f"num_paths:           {num_paths}")
print(f"T:                   {T}")
print()
print(f"Empirical mean:      {empirical_mean:.6f}")
print(f"Exact mean:          {exact_mean:.6f}")
skip_note = " (skipped (near-zero))" if near_zero_mean else ""
print(f"Mean rel. error:     {mean_rel_err:.4f}  ({'PASS' if mean_passes else 'FAIL'}{skip_note})")
print()
print(f"Empirical variance:  {empirical_var:.6f}")
print(f"Exact variance:      {exact_var:.6f}")
print(f"Variance rel. error: {var_rel_err:.4f}  ({'PASS' if variance_passes else 'FAIL'})")
print()
print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
