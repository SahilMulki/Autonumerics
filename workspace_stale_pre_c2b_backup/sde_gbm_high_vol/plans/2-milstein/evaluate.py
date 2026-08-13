import sys
import os
import numpy as np

# Add plan directory to path for solver import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import solve_sde

# Problem parameters
X_0 = 1.0
mu = 0.05
sigma = 1.0
T = 1.0

# Hyperparameters from SOLUTION.md (updated: 500000 paths as per latest solver results)
num_paths = 500000
dt = 0.005
seed = 42

# Run solver
result = solve_sde(num_paths=num_paths, dt=dt, T=T, seed=seed)

empirical_mean = result["empirical_mean"]
empirical_var  = result["empirical_variance"]

# Exact moments at T
t = T
exact_mean = X_0 * np.exp(mu * t)
exact_var  = X_0**2 * np.exp(2 * mu * t) * (np.exp(sigma**2 * t) - 1)

# Relative errors
mean_rel_err = abs(empirical_mean - exact_mean) / max(abs(exact_mean), 1e-10)
var_rel_err  = abs(empirical_var  - exact_var)  / max(abs(exact_var),  1e-10)

# Thresholds
mean_threshold = 0.05
var_threshold  = 0.10

near_zero_mean   = abs(exact_mean) < 0.01
variance_passes  = var_rel_err < var_threshold
mean_passes      = near_zero_mean or (mean_rel_err < mean_threshold)
overall_pass     = variance_passes and mean_passes

scheme = "Milstein"

print("=== Evaluation Results ===")
print(f"Scheme:              {scheme}")
print(f"dt:                  {dt}")
print(f"num_paths:           {num_paths}")
print(f"T:                   {T}")
print()
print(f"Empirical mean:      {empirical_mean:.6f}")
print(f"Exact mean:          {exact_mean:.6f}")
mean_status = "skipped (near-zero)" if near_zero_mean else ("PASS" if mean_passes else "FAIL")
print(f"Mean rel. error:     {mean_rel_err:.4f}  ({'PASS' if mean_passes else 'FAIL'} | {mean_status if near_zero_mean else ''})")
print()
print(f"Empirical variance:  {empirical_var:.6f}")
print(f"Exact variance:      {exact_var:.6f}")
print(f"Variance rel. error: {var_rel_err:.4f}  ({'PASS' if variance_passes else 'FAIL'})")
print()
print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
