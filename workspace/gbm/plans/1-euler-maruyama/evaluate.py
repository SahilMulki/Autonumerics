import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from solver import solve_sde

mu = 0.1
sigma = 0.2
X0 = 1.0
T = 1.0

result = solve_sde(num_paths=50000, dt=0.01, T=T, seed=42)

empirical_mean = result["empirical_mean"]
empirical_var = result["empirical_variance"]

exact_mean = X0 * np.exp(mu * T)
exact_var = X0**2 * np.exp(2 * mu * T) * (np.exp(sigma**2 * T) - 1)

mean_rel_err = abs(empirical_mean - exact_mean) / max(abs(exact_mean), 1e-10)
var_rel_err = abs(empirical_var - exact_var) / max(abs(exact_var), 1e-10)

var_threshold = 0.10
mean_threshold = 0.05
near_zero_mean_threshold = 0.01

variance_passes = var_rel_err < var_threshold
near_zero_mean = abs(exact_mean) < near_zero_mean_threshold
mean_passes = near_zero_mean or (mean_rel_err < mean_threshold)
overall_pass = variance_passes and mean_passes

scheme = "euler-maruyama"
print("=== Evaluation Results ===")
print(f"Scheme:              {scheme}")
print(f"dt:                  {result['dt']:.6f}")
print(f"num_paths:           {result['num_paths']}")
print(f"T:                   {T}")
print()
print(f"Empirical mean:      {empirical_mean:.6f}")
print(f"Exact mean:          {exact_mean:.6f}")
print(f"Mean rel. error:     {mean_rel_err:.4f}  ({'PASS' if mean_passes else 'FAIL'}{' | skipped (near-zero)' if near_zero_mean else ''})")
print()
print(f"Empirical variance:  {empirical_var:.6f}")
print(f"Exact variance:      {exact_var:.6f}")
print(f"Variance rel. error: {var_rel_err:.4f}  ({'PASS' if variance_passes else 'FAIL'})")
print()
print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
