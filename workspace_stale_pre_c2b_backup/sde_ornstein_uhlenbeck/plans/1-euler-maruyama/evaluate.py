import sys
import numpy as np

sys.path.insert(0, "/Users/sahilmulki/Autonumerics/workspace/sde_ornstein_uhlenbeck/plans/1-euler-maruyama")
from solver import solve_sde

# Hyperparameters from SOLUTION.md / problem_spec.json
num_paths = 50000
dt        = 0.01
T         = 1.0
seed      = 42

# OU parameters
X_0   = 2.0
theta = 1.5
mu    = 0.0
sigma = 0.5

# Run solver
result = solve_sde(num_paths=num_paths, dt=dt, T=T, seed=seed)

empirical_mean = result["empirical_mean"]
empirical_var  = result["empirical_variance"]
dt_used        = result["dt"]

# Analytic moments at T
t = T
exact_mean = mu + (X_0 - mu) * np.exp(-theta * t)
exact_var  = sigma**2 / (2 * theta) * (1 - np.exp(-2 * theta * t))

# Relative errors
mean_rel_err = abs(empirical_mean - exact_mean) / max(abs(exact_mean), 1e-10)
var_rel_err  = abs(empirical_var  - exact_var)  / max(abs(exact_var),  1e-10)

# Pass/fail thresholds
var_threshold  = 0.10
mean_threshold = 0.05
near_zero_mean_threshold = 0.01

variance_passes = var_rel_err < var_threshold
near_zero_mean  = abs(exact_mean) < near_zero_mean_threshold
mean_passes     = near_zero_mean or (mean_rel_err < mean_threshold)
overall_pass    = variance_passes and mean_passes

scheme = "Euler-Maruyama"

print("=== Evaluation Results ===")
print(f"Scheme:              {scheme}")
print(f"dt:                  {dt_used}")
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
