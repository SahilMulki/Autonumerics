import sys
import numpy as np

sys.path.insert(0, "/Users/sahilmulki/Autonumerics/workspace/sde_linear_additive/plans/1-euler-maruyama")
from solver import solve_sde

# Hyperparameters from problem_spec.json / SOLUTION.md
num_paths = 50000
dt = 0.01
T = 1.0
seed = 42
scheme = "Euler-Maruyama"

# Run solver
result = solve_sde(num_paths=num_paths, dt=dt, T=T, seed=seed)
empirical_mean = result["empirical_mean"]
empirical_var  = result["empirical_variance"]

# Analytic moments — linear_sde_additive
# Parameters
X_0 = 0.0
a = 2.0
b = -1.0
c = 0.5
t = T

exact_mean = np.exp(b * t) * (X_0 + a / b) - a / b
exact_var  = c**2 / (2 * b) * (np.exp(2 * b * t) - 1)

# Relative errors
mean_rel_err = abs(empirical_mean - exact_mean) / max(abs(exact_mean), 1e-10)
var_rel_err  = abs(empirical_var  - exact_var)  / max(abs(exact_var),  1e-10)

# Thresholds
var_threshold  = 0.10
mean_threshold = 0.05
near_zero_threshold = 0.01

variance_passes = var_rel_err < var_threshold
near_zero_mean  = abs(exact_mean) < near_zero_threshold
mean_passes     = near_zero_mean or (mean_rel_err < mean_threshold)
overall_pass    = variance_passes and mean_passes

print("=== Evaluation Results ===")
print(f"Scheme:              {scheme}")
print(f"dt:                  {result['dt']}")
print(f"num_paths:           {num_paths}")
print(f"T:                   {T}")
print()
print(f"Empirical mean:      {empirical_mean:.6f}")
print(f"Exact mean:          {exact_mean:.6f}")
skipped_str = " (skipped: near-zero)" if near_zero_mean else ""
print(f"Mean rel. error:     {mean_rel_err:.4f}  ({'PASS' if mean_passes else 'FAIL'}{skipped_str})")
print()
print(f"Empirical variance:  {empirical_var:.6f}")
print(f"Exact variance:      {exact_var:.6f}")
print(f"Variance rel. error: {var_rel_err:.4f}  ({'PASS' if variance_passes else 'FAIL'})")
print()
print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
