import sys
import numpy as np

sys.path.insert(0, "/Users/sahilmulki/Autonumerics/workspace/sde_oscillator_long_horizon/plans/1-euler-maruyama")
from solver import solve_sde

# Hyperparameters from SOLUTION.md / problem_spec.json
num_paths = 50000
dt = 0.001
T = 10 * np.pi
seed = 42

scheme = "euler-maruyama"

# Run solver
result = solve_sde(num_paths=num_paths, dt=dt, T=T, seed=seed)

empirical_mean = np.array(result["empirical_mean"])      # [mean_X, mean_Y]
empirical_var  = np.array(result["empirical_variance"])  # [var_X, var_Y]

# Analytic moments at t = T
X_0 = 1.0
Y_0 = 0.0
sigma = 0.3
t = T

exact_mean_X = X_0 * np.cos(t) + Y_0 * np.sin(t)
exact_mean_Y = -X_0 * np.sin(t) + Y_0 * np.cos(t)
exact_var_X  = sigma**2 / 2 * (t - np.sin(t) * np.cos(t))
exact_var_Y  = sigma**2 / 2 * (t + np.sin(t) * np.cos(t))

exact_mean = np.array([exact_mean_X, exact_mean_Y])
exact_var  = np.array([exact_var_X,  exact_var_Y])

# Thresholds
var_threshold  = 0.10
mean_threshold = 0.05
near_zero_threshold = 0.01

components = ["X", "Y"]
all_pass = True

print("=== Evaluation Results ===")
print(f"Scheme:              {scheme}")
print(f"dt:                  {dt}")
print(f"num_paths:           {num_paths}")
print(f"T:                   {T:.6f}")
print()

for i, comp in enumerate(components):
    em = empirical_mean[i]
    ev = empirical_var[i]
    xm = exact_mean[i]
    xv = exact_var[i]

    mean_rel_err = abs(em - xm) / max(abs(xm), 1e-10)
    var_rel_err  = abs(ev - xv) / max(abs(xv),  1e-10)

    near_zero_mean    = abs(xm) < near_zero_threshold
    variance_passes   = var_rel_err  < var_threshold
    mean_passes       = near_zero_mean or (mean_rel_err < mean_threshold)

    comp_pass = variance_passes and mean_passes
    if not comp_pass:
        all_pass = False

    mean_status = "skipped (near-zero)" if near_zero_mean else ("PASS" if mean_passes else "FAIL")
    var_status  = "PASS" if variance_passes else "FAIL"

    print(f"--- Component {comp} ---")
    print(f"Empirical mean:      {em:.6f}")
    print(f"Exact mean:          {xm:.6f}")
    print(f"Mean rel. error:     {mean_rel_err:.4f}  ({mean_status})")
    print()
    print(f"Empirical variance:  {ev:.6f}")
    print(f"Exact variance:      {xv:.6f}")
    print(f"Variance rel. error: {var_rel_err:.4f}  ({var_status})")
    print(f"Component {comp}: {'PASS' if comp_pass else 'FAIL'}")
    print()

print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
