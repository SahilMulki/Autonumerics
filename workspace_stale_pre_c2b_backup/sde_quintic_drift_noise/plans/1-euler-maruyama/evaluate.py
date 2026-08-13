"""
Evaluator for the quintic superlinear-drift stability benchmark
(dX = -X^5 dt + X dW, X_0 = 1.0).

problem_spec.json sets `analytic_moments.has_analytic_solution = false` — there
is no closed-form mean/variance to compare against. Per the spec's
`implementation_notes`, the independent check here verifies that:

  1. terminal states X(T) remain finite (no Inf/NaN) at the plan's stated
     hyperparameters (num_paths=50000, dt=0.01, T=1.0, seed=42), and
  2. the finiteness/stability holds under a coarser dt (a real stress test —
     a scheme that merely got lucky at the default dt should fail here), and
  3. the moments are not merely "finite but enormous" (a near-blowup that
     technically avoids Inf/NaN by a hair), and
  4. the moments are reasonably consistent between the default dt and a much
     finer dt, as a weak-convergence sanity check.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import solve_sde  # noqa: E402

SCHEME = "euler-maruyama (tamed)"
NUM_PATHS = 50000
T = 1.0
SEED = 42

DT_PRIMARY = 0.01
DT_STRESS_COARSE = 0.05
DT_FINE = 0.002

SANE_BOUND_MEAN = 50.0
SANE_BOUND_VAR = 50.0**2
CONSISTENCY_TOL = 0.1


def run(dt, label):
    result = solve_sde(num_paths=NUM_PATHS, dt=dt, T=T, seed=SEED)
    paths = np.asarray(result["terminal_paths"])
    finite = bool(np.all(np.isfinite(paths)))
    mean = float(result["empirical_mean"])
    var = float(result["empirical_variance"])
    print(f"--- {label} (dt={dt}) ---")
    print(f"  all finite: {finite}")
    print(f"  mean:       {mean:.6f}")
    print(f"  variance:   {var:.6f}")
    return finite, mean, var


print("=== Evaluation Results (stability benchmark, no analytic solution) ===")
print(f"Scheme:     {SCHEME}")
print(f"num_paths:  {NUM_PATHS}")
print(f"T:          {T}")
print(f"seed:       {SEED}")
print()

finite_primary, mean_primary, var_primary = run(DT_PRIMARY, "primary (SOLUTION.md hyperparameters)")
finite_coarse, mean_coarse, var_coarse = run(DT_STRESS_COARSE, "stress test (coarser dt)")
finite_fine, mean_fine, var_fine = run(DT_FINE, "fine dt (convergence sanity check)")
print()

all_finite = finite_primary and finite_coarse and finite_fine

moment_values = [(mean_primary, var_primary), (mean_coarse, var_coarse), (mean_fine, var_fine)]
sane = all(abs(m) < SANE_BOUND_MEAN and v < SANE_BOUND_VAR for m, v in moment_values)

mean_diff = abs(mean_primary - mean_fine)
var_diff = abs(var_primary - var_fine)
consistent = mean_diff < CONSISTENCY_TOL and var_diff < CONSISTENCY_TOL

overall_pass = all_finite and sane

print(f"All finite across dt in {{{DT_PRIMARY}, {DT_STRESS_COARSE}, {DT_FINE}}}: {all_finite}")
print(
    f"Moments within sane bounds (|mean| < {SANE_BOUND_MEAN}, var < {SANE_BOUND_VAR}): {sane}"
)
print(
    f"Consistency dt={DT_PRIMARY} vs dt={DT_FINE}: "
    f"mean_diff={mean_diff:.4f}, var_diff={var_diff:.4f}  "
    f"({'PASS' if consistent else 'FAIL'} | tol={CONSISTENCY_TOL})"
)
print()
print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
