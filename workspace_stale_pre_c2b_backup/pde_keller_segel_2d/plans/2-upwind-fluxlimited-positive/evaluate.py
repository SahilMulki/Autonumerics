import json
import os

import numpy as np

from solver import solve_pde

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_PATH = os.path.join(HERE, "..", "..", "problem_spec.json")

with open(SPEC_PATH) as f:
    spec = json.load(f)

thresholds = spec.get("evaluation_thresholds", {})


def rel_err(u_num, u_exact, metric="l2", mask=None):
    d = u_num - u_exact
    ref = u_exact
    if mask is not None:
        d, ref = d[mask], ref[mask]
    if metric == "l1":
        return float(np.mean(np.abs(d)) / (np.mean(np.abs(ref)) + 1e-14))
    return float(np.sqrt(np.mean(d**2)) / (np.sqrt(np.mean(ref**2)) + 1e-14))


grid_N = thresholds.get("grid_N", 64)
min_ord = thresholds.get("min_spatial_order", 1.0)
order_on = thresholds.get("order_check", True)
tol = thresholds.get("rel_l2_err_max", 0.01)
metric = thresholds.get("metric", "l2")
axes_nm = thresholds.get("axes", ["x", "y", "z"])
fields = thresholds.get("fields")
primary = thresholds.get("primary_field")
required = thresholds.get("required_fields")
gauge = set(thresholds.get("gauge_fields", []))
diagnostics = thresholds.get("diagnostics", [])

Ns = [grid_N, 2 * grid_N] if order_on else [grid_N]

analytic = spec["analytic_solution"]["fields"]


def _rho_exact(X, Y, t):
    return eval(analytic["rho"], {"np": np}, {"X": X, "Y": Y, "t": t})


def _c_exact(X, Y, t):
    return eval(analytic["c"], {"np": np}, {"X": X, "Y": Y, "t": t})


field_errs_per_grid = []
viol = []
results = []

for N in Ns:
    result = solve_pde(N)
    results.append(result)
    x = result["grid"]["x"]
    y = result["grid"]["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    t = result.get("t_final")

    exact = {
        "rho": _rho_exact(X, Y, t),
        "c": _c_exact(X, Y, t),
    }
    num = result["fields"]

    fe = {}
    for name in fields:
        un = np.asarray(num[name], float).reshape(exact[name].shape)
        ue = exact[name]
        if name in gauge:
            un = un - np.mean(un)
            ue = ue - np.mean(ue)
        fe[name] = rel_err(un, ue, metric)
    field_errs_per_grid.append(fe)

# --- structural diagnostics on the finest grid ---
fine_result = results[-1]
fine_num = fine_result["fields"]
for diag in diagnostics:
    name = diag.get("name")
    gate = diag.get("gate", False)
    if name == "min_density":
        min_rho = float(np.min(fine_num["rho"]))
        min_c = float(np.min(fine_num["c"]))
        ok = (min_rho >= 0.0) and (min_c >= 0.0)
        if gate and not ok:
            viol.append(f"min_density (min_rho={min_rho:.6e}, min_c={min_c:.6e} < 0)")

req = required or fields
e_fine = max(field_errs_per_grid[-1][f] for f in req)
converged = e_fine < tol

primary_field = primary or fields[0]
p_coarse = field_errs_per_grid[0][primary_field]
p_fine = field_errs_per_grid[-1][primary_field]

if not order_on or p_fine < 1e-9 or p_coarse < 1e-9:
    order = float("inf")
    order_ok = True
else:
    order = float(np.log2(p_coarse / p_fine))
    order_ok = order >= min_ord

passes = converged and order_ok and not viol

print("=== PDE Evaluation Results ===")
print("Scheme:            finite-volume-upwind (van Leer flux-limited chemotaxis, SSP-RK2)")
print(f"Resolutions:       N = {Ns}   metric = {metric}")
print(
    f"Per-field error:   {field_errs_per_grid[-1]} (fine grid; required max = {e_fine:.6e}, tol {tol})"
)
print(
    f"Observed order:    {order:.3f}  (primary '{primary_field}', min {min_ord}; "
    f"{'ok' if order_ok else 'TOO LOW'})"
)
print(f"Constraints:       {'ok' if not viol else 'VIOLATED: ' + ', '.join(viol)}")
print(f"Overall: {'PASS' if passes else 'FAIL'}")

print()
print("Full per-grid errors:", field_errs_per_grid)
