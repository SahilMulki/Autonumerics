import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import solve_pde

SPEC_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "problem_spec.json"
)
with open(SPEC_PATH) as f:
    spec = json.load(f)

thresholds = spec.get("evaluation_thresholds", {})
nu = spec["parameters"]["nu"]


def rel_err(u_num, u_exact, metric="l2", mask=None):
    d = u_num - u_exact
    ref = u_exact
    if mask is not None:
        d, ref = d[mask], ref[mask]
    if metric == "l1":
        return float(np.mean(np.abs(d)) / (np.mean(np.abs(ref)) + 1e-14))
    return float(np.sqrt(np.mean(d ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-14))


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

field_errs_per_grid = []
div_norms = []
viol = []

for N in Ns:
    result = solve_pde(N)
    coords = np.meshgrid(*[result["grid"][a] for a in axes_nm if a in result["grid"]], indexing="ij")
    X, Y = coords
    t = result.get("t_final")

    exact = {
        "u": np.sin(X) * np.cos(Y) * np.exp(-2 * nu * t),
        "v": -np.cos(X) * np.sin(Y) * np.exp(-2 * nu * t),
        "p": 0.25 * (np.cos(2 * X) + np.cos(2 * Y)) * np.exp(-4 * nu * t),
    }
    num = result["fields"] if fields else {"u": result["numerical_solution"]}

    fe = {}
    for name in (fields or ["u"]):
        un = np.asarray(num[name], float).reshape(exact[name].shape)
        ue = exact[name]
        if name in gauge:
            un = un - np.mean(un)
            ue = ue - np.mean(ue)
        fe[name] = rel_err(un, ue, metric)
    field_errs_per_grid.append(fe)

    # structural diagnostic: divergence_norm (hard gate)
    dn = result.get("divergence_norm")
    if dn is None:
        # fall back: compute divergence via spectral differentiation if not provided
        u_num = np.asarray(num["u"], float)
        v_num = np.asarray(num["v"], float)
        dx = X[1, 0] - X[0, 0]
        dy = Y[0, 1] - Y[0, 0]
        dudx = np.gradient(u_num, dx, axis=0)
        dvdy = np.gradient(v_num, dy, axis=1)
        dn = float(np.sqrt(np.mean((dudx + dvdy) ** 2)))
    div_norms.append(dn)

# divergence gate check on finest grid
gate_names = {d["name"] for d in diagnostics if d.get("gate")}
if "divergence_norm" in gate_names:
    dn_fine = div_norms[-1]
    # threshold: near-zero relative to velocity magnitude scale (O(1)); use 1e-6 as "near round-off/negligible"
    if dn_fine > 1e-6:
        viol.append(f"divergence_norm={dn_fine:.3e}")

req = required or (fields or ["u"])
e_fine = max(field_errs_per_grid[-1][f] for f in req)
converged = e_fine < tol

prim = primary or (fields or ["u"])[0]
p_coarse = field_errs_per_grid[0][prim]
p_fine = field_errs_per_grid[-1][prim]
if not order_on or p_fine < 1e-9 or p_coarse < 1e-9:
    order = float("inf")
    order_ok = True
else:
    order = float(np.log2(p_coarse / p_fine))
    order_ok = order >= min_ord

passes = converged and order_ok and not viol

print("=== PDE Evaluation Results ===")
print("Scheme:            Fourier pseudo-spectral, integrating-factor RK4, Leray projection")
print(f"Resolutions:       N = {Ns}   metric = {metric}")
print(f"Per-field error:   {field_errs_per_grid[-1]} (fine grid; required max = {e_fine:.6e}, tol {tol})")
print(f"Observed order:    {order:.3f}  (primary '{prim}', min {min_ord}; {'ok' if order_ok else 'TOO LOW'})")
print(f"Divergence norm:   {div_norms} (gate near-zero)")
print(f"Constraints:       {'ok' if not viol else 'VIOLATED: ' + ', '.join(viol)}")
print(f"Overall: {'PASS' if passes else 'FAIL'}")
