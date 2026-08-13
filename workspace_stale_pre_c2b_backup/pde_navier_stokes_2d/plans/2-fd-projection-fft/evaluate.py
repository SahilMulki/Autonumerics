import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import solve_pde  # noqa: E402

PLAN_DIR = os.path.dirname(os.path.abspath(__file__))
PROBLEM_DIR = os.path.abspath(os.path.join(PLAN_DIR, "..", ".."))

with open(os.path.join(PROBLEM_DIR, "problem_spec.json")) as f:
    spec = json.load(f)

thresholds = spec.get("evaluation_thresholds", {})
nu = spec.get("parameters", {}).get("nu", 0.1)


def rel_err(u_num, u_exact, metric="l2", mask=None):
    d = u_num - u_exact
    ref = u_exact
    if mask is not None:
        d, ref = d[mask], ref[mask]
    if metric == "l1":
        return float(np.mean(np.abs(d)) / (np.mean(np.abs(ref)) + 1e-14))
    return float(np.sqrt(np.mean(d ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-14))


def spectral_divergence(u, v, x, y):
    """Discrete divergence via spectral (FFT) derivatives -- appropriate on the
    periodic box and independent of whatever internal operators the solver used."""
    N = u.shape[0]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    dudx = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(u)))
    dvdy = np.real(np.fft.ifft2(1j * KY * np.fft.fft2(v)))
    return dudx + dvdy


grid_N = thresholds.get("grid_N", 64)
min_ord = thresholds.get("min_spatial_order", 1.0)
order_on = thresholds.get("order_check", True)
tol = thresholds.get("rel_l2_err_max", 0.01)
metric = thresholds.get("metric", "l2")
axes_nm = thresholds.get("axes", ["x", "y", "z"])
fields = thresholds.get("fields")
primary = thresholds.get("primary_field", "u")
required = thresholds.get("required_fields", ["u", "v"])
gauge = set(thresholds.get("gauge_fields", []))
diagnostics = thresholds.get("diagnostics", [])
gate_names = {d["name"] for d in diagnostics if d.get("gate")}

Ns = [grid_N, 2 * grid_N] if order_on else [grid_N]

field_errs_per_grid = []
viol = []
div_norms = {}

for N in Ns:
    result = solve_pde(N)
    x = result["grid"]["x"]
    y = result["grid"]["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    t = result.get("t_final", 0.5)

    exact = {
        "u": np.sin(X) * np.cos(Y) * np.exp(-2 * nu * t),
        "v": -np.cos(X) * np.sin(Y) * np.exp(-2 * nu * t),
        "p": 0.25 * (np.cos(2 * X) + np.cos(2 * Y)) * np.exp(-4 * nu * t),
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

    # --- structural diagnostic: divergence-free velocity (hard gate) ---
    u_num = np.asarray(num["u"], float)
    v_num = np.asarray(num["v"], float)
    div = spectral_divergence(u_num, v_num, x, y)
    vel_scale = np.sqrt(np.mean(u_num ** 2) + np.mean(v_num ** 2)) + 1e-14
    div_rel = float(np.sqrt(np.mean(div ** 2)) / vel_scale)
    div_norms[N] = div_rel
    if "divergence_norm" in gate_names and div_rel > 1e-3:
        viol.append(f"divergence_norm(N={N})={div_rel:.3e}")

req = required or (fields or ["u"])
e_fine = max(field_errs_per_grid[-1][f] for f in req)
converged = e_fine < tol
p_coarse = field_errs_per_grid[0][primary]
p_fine = field_errs_per_grid[-1][primary]
if not order_on or p_fine < 1e-9 or p_coarse < 1e-9:
    order = float("inf")
    order_ok = True
else:
    order = float(np.log2(p_coarse / p_fine))
    order_ok = order >= min_ord

passes = converged and order_ok and not viol

print("=== PDE Evaluation Results ===")
print("Scheme:            fd-projection-fft (Chorin projection, FFT pressure-Poisson, RK2)")
print(f"Resolutions:       N = {Ns}   metric = {metric}")
print(f"Per-field error:   {field_errs_per_grid[-1]} (fine grid; required max = {e_fine:.6e}, tol {tol})")
print(f"Observed order:    {order:.3f}  (primary '{primary}', min {min_ord}; {'ok' if order_ok else 'TOO LOW'})")
print(f"Divergence norms:  {div_norms}")
print(f"Constraints:       {'ok' if not viol else 'VIOLATED: ' + ', '.join(viol)}")
print(f"Overall: {'PASS' if passes else 'FAIL'}")
