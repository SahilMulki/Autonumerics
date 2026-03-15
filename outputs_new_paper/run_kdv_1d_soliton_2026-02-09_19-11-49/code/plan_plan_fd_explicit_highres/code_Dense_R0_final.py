```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract domain and discretization parameters ---
    domain = pde_spec["domain"]
    x_min, x_max = domain["bounds"]["x"]
    Nx = plan["spatial_discretization"]["Nx"]
    dx = (x_max - x_min) / Nx
    x = np.linspace(x_min, x_max - dx, Nx)  # periodic grid: last point = x_max - dx

    # --- Time stepping parameters ---
    t_final = plan["time_stepping"].get("t_final", 1.0)
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt by CFL for KdV: dt < C * dx^3 (very restrictive for explicit)
        # Use C = 0.4 as a safe guess
        dt = 0.4 * dx**3
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt  # adjust dt to hit t_final exactly
    else:
        dt = t_final / Nt
    t_array = np.linspace(0, t_final, Nt + 1)

    # --- Initial condition ---
    # u(x,0) = 0.5 * (1 / np.cosh(0.5 * x))**2
    u = 0.5 * (1 / np.cosh(0.5 * x))**2

    # --- Helper: periodic finite difference operators (4th order centered) ---
    def periodic_diff(arr, order=1):
        # 4th order centered finite difference for 1st and 3rd derivatives
        # https://en.wikipedia.org/wiki/Finite_difference_coefficient
        if order == 1:
            # 4th order centered: f'(x) ≈ (f_{-2} - 8 f_{-1} + 8 f_{+1} - f_{+2}) / (12 dx)
            return (np.roll(arr, -2) - 8 * np.roll(arr, -1) + 8 * np.roll(arr, 1) - np.roll(arr, 2)) / (12 * dx)
        elif order == 3:
            # 4th order centered for 3rd derivative:
            # f'''(x) ≈ (-f_{-2} + 2 f_{-1} - 2 f_{+1} + f_{+2}) / (2 dx^3)
            return (-np.roll(arr, -2) + 2 * np.roll(arr, -1) - 2 * np.roll(arr, 1) + np.roll(arr, 2)) / (2 * dx**3)
        else:
            raise ValueError("Only 1st and 3rd derivatives supported.")

    # --- RHS function for KdV: u_t = -6 u u_x - u_xxx ---
    def kdv_rhs(u):
        u_x = periodic_diff(u, order=1)
        u_xxx = periodic_diff(u, order=3)
        return -6 * u * u_x - u_xxx

    # --- Time stepping: RK4 ---
    # Memory safety: only store current u (not full history)
    for n in range(Nt):
        k1 = kdv_rhs(u)
        k2 = kdv_rhs(u + 0.5 * dt * k1)
        k3 = kdv_rhs(u + 0.5 * dt * k2)
        k4 = kdv_rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # --- Compute residual at final time ---
    # Residual: R = u_t + 6 u u_x + u_xxx ≈ (u_final - u_prev) / dt + 6 u u_x + u_xxx
    # For best accuracy, recompute u_t at final step using one backward Euler step:
    # u_t ≈ (u_final - u_prev) / dt
    # But since we don't have u_prev, use the PDE: R = kdv_rhs(u_final) + u_t (should be ~0)
    # Instead, for explicit RK4, estimate u_t as kdv_rhs(u_final)
    # So residual = u_t + 6 u u_x + u_xxx = kdv_rhs(u_final) + 6 u u_x + u_xxx = 0
    # But this is just 2 * kdv_rhs(u_final), which is not informative.
    # Instead, use a finite difference in time:
    # Save u_prev at last step to estimate u_t.
    # So, rerun last step to get u_prev:
    u_prev = u.copy()
    # Step backward by -dt (approximate):
    # (This is not exact, but for residual estimation it's acceptable)
    # Actually, let's store u_prev during the last step:
    # So, rerun the time loop, but only store u_prev at last step:
    u_temp = 0.5 * (1 / np.cosh(0.5 * x))**2
    for n in range(Nt - 1):
        k1 = kdv_rhs(u_temp)
        k2 = kdv_rhs(u_temp + 0.5 * dt * k1)
        k3 = kdv_rhs(u_temp + 0.5 * dt * k2)
        k4 = kdv_rhs(u_temp + dt * k3)
        u_temp = u_temp + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    u_prev = u_temp.copy()  # u at t = t_final - dt

    # Now, estimate u_t ≈ (u - u_prev) / dt at t_final
    u_t_num = (u - u_prev) / dt
    u_x = periodic_diff(u, order=1)
    u_xxx = periodic_diff(u, order=3)
    residual_grid = u_t_num + 6 * u * u_x + u_xxx

    # --- Output ---
    result = {
        "u": u.copy(),  # shape (Nx,)
        "coords": {"x": x.copy()},
        "t": t_array,
        "residual": residual_grid.copy()
    }
    return result
```
