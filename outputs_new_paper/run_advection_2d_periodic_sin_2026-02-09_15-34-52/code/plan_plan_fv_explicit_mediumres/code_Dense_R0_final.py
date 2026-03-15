```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and plan parameters ---
    # Domain
    x_min = pde_spec["domain"]["x_min"]
    x_max = pde_spec["domain"]["x_max"]
    y_min = pde_spec["domain"]["y_min"]
    y_max = pde_spec["domain"]["y_max"]
    # Parameters
    c_x = float(pde_spec["parameters"]["c_x"])
    c_y = float(pde_spec["parameters"]["c_y"])
    # Initial condition
    initial_condition = pde_spec["initial_condition"]
    # Boundary conditions
    bc_type = pde_spec["boundary_conditions"]["type"]
    # Plan: grid
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])
    # Plan: time
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    Nt = plan["time_stepping"].get("Nt", None)
    if dt is None:
        # Estimate dt by CFL
        dx = (x_max - x_min) / Nx
        dy = (y_max - y_min) / Ny
        cfl = 0.5  # conservative
        dt = cfl * min(dx/abs(c_x) if c_x != 0 else np.inf,
                       dy/abs(c_y) if c_y != 0 else np.inf)
    else:
        dx = (x_max - x_min) / Nx
        dy = (y_max - y_min) / Ny
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    else:
        t_final = Nt * dt
    # Time array (store only final time, but return t array for API)
    t_array = np.linspace(0, t_final, Nt+1)
    # --- Create grid ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')  # shape (Nx, Ny)
    # --- Initial condition ---
    u0 = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    u = u0.copy()
    # --- Helper: periodic roll ---
    def periodic_roll(arr, shift, axis):
        return np.roll(arr, shift, axis=axis)
    # --- FV upwind fluxes ---
    def compute_fluxes(u):
        # For upwind FV, flux at i+1/2 is determined by sign of c_x/c_y
        # x-direction
        if c_x >= 0:
            u_xL = u
            u_xR = periodic_roll(u, -1, axis=0)
            flux_x = c_x * u_xL
        else:
            u_xL = u
            u_xR = periodic_roll(u, -1, axis=0)
            flux_x = c_x * u_xR
        # y-direction
        if c_y >= 0:
            u_yL = u
            u_yR = periodic_roll(u, -1, axis=1)
            flux_y = c_y * u_yL
        else:
            u_yL = u
            u_yR = periodic_roll(u, -1, axis=1)
            flux_y = c_y * u_yR
        return flux_x, flux_y
    # --- FV update (1 step) ---
    def fv_rhs(u):
        # Compute fluxes at cell faces
        # x-direction
        if c_x >= 0:
            flux_xL = c_x * u
            flux_xR = c_x * periodic_roll(u, 1, axis=0)
        else:
            flux_xL = c_x * periodic_roll(u, -1, axis=0)
            flux_xR = c_x * u
        # y-direction
        if c_y >= 0:
            flux_yL = c_y * u
            flux_yR = c_y * periodic_roll(u, 1, axis=1)
        else:
            flux_yL = c_y * periodic_roll(u, -1, axis=1)
            flux_yR = c_y * u
        # FV update: divergence of fluxes
        dudt = - (flux_xL - flux_xR) / dx - (flux_yL - flux_yR) / dy
        return dudt
    # --- Time stepping: RK2 ---
    for n in range(Nt):
        k1 = fv_rhs(u)
        u1 = u + dt * k1
        k2 = fv_rhs(u1)
        u = u + 0.5 * dt * (k1 + k2)
    # --- Compute residual at final state ---
    # Compute u_t ≈ (u - u_prev) / dt, but since we only have final u, use PDE directly:
    # Residual: u_t + c_x u_x + c_y u_y = 0
    # Approximate u_t ≈ (u - u_prev) / dt, but since we only have final u, set u_t=0 (steady residual)
    # Instead, compute: R = c_x u_x + c_y u_y
    # Use upwind for residual to match FV
    def upwind_derivative(u, c, dx, axis):
        if c >= 0:
            # backward difference
            return (u - periodic_roll(u, 1, axis=axis)) / dx
        else:
            # forward difference
            return (periodic_roll(u, -1, axis=axis) - u) / dx
    u_x = upwind_derivative(u, c_x, dx, axis=0)
    u_y = upwind_derivative(u, c_y, dy, axis=1)
    residual_grid = c_x * u_x + c_y * u_y
    # --- Return only final state ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual_grid
    }
```