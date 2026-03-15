import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and plan parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    # Grid
    Nx = plan["spatial_discretization"]["Nx"]
    Ny = plan["spatial_discretization"]["Ny"]
    order = plan["spatial_discretization"].get("order", 4)
    # Time
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", 1.0)
    Nt = plan["time_stepping"].get("Nt", None)
    # Parameters
    c_x = float(pde_spec["parameters"]["c_x"])
    c_y = float(pde_spec["parameters"]["c_y"])
    # Periodic BCs
    periodic = pde_spec["boundary_conditions"]["type"] == "periodic"

    # --- Build grid ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial condition ---
    u0 = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    u = u0.copy()

    # --- Time stepping setup ---
    # Compute stable dt using CFL for 4th order upwind (CFL < 1/2 for 1D, more restrictive in 2D)
    # For 4th order upwind, max CFL ~ 0.25 (for stability margin)
    # For 2D, use more restrictive CFL, e.g., 0.08
    max_cfl = 0.08
    if dt is None:
        cfl = max_cfl
        dt = cfl / (abs(c_x)/dx + abs(c_y)/dy)
    else:
        # Check user dt against recommended CFL
        cfl = dt * (abs(c_x)/dx + abs(c_y)/dy)
        if cfl > max_cfl:
            dt = max_cfl / (abs(c_x)/dx + abs(c_y)/dy)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt  # adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)
    t = 0.0

    # --- 4th order upwind finite difference coefficients ---
    # For positive advection speed (upwind left-biased)
    # Coefficients for 4th order upwind (left-biased) for df/dx at i:
    # f'_i ≈ ( 25/12 f_i - 4 f_{i-1} + 3 f_{i-2} - 4/3 f_{i-3} + 1/4 f_{i-4} ) / dx
    # For negative advection speed, use right-biased (reverse stencil)
    def upwind_4th_x(u, c):
        if c >= 0:
            # left-biased
            return (
                25/12 * u
                - 4 * np.roll(u, 1, axis=0)
                + 3 * np.roll(u, 2, axis=0)
                - 4/3 * np.roll(u, 3, axis=0)
                + 1/4 * np.roll(u, 4, axis=0)
            ) / dx
        else:
            # right-biased
            return (
                -25/12 * u
                + 4 * np.roll(u, -1, axis=0)
                - 3 * np.roll(u, -2, axis=0)
                + 4/3 * np.roll(u, -3, axis=0)
                - 1/4 * np.roll(u, -4, axis=0)
            ) / dx

    def upwind_4th_y(u, c):
        if c >= 0:
            # left-biased
            return (
                25/12 * u
                - 4 * np.roll(u, 1, axis=1)
                + 3 * np.roll(u, 2, axis=1)
                - 4/3 * np.roll(u, 3, axis=1)
                + 1/4 * np.roll(u, 4, axis=1)
            ) / dy
        else:
            # right-biased
            return (
                -25/12 * u
                + 4 * np.roll(u, -1, axis=1)
                - 3 * np.roll(u, -2, axis=1)
                + 4/3 * np.roll(u, -3, axis=1)
                - 1/4 * np.roll(u, -4, axis=1)
            ) / dy

    # --- RHS function for advection ---
    def rhs(u):
        ux = upwind_4th_x(u, c_x)
        uy = upwind_4th_y(u, c_y)
        return -c_x * ux - c_y * uy

    # --- SSPRK(3,3) time stepping ---
    for n in range(Nt):
        # Stage 1
        k1 = rhs(u)
        u1 = u + dt * k1
        # Stage 2
        k2 = rhs(u1)
        u2 = 0.75 * u + 0.25 * (u1 + dt * k2)
        # Stage 3
        k3 = rhs(u2)
        u = (1/3) * u + (2/3) * (u2 + dt * k3)
        t += dt

    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": t_array
    }