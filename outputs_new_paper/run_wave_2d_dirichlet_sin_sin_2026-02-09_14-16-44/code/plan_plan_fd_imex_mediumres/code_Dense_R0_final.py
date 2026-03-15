import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    t_min, t_max = pde_spec["domain"]["bounds"]["t"]

    # PDE parameters
    c = float(pde_spec["parameters"]["c"])

    # Grid sizes
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])

    # Time stepping
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", t_max)
    if dt is None:
        # Estimate dt by CFL: dt <= dx/(c*sqrt(2)) for 2D wave, safety factor 0.9
        dx = (x_max - x_min) / (Nx - 1)
        dy = (y_max - y_min) / (Ny - 1)
        dx_min = min(dx, dy)
        dt = 0.9 * dx_min / (c * np.sqrt(2))
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil((t_final - t_min) / dt))
    else:
        dt = (t_final - t_min) / Nt
    t_array = np.linspace(t_min, t_final, Nt+1)

    # --- 2. Create grids ---
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')  # shape (Nx, Ny)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # --- 3. Initial conditions ---
    # u(x, y, 0) = sin(pi x) sin(pi y)
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
    # u_t(x, y, 0) = 0
    v0 = np.zeros_like(u0)

    # --- 4. Boundary mask ---
    boundary_mask = np.zeros((Nx, Ny), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True

    # --- 5. Laplacian operator (5-point FD) ---
    def laplacian(U):
        # U: (Nx, Ny)
        L = np.zeros_like(U)
        # interior points
        L[1:-1,1:-1] = (
            (U[2:,1:-1] - 2*U[1:-1,1:-1] + U[0:-2,1:-1]) / dx**2 +
            (U[1:-1,2:] - 2*U[1:-1,1:-1] + U[1:-1,0:-2]) / dy**2
        )
        return L

    # --- 6. Time stepping (Leapfrog scheme for wave equation) ---
    # For memory: only keep u, v at current and previous step.

    u_nm1 = u0.copy()  # u^{n-1}
    u_n = u0.copy()    # u^{n}
    v_n = v0.copy()    # v^{n}

    # For first step, use a Taylor expansion for u^1:
    #   u^1 = u^0 + dt*v^0 + 0.5*dt^2*c^2*Lap(u^0)
    Lap_u0 = laplacian(u0)
    u_np1 = u0 + dt * v0 + 0.5 * dt**2 * c**2 * Lap_u0

    # Enforce Dirichlet BCs at t=0 and t=dt
    u_n[boundary_mask] = 0.0
    u_np1[boundary_mask] = 0.0

    # For v, use central difference:
    v_np1 = (u_np1 - u_n) / dt

    # Time stepping loop
    for n in range(1, Nt):
        # Leapfrog: u^{n+1} = 2u^n - u^{n-1} + dt^2 * c^2 * Lap(u^n)
        Lap_u_n = laplacian(u_n)
        u_new = 2 * u_n - u_nm1 + dt**2 * c**2 * Lap_u_n

        # Dirichlet BCs
        u_new[boundary_mask] = 0.0

        # Update v (approximate u_t at n+1)
        v_new = (u_new - u_n) / dt

        # Rotate variables
        u_nm1, u_n = u_n, u_new
        v_n = v_new

    # After loop, u_n is u at t_final
    u = u_n.copy()

    # --- 7. Residual calculation (L2 norm of PDE residual at final time) ---
    # PDE: u_tt = c^2 (u_xx + u_yy)
    # Approximate u_tt by central difference at t_final:
    #   u_tt ≈ (u^{n+1} - 2u^n + u^{n-1}) / dt^2
    # At final time, u_n = u^{Nt}, u_nm1 = u^{Nt-1}
    # For u^{Nt+1}, we can use the leapfrog formula one more time:
    Lap_u_n = laplacian(u_n)
    u_np1 = 2 * u_n - u_nm1 + dt**2 * c**2 * Lap_u_n
    u_tt = (u_np1 - 2 * u_n + u_nm1) / dt**2
    Lap_u = laplacian(u_n)
    residual = u_tt - c**2 * Lap_u
    # Mask out boundaries for L2 norm
    interior_mask = ~boundary_mask
    residual_L2 = np.sqrt(np.mean(residual[interior_mask]**2))

    # --- 8. Return ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual_L2
    }