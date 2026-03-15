import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    D = float(pde_spec["parameters"]["D"])
    r = float(pde_spec["parameters"]["r"])
    domain = pde_spec["domain"]
    x_min, x_max = domain["bounds"]["x"]
    y_min, y_max = domain["bounds"]["y"]
    bc_type = pde_spec["boundary_conditions"]["type"]
    bc_val = pde_spec["boundary_conditions"]["values"]["u"]

    # --- Extract discretization parameters ---
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny

    # --- Time stepping ---
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    Nt = plan["time_stepping"].get("Nt", None)
    if dt is None:
        dt = 0.25 * min(dx**2, dy**2) / D
    if t_final is not None:
        Nt = int(np.ceil(t_final / dt))
        t_final = Nt * dt
    elif Nt is not None:
        t_final = Nt * dt
    else:
        raise ValueError("Either t_final or Nt must be specified in the plan.")

    # --- Grids ---
    x = np.linspace(x_min + dx/2, x_max - dx/2, Nx)
    y = np.linspace(y_min + dy/2, y_max - dy/2, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial condition ---
    u = np.sin(np.pi * X) * np.sin(np.pi * Y)

    # --- Dirichlet boundary mask ---
    def apply_dirichlet(u):
        u[0, :] = bc_val
        u[-1, :] = bc_val
        u[:, 0] = bc_val
        u[:, -1] = bc_val
        return u

    # --- Precompute boundary mask for flat array ---
    boundary_mask = np.zeros((Nx, Ny), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    boundary_mask_flat = boundary_mask.ravel()

    # --- Helper: matrix-vector product for implicit step ---
    def matvec(u_flat):
        u_grid = u_flat.reshape((Nx, Ny))
        lap = np.zeros_like(u_grid)
        # Interior points only
        lap[1:-1,1:-1] = (
            (u_grid[2:,1:-1] - 2*u_grid[1:-1,1:-1] + u_grid[0:-2,1:-1]) / dx**2 +
            (u_grid[1:-1,2:] - 2*u_grid[1:-1,1:-1] + u_grid[1:-1,0:-2]) / dy**2
        )
        out = u_grid + dt * (D * lap + r * u_grid)
        out[boundary_mask] = u_grid[boundary_mask]  # Dirichlet BCs
        return out.ravel()

    # --- Time stepping (Backward Euler, iterative linear solver) ---
    u_flat = u.ravel()
    t_array = np.linspace(0, t_final, Nt+1)
    tol = 1e-8
    maxiter = 200

    for n in range(Nt):
        b = u_flat.copy()
        b[boundary_mask_flat] = bc_val

        # Use simple Jacobi iteration for the implicit solve
        u_new = u_flat.copy()
        for it in range(maxiter):
            u_old = u_new.copy()
            u_grid = u_old.reshape((Nx, Ny))
            rhs = b.reshape((Nx, Ny))
            u_next = u_grid.copy()
            # Update only interior points
            u_next[1:-1,1:-1] = (
                rhs[1:-1,1:-1]
                + dt * D * (
                    (u_grid[2:,1:-1] + u_grid[0:-2,1:-1]) / dx**2 +
                    (u_grid[1:-1,2:] + u_grid[1:-1,0:-2]) / dy**2
                )
            ) / (
                1 + dt * (2*D/dx**2 + 2*D/dy**2) - dt*r
            )
            # Dirichlet BCs
            u_next[0,:] = bc_val
            u_next[-1,:] = bc_val
            u_next[:,0] = bc_val
            u_next[:,-1] = bc_val
            u_new = u_next.ravel()
            if np.linalg.norm(u_new - u_old) < tol:
                break
        u_flat = u_new

    # --- Final solution ---
    u = u_flat.reshape((Nx, Ny))
    u = apply_dirichlet(u)

    # --- Residual calculation (L2 error with analytic solution if available) ---
    t = t_final
    analytic = np.exp((r - 2*D*np.pi**2)*t) * np.sin(np.pi * X) * np.sin(np.pi * Y)
    residual = np.sqrt(np.mean((u - analytic)**2))

    coords = {"x": x, "y": y}
    return {
        "u": u,
        "coords": coords,
        "t": t_array,
        "residual": residual
    }