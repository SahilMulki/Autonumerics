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
        # Estimate dt by CFL for diffusion: dt <= 0.25 * min(dx^2, dy^2) / D
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

    # --- Assemble sparse matrix for implicit step (Backward Euler) ---
    from scipy.sparse import diags
    from scipy.sparse.linalg import cg

    N = Nx * Ny
    main_diag = (1 + dt * (2*D/dx**2 + 2*D/dy**2) - dt*r) * np.ones(N)
    off_x = -dt * D / dx**2 * np.ones(N-1)
    off_y = -dt * D / dy**2 * np.ones(N-Ny)

    # Zero out off-diagonal connections at boundaries in x
    for i in range(1, Nx):
        off_x[i*Ny-1] = 0

    diagonals = [main_diag, off_x, off_x, off_y, off_y]
    offsets = [0, -1, 1, -Ny, Ny]
    from scipy.sparse import csr_matrix
    A = diags(diagonals, offsets, shape=(N, N), format='csr')

    # Precompute boundary mask for flat array
    boundary_mask = np.zeros((Nx, Ny), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    boundary_mask_flat = boundary_mask.ravel()

    # --- Time stepping ---
    u_flat = u.ravel()
    t_array = np.linspace(0, t_final, Nt+1)
    for n in range(Nt):
        b = u_flat.copy()
        b[boundary_mask_flat] = bc_val
        u_new_flat, info = cg(A, b, x0=u_flat, maxiter=200, tol=1e-8)
        if info != 0:
            raise RuntimeError(f"Linear solver did not converge at step {n}, info={info}")
        u_new_flat[boundary_mask_flat] = bc_val
        u_flat = u_new_flat

    # --- Final solution ---
    u = u_flat.reshape((Nx, Ny))
    u = apply_dirichlet(u)

    # --- Residual computation ---
    # To do this, we need to rerun the last step to get u_prev
    u_flat_prev = u_flat.copy()
    b = u_flat_prev.copy()
    b[boundary_mask_flat] = bc_val
    u_flat_older, info = cg(A, b, x0=u_flat_prev, maxiter=200, tol=1e-8)
    if info != 0:
        raise RuntimeError(f"Linear solver did not converge for residual, info={info}")
    u_flat_older[boundary_mask_flat] = bc_val
    u_prev = u_flat_older.reshape((Nx, Ny))
    u_prev = apply_dirichlet(u_prev)

    # Compute u_t at final time
    u_t = (u - u_prev) / dt

    # Compute Laplacian using central differences (second order)
    u_xx = np.zeros_like(u)
    u_yy = np.zeros_like(u)
    u_xx[1:-1, :] = (u[2:, :] - 2*u[1:-1, :] + u[:-2, :]) / dx**2
    u_yy[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]) / dy**2
    lap = u_xx + u_yy

    residual = u_t - D * lap - r * u
    residual = apply_dirichlet(residual)

    coords = {"x": x, "y": y}
    return {
        "u": u,
        "coords": coords,
        "t": t_array
    }