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

    # --- Assemble matrix for implicit step (Backward Euler) ---
    N = Nx * Ny
    main_diag = (1 + dt * (2*D/dx**2 + 2*D/dy**2) - dt*r) * np.ones(N)
    off_x = -dt * D / dx**2 * np.ones(N-1)
    off_y = -dt * D / dy**2 * np.ones(N-Ny)

    # Zero out off-diagonal connections at boundaries in x
    for i in range(1, Nx):
        off_x[i*Ny-1] = 0

    # Construct the matrix A in dense format
    A = np.zeros((N, N))
    # Main diagonal
    np.fill_diagonal(A, main_diag)
    # Off-diagonals in y (±Ny)
    for i in range(N-Ny):
        A[i, i+Ny] = off_y[i]
        A[i+Ny, i] = off_y[i]
    # Off-diagonals in x (±1)
    for i in range(N-1):
        if off_x[i] != 0:
            A[i, i+1] = off_x[i]
            A[i+1, i] = off_x[i]

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
        # For Dirichlet BCs, enforce solution at boundary
        A_mod = A.copy()
        b_mod = b.copy()
        for idx in np.where(boundary_mask_flat)[0]:
            A_mod[idx, :] = 0
            A_mod[idx, idx] = 1
            b_mod[idx] = bc_val
        # Solve the linear system
        u_new_flat = np.linalg.solve(A_mod, b_mod)
        u_new_flat[boundary_mask_flat] = bc_val
        u_flat = u_new_flat

    # --- Final solution ---
    u = u_flat.reshape((Nx, Ny))
    u = apply_dirichlet(u)

    # --- Residual calculation (L2 error with analytic solution if available) ---
    # Analytic: exp((r-2*D*pi^2)*t)*sin(pi*x)*sin(pi*y)
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