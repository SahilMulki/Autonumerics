import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters ---
    # PDE parameters
    D = float(pde_spec['parameters']['D'])
    lam = float(pde_spec['parameters']['lambda'])
    domain = pde_spec['domain']
    x_min, x_max = domain['bounds']['x']
    y_min, y_max = domain['bounds']['y']

    # FEM grid parameters
    Nx = int(plan['spatial_discretization']['Nx'])
    Ny = int(plan['spatial_discretization']['Ny'])

    # Time stepping
    dt = float(plan['time_stepping'].get('dt', 0.005))
    t_final = float(plan['time_stepping'].get('t_final', 1.0))
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # ensure final time is exactly t_final
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Mesh ---
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    X, Y = np.meshgrid(x, y, indexing='ij')  # shape (Nx, Ny)

    # --- Initial condition ---
    rho0 = (1/(2*np.pi)) * np.exp(-(X**2 + Y**2)/2)
    u = rho0.copy()

    # --- Dirichlet BC mask ---
    bc_mask = np.zeros((Nx, Ny), dtype=bool)
    bc_mask[0, :] = True
    bc_mask[-1, :] = True
    bc_mask[:, 0] = True
    bc_mask[:, -1] = True

    # Helper for BC value at (x, y, t)
    def sigma2_t(t):
        return D/lam + (1 - D/lam)*np.exp(-2*lam*t)
    def bc_value(xb, yb, t):
        s2 = sigma2_t(t)
        return (1/(2*np.pi*s2)) * np.exp(-(xb**2 + yb**2)/(2*s2))

    # Precompute indices for interior
    interior = (~bc_mask)
    Ntot = Nx * Ny
    idx_map = np.arange(Ntot).reshape((Nx, Ny))

    # --- Assemble sparse matrix for backward Euler step ---
    dx2 = dx*dx
    dy2 = dy*dy

    # Build mapping from (i,j) to local interior index
    interior_idx = np.where(interior.ravel())[0]
    n_interior = len(interior_idx)
    global2local = -np.ones(Ntot, dtype=int)
    global2local[interior_idx] = np.arange(n_interior)

    # Precompute drift at each grid point
    drift_x = lam * X
    drift_y = lam * Y

    # Stencil coefficients
    center = np.ones((Nx, Ny)) + dt * 2*D*(1/dx2 + 1/dy2)
    left   = -dt * D / dx2 - 0.5 * dt * drift_x / dx
    right  = -dt * D / dx2 + 0.5 * dt * drift_x / dx
    down   = -dt * D / dy2 - 0.5 * dt * drift_y / dy
    up     = -dt * D / dy2 + 0.5 * dt * drift_y / dy

    # --- Time stepping ---
    u_n = u.copy()
    # Precompute BC values for all time steps for efficiency
    bc_vals_time = []
    for n in range(Nt+1):
        t_n = t_array[n]
        bc_vals = np.zeros((Nx, Ny))
        for i in [0, Nx-1]:
            for j in range(Ny):
                bc_vals[i, j] = bc_value(x[i], y[j], t_n)
        for i in range(1, Nx-1):
            for j in [0, Ny-1]:
                bc_vals[i, j] = bc_value(x[i], y[j], t_n)
        bc_vals_time.append(bc_vals)

    # Jacobi iterative solver parameters
    max_jacobi_iter = 100
    jacobi_tol = 1e-7

    for n in range(Nt):
        t_np1 = t_array[n+1]
        bc_vals_np1 = bc_vals_time[n+1]

        # Build RHS: u^n at interior, plus BCs
        b = u_n.copy()

        # Add BC contributions from neighbors for interior points
        rhs = b.copy()

        # For each interior point, subtract the BC contributions from neighbors
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                if not interior[i, j]:
                    continue
                # left neighbor
                if bc_mask[i-1, j]:
                    rhs[i, j] -= left[i, j] * bc_vals_np1[i-1, j]
                # right neighbor
                if bc_mask[i+1, j]:
                    rhs[i, j] -= right[i, j] * bc_vals_np1[i+1, j]
                # down neighbor
                if bc_mask[i, j-1]:
                    rhs[i, j] -= down[i, j] * bc_vals_np1[i, j-1]
                # up neighbor
                if bc_mask[i, j+1]:
                    rhs[i, j] -= up[i, j] * bc_vals_np1[i, j+1]

        # Initial guess: previous solution
        u_guess = u_n.copy()

        # Jacobi iterations
        for it in range(max_jacobi_iter):
            u_new = u_guess.copy()
            # Only update interior points
            for i in range(1, Nx-1):
                for j in range(1, Ny-1):
                    if not interior[i, j]:
                        continue
                    num = rhs[i, j]
                    num += left[i, j] * u_guess[i-1, j]
                    num += right[i, j] * u_guess[i+1, j]
                    num += down[i, j] * u_guess[i, j-1]
                    num += up[i, j] * u_guess[i, j+1]
                    u_new[i, j] = num / center[i, j]
            # Dirichlet BCs
            u_new[bc_mask] = bc_vals_np1[bc_mask]
            # Check convergence
            err = np.linalg.norm(u_new - u_guess, ord=np.inf)
            if err < jacobi_tol:
                break
            u_guess = u_new

        u_n = u_new

    u_final = u_n

    # --- Residual calculation (L2 error to analytic solution at final time) ---
    # Analytic solution at t_final
    s2 = sigma2_t(t_final)
    u_analytic = (1/(2*np.pi*s2)) * np.exp(-(X**2 + Y**2)/(2*s2))
    # Compute L2 error (integral over domain)
    diff2 = (u_final - u_analytic)**2
    # Use the trapezoidal rule for 2D integration
    residual = np.sqrt(np.sum(diff2) * dx * dy)

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual
    }