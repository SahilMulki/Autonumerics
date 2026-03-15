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

    # Precompute the sparse matrix A for the implicit step
    # Only for interior points
    A = np.zeros((n_interior, n_interior))

    # For each interior node (i,j)
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            k = idx_map[i, j]
            loc = global2local[k]
            if loc == -1:
                continue

            # Laplacian
            center = 1.0 + dt * 2*D*(1/dx2 + 1/dy2)
            left   = -dt * D / dx2
            right  = -dt * D / dx2
            down   = -dt * D / dy2
            up     = -dt * D / dy2

            # Drift (central diff)
            xij = x[i]
            yij = y[j]
            drift_x = lam * xij
            drift_y = lam * yij
            drift_left  = -0.5 * dt * drift_x / dx
            drift_right =  0.5 * dt * drift_x / dx
            drift_down  = -0.5 * dt * drift_y / dy
            drift_up    =  0.5 * dt * drift_y / dy

            # Center
            A[loc, loc] += center

            # Left neighbor (i-1, j)
            kl = idx_map[i-1, j]
            locl = global2local[kl]
            if locl != -1:
                A[loc, locl] += left + drift_left

            # Right neighbor (i+1, j)
            kr = idx_map[i+1, j]
            locr = global2local[kr]
            if locr != -1:
                A[loc, locr] += right + drift_right

            # Down neighbor (i, j-1)
            kd = idx_map[i, j-1]
            locd = global2local[kd]
            if locd != -1:
                A[loc, locd] += down + drift_down

            # Up neighbor (i, j+1)
            ku = idx_map[i, j+1]
            locu = global2local[ku]
            if locu != -1:
                A[loc, locu] += up + drift_up

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
    # Time stepping loop
    for n in range(Nt):
        t_np1 = t_array[n+1]
        bc_vals_np1 = bc_vals_time[n+1]

        # Build RHS: u^n at interior, plus BCs
        b = u_n[interior].copy()  # shape (n_interior,)

        # Add BC contributions from neighbors
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                k = idx_map[i, j]
                loc = global2local[k]
                if loc == -1:
                    continue
                # neighbors
                neighbors = [
                    (i-1, j, -dt*D/dx2 - 0.5*dt*lam*x[i]/dx),   # left
                    (i+1, j, -dt*D/dx2 + 0.5*dt*lam*x[i]/dx),   # right
                    (i, j-1, -dt*D/dy2 - 0.5*dt*lam*y[j]/dy),   # down
                    (i, j+1, -dt*D/dy2 + 0.5*dt*lam*y[j]/dy),   # up
                ]
                for ni, nj, coeff in neighbors:
                    if bc_mask[ni, nj]:
                        b[loc] -= coeff * bc_vals_np1[ni, nj]

        # Solve for interior
        # Use conjugate gradient for symmetric positive definite A
        # For dense, use np.linalg.solve, but for large n_interior, use iterative
        # For Nx=150, n_interior ~ 21,000, so use a few CG iterations for speed
        # But since A is not strictly symmetric due to drift, fallback to np.linalg.solve for robustness
        u_interior = np.linalg.solve(A, b)

        # Update u
        u_new = np.zeros_like(u_n)
        u_new[interior] = u_interior

        # Dirichlet BCs at t_{n+1}
        u_new[bc_mask] = bc_vals_np1[bc_mask]

        u_n = u_new

    u_final = u_n

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x, "y": y},
        "t": t_array
    }