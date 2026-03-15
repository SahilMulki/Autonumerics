import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters ---
    # PDE parameters
    D = float(pde_spec['parameters']['D'])
    lam = float(pde_spec['parameters']['lambda'])
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']

    # Plan parameters
    Nx = int(plan['spatial_discretization']['Nx'])
    Ny = int(plan['spatial_discretization']['Ny'])
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    Nt = plan['time_stepping'].get('Nt', None)
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    if dt is None:
        dt = 0.4 * min(dx, dy)**2 / (4*D)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    else:
        t_final = Nt * dt

    # --- Grids ---
    # Cell centers (finite volume)
    x = np.linspace(x_min + dx/2, x_max - dx/2, Nx)
    y = np.linspace(y_min + dy/2, y_max - dy/2, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial condition ---
    u = (1/(2*np.pi)) * np.exp(-(X**2 + Y**2)/2)

    # --- Boundary condition function ---
    def sigma2_t(t):
        return D/lam + (1 - D/lam)*np.exp(-2*lam*t)
    def bc_func(xb, yb, t):
        s2 = sigma2_t(t)
        return (1/(2*np.pi*s2)) * np.exp(-(xb**2 + yb**2)/(2*s2))

    # --- Precompute for efficiency ---
    x_faces = np.linspace(x_min, x_max, Nx+1)
    y_faces = np.linspace(y_min, y_max, Ny+1)
    x_c = x
    y_c = y
    x_f = x_faces
    y_f = y_faces

    # --- Matrix assembly ---
    N = Nx * Ny
    def idx(i, j):
        return i*Ny + j

    # Precompute x, y arrays for all cells
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    # Build operator A
    A = np.zeros((N, N), dtype=np.float64)
    for i in range(Nx):
        for j in range(Ny):
            p = idx(i, j)
            xi = x_c[i]
            yj = y_c[j]

            # Diffusion coefficients
            if i > 0:
                A[p, idx(i-1, j)] += D / dx**2
            if i < Nx-1:
                A[p, idx(i+1, j)] += D / dx**2
            A[p, p] += -2*D / dx**2

            if j > 0:
                A[p, idx(i, j-1)] += D / dy**2
            if j < Ny-1:
                A[p, idx(i, j+1)] += D / dy**2
            A[p, p] += -2*D / dy**2

            # Drift terms (FV: central at faces)
            # x drift
            if i < Nx-1:
                A[p, idx(i+1, j)] += (lam * x_f[i+1]) / (2*dx)
            if i > 0:
                A[p, idx(i-1, j)] += -(lam * x_f[i]) / (2*dx)
            diag_drift_x = 0.0
            if i < Nx-1:
                diag_drift_x += (lam * x_f[i+1]) / (2*dx)
            if i > 0:
                diag_drift_x += -(lam * x_f[i]) / (2*dx)
            A[p, p] += -diag_drift_x

            # y drift
            if j < Ny-1:
                A[p, idx(i, j+1)] += (lam * y_f[j+1]) / (2*dy)
            if j > 0:
                A[p, idx(i, j-1)] += -(lam * y_f[j]) / (2*dy)
            diag_drift_y = 0.0
            if j < Ny-1:
                diag_drift_y += (lam * y_f[j+1]) / (2*dy)
            if j > 0:
                diag_drift_y += -(lam * y_f[j]) / (2*dy)
            A[p, p] += -diag_drift_y

    # --- Time stepping ---
    # Crank-Nicolson: (I - 0.5*dt*A) u^{n+1} = (I + 0.5*dt*A) u^n + b
    I = np.eye(N)
    M_lhs_base = I - 0.5*dt*A
    M_rhs_base = I + 0.5*dt*A

    # Time array (only store t=0 and t_final)
    t_array = np.array([0.0, t_final])

    u_flat = u.ravel().copy()
    t = 0.0

    # Precompute boundary indices and masks
    boundary_mask = np.zeros((Nx, Ny), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    boundary_idx = np.where(boundary_mask.ravel())[0]

    # For efficiency, precompute which indices are on which boundaries
    x0_idx = [idx(0, j) for j in range(Ny)]
    xN_idx = [idx(Nx-1, j) for j in range(Ny)]
    y0_idx = [idx(i, 0) for i in range(Nx)]
    yN_idx = [idx(i, Ny-1) for i in range(Nx)]

    # For each time step
    for n in range(Nt):
        t_next = t + dt

        # Copy matrices to avoid modifying base matrices
        M_lhs = M_lhs_base.copy()
        M_rhs = M_rhs_base.copy()

        # Set up RHS
        rhs = M_rhs @ u_flat

        # Set BC values at t_next
        u_bc = u_flat.copy()
        # x boundaries
        for j in range(Ny):
            u_bc[idx(0, j)] = bc_func(x_c[0], y_c[j], t_next)
            u_bc[idx(Nx-1, j)] = bc_func(x_c[-1], y_c[j], t_next)
        # y boundaries
        for i in range(Nx):
            u_bc[idx(i, 0)] = bc_func(x_c[i], y_c[0], t_next)
            u_bc[idx(i, Ny-1)] = bc_func(x_c[i], y_c[-1], t_next)

        # Enforce BCs in system
        for p in boundary_idx:
            M_lhs[p, :] = 0.0
            M_lhs[p, p] = 1.0
            rhs[p] = u_bc[p]

        # Solve linear system
        u_flat = np.linalg.solve(M_lhs, rhs)

        t = t_next

    # Reshape to grid
    u = u_flat.reshape((Nx, Ny))

    # --- Output ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": t_array
    }