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

    # Assemble sparse matrix using banded structure for efficiency
    # We'll use scipy.sparse for memory and speed
    # But since only numpy is allowed, we will use a block tridiagonal approach
    # and solve with Thomas algorithm for block tridiagonal matrices
    # However, for generality, let's use Jacobi iterations for the implicit solve

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

    # Precompute coefficients for the operator
    # These are for the interior points only
    # For each cell, we need the coefficients for itself and its 4 neighbors
    # We'll store the coefficients in arrays for vectorized computation

    # Prepare arrays for coefficients
    main_diag = np.zeros((Nx, Ny))
    x_plus = np.zeros((Nx, Ny))
    x_minus = np.zeros((Nx, Ny))
    y_plus = np.zeros((Nx, Ny))
    y_minus = np.zeros((Nx, Ny))

    for i in range(Nx):
        for j in range(Ny):
            xi = x_c[i]
            yj = y_c[j]
            # Diffusion
            main_diag[i, j] += -2*D/dx**2 - 2*D/dy**2
            if i < Nx-1:
                x_plus[i, j] = D/dx**2 + (lam * x_f[i+1])/(2*dx)
            if i > 0:
                x_minus[i, j] = D/dx**2 - (lam * x_f[i])/(2*dx)
            if j < Ny-1:
                y_plus[i, j] = D/dy**2 + (lam * y_f[j+1])/(2*dy)
            if j > 0:
                y_minus[i, j] = D/dy**2 - (lam * y_f[j])/(2*dy)
            # Diagonal drift terms are already included in off-diagonal

    # Time array (only store t=0 and t_final)
    t_array = np.array([0.0, t_final])

    u_curr = u.copy()
    t = 0.0

    # Helper for Jacobi iteration
    def apply_operator(u_grid):
        # u_grid: shape (Nx, Ny)
        out = main_diag * u_grid
        out[:-1, :] += x_plus[:-1, :] * u_grid[1:, :]
        out[1:, :] += x_minus[1:, :] * u_grid[:-1, :]
        out[:, :-1] += y_plus[:, :-1] * u_grid[:, 1:]
        out[:, 1:] += y_minus[:, 1:] * u_grid[:, :-1]
        return out

    # Crank-Nicolson: (I - 0.5*dt*A) u^{n+1} = (I + 0.5*dt*A) u^n + b
    # We'll use Jacobi iterations for the implicit solve
    max_jacobi_iter = 100
    jacobi_tol = 1e-8

    for n in range(Nt):
        t_next = t + dt

        # Set up RHS: (I + 0.5*dt*A) u^n
        rhs_grid = u_curr + 0.5*dt*apply_operator(u_curr)

        # Set BC values at t_next
        u_bc = u_curr.copy()
        # x boundaries
        for j in range(Ny):
            u_bc[0, j] = bc_func(x_c[0], y_c[j], t_next)
            u_bc[-1, j] = bc_func(x_c[-1], y_c[j], t_next)
        # y boundaries
        for i in range(Nx):
            u_bc[i, 0] = bc_func(x_c[i], y_c[0], t_next)
            u_bc[i, -1] = bc_func(x_c[i], y_c[-1], t_next)

        # Enforce BCs in system
        rhs_grid[0, :] = u_bc[0, :]
        rhs_grid[-1, :] = u_bc[-1, :]
        rhs_grid[:, 0] = u_bc[:, 0]
        rhs_grid[:, -1] = u_bc[:, -1]

        # Jacobi iteration for (I - 0.5*dt*A) u^{n+1} = rhs_grid
        u_new = u_curr.copy()
        for it in range(max_jacobi_iter):
            u_old = u_new.copy()
            # Only update interior points
            # For interior: i=1..Nx-2, j=1..Ny-2
            # Jacobi: u_new[i,j] = (rhs - sum_{neighbors} coeff*neighbor)/(1 - 0.5*dt*main_diag)
            denom = 1 - 0.5*dt*main_diag[1:-1, 1:-1]
            rhs = rhs_grid[1:-1, 1:-1].copy()
            # x neighbors
            rhs -= 0.5*dt*x_plus[1:-1, 1:-1]*u_old[2:, 1:-1]
            rhs -= 0.5*dt*x_minus[1:-1, 1:-1]*u_old[:-2, 1:-1]
            # y neighbors
            rhs -= 0.5*dt*y_plus[1:-1, 1:-1]*u_old[1:-1, 2:]
            rhs -= 0.5*dt*y_minus[1:-1, 1:-1]*u_old[1:-1, :-2]
            u_new[1:-1, 1:-1] = rhs / denom
            # Enforce BCs
            u_new[0, :] = u_bc[0, :]
            u_new[-1, :] = u_bc[-1, :]
            u_new[:, 0] = u_bc[:, 0]
            u_new[:, -1] = u_bc[:, -1]
            # Check convergence
            if np.linalg.norm(u_new - u_old, ord=np.inf) < jacobi_tol:
                break

        u_curr = u_new
        t = t_next

    u = u_curr

    # --- Output ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": t_array
    }