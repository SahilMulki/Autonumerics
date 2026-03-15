import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # PDE parameters
    D = float(pde_spec["parameters"]["D"])
    lam = float(pde_spec["parameters"]["lambda"])
    domain = pde_spec["domain"]
    x_min, x_max = domain["bounds"]["x"]
    y_min, y_max = domain["bounds"]["y"]
    # Grid parameters
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    # Time stepping
    t_final = float(plan["time_stepping"]["t_final"])
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        dx_min = min(dx, dy)
        dt = 0.2 * dx_min**2 / (4*D + lam*dx_min**2)
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust dt to hit t_final exactly
    # --- 2. Create grids ---
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    coords = {"x": x, "y": y}
    t_array = np.linspace(0, t_final, Nt+1)
    # --- 3. Initial condition ---
    u = (1/(2*np.pi)) * np.exp(-(X**2 + Y**2)/2)
    # --- 4. Boundary condition function ---
    def sigma2_t(t):
        return D/lam + (1 - D/lam)*np.exp(-2*lam*t)
    def bc_func(xb, yb, t):
        s2 = sigma2_t(t)
        return (1/(2*np.pi*s2)) * np.exp(-(xb**2 + yb**2)/(2*s2))
    # --- 5. Precompute coefficients for FD operator ---
    boundary_mask = np.zeros((Nx, Ny), dtype=bool)
    boundary_mask[0,:] = True
    boundary_mask[-1,:] = True
    boundary_mask[:,0] = True
    boundary_mask[:,-1] = True
    interior_mask = ~boundary_mask
    interior_points = np.argwhere(interior_mask)
    Nint = interior_points.shape[0]
    ij_to_int = -np.ones((Nx, Ny), dtype=int)
    for idx, (i, j) in enumerate(interior_points):
        ij_to_int[i, j] = idx
    # --- 6. Define operator application for interior points ---
    def apply_L_interior(u_grid):
        # u_grid: (Nx, Ny)
        Lu = np.zeros_like(u_grid)
        # Laplacian
        Lu[1:-1,1:-1] += D * (
            (u_grid[2:,1:-1] - 2*u_grid[1:-1,1:-1] + u_grid[:-2,1:-1])/(dx*dx)
            + (u_grid[1:-1,2:] - 2*u_grid[1:-1,1:-1] + u_grid[1:-1,:-2])/(dy*dy)
        )
        # Drift: x*u_x
        xi = x[1:-1,None]
        Lu[1:-1,1:-1] += lam * xi * (u_grid[2:,1:-1] - u_grid[:-2,1:-1])/(2*dx)
        # Drift: y*u_y
        yj = y[None,1:-1]
        Lu[1:-1,1:-1] += lam * yj * (u_grid[1:-1,2:] - u_grid[1:-1,:-2])/(2*dy)
        # +2*lam*u
        Lu[1:-1,1:-1] += 2*lam * u_grid[1:-1,1:-1]
        return Lu
    # --- 7. Crank-Nicolson time stepping (matrix-free, Jacobi iteration) ---
    u_n = u.copy()
    diag = np.ones(Nint)
    for idx, (i, j) in enumerate(interior_points):
        diag[idx] = 1 - 0.5*dt*(-2*D/(dx*dx) -2*D/(dy*dy) + 2*lam)
    max_iter = 200  # Jacobi iterations per step
    tol = 1e-8
    for n in range(Nt):
        t = t_array[n+1]
        # 1. Set Dirichlet BCs at time t
        u_bc = u_n.copy()
        u_bc[0,:] = bc_func(x[0], y, t)
        u_bc[-1,:] = bc_func(x[-1], y, t)
        u_bc[:,0] = bc_func(x, y[0], t)
        u_bc[:,-1] = bc_func(x, y[-1], t)
        # 2. Build RHS for interior: (I + 0.5*dt*L)u^n + BC contributions
        Lu_n = apply_L_interior(u_n)
        rhs_grid = u_n + 0.5*dt*Lu_n
        rhs_grid[boundary_mask] = u_bc[boundary_mask]
        rhs = np.zeros(Nint)
        for idx, (i, j) in enumerate(interior_points):
            rhs[idx] = rhs_grid[i, j]
        # 3. Jacobi iteration for (I - 0.5*dt*L)u^{n+1} = rhs
        u_int = np.array([u_bc[i, j] for (i, j) in interior_points])
        for it in range(max_iter):
            u_grid_temp = u_bc.copy()
            for idx, (i, j) in enumerate(interior_points):
                u_grid_temp[i, j] = u_int[idx]
            Lu_temp = apply_L_interior(u_grid_temp)
            Au = np.zeros(Nint)
            for idx, (i, j) in enumerate(interior_points):
                Au[idx] = u_int[idx] - 0.5*dt*Lu_temp[i, j]
            res = rhs - Au
            err = np.linalg.norm(res, ord=np.inf)
            if err < tol:
                break
            u_int += res / diag
        u_new = u_bc.copy()
        for idx, (i, j) in enumerate(interior_points):
            u_new[i, j] = u_int[idx]
        u_n = u_new
    # --- 8. Compute residual (L2 error to analytic solution at t_final) ---
    # Analytic solution at t_final
    s2 = sigma2_t(t_final)
    u_exact = (1/(2*np.pi*s2)) * np.exp(-(X**2 + Y**2)/(2*s2))
    # L2 norm over the domain (trapezoidal rule)
    diff2 = (u_n - u_exact)**2
    # Use composite trapezoidal rule in 2D
    l2_num = np.trapz(np.trapz(diff2, y, axis=1), x)
    l2_den = np.trapz(np.trapz(u_exact**2, y, axis=1), x)
    residual = np.sqrt(l2_num / l2_den)
    # --- 9. Return ---
    return {
        "u": u_n,
        "coords": coords,
        "t": t_array,
        "residual": residual
    }