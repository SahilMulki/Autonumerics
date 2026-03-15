import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract grid and domain info ---
    Nx = int(plan['spatial_discretization'].get('Nx', 100))
    Ny = int(plan['spatial_discretization'].get('Ny', 100))
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)

    # --- 2. Setup boundary conditions ---
    # BCs are Dirichlet, given as formulas in x or y
    # u(x,0) = 0
    # u(x,1) = sin(pi*x)
    # u(0,y) = 0
    # u(1,y) = 0

    # --- 3. Initialize solution array ---
    u = np.zeros((Nx, Ny), dtype=np.float64)

    # Apply BCs
    # Bottom y=0
    u[:, 0] = 0.0
    # Top y=1
    u[:, -1] = np.sin(np.pi * x)
    # Left x=0
    u[0, :] = 0.0
    # Right x=1
    u[-1, :] = 0.0

    # --- 4. Multigrid solver (V-cycle, 5-point Laplacian, Gauss-Seidel smoothing) ---

    def laplace_residual(u, f, dx, dy):
        # f is zero for Laplace
        r = np.zeros_like(u)
        # Only interior points
        r[1:-1,1:-1] = (
            (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[0:-2,1:-1]) / dx**2 +
            (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,0:-2]) / dy**2
        ) - f[1:-1,1:-1]
        return r

    def gauss_seidel(u, f, dx, dy, n_iter):
        # Red-black Gauss-Seidel for smoothing
        Nx, Ny = u.shape
        dx2 = dx*dx
        dy2 = dy*dy
        denom = 2.0*(1.0/dx2 + 1.0/dy2)
        for _ in range(n_iter):
            # Update red-black ordering for better convergence
            for color in [0, 1]:
                for i in range(1, Nx-1):
                    for j in range(1 + (i+color)%2, Ny-1, 2):
                        u[i,j] = (
                            (u[i+1,j] + u[i-1,j]) / dx2 +
                            (u[i,j+1] + u[i,j-1]) / dy2 -
                            f[i,j]
                        ) / denom
        return u

    def restrict(res):
        # Full-weighting restriction (from fine to coarse)
        # Input: shape (Nxf, Nyf)
        # Output: shape (Nxc, Nyc) where Nxc = (Nxf-1)//2 + 1, Nyc = (Nyf-1)//2 + 1
        Nxf, Nyf = res.shape
        Nxc = (Nxf-1)//2 + 1
        Nyc = (Nyf-1)//2 + 1
        res_c = np.zeros((Nxc, Nyc), dtype=res.dtype)
        # Interior
        for i_c in range(1, Nxc-1):
            i_f = 2*i_c
            for j_c in range(1, Nyc-1):
                j_f = 2*j_c
                res_c[i_c, j_c] = (
                    1/16 * (
                        4*res[i_f, j_f] +
                        2*(res[i_f-1, j_f] + res[i_f+1, j_f] + res[i_f, j_f-1] + res[i_f, j_f+1]) +
                        (res[i_f-1, j_f-1] + res[i_f-1, j_f+1] + res[i_f+1, j_f-1] + res[i_f+1, j_f+1])
                    )
                )
        # Boundaries: inject
        res_c[0, :] = res[0, ::2]
        res_c[-1, :] = res[-1, ::2]
        res_c[:, 0] = res[::2, 0]
        res_c[:, -1] = res[::2, -1]
        return res_c

    def prolong(e):
        # Bilinear interpolation (from coarse to fine)
        Nc_x, Nc_y = e.shape
        Nf_x, Nf_y = 2*Nc_x-1, 2*Nc_y-1
        ef = np.zeros((Nf_x, Nf_y), dtype=e.dtype)
        # Copy coarse points
        ef[0::2,0::2] = e
        # Interpolate in x
        ef[1::2,0::2] = 0.5 * (e[:-1,:] + e[1:,:])
        # Interpolate in y
        ef[0::2,1::2] = 0.5 * (e[:,:-1] + e[:,1:])
        # Interpolate diagonals
        ef[1::2,1::2] = 0.25 * (e[:-1,:-1] + e[1:,:-1] + e[:-1,1:] + e[1:,1:])
        return ef

    def multigrid_vcycle(u, f, dx, dy, pre_smooth, post_smooth, level=0, max_levels=10):
        Nx, Ny = u.shape
        # Stop coarsening if grid is too small
        if min(Nx, Ny) <= 3 or level >= max_levels:
            # Direct solve (few GS sweeps)
            u = gauss_seidel(u, f, dx, dy, 30)
            return u

        # Pre-smoothing
        u = gauss_seidel(u, f, dx, dy, pre_smooth)

        # Compute residual
        res = laplace_residual(u, f, dx, dy)

        # Restrict residual to coarse grid
        res_c = restrict(res)
        # Coarse grid size
        Nxc, Nyc = res_c.shape
        # Coarse grid error
        e_c = np.zeros((Nxc, Nyc), dtype=u.dtype)

        # Coarse grid spacing
        dx_c = dx * (Nx-1)/(Nxc-1)
        dy_c = dy * (Ny-1)/(Nyc-1)

        # Recursive V-cycle on coarse grid
        e_c = multigrid_vcycle(e_c, res_c, dx_c, dy_c, pre_smooth, post_smooth, level+1, max_levels)

        # Prolongate error and correct
        e_f = prolong(e_c)
        # Correction: match only the interior of u (avoid shape mismatch)
        # e_f shape: (Nx, Ny) or (Nx-1, Ny-1)
        # We want to add e_f to u[1:-1,1:-1], so e_f must have shape (Nx-2, Ny-2)
        # But prolong returns (Nx, Ny), so we need to slice accordingly
        # Actually, prolong returns (2*Nc_x-1, 2*Nc_y-1) = (Nx, Ny)
        # So e_f[1:-1,1:-1] matches u[1:-1,1:-1]
        u[1:-1,1:-1] += e_f[1:-1,1:-1]

        # Post-smoothing
        u = gauss_seidel(u, f, dx, dy, post_smooth)
        return u

    # --- 5. Solve Laplace equation ---
    # f = 0 (Laplace)
    f = np.zeros_like(u)
    pre_smooth = int(plan['time_stepping']['extra_parameters'].get('pre_smoothing', 3))
    post_smooth = int(plan['time_stepping']['extra_parameters'].get('post_smoothing', 3))

    # Initial guess is already zeros with BCs set
    # Multigrid cycles until convergence
    max_cycles = 100
    tol = 1e-8
    for cycle in range(max_cycles):
        u_old = u.copy()
        u = multigrid_vcycle(u, f, dx, dy, pre_smooth, post_smooth)
        # Enforce Dirichlet BCs after each cycle
        u[:, 0] = 0.0
        u[:, -1] = np.sin(np.pi * x)
        u[0, :] = 0.0
        u[-1, :] = 0.0
        # Check convergence (max norm of update)
        diff = np.max(np.abs(u - u_old))
        if diff < tol:
            break

    # --- 6. Compute residual grid ---
    # Residual: r = u_xx + u_yy (should be ~0)
    residual = np.zeros_like(u)
    residual[1:-1,1:-1] = (
        (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[0:-2,1:-1]) / dx**2 +
        (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,0:-2]) / dy**2
    )

    # --- 7. Return result ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": None,
        "residual": residual
    }