```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Parse domain and grid ---
    domain = pde_spec["domain"]
    bounds = domain["bounds"]
    Nx = plan["spatial_discretization"]["Nx"]
    Ny = plan["spatial_discretization"]["Ny"]
    x_min, x_max = bounds["x"]
    y_min, y_max = bounds["y"]
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    X, Y = np.meshgrid(x, y, indexing='ij')  # shape (Nx, Ny)

    # --- 2. Define f(x, y) ---
    # For -Δu = f, with analytic u = sin(pi x) sin(pi y)
    # Δu = -2π² sin(pi x) sin(pi y) => -Δu = 2π² sin(pi x) sin(pi y)
    pi = np.pi
    f = 2 * pi ** 2 * np.sin(pi * X) * np.sin(pi * Y)

    # --- 3. Setup boundary conditions ---
    # Dirichlet u=0 on all boundaries
    # We'll solve for interior points only

    # --- 4. Multigrid solver (V-cycle, Jacobi smoothing) ---
    # Helper functions for restriction, prolongation, smoothing

    def jacobi(u, f, dx, dy, mask, num_iter=5, omega=2/3):
        """Weighted Jacobi smoothing for Au = f, only on interior (mask==True)."""
        Nx, Ny = u.shape
        dx2 = dx * dx
        dy2 = dy * dy
        denom = 2 * (1/dx2 + 1/dy2)
        for _ in range(num_iter):
            u_new = u.copy()
            # 5-point Laplacian
            u_new[1:-1,1:-1] = (1-omega)*u[1:-1,1:-1] + omega * 0.5 * (
                (u[2:,1:-1] + u[:-2,1:-1]) / dx2 +
                (u[1:-1,2:] + u[1:-1,:-2]) / dy2 -
                f[1:-1,1:-1]
            ) / denom
            # Enforce Dirichlet BCs
            u_new[0,:] = 0
            u_new[-1,:] = 0
            u_new[:,0] = 0
            u_new[:,-1] = 0
            u = u_new
        return u

    def restrict(res):
        """Full-weighting restriction from fine to coarse grid."""
        # res shape: (Nxf, Nyf)
        Nxf, Nyf = res.shape
        Nxc = (Nxf - 1) // 2 + 1
        Nyc = (Nyf - 1) // 2 + 1
        res_c = np.zeros((Nxc, Nyc))
        # Interior
        res_c[1:-1,1:-1] = (
            1/16 * (
                4*res[2:-1:2,2:-1:2] +
                2*(res[1:-2:2,2:-1:2] + res[3::2,2:-1:2] + res[2:-1:2,1:-2:2] + res[2:-1:2,3::2]) +
                (res[1:-2:2,1:-2:2] + res[1:-2:2,3::2] + res[3::2,1:-2:2] + res[3::2,3::2])
            )
        )
        # Boundaries (Dirichlet, so zero)
        return res_c

    def prolong(e_c):
        """Bilinear prolongation from coarse to fine grid."""
        Nxc, Nyc = e_c.shape
        Nxf = 2 * (Nxc - 1) + 1
        Nyf = 2 * (Nyc - 1) + 1
        e_f = np.zeros((Nxf, Nyf))
        # Copy coarse grid points
        e_f[::2,::2] = e_c
        # Interpolate in x
        e_f[1::2,::2] = 0.5 * (e_c[:-1,:] + e_c[1:,:])
        # Interpolate in y
        e_f[::2,1::2] = 0.5 * (e_c[:,:-1] + e_c[:,1:])
        # Interpolate in both
        e_f[1::2,1::2] = 0.25 * (e_c[:-1,:-1] + e_c[1:,:-1] + e_c[:-1,1:] + e_c[1:,1:])
        return e_f

    def compute_residual(u, f, dx, dy):
        """Compute residual r = f - Au on the grid."""
        r = np.zeros_like(u)
        dx2 = dx * dx
        dy2 = dy * dy
        # 5-point Laplacian
        r[1:-1,1:-1] = (
            f[1:-1,1:-1] -
            (
                (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]) / dx2 +
                (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]) / dy2
            )
        )
        # Dirichlet BC: residual is zero on boundary
        return r

    def v_cycle(u, f, dx, dy, level, max_level):
        """Recursive V-cycle multigrid solver."""
        Nx, Ny = u.shape
        # Stop coarsening at 5x5 grid
        if level == max_level or min(Nx, Ny) <= 5:
            # Direct solve (Jacobi smoothing with many steps)
            u = jacobi(u, f, dx, dy, None, num_iter=50)
            return u
        # Pre-smoothing
        u = jacobi(u, f, dx, dy, None, num_iter=5)
        # Compute residual
        r = compute_residual(u, f, dx, dy)
        # Restrict residual to coarse grid
        r_c = restrict(r)
        # Zero initial guess for error on coarse grid
        e_c = np.zeros_like(r_c)
        # Grid spacings on coarse grid
        dx_c = dx * 2
        dy_c = dy * 2
        # Recursively solve for error on coarse grid
        e_c = v_cycle(e_c, r_c, dx_c, dy_c, level+1, max_level)
        # Prolongate error to fine grid and correct
        e_f = prolong(e_c)
        u += e_f
        # Post-smoothing
        u = jacobi(u, f, dx, dy, None, num_iter=5)
        return u

    # --- 5. Solve using multigrid V-cycles ---
    u = np.zeros((Nx, Ny), dtype=np.float64)
    max_level = int(np.log2(min(Nx, Ny))) - 2  # Don't coarsen below 5x5
    num_cycles = 15
    for _ in range(num_cycles):
        u = v_cycle(u, f, dx, dy, level=0, max_level=max_level)

    # --- 6. Compute residual grid ---
    residual = compute_residual(u, f, dx, dy)

    # --- 7. Output ---
    result = {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": None,
        "residual": residual
    }
    return result
```