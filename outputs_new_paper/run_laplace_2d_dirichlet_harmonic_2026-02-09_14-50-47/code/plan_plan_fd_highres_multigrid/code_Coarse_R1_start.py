```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract grid and domain info ---
    Nx = int(plan['spatial_discretization']['Nx'])
    Ny = int(plan['spatial_discretization']['Ny'])
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    X, Y = np.meshgrid(x, y, indexing='ij')  # shape (Nx, Ny)

    # --- 2. Boundary conditions ---
    # u(x,0) = 0
    # u(x,1) = sin(pi*x)
    # u(0,y) = 0
    # u(1,y) = 0

    u = np.zeros((Nx, Ny), dtype=np.float64)

    # Bottom boundary y=0
    u[:, 0] = 0.0
    # Top boundary y=1
    u[:, -1] = np.sin(np.pi * x)
    # Left boundary x=0
    u[0, :] = 0.0
    # Right boundary x=1
    u[-1, :] = 0.0

    # --- 3. Multigrid solver (V-cycle, 2-level, Jacobi smoothing) ---
    # For simplicity, we implement a basic multigrid V-cycle with Jacobi smoothing.

    def laplace_residual(u, f, dx, dy):
        # Compute residual r = f - A u
        r = np.zeros_like(u)
        # Only compute at interior points
        r[1:-1, 1:-1] = (
            f[1:-1, 1:-1]
            - (
                (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx ** 2
                + (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy ** 2
            )
        )
        return r

    def jacobi(u, f, dx, dy, num_iter, boundary_mask):
        # Jacobi smoothing for Laplace: Au = f
        # boundary_mask: True where Dirichlet BC is set
        u_new = u.copy()
        dx2 = dx * dx
        dy2 = dy * dy
        denom = 2 * (1/dx2 + 1/dy2)
        for _ in range(num_iter):
            u_old = u_new.copy()
            u_new[1:-1, 1:-1] = (
                (
                    (u_old[2:, 1:-1] + u_old[:-2, 1:-1]) / dx2
                    + (u_old[1:-1, 2:] + u_old[1:-1, :-2]) / dy2
                    - f[1:-1, 1:-1]
                ) / denom
            )
            # Re-impose Dirichlet BCs
            u_new[boundary_mask] = u[boundary_mask]
        return u_new

    def restrict(r):
        # Full-weighting restriction from fine (n) to coarse (n//2+1)
        # r shape: (Nxf, Nyf)
        Nxf, Nyf = r.shape
        Nxc = (Nxf - 1) // 2 + 1
        Nyc = (Nyf - 1) // 2 + 1
        rc = np.zeros((Nxc, Nyc), dtype=r.dtype)
        # Interior points
        rc[1:-1, 1:-1] = (
            1/16 * (
                4 * r[2:-1:2, 2:-1:2] +
                2 * (r[1:-2:2, 2:-1:2] + r[3::2, 2:-1:2] + r[2:-1:2, 1:-2:2] + r[2:-1:2, 3::2]) +
                (r[1:-2:2, 1:-2:2] + r[1:-2:2, 3::2] + r[3::2, 1:-2:2] + r[3::2, 3::2])
            )
        )
        # Boundaries: inject
        rc[0, :] = r[0, ::2]
        rc[-1, :] = r[-1, ::2]
        rc[:, 0] = r[::2, 0]
        rc[:, -1] = r[::2, -1]
        return rc

    def prolong(e):
        # Bilinear interpolation from coarse (nc) to fine (2*nc-1)
        ncx, ncy = e.shape
        nfx = 2 * (ncx - 1) + 1
        nfy = 2 * (ncy - 1) + 1
        ef = np.zeros((nfx, nfy), dtype=e.dtype)
        # Copy coarse points
        ef[::2, ::2] = e
        # Interpolate in x
        ef[1::2, ::2] = 0.5 * (e[:-1, :] + e[1:, :])
        # Interpolate in y
        ef[::2, 1::2] = 0.5 * (e[:, :-1] + e[:, 1:])
        # Interpolate in both
        ef[1::2, 1::2] = 0.25 * (e[:-1, :-1] + e[1:, :-1] + e[:-1, 1:] + e[1:, 1:])
        return ef

    def v_cycle(u, f, dx, dy, pre_smooth, post_smooth, boundary_mask, min_size=5):
        n, m = u.shape
        if min(n, m) <= min_size:
            # Direct solve (Jacobi with many iterations)
            u = jacobi(u, f, dx, dy, 100, boundary_mask)
            return u
        # Pre-smoothing
        u = jacobi(u, f, dx, dy, pre_smooth, boundary_mask)
        # Compute residual
        r = laplace_residual(u, f, dx, dy)
        # Restrict residual and error
        r_coarse = restrict(r)
        # Zero initial guess for error on coarse grid
        e_coarse = np.zeros_like(r_coarse)
        # Coarse grid spacing
        dx2 = dx * 2
        dy2 = dy * 2
        # Boundary mask for coarse grid
        mask_coarse = restrict(boundary_mask.astype(np.float64)) > 0.5
        # Recursively solve for error on coarse grid
        e_coarse = v_cycle(e_coarse, r_coarse, dx2, dy2, pre_smooth, post_smooth, mask_coarse, min_size)
        # Prolongate error and correct
        e_fine = prolong(e_coarse)
        # Add correction (only at interior)
        u[~boundary_mask] += e_fine[~boundary_mask]
        # Post-smoothing
        u = jacobi(u, f, dx, dy, post_smooth, boundary_mask)
        return u

    # --- 4. Setup for multigrid ---
    # Right-hand side f = 0 for Laplace
    f = np.zeros_like(u)
    # Dirichlet boundary mask: True where BC is set
    boundary_mask = np.zeros_like(u, dtype=bool)
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True

    # --- 5. Multigrid solve ---
    max_cycles = 50
    tol = 1e-8
    pre_smooth = int(plan['time_stepping']['extra_parameters'].get('pre_smoothing', 3))
    post_smooth = int(plan['time_stepping']['extra_parameters'].get('post_smoothing', 3))

    for cycle in range(max_cycles):
        u_old = u.copy()
        u = v_cycle(u, f, dx, dy, pre_smooth, post_smooth, boundary_mask)
        # Re-impose Dirichlet BCs (in case of roundoff)
        u[:, 0] = 0.0
        u[:, -1] = np.sin(np.pi * x)
        u[0, :] = 0.0
        u[-1, :] = 0.0
        # Check convergence (max norm of update)
        diff = np.max(np.abs(u - u_old))
        if diff < tol:
            break

    # --- 6. Compute residual grid ---
    # For Laplace: residual = u_xx + u_yy (should be ~0)
    residual = np.zeros_like(u)
    residual[1:-1, 1:-1] = (
        (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx ** 2
        + (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy ** 2
    )
    # Dirichlet boundaries: set residual to 0 (since BCs are enforced)
    residual[boundary_mask] = 0.0

    # --- 7. Return final state ---
    result = {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": np.array([]),  # No time dimension for Laplace
        "residual": residual
    }
    return result
```