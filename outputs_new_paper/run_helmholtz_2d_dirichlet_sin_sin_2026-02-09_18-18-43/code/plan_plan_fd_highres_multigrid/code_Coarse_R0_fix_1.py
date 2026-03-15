import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # Domain
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    # Grid
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    order = plan['spatial_discretization'].get('order', 2)
    # PDE parameters
    k = float(pde_spec['parameters']['k'])
    pi = np.pi

    # --- 2. Build grid ---
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- 3. Source term f(x, y) ---
    f = (2 * pi ** 2 + k ** 2) * np.sin(pi * X) * np.sin(pi * Y)

    # --- 4. Boundary conditions ---
    # Dirichlet u=0 on all boundaries
    def apply_bc(U):
        U[0, :] = 0
        U[-1, :] = 0
        U[:, 0] = 0
        U[:, -1] = 0
        return U

    # --- 5. 2D 4th-order finite difference Laplacian operator ---
    # 5-point 4th-order stencil for Laplacian (interior points only)
    def laplacian_2d_4th(U, dx, dy):
        # U: (Nx, Ny)
        lap = np.zeros_like(U)
        # x-direction
        lap[2:-2, 2:-2] = (
            - (1/12) * U[0:-4, 2:-2]
            + (4/3) * U[1:-3, 2:-2]
            - (5/2) * U[2:-2, 2:-2]
            + (4/3) * U[3:-1, 2:-2]
            - (1/12) * U[4:, 2:-2]
        ) / dx**2
        # y-direction
        lap[2:-2, 2:-2] += (
            - (1/12) * U[2:-2, 0:-4]
            + (4/3) * U[2:-2, 1:-3]
            - (5/2) * U[2:-2, 2:-2]
            + (4/3) * U[2:-2, 3:-1]
            - (1/12) * U[2:-2, 4:]
        ) / dy**2
        return lap

    # --- 6. Multigrid solver (V-cycle, geometric, memory-safe) ---
    # Restriction: full-weighting (from fine to coarse)
    def restrict(res):
        # res: (Nxf, Nyf)
        Nxf, Nyf = res.shape
        Nxc = (Nxf - 1) // 2 + 1
        Nyc = (Nyf - 1) // 2 + 1
        rc = np.zeros((Nxc, Nyc))
        # Only interior points
        # Full-weighting restriction
        # For i,j in 1..Nxc-2, 1..Nyc-2
        # Map to fine grid: If, Jf = 2*i, 2*j
        for i in range(1, Nxc-1):
            for j in range(1, Nyc-1):
                If = 2*i
                Jf = 2*j
                rc[i, j] = (
                    1/16 * (
                        4 * res[If, Jf]
                        + 2 * (res[If-1, Jf] + res[If+1, Jf] + res[If, Jf-1] + res[If, Jf+1])
                        + (res[If-1, Jf-1] + res[If-1, Jf+1] + res[If+1, Jf-1] + res[If+1, Jf+1])
                    )
                )
        # Boundaries (Dirichlet, so zero)
        return rc

    # Prolongation: bilinear interpolation (from coarse to fine)
    def prolong(ec):
        Nxc, Nyc = ec.shape
        Nxf = 2 * (Nxc - 1) + 1
        Nyf = 2 * (Nyc - 1) + 1
        ef = np.zeros((Nxf, Nyf))
        # Copy coarse points
        ef[::2, ::2] = ec
        # Interpolate in x
        ef[1::2, ::2] = 0.5 * (ec[:-1, :] + ec[1:, :])
        # Interpolate in y
        ef[::2, 1::2] = 0.5 * (ec[:, :-1] + ec[:, 1:])
        # Interpolate diagonals
        ef[1::2, 1::2] = 0.25 * (ec[:-1, :-1] + ec[1:, :-1] + ec[:-1, 1:] + ec[1:, 1:])
        return ef

    # Smoother: weighted Jacobi
    def weighted_jacobi(U, f, dx, dy, k, omega=0.8, n_iter=5):
        # Only update interior points (2:Nx-2, 2:Ny-2)
        for _ in range(n_iter):
            lapU = laplacian_2d_4th(U, dx, dy)
            # Diagonal of operator: -(-5/2/dx^2 -5/2/dy^2) + k^2
            D = (5/2)*(1/dx**2 + 1/dy**2) + k**2
            res = f - (-lapU + k**2 * U)
            U[2:-2, 2:-2] += omega * res[2:-2, 2:-2] / D
            apply_bc(U)
        return U

    # V-cycle
    def v_cycle(U, f, dx, dy, k, level, max_level):
        # Pre-smoothing
        U = weighted_jacobi(U, f, dx, dy, k, omega=0.8, n_iter=5)
        # Compute residual
        res = f - (-laplacian_2d_4th(U, dx, dy) + k**2 * U)
        # Restrict residual to coarse grid
        if level < max_level and U.shape[0] > 9 and U.shape[1] > 9:
            rc = restrict(res)
            # Zero initial guess for error on coarse grid
            ec = np.zeros_like(rc)
            # Grid spacings double
            dx2 = dx * 2
            dy2 = dy * 2
            # Recursively solve error equation
            ec = v_cycle(ec, rc, dx2, dy2, k, level+1, max_level)
            # Prolongate error and correct
            ef = prolong(ec)
            # Only correct interior
            # The interior of U is [2:-2,2:-2], ef is same shape as U
            U[2:-2, 2:-2] += ef[2:-2, 2:-2]
            apply_bc(U)
            # Post-smoothing
            U = weighted_jacobi(U, f, dx, dy, k, omega=0.8, n_iter=5)
        return U

    # Multigrid solver: F-cycle (several V-cycles)
    def multigrid_solver(U, f, dx, dy, k, n_cycles=10):
        for _ in range(n_cycles):
            U = v_cycle(U, f, dx, dy, k, level=1, max_level=5)
        return U

    # --- 7. Initial guess ---
    U = np.zeros((Nx, Ny))
    apply_bc(U)

    # --- 8. Solve using multigrid ---
    U = multigrid_solver(U, f, dx, dy, k, n_cycles=15)
    apply_bc(U)

    # --- 9. Compute residual grid ---
    residual = f - (-laplacian_2d_4th(U, dx, dy) + k**2 * U)

    # --- 10. Return ---
    result = {
        "u": U,
        "coords": {"x": x, "y": y},
        "t": None
    }
    return result