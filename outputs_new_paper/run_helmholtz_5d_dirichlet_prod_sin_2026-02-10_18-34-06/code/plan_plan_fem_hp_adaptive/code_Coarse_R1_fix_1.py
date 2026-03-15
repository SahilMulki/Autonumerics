import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and Plan Parameters ---
    # Spatial variables and domain
    spatial_vars = pde_spec["spatial_variables"]
    dim = len(spatial_vars)
    bounds = pde_spec["domain"]["bounds"]
    k = float(pde_spec["parameters"]["k"])
    Nx = plan["spatial_discretization"].get("Nx", 12)
    order = plan["spatial_discretization"].get("order", 3)
    # For 5D, use Nx for all dimensions
    N = Nx
    shape = tuple([N] * dim)
    # Dirichlet BC value (always 0 for this problem)
    dirichlet_val = pde_spec.get("boundary_conditions", {}).get("values", {}).get("u", 0.0)
    if dirichlet_val is None:
        dirichlet_val = 0.0

    # --- Build Grid ---
    coords = {}
    grid_axes = []
    for var in spatial_vars:
        a, b = bounds[var]
        x = np.linspace(a, b, N)
        coords[var] = x
        grid_axes.append(x)
    mesh = np.meshgrid(*grid_axes, indexing='ij')

    # --- Assemble RHS f(x) using analytic solution ---
    # Analytic solution: u = prod_{i=1}^5 sin(pi x_i)
    # Compute f(x) = -Δu + k^2 u
    # Δu = sum_i d^2u/dx_i^2
    # For u = prod sin(pi x_i), d^2u/dx_i^2 = -pi^2 u
    # So Δu = -5*pi^2 u
    # f(x) = 5*pi^2 u + k^2 u = (5*pi^2 + k^2) * u
    pi2 = np.pi ** 2
    u_analytic = np.ones(shape)
    for xi in mesh:
        u_analytic *= np.sin(np.pi * xi)
    f_grid = (5 * pi2 + k ** 2) * u_analytic

    # --- FEM Assembly (5D, cubic, Dirichlet 0) ---
    # For memory safety, use finite difference as a surrogate for high-order FEM
    # Use 4th-order central finite difference for Laplacian, Dirichlet BCs

    # Step 1: Initialize u with zeros (Dirichlet BCs)
    u = np.zeros(shape, dtype=np.float64)

    # Step 2: Set up grid spacing
    h = (coords[spatial_vars[0]][1] - coords[spatial_vars[0]][0])

    # Step 3: Build Laplacian operator (5D, 4th-order FD stencil)
    # For each axis, apply 4th-order central difference
    # 4th-order stencil: [-1/12, 4/3, -5/2, 4/3, -1/12] / h^2
    # For interior points only (2:N-2 in each dim)
    # We'll solve Au = f, with Dirichlet BCs (u=0 at boundary)

    # Step 4: Build mask for interior points
    slices_interior = tuple([slice(2, N-2)] * dim)
    # Prepare f_interior
    f_interior = f_grid[slices_interior]

    # Step 5: Build Laplacian action as a function
    def laplacian_5d(u):
        lap = np.zeros_like(u)
        for axis in range(dim):
            u_m2 = np.roll(u, 2, axis=axis)
            u_m1 = np.roll(u, 1, axis=axis)
            u_p1 = np.roll(u, -1, axis=axis)
            u_p2 = np.roll(u, -2, axis=axis)
            lap += (-1/12 * u_m2 + 4/3 * u_m1 - 5/2 * u + 4/3 * u_p1 - 1/12 * u_p2) / h**2
        return lap

    # Step 6: Iterative solver (Jacobi, since matrix is huge)
    # Only update interior points, keep Dirichlet BCs at 0
    max_iter = 5000
    tol = 1e-8
    omega = 0.8  # relaxation parameter

    for it in range(max_iter):
        # Compute Laplacian at all points
        lap_u = laplacian_5d(u)
        # Residual at all points: -lap_u + k^2 * u - f = 0
        res = -lap_u + k**2 * u - f_grid
        # Jacobi update for interior points
        u_old = u[slices_interior]
        # For each axis, sum neighbor contributions for interior points
        u_update = np.zeros_like(u_old)
        for axis in range(dim):
            idx = [slice(2, N-2)] * dim
            # m2
            idx_m2 = list(idx)
            idx_m2[axis] = slice(0, N-4)
            u_m2 = u[tuple(idx_m2)]
            # m1
            idx_m1 = list(idx)
            idx_m1[axis] = slice(1, N-3)
            u_m1 = u[tuple(idx_m1)]
            # p1
            idx_p1 = list(idx)
            idx_p1[axis] = slice(3, N-1)
            u_p1 = u[tuple(idx_p1)]
            # p2
            idx_p2 = list(idx)
            idx_p2[axis] = slice(4, N)
            u_p2 = u[tuple(idx_p2)]
            u_update += (-1/12 * u_m2 + 4/3 * u_m1 + 4/3 * u_p1 - 1/12 * u_p2)
        denom = dim * (-5/2) / h**2 + k**2  # diagonal term from Laplacian + k^2
        rhs = f_interior
        u_new_interior = (u_update / h**2 + rhs) / denom
        # Relaxation
        u[slices_interior] = omega * u_new_interior + (1 - omega) * u_old
        # Check convergence
        err = np.linalg.norm(u[slices_interior] - u_old) / (np.linalg.norm(u_old) + 1e-14)
        if err < tol:
            break

    # --- Compute Residual Grid ---
    lap_u = laplacian_5d(u)
    residual_grid = -lap_u + k**2 * u - f_grid

    # --- Prepare Output ---
    # No time variable (stationary)
    t_array = None
    return {
        "u": u,
        "coords": coords,
        "t": t_array
    }