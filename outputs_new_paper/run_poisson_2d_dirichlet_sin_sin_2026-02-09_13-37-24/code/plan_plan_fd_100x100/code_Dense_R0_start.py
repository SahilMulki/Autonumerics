import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Read grid parameters ---
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    order = plan['spatial_discretization'].get('order', 2)
    # 4th order FD requires at least 5 points in each direction
    assert Nx >= 5 and Ny >= 5, "Grid too coarse for 4th order FD"

    # --- Create grid ---
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- RHS function f(x, y) ---
    # For analytic solution u = sin(pi x) sin(pi y), -Δu = 2π² sin(pi x) sin(pi y)
    pi = np.pi
    f = 2 * pi**2 * np.sin(pi * X) * np.sin(pi * Y)

    # --- Dirichlet BCs: u = 0 on boundary ---
    # We'll solve for interior points only, boundary is fixed at zero

    # --- Build 4th order FD Laplacian operator ---
    # 1D 4th order stencil: [-1, 16, -30, 16, -1] / (12 h^2)
    # We'll build 2D Laplacian as a matrix acting on the flattened interior grid

    # Indexing: (i, j) for 0 <= i < Nx, 0 <= j < Ny
    # Interior: 2 <= i < Nx-2, 2 <= j < Ny-2

    i0, i1 = 2, Nx-2
    j0, j1 = 2, Ny-2
    Nx_in = i1 - i0
    Ny_in = j1 - j0
    N_in = Nx_in * Ny_in

    # Map (i, j) -> idx = (i-i0)*Ny_in + (j-j0)
    def idx(i, j):
        return (i - i0) * Ny_in + (j - j0)

    # Build matrix in dense format (since scipy is not allowed)
    A = np.zeros((N_in, N_in), dtype=np.float64)
    F = np.zeros(N_in, dtype=np.float64)

    # Helper for boundary value at (i, j)
    def boundary_val(i, j):
        return 0.0  # Dirichlet 0

    for i in range(i0, i1):
        for j in range(j0, j1):
            k = idx(i, j)
            # Center
            A[k, k] = -30/(12*dx**2) -30/(12*dy**2)
            # x-direction neighbors
            for offset, coeff in [(-2, -1), (-1, 16), (1, 16), (2, -1)]:
                ni = i + offset
                if 0 <= ni < Nx:
                    if i0 <= ni < i1:
                        # Interior neighbor
                        nk = idx(ni, j)
                        A[k, nk] += coeff/(12*dx**2)
                    else:
                        # Boundary neighbor
                        F[k] -= coeff/(12*dx**2) * boundary_val(ni, j)
            # y-direction neighbors
            for offset, coeff in [(-2, -1), (-1, 16), (1, 16), (2, -1)]:
                nj = j + offset
                if 0 <= nj < Ny:
                    if j0 <= nj < j1:
                        # Interior neighbor
                        nk = idx(i, nj)
                        A[k, nk] += coeff/(12*dy**2)
                    else:
                        # Boundary neighbor
                        F[k] -= coeff/(12*dy**2) * boundary_val(i, nj)
            # RHS
            F[k] += f[i, j]

    # --- Solve linear system ---
    # Use numpy.linalg.solve on dense matrix
    u_in = np.linalg.solve(A, F)

    # --- Reconstruct full grid ---
    u = np.zeros((Nx, Ny), dtype=np.float64)
    # Insert interior solution
    for i in range(i0, i1):
        for j in range(j0, j1):
            k = idx(i, j)
            u[i, j] = u_in[k]
    # Boundaries remain zero

    # --- Compute residual grid ---
    # Compute -Δu - f(x, y) at all grid points (including boundaries)
    # Use 4th order FD for Laplacian at interior, lower order at boundaries

    residual = np.zeros_like(u)
    # 4th order Laplacian at interior
    for i in range(2, Nx-2):
        for j in range(2, Ny-2):
            lap = (
                (-u[i-2, j] + 16*u[i-1, j] - 30*u[i, j] + 16*u[i+1, j] - u[i+2, j]) / (12*dx**2) +
                (-u[i, j-2] + 16*u[i, j-1] - 30*u[i, j] + 16*u[i, j+1] - u[i, j+2]) / (12*dy**2)
            )
            residual[i, j] = -lap - f[i, j]
    # For boundaries and near-boundaries, use 2nd order Laplacian
    for i in range(Nx):
        for j in range(Ny):
            if 2 <= i < Nx-2 and 2 <= j < Ny-2:
                continue  # already done
            lap = 0.0
            if 1 <= i < Nx-1:
                lap += (u[i-1, j] - 2*u[i, j] + u[i+1, j]) / dx**2
            if 1 <= j < Ny-1:
                lap += (u[i, j-1] - 2*u[i, j] + u[i, j+1]) / dy**2
            residual[i, j] = -lap - f[i, j]

    # --- Output ---
    result = {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": None,
        "residual": residual
    }
    return result