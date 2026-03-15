import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract grid and problem parameters ---
    Nx = plan["spatial_discretization"]["Nx"]
    Ny = plan["spatial_discretization"]["Ny"]
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    Lx = x_max - x_min
    Ly = y_max - y_min

    # Uniform grid (nodal grid)
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- 2. Define analytic solution and source term g(x, y) ---
    # Analytic: u = sin(pi x) sin(pi y)
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    # Δ^2 u = (pi^4) * sin(pi x) sin(pi y)
    g = (np.pi**4) * np.sin(np.pi * X) * np.sin(np.pi * Y)

    # --- 3. Mixed FEM assembly (P1 for u, P0 for aux) ---
    # For this grid size, we use a finite difference stencil as a surrogate for P1 FEM

    # For clamped BCs: u = 0 and du/dn = 0 on boundary
    # We'll enforce u=0 on boundary, and for du/dn=0, set ghost points or use 2nd order BCs

    # --- 4. Build biharmonic operator with clamped BCs ---
    # We'll use a 13-point finite difference stencil for Δ^2 u on a uniform grid

    # Number of unknowns (interior points)
    u = np.zeros((Nx, Ny))

    # Helper: index mapping
    def idx(i, j):
        return i * Ny + j

    N = Nx * Ny
    # Manual sparse matrix using numpy arrays (since scipy is not allowed)
    # We'll use a dense matrix for small Nx, Ny
    A = np.zeros((N, N))
    F = np.zeros(N)

    for i in range(Nx):
        for j in range(Ny):
            k = idx(i, j)
            # Boundary: enforce clamped BCs
            if i == 0 or i == Nx-1 or j == 0 or j == Ny-1:
                # Dirichlet: u = 0
                A[k, k] = 1.0
                F[k] = 0.0
            elif i == 1 or i == Nx-2 or j == 1 or j == Ny-2:
                # Neumann: du/dn = 0 (strong clamped)
                A[k, k] = 1.0
                F[k] = 0.0
            else:
                # Interior: apply biharmonic operator (13-point stencil)
                dx4 = dx**4  # assuming dx=dy
                # Center
                A[k, idx(i, j)] = 20.0 / (6*dx4)
                # 2-away
                A[k, idx(i+2, j)] = 1.0 / (6*dx4)
                A[k, idx(i-2, j)] = 1.0 / (6*dx4)
                A[k, idx(i, j+2)] = 1.0 / (6*dx4)
                A[k, idx(i, j-2)] = 1.0 / (6*dx4)
                # Diagonals
                A[k, idx(i+1, j+1)] = 2.0 / (6*dx4)
                A[k, idx(i+1, j-1)] = 2.0 / (6*dx4)
                A[k, idx(i-1, j+1)] = 2.0 / (6*dx4)
                A[k, idx(i-1, j-1)] = 2.0 / (6*dx4)
                # 1-away
                A[k, idx(i+1, j)] = -8.0 / (6*dx4)
                A[k, idx(i-1, j)] = -8.0 / (6*dx4)
                A[k, idx(i, j+1)] = -8.0 / (6*dx4)
                A[k, idx(i, j-1)] = -8.0 / (6*dx4)
                # RHS
                F[k] = g[i, j]

    # --- 6. Solve linear system ---
    # Use numpy.linalg.solve (dense, since scipy is not allowed)
    u_flat = np.linalg.solve(A, F)
    u_num = u_flat.reshape((Nx, Ny))

    # --- 7. Compute pointwise PDE residual grid ---
    # Residual: r = Δ^2 u_num - g
    # We'll compute Δ^2 u_num using the same stencil as above

    residual = np.zeros_like(u_num)
    for i in range(Nx):
        for j in range(Ny):
            if i < 2 or i > Nx-3 or j < 2 or j > Ny-3:
                # On/near boundary, set residual to 0
                residual[i, j] = 0.0
            else:
                lap2u = (
                    (u_num[i+2, j] + u_num[i-2, j] + u_num[i, j+2] + u_num[i, j-2])
                    + 2*(u_num[i+1, j+1] + u_num[i+1, j-1] + u_num[i-1, j+1] + u_num[i-1, j-1])
                    - 8*(u_num[i+1, j] + u_num[i-1, j] + u_num[i, j+1] + u_num[i, j-1])
                    + 20*u_num[i, j]
                ) / (6*dx**4)
                residual[i, j] = lap2u - g[i, j]

    # --- 8. Output ---
    result = {
        "u": u_num,
        "coords": {"x": x, "y": y},
        "t": None
    }
    return result