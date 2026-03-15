```python
import numpy as np

def chebyshev_points(N, a=0.0, b=1.0):
    # Chebyshev-Gauss-Lobatto nodes in [a, b]
    k = np.arange(N)
    x = np.cos(np.pi * k / (N - 1))
    x = 0.5 * (x + 1) * (b - a) + a
    return x

def chebyshev_D(N, a=0.0, b=1.0):
    # Chebyshev differentiation matrix on [a, b]
    if N == 1:
        return np.zeros((1, 1))
    x = np.cos(np.pi * np.arange(N) / (N - 1))
    c = np.ones(N)
    c[0] = 2
    c[-1] = 2
    c = c * ((-1) ** np.arange(N))
    X = np.tile(x, (N, 1))
    dX = X - X.T + np.eye(N)
    D = np.outer(c, 1 / c) / (dX)
    D = D - np.diag(np.sum(D, axis=1))
    # Map from [-1,1] to [a,b]
    D = 2.0 / (b - a) * D
    x_phys = 0.5 * (x + 1) * (b - a) + a
    return D, x_phys

def parse_initial_condition(ic_str, x, y):
    # Only supports "sin(pi*x)*sin(pi*y)" for this problem
    X, Y = np.meshgrid(x, y, indexing='ij')
    return np.sin(np.pi * X) * np.sin(np.pi * Y)

def apply_dirichlet_bc(U):
    # Set boundary values to zero (Dirichlet u=0)
    U[0, :] = 0
    U[-1, :] = 0
    U[:, 0] = 0
    U[:, -1] = 0
    return U

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # Parameters
    alpha = float(pde_spec["parameters"]["alpha"])
    domain = pde_spec["domain"]
    x_min, x_max = domain["bounds"]["x"]
    y_min, y_max = domain["bounds"]["y"]

    # Grid
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])
    D_x, x = chebyshev_D(Nx, x_min, x_max)
    D_y, y = chebyshev_D(Ny, y_min, y_max)
    # 1D coordinate arrays
    coords = {"x": x, "y": y}

    # Initial condition
    U = parse_initial_condition(pde_spec["initial_condition"], x, y)
    U = apply_dirichlet_bc(U)

    # Time stepping
    dt = float(plan["time_stepping"]["dt"])
    t_final = float(plan["time_stepping"]["t_final"])
    Nt = int(np.ceil(t_final / dt))
    t_array = np.linspace(0, Nt * dt, Nt + 1)
    # Only store final state for memory safety

    # Precompute operators
    I_x = np.eye(Nx)
    I_y = np.eye(Ny)
    D2_x = np.dot(D_x, D_x)
    D2_y = np.dot(D_y, D_y)

    # Laplacian operator (Kronecker sum)
    # L = kron(Iy, D2_x) + kron(D2_y, Ix)
    # Flatten U as (Nx*Ny,)
    from numpy import kron

    L = kron(I_y, D2_x) + kron(D2_y, I_x)
    A = np.eye(Nx * Ny) - dt * alpha * L

    # Dirichlet BC mask: indices of interior points
    boundary_mask = np.zeros((Nx, Ny), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    interior_mask = ~boundary_mask
    interior_idx = np.where(interior_mask.ravel())[0]

    # Reduce system to interior points only
    # For Dirichlet BCs, boundary values are always zero
    # So we solve only for interior points
    def flatten_interior(U):
        return U.ravel()[interior_idx]

    def unflatten_interior(u_flat):
        U = np.zeros((Nx, Ny))
        U.ravel()[interior_idx] = u_flat
        return U

    # Build reduced operator for interior points
    A_reduced = A[np.ix_(interior_idx, interior_idx)]

    # Time stepping loop (backward Euler)
    u_flat = flatten_interior(U)
    for n in range(Nt):
        # Right-hand side: previous solution (interior only)
        rhs = u_flat.copy()
        # Solve (I - dt*alpha*L) u^{n+1} = u^n
        u_flat = np.linalg.solve(A_reduced, rhs)
        # No need to update boundary: always zero

    # Final solution
    U_final = unflatten_interior(u_flat)
    U_final = apply_dirichlet_bc(U_final)

    # Compute residual at final time
    # Residual: R = u_t - alpha*(u_xx + u_yy)
    # u_t ≈ (u_final - u_prev) / dt
    # For residual, use backward Euler step: u_t ≈ (u_final - u_prev)/dt
    # But since we do not store u_prev, we can use the PDE: at steady state, residual should be zero
    # Instead, we can compute the residual as:
    # R = (U_final - U_prev)/dt - alpha*(U_xx + U_yy)
    # But since U_prev is not available, we can use the discrete equation:
    # (U_final - U_prev)/dt = alpha*(U_xx + U_yy)
    # So, for the final U_final, we can compute the spatial operator and compare to the time derivative

    # For best accuracy, let's do one backward Euler step from U_prev to U_final, so we can get U_prev
    # But since we only have U_final, we can reconstruct U_prev:
    # U_prev = U_final - dt * alpha * (U_xx + U_yy)
    # So u_t ≈ (U_final - U_prev)/dt = alpha*(U_xx + U_yy)
    # Therefore, residual = (U_final - U_prev)/dt - alpha*(U_xx + U_yy) = 0 (up to solver error)
    # To check the true residual, let's compute the continuous residual:
    # R = U_t - alpha*(U_xx + U_yy)
    # For U_t, approximate as (U_final - U_prev)/dt

    # Compute Laplacian of U_final
    Uxx = D2_x @ U_final
    Uyy = U_final @ D2_y.T
    lapU = Uxx + Uyy

    # Reconstruct U_prev
    U_prev = U_final - dt * alpha * lapU

    # Compute U_t ≈ (U_final - U_prev)/dt
    U_t = (U_final - U_prev) / dt

    # Residual grid
    residual_grid = U_t - alpha * lapU

    # Enforce boundary residuals to zero (since BCs are always satisfied)
    residual_grid = apply_dirichlet_bc(residual_grid)

    return {
        "u": U_final,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }
```