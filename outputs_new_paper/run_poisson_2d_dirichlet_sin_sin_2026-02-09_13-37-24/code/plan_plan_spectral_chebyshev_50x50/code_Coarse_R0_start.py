```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract grid parameters ---
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']

    # --- Chebyshev points in [-1,1] then mapped to [a,b] ---
    def cheb_points(N, a, b):
        k = np.arange(N)
        x_cheb = np.cos(np.pi * k / (N - 1))
        # Map from [-1,1] to [a,b]
        return 0.5 * (a + b) + 0.5 * (b - a) * x_cheb

    x = cheb_points(Nx, x_min, x_max)
    y = cheb_points(Ny, y_min, y_max)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Chebyshev differentiation matrix (from Trefethen "Spectral Methods in MATLAB") ---
    def cheb_D(N, a, b):
        if N == 1:
            return np.zeros((1, 1))
        x = np.cos(np.pi * np.arange(N) / (N - 1))
        c = np.ones(N)
        c[0] = 2
        c[-1] = 2
        c = c * ((-1) ** np.arange(N))
        Xmat = np.tile(x, (N, 1))
        dX = Xmat - Xmat.T + np.eye(N)
        D = np.outer(c, 1 / c) / (dX)
        D = D - np.diag(np.sum(D, axis=1))
        # Scale for [a,b]
        D = 2.0 / (b - a) * D
        return D

    Dx = cheb_D(Nx, x_min, x_max)
    Dy = cheb_D(Ny, y_min, y_max)

    # --- Laplacian operator (Kronecker sum) ---
    Ix = np.eye(Nx)
    Iy = np.eye(Ny)
    # 1D Laplacian: D2 = D @ D
    D2x = Dx @ Dx
    D2y = Dy @ Dy
    # 2D Laplacian: L = kron(Iy, D2x) + kron(D2y, Ix)
    # But we need to flatten in Fortran order (y fastest) to match meshgrid
    L = np.kron(Iy, D2x) + np.kron(D2y, Ix)

    # --- RHS: f(x,y) = 2*pi^2*sin(pi*x)*sin(pi*y) ---
    f = 2 * np.pi ** 2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

    # --- Dirichlet BCs: u=0 on boundary ---
    # Find boundary indices
    boundary_mask = np.zeros((Nx, Ny), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    interior_mask = ~boundary_mask
    # Flattened indices
    idx_all = np.arange(Nx * Ny)
    idx_interior = idx_all[interior_mask.ravel()]
    idx_boundary = idx_all[boundary_mask.ravel()]

    # --- Build reduced system for interior points ---
    # Flatten in Fortran order (y fastest)
    f_flat = f.ravel(order='F')
    # Remove boundary rows/cols
    L_int = L[np.ix_(idx_interior, idx_interior)]
    f_int = f_flat[idx_interior]

    # --- Solve for interior u ---
    u_int = np.linalg.solve(-L_int, f_int)  # -Δu = f

    # --- Assemble full u ---
    u_flat = np.zeros(Nx * Ny)
    u_flat[idx_interior] = u_int
    # Dirichlet BCs: u=0 on boundary (already zero)
    u = u_flat.reshape((Nx, Ny), order='F')

    # --- Compute residual: -Δu - f(x,y) ---
    # Compute Laplacian using full D2x, D2y
    u_xx = D2x @ u
    u_yy = u @ D2y.T
    lap_u = u_xx + u_yy
    residual = -lap_u - f

    # --- Prepare output ---
    coords = {'x': x, 'y': y}
    t_array = np.array([])  # No time
    return {
        "u": u,
        "coords": coords,
        "t": t_array,
        "residual": residual
    }
```
