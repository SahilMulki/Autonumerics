```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    # Spectral grid sizes
    Nx = plan["spatial_discretization"]["Nx"]
    Ny = plan["spatial_discretization"]["Ny"]
    # PDE parameters
    k = float(pde_spec["parameters"]["k"])
    pi = np.pi

    # --- 2. Chebyshev-Gauss-Lobatto points and differentiation matrices ---
    def cheb(N):
        if N == 1:
            x = np.array([0.])
            D = np.array([[0.]])
            return x, D
        x = np.cos(np.pi * np.arange(N) / (N - 1))
        c = np.ones(N)
        c[0] = 2
        c[-1] = 2
        c = c * ((-1) ** np.arange(N))
        X = np.tile(x, (N, 1))
        dX = X - X.T + np.eye(N)
        D = (np.outer(c, 1/c)) / (dX)
        D = D - np.diag(np.sum(D, axis=1))
        return x, D

    # Chebyshev points in [-1,1], map to [a,b]
    x_cheb, Dx = cheb(Nx)
    y_cheb, Dy = cheb(Ny)
    # Map to [x_min, x_max], [y_min, y_max]
    x = 0.5 * (x_cheb + 1) * (x_max - x_min) + x_min
    y = 0.5 * (y_cheb + 1) * (y_max - y_min) + y_min
    # Scaling for derivatives
    Dx = 2.0 / (x_max - x_min) * Dx
    Dy = 2.0 / (y_max - y_min) * Dy

    # --- 3. 2D grid ---
    X, Y = np.meshgrid(x, y, indexing='ij')  # shape (Nx, Ny)

    # --- 4. Source term f(x, y) ---
    f = (2 * pi ** 2 + k ** 2) * np.sin(pi * X) * np.sin(pi * Y)

    # --- 5. Dirichlet BCs: u=0 on boundary ---
    # We'll solve for interior points only, set boundary to zero

    # Indices for interior points
    ix = np.arange(1, Nx-1)
    iy = np.arange(1, Ny-1)
    Nix = Nx - 2
    Niy = Ny - 2

    # --- 6. Build Laplacian operator ---
    # 1D second derivative matrices
    D2x = np.dot(Dx, Dx)
    D2y = np.dot(Dy, Dy)
    # Restrict to interior
    D2x_in = D2x[1:-1,1:-1]
    D2y_in = D2y[1:-1,1:-1]

    # 2D Laplacian via Kronecker sum
    Ix = np.eye(Nix)
    Iy = np.eye(Niy)
    Lx = np.kron(D2x_in, Iy)
    Ly = np.kron(Ix, D2y_in)
    Lap = Lx + Ly  # shape ((Nix*Niy), (Nix*Niy))

    # --- 7. Assemble linear system: (-Lap + k^2*I) u = f_interior ---
    A = -Lap + k**2 * np.eye(Nix*Niy)
    # Flatten f at interior
    f_interior = f[1:-1,1:-1].reshape(-1)

    # --- 8. Solve linear system ---
    u_interior = np.linalg.solve(A, f_interior)

    # --- 9. Build full solution grid ---
    u = np.zeros((Nx, Ny))
    u[1:-1,1:-1] = u_interior.reshape((Nix, Niy))

    # --- 10. Compute pointwise PDE residual ---
    # Compute -Δu + k^2 u - f at all grid points
    # For boundary, residual is just u (since u=0, residual=0)
    # For interior, compute Laplacian using spectral D2x, D2y

    # Compute second derivatives
    u_xx = D2x @ u  # shape (Nx, Ny)
    u_yy = u @ D2y.T  # shape (Nx, Ny)
    Lap_u = u_xx + u_yy
    residual = -Lap_u + k**2 * u - f

    # --- 11. Return ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": None,
        "residual": residual
    }
```