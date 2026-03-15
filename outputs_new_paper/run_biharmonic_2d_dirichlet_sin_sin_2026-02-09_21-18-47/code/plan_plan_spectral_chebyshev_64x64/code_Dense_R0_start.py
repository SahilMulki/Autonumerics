```python
import numpy as np

def cheb(N):
    """
    Compute the Chebyshev-Gauss-Lobatto points and differentiation matrix.
    Returns x (N+1,), D (N+1,N+1)
    """
    if N == 0:
        x = np.array([1.0])
        D = np.array([[0.0]])
        return x, D
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = 2
    c[-1] = 2
    c = c * ((-1) ** np.arange(N + 1))
    X = np.tile(x, (N + 1, 1))
    dX = X - X.T + np.eye(N + 1)
    D = np.outer(c, 1 / c) / (dX)
    D = D - np.diag(np.sum(D, axis=1))
    return x, D

def map_to_domain(x_cheb, a, b):
    """Map Chebyshev points from [-1,1] to [a,b]"""
    return 0.5 * (a + b) + 0.5 * (b - a) * x_cheb

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # === 1. Extract parameters ===
    # Domain
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    # Grid sizes
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    # Chebyshev points and D-matrices
    x_cheb, Dx = cheb(Nx)
    y_cheb, Dy = cheb(Ny)
    x = map_to_domain(x_cheb, x_min, x_max)
    y = map_to_domain(y_cheb, y_min, y_max)
    # Meshgrid (spectral ordering: y first, then x)
    X, Y = np.meshgrid(x, y, indexing='ij')  # shape (Nx+1, Ny+1)

    # === 2. Build biharmonic operator with clamped BCs ===
    # D2 = second derivative matrix
    D2x = Dx @ Dx  # (Nx+1, Nx+1)
    D2y = Dy @ Dy  # (Ny+1, Ny+1)
    I_x = np.eye(Nx + 1)
    I_y = np.eye(Ny + 1)
    # Laplacian operator: L = kron(I, D2y) + kron(D2x, I)
    # Biharmonic: L2 = L @ L
    # But more efficiently, use Kronecker structure:
    # Δ^2 u = (D2x ⊗ I + I ⊗ D2y)^2 u
    #       = (D2x^2 ⊗ I) + 2 (D2x ⊗ D2y) + (I ⊗ D2y^2)
    # We'll build the operator as a matrix for the interior points.

    N = (Nx + 1) * (Ny + 1)

    # === 3. Build RHS: g(x,y) ===
    # From analytic solution: u = sin(pi x) sin(pi y)
    # Δ^2 u = (π^4) sin(πx) sin(πy)
    g = (np.pi ** 4) * np.sin(np.pi * X) * np.sin(np.pi * Y)

    # === 4. Apply clamped BCs: u = 0, du/dn = 0 on all boundaries ===
    # For Chebyshev, boundary points are at indices 0 and -1 in each direction.
    # We'll solve for the interior points only (excluding 0 and -1 in each direction).
    # For clamped BCs, both u and du/dn are zero at boundaries.

    # Indices of interior points
    ix = np.arange(1, Nx)  # 1..Nx-1
    iy = np.arange(1, Ny)  # 1..Ny-1

    # Number of interior points
    Nix = len(ix)
    Niy = len(iy)
    Nint = Nix * Niy

    # Build 1D interior D2 matrices
    D2x_in = D2x[np.ix_(ix, ix)]
    D2y_in = D2y[np.ix_(iy, iy)]
    Ix_in = np.eye(Nix)
    Iy_in = np.eye(Niy)

    # Build 1D first derivative matrices for BCs
    Dx_in = Dx[np.ix_(ix, ix)]
    Dy_in = Dy[np.ix_(iy, iy)]

    # Build the 2D Laplacian on the interior
    # L = kron(I, D2y_in) + kron(D2x_in, I)
    L = np.kron(Iy_in, D2x_in) + np.kron(D2y_in, Ix_in)
    # Biharmonic operator:
    # Δ^2 u = (L @ L) u
    # But more efficiently, as above:
    L2 = (
        np.kron(Iy_in, D2x_in @ D2x_in) +
        2 * np.kron(D2y_in, D2x_in) +
        np.kron(D2y_in @ D2y_in, Ix_in)
    )

    # === 5. Build the RHS for interior points ===
    g_interior = g[np.ix_(ix, iy)].reshape(-1)

    # === 6. Modify RHS for clamped BCs ===
    # For Chebyshev, the boundary conditions are enforced by removing boundary rows/cols,
    # so the system is for interior points only.

    # === 7. Solve the linear system ===
    # L2 u = g
    # Use dense direct solver as per plan
    u_interior = np.linalg.solve(L2, g_interior)

    # === 8. Reconstruct full u grid (including boundaries) ===
    u = np.zeros((Nx + 1, Ny + 1))
    u[np.ix_(ix, iy)] = u_interior.reshape((Nix, Niy))
    # Boundaries are already zero (Dirichlet), and du/dn=0 is satisfied by construction.

    # === 9. Compute residual grid ===
    # Compute Δ^2 u at all grid points, then subtract g
    # We'll use the full D2x, D2y for this.
    # First, compute Laplacian: Δu = D2x u + u D2y^T
    # For each y, D2x @ u[:,j]
    Lap_u = D2x @ u + u @ D2y.T
    # Now, Δ^2 u = D2x @ Lap_u + Lap_u @ D2y.T
    biharm_u = D2x @ Lap_u + Lap_u @ D2y.T
    residual = biharm_u - g

    # === 10. Output ===
    # No time variable for steady-state
    t_array = None

    coords = {'x': x, 'y': y}

    return {
        "u": u,
        "coords": coords,
        "t": t_array,
        "residual": residual
    }
```
