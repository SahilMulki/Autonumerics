```python
import numpy as np

def chebyshev_points(N, a=0.0, b=1.0):
    # Gauss-Lobatto points on [-1,1]
    k = np.arange(N)
    x_cheb = np.cos(np.pi * k / (N - 1))
    # Map to [a, b]
    x_phys = 0.5 * (a + b) + 0.5 * (b - a) * x_cheb
    return x_phys

def chebyshev_D_matrix(N, a=0.0, b=1.0):
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
    D = np.outer(c, 1/c) / (dX)
    D = D - np.diag(np.sum(D, axis=1))
    # Map from [-1,1] to [a,b]
    D = 2.0 / (b - a) * D
    return D

def parse_bc_expr(expr):
    # expr is a string like "sin(pi*x)"
    # returns a lambda function of x or y
    import math
    from numpy import sin, cos, pi, exp, sinh, cosh
    # Only safe math functions
    def f(x, y=None):
        return eval(expr, {"x": x, "y": y, "sin": np.sin, "cos": np.cos, "pi": np.pi,
                           "exp": np.exp, "sinh": np.sinh, "cosh": np.cosh, "np": np})
    return f

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # 1. Parse grid size and domain
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']

    # 2. Chebyshev collocation points and differentiation matrices
    x = chebyshev_points(Nx, x_min, x_max)
    y = chebyshev_points(Ny, y_min, y_max)
    Dx = chebyshev_D_matrix(Nx, x_min, x_max)
    Dy = chebyshev_D_matrix(Ny, y_min, y_max)

    # 3. Meshgrid for evaluation
    X, Y = np.meshgrid(x, y, indexing='ij')  # shape (Nx, Ny)

    # 4. Boundary conditions
    bc = pde_spec['boundary_conditions']['values']
    # Parse BC expressions
    bc_x0 = parse_bc_expr(bc['u(0,y)'])
    bc_x1 = parse_bc_expr(bc['u(1,y)'])
    bc_y0 = parse_bc_expr(bc['u(x,0)'])
    bc_y1 = parse_bc_expr(bc['u(x,1)'])

    # 5. Build the linear system Au = 0 with Dirichlet BCs
    # Unknowns: u[1:Nx-2, 1:Ny-2] (interior points)
    u = np.zeros((Nx, Ny))

    # Set boundary values
    u[0, :] = bc_x0(x[0], y)
    u[-1, :] = bc_x1(x[-1], y)
    u[:, 0] = bc_y0(x, y[0])
    u[:, -1] = bc_y1(x, y[-1])

    # Flatten interior indices
    interior_idx = [(i, j) for i in range(1, Nx-1) for j in range(1, Ny-1)]
    N_interior = len(interior_idx)

    # Build system matrix and RHS
    # For each interior point, row in A corresponds to:
    # (D2x u + u D2y^T)[i,j] = 0
    D2x = Dx @ Dx
    D2y = Dy @ Dy

    A = np.zeros((N_interior, N_interior))
    b = np.zeros(N_interior)

    # Map (i,j) <-> flat index
    idx_map = {(i, j): k for k, (i, j) in enumerate(interior_idx)}

    for row, (i, j) in enumerate(interior_idx):
        # Laplacian at (i,j): sum over k: D2x[i,k] * u[k,j] + sum over l: u[i,l] * D2y[j,l]
        # For each unknown u[p,q], if (p,q) is interior, its coefficient is D2x[i,p] * delta_{q,j} + delta_{i,p} * D2y[j,q}
        for p in range(1, Nx-1):
            if (p, j) in idx_map:
                col = idx_map[(p, j)]
                A[row, col] += D2x[i, p]
        for q in range(1, Ny-1):
            if (i, q) in idx_map:
                col = idx_map[(i, q)]
                A[row, col] += D2y[j, q]
        # Move BCs to RHS
        # x boundaries
        b[row] -= D2x[i, 0] * u[0, j] + D2x[i, Nx-1] * u[Nx-1, j]
        # y boundaries
        b[row] -= D2y[j, 0] * u[i, 0] + D2y[j, Ny-1] * u[i, Ny-1]

    # 6. Solve the linear system
    u_interior = np.linalg.solve(A, b)

    # 7. Fill in the solution array
    for k, (i, j) in enumerate(interior_idx):
        u[i, j] = u_interior[k]

    # 8. Compute the residual grid at all points
    # Residual: r = u_xx + u_yy
    # Use spectral differentiation for all points
    u_xx = Dx @ (Dx @ u)
    u_yy = (u @ Dy.T) @ Dy.T
    residual = u_xx + u_yy

    # 9. Prepare output
    coords = {'x': x, 'y': y}
    t_array = np.array([])  # No time
    return {
        "u": u,
        "coords": coords,
        "t": t_array,
        "residual": residual
    }
```
