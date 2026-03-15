```python
import numpy as np

def chebyshev_points(N, a=0.0, b=1.0):
    # Gauss-Lobatto Chebyshev points on [a, b]
    k = np.arange(N)
    x = np.cos(np.pi * k / (N - 1))
    # Map from [-1, 1] to [a, b]
    x_mapped = 0.5 * (a + b) + 0.5 * (b - a) * x
    return x_mapped[::-1]  # ascending order

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
    # Scale for [a, b]
    D = 2.0 / (b - a) * D
    return D

def parse_bc_expr(expr):
    # Accepts e.g. "sin(pi*x)", returns a lambda x: ...
    import math
    def f(x):
        return eval(expr, {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "pi": np.pi, "exp": np.exp, "sinh": np.sinh, "cosh": np.cosh, "math": math})
    return f

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract grid parameters ---
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']

    # --- 2. Chebyshev collocation points and differentiation matrices ---
    x = chebyshev_points(Nx, x_min, x_max)
    y = chebyshev_points(Ny, y_min, y_max)
    Dx = chebyshev_D(Nx, x_min, x_max)
    Dy = chebyshev_D(Ny, y_min, y_max)

    # --- 3. Boundary conditions ---
    bc = pde_spec['boundary_conditions']['values']
    # Parse boundary expressions
    bc_x0 = parse_bc_expr(bc['u(0,y)'])
    bc_x1 = parse_bc_expr(bc['u(1,y)'])
    bc_y0 = parse_bc_expr(bc['u(x,0)'])
    bc_y1 = parse_bc_expr(bc['u(x,1)'])

    # --- 4. Build the 2D Laplace operator (Kronecker sum) ---
    # u_xx + u_yy = 0
    # For interior points only (excluding boundaries)
    # Indices: 0 ... Nx-1, 0 ... Ny-1
    # Boundary: i=0, i=Nx-1, j=0, j=Ny-1
    # Interior: i=1..Nx-2, j=1..Ny-2
    ix_interior = np.arange(1, Nx-1)
    iy_interior = np.arange(1, Ny-1)
    Nix = len(ix_interior)
    Niy = len(iy_interior)
    N_unknowns = Nix * Niy

    # 1D interior D2
    D2x = Dx @ Dx
    D2y = Dy @ Dy
    D2x_int = D2x[np.ix_(ix_interior, ix_interior)]
    D2y_int = D2y[np.ix_(iy_interior, iy_interior)]

    # 2D Laplacian: kron(Iy, D2x) + kron(D2y, Ix)
    Ix = np.eye(Nix)
    Iy = np.eye(Niy)
    L = np.kron(Iy, D2x_int) + np.kron(D2y_int, Ix)  # shape (N_unknowns, N_unknowns)

    # --- 5. Build the right-hand side (incorporate Dirichlet BCs) ---
    # u is (Nx, Ny), but we solve for u[1:-1, 1:-1] (flattened)
    rhs = np.zeros(N_unknowns)

    # For each interior point, add BC contributions from boundaries
    # For each (i, j) in interior:
    for j, jj in enumerate(iy_interior):
        yj = y[jj]
        for i, ii in enumerate(ix_interior):
            xi = x[ii]
            idx = j * Nix + i
            # x boundaries
            if ii == 1:  # left neighbor is boundary at i=0
                rhs[idx] -= D2x[ii, 0] * bc_x0(yj)
            if ii == Nx-2:  # right neighbor is boundary at i=Nx-1
                rhs[idx] -= D2x[ii, Nx-1] * bc_x1(yj)
            # y boundaries
            if jj == 1:  # bottom neighbor is boundary at j=0
                rhs[idx] -= D2y[jj, 0] * bc_y0(xi)
            if jj == Ny-2:  # top neighbor is boundary at j=Ny-1
                rhs[idx] -= D2y[jj, Ny-1] * bc_y1(xi)

    # --- 6. Solve the linear system ---
    # L u_int = rhs
    u_int = np.linalg.solve(L, rhs)

    # --- 7. Assemble full solution grid ---
    u = np.zeros((Nx, Ny))
    # Set interior
    u[np.ix_(ix_interior, iy_interior)] = u_int.reshape((Nix, Niy))
    # Set boundaries
    # x=0 and x=1
    for j in range(Ny):
        u[0, j] = bc_x0(y[j])
        u[-1, j] = bc_x1(y[j])
    # y=0 and y=1
    for i in range(Nx):
        u[i, 0] = bc_y0(x[i])
        u[i, -1] = bc_y1(x[i])

    # --- 8. Compute pointwise PDE residual ---
    # Compute u_xx + u_yy at all points
    u_xx = Dx @ (Dx @ u)
    u_yy = (u @ Dy.T) @ Dy.T
    residual = u_xx + u_yy

    # --- 9. Prepare output ---
    coords = {'x': x, 'y': y}
    t_array = np.array([])  # No time
    return {
        "u": u,
        "coords": coords,
        "t": t_array,
        "residual": residual
    }
```
