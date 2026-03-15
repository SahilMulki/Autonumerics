import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    """
    Solve the steady 2D Stokes equations with manufactured solution using a spectral (Chebyshev) method
    and tau method for Dirichlet BCs, as specified in the plan.

    Returns:
        {
          "u": u,                           # ndarray solution (u, v, p) at grid points
          "coords": {"x": x, "y": y},       # 1D coordinate arrays
          "t": None,                        # 1D array of time steps (None for steady)
          "residual": residual              # L2 error between numerical and analytic solution (u,v)
        }
    """
    # --- 1. Parse grid parameters ---
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']

    # --- 2. Build Chebyshev grid (for tau method with Dirichlet BCs) ---
    # Chebyshev-Gauss-Lobatto points in [-1,1]
    def cheb_points(N):
        k = np.arange(N)
        return np.cos(np.pi * k / (N - 1))

    x_cheb = cheb_points(Nx)
    y_cheb = cheb_points(Ny)
    # Map [-1,1] -> [x_min,x_max], [y_min,y_max]
    x = 0.5 * (x_cheb + 1) * (x_max - x_min) + x_min
    y = 0.5 * (y_cheb + 1) * (y_max - y_min) + y_min

    # 2D meshgrid (y, x) for array broadcasting
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- 3. Chebyshev differentiation matrices ---
    # See Trefethen, "Spectral Methods in MATLAB", p.54
    def cheb_D(N):
        if N == 1:
            return np.zeros((1, 1))
        x = np.cos(np.pi * np.arange(N) / (N - 1))
        c = np.ones(N)
        c[0] = 2
        c[-1] = 2
        c = c * ((-1) ** np.arange(N))
        Xmat = np.tile(x, (N, 1))
        dX = Xmat - Xmat.T
        D = np.outer(c, 1 / c) / (dX + np.eye(N))
        D = D - np.diag(np.sum(D, axis=1))
        return D

    Dx = cheb_D(Nx)
    Dy = cheb_D(Ny)

    # --- 4. Build Laplacian operator (Kronecker sum) ---
    Ix = np.eye(Nx)
    Iy = np.eye(Ny)
    D2x = Dx @ Dx
    D2y = Dy @ Dy

    # Laplacian: L = kron(Iy, D2x) + kron(D2y, Ix)
    # For 2D flattening: (y,x) -> i = y*Nx + x
    def kron_sum(A, B):
        return np.kron(np.eye(B.shape[0]), A) + np.kron(B, np.eye(A.shape[0]))

    Lap = kron_sum(D2x, D2y)  # Shape: (Nx*Ny, Nx*Ny)

    # --- 5. Manufactured solution and source terms ---
    pi = np.pi
    u_exact = np.sin(pi * X) * np.sin(pi * Y)
    v_exact = np.cos(pi * X) * np.cos(pi * Y)
    p_exact = np.sin(pi * X) * np.cos(pi * Y)

    # Compute f1, f2 from analytic solution
    # -Δu + p_x = f1
    # -Δv + p_y = f2

    # Compute derivatives using analytic expressions
    u_xx = -pi ** 2 * np.sin(pi * X) * np.sin(pi * Y)
    u_yy = -pi ** 2 * np.sin(pi * X) * np.sin(pi * Y)
    v_xx = -pi ** 2 * np.cos(pi * X) * np.cos(pi * Y)
    v_yy = -pi ** 2 * np.cos(pi * X) * np.cos(pi * Y)
    p_x = pi * np.cos(pi * X) * np.cos(pi * Y)
    p_y = -pi * np.sin(pi * X) * np.sin(pi * Y)

    f1 = - (u_xx + u_yy) + p_x
    f2 = - (v_xx + v_yy) + p_y

    # Flatten for linear system
    f1_flat = f1.ravel()
    f2_flat = f2.ravel()

    # --- 6. Dirichlet BCs: tau method ---
    # For each variable, enforce Dirichlet BCs at boundaries (x=ends, y=ends)
    # We'll solve for [u, v, p] as a block system
    N = Nx * Ny
    # Unknowns: [u | v | p], each of length N
    n_unknowns = 3 * N

    # Build block system:
    # [ -Δ   0   Dx ] [u]   [f1]
    # [  0  -Δ   Dy ] [v] = [f2]
    # [ Dx  Dy   0  ] [p]   [0 ]
    #
    # Where Dx, Dy are discrete derivative matrices

    # Discrete derivative matrices for p_x, p_y, u_x, v_y
    # For p_x: kron(Iy, Dx)
    Dx_big = np.kron(Iy, Dx)
    Dy_big = np.kron(Dy, Ix)

    # Build blocks
    zero = np.zeros((N, N))
    # Laplacian blocks
    Lap_block = Lap.copy()
    # System matrix
    A = np.zeros((3 * N, 3 * N))
    # Fill blocks
    # Row 0: -Δu + p_x = f1
    A[0:N, 0:N] = -Lap_block
    A[0:N, 2 * N:3 * N] = Dx_big
    # Row 1: -Δv + p_y = f2
    A[N:2 * N, N:2 * N] = -Lap_block
    A[N:2 * N, 2 * N:3 * N] = Dy_big
    # Row 2: u_x + v_y = 0 (divergence-free)
    A[2 * N:3 * N, 0:N] = Dx_big
    A[2 * N:3 * N, N:2 * N] = Dy_big
    # No p block in last row

    # RHS
    rhs = np.zeros(3 * N)
    rhs[0:N] = f1_flat
    rhs[N:2 * N] = f2_flat
    # rhs[2N:3N] = 0

    # --- 7. Enforce Dirichlet BCs (tau method) ---
    # For each boundary (x=0,Nx-1, y=0,Ny-1), replace corresponding equations with u/v = BC
    # Find boundary indices
    def boundary_indices(Nx, Ny):
        idxs = []
        for j in range(Ny):
            for i in range(Nx):
                if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:
                    idxs.append(j * Nx + i)
        return np.array(sorted(set(idxs)))

    bidx = boundary_indices(Nx, Ny)
    # For u and v, enforce Dirichlet at boundary indices
    for var in [0, 1]:  # 0: u, 1: v
        for idx in bidx:
            row = var * N + idx
            # Zero out row
            A[row, :] = 0
            # Set diagonal to 1
            A[row, var * N + idx] = 1
            # Set RHS to BC value
            if var == 0:
                rhs[row] = u_exact.ravel()[idx]
            else:
                rhs[row] = v_exact.ravel()[idx]

    # For pressure: fix mean to zero (to make system non-singular)
    # Replace one p unknown equation (pick an interior point, not a boundary point)
    # Find first interior grid point (not in bidx)
    all_idxs = np.arange(N)
    interior_idxs = np.setdiff1d(all_idxs, bidx)
    if len(interior_idxs) == 0:
        # fallback: just pick the first p unknown
        p_fix_idx = 0
    else:
        p_fix_idx = interior_idxs[0]
    p_row = 2 * N + p_fix_idx  # row in the divergence-free block
    # Replace this row with mean(p) = 0
    A[p_row, :] = 0
    A[p_row, 2 * N:3 * N] = 1
    rhs[p_row] = 0

    # --- 8. Solve the linear system ---
    sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    # The solution is ordered as [u, v, p], each of length N = Nx*Ny
    # Reshape to (Ny, Nx) for each variable (row-major order)
    u_num = sol[0:N].reshape((Ny, Nx))
    v_num = sol[N:2 * N].reshape((Ny, Nx))
    p_num = sol[2 * N:3 * N].reshape((Ny, Nx))

    # --- 9. Compute residual (L2 error in u,v) ---
    # Interpolate analytic solution to grid (already done: u_exact, v_exact)
    # Compute L2 norm of error in u and v (not including p)
    # Use Chebyshev quadrature weights for integration
    def cheb_weights(N):
        # Chebyshev-Gauss-Lobatto quadrature weights on [-1,1]
        w = np.ones(N)
        w[0] = w[-1] = 0.5
        for k in range(1, N-1):
            w[k] = 1
        w = w * np.pi / (N - 1)
        return w

    wx = cheb_weights(Nx)
    wy = cheb_weights(Ny)
    # 2D quadrature weights
    W2d = np.outer(wy, wx)
    # Map weights from [-1,1] to [x_min,x_max], [y_min,y_max]
    W2d = W2d * 0.25 * (x_max - x_min) * (y_max - y_min)

    err_u = u_num - u_exact.T  # X, Y meshgrid is (x, y) with 'ij', but u_num is (Ny, Nx)
    err_v = v_num - v_exact.T

    # L2 norm over domain for u and v
    l2_u = np.sqrt(np.sum((err_u.T) ** 2 * W2d))
    l2_v = np.sqrt(np.sum((err_v.T) ** 2 * W2d))
    residual = np.sqrt(l2_u ** 2 + l2_v ** 2)

    # --- 10. Output ---
    # Only final state (steady), so t=None
    return {
        "u": np.stack([u_num, v_num, p_num], axis=0),  # shape (3, Ny, Nx)
        "coords": {"x": x, "y": y},
        "t": None,
        "residual": residual
    }