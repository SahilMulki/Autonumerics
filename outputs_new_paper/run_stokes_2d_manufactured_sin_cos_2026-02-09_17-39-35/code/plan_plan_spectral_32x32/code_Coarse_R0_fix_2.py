import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    """
    Solve the steady 2D Stokes equations with manufactured solution using a spectral (Chebyshev) method
    and tau method for Dirichlet BCs. Return only the final solution.
    """

    # --- 1. Extract grid and domain info ---
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    Lx = x_max - x_min
    Ly = y_max - y_min

    # Use Chebyshev grid for non-periodic Dirichlet BCs (tau method)
    def chebyshev_nodes(N, a, b):
        k = np.arange(N)
        x = np.cos(np.pi * k / (N - 1))
        # Map from [-1,1] to [a,b]
        return 0.5 * (a + b) + 0.5 * (b - a) * x[::-1]

    x = chebyshev_nodes(Nx, x_min, x_max)
    y = chebyshev_nodes(Ny, y_min, y_max)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- 2. Analytic solution and source terms ---
    pi = np.pi
    u_exact = np.sin(pi * X) * np.sin(pi * Y)
    v_exact = np.cos(pi * X) * np.cos(pi * Y)
    p_exact = np.sin(pi * X) * np.cos(pi * Y)

    # Compute f1, f2 from manufactured solution
    # -Δu + p_x = f1
    # -Δv + p_y = f2

    # Laplacians
    u_xx = -pi**2 * np.sin(pi * X) * np.sin(pi * Y)
    u_yy = -pi**2 * np.sin(pi * X) * np.sin(pi * Y)
    v_xx = -pi**2 * np.cos(pi * X) * np.cos(pi * Y)
    v_yy = -pi**2 * np.cos(pi * X) * np.cos(pi * Y)

    # Pressure derivatives
    p_x = pi * np.cos(pi * X) * np.cos(pi * Y)
    p_y = -pi * np.sin(pi * X) * np.sin(pi * Y)

    f1 = - (u_xx + u_yy) + p_x
    f2 = - (v_xx + v_yy) + p_y

    # --- 3. Construct Chebyshev differentiation matrices ---
    # (We use the spectral Chebyshev collocation method)
    def cheb_D(N, a, b):
        # Chebyshev differentiation matrix on [a,b]
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
        # Map from [-1,1] to [a,b]
        D = 2.0 / (b - a) * D
        return D

    Dx = cheb_D(Nx, x_min, x_max)
    Dy = cheb_D(Ny, y_min, y_max)

    I_x = np.eye(Nx)
    I_y = np.eye(Ny)

    # 2D Laplacian operator (Kronecker sum)
    # Δu ≈ (D2x ⊗ Iy + Ix ⊗ D2y) u_flat
    D2x = Dx @ Dx
    D2y = Dy @ Dy

    # --- 4. Build the block system for (u, v, p) ---
    # Unknowns: u (Nx*Ny), v (Nx*Ny), p (Nx*Ny)
    N = Nx * Ny
    # Flattening: u[i,j] -> u[i*Ny + j]

    # Helper for Kronecker products
    def kron2(A, B):
        return np.kron(A, B)

    # Laplacian blocks
    L = kron2(D2x, I_y) + kron2(I_x, D2y)  # shape (N, N)

    # Pressure gradient blocks
    Dx_big = kron2(Dx, I_y)  # d/dx
    Dy_big = kron2(I_x, Dy)  # d/dy

    # Divergence blocks
    Div_x = Dx_big.copy()
    Div_y = Dy_big.copy()

    # Zero blocks
    Z = np.zeros((N, N))

    # --- 5. Tau method: enforce Dirichlet BCs by replacing rows in the system ---
    # Identify boundary indices
    boundary_mask = np.zeros((Nx, Ny), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    boundary_idx = np.flatnonzero(boundary_mask.ravel())
    interior_mask = ~boundary_mask
    interior_idx = np.flatnonzero(interior_mask.ravel())

    # For u and v: Dirichlet BCs at boundary points
    # For p: no BCs (pressure determined up to constant, fix p at one point)

    # --- 6. Assemble the full system ---
    # System: [A]{U} = {F}
    # [ -Δ   0   d/dx ] [u]   [f1]
    # [  0  -Δ   d/dy ] [v] = [f2]
    # [d/dx d/dy   0  ] [p]   [0 ]
    #
    # U = [u_flat, v_flat, p_flat]

    # Block matrices
    # Row 1: u equation
    A11 = L.copy()
    A12 = Z.copy()
    A13 = Dx_big.copy()
    # Row 2: v equation
    A21 = Z.copy()
    A22 = L.copy()
    A23 = Dy_big.copy()
    # Row 3: continuity
    A31 = Dx_big.copy()
    A32 = Dy_big.copy()
    A33 = Z.copy()

    # Stack blocks
    top = np.hstack([A11, A12, A13])
    mid = np.hstack([A21, A22, A23])
    bot = np.hstack([A31, A32, A33])
    A = np.vstack([top, mid, bot])  # shape (3N, 3N)

    # RHS
    F = np.concatenate([f1.ravel(), f2.ravel(), np.zeros(N)])

    # --- 7. Tau method: enforce Dirichlet BCs for u, v ---
    # For each boundary point, replace corresponding row in A and F
    # For u: rows 0..N-1, for v: rows N..2N-1
    for idx in boundary_idx:
        # u Dirichlet
        A[idx, :] = 0
        A[idx, idx] = 1
        F[idx] = u_exact.ravel()[idx]
        # v Dirichlet
        A[N + idx, :] = 0
        A[N + idx, N + idx] = 1
        F[N + idx] = v_exact.ravel()[idx]

    # For pressure: fix p at one interior point (not on the boundary)
    # This avoids singularity due to over-constraining p at a boundary (where u,v are already set)
    # If all points are boundary (very coarse grid), just pick the first available
    if interior_idx.size > 0:
        p0_idx = 2 * N + interior_idx[0]
        F[p0_idx] = p_exact.ravel()[interior_idx[0]]
    else:
        p0_idx = 2 * N  # fallback
        F[p0_idx] = p_exact.ravel()[0]
    A[p0_idx, :] = 0
    A[p0_idx, p0_idx] = 1

    # --- 8. Solve the linear system ---
    U = np.linalg.solve(A, F)
    u_num = U[:N].reshape((Nx, Ny))
    v_num = U[N:2*N].reshape((Nx, Ny))
    p_num = U[2*N:].reshape((Nx, Ny))

    # --- 9. Return final state ---
    # For steady problem, t is None
    coords = {"x": x, "y": y}
    u = np.stack([u_num, v_num], axis=0)  # shape (2, Nx, Ny)
    t_array = None

    return {
        "u": u,
        "coords": coords,
        "t": t_array
    }