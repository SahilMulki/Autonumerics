import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # Domain
    x0, x1 = pde_spec["domain"]["bounds"]["x"]
    y0, y1 = pde_spec["domain"]["bounds"]["y"]
    t0, t1 = pde_spec["domain"]["bounds"]["t"]
    # PDE parameters
    c = float(pde_spec["parameters"]["c"])
    # Grid
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])
    dx = (x1 - x0) / (Nx - 1)
    dy = (y1 - y0) / (Ny - 1)
    x = np.linspace(x0, x1, Nx)
    y = np.linspace(y0, y1, Ny)
    coords = {"x": x, "y": y}
    # Time
    dt = float(plan["time_stepping"].get("dt", None))
    t_final = float(plan["time_stepping"].get("t_final", t1))
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil((t_final - t0) / dt))
    else:
        Nt = int(Nt)
        dt = (t_final - t0) / Nt
    t_array = np.linspace(t0, t_final, Nt+1)

    # --- 2. FEM Assembly (Structured Q1 elements, but use FD for speed) ---
    # For high-res, use finite difference mass and stiffness (diagonal mass)
    # Mass matrix (lumped)
    M_diag = np.ones(Nx*Ny) * dx * dy

    # Stiffness matrix (Laplacian, 5-point stencil)
    def laplacian_2d_fd(Nx, Ny, dx, dy):
        N = Nx * Ny
        data = []
        rows = []
        cols = []
        for j in range(Ny):
            for i in range(Nx):
                idx = j*Nx + i
                if i == 0 or i == Nx-1 or j == 0 or j == Ny-1:
                    # Dirichlet BC node
                    rows.append(idx)
                    cols.append(idx)
                    data.append(1.0)
                else:
                    # Center
                    rows.append(idx)
                    cols.append(idx)
                    data.append(-2.0/dx**2 -2.0/dy**2)
                    # Left
                    rows.append(idx)
                    cols.append(idx-1)
                    data.append(1.0/dx**2)
                    # Right
                    rows.append(idx)
                    cols.append(idx+1)
                    data.append(1.0/dx**2)
                    # Down
                    rows.append(idx)
                    cols.append(idx-Nx)
                    data.append(1.0/dy**2)
                    # Up
                    rows.append(idx)
                    cols.append(idx+Nx)
                    data.append(1.0/dy**2)
        from scipy.sparse import coo_matrix
        K = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsc()
        return K

    try:
        from scipy.sparse import csc_matrix, diags
        from scipy.sparse.linalg import factorized
        sparse_ok = True
    except ImportError:
        sparse_ok = False

    if sparse_ok:
        K = laplacian_2d_fd(Nx, Ny, dx, dy)
        M_diag = np.ones(Nx*Ny) * dx * dy
    else:
        # fallback to dense
        K = np.zeros((Nx*Ny, Nx*Ny))
        for j in range(Ny):
            for i in range(Nx):
                idx = j*Nx + i
                if i == 0 or i == Nx-1 or j == 0 or j == Ny-1:
                    K[idx, idx] = 1.0
                else:
                    K[idx, idx] = -2.0/dx**2 -2.0/dy**2
                    K[idx, idx-1] = 1.0/dx**2
                    K[idx, idx+1] = 1.0/dx**2
                    K[idx, idx-Nx] = 1.0/dy**2
                    K[idx, idx+Nx] = 1.0/dy**2

    # --- 3. Initial conditions ---
    X, Y = np.meshgrid(x, y, indexing='ij')
    u0_grid = np.sin(np.pi*X) * np.sin(np.pi*Y)
    v0_grid = np.zeros_like(u0_grid)
    u0 = u0_grid.flatten()
    v0 = v0_grid.flatten()

    # --- 4. Dirichlet BCs ---
    # Find boundary indices
    boundary = np.zeros((Nx, Ny), dtype=bool)
    boundary[0,:] = True
    boundary[-1,:] = True
    boundary[:,0] = True
    boundary[:,-1] = True
    boundary_idx = np.where(boundary.flatten())[0]
    interior_idx = np.setdiff1d(np.arange(Nx*Ny), boundary_idx)

    # --- 5. Crank-Nicolson for 2nd order wave eq ---
    # Use diagonal mass for speed
    # (M/dt^2 + (c^2/2)K) u^{n+1} = 2M/dt^2 u^n - (M/dt^2 - (c^2/2)K) u^{n-1}
    # Only solve for interior

    # Precompute diagonal mass for interior
    M_diag_int = M_diag[interior_idx]

    # Precompute K_int for interior
    if sparse_ok:
        K_int = K[interior_idx, :][:, interior_idx]
        # Precompute A, B, C as sparse matrices
        A = diags(M_diag_int/dt**2) + (c**2/2)*K_int
        B = diags(2*M_diag_int/dt**2)
        C = diags(M_diag_int/dt**2) - (c**2/2)*K_int
        # Pre-factorize A for fast repeated solves
        solve_A = factorized(A)
    else:
        K_int = K[np.ix_(interior_idx, interior_idx)]
        A = np.diag(M_diag_int/dt**2) + (c**2/2)*K_int
        B = np.diag(2*M_diag_int/dt**2)
        C = np.diag(M_diag_int/dt**2) - (c**2/2)*K_int

    # Initial step: U^1 = U^0 + dt*V^0 + 0.5*dt^2*M^{-1}*(-K*U^0)
    u0_int = u0[interior_idx]
    v0_int = v0[interior_idx]
    if sparse_ok:
        KU0 = K_int @ u0_int
        u1_int = u0_int + dt*v0_int + 0.5*dt**2 * (-KU0 / M_diag_int)
    else:
        KU0 = K_int @ u0_int
        u1_int = u0_int + dt*v0_int + 0.5*dt**2 * (-KU0 / M_diag_int)

    # Storage for last two steps
    u_nm1 = u0_int.copy()
    u_n = u1_int.copy()

    # Only store the final solution for memory efficiency
    for n in range(1, Nt):
        # rhs = B @ u_n - C @ u_nm1
        if sparse_ok:
            rhs = B.dot(u_n) - C.dot(u_nm1)
            u_np1 = solve_A(rhs)
        else:
            rhs = B @ u_n - C @ u_nm1
            u_np1 = np.linalg.solve(A, rhs)
        u_nm1, u_n = u_n, u_np1

    # --- 6. Insert Dirichlet BCs for output ---
    u_final_flat = np.zeros(Nx*Ny)
    u_final_flat[boundary_idx] = 0.0
    u_final_flat[interior_idx] = u_n
    u_final = u_final_flat.reshape((Nx, Ny))

    # --- 7. Return ---
    return {
        "u": u_final,
        "coords": coords,
        "t": t_array
    }