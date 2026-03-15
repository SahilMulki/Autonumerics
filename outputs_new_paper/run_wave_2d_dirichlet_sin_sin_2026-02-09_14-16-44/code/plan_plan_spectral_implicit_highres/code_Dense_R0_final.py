import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and Plan Parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    t_min, t_max = pde_spec["domain"]["bounds"]["t"]
    # Parameters
    c = float(pde_spec["parameters"]["c"])
    # Grid sizes
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])
    # Time stepping
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        dx = (x_max - x_min) / (Nx - 1)
        dy = (y_max - y_min) / (Ny - 1)
        dt = 0.8 * min(dx, dy) / (np.sqrt(2)*c)
    t_final = plan["time_stepping"].get("t_final", t_max)
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil((t_final - t_min) / dt))
    else:
        dt = (t_final - t_min) / Nt

    # Chebyshev grid (Dirichlet: use all points, BCs enforced by zeroing boundaries)
    def cheb_points(N):
        k = np.arange(N)
        return np.cos(np.pi * k / (N - 1))
    def map_to_domain(xi, a, b):
        return 0.5*(b-a)*xi + 0.5*(b+a)
    x_cheb = cheb_points(Nx)
    y_cheb = cheb_points(Ny)
    x = map_to_domain(x_cheb, x_min, x_max)
    y = map_to_domain(y_cheb, y_min, y_max)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Chebyshev Differentiation Matrices ---
    def cheb_D(N):
        if N == 1:
            return np.zeros((1,1))
        x = np.cos(np.pi * np.arange(N) / (N-1))
        c = np.ones(N)
        c[0] = 2
        c[-1] = 2
        c = c * ((-1) ** np.arange(N))
        Xmat = np.tile(x, (N,1)).T
        dX = Xmat - Xmat.T + np.eye(N)
        D = np.outer(c, 1/c) / (dX)
        D = D - np.diag(np.sum(D, axis=1))
        return D
    Dx = cheb_D(Nx)
    Dy = cheb_D(Ny)
    D2x = Dx @ Dx
    D2y = Dy @ Dy

    # --- Initial Conditions ---
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
    v0 = np.zeros_like(u0)

    # --- Dirichlet BCs: enforce u=0 at boundaries at all times ---
    def enforce_bc(U):
        U[0,:] = 0
        U[-1,:] = 0
        U[:,0] = 0
        U[:,-1] = 0
        return U

    u0 = enforce_bc(u0)
    v0 = enforce_bc(v0)

    # --- Precompute Laplacian operator as a function ---
    def laplacian(U):
        # U: (Nx, Ny)
        # Lap U = D2x @ U + U @ D2y^T
        return D2x @ U + U @ D2y.T

    # --- Precompute eigendecomposition for Laplacian (for fast implicit solves) ---
    Nx_in = Nx - 2
    Ny_in = Ny - 2

    D2x_in = D2x[1:-1,1:-1]
    D2y_in = D2y[1:-1,1:-1]

    # Eigendecomposition
    Lx, Vx = np.linalg.eigh(D2x_in)
    Ly, Vy = np.linalg.eigh(D2y_in)

    # --- Initial Conditions (interior only) ---
    u_nm1 = u0.copy()
    v_nm1 = v0.copy()
    u_n = u0.copy()
    v_n = v0.copy()

    # --- Helper: solve (A - c^2*Laplacian)U = RHS for U (interior only) ---
    def solve_helmholtz(rhs, alpha_coef):
        # rhs: (Nx,Ny) with BCs already enforced (so boundaries are zero)
        # Only solve for interior
        rhs_in = rhs[1:-1,1:-1]
        # Transform to eigenbasis
        rhs_hat = Vx.T @ rhs_in @ Vy
        # Solve in spectral space
        denom = alpha_coef - c**2 * (Lx[:,None] + Ly[None,:])
        u_hat = rhs_hat / denom
        # Transform back
        u_in = Vx @ u_hat @ Vy.T
        # Insert into full grid
        u_full = np.zeros_like(rhs)
        u_full[1:-1,1:-1] = u_in
        return u_full

    # --- Time stepping ---
    t = t_min
    for n in range(Nt):
        t = t_min + (n+1)*dt
        if n == 0:
            # BDF1 step (Backward Euler)
            # (u^{n+1} - 2u^n + u^{n-1})/dt^2 = c^2 Lap u^{n+1}
            # For first step, use Taylor expansion for u^{-1}:
            # u^{-1} = u^0 - dt*v^0
            u_nm1 = u0 - dt*v0
            # (u^{n+1} - 2u^n + u^{n-1})/dt^2 = c^2 Lap u^{n+1}
            # => (1/dt^2) u^{n+1} - c^2 Lap u^{n+1} = (2u^n - u^{n-1})/dt^2
            rhs = (2*u_n - u_nm1) / dt**2
            alpha_coef = 1.0/dt**2
            u_np1 = solve_helmholtz(rhs, alpha_coef)
            u_np1 = enforce_bc(u_np1)
            v_np1 = (u_np1 - u_n)/dt
            v_np1 = enforce_bc(v_np1)
        else:
            # BDF2 step
            # (3u^{n+1} - 4u^n + u^{n-1})/(2dt^2) = c^2 Lap u^{n+1}
            # => (3/(2dt^2)) u^{n+1} - c^2 Lap u^{n+1} = (4u^n - u^{n-1})/(2dt^2)
            rhs = (4*u_n - u_nm1) / (2*dt**2)
            alpha_coef = 3.0/(2*dt**2)
            u_np1 = solve_helmholtz(rhs, alpha_coef)
            u_np1 = enforce_bc(u_np1)
            v_np1 = (3*u_np1 - 4*u_n + u_nm1)/(2*dt)
            v_np1 = enforce_bc(v_np1)
        u_nm1, u_n = u_n, u_np1
        v_nm1, v_n = v_n, v_np1

    u = u_n.copy()
    coords = {"x": x, "y": y}
    t_array = np.array([t_min + Nt*dt])

    # --- Residual calculation (L2 error of PDE at final time) ---
    # PDE: u_tt = c^2 (u_xx + u_yy)
    # Approximate u_tt at final time using BDF2
    # u_tt ≈ (u^{n+1} - 2u^n + u^{n-1}) / dt^2
    # Laplacian at final time: laplacian(u)
    # Use u_n (final), u_nm1 (previous), and u_nm2 (second previous)
    # To get u_nm2, we need to store it during stepping

    # To get u_tt at final time, we need u_n (current), u_nm1 (previous), u_nm2 (second previous)
    # We'll rerun the last two steps to get u_nm2

    # Rewind two steps to get u_nm2, u_nm1, u_n
    # We'll do this by rerunning the time stepping and storing the last three
    u_nm2 = u0.copy()
    u_nm1 = u0.copy()
    u_n = u0.copy()
    v_nm1 = v0.copy()
    v_n = v0.copy()
    for n in range(Nt):
        if n == 0:
            u_nm1 = u0 - dt*v0
            rhs = (2*u_n - u_nm1) / dt**2
            alpha_coef = 1.0/dt**2
            u_np1 = solve_helmholtz(rhs, alpha_coef)
            u_np1 = enforce_bc(u_np1)
        else:
            rhs = (4*u_n - u_nm1) / (2*dt**2)
            alpha_coef = 3.0/(2*dt**2)
            u_np1 = solve_helmholtz(rhs, alpha_coef)
            u_np1 = enforce_bc(u_np1)
        u_nm2, u_nm1, u_n = u_nm1, u_n, u_np1

    # Now u_nm2 = u^{n-1}, u_nm1 = u^n, u_n = u^{n+1} (final)
    # Compute u_tt at final time
    u_tt = (u_n - 2*u_nm1 + u_nm2) / dt**2
    lap_u = laplacian(u_n)
    residual_grid = u_tt - c**2 * lap_u
    # Mask out boundaries (where Dirichlet BCs are enforced)
    mask = np.ones_like(u_n, dtype=bool)
    mask[0,:] = False
    mask[-1,:] = False
    mask[:,0] = False
    mask[:,-1] = False
    residual = np.sqrt(np.sum(residual_grid[mask]**2) / np.sum(mask))

    return {
        "u": u,
        "coords": coords,
        "t": t_array,
        "residual": residual
    }