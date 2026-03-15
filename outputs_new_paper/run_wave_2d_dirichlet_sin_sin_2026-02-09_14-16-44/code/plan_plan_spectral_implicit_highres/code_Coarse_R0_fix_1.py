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

    # --- BDF2 coefficients ---
    alpha = 3.0/(2*dt)

    # --- Storage for time stepping ---
    u_nm1 = u0.copy()
    v_nm1 = v0.copy()
    u_n = u0.copy()
    v_n = v0.copy()

    # --- Time stepping ---
    for n in range(Nt):
        t = t_min + (n+1)*dt
        if n == 0:
            # BDF1 step
            RHS = (u_n/dt**2) + (v_n/dt)
            U_flat = u_n.flatten()
            RHS_flat = RHS.flatten()
            def apply_A(Uvec):
                Umat = Uvec.reshape((Nx,Ny))
                return (Umat/dt**2 - c**2/dt * laplacian(Umat)).flatten()
            U1 = U_flat.copy()
            for _ in range(10):
                AU = apply_A(U1)
                res = RHS_flat - AU
                U1 += 0.8*res
            u_np1 = U1.reshape((Nx,Ny))
            v_np1 = (u_np1 - u_n)/dt
        else:
            RHS1 = (4*u_n - u_nm1)/(2*dt)
            RHS2 = (4*v_n - v_nm1)/(2*dt)
            RHS = alpha*RHS1 + RHS2
            RHS_flat = RHS.flatten()
            def apply_A(Uvec):
                Umat = Uvec.reshape((Nx,Ny))
                return (alpha**2 * Umat - c**2 * laplacian(Umat)).flatten()
            U_guess = u_n.flatten()
            U_np1 = U_guess.copy()
            for _ in range(10):
                AU = apply_A(U_np1)
                res = RHS_flat - AU
                U_np1 += 0.8*res
            u_np1 = U_np1.reshape((Nx,Ny))
            v_np1 = (3*u_np1 - 4*u_n + u_nm1)/(2*dt)
        u_np1 = enforce_bc(u_np1)
        v_np1 = enforce_bc(v_np1)
        u_nm1, u_n = u_n, u_np1
        v_nm1, v_n = v_n, v_np1

    u = u_n.copy()
    coords = {"x": x, "y": y}
    t_array = np.array([t_min + Nt*dt])

    # --- Compute residual grid ---
    # u_tt ≈ (v_n - v_nm1)/dt
    u_tt = (v_n - v_nm1)/dt
    lap_u = laplacian(u)
    residual = u_tt - c**2 * lap_u
    residual = enforce_bc(residual)

    return {
        "u": u,
        "coords": coords,
        "t": t_array
    }