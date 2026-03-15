import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # PDE parameters
    D = float(pde_spec['parameters']['D'])
    r = float(pde_spec['parameters']['r'])
    # Domain
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    # Initial condition: sin(pi x) sin(pi y)
    # BCs: Dirichlet u=0

    # --- 2. Plan parameters ---
    Nx = int(plan['spatial_discretization']['Nx'])
    Ny = int(plan['spatial_discretization']['Ny'])
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    Nt = plan['time_stepping'].get('Nt', None)
    if t_final is not None and dt is not None:
        Nt = int(np.ceil(t_final / dt))
    elif Nt is not None and dt is None and t_final is not None:
        dt = t_final / Nt
    elif Nt is not None and dt is not None:
        t_final = Nt * dt
    else:
        raise ValueError("Insufficient time stepping info in plan.")

    # --- 3. Chebyshev grid (Dirichlet 0 at boundaries) ---
    def cheb_points(N):
        k = np.arange(N)
        return np.cos(np.pi * k / (N - 1))
    x_cheb = cheb_points(Nx)
    y_cheb = cheb_points(Ny)
    x = 0.5 * (x_cheb + 1) * (x_max - x_min) + x_min
    y = 0.5 * (y_cheb + 1) * (y_max - y_min) + y_min

    # --- 4. Chebyshev differentiation matrix (1D) ---
    def cheb_D(N, a, b):
        if N == 1:
            return np.zeros((1, 1)), np.array([0.5 * (a + b)])
        x = np.cos(np.pi * np.arange(N) / (N - 1))
        c = np.ones(N)
        c[0] = 2
        c[-1] = 2
        c = c * ((-1) ** np.arange(N))
        X = np.tile(x, (N, 1))
        dX = X - X.T + np.eye(N)
        D = (np.outer(c, 1 / c)) / (dX)
        D = D - np.diag(np.sum(D, axis=1))
        D = 2.0 / (b - a) * D
        x_phys = 0.5 * (x + 1) * (b - a) + a
        return D, x_phys

    Dx, x_phys = cheb_D(Nx, x_min, x_max)
    Dy, y_phys = cheb_D(Ny, y_min, y_max)

    # --- 5. Initial condition ---
    X, Y = np.meshgrid(x_phys, y_phys, indexing='ij')
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)

    # --- 6. Dirichlet BCs: enforce u=0 at boundaries (tau method) ---
    # For Chebyshev, Dirichlet BCs are at first and last grid points in each direction.

    # --- 7. IMEX Runge-Kutta coefficients (ARK3(2)4L[2]SA, 3rd order) ---
    # Explicit (reaction): A_e, b_e; Implicit (diffusion): A_i, b_i
    # Coefficients for ARK3(2)4L[2]SA (simplified for 3 stages)
    A_e = np.array([
        [0,   0,   0],
        [0.5, 0,   0],
        [-1,  2,   0]
    ])
    b_e = np.array([1/6, 1/6, 2/3])
    c_e = np.array([0, 0.5, 1])

    A_i = np.array([
        [0.5, 0,   0],
        [0,   0.5, 0],
        [0,   0,   1]
    ])
    b_i = np.array([1/6, 1/6, 2/3])
    c_i = np.array([0.5, 0.5, 1])

    s = 3  # number of stages

    # --- 8. Precompute Laplacian operator (for implicit solve) ---
    def interior_idx(N):
        return slice(1, N-1)
    ix = interior_idx(Nx)
    iy = interior_idx(Ny)
    Nxi = Nx - 2
    Nyi = Ny - 2

    # 1D second derivative matrices (interior only)
    D2x = np.dot(Dx, Dx)
    D2y = np.dot(Dy, Dy)
    D2x_int = D2x[ix, ix]
    D2y_int = D2y[iy, iy]

    # 2D Laplacian with Dirichlet BCs: kron(I, D2x) + kron(D2y, I)
    Ix_ = np.eye(Nxi)
    Iy_ = np.eye(Nyi)
    Lx = np.kron(Iy_, D2x_int)
    Ly = np.kron(D2y_int, Ix_)
    Lap = Lx + Ly  # shape: (Nxi*Nyi, Nxi*Nyi)

    # --- 9. Flattened unknowns: u_int = u[ix,iy].flatten('C') ---
    def u_to_int(u):
        return u[ix, :][:, iy].reshape(-1)
    def int_to_u(u_int):
        u = np.zeros((Nx, Ny))
        u[ix, :][:, iy] = u_int.reshape(Nxi, Nyi)
        return u

    # --- 10. Time stepping ---
    u = u0.copy()
    u_int = u_to_int(u)
    t = 0.0
    t_array = np.array([0.0, t_final])  # Only initial and final times

    I_int = np.eye(Nxi * Nyi)

    for n in range(Nt):
        t_n = t
        u_stages = np.zeros((s, Nxi*Nyi))
        for i in range(s):
            # Explicit (reaction) part
            expl = np.zeros(Nxi*Nyi)
            for j in range(i):
                expl += A_e[i, j] * r * u_stages[j]
            expl = u_int + dt * expl

            # Implicit (diffusion) part
            a_ii = A_i[i, i]
            if a_ii == 0:
                u_new = expl
            else:
                rhs = expl.copy()
                for j in range(i):
                    rhs += dt * A_i[i, j] * D * (Lap @ u_stages[j])
                M = I_int - dt * a_ii * D * Lap
                u_new = np.linalg.solve(M, rhs)
            u_stages[i] = u_new
        # Combine stages for next step
        incr = np.zeros(Nxi*Nyi)
        for i in range(s):
            incr += b_e[i] * r * u_stages[i]
        incr = u_int + dt * incr
        rhs = incr.copy()
        for i in range(s):
            rhs += dt * b_i[i] * D * (Lap @ u_stages[i])
        u_int = rhs  # No further implicit solve needed

        t += dt

    # --- 11. Reconstruct full grid with BCs ---
    u_final = int_to_u(u_int)

    # --- 12. Compute residual grid ---
    # Residual: R = u_t - D*(u_xx + u_yy) - r*u
    # Approximate u_t by (u_final - u0)/t_final
    u_t_approx = (u_final - u0) / t_final

    # Compute Laplacian at all points (including boundaries)
    # For Chebyshev, Laplacian is Dx^2 in x and Dy^2 in y, but need to apply to both axes
    # u_xx = Dx @ u @ I
    # u_yy = I @ u @ Dy^T
    u_xx = Dx @ u_final
    u_xx = Dx @ u_xx  # (Nx, Ny)
    u_yy = u_final @ Dy.T
    u_yy = u_yy @ Dy.T  # (Nx, Ny)
    lap_u = u_xx + u_yy

    residual_grid = u_t_approx - D * lap_u - r * u_final

    # --- 13. Compute L2 residual norm ---
    # Use Chebyshev quadrature weights for integration
    def cheb_weights(N):
        w = np.ones(N)
        w[0] = w[-1] = 0.5
        for k in range(1, N-1):
            w[k] = 1
        w = w * np.pi / (N - 1)
        return w

    wx = cheb_weights(Nx)
    wy = cheb_weights(Ny)
    # 2D quadrature weights
    W = np.outer(wx, wy)
    # L2 norm: sqrt( sum |residual|^2 * W )
    residual_L2 = np.sqrt(np.sum((residual_grid**2) * W))

    # --- 14. Output ---
    result = {
        "u": u_final.copy(),
        "coords": {"x": x_phys.copy(), "y": y_phys.copy()},
        "t": t_array.copy(),
        "residual": residual_L2
    }
    return result