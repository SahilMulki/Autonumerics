import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # PDE parameters
    D = float(pde_spec["parameters"]["D"])
    lam = float(pde_spec["parameters"]["lambda"])
    L = float(pde_spec["parameters"]["L"])
    x_min, x_max = float(pde_spec["domain"]["x_min"]), float(pde_spec["domain"]["x_max"])
    y_min, y_max = float(pde_spec["domain"]["y_min"]), float(pde_spec["domain"]["y_max"])
    # Plan parameters
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])
    dt = float(plan["time_stepping"].get("dt", 0.005))
    t_final = float(plan["time_stepping"].get("t_final", 1.0))
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust dt to land exactly on t_final

    # --- 2. Chebyshev grid and differentiation matrices ---
    def cheb(N, a, b):
        # Chebyshev points on [a, b]
        k = np.arange(N)
        x = np.cos(np.pi * k / (N - 1))
        x = 0.5 * (b - a) * x + 0.5 * (b + a)
        return x

    # Chebyshev differentiation matrix (1D)
    def cheb_D(N, a, b):
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

    # Grids and differentiation matrices
    x = cheb(Nx, x_min, x_max)
    y = cheb(Ny, y_min, y_max)
    Dx = cheb_D(Nx, x_min, x_max)
    Dy = cheb_D(Ny, y_min, y_max)
    X, Y = np.meshgrid(x, y, indexing='ij')  # shape (Nx, Ny)

    # --- 3. Initial condition ---
    # IC: (1/(2*np.pi))*np.exp(-(x**2 + y**2)/2)
    u = (1/(2*np.pi)) * np.exp(-(X**2 + Y**2)/2)

    # --- 4. Dirichlet boundary condition function ---
    def sigma2_t(t):
        return D/lam + (1 - D/lam) * np.exp(-2*lam*t)
    def bc_func(xv, yv, t):
        s2 = sigma2_t(t)
        return (1/(2*np.pi*s2)) * np.exp(-(xv**2 + yv**2)/(2*s2))

    # --- 5. Helper: Apply Dirichlet BCs (overwrite boundaries) ---
    def apply_bc(u, t):
        # Overwrite all boundaries with analytic BC
        # x boundaries
        u[0, :] = bc_func(x[0], y, t)
        u[-1, :] = bc_func(x[-1], y, t)
        # y boundaries
        u[:, 0] = bc_func(x, y[0], t)
        u[:, -1] = bc_func(x, y[-1], t)
        return u

    # --- 6. Operators for IMEX ---
    # Laplacian: D*(u_xx + u_yy)
    # Use Kronecker structure for implicit solve
    Ix = np.eye(Nx)
    Iy = np.eye(Ny)
    D2x = Dx @ Dx
    D2y = Dy @ Dy

    # For implicit solve: (I - dt*D*L) u^{n+1} = rhs
    # L = D2x kron Iy + Ix kron D2y
    # We'll use the fact that for Dirichlet BCs, we can mask out boundaries and solve only for interior points

    # Indices for interior points
    ix = np.arange(1, Nx-1)
    iy = np.arange(1, Ny-1)
    N_in = (Nx-2)*(Ny-2)

    # Precompute implicit operator for interior
    D2x_in = D2x[np.ix_(ix, ix)]
    D2y_in = D2y[np.ix_(iy, iy)]
    Ix_in = np.eye(Nx-2)
    Iy_in = np.eye(Ny-2)
    # Laplacian on interior
    L_in = np.kron(Iy_in, D2x_in) + np.kron(D2y_in, Ix_in)
    # A = np.eye(N_in) - dt * D * L_in  # (N_in, N_in)  # Not used directly, see below

    # --- 7. IMEX Runge-Kutta coefficients (ARK3(2)4L[2]SA) ---
    # For simplicity, use ARK3(2)4L[2]SA coefficients (see Kennedy & Carpenter 2003)
    # Explicit tableau (A_ex, b_ex, c_ex), Implicit tableau (A_im, b_im, c_im)
    # Only 3 stages needed for order 3
    gamma = 0.4358665215
    bE = np.array([0.140737774726167, 0.367893561216354, 0.491368664057479])
    bI = np.array([0.140737774726167, 0.367893561216354, 0.491368664057479])
    c = np.array([0, gamma, 1.0])
    AE = np.array([
        [0, 0, 0],
        [gamma, 0, 0],
        [0.552929148035939, 0.447070851964061, 0]
    ])
    AI = np.array([
        [gamma, 0, 0],
        [0.0, gamma, 0],
        [0.243100786275166, 0.206899213724834, gamma]
    ])
    s = 3

    # --- 8. Time stepping ---
    t = 0.0
    t_array = np.linspace(0, t_final, Nt+1)
    u0 = u.copy()  # for residual at the end
    for n in range(Nt):
        u0 = u.copy()
        t_n = t
        dt_n = min(dt, t_final - t_n)
        # Stage arrays
        U = [None]*s
        F = [None]*s
        G = [None]*s
        # For each stage
        for i in range(s):
            # Compute stage time
            t_stage = t_n + c[i]*dt_n
            # Stage value
            u_stage = u0.copy()
            # Explicit sum
            for j in range(i):
                if AE[i, j] != 0:
                    u_stage += dt_n * AE[i, j] * F[j]
            # Implicit sum
            rhs = u_stage.copy()
            for j in range(i):
                if AI[i, j] != 0:
                    rhs += dt_n * AI[i, j] * G[j]
            # Apply BCs to rhs
            rhs = apply_bc(rhs, t_stage)
            # Extract interior
            u_in = rhs[1:-1, 1:-1].reshape(-1)
            # Implicit solve: (I - dt*gamma*D*L) U = rhs
            A_stage = np.eye(N_in) - dt_n * gamma * D * L_in
            u_in_new = np.linalg.solve(A_stage, u_in)
            # Place back into array
            u_stage_new = u_stage.copy()
            u_stage_new[1:-1, 1:-1] = u_in_new.reshape((Nx-2, Ny-2))
            # Apply BCs
            u_stage_new = apply_bc(u_stage_new, t_stage)
            U[i] = u_stage_new
            # Explicit (advection) part: F = -div(lambda*[x,y]*u)
            # Compute divergence: -lam*(d/dx(x*u) + d/dy(y*u))
            xU = X * U[i]
            yU = Y * U[i]
            dxd = Dx @ xU
            dyd = yU @ Dy.T
            div = (dxd + dyd)
            F[i] = -lam * div
            # Implicit (diffusion) part: G = D*(u_xx + u_yy)
            uxx = Dx @ (Dx @ U[i])
            uyy = (U[i] @ Dy) @ Dy
            G[i] = D * (uxx + uyy)
        # Combine stages for next step
        u_new = u0.copy()
        for i in range(s):
            u_new += dt_n * (bE[i]*F[i] + bI[i]*G[i])
        # Apply BCs at t_{n+1}
        t = t_n + dt_n
        u = apply_bc(u_new, t)
    # Final time
    u_final = u

    # --- 9. Compute residual grid at final time ---
    # PDE: rho_t = D*(rho_xx + rho_yy) + div(lam*[x,y]*rho)
    # Approximate rho_t by backward difference
    # For residual, need u_t, u_xx, u_yy, div(lam*[x,y]*u)
    # Compute u_t ~ (u_final - u0)/dt
    u_t = (u_final - u0) / dt
    # u_xx, u_yy
    u_xx = Dx @ (Dx @ u_final)
    u_yy = (u_final @ Dy) @ Dy
    # div(lam*[x,y]*u)
    xU = X * u_final
    yU = Y * u_final
    dxd = Dx @ xU
    dyd = yU @ Dy.T
    div = (dxd + dyd)
    adv = -lam * div
    # Residual: u_t - [D*(u_xx + u_yy) + adv]
    residual_grid = u_t - (D*(u_xx + u_yy) + adv)

    # --- 10. Output ---
    coords = {"x": x, "y": y}
    return {
        "u": u_final,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }