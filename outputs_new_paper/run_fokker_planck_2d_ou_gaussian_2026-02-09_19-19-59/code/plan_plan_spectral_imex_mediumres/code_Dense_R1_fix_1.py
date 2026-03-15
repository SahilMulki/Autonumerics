import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters ---
    # PDE parameters
    D = float(pde_spec['parameters']['D'])
    lam = float(pde_spec['parameters']['lambda'])
    L = float(pde_spec['parameters']['L'])
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']

    # Plan parameters
    Nx = int(plan['spatial_discretization']['Nx'])
    Ny = int(plan['spatial_discretization']['Ny'])
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    Nt = plan['time_stepping'].get('Nt', None)
    if dt is None:
        dx = (x_max - x_min) / (Nx-1)
        dy = (y_max - y_min) / (Ny-1)
        dt = 0.4 * min(dx, dy)**2 / (4*D)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    else:
        t_final = Nt * dt
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Chebyshev grid and differentiation matrices ---
    def chebyshev_D(N, a, b):
        x = np.cos(np.pi * np.arange(N) / (N - 1))
        c = np.ones(N)
        c[0] = 2
        c[-1] = 2
        c = c * ((-1) ** np.arange(N))
        X = np.tile(x, (N,1)).T
        dX = X - X.T + np.eye(N)
        D = np.outer(c, 1/c) / (dX)
        D = D - np.diag(np.sum(D, axis=1))
        D = 2.0/(b-a) * D
        x_phys = 0.5*(b-a)*x + 0.5*(b+a)
        return D, x_phys

    Dx, x = chebyshev_D(Nx, x_min, x_max)
    Dy, y = chebyshev_D(Ny, y_min, y_max)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial condition ---
    u = (1/(2*np.pi))*np.exp(-(X**2 + Y**2)/2)

    # --- Dirichlet BC function (analytic solution) ---
    def sigma2_t(t):
        return D/lam + (1 - D/lam)*np.exp(-2*lam*t)
    def bc_func(xv, yv, t):
        s2 = sigma2_t(t)
        return (1/(2*np.pi*s2))*np.exp(-(xv**2 + yv**2)/(2*s2))

    # --- IMEX Runge-Kutta coefficients (ARS(2,3,2) 3rd order) ---
    gamma = 0.4358665215
    ARS_A = np.array([
        [gamma, 0, 0],
        [(-1+2*gamma), gamma, 0],
        [(2-3*gamma), (3*gamma-2), gamma]
    ])
    ARS_B = np.array([
        [gamma, 0, 0],
        [(-1+2*gamma), gamma, 0],
        [(2-3*gamma), (3*gamma-2), gamma]
    ])
    ARS_c = np.array([gamma, -1+2*gamma, 1])
    ARS_bE = np.array([0, 0, 1])
    ARS_bI = np.array([0, 0, 1])

    # --- Precompute for implicit solve ---
    I_x = np.eye(Nx)
    I_y = np.eye(Ny)
    Dx2 = Dx @ Dx
    Dy2 = Dy @ Dy

    # --- Helper: enforce Dirichlet BCs ---
    def enforce_bc(u, t):
        u[0, :] = bc_func(x[0], y, t)
        u[-1, :] = bc_func(x[-1], y, t)
        u[:, 0] = bc_func(x, y[0], t)
        u[:, -1] = bc_func(x, y[-1], t)
        return u

    # --- Helper: explicit advection operator ---
    def adv_op(u):
        # Compute divergence of (lambda*[x,y]*u)
        # div(lambda*[x,y]*u) = lambda*(d/dx(x*u) + d/dy(y*u))
        xu = X * u
        yu = Y * u
        dx_xu = Dx @ xu
        dy_yu = yu @ Dy.T
        return lam * (dx_xu + dy_yu)

    # --- Main time stepping loop ---
    u = enforce_bc(u, t_array[0])
    u_new = np.empty_like(u)

    # Reduce dt for stability (IMEX ARS(2,3,2) is only weakly stable for explicit advection at large dt)
    # Use a more conservative dt if necessary
    dx_min = np.min(np.abs(np.diff(x)))
    dy_min = np.min(np.abs(np.diff(y)))
    dt_cfl = 0.2 * min(dx_min, dy_min) / (np.abs(lam)*L + 1e-12)
    dt_diff = 0.2 * min(dx_min, dy_min)**2 / (4*D)
    dt_safe = min(dt, dt_cfl, dt_diff)
    if dt_safe < dt:
        Nt = int(np.ceil(t_final / dt_safe))
        t_array = np.linspace(0, t_final, Nt+1)
        dt = dt_safe

    for n in range(Nt):
        t_n = t_array[n]
        u_stage = u.copy()
        for s in range(3):
            # Explicit advection
            adv = adv_op(u_stage)
            rhs = u.copy()
            for j in range(s):
                rhs += dt * ARS_A[s, j] * adv_op(u_stage)
            rhs += dt * ARS_B[s, s] * adv
            # Implicit diffusion via ADI
            A_x = I_x - dt * ARS_B[s, s] * D * Dx2
            tmp = np.linalg.solve(A_x, rhs)
            A_y = I_y - dt * ARS_B[s, s] * D * Dy2
            tmp = np.linalg.solve(A_y, tmp.T).T
            u_stage = tmp
            u_stage = enforce_bc(u_stage, t_n + ARS_c[s]*dt)
        u_new[...] = u_stage
        u_new = enforce_bc(u_new, t_array[n+1])
        u, u_new = u_new, u  # swap

    u_final = u

    # --- Compute residual at final time ---
    # Use backward difference for u_t
    if Nt > 0:
        # Recompute u_prev at t = t_final - dt
        u_prev = (1/(2*np.pi))*np.exp(-(X**2 + Y**2)/2)
        u_prev = enforce_bc(u_prev, t_array[0])
        u_tmp = np.empty_like(u_prev)
        for n in range(Nt-1):
            t_n = t_array[n]
            u_stage = u_prev.copy()
            for s in range(3):
                adv = adv_op(u_stage)
                rhs = u_prev.copy()
                for j in range(s):
                    rhs += dt * ARS_A[s, j] * adv_op(u_stage)
                rhs += dt * ARS_B[s, s] * adv
                A_x = I_x - dt * ARS_B[s, s] * D * Dx2
                tmp = np.linalg.solve(A_x, rhs)
                A_y = I_y - dt * ARS_B[s, s] * D * Dy2
                tmp = np.linalg.solve(A_y, tmp.T).T
                u_stage = tmp
                u_stage = enforce_bc(u_stage, t_n + ARS_c[s]*dt)
            u_tmp[...] = u_stage
            u_tmp = enforce_bc(u_tmp, t_array[n+1])
            u_prev, u_tmp = u_tmp, u_prev
    else:
        u_prev = u_final

    u_t = (u_final - u_prev) / dt

    u_xx = Dx2 @ u_final
    u_yy = u_final @ Dy2.T
    lap = u_xx + u_yy

    xu = X * u_final
    yu = Y * u_final
    dx_xu = Dx @ xu
    dy_yu = yu @ Dy.T
    adv = lam * (dx_xu + dy_yu)

    residual_grid = u_t - D * lap - adv

    # Compute L2 norm of residual (over grid, using Chebyshev quadrature weights)
    # Chebyshev-Gauss-Lobatto weights
    def cheb_weights(N, a, b):
        w = np.ones(N)
        w[0] = w[-1] = 0.5
        w = w * np.pi / (N - 1)
        return w * (b - a) / 2

    wx = cheb_weights(Nx, x_min, x_max)
    wy = cheb_weights(Ny, y_min, y_max)
    W = np.outer(wx, wy)
    residual_L2 = np.sqrt(np.sum((residual_grid**2) * W))

    coords = {'x': x, 'y': y}
    return {
        "u": u_final,
        "coords": coords,
        "t": t_array,
        "residual": residual_L2
    }