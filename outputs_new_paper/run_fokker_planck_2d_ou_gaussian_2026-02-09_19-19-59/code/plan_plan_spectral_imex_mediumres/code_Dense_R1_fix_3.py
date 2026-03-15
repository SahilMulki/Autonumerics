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
    if t_final is None and Nt is not None:
        t_final = Nt * dt
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    else:
        t_final = Nt * dt
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Chebyshev grid and differentiation matrices ---
    def chebyshev_D(N, a, b):
        if N == 1:
            x = np.array([1.0])
            D = np.zeros((1, 1))
            x_phys = 0.5*(b-a)*x + 0.5*(b+a)
            return D, x_phys
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
    dx_min = np.min(np.abs(np.diff(x)))
    dy_min = np.min(np.abs(np.diff(y)))
    dt_cfl = 0.2 * min(dx_min, dy_min) / (np.abs(lam)*L + 1e-12)
    dt_diff = 0.2 * min(dx_min, dy_min)**2 / (4*D)
    dt_safe = min(dt, dt_cfl, dt_diff)
    if dt_safe < dt:
        Nt = int(np.ceil(t_final / dt_safe))
        t_array = np.linspace(0, t_final, Nt+1)
        dt = dt_safe

    # Precompute "LU" decompositions for implicit solves (ADI method)
    # For Chebyshev, matrices are dense but small, so we use numpy.linalg.solve directly
    unique_betas = np.unique(np.diag(ARS_B))
    Ax_mat = {}
    Ay_mat = {}
    for beta in unique_betas:
        if beta == 0:
            continue
        Ax = I_x - dt * beta * D * Dx2
        Ay = I_y - dt * beta * D * Dy2
        Ax_mat[beta] = Ax
        Ay_mat[beta] = Ay

    # For beta=0, the solve is identity (no-op)
    def solve_Ax(beta, rhs):
        if beta == 0:
            return rhs
        # Solve Ax_mat[beta] @ X = rhs for X, for each column of rhs
        return np.linalg.solve(Ax_mat[beta], rhs)
    def solve_Ay(beta, rhs):
        if beta == 0:
            return rhs
        # Solve Ay_mat[beta] @ X = rhs for X, for each row of rhs (so transpose, solve, transpose back)
        return np.linalg.solve(Ay_mat[beta], rhs)

    # Time stepping
    for n in range(Nt):
        t_n = t_array[n]
        u_stage = u.copy()
        for s in range(3):
            beta = ARS_B[s, s]
            # Explicit advection
            adv = adv_op(u_stage)
            rhs = u.copy()
            for j in range(s):
                rhs += dt * ARS_A[s, j] * adv_op(u_stage)
            rhs += dt * beta * adv
            # Implicit diffusion via ADI (x then y)
            tmp = solve_Ax(beta, rhs)
            tmp = solve_Ay(beta, tmp.T).T
            u_stage = tmp
            u_stage = enforce_bc(u_stage, t_n + ARS_c[s]*dt)
        u_new[...] = u_stage
        u_new = enforce_bc(u_new, t_array[n+1])
        u, u_new = u_new, u  # swap

    u_final = u

    coords = {'x': x, 'y': y}
    return {
        "u": u_final,
        "coords": coords,
        "t": t_array
    }