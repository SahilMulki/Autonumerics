```python
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
        # Estimate dt by CFL for diffusion: dt < dx^2/(4*D)
        dx = (x_max - x_min) / (Nx-1)
        dy = (y_max - y_min) / (Ny-1)
        dt = 0.4 * min(dx, dy)**2 / (4*D)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    else:
        t_final = Nt * dt
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Chebyshev grid and differentiation matrices ---
    def chebyshev_nodes(N, a, b):
        k = np.arange(N)
        x = np.cos(np.pi * k / (N - 1))
        return 0.5*(b-a)*x + 0.5*(b+a)

    def chebyshev_D(N, a, b):
        # Chebyshev differentiation matrix on [a,b]
        x = np.cos(np.pi * np.arange(N) / (N - 1))
        c = np.ones(N)
        c[0] = 2
        c[-1] = 2
        c = c * ((-1) ** np.arange(N))
        X = np.tile(x, (N,1)).T
        dX = X - X.T + np.eye(N)
        D = np.outer(c, 1/c) / (dX)
        D = D - np.diag(np.sum(D, axis=1))
        # Scale to [a,b]
        D = 2.0/(b-a) * D
        x_phys = 0.5*(b-a)*x + 0.5*(b+a)
        return D, x_phys

    Dx, x = chebyshev_D(Nx, x_min, x_max)
    Dy, y = chebyshev_D(Ny, y_min, y_max)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial condition ---
    # (1/(2*np.pi))*np.exp(-(x**2 + y**2)/2)
    u = (1/(2*np.pi))*np.exp(-(X**2 + Y**2)/2)

    # --- Dirichlet BC function (analytic solution) ---
    def sigma2_t(t):
        return D/lam + (1 - D/lam)*np.exp(-2*lam*t)
    def bc_func(x, y, t):
        s2 = sigma2_t(t)
        return (1/(2*np.pi*s2))*np.exp(-(x**2 + y**2)/(2*s2))

    # --- Spectral Laplacian operator ---
    # For Chebyshev, Laplacian: L[u] = D*(u_xx + u_yy)
    # We'll use matrix multiplication for each direction
    # For Dirichlet BCs, we enforce u at boundaries at each step

    # --- IMEX Runge-Kutta coefficients (ARS(2,3,2) as a standard 3rd order IMEX) ---
    # https://arxiv.org/pdf/1110.1876.pdf Table 2.1
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
    # For each direction, build the implicit operator for Laplacian
    # We solve (I - dt*gamma*D*L) u = rhs at each stage
    # L = D*(Dx2 + Dy2)
    I_x = np.eye(Nx)
    I_y = np.eye(Ny)
    Dx2 = np.matmul(Dx, Dx)
    Dy2 = np.matmul(Dy, Dy)
    # For each stage, the implicit solve is (I - dt*gamma*D*Dx2) in x and (I - dt*gamma*D*Dy2) in y
    # We'll use ADI (alternating direction implicit) for efficiency

    # --- Helper: enforce Dirichlet BCs ---
    def enforce_bc(u, t):
        # Set boundary values to analytic solution
        # x boundaries
        u[0, :] = bc_func(x[0], y, t)
        u[-1, :] = bc_func(x[-1], y, t)
        # y boundaries
        u[:, 0] = bc_func(x, y[0], t)
        u[:, -1] = bc_func(x, y[-1], t)
        return u

    # --- Helper: explicit advection operator ---
    def adv_op(u):
        # div(lambda*[x,y]*u) = lambda*(d/dx(x*u) + d/dy(y*u))
        # Compute x-derivative
        xu = X * u
        yu = Y * u
        dx_xu = Dx @ xu
        dy_yu = yu @ Dy.T
        return lam * (dx_xu + dy_yu)

    # --- Main time stepping loop ---
    u = enforce_bc(u, t_array[0])
    u_new = np.empty_like(u)
    for n in range(Nt):
        t_n = t_array[n]
        dt_stage = dt

        # IMEX ARS(2,3,2) 3-stage
        u_stage = u.copy()
        for s in range(3):
            # Compute explicit part (advection) at this stage
            adv = adv_op(u_stage)
            # Right-hand side for implicit solve
            rhs = u
            for j in range(s):
                rhs += dt_stage * ARS_A[s, j] * adv_op(u_stage)
            rhs += dt_stage * ARS_B[s, s] * adv
            # Implicit solve: (I - dt*gamma*D*L) u = rhs
            # Use ADI: first x, then y
            # Step 1: solve in x for each y
            A_x = I_x - dt_stage * ARS_B[s, s] * D * Dx2
            tmp = np.linalg.solve(A_x, rhs)
            # Step 2: solve in y for each x
            A_y = I_y - dt_stage * ARS_B[s, s] * D * Dy2
            tmp = np.linalg.solve(A_y, tmp.T).T
            u_stage = tmp
            # Enforce BCs at each stage
            u_stage = enforce_bc(u_stage, t_n + ARS_c[s]*dt_stage)
        u_new[...] = u_stage
        u_new = enforce_bc(u_new, t_array[n+1])
        u, u_new = u_new, u  # swap

    # --- Final solution ---
    u_final = u

    # --- Compute residual at final time ---
    # PDE: rho_t = D*(rho_xx + rho_yy) + div(lambda*[x,y]*rho)
    # Residual: R = u_t - D*(u_xx + u_yy) - div(lambda*[x,y]*u)
    # Approximate u_t by backward difference
    # For u_t, need previous step
    # We'll recompute one step back if needed
    if Nt > 0:
        # Step back one dt
        u_prev = u_final.copy()
        u = (1/(2*np.pi))*np.exp(-(X**2 + Y**2)/2)
        u = enforce_bc(u, t_array[0])
        u_new = np.empty_like(u)
        for n in range(Nt-1):
            t_n = t_array[n]
            u_stage = u.copy()
            for s in range(3):
                adv = adv_op(u_stage)
                rhs = u
                for j in range(s):
                    rhs += dt * ARS_A[s, j] * adv_op(u_stage)
                rhs += dt * ARS_B[s, s] * adv
                A_x = I_x - dt * ARS_B[s, s] * D * Dx2
                tmp = np.linalg.solve(A_x, rhs)
                A_y = I_y - dt * ARS_B[s, s] * D * Dy2
                tmp = np.linalg.solve(A_y, tmp.T).T
                u_stage = tmp
                u_stage = enforce_bc(u_stage, t_n + ARS_c[s]*dt)
            u_new[...] = u_stage
            u_new = enforce_bc(u_new, t_array[n+1])
            u, u_new = u_new, u
        u_prev = u
    else:
        u_prev = u_final

    u_t = (u_final - u_prev) / dt

    # Compute Laplacian
    u_xx = Dx2 @ u_final
    u_yy = u_final @ Dy2.T
    lap = u_xx + u_yy

    # Compute advection (div(lambda*[x,y]*u))
    xu = X * u_final
    yu = Y * u_final
    dx_xu = Dx @ xu
    dy_yu = yu @ Dy.T
    adv = lam * (dx_xu + dy_yu)

    residual_grid = u_t - D * lap - adv

    # --- Output ---
    coords = {'x': x, 'y': y}
    return {
        "u": u_final,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }
```
**Notes:**
- Only the final state `u` is stored (memory safe).
- Residual is computed pointwise as required.
- Chebyshev spectral method with IMEX ARS(2,3,2) (3rd order) is used.
- Dirichlet BCs are enforced at every stage using the analytic solution.
- No external libraries except NumPy are used.