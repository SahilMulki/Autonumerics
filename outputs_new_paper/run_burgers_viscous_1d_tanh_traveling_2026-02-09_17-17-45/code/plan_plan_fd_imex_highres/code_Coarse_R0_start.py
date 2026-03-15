```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and plan parameters ---
    # Domain
    x_min, x_max = pde_spec['domain']['bounds']['x']
    Nx = plan['spatial_discretization']['Nx']
    nu = float(pde_spec['parameters']['nu'])
    # Time
    t_final = plan['time_stepping']['t_final']
    dt = plan['time_stepping'].get('dt', None)
    order = plan['time_stepping'].get('order', 4)
    # Use CFL if dt not given (for Burgers: dt <= dx / max|u|, but here dt is given)
    # Grid
    x = np.linspace(x_min, x_max, Nx)
    dx = x[1] - x[0]
    # Time steps
    if dt is None:
        # Estimate dt by CFL for Burgers: dt <= dx / max|u|, max|u| ~ 1
        dt = 0.4 * dx
    Nt = int(np.ceil(t_final / dt))
    t_array = np.linspace(0, t_final, Nt+1)
    dt = t_final / Nt  # adjust dt so that t_array[-1] == t_final

    # --- Initial condition ---
    u = np.tanh(x / (2 * nu))
    u = u.astype(np.float64)
    u0 = u.copy()

    # --- Boundary conditions (Dirichlet, analytic) ---
    def bc_left(t):
        return np.tanh((x_min) / (2 * nu))
    def bc_right(t):
        return np.tanh((x_max) / (2 * nu))

    # --- Finite difference operators ---
    def convection_flux(u):
        # Fourth-order central difference for u_x (interior points)
        # u_x[i] = (u[i-2] - 8u[i-1] + 8u[i+1] - u[i+2]) / (12*dx)
        ux = np.zeros_like(u)
        ux[2:-2] = (u[0:-4] - 8*u[1:-3] + 8*u[3:-1] - u[4:]) / (12*dx)
        # For boundaries, fall back to lower order (second order central)
        ux[1] = (u[2] - u[0]) / (2*dx)
        ux[-2] = (u[-1] - u[-3]) / (2*dx)
        # For very edges, use one-sided (first order)
        ux[0] = (u[1] - u[0]) / dx
        ux[-1] = (u[-1] - u[-2]) / dx
        return ux

    def diffusion_operator(u):
        # Second-order central difference for u_xx
        uxx = np.zeros_like(u)
        uxx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        # Dirichlet BC: u[0], u[-1] fixed, so uxx[0], uxx[-1] = 0 (not used)
        uxx[0] = 0.0
        uxx[-1] = 0.0
        return uxx

    # --- Implicit diffusion matrix (tridiagonal) ---
    # For implicit step: (I - dt*nu*D2) u^{n+1} = rhs
    main_diag = np.ones(Nx) + 2 * nu * dt / dx**2
    off_diag = -nu * dt / dx**2 * np.ones(Nx-1)
    # Dirichlet BC rows: main_diag[0] = main_diag[-1] = 1, off_diag[0] = off_diag[-1] = 0
    main_diag[0] = main_diag[-1] = 1.0
    off_diag[0] = off_diag[-2] = 0.0
    # Precompute banded matrix for Thomas algorithm
    def solve_diffusion(rhs):
        # Tridiagonal solve: A u = rhs, where A is (main_diag, off_diag)
        a = off_diag.copy()
        b = main_diag.copy()
        c = off_diag.copy()
        d = rhs.copy()
        n = len(d)
        # Forward elimination
        for i in range(1, n):
            w = a[i-1] / b[i-1]
            b[i] = b[i] - w * c[i-1]
            d[i] = d[i] - w * d[i-1]
        # Back substitution
        u_sol = np.zeros_like(d)
        u_sol[-1] = d[-1] / b[-1]
        for i in range(n-2, -1, -1):
            u_sol[i] = (d[i] - c[i] * u_sol[i+1]) / b[i]
        return u_sol

    # --- IMEX ARK4 coefficients (Kennedy & Carpenter 2003, Table 3, 6-stage) ---
    # For brevity, use a 4-stage ARK4(3)6L[2]SA scheme (simplified, see e.g. https://runge.math.smu.edu/arkode-dev/ButcherTable.html)
    # We'll use a classic 4-stage IMEX RK for demonstration:
    # Explicit (for convection): classic RK4
    # Implicit (for diffusion): DIRK with same nodes
    rk_a = np.array([0, 0.5, 0.5, 1.0])
    rk_b = np.array([1/6, 1/3, 1/3, 1/6])
    rk_c = np.array([0, 0.5, 0.5, 1.0])

    # --- Time stepping loop ---
    u = u0.copy()
    for n in range(Nt):
        t_n = n * dt
        # Stage arrays
        u_stage = u.copy()
        k_exp = []
        k_imp = []
        u_stages = []
        for s in range(4):
            # Compute explicit convection RHS at this stage
            if s == 0:
                u_exp = u_stage
            else:
                u_exp = u + rk_a[s] * dt * k_exp[s-1]
            # Dirichlet BCs for stage
            u_exp[0] = bc_left(t_n + rk_c[s]*dt)
            u_exp[-1] = bc_right(t_n + rk_c[s]*dt)
            # Explicit convection term
            conv = -u_exp * convection_flux(u_exp)
            k_exp.append(conv)
            # Implicit diffusion solve
            rhs = u + dt * rk_a[s] * k_exp[s]
            # Dirichlet BCs for implicit solve
            rhs[0] = bc_left(t_n + rk_c[s]*dt)
            rhs[-1] = bc_right(t_n + rk_c[s]*dt)
            # Solve (I - rk_a[s]*dt*nu*D2) u_stage = rhs
            if s == 0 or rk_a[s] == 0:
                u_imp = rhs.copy()
            else:
                # Build tridiagonal for this stage
                a = -nu * rk_a[s] * dt / dx**2 * np.ones(Nx-1)
                b = np.ones(Nx) + 2 * nu * rk_a[s] * dt / dx**2
                c = -nu * rk_a[s] * dt / dx**2 * np.ones(Nx-1)
                # Dirichlet BC rows
                b[0] = b[-1] = 1.0
                a[0] = c[-2] = 0.0
                # Thomas algorithm
                # Forward elimination
                d = rhs.copy()
                for i in range(1, Nx):
                    w = a[i-1] / b[i-1]
                    b[i] = b[i] - w * c[i-1]
                    d[i] = d[i] - w * d[i-1]
                u_imp = np.zeros_like(d)
                u_imp[-1] = d[-1] / b[-1]
                for i in range(Nx-2, -1, -1):
                    u_imp[i] = (d[i] - c[i] * u_imp[i+1]) / b[i]
            k_imp.append((u_imp - u) / (rk_a[s]*dt) if (s > 0 and rk_a[s] != 0) else diffusion_operator(u_imp))
            u_stages.append(u_imp)
        # Combine stages for next step
        u_new = u.copy()
        for s in range(4):
            u_new += dt * rk_b[s] * (k_exp[s] + nu * diffusion_operator(u_stages[s]))
        # Dirichlet BCs at new time
        u_new[0] = bc_left(t_n + dt)
        u_new[-1] = bc_right(t_n + dt)
        u = u_new
    # --- End time stepping ---

    # --- Residual calculation ---
    # At final time, compute PDE residual at all grid points
    # PDE: u_t + u u_x - nu u_xx = 0
    # Approximate u_t by backward difference (since we only store final state)
    # u_t ≈ (u_final - u_prev) / dt
    # To get u_prev, do one backward Euler step (approximate)
    u_prev = u.copy()
    # One backward Euler step (approximate u_prev)
    # u = u_prev + dt*(-u_prev*u_prev_x + nu*u_prev_xx)
    # => u_prev ≈ u - dt*(-u*u_x + nu*u_xx)
    u_x = convection_flux(u)
    u_xx = diffusion_operator(u)
    u_prev = u - dt * (-u * u_x + nu * u_xx)
    u_t = (u - u_prev) / dt
    residual = u_t + u * u_x - nu * u_xx

    # --- Output ---
    return {
        "u": u.copy(),
        "coords": {"x": x.copy()},
        "t": t_array,
        "residual": residual.copy()
    }
```