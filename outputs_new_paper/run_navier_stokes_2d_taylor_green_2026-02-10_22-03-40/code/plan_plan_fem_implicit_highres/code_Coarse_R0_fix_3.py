import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Parse Plan and PDE Spec ---
    # Domain
    x_min, x_max = pde_spec['domain']['x_min'], pde_spec['domain']['x_max']
    y_min, y_max = pde_spec['domain']['y_min'], pde_spec['domain']['y_max']
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    nu = float(pde_spec['parameters']['nu'])
    # Time
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', 1.0)
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    # Estimate dt by CFL for viscous and advective terms (conservative)
    u_max = 1.0  # max |u|, conservative for sin/cos
    cfl_adv = 0.2  # even more conservative for explicit advection
    cfl_visc = 0.4
    dt_cfl_adv = cfl_adv * min(dx, dy) / u_max
    dt_cfl_visc = cfl_visc * min(dx, dy)**2 / (4*nu)
    if dt is None:
        dt = min(dt_cfl_adv, dt_cfl_visc)
    else:
        dt = min(dt, dt_cfl_adv, dt_cfl_visc)
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)
    # Coordinates
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- 2. Initial Condition ---
    u = np.sin(X) * np.cos(Y)
    v = -np.cos(X) * np.sin(Y)
    # For BDF2, need u^n and u^{n-1}
    u_prev = u.copy()
    v_prev = v.copy()

    # FFT-based periodic operators
    def laplacian(f):
        return (
            (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / dx**2 +
            (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / dy**2
        )
    def grad_x(f):
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2*dx)
    def grad_y(f):
        return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2*dy)
    def div(u, v):
        return grad_x(u) + grad_y(v)
    def project(u, v):
        # Solve Poisson for pressure correction phi: laplacian(phi) = div(u*,v*)/dt
        rhs = div(u, v) / dt
        phi = poisson_periodic(rhs, dx, dy)
        u_corr = u - grad_x(phi) * dt
        v_corr = v - grad_y(phi) * dt
        return u_corr, v_corr, phi

    def poisson_periodic(rhs, dx, dy):
        # Solve laplacian(phi) = rhs with periodic BCs using FFT
        Nx_, Ny_ = rhs.shape
        kx = 2*np.pi*np.fft.fftfreq(Nx_, d=dx)
        ky = 2*np.pi*np.fft.fftfreq(Ny_, d=dy)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        denom = KX**2 + KY**2
        rhs_hat = np.fft.fft2(rhs)
        denom[0,0] = 1.0
        phi_hat = rhs_hat / denom
        phi_hat[0,0] = 0.0  # set mean to zero
        phi = np.fft.ifft2(phi_hat).real
        return phi

    # --- 3. Time Stepping: BDF2, Implicit Viscosity, Explicit Nonlinear, Projection ---
    # Store previous nonlinear terms for AB2
    u_adv_prev = u * grad_x(u) + v * grad_y(u)
    v_adv_prev = u * grad_x(v) + v * grad_y(v)

    # For BDF2, store previous solution for time derivative
    u_hist = [u_prev.copy(), u.copy()]
    v_hist = [v_prev.copy(), v.copy()]

    for n in range(Nt):
        t = t_array[n+1]
        # BDF2 coefficients
        if n == 0:
            # Backward Euler for first step
            alpha = 1.0/dt
            beta = 1.0/dt
            gamma = 0.0
        else:
            alpha = 1.5/dt
            beta = -2.0/dt
            gamma = 0.5/dt
        # Nonlinear terms (explicit, Adams-Bashforth 2)
        if n == 0:
            # Use Euler for first step
            u_adv = u * grad_x(u) + v * grad_y(u)
            v_adv = u * grad_x(v) + v * grad_y(v)
        else:
            # AB2: 1.5*N^n - 0.5*N^{n-1}
            u_adv = 1.5 * (u * grad_x(u) + v * grad_y(u)) - 0.5 * u_adv_prev
            v_adv = 1.5 * (u * grad_x(v) + v * grad_y(v)) - 0.5 * v_adv_prev
        # Right-hand side for implicit solve
        if n == 0:
            rhs_u = -u_adv + u / dt
            rhs_v = -v_adv + v / dt
        else:
            rhs_u = -u_adv + (2*u - 0.5*u_prev) / dt
            rhs_v = -v_adv + (2*v - 0.5*v_prev) / dt
        # Implicit viscous solve: (alpha - nu*L) u^{n+1} = rhs
        def implicit_solve(rhs):
            rhs_hat = np.fft.fft2(rhs)
            kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
            ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            denom = alpha + nu * (KX**2 + KY**2)
            denom[0,0] = 1.0  # avoid divide by zero
            u_hat = rhs_hat / denom
            u_hat[0,0] = 0.0  # mean velocity is zero for Taylor-Green
            u_new = np.fft.ifft2(u_hat).real
            return u_new
        u_star = implicit_solve(rhs_u)
        v_star = implicit_solve(rhs_v)
        # Projection step to enforce incompressibility
        u_new, v_new, p_corr = project(u_star, v_star)
        # Prepare for next step
        u_prev, v_prev = u, v
        u, v = u_new, v_new
        u_adv_prev = u_adv
        v_adv_prev = v_adv
        # Store for BDF2 time derivative at the end
        if len(u_hist) == 2:
            u_hist.append(u.copy())
            v_hist.append(v.copy())
        else:
            u_hist.pop(0)
            v_hist.pop(0)
            u_hist.append(u.copy())
            v_hist.append(v.copy())

    # --- 4. Compute Pressure Field (Poisson solve) ---
    # At final time, reconstruct pressure by solving Poisson: laplacian(p) = -div(N)
    N1 = u * grad_x(u) + v * grad_y(u)
    N2 = u * grad_x(v) + v * grad_y(v)
    rhs_p = -div(N1, N2)
    p = poisson_periodic(rhs_p, dx, dy)

    # --- 5. Compute Residual Grid ---
    # Residuals for u and v equations at final time
    # BDF2 time derivative (approximate u_t at final step)
    if Nt >= 2:
        u_nm1 = u_hist[-2]
        u_nm2 = u_hist[-3] if len(u_hist) >= 3 else u_hist[-2]
        v_nm1 = v_hist[-2]
        v_nm2 = v_hist[-3] if len(v_hist) >= 3 else v_hist[-2]
        u_t = (1.5*u - 2*u_nm1 + 0.5*u_nm2) / dt
        v_t = (1.5*v - 2*v_nm1 + 0.5*v_nm2) / dt
    else:
        u_nm1 = u_hist[-2]
        v_nm1 = v_hist[-2]
        u_t = (u - u_nm1) / dt
        v_t = (v - v_nm1) / dt
    # Nonlinear terms
    u_adv = u * grad_x(u) + v * grad_y(u)
    v_adv = u * grad_x(v) + v * grad_y(v)
    # Pressure gradients
    p_x = grad_x(p)
    p_y = grad_y(p)
    # Viscous terms
    u_lap = laplacian(u)
    v_lap = laplacian(v)
    # Residuals
    res_u = u_t + u_adv + p_x - nu * u_lap
    res_v = v_t + v_adv + p_y - nu * v_lap
    # Incompressibility residual
    res_div = div(u, v)
    # Stack residuals: shape (3, Nx, Ny)
    residual_grid = np.stack([res_u, res_v, res_div], axis=0)

    # --- 6. Return ---
    # Output: u, v as a single array (shape (2, Nx, Ny))
    u_out = np.stack([u, v], axis=0)
    coords = {'x': x, 'y': y}
    return {
        "u": u_out,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }