import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters from pde_spec and plan ---
    # Domain
    x_min, x_max = pde_spec["domain"]["x_min"], pde_spec["domain"]["x_max"]
    y_min, y_max = pde_spec["domain"]["y_min"], pde_spec["domain"]["y_max"]
    # Grid
    Nx = plan["spatial_discretization"]["Nx"]
    Ny = plan["spatial_discretization"]["Ny"]
    # Time
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None and t_final is not None and dt is not None:
        Nt = int(np.ceil(t_final / dt))
    elif Nt is not None and dt is None and t_final is not None:
        dt = t_final / Nt
    elif Nt is not None and dt is not None:
        t_final = Nt * dt
    else:
        # Fallback: estimate dt by CFL for viscous term
        nu = pde_spec["parameters"]["nu"]
        dx = (x_max - x_min) / Nx
        dy = (y_max - y_min) / Ny
        dt = 0.4 * min(dx, dy)**2 / (4*nu)
        Nt = int(np.ceil(t_final / dt))
    # Physical parameter
    nu = pde_spec["parameters"]["nu"]

    # --- Create grid ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial conditions ---
    u = np.sin(X) * np.cos(Y)
    v = -np.cos(X) * np.sin(Y)
    # Pressure is not evolved, but needed for residual
    p = -0.25 * (np.cos(2*X) + np.cos(2*Y))

    # For BDF2, need u^{n-1} and u^{n}
    u_nm1 = u.copy()
    v_nm1 = v.copy()
    p_nm1 = p.copy()

    # Laplacian and gradient operators (periodic)
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

    def nonlinear_advect(u, v, f):
        return u * grad_x(f) + v * grad_y(f)

    # Projection method: Chorin's method
    def project(u_star, v_star):
        # Solve Poisson: Lap p = div(u_star)/dt
        rhs = div(u_star, v_star) / dt
        p_hat = poisson_solve(rhs)
        # Subtract grad p
        u_proj = u_star - dt * grad_x(p_hat)
        v_proj = v_star - dt * grad_y(p_hat)
        return u_proj, v_proj, p_hat

    def poisson_solve(rhs):
        # Solve Lap p = rhs with periodic BCs using FFT
        rhs_hat = np.fft.fft2(rhs)
        kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        denom = -(KX**2 + KY**2)
        denom[0,0] = 1.0  # avoid division by zero for mean
        p_hat = rhs_hat / denom
        p_hat[0,0] = 0.0  # set mean to zero (pressure defined up to const)
        p = np.fft.ifft2(p_hat).real
        return p

    # First step: backward Euler (BDF1)
    for step in range(1):
        # Nonlinear term at n
        N_u = nonlinear_advect(u, v, u)
        N_v = nonlinear_advect(u, v, v)
        # Implicit viscous term: (I - dt*nu*L) u* = u^n - dt*N(u^n)
        rhs_u = u - dt * N_u
        rhs_v = v - dt * N_v
        def implicit_solve(rhs):
            # Solve (I - dt*nu*L) f_new = rhs using FFT (periodic)
            f_hat = np.fft.fft2(rhs)
            kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
            ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            denom = 1 + dt * nu * (KX**2 + KY**2)
            f_new_hat = f_hat / denom
            return np.fft.ifft2(f_new_hat).real
        u_star = implicit_solve(rhs_u)
        v_star = implicit_solve(rhs_v)
        # Projection
        u1, v1, p1 = project(u_star, v_star)
        # Store for BDF2
        u_nm1 = u.copy()
        v_nm1 = v.copy()
        p_nm1 = p.copy()
        u = u1
        v = v1
        p = p1

    # --- Main BDF2 time stepping ---
    t_array = np.arange(0, Nt+1) * dt
    t = dt  # current time
    for n in range(1, Nt):
        t += dt
        # Nonlinear terms at n and n-1
        N_u_n = nonlinear_advect(u, v, u)
        N_v_n = nonlinear_advect(u, v, v)
        N_u_nm1 = nonlinear_advect(u_nm1, v_nm1, u_nm1)
        N_v_nm1 = nonlinear_advect(u_nm1, v_nm1, v_nm1)
        # BDF2 coefficients: (3/2 u^{n+1} - 2 u^n + 0.5 u^{n-1})/dt = RHS
        # RHS = -N(u^n) + 0.5*N(u^{n-1}) + nu*Lap(u^{n+1}) - grad p^{n+1}
        rhs_u = (2*u - 0.5*u_nm1) - dt * (2*N_u_n - 0.5*N_u_nm1)
        rhs_v = (2*v - 0.5*v_nm1) - dt * (2*N_v_n - 0.5*N_v_nm1)
        def implicit_solve_bdf2(rhs):
            f_hat = np.fft.fft2(rhs)
            kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
            ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            denom = (3/2) + dt * nu * (KX**2 + KY**2)
            f_new_hat = f_hat / denom
            return np.fft.ifft2(f_new_hat).real
        u_star = implicit_solve_bdf2(rhs_u)
        v_star = implicit_solve_bdf2(rhs_v)
        # Projection
        u_new, v_new, p_new = project(u_star, v_star)
        # Update for next step
        u_nm1, v_nm1, p_nm1 = u, v, p
        u, v, p = u_new, v_new, p_new

    # --- Compute residual at final time ---
    # u_t ≈ (u^{n} - u^{n-1}) / dt (BDF1 for residual)
    u_t = (u - u_nm1) / dt
    v_t = (v - v_nm1) / dt
    u_x = grad_x(u)
    u_y = grad_y(u)
    v_x = grad_x(v)
    v_y = grad_y(v)
    u_xx = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dx**2
    u_yy = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dy**2
    v_xx = (np.roll(v, -1, axis=0) - 2*v + np.roll(v, 1, axis=0)) / dx**2
    v_yy = (np.roll(v, -1, axis=1) - 2*v + np.roll(v, 1, axis=1)) / dy**2
    p_x = grad_x(p)
    p_y = grad_y(p)
    # Residuals
    res_u = u_t + u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
    res_v = v_t + u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)
    res_div = grad_x(u) + grad_y(v)
    # Stack residuals: shape (3, Nx, Ny)
    residual_grid = np.stack([res_u, res_v, res_div], axis=0)

    # --- Output ---
    u_out = np.stack([u, v], axis=0)  # shape (2, Nx, Ny)
    coords = {"x": x, "y": y}
    return {
        "u": u_out,
        "coords": coords,
        "t": t_array[-1],
        "residual": residual_grid
    }