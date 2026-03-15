import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # Domain
    Lx = pde_spec["domain"]["x_max"] - pde_spec["domain"]["x_min"]
    Ly = pde_spec["domain"]["y_max"] - pde_spec["domain"]["y_min"]
    x_min = pde_spec["domain"]["x_min"]
    y_min = pde_spec["domain"]["y_min"]
    x_max = pde_spec["domain"]["x_max"]
    y_max = pde_spec["domain"]["y_max"]

    # Grid
    Nx = plan["spatial_discretization"]["Nx"]
    Ny = plan["spatial_discretization"]["Ny"]

    # Time
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    Nt = plan["time_stepping"].get("Nt", None)
    if dt is None:
        raise ValueError("dt must be specified in plan for implicit BDF2 spectral")
    if t_final is None and Nt is None:
        raise ValueError("Either t_final or Nt must be specified")
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    if t_final is None:
        t_final = Nt * dt
    t_array = np.linspace(0, Nt*dt, Nt+1)
    t_final = t_array[-1]

    # PDE parameters
    nu = float(pde_spec["parameters"]["nu"])

    # --- 2. Create grid and wavenumbers ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Fourier wavenumbers (for spectral derivatives)
    kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0,0] = 1.0  # avoid division by zero for pressure Poisson

    # --- 3. Initial condition ---
    u0 = np.sin(X) * np.cos(Y)
    v0 = -np.cos(X) * np.sin(Y)
    # p0 = -0.25 * (np.cos(2*X) + np.cos(2*Y))  # not needed for time stepping

    # --- 4. Allocate fields (in physical and spectral space) ---
    u = u0.copy()
    v = v0.copy()
    # For BDF2, need u^n and u^{n-1}
    u_prev = u0.copy()
    v_prev = v0.copy()

    # Storage for spectral transforms
    def fft2(f):
        return np.fft.fft2(f)
    def ifft2(f_hat):
        return np.fft.ifft2(f_hat).real

    # Helper: compute nonlinear terms in physical space
    def nonlinear_terms(u, v):
        u_hat = fft2(u)
        v_hat = fft2(v)
        u_x = ifft2(1j*KX * u_hat)
        u_y = ifft2(1j*KY * u_hat)
        v_x = ifft2(1j*KX * v_hat)
        v_y = ifft2(1j*KY * v_hat)
        N1 = u * u_x + v * u_y
        N2 = u * v_x + v * v_y
        return N1, N2

    # Helper: project velocity field to divergence-free (pressure projection)
    def project(u_hat, v_hat):
        div_hat = 1j*KX*u_hat + 1j*KY*v_hat
        p_hat = div_hat / K2
        u_hat_proj = u_hat - 1j*KX*p_hat
        v_hat_proj = v_hat - 1j*KY*p_hat
        return u_hat_proj, v_hat_proj

    # --- First step: Backward Euler (BDF1) to get u^1 ---
    N1_0, N2_0 = nonlinear_terms(u0, v0)
    u0_hat = fft2(u0)
    v0_hat = fft2(v0)
    N1_0_hat = fft2(N1_0)
    N2_0_hat = fft2(N2_0)
    denom = 1 + dt*nu*K2
    rhs_u_hat = u0_hat - dt*N1_0_hat
    rhs_v_hat = v0_hat - dt*N2_0_hat
    u1_hat = rhs_u_hat / denom
    v1_hat = rhs_v_hat / denom
    # Project to divergence-free
    u1_hat, v1_hat = project(u1_hat, v1_hat)
    u1 = ifft2(u1_hat)
    v1 = ifft2(v1_hat)

    # Prepare for main loop
    u_prevprev = u0.copy()
    v_prevprev = v0.copy()
    u_prev = u1.copy()
    v_prev = v1.copy()
    u = u1.copy()
    v = v1.copy()
    u_hat = u1_hat.copy()
    v_hat = v1_hat.copy()

    # --- Main BDF2 loop ---
    alpha0 = 3/2
    for n in range(1, Nt):
        # Nonlinear terms at current step
        N1, N2 = nonlinear_terms(u, v)
        N1_hat = fft2(N1)
        N2_hat = fft2(N2)
        # Nonlinear terms at previous step
        N1_prev, N2_prev = nonlinear_terms(u_prev, v_prev)
        N1_prev_hat = fft2(N1_prev)
        N2_prev_hat = fft2(N2_prev)
        # BDF2 RHS in spectral space
        rhs_u_hat = (2*fft2(u) - 0.5*fft2(u_prev))
        rhs_v_hat = (2*fft2(v) - 0.5*fft2(v_prev))
        rhs_u_hat = rhs_u_hat - dt*(2*N1_hat - N1_prev_hat)/2
        rhs_v_hat = rhs_v_hat - dt*(2*N2_hat - N2_prev_hat)/2
        denom = alpha0 + dt*nu*K2
        u_new_hat = rhs_u_hat / denom
        v_new_hat = rhs_v_hat / denom
        # Project to divergence-free
        u_new_hat, v_new_hat = project(u_new_hat, v_new_hat)
        # Transform back to physical space
        u_new = ifft2(u_new_hat)
        v_new = ifft2(v_new_hat)
        # Prepare for next step
        u_prevprev = u_prev.copy()
        v_prevprev = v_prev.copy()
        u_prev = u.copy()
        v_prev = v.copy()
        u = u_new
        v = v_new
        u_hat = u_new_hat
        v_hat = v_new_hat

    # --- 6. Compute pressure field at final time (for residual) ---
    N1, N2 = nonlinear_terms(u, v)
    u_hat = fft2(u)
    v_hat = fft2(v)
    Lap_u = ifft2(-K2 * u_hat)
    Lap_v = ifft2(-K2 * v_hat)
    # Compute time derivatives (BDF2)
    # u_t ≈ (3u^n - 4u^{n-1} + u^{n-2})/(2dt)
    u_t = (3*u - 4*u_prev + u_prevprev) / (2*dt)
    v_t = (3*v - 4*v_prev + v_prevprev) / (2*dt)

    # Compute pressure gradient via Poisson equation
    N1_hat = fft2(N1)
    N2_hat = fft2(N2)
    div_N_hat = 1j*KX*N1_hat + 1j*KY*N2_hat
    div_Lap_hat = 1j*KX*fft2(Lap_u) + 1j*KY*fft2(Lap_v)
    p_hat = (-div_N_hat + nu*div_Lap_hat) / K2
    p_hat[0,0] = 0.0  # set mean to zero
    p = ifft2(p_hat)

    # --- 7. Compute pointwise residuals ---
    # Residuals for u and v equations:
    # Ru = u_t + u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
    # Rv = v_t + u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)
    p_hat = fft2(p)
    u_x = ifft2(1j*KX * u_hat)
    u_y = ifft2(1j*KY * u_hat)
    v_x = ifft2(1j*KX * v_hat)
    v_y = ifft2(1j*KY * v_hat)
    p_x = ifft2(1j*KX * p_hat)
    p_y = ifft2(1j*KY * p_hat)
    u_xx = ifft2(-KX**2 * u_hat)
    u_yy = ifft2(-KY**2 * u_hat)
    v_xx = ifft2(-KX**2 * v_hat)
    v_yy = ifft2(-KY**2 * v_hat)

    Ru = u_t + u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
    Rv = v_t + u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)
    # Divergence-free residual
    div = u_x + v_y

    # Stack residuals into a single ndarray: shape (3, Nx, Ny)
    residual_grid = np.stack([Ru, Rv, div], axis=0)

    # --- 8. Return only final state (memory safe) ---
    # Output: u, v, coords, t, residual
    # For 2D velocity, stack as (2, Nx, Ny)
    u_out = np.stack([u, v], axis=0)
    coords = {"x": x, "y": y}

    return {
        "u": u_out,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }