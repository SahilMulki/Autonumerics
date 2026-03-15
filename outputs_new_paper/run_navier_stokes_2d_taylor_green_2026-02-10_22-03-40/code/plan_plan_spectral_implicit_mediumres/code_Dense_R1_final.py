import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters ---
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
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    else:
        t_final = Nt * dt
    # PDE parameters
    nu = pde_spec["parameters"]["nu"]

    # --- Grids and wavenumbers ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Fourier wavenumbers
    kx = np.fft.fftfreq(Nx, d=(x_max - x_min)/Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=(y_max - y_min)/Ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0,0] = 1e-20  # avoid division by zero for pressure projection

    # --- Initial condition ---
    u0 = np.sin(X) * np.cos(Y)
    v0 = -np.cos(X) * np.sin(Y)
    # p0 = -0.25 * (np.cos(2*X) + np.cos(2*Y))  # not needed for evolution

    # --- Allocate fields ---
    u = u0.copy()
    v = v0.copy()
    # For BDF2, need previous two steps
    u_prev = u0.copy()
    v_prev = v0.copy()
    # For BDF2, need u_prev2, v_prev2 for residual at the end
    u_prev2 = u0.copy()
    v_prev2 = v0.copy()
    # First step: use BDF1 (Backward Euler) for bootstrapping

    # --- Time stepping ---
    t_array = np.arange(0, t_final+dt, dt)
    if t_array[-1] > t_final + 1e-12:
        t_array = t_array[:-1]
    Nt = len(t_array)

    # Only store final state for memory safety
    for n in range(Nt):
        t = t_array[n]
        # --- Nonlinear terms (in physical space) ---
        # Compute nonlinear terms at current step
        # Spectral derivatives
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)
        u_x = np.fft.ifft2(1j*KX*u_hat).real
        u_y = np.fft.ifft2(1j*KY*u_hat).real
        v_x = np.fft.ifft2(1j*KX*v_hat).real
        v_y = np.fft.ifft2(1j*KY*v_hat).real

        # Nonlinear terms
        N1 = u*u_x + v*u_y  # for u
        N2 = u*v_x + v*v_y  # for v

        # Dealiasing (2/3 rule)
        def dealias(f_hat):
            kx_cut = int(Nx/3)
            ky_cut = int(Ny/3)
            f_hat_dealiased = f_hat.copy()
            if kx_cut > 0:
                f_hat_dealiased[kx_cut:-kx_cut, :] = 0
            if ky_cut > 0:
                f_hat_dealiased[:, ky_cut:-ky_cut] = 0
            return f_hat_dealiased

        N1_hat = np.fft.fft2(N1)
        N2_hat = np.fft.fft2(N2)
        N1_hat = dealias(N1_hat)
        N2_hat = dealias(N2_hat)
        N1 = np.fft.ifft2(N1_hat).real
        N2 = np.fft.ifft2(N2_hat).real

        # --- BDF coefficients ---
        if n == 0:
            # Backward Euler for first step
            a0, a1, a2 = 1, -1, 0
            b = 1
            # For first step, use previous only
            rhs_u = u + dt*(-N1)
            rhs_v = v + dt*(-N2)
        else:
            # BDF2: (3u^{n+1} - 4u^n + u^{n-1})/(2dt) = RHS
            a0, a1, a2 = 3, -4, 1
            b = 2
            rhs_u = (4*u - u_prev)/b + dt*(-N1)
            rhs_v = (4*v - v_prev)/b + dt*(-N2)

        rhs_u_hat = np.fft.fft2(rhs_u)
        rhs_v_hat = np.fft.fft2(rhs_v)

        # --- Solve for tentative velocity (without pressure) ---
        # (a0/b/dt - nu*L) * u_hat_new = rhs_hat
        # L = -K2, so nu*L = -nu*K2
        factor = (a0/(b*dt) + nu*K2)
        u_hat_new = rhs_u_hat / factor
        v_hat_new = rhs_v_hat / factor

        # --- Pressure projection (enforce incompressibility) ---
        # Compute divergence of tentative velocity
        div_hat = 1j*KX*u_hat_new + 1j*KY*v_hat_new
        # Solve Poisson: K2 * p_hat = divergence / dt
        p_hat = div_hat / (K2 * (a0/(b*dt)))
        p_hat[0,0] = 0.0  # set mean pressure to zero

        # Subtract pressure gradient
        u_hat_new -= (a0/(b*dt)) * 1j*KX*p_hat / factor
        v_hat_new -= (a0/(b*dt)) * 1j*KY*p_hat / factor

        # Transform back to physical space
        u_new = np.fft.ifft2(u_hat_new).real
        v_new = np.fft.ifft2(v_hat_new).real

        # Prepare for next step
        if n == 0:
            u_prev2 = u_prev.copy()
            v_prev2 = v_prev.copy()
        else:
            u_prev2 = u_prev.copy()
            v_prev2 = v_prev.copy()
        u_prev, v_prev = u.copy(), v.copy()
        u, v = u_new, v_new

    # --- Compute residual at final time ---
    # u, v are at final time
    # Compute all terms in physical space
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    # Derivatives
    u_x = np.fft.ifft2(1j*KX*u_hat).real
    u_y = np.fft.ifft2(1j*KY*u_hat).real
    v_x = np.fft.ifft2(1j*KX*v_hat).real
    v_y = np.fft.ifft2(1j*KY*v_hat).real
    u_xx = np.fft.ifft2(-(KX**2)*u_hat).real
    u_yy = np.fft.ifft2(-(KY**2)*u_hat).real
    v_xx = np.fft.ifft2(-(KX**2)*v_hat).real
    v_yy = np.fft.ifft2(-(KY**2)*v_hat).real

    # Pressure: solve Poisson equation for p
    # div(u) = 0, so take divergence of momentum equation to get pressure
    # Laplacian(p) = -div(NL)
    N1 = u*u_x + v*u_y
    N2 = u*v_x + v*v_y
    NL1_hat = np.fft.fft2(N1)
    NL2_hat = np.fft.fft2(N2)
    div_NL_hat = 1j*KX*NL1_hat + 1j*KY*NL2_hat
    p_hat = div_NL_hat / K2
    p_hat[0,0] = 0.0
    p = np.fft.ifft2(p_hat).real
    # Pressure gradients
    p_x = np.fft.ifft2(1j*KX*p_hat).real
    p_y = np.fft.ifft2(1j*KY*p_hat).real

    # Time derivative: use backward difference (first order) as approximation
    u_t = (u - u_prev) / dt
    v_t = (v - v_prev) / dt

    # Residuals
    res_u = u_t + N1 + p_x - nu*(u_xx + u_yy)
    res_v = v_t + N2 + p_y - nu*(v_xx + v_yy)
    res_div = u_x + v_y  # incompressibility

    # Stack residuals into a single array: shape (Nx, Ny, 3)
    residual_grid = np.stack([res_u, res_v, res_div], axis=-1)

    # --- Output ---
    u_out = np.stack([u, v], axis=-1)  # shape (Nx, Ny, 2)
    coords = {"x": x, "y": y}
    return {
        "u": u_out,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }