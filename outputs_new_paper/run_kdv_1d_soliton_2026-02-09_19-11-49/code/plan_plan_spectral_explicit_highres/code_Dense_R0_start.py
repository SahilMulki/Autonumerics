import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters from plan and spec ---
    # Spatial grid
    Nx = plan['spatial_discretization']['Nx']
    x_min, x_max = pde_spec['domain']['bounds']['x']
    L = x_max - x_min
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = L / Nx

    # Time stepping
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    Nt = plan['time_stepping'].get('Nt', None)
    if t_final is not None:
        if dt is not None:
            Nt = int(np.round(t_final / dt))
            t_array = np.linspace(0, Nt*dt, Nt+1)
        elif Nt is not None:
            dt = t_final / Nt
            t_array = np.linspace(0, t_final, Nt+1)
        else:
            # Estimate dt using CFL for KdV: dt < C * dx^3 (very conservative)
            dt = 0.2 * dx**3
            Nt = int(np.round(t_final / dt))
            t_array = np.linspace(0, Nt*dt, Nt+1)
    else:
        if Nt is not None and dt is not None:
            t_final = Nt * dt
            t_array = np.linspace(0, t_final, Nt+1)
        else:
            raise ValueError("Either t_final or Nt must be specified in the plan.")

    # --- Initial condition ---
    # u0 = 0.5 * (1 / np.cosh(0.5 * x))**2
    u = 0.5 * (1 / np.cosh(0.5 * x))**2

    # --- Precompute spectral operators ---
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)  # wave numbers
    ik = 1j * k
    ik3 = (1j * k) ** 3

    # Linear operator in Fourier space: L = -d^3/dx^3
    L_op = -ik3

    # ETDRK4 scalar coefficients (Kassam & Trefethen 2005)
    E = np.exp(L_op * dt)
    E2 = np.exp(L_op * dt / 2.0)
    # Avoid division by zero for L_op=0
    L_op_safe = L_op.copy()
    L_op_safe[np.abs(L_op_safe) == 0] = 1e-20
    M = 32  # for contour integral
    r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)
    LR = dt * L_op[:, None] + r[None, :]
    Q = dt * np.mean((np.exp(LR/2.0) - 1) / LR, axis=1)
    f1 = dt * np.mean((-4 - LR + np.exp(LR)*(4 - 3*LR + LR**2)) / LR**3, axis=1)
    f2 = dt * np.mean((2 + LR + np.exp(LR)*(-2 + LR)) / LR**3, axis=1)
    f3 = dt * np.mean((-4 - 3*LR - LR**2 + np.exp(LR)*(4 - LR)) / LR**3, axis=1)

    # --- Time stepping (ETDRK4) ---
    u_hat = np.fft.fft(u)
    u_hat_hist = [u_hat.copy()]  # store for residual computation
    for n in range(Nt):
        # Nonlinear term N(u) = -6u u_x
        u_phys = np.fft.ifft(u_hat).real
        u_x = np.fft.ifft(ik * u_hat).real
        N1 = -6.0 * u_phys * u_x
        N1_hat = np.fft.fft(N1)

        a_hat = E2 * u_hat + Q * N1_hat
        a = np.fft.ifft(a_hat).real
        a_x = np.fft.ifft(ik * a_hat).real
        N2 = -6.0 * a * a_x
        N2_hat = np.fft.fft(N2)

        b_hat = E2 * u_hat + Q * N2_hat
        b = np.fft.ifft(b_hat).real
        b_x = np.fft.ifft(ik * b_hat).real
        N3 = -6.0 * b * b_x
        N3_hat = np.fft.fft(N3)

        c_hat = E * u_hat + Q * (2*N3_hat)
        c = np.fft.ifft(c_hat).real
        c_x = np.fft.ifft(ik * c_hat).real
        N4 = -6.0 * c * c_x
        N4_hat = np.fft.fft(N4)

        u_hat = (E * u_hat +
                 f1 * N1_hat +
                 2 * f2 * (N2_hat + N3_hat) +
                 f3 * N4_hat)
        if n == Nt-2:
            u_hat_hist.append(u_hat.copy())

    # Final solution in physical space
    u_final = np.fft.ifft(u_hat).real

    # --- Compute residual grid ---
    # Residual: R = u_t + 6u u_x + u_xxx
    # Approximate u_t by backward difference
    if len(u_hat_hist) == 2:
        u_prev = np.fft.ifft(u_hat_hist[0]).real
    else:
        # If only one time step, use initial condition as previous
        u_prev = np.fft.ifft(np.fft.fft(0.5 * (1 / np.cosh(0.5 * x))**2)).real

    u_t = (u_final - u_prev) / dt

    # u_x
    u_x = np.fft.ifft(ik * np.fft.fft(u_final)).real
    # u_xxx
    u_xxx = np.fft.ifft((ik)**3 * np.fft.fft(u_final)).real

    # Residual at each grid point
    residual_grid = u_t + 6 * u_final * u_x + u_xxx

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual_grid
    }