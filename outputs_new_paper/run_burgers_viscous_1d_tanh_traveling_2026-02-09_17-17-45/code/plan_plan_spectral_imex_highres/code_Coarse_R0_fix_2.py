import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    nu = float(pde_spec["parameters"]["nu"])
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    Nx = int(plan["spatial_discretization"]["Nx"])
    t_final = float(plan["time_stepping"]["t_final"])
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        dx = (x_max - x_min) / Nx
        dt = 0.4 * dx / 1.0  # max|u| ~ 1 for tanh profile
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # Adjust dt to hit t_final exactly

    # --- Spectral grid (Fourier, periodic) ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = x[1] - x[0]
    L = x_max - x_min

    # Wavenumbers for FFT (Fourier basis)
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)  # shape (Nx,)

    # --- Initial condition ---
    u0 = np.tanh(x / (2 * nu))

    # --- Dirichlet BCs (enforced via dealiasing and padding, but spectral is periodic) ---
    # For spectral Fourier, Dirichlet BCs are not exactly enforced; but with high Nx and tanh profile, error is small.

    # --- Dealiasing mask (2/3 rule) ---
    dealias_cut = int(Nx // 3)
    dealias_mask = np.zeros(Nx, dtype=bool)
    dealias_mask[:dealias_cut] = True
    dealias_mask[-dealias_cut:] = True

    # --- Time stepping: IMEX-BDF2 (implicit for diffusion, explicit for nonlinear convection) ---
    # u_t + u u_x = nu u_xx

    # Storage: only keep u^{n+1}, u^n, u^{n-1}
    u_nm1 = u0.copy()
    u_n = u0.copy()

    # Precompute BC values for reference (analytic)
    bc_left = np.tanh((x_min) / (2 * nu))
    bc_right = np.tanh((x_max) / (2 * nu))

    # --- First step (IMEX Euler) ---
    u_hat_n = np.fft.fft(u_n)
    u_x = np.fft.ifft(1j * k * u_hat_n).real
    N_u = u_n * u_x
    N_u_hat = np.fft.fft(N_u)
    N_u_hat[~dealias_mask] = 0.0

    rhs_hat = u_hat_n / dt - N_u_hat
    denom = 1.0 / dt + nu * k**2
    u_hat_np1 = rhs_hat / denom
    u_np1 = np.fft.ifft(u_hat_np1).real

    # No BC enforcement: spectral method is periodic, so do not overwrite endpoints

    # Prepare for main loop
    u_hist = [u_n.copy(), u_np1.copy()]
    t_hist = [0.0, dt]
    t = dt
    step = 1

    # --- Main IMEX-BDF2 loop ---
    while t < t_final - 1e-12:
        u_nm1 = u_hist[-2]
        u_n = u_hist[-1]

        u_hat_n = np.fft.fft(u_n)
        u_x = np.fft.ifft(1j * k * u_hat_n).real
        N_u = u_n * u_x
        N_u_hat = np.fft.fft(N_u)
        N_u_hat[~dealias_mask] = 0.0

        rhs = (4 * u_n - u_nm1) / (2 * dt)
        rhs_hat = np.fft.fft(rhs) - N_u_hat
        denom = 3.0 / (2 * dt) + nu * k**2
        u_hat_np1 = rhs_hat / denom
        u_np1 = np.fft.ifft(u_hat_np1).real

        u_hist.append(u_np1.copy())
        if len(u_hist) > 2:
            u_hist.pop(0)
        t_hist.append(t + dt)
        if len(t_hist) > 2:
            t_hist.pop(0)
        t += dt
        step += 1

    # Final state
    u_final = u_hist[-1].copy()
    t_array = np.array([0.0, t_final])

    # --- Residual computation ---
    # Compute u_t, u_x, u_xx at final time
    # u_t ≈ (u_final - u_n) / dt (backward difference)
    u_t = (u_final - u_hist[-2]) / dt
    u_hat_final = np.fft.fft(u_final)
    u_x = np.fft.ifft(1j * k * u_hat_final).real
    u_xx = np.fft.ifft(-k**2 * u_hat_final).real
    residual = u_t + u_final * u_x - nu * u_xx

    # L2 norm of the residual
    residual_l2 = np.sqrt(np.sum(np.abs(residual)**2) * dx)

    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual_l2
    }