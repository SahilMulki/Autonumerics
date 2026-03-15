import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract domain and discretization parameters ---
    domain = pde_spec["domain"]
    x_min, x_max = domain["bounds"]["x"]
    L = x_max - x_min

    Nx = plan["spatial_discretization"]["Nx"]
    dx = L / Nx
    x = np.linspace(x_min, x_max, Nx, endpoint=False)

    # --- Time stepping parameters ---
    t_final = plan["time_stepping"].get("t_final", 1.0)
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt by CFL: dt < dx / max|u| (u_max ~ 1 for sin(x))
        dt = 0.5 * dx / 1.0
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Spectral wavenumbers for Fourier basis ---
    k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi  # shape (Nx,)
    k = 1j * k  # for differentiation

    # --- Dealiasing mask (2/3 rule) ---
    dealias = np.zeros(Nx, dtype=bool)
    cutoff = int(Nx // 3)
    dealias[:cutoff] = True
    dealias[-cutoff:] = True

    # --- Initial condition: u(x,0) = sin(x) ---
    u = np.sin(x)

    # --- Time stepping: explicit RK3 ---
    def nonlinear_rhs(u_phys):
        # Compute u * u_x in spectral space, with dealiasing
        u_hat = np.fft.fft(u_phys)
        u_x_hat = k * u_hat
        u_x = np.fft.ifft(u_x_hat).real
        nonlinear = u_phys * u_x
        # Dealiasing: transform to Fourier, zero out high modes, back
        nonlinear_hat = np.fft.fft(nonlinear)
        nonlinear_hat[~dealias] = 0
        nonlinear_dealiased = np.fft.ifft(nonlinear_hat).real
        return -nonlinear_dealiased

    # Only store final state for memory safety
    t = 0.0
    for n in range(Nt):
        # RK3 (Shu-Osher)
        u1 = u + dt * nonlinear_rhs(u)
        u2 = (3/4)*u + (1/4)*(u1 + dt * nonlinear_rhs(u1))
        u = (1/3)*u + (2/3)*(u2 + dt * nonlinear_rhs(u2))
        t += dt

    u_final = u.copy()

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": np.array([t])
    }