import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Parse domain and discretization ---
    # Spatial grid
    Nx = plan['spatial_discretization']['Nx']
    x_min, x_max = pde_spec['domain']['bounds']['x']
    L = x_max - x_min
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = L / Nx

    # Time grid
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    Nt = plan['time_stepping'].get('Nt', None)
    if Nt is None:
        if dt is not None and t_final is not None:
            Nt = int(np.round(t_final / dt))
        else:
            # Estimate dt via CFL for KdV: dt < C * dx^3 (for stability, C ~ 0.4)
            dt = 0.4 * dx**3
            t_final = 1.0
            Nt = int(np.round(t_final / dt))
    else:
        if dt is None:
            dt = t_final / Nt
    t_array = np.arange(Nt + 1) * dt

    # --- Initial condition ---
    u = 0.5 * (1 / np.cosh(0.5 * x))**2

    # --- Spectral differentiation setup ---
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)  # wave numbers
    ik = 1j * k
    ik3 = (1j * k) ** 3

    # --- Precompute linear evolution operator for integrating factor ---
    # For explicit RK4, instability is likely due to the stiff linear term u_xxx.
    # Use integrating factor method (Fourier space) to stabilize:
    # Let v = exp(-L*dt) * u_hat, where L = -ik3 (linear part)
    # The nonlinear part is handled in physical space.

    exp_Ldt = np.exp(ik3 * dt)
    exp_Ldt2 = np.exp(ik3 * dt / 2)

    def nonlinear(u_phys):
        u_hat = np.fft.fft(u_phys)
        u_x = np.fft.ifft(ik * u_hat).real
        return -6 * u_phys * u_x

    # --- Time stepping: Integrating Factor RK4 ---
    u_hat = np.fft.fft(u)
    for n in range(Nt):
        # Step 1
        u_phys = np.fft.ifft(u_hat).real
        N1 = nonlinear(u_phys)
        a = dt * np.fft.fft(N1)
        # Step 2
        u_hat2 = u_hat * exp_Ldt2 + 0.5 * a * exp_Ldt2
        u_phys2 = np.fft.ifft(u_hat2).real
        N2 = nonlinear(u_phys2)
        b = dt * np.fft.fft(N2)
        # Step 3
        u_hat3 = u_hat * exp_Ldt2 + 0.5 * b * exp_Ldt2
        u_phys3 = np.fft.ifft(u_hat3).real
        N3 = nonlinear(u_phys3)
        c = dt * np.fft.fft(N3)
        # Step 4
        u_hat4 = u_hat * exp_Ldt + c * exp_Ldt
        u_phys4 = np.fft.ifft(u_hat4).real
        N4 = nonlinear(u_phys4)
        d = dt * np.fft.fft(N4)
        # RK4 update in Fourier space
        u_hat = u_hat * exp_Ldt + (a + 2*b + 2*c + d) / 6.0 * exp_Ldt

    u_final = np.fft.ifft(u_hat).real

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array
    }