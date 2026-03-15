import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract domain and discretization ---
    # Domain
    x_min = pde_spec["domain"]["x_min"]
    x_max = pde_spec["domain"]["x_max"]
    L = x_max - x_min

    # Discretization
    Nx = plan["spatial_discretization"]["Nx"]
    dx = L / Nx
    x = np.linspace(x_min, x_max, Nx, endpoint=False)

    # Time stepping
    t_final = plan["time_stepping"].get("t_final", 1.0)
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt by CFL: dt <= dx / max|u| (u_max ~ 1 for sin(x))
        dt = 0.5 * dx / 1.0
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt  # adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Initial condition ---
    u = np.sin(x)

    # --- Spectral setup ---
    k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi  # wave numbers
    k = k.astype(np.float64)

    # Dealiasing mask (2/3 rule)
    dealias = np.ones(Nx, dtype=bool)
    k_cut = (2.0/3.0) * (Nx//2)
    k_abs = np.abs(np.fft.fftfreq(Nx, d=dx) * Nx)
    dealias[k_abs > k_cut] = False

    # --- RHS function for Burgers: du/dt = -u u_x ---
    def rhs(u_phys):
        # Compute nonlinear term in physical space, transform to spectral, dealias
        u_hat = np.fft.fft(u_phys)
        u_x_hat = 1j * k * u_hat
        u_x = np.fft.ifft(u_x_hat).real
        nonlinear = u_phys * u_x
        nonlinear_hat = np.fft.fft(nonlinear)
        # Dealias
        nonlinear_hat = nonlinear_hat * dealias
        # Back to physical
        rhs_phys = -np.fft.ifft(nonlinear_hat).real
        return rhs_phys

    # --- Time stepping: RK4 ---
    t = 0.0
    u_curr = u.copy()
    for n in range(Nt):
        k1 = rhs(u_curr)
        k2 = rhs(u_curr + 0.5*dt*k1)
        k3 = rhs(u_curr + 0.5*dt*k2)
        k4 = rhs(u_curr + dt*k3)
        u_next = u_curr + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        u_curr = u_next
        t += dt

    u_final = u_curr

    # --- Residual calculation ---
    # Compute u_t at final step using one backward Euler step (finite difference)
    # u_t ≈ (u_final - u_prev) / dt
    # To get u_prev, step back one dt using RK4 with negative dt
    u_prev = u_final.copy()
    for _ in range(1):  # one step back
        k1 = rhs(u_prev)
        k2 = rhs(u_prev - 0.5*dt*k1)
        k3 = rhs(u_prev - 0.5*dt*k2)
        k4 = rhs(u_prev - dt*k3)
        u_prev = u_prev - (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    u_t = (u_final - u_prev) / dt

    # Compute u_x at final step (spectral derivative)
    u_hat = np.fft.fft(u_final)
    u_x_hat = 1j * k * u_hat
    u_x = np.fft.ifft(u_x_hat).real

    # Residual: u_t + u * u_x
    residual_grid = u_t + u_final * u_x

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual_grid
    }