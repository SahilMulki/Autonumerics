import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters from pde_spec and plan ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    # Spectral grid
    Nx = plan["spatial_discretization"]["Nx"]
    Lx = x_max - x_min
    dx = Lx / Nx
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    coords = {"x": x}
    # Parameters
    eps = float(pde_spec["parameters"]["eps"])
    # Initial condition
    u0 = 0.1 * np.cos(2 * np.pi * x)
    # Time stepping
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    if t_final is None:
        raise ValueError("t_final must be specified in plan['time_stepping']")
    if dt is None:
        dt = 0.1 * dx**4 / eps**2
    dt = min(dt, 2e-6)
    Nt = int(np.ceil(t_final / dt))
    t_array = np.linspace(0, t_final, Nt+1)
    # --- Spectral setup ---
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    k2 = k**2
    k4 = k2**2
    # Dealiasing mask (2/3 rule)
    kfreq = np.fft.fftfreq(Nx)
    dealias = np.abs(kfreq) < (1/3)
    # --- IMEX CN-AB2 coefficients ---
    L_hat = -eps**2 * k4
    A = 1 - 0.5*dt*L_hat
    B = 1 + 0.5*dt*L_hat
    # --- Nonlinear function ---
    def nonlinear_term(u):
        u3_minus_u = u**3 - u
        u3_minus_u_hat = np.fft.fft(u3_minus_u)
        u3_minus_u_hat = u3_minus_u_hat * dealias
        nonlinear_hat = -k2 * u3_minus_u_hat
        return np.fft.ifft(nonlinear_hat).real
    # --- Time stepping ---
    u = u0.copy()
    u_hat = np.fft.fft(u)
    NL0 = nonlinear_term(u)
    rhs_hat = B * u_hat + dt * np.fft.fft(NL0)
    u_hat_new = rhs_hat / A
    u_new = np.fft.ifft(u_hat_new).real
    NL1 = nonlinear_term(u_new)
    for n in range(1, Nt):
        NL_ab2 = 1.5*NL1 - 0.5*NL0
        rhs_hat = B * u_hat_new + dt * np.fft.fft(NL_ab2)
        u_hat_next = rhs_hat / A
        u_next = np.fft.ifft(u_hat_next).real
        u_hat, u_hat_new = u_hat_new, u_hat_next
        u, u_new = u_new, u_next
        NL0, NL1 = NL1, nonlinear_term(u_new)
    u_final = u_new

    # --- Residual calculation ---
    # Compute u_t (time derivative) using backward difference
    # Re-run last step to get u_prev
    # (We already have u and u_final: u is previous, u_final is current)
    u_t = (u_final - u) / dt

    # Compute spatial derivatives in Fourier space
    u_hat_final = np.fft.fft(u_final)
    u_hat_prev = np.fft.fft(u)
    # Linear term: -eps^2 * u_xxxx
    u_xxxx = np.fft.ifft(k4 * u_hat_final).real
    linear_term = -eps**2 * u_xxxx
    # Nonlinear term: ((u^3 - u)_xx)
    u3_minus_u = u_final**3 - u_final
    u3_minus_u_hat = np.fft.fft(u3_minus_u)
    u3_minus_u_hat = u3_minus_u_hat * dealias
    nonlinear_term_xx = np.fft.ifft(-k2 * u3_minus_u_hat).real
    # Residual: u_t + eps^2 u_xxxx + ((u^3-u)_xx)
    residual = u_t + linear_term + nonlinear_term_xx
    # L2 norm of residual
    residual_l2 = np.sqrt(np.sum(np.abs(residual)**2) * dx)

    return {
        "u": u_final,
        "coords": coords,
        "t": t_array,
        "residual": residual_l2
    }