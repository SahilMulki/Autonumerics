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
        # Estimate dt by CFL (not needed here, but for robustness)
        dt = 0.1 * dx**4 / eps**2
    # Reduce dt for stability (Cahn-Hilliard is very stiff)
    dt = min(dt, 2e-5)
    Nt = int(np.ceil(t_final / dt))
    t_array = np.linspace(0, t_final, Nt+1)
    # --- Spectral setup ---
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)  # shape (Nx,)
    k2 = k**2
    k4 = k2**2
    # Dealiasing mask (2/3 rule)
    kfreq = np.fft.fftfreq(Nx)
    dealias = np.abs(kfreq) < (1/3)
    # --- IMEX CN-AB2 coefficients ---
    # Linear operator in Fourier: L = -eps^2 * d^4/dx^4
    L_hat = -eps**2 * k4
    # For Crank-Nicolson: (I - dt/2*L) u^{n+1} = (I + dt/2*L) u^n + dt*NL
    A = 1 - 0.5*dt*L_hat
    B = 1 + 0.5*dt*L_hat
    # --- Nonlinear function ---
    def nonlinear_term(u):
        # u: real space, shape (Nx,)
        u3_minus_u = u**3 - u
        u3_minus_u_hat = np.fft.fft(u3_minus_u)
        # Dealias
        u3_minus_u_hat = u3_minus_u_hat * dealias
        # Compute second derivative in Fourier
        nonlinear_hat = -k2 * u3_minus_u_hat
        # Return to real space
        return np.fft.ifft(nonlinear_hat).real
    # --- Time stepping ---
    u = u0.copy()
    u_hat = np.fft.fft(u)
    NL0 = nonlinear_term(u)
    # First step: Forward Euler for NL, CN for L
    rhs_hat = B * u_hat + dt * np.fft.fft(NL0)
    u_hat_new = rhs_hat / A
    u_new = np.fft.ifft(u_hat_new).real
    NL1 = nonlinear_term(u_new)
    # Main loop
    for n in range(1, Nt):
        # AB2 for NL: NL^{n+1/2} = 1.5*NL^n - 0.5*NL^{n-1}
        NL_ab2 = 1.5*NL1 - 0.5*NL0
        rhs_hat = B * u_hat_new + dt * np.fft.fft(NL_ab2)
        u_hat_next = rhs_hat / A
        u_next = np.fft.ifft(u_hat_next).real
        # Rotate variables
        u_hat, u_hat_new = u_hat_new, u_hat_next
        u, u_new = u_new, u_next
        NL0, NL1 = NL1, nonlinear_term(u_new)
    # Final state
    u_final = u_new

    # --- Residual calculation ---
    # Compute u_t ≈ (u_final - u_prev) / dt
    # Use last available time step for residual
    u_t = (u_final - u) / dt
    # Compute -(eps^2 * u_xxxx + (u^3 - u)_xx)
    # Compute u_xxxx in Fourier
    u_hat_final = np.fft.fft(u_final)
    u_xxxx_hat = k4 * u_hat_final
    u_xxxx = np.fft.ifft(u_xxxx_hat).real
    # Compute (u^3 - u)_xx
    u3_minus_u = u_final**3 - u_final
    u3_minus_u_hat = np.fft.fft(u3_minus_u)
    # Dealias
    u3_minus_u_hat = u3_minus_u_hat * dealias
    u3_minus_u_xx_hat = -k2 * u3_minus_u_hat
    u3_minus_u_xx = np.fft.ifft(u3_minus_u_xx_hat).real
    # Residual at final time
    residual_pointwise = u_t + eps**2 * u_xxxx + u3_minus_u_xx
    # L2 norm over domain
    residual = np.sqrt(np.sum(np.abs(residual_pointwise)**2) * dx)

    # --- Return ---
    return {
        "u": u_final,
        "coords": coords,
        "t": t_array,
        "residual": residual
    }