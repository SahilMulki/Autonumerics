import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    Nx = plan["spatial_discretization"]["Nx"]
    L = x_max - x_min

    # Time
    t_final = plan["time_stepping"].get("t_final", 1.0)
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        dx = (x_max - x_min) / Nx
        dt = 0.2 * dx**3
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt  # Adjust so we land exactly at t_final

    # --- 2. Set up grid ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = x[1] - x[0]
    t_array = np.linspace(0, t_final, Nt+1)

    # --- 3. Initial condition ---
    u = 0.5 * (1 / np.cosh(0.5 * x))**2

    # --- 4. Spectral operators ---
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)  # wave numbers
    ik = 1j * k
    ik3 = (1j * k) ** 3

    # --- 5. IMEX-RK3 coefficients (ARS(2,3,2)) ---
    gamma = (2 - np.sqrt(2)) / 2

    def nonlinear(u_phys):
        # 6 * u * u_x
        u_hat = np.fft.fft(u_phys)
        u_x = np.fft.ifft(ik * u_hat).real
        return -6 * u_phys * u_x  # negative sign because equation is u_t + ... = 0

    # Precompute implicit operator for each stage
    E1 = 1.0 / (1 - gamma * dt * ik3)
    E2 = 1.0 / (1 - gamma * dt * ik3)
    E3 = 1.0 / (1 - gamma * dt * ik3)

    # --- 6. Time stepping ---
    u_hat = np.fft.fft(u)
    for n in range(Nt):
        # Stage 1
        u1 = np.fft.ifft(u_hat).real
        N1 = nonlinear(u1)
        # Stage 2
        rhs2 = u_hat + gamma * dt * np.fft.fft(N1)
        u2_hat = E1 * rhs2
        u2 = np.fft.ifft(u2_hat).real
        N2 = nonlinear(u2)
        # Stage 3
        rhs3 = u_hat + dt * ((1 - gamma) * np.fft.fft(N2) + (1 - 2 * gamma) * np.fft.fft(N1))
        u3_hat = E3 * rhs3
        u3 = np.fft.ifft(u3_hat).real
        N3 = nonlinear(u3)
        # Combine for next step (bE = [1/6, 2/3, 1/6], bI = [1/6, 2/3, 1/6])
        u_hat = (1/6) * u_hat + (2/3) * u2_hat + (1/6) * u3_hat \
              + dt * ((1/6) * np.fft.fft(N1) + (2/3) * np.fft.fft(N2) + (1/6) * np.fft.fft(N3))

    u_final = np.fft.ifft(u_hat).real

    # --- 7. Output ---
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array
    }