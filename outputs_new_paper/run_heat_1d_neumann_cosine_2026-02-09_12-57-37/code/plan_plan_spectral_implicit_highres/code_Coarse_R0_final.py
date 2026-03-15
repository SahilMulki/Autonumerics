import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and plan parameters ---
    # Domain
    x_min = pde_spec["domain"].get("x_min", pde_spec["domain"]["bounds"]["x"][0])
    x_max = pde_spec["domain"].get("x_max", pde_spec["domain"]["bounds"]["x"][1])
    L = x_max - x_min

    # Parameters
    alpha = float(pde_spec["parameters"]["alpha"])

    # Spatial discretization
    Nx = int(plan["spatial_discretization"]["Nx"])
    x = np.linspace(x_min, x_max, Nx, endpoint=False)  # grid for DCT

    dx = L / Nx

    # Time stepping
    t_final = float(plan["time_stepping"]["t_final"])
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        dt = 0.4 * dx**2 / alpha
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt  # adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Initial condition ---
    # u(x,0) = cos(pi x)
    u = np.cos(np.pi * x)

    # --- DCT-I and IDCT-I implementation (Neumann BCs) ---
    def dct1(u):
        N = len(u)
        v = np.concatenate((u, u[-2:0:-1]))
        U = np.real(np.fft.fft(v))
        return U[:N] / (N-1) * 2

    def idct1(U):
        N = len(U)
        V = np.zeros(2*N-2, dtype=float)
        V[:N] = U
        V[N:] = U[-2:0:-1]
        v = np.fft.ifft(V)
        # DCT-I normalization
        u = np.real(v[:N]) * (N-1) / 2
        return u

    # Precompute eigenvalues for DCT-I (Neumann BCs)
    n = np.arange(Nx)
    k = n * np.pi / L
    lap_eigs = -k**2

    # --- Time stepping: Backward Euler in spectral space ---
    U = dct1(u)
    denom = 1 - dt * alpha * lap_eigs

    # Store for time output if needed
    u_hist = None
    if plan["time_stepping"].get("store_all", False):
        u_hist = np.zeros((Nt+1, Nx))
        u_hist[0] = u.copy()

    for nstep in range(Nt):
        U = U / denom  # implicit backward Euler step in spectral space
        if u_hist is not None:
            u_hist[nstep+1] = idct1(U)

    u_final = idct1(U)

    # --- Compute residual at final time ---
    # To get u_prev, step backward one step
    U_prev = U * denom
    u_prev = idct1(U_prev)
    u_t = (u_final - u_prev) / dt

    # Compute u_xx via DCT
    U_final = dct1(u_final)
    u_xx = idct1(U_final * lap_eigs)

    residual = u_t - alpha * u_xx

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual
    }