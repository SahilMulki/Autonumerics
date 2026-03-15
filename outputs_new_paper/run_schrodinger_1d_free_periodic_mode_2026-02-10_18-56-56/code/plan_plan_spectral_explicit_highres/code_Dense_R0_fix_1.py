import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # Domain
    x_min, x_max = pde_spec['domain']['bounds']['x']
    Nx = plan['spatial_discretization']['Nx']
    L = x_max - x_min

    # Time
    t_final = plan['time_stepping']['t_final']
    dt = plan['time_stepping'].get('dt', None)
    if dt is None:
        dx = L / Nx
        dt = 0.5 * dx**2
    Nt = int(np.round(t_final / dt))
    t_array = np.linspace(0, t_final, Nt+1)
    dt = t_array[1] - t_array[0]  # recompute for exact spacing

    # PDE parameters
    m = pde_spec['parameters'].get('m', 1)

    # --- 2. Set up spatial grid and wavenumbers ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi  # Fourier wavenumbers

    # --- 3. Initial condition ---
    u0 = np.exp(1j * 1 * x)

    # --- 4. Time stepping: Strang splitting (exact linear evolution) ---
    # For the linear Schrödinger equation with periodic BCs, the exact solution in spectral space is:
    # u_hat(t+dt) = u_hat(t) * exp(-0.5j * k^2 * dt)
    # This is unconditionally stable and preserves norm.
    u = u0.copy()
    u_hat = np.fft.fft(u)

    # Precompute the linear evolution operator
    linear_evolution = np.exp(-0.5j * (k**2) * dt)

    # Store solution at all times if desired (here, only final state is required)
    for n in range(Nt):
        u_hat = u_hat * linear_evolution

    u = np.fft.ifft(u_hat)

    # --- 5. Output ---
    result = {
        "u": u,  # shape (Nx,)
        "coords": {"x": x},
        "t": t_array
    }
    return result