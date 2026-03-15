```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract domain and discretization parameters ---
    # Domain
    x_min = pde_spec['domain']['x_min']
    x_max = pde_spec['domain']['x_max']
    L = x_max - x_min

    # Grid
    Nx = plan['spatial_discretization']['Nx']
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = L / Nx

    # Time
    t_final = plan['time_stepping']['t_final']
    dt = plan['time_stepping'].get('dt', None)
    if dt is None:
        # Estimate dt by CFL: dt < dx / max_wave_speed
        H = pde_spec['parameters']['H']
        g = pde_spec['parameters']['g']
        c = np.sqrt(g * H)
        dt = 0.5 * dx / c
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # Adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # PDE parameters
    H = pde_spec['parameters']['H']
    g = pde_spec['parameters']['g']

    # --- Initial conditions ---
    eta0 = np.sin(x)
    u0 = np.zeros_like(x)

    # --- Spectral wavenumbers ---
    k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi  # shape (Nx,)
    ik = 1j * k

    # --- Storage for final state only (memory safe) ---
    eta = eta0.copy()
    u = u0.copy()

    # --- Right-hand side function for ODE system ---
    def rhs(eta, u):
        # Compute spatial derivatives via FFT
        eta_hat = np.fft.fft(eta)
        u_hat = np.fft.fft(u)
        eta_x = np.fft.ifft(ik * eta_hat).real
        u_x = np.fft.ifft(ik * u_hat).real
        # Linearized shallow water equations
        deta_dt = -H * u_x
        du_dt = -g * eta_x
        return deta_dt, du_dt

    # --- Time stepping: RK4 ---
    for n in range(Nt):
        # RK4 steps
        k1_eta, k1_u = rhs(eta, u)
        k2_eta, k2_u = rhs(eta + 0.5*dt*k1_eta, u + 0.5*dt*k1_u)
        k3_eta, k3_u = rhs(eta + 0.5*dt*k2_eta, u + 0.5*dt*k2_u)
        k4_eta, k4_u = rhs(eta + dt*k3_eta, u + dt*k3_u)
        eta += (dt/6.0)*(k1_eta + 2*k2_eta + 2*k3_eta + k4_eta)
        u   += (dt/6.0)*(k1_u   + 2*k2_u   + 2*k3_u   + k4_u)

    # --- Compute residuals at final time ---
    # eta_t, u_t: time derivatives at t_final (using RHS)
    eta_t, u_t = rhs(eta, u)
    # eta_x, u_x: spatial derivatives at t_final
    eta_hat = np.fft.fft(eta)
    u_hat = np.fft.fft(u)
    eta_x = np.fft.ifft(ik * eta_hat).real
    u_x = np.fft.ifft(ik * u_hat).real

    # Residuals:
    # eq1: eta_t + H * u_x = 0
    # eq2: u_t + g * eta_x = 0
    residual_eta = eta_t + H * u_x
    residual_u   = u_t   + g * eta_x

    # Stack residuals into a (2, Nx) array: [residual_eta, residual_u]
    residual_grid = np.stack([residual_eta, residual_u], axis=0)

    # Stack solution into a (2, Nx) array: [eta, u]
    u_out = np.stack([eta, u], axis=0)

    # Output
    return {
        "u": u_out,  # shape (2, Nx)
        "coords": {"x": x},
        "t": t_array,
        "residual": residual_grid  # shape (2, Nx)
    }
```