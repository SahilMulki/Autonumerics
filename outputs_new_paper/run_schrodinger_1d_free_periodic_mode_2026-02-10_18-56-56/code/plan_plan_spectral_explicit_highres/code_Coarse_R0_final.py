```python
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
        # Estimate dt by CFL for Schrödinger: dt < C * dx^2 (C ~ 0.5 for stability)
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
    # IC: np.exp(1j*1*x)
    u0 = np.exp(1j * 1 * x)

    # --- 4. Time stepping: RK4 in spectral space ---
    u = u0.copy()
    u_hat = np.fft.fft(u)

    # Linear operator in spectral space: 1j*u_t = -0.5*u_xx
    # => u_t = 1j*0.5*u_xx = 1j*0.5*(-k^2)*u_hat = -0.5j*k^2*u_hat
    def rhs(u_hat):
        return -0.5j * (k**2) * u_hat

    for n in range(Nt):
        k1 = rhs(u_hat)
        k2 = rhs(u_hat + 0.5 * dt * k1)
        k3 = rhs(u_hat + 0.5 * dt * k2)
        k4 = rhs(u_hat + dt * k3)
        u_hat = u_hat + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    u = np.fft.ifft(u_hat)

    # --- 5. Residual calculation ---
    # PDE: 1j*u_t = -0.5*u_xx
    # Residual: R = 1j*u_t + 0.5*u_xx
    # Approximate u_t by backward difference: (u^n - u^{n-1})/dt
    # For u^{n-1}, step back one dt using the same scheme (or just one Euler step for residual)
    # For high accuracy, we can do one backward Euler step from u_hat

    # Step back one dt to get u_prev
    u_hat_prev = u_hat.copy()
    # One backward Euler step (for residual only, not for solution)
    # u_hat_prev = u_hat - dt * rhs(u_hat)
    # But for accuracy, let's do one RK4 backward step
    k1 = rhs(u_hat)
    k2 = rhs(u_hat - 0.5 * dt * k1)
    k3 = rhs(u_hat - 0.5 * dt * k2)
    k4 = rhs(u_hat - dt * k3)
    u_hat_prev = u_hat - (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    u_prev = np.fft.ifft(u_hat_prev)

    # u_t ≈ (u - u_prev)/dt
    u_t = (u - u_prev) / dt

    # u_xx in spectral: u_xx_hat = -(k^2) * u_hat
    u_xx_hat = -(k**2) * u_hat
    u_xx = np.fft.ifft(u_xx_hat)

    # Residual: 1j*u_t + 0.5*u_xx
    residual = 1j * u_t + 0.5 * u_xx
    # Return as ndarray (complex)

    # --- 6. Output ---
    result = {
        "u": u,  # shape (Nx,)
        "coords": {"x": x},
        "t": t_array,
        "residual": residual
    }
    return result
```