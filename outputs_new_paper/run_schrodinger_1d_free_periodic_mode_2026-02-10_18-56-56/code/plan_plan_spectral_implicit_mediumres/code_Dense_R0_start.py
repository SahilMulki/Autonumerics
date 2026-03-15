```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # === 1. Extract parameters ===
    # Domain and grid
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    Nx = plan["spatial_discretization"]["Nx"]
    L = x_max - x_min
    dx = L / Nx
    x = np.linspace(x_min, x_max, Nx, endpoint=False)

    # Time stepping
    t_final = plan["time_stepping"]["t_final"]
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt by CFL: dt < dx^2 for stability (not critical for implicit, but safe)
        dt = 0.5 * dx**2
    Nt = int(np.round(t_final / dt))
    dt = t_final / Nt  # Adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # PDE parameters
    m = pde_spec["parameters"].get("m", 1)

    # === 2. Initial condition ===
    # IC: np.exp(1j*1*x)
    u0 = np.exp(1j * m * x)
    u = u0.copy()

    # === 3. Spectral setup ===
    k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi  # wave numbers
    k2 = k**2

    # === 4. Precompute spectral operator for implicit midpoint ===
    # 1j*u_t = -0.5*u_xx  => u_t = -0.5j*u_xx
    # In Fourier: du_hat/dt = -0.5j*(-k^2)*u_hat = 0.5j*k^2*u_hat
    # Implicit midpoint: (u^{n+1} - u^n)/dt = RHS at midpoint
    # For linear: u^{n+1} = u^n + dt * RHS( (u^{n+1} + u^n)/2 )
    # In Fourier: 
    #   (u_hat^{n+1} - u_hat^n)/dt = 0.5j*k^2 * (u_hat^{n+1} + u_hat^n)/2
    #   => (u_hat^{n+1} - u_hat^n) = (dt/2) * 0.5j*k^2 * (u_hat^{n+1} + u_hat^n)
    #   => (1 - (dt/4)*1j*k^2) u_hat^{n+1} = (1 + (dt/4)*1j*k^2) u_hat^n
    #   => u_hat^{n+1} = A * u_hat^n, with
    A_num = 1 + (dt/4)*1j*k2
    A_den = 1 - (dt/4)*1j*k2

    # === 5. Time stepping ===
    u_hat = np.fft.fft(u)
    for n in range(Nt):
        u_hat = (A_num * u_hat) / A_den
    u = np.fft.ifft(u_hat)

    # === 6. Residual calculation ===
    # Compute u_t numerically (backward difference)
    # At t_final: u_t ≈ (u_final - u_prev) / dt
    # To avoid storing all steps, do one more step from u_prev to u_final
    # So, step to u_prev at t = t_final - dt
    u_hat_prev = np.fft.fft(u0)
    for n in range(Nt-1):
        u_hat_prev = (A_num * u_hat_prev) / A_den
    u_prev = np.fft.ifft(u_hat_prev)

    u_t = (u - u_prev) / dt

    # Compute u_xx via spectral
    u_hat_final = np.fft.fft(u)
    u_xx = np.fft.ifft(-k2 * u_hat_final)

    # Residual: 1j*u_t + 0.5*u_xx  (should be zero)
    residual = 1j * u_t + 0.5 * u_xx

    # === 7. Output ===
    # Only return final state (memory safe)
    return {
        "u": u,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual
    }
```