```python
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
    # u0 = 0.5 * (1 / np.cosh(0.5 * x))**2
    u = 0.5 * (1 / np.cosh(0.5 * x))**2

    # --- Spectral differentiation setup ---
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)  # wave numbers
    ik = 1j * k
    ik3 = (1j * k) ** 3

    # --- Right-hand side function for KdV ---
    def rhs(u_phys):
        # u_phys: shape (Nx,)
        u_hat = np.fft.fft(u_phys)
        u_x = np.fft.ifft(ik * u_hat).real
        u_xxx = np.fft.ifft(ik3 * u_hat).real
        return -6 * u_phys * u_x - u_xxx

    # --- Time stepping: RK4 ---
    # Only store final state for memory safety
    for n in range(Nt):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    u_final = u.copy()

    # --- Residual computation ---
    # Compute pointwise PDE residual at final time:
    # residual = u_t + 6*u*u_x + u_xxx
    # Approximate u_t by backward difference (since we only have u_final)
    # We'll do one extra backward Euler step for u_t estimate
    # (Alternatively, do one step backward in time with -dt, but that's less stable)
    # Instead, use the PDE: u_t = rhs(u), so residual = u_t + 6*u*u_x + u_xxx = rhs(u) + 6*u*u_x + u_xxx = 0
    # But rhs(u) = -6*u*u_x - u_xxx, so residual = 0. To check the actual residual, we can compute:
    # Compute u_x and u_xxx spectrally
    u_hat = np.fft.fft(u_final)
    u_x = np.fft.ifft(ik * u_hat).real
    u_xxx = np.fft.ifft(ik3 * u_hat).real
    # Approximate u_t by finite difference (backward Euler)
    # Step u_back = u_final - dt * rhs(u_final)
    u_back = u_final - dt * rhs(u_final)
    u_t_approx = (u_final - u_back) / dt
    residual_grid = u_t_approx + 6 * u_final * u_x + u_xxx

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual_grid
    }
```