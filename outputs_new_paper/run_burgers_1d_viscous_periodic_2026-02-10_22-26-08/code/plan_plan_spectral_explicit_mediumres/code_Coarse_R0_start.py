```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and plan parameters ---
    # Domain
    x_min = pde_spec['domain'].get('x_min', pde_spec['domain']['bounds']['x'][0])
    x_max = pde_spec['domain'].get('x_max', pde_spec['domain']['bounds']['x'][1])
    L = x_max - x_min

    # Spectral grid
    Nx = plan['spatial_discretization']['Nx']
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = L / Nx

    # Viscosity
    nu = float(pde_spec['parameters']['nu'])

    # Time stepping
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    Nt = plan['time_stepping'].get('Nt', None)
    if t_final is not None:
        if dt is not None:
            Nt = int(np.ceil(t_final / dt))
            t_array = np.linspace(0, Nt*dt, Nt+1)
        else:
            # Estimate dt via CFL for Burgers: dt < dx / max|u|
            # Assume max|u| ~ 1 (from initial condition sin(2pi x))
            dt = 0.4 * dx
            Nt = int(np.ceil(t_final / dt))
            t_array = np.linspace(0, Nt*dt, Nt+1)
    elif Nt is not None and dt is not None:
        t_array = np.linspace(0, Nt*dt, Nt+1)
    else:
        raise ValueError("Either t_final or Nt must be specified in plan['time_stepping'].")

    # --- Initial condition ---
    u0 = np.sin(2 * np.pi * x)

    # --- Spectral setup ---
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)  # shape (Nx,)
    ik = 1j * k
    k2 = k**2

    # --- Time stepping: RK4 ---
    u = u0.copy()
    t = 0.0

    # Only store final state for memory safety
    for n in range(Nt):
        # RK4 steps in spectral space
        u_hat = np.fft.fft(u)

        def rhs(u_phys):
            u_hat = np.fft.fft(u_phys)
            u_x = np.fft.ifft(ik * u_hat).real
            nonlinear = u_phys * u_x
            nonlinear_hat = np.fft.fft(nonlinear)
            # Burgers: u_t = -u u_x + nu u_xx
            du_hat_dt = -nonlinear_hat + (-nu) * (k2 * u_hat)
            du_dt = np.fft.ifft(du_hat_dt).real
            return du_dt

        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t += dt

    u_final = u.copy()

    # --- Residual computation ---
    # Compute u_t (time derivative) at final time using one-sided finite difference
    # Use one backward Euler step for u_t at final time
    # u_t ≈ (u_final - u_prev) / dt
    # So, we need u_prev (one step before final)
    # We'll recompute u_prev by integrating up to Nt-1
    u = u0.copy()
    t = 0.0
    for n in range(Nt-1):
        u_hat = np.fft.fft(u)
        def rhs(u_phys):
            u_hat = np.fft.fft(u_phys)
            u_x = np.fft.ifft(ik * u_hat).real
            nonlinear = u_phys * u_x
            nonlinear_hat = np.fft.fft(nonlinear)
            du_hat_dt = -nonlinear_hat + (-nu) * (k2 * u_hat)
            du_dt = np.fft.ifft(du_hat_dt).real
            return du_dt
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t += dt
    u_prev = u.copy()

    # u_t at final time
    u_t = (u_final - u_prev) / dt

    # Compute u_x and u_xx at final time using spectral derivatives
    u_hat_final = np.fft.fft(u_final)
    u_x = np.fft.ifft(ik * u_hat_final).real
    u_xx = np.fft.ifft(-k2 * u_hat_final).real

    # Pointwise residual: u_t + u*u_x - nu*u_xx
    residual = u_t + u_final * u_x - nu * u_xx

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual
    }
```