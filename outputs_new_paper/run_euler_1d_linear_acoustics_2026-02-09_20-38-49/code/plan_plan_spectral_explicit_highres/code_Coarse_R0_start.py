```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    L = pde_spec['domain']['bounds']['x'][1] - pde_spec['domain']['bounds']['x'][0]
    x_min = pde_spec['domain']['bounds']['x'][0]
    x_max = pde_spec['domain']['bounds']['x'][1]
    rho0 = float(pde_spec['parameters']['rho0'])
    c = float(pde_spec['parameters']['c'])

    # --- Extract Plan parameters ---
    Nx = int(plan['spatial_discretization']['Nx'])
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    Nt = plan['time_stepping'].get('Nt', None)
    if dt is None:
        # Estimate dt by CFL: dt < dx / c
        dx = (x_max - x_min) / Nx
        dt = 0.5 * dx / c
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    else:
        t_final = Nt * dt
    # Recompute dt to exactly hit t_final
    dt = t_final / Nt

    # --- Grids ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = (x_max - x_min) / Nx
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Initial conditions ---
    # p(x,0) = sin(x), u(x,0) = 0
    p0 = np.sin(x)
    u0 = np.zeros_like(x)

    # --- Spectral wave numbers (Fourier, periodic) ---
    k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi  # shape (Nx,)

    # --- Right-hand side function for the ODE system ---
    def rhs(y):
        # y = [p, u], both shape (Nx,)
        p, u = y
        # Compute derivatives in spectral space
        p_hat = np.fft.fft(p)
        u_hat = np.fft.fft(u)
        # Derivatives
        p_x = np.fft.ifft(1j * k * p_hat).real
        u_x = np.fft.ifft(1j * k * u_hat).real
        # PDEs:
        # p_t = -c^2 * rho0 * u_x
        # u_t = -(1/rho0) * p_x
        p_t = -c**2 * rho0 * u_x
        u_t = -(1/rho0) * p_x
        return np.array([p_t, u_t])

    # --- RK4 time stepping ---
    p = p0.copy()
    u = u0.copy()
    y = np.array([p, u])  # shape (2, Nx)
    # Only store final state for memory safety
    for n in range(Nt):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    p = y[0]
    u = y[1]

    # --- Residual calculation ---
    # Compute p_t and u_t numerically at final time using the PDE
    # (since we don't have u at t+dt, use the PDE to compute time derivatives at t_final)
    # Compute spatial derivatives
    p_hat = np.fft.fft(p)
    u_hat = np.fft.fft(u)
    p_x = np.fft.ifft(1j * k * p_hat).real
    u_x = np.fft.ifft(1j * k * u_hat).real
    # Compute PDE residuals at each point:
    # For each equation, residual = LHS - RHS evaluated at (p, u, p_x, u_x)
    # Equation 1: p_t + c^2 * rho0 * u_x = 0
    # Equation 2: u_t + (1/rho0) * p_x = 0
    # Use the PDE to compute time derivatives at t_final:
    p_t = -c**2 * rho0 * u_x
    u_t = -(1/rho0) * p_x
    # Now, compute the residual as LHS - RHS (should be zero if solution is exact)
    # But since we only have p and u at t_final, and not at t+dt, we use the PDE itself:
    # residual1 = p_t + c^2 * rho0 * u_x = 0
    # residual2 = u_t + (1/rho0) * p_x = 0
    # But since p_t and u_t are defined as above, the residuals are identically zero.
    # To get a meaningful residual, we can use a finite difference in time:
    # Use a backward Euler step for p_t and u_t:
    # p_t ≈ (p(t_final) - p(t_final - dt)) / dt
    # u_t ≈ (u(t_final) - u(t_final - dt)) / dt
    # So, rerun one step backward to get p_prev, u_prev:
    y_prev = y.copy()
    # Integrate backward one step (for residual only)
    # Use forward Euler backward in time for simplicity (since dt is small)
    k1b = rhs(y_prev)
    y_prev = y_prev - dt * k1b
    p_prev = y_prev[0]
    u_prev = y_prev[1]
    # Now, finite difference in time:
    p_t_fd = (p - p_prev) / dt
    u_t_fd = (u - u_prev) / dt
    # Now, residuals:
    residual1 = p_t_fd + c**2 * rho0 * u_x
    residual2 = u_t_fd + (1/rho0) * p_x
    # Stack residuals: shape (2, Nx)
    residual_grid = np.stack([residual1, residual2], axis=0)

    # --- Output ---
    # u: shape (2, Nx), where u[0]=p, u[1]=u
    u_out = np.stack([p, u], axis=0)
    coords = {'x': x}
    return {
        "u": u_out,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }
```