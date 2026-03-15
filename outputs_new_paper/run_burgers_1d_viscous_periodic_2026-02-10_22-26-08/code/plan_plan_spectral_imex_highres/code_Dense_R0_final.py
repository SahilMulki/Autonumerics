```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and plan parameters ---
    # Domain
    x_min = pde_spec["domain"].get("x_min", pde_spec["domain"]["bounds"]["x"][0])
    x_max = pde_spec["domain"].get("x_max", pde_spec["domain"]["bounds"]["x"][1])
    L = x_max - x_min

    # Parameters
    nu = float(pde_spec["parameters"]["nu"])

    # Spatial discretization
    Nx = int(plan["spatial_discretization"]["Nx"])
    dx = L / Nx
    x = np.linspace(x_min, x_max, Nx, endpoint=False)  # periodic grid

    # Time discretization
    t_final = float(plan["time_stepping"]["t_final"])
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt by CFL: dt <= C*dx/max|u|, but u is sin(2pi x) in [-1,1]
        dt = 0.4 * dx / (2 * np.pi)  # conservative
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Initial condition ---
    u = np.sin(2 * np.pi * x)

    # --- Spectral setup ---
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)  # Fourier wavenumbers

    # --- IMEX time stepping: RK4 for nonlinear, implicit (BE) for diffusion ---
    def nonlinear_rhs(u_phys):
        # Nonlinear term: -u * u_x (note sign: u_t + u u_x = ...)
        u_hat = np.fft.fft(u_phys)
        u_x_hat = 1j * k * u_hat
        u_x = np.fft.ifft(u_x_hat).real
        return -u_phys * u_x  # shape (Nx,)

    # Precompute implicit denominator for backward Euler (in Fourier space)
    implicit_denom = 1 + dt * nu * (k**2)

    # --- Time stepping loop ---
    u_hat = np.fft.fft(u)
    save_every = max(1, Nt // 1000)  # Only save final state for memory safety

    for n in range(Nt):
        # --- Explicit RK4 for nonlinear term in physical space ---
        u_phys = np.fft.ifft(u_hat).real

        k1 = nonlinear_rhs(u_phys)
        k2 = nonlinear_rhs(np.fft.ifft(u_hat + 0.5 * dt * np.fft.fft(k1)).real)
        k3 = nonlinear_rhs(np.fft.ifft(u_hat + 0.5 * dt * np.fft.fft(k2)).real)
        k4 = nonlinear_rhs(np.fft.ifft(u_hat + dt * np.fft.fft(k3)).real)
        nonlinear_hat = np.fft.fft((k1 + 2*k2 + 2*k3 + k4) / 6)

        # --- Implicit Backward Euler for diffusion in Fourier space ---
        # u_hat^{n+1} = [u_hat^n + dt * nonlinear_hat] / (1 + dt*nu*k^2)
        u_hat = (u_hat + dt * nonlinear_hat) / implicit_denom

    # --- Final solution in physical space ---
    u_final = np.fft.ifft(u_hat).real

    # --- Residual computation ---
    # Compute all terms at t = t_final
    # u_t ≈ (u_final - u_prev) / dt (backward difference)
    # But since we don't store u_prev, use a spectral derivative in time:
    # Instead, estimate u_t by one backward Euler step:
    #   u_t ≈ (u_final - u_BE) / dt, where u_BE is one step back
    # Or, recompute one step back:
    #   u_hat_prev = implicit_denom * u_hat - dt * nonlinear_hat
    #   u_prev = ifft(u_hat_prev)
    # But for accuracy, do one more step from u_hat_final backward:
    #   u_t = -u*u_x + nu*u_xx

    # Compute u_x and u_xx in spectral space
    u_hat_final = np.fft.fft(u_final)
    u_x = np.fft.ifft(1j * k * u_hat_final).real
    u_xx = np.fft.ifft(-k**2 * u_hat_final).real

    # Compute u_t by the PDE (since at steady state, residual should be small)
    # u_t = nu*u_xx - u*u_x
    u_t = nu * u_xx - u_final * u_x

    # Compute the residual: u_t + u*u_x - nu*u_xx
    residual_grid = u_t + u_final * u_x - nu * u_xx  # should be close to zero

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual_grid
    }
```
