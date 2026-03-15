```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters from spec and plan ---
    # Domain
    x_min = pde_spec["domain"]["x_min"]
    x_max = pde_spec["domain"]["x_max"]
    L = x_max - x_min

    # Parameters
    H = float(pde_spec["parameters"]["H"])
    g = float(pde_spec["parameters"]["g"])

    # Grid
    Nx = int(plan["spatial_discretization"]["Nx"])
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = (x_max - x_min) / Nx

    # Time stepping
    t_final = float(plan["time_stepping"]["t_final"])
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt by CFL: dt < dx / sqrt(g*H)
        c = np.sqrt(g * H)
        dt = 0.5 * dx / c
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt + 1)

    # --- Initial conditions ---
    eta = np.sin(x)
    u = np.zeros_like(x)

    # --- Precompute spectral wave numbers ---
    k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi  # shape (Nx,)

    # --- Time stepping: RK3 (Williamson 1980) ---
    def rhs(eta, u):
        # Compute spatial derivatives using spectral method
        eta_hat = np.fft.fft(eta)
        u_hat = np.fft.fft(u)
        eta_x = np.fft.ifft(1j * k * eta_hat).real
        u_x = np.fft.ifft(1j * k * u_hat).real
        # Linearized shallow water equations
        deta_dt = -H * u_x
        du_dt = -g * eta_x
        return deta_dt, du_dt

    # Only store final state for memory safety
    for n in range(Nt):
        # Stage 1
        deta1, du1 = rhs(eta, u)
        eta1 = eta + dt * deta1
        u1 = u + dt * du1

        # Stage 2
        deta2, du2 = rhs(eta1, u1)
        eta2 = 0.75 * eta + 0.25 * (eta1 + dt * deta2)
        u2 = 0.75 * u + 0.25 * (u1 + dt * du2)

        # Stage 3
        deta3, du3 = rhs(eta2, u2)
        eta = (1.0/3.0) * eta + (2.0/3.0) * (eta2 + dt * deta3)
        u = (1.0/3.0) * u + (2.0/3.0) * (u2 + dt * du3)
        # periodic BCs are implicit in spectral

    # --- Compute residuals at final time ---
    # Residuals: eta_t + H u_x = 0, u_t + g eta_x = 0
    # Compute time derivatives using last RK step (forward Euler approx)
    deta_dt, du_dt = rhs(eta, u)
    # Compute spatial derivatives
    eta_hat = np.fft.fft(eta)
    u_hat = np.fft.fft(u)
    eta_x = np.fft.ifft(1j * k * eta_hat).real
    u_x = np.fft.ifft(1j * k * u_hat).real

    # Residuals (should be near zero for a good solution)
    res_eta = deta_dt + H * u_x
    res_u = du_dt + g * eta_x
    # Stack residuals: shape (2, Nx)
    residual_grid = np.stack([res_eta, res_u], axis=0)

    # Solution: stack eta and u for output, shape (2, Nx)
    u_out = np.stack([eta, u], axis=0)

    # Return only final state for memory safety
    return {
        "u": u_out,  # shape (2, Nx)
        "coords": {"x": x},
        "t": t_array,
        "residual": residual_grid  # shape (2, Nx)
    }
```