```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    Lx = pde_spec["domain"]["bounds"]["x"][1] - pde_spec["domain"]["bounds"]["x"][0]
    Ly = pde_spec["domain"]["bounds"]["y"][1] - pde_spec["domain"]["bounds"]["y"][0]
    x_min = pde_spec["domain"]["bounds"]["x"][0]
    y_min = pde_spec["domain"]["bounds"]["y"][0]
    x_max = pde_spec["domain"]["bounds"]["x"][1]
    y_max = pde_spec["domain"]["bounds"]["y"][1]
    c_x = float(pde_spec["parameters"]["c_x"])
    c_y = float(pde_spec["parameters"]["c_y"])
    nu = float(pde_spec["parameters"]["nu"])

    # --- Extract plan parameters ---
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    Nt = plan["time_stepping"].get("Nt", None)

    # Estimate dt if not given (CFL for convection-diffusion, spectral)
    if dt is None:
        dx = Lx / Nx
        dy = Ly / Ny
        cfl = 0.4
        dt_adv = cfl * min(dx/abs(c_x) if c_x != 0 else np.inf,
                           dy/abs(c_y) if c_y != 0 else np.inf)
        dt_diff = cfl * min(dx*dx, dy*dy) / (4*nu) if nu > 0 else np.inf
        dt = min(dt_adv, dt_diff)
    # Compute Nt if not given
    if Nt is None:
        if t_final is None:
            raise ValueError("Either Nt or t_final must be specified.")
        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt  # adjust dt to hit t_final exactly
    else:
        if t_final is None:
            t_final = Nt * dt

    # --- Set up spatial grid ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    dx = Lx / Nx
    dy = Ly / Ny
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial condition ---
    u0 = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    u_hat = np.fft.fft2(u0)

    # --- Spectral wavenumbers ---
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    # --- Time stepping (RK4) ---
    def rhs(u_hat):
        # u_hat: spectral coefficients
        # Compute derivatives in spectral space
        u = np.fft.ifft2(u_hat).real
        u_x_hat = 1j * KX * u_hat
        u_y_hat = 1j * KY * u_hat
        u_xx_hat = -KX**2 * u_hat
        u_yy_hat = -KY**2 * u_hat

        # Convection terms
        conv_x_hat = c_x * u_x_hat
        conv_y_hat = c_y * u_y_hat

        # Diffusion terms
        diff_hat = nu * (u_xx_hat + u_yy_hat)

        # RHS in spectral space: -convection + diffusion
        rhs_hat = -conv_x_hat - conv_y_hat + diff_hat
        return rhs_hat

    t = 0.0
    for n in range(Nt):
        k1 = rhs(u_hat)
        k2 = rhs(u_hat + 0.5 * dt * k1)
        k3 = rhs(u_hat + 0.5 * dt * k2)
        k4 = rhs(u_hat + dt * k3)
        u_hat = u_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t += dt

    # Final solution in physical space
    u = np.fft.ifft2(u_hat).real

    # --- Compute residual grid ---
    # u_t ≈ (u - u_prev) / dt, but since we only have u at final time, use PDE directly:
    # Compute all terms at final time
    # u_x, u_y, u_xx, u_yy via spectral derivatives
    u_hat_final = np.fft.fft2(u)
    u_x = np.fft.ifft2(1j * KX * u_hat_final).real
    u_y = np.fft.ifft2(1j * KY * u_hat_final).real
    u_xx = np.fft.ifft2(-KX**2 * u_hat_final).real
    u_yy = np.fft.ifft2(-KY**2 * u_hat_final).real

    # Approximate u_t using backward difference (since we don't store all time steps)
    # We'll do a single backward Euler step for residual: u_t ≈ (u - u_prev) / dt
    # To get u_prev, step backward one dt using explicit Euler (approximate)
    u_hat_prev = u_hat - dt * rhs(u_hat)
    u_prev = np.fft.ifft2(u_hat_prev).real
    u_t = (u - u_prev) / dt

    # Residual: u_t + c_x u_x + c_y u_y - nu(u_xx + u_yy)
    residual = u_t + c_x * u_x + c_y * u_y - nu * (u_xx + u_yy)

    # --- Output ---
    t_array = np.array([t])  # Only final time
    coords = {"x": x, "y": y}
    return {
        "u": u,
        "coords": coords,
        "t": t_array,
        "residual": residual
    }
```
