```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and plan parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    # Grid sizes
    Nx = plan["spatial_discretization"]["Nx"]
    Ny = plan["spatial_discretization"]["Ny"]
    # Parameters
    c_x = float(pde_spec["parameters"]["c_x"])
    c_y = float(pde_spec["parameters"]["c_y"])
    # Time stepping
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    else:
        t_final = Nt * dt
    # Coordinates
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    X, Y = np.meshgrid(x, y, indexing='ij')
    # Initial condition
    u0 = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    # --- Spectral setup ---
    kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi  # shape (Nx,)
    ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi  # shape (Ny,)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')  # shape (Nx, Ny)
    # --- IMEX RK3 coefficients (Kennedy & Carpenter 2003, ARS(2,3,2)) ---
    # For pure advection, all terms are linear, so IMEX is equivalent to explicit RK
    # But we implement a 3rd order explicit RK (Shu-Osher) for accuracy
    def rhs(u_hat):
        # du/dt = -c_x u_x - c_y u_y
        # In Fourier: u_x_hat = i*kx*u_hat, u_y_hat = i*ky*u_hat
        return -1j * (c_x * KX + c_y * KY) * u_hat

    # --- Time stepping ---
    u_hat = np.fft.fft2(u0)
    t = 0.0
    t_array = np.array([0.0, t_final])  # Only store initial and final time for memory safety

    for n in range(Nt):
        # Shu-Osher 3rd order explicit RK (suitable for linear advection)
        # Stage 1
        k1 = rhs(u_hat)
        u1 = u_hat + dt * k1
        # Stage 2
        k2 = rhs(u1)
        u2 = 0.75 * u_hat + 0.25 * (u1 + dt * k2)
        # Stage 3
        k3 = rhs(u2)
        u_hat = (1.0/3.0) * u_hat + (2.0/3.0) * (u2 + dt * k3)
        t += dt

    # --- Final solution ---
    u = np.fft.ifft2(u_hat).real  # shape (Nx, Ny)

    # --- Residual computation ---
    # Compute u_t ≈ (u_final - u_prev) / dt, but since we only have u_final, use spectral PDE directly:
    # PDE: u_t + c_x u_x + c_y u_y = 0
    # Compute u_x, u_y spectrally
    u_hat_final = np.fft.fft2(u)
    u_x = np.fft.ifft2(1j * KX * u_hat_final).real
    u_y = np.fft.ifft2(1j * KY * u_hat_final).real
    # Compute u_t using the PDE (since the equation is linear and time-independent coefficients)
    # At final time, u_t = -c_x u_x - c_y u_y
    u_t = -c_x * u_x - c_y * u_y
    # Compute the actual time derivative using finite difference (backward)
    # To do this, we need u at t_final - dt; recompute one step back:
    u_hat_prev = u_hat.copy()
    for _ in range(1):  # Step back one dt
        # Reverse step: u_hat_prev = (u_hat - (2.0/3.0) * dt * k3 - (2.0/3.0) * u2) / (1.0/3.0)
        # But that's messy; instead, just step forward from t_final-dt
        # So, rerun the time loop for one less step to get u_prev
        u_hat_tmp = np.fft.fft2(u0)
        t_tmp = 0.0
        for n in range(Nt-1):
            k1 = rhs(u_hat_tmp)
            u1 = u_hat_tmp + dt * k1
            k2 = rhs(u1)
            u2 = 0.75 * u_hat_tmp + 0.25 * (u1 + dt * k2)
            k3 = rhs(u2)
            u_hat_tmp = (1.0/3.0) * u_hat_tmp + (2.0/3.0) * (u2 + dt * k3)
            t_tmp += dt
        u_prev = np.fft.ifft2(u_hat_tmp).real
        u_t_fd = (u - u_prev) / dt
    # Residual: r = u_t_fd + c_x u_x + c_y u_y
    residual_grid = u_t_fd + c_x * u_x + c_y * u_y

    # --- Output ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual_grid
    }
```
**Notes:**
- Only the final state `u` is stored for memory safety.
- The residual is computed as the pointwise grid of `u_t + c_x u_x + c_y u_y` at the final time, using a finite-difference time derivative and spectral spatial derivatives.
- The time array contains only `[0, t_final]` for minimal memory.
- The code is robust and uses only NumPy.