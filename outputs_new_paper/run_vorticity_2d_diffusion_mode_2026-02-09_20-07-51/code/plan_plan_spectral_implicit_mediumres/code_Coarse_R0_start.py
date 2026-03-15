```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters from pde_spec and plan ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    t_min, t_max = pde_spec["domain"]["bounds"]["t"]

    # Parameters
    nu = float(pde_spec["parameters"]["nu"])

    # Grid sizes
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])

    # Time stepping
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", t_max)
    if dt is None:
        # Estimate dt by CFL for diffusion: dt < dx^2/(4*nu)
        dx = (x_max - x_min) / Nx
        dy = (y_max - y_min) / Ny
        dt = 0.4 * min(dx, dy)**2 / (4 * nu)
    Nt = int(np.ceil((t_final - t_min) / dt))
    dt = (t_final - t_min) / Nt  # Adjust dt to hit t_final exactly

    # --- Build spatial grid ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial condition ---
    # omega(x, y, 0) = sin(2*pi*x) * sin(2*pi*y)
    u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    # --- Spectral wavenumbers ---
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=(x_max - x_min) / Nx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=(y_max - y_min) / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    laplacian_symbol = -(KX**2 + KY**2)

    # --- Precompute denominator for backward Euler ---
    # (I - dt*nu*L) in Fourier space: denominator = 1 - dt*nu*(-k^2)
    denom = 1 - dt * nu * laplacian_symbol

    # --- Time stepping loop (only keep final state for memory safety) ---
    u_hat = np.fft.fft2(u)
    for n in range(Nt):
        u_hat = u_hat / denom
    u = np.fft.ifft2(u_hat).real  # Final state

    # --- Time array (only initial and final for memory safety) ---
    t_array = np.array([t_min, t_max])

    # --- Compute residual at final time ---
    # omega_t ≈ (u_final - u_prev) / dt
    # For best accuracy, do one more backward Euler step from u_prev to u_final
    # So, recompute u_prev by stepping backward
    # u_prev_hat = denom * u_hat (since u_hat = u_prev_hat / denom)
    u_prev_hat = u_hat * denom
    u_prev = np.fft.ifft2(u_prev_hat).real

    omega_t = (u - u_prev) / dt

    # Compute Laplacian in spectral space
    u_hat_final = np.fft.fft2(u)
    lap_u_hat = laplacian_symbol * u_hat_final
    lap_u = np.fft.ifft2(lap_u_hat).real

    residual = omega_t - nu * lap_u  # Should be close to zero everywhere

    # --- Return only final state and required outputs ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual
    }
```
**Key points:**
- Only the final state is stored for memory safety.
- Residual is computed as the pointwise difference between the time derivative and the right-hand side at the final time.
- All parameters (Nx, Ny, dt, etc.) are dynamically read from the plan.
- Spectral (Fourier) method is used for periodic BCs.
- The time array contains only initial and final times for memory efficiency.