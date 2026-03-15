```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    # Grid
    Nx = plan["spatial_discretization"]["Nx"]
    Ny = plan["spatial_discretization"]["Ny"]
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    coords = {"x": x, "y": y}
    # Time
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    Nt = plan["time_stepping"].get("Nt", None)
    if dt is None:
        # Estimate dt by CFL for diffusion (D_u largest)
        D_u = pde_spec["parameters"]["D_u"]
        dt = 0.9 * min(dx, dy)**2 / (4 * D_u)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    else:
        t_final = Nt * dt
    t_array = np.linspace(0, t_final, Nt+1)
    # Parameters
    D_u = pde_spec["parameters"]["D_u"]
    D_v = pde_spec["parameters"]["D_v"]
    F = pde_spec["parameters"]["F"]
    k = pde_spec["parameters"]["k"]

    # --- Initial condition ---
    u = np.ones((Nx, Ny), dtype=np.float64)
    v = np.zeros((Nx, Ny), dtype=np.float64)
    # Patch: small square centered at (0.5, 0.5)
    patch_size = 0.1  # width of patch
    x0, y0 = 0.5, 0.5
    x1, x2 = x0 - patch_size/2, x0 + patch_size/2
    y1, y2 = y0 - patch_size/2, y0 + patch_size/2
    X, Y = np.meshgrid(x, y, indexing='ij')
    patch = (X >= x1) & (X < x2) & (Y >= y1) & (Y < y2)
    u[patch] = 0.5
    v[patch] = 0.25

    # --- Helper: Laplacian with periodic BCs, 2D, 2nd order FD ---
    def laplacian(f):
        # f: (Nx, Ny)
        return (
            (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / dx**2 +
            (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / dy**2
        )

    # --- Helper: Implicit solve (ADI not used, use bicgstab on sparse) ---
    # For periodic BCs, the Laplacian is circulant, so we can use FFT for implicit solve
    def implicit_diffusion_fft(u0, D, dt):
        # Solve (I - dt*D*L) u = rhs, where L is Laplacian with periodic BCs
        # Use FFT diagonalization
        u_hat = np.fft.fft2(u0)
        kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        lap_eig = - (KX**2 + KY**2)
        denom = 1 - dt * D * lap_eig
        # Avoid division by zero at (0,0) (DC mode, but that's fine here)
        u1_hat = u_hat / denom
        u1 = np.fft.ifft2(u1_hat).real
        return u1

    # --- Time stepping: IMEX Euler ---
    # Only store final state for memory safety
    for n in range(Nt):
        # Explicit reaction
        uv2 = u * v * v
        u_react = -uv2 + F * (1 - u)
        v_react = uv2 - (F + k) * v
        u_rhs = u + dt * u_react
        v_rhs = v + dt * v_react
        # Implicit diffusion (FFT solve)
        u = implicit_diffusion_fft(u_rhs, D_u, dt)
        v = implicit_diffusion_fft(v_rhs, D_v, dt)
        # No need to enforce BCs: periodic handled by FFT

    # --- Compute residual at final state ---
    # u_t ≈ (u_final - u_prev) / dt, but since we don't have u_prev, use PDE at final state
    # Compute pointwise residuals:
    # residual_u = u_t - [D_u * lap(u) - u*v^2 + F*(1-u)]
    # residual_v = v_t - [D_v * lap(v) + u*v^2 - (F+k)*v]
    # Since we don't have u_t, we can compute the steady-state residual (should be small if converged)
    lap_u = laplacian(u)
    lap_v = laplacian(v)
    uv2 = u * v * v
    residual_u = D_u * lap_u - uv2 + F * (1 - u)
    residual_v = D_v * lap_v + uv2 - (F + k) * v
    # Stack residuals as (2, Nx, Ny)
    residual_grid = np.stack([residual_u, residual_v], axis=0)
    # Solution: stack as (2, Nx, Ny)
    u_out = np.stack([u, v], axis=0)

    return {
        "u": u_out,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }
```
