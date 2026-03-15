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
    if dt is None:
        # Estimate dt by CFL (diffusion)
        D_u = pde_spec["parameters"]["D_u"]
        D_v = pde_spec["parameters"]["D_v"]
        D = max(D_u, D_v)
        dt = 0.2 * min(dx, dy)**2 / D
    if t_final is not None:
        Nt = int(np.ceil(t_final / dt))
        t_array = np.linspace(0, Nt*dt, Nt+1)
    else:
        Nt = plan["time_stepping"]["Nt"]
        t_array = np.linspace(0, Nt*dt, Nt+1)
    # PDE parameters
    D_u = pde_spec["parameters"]["D_u"]
    D_v = pde_spec["parameters"]["D_v"]
    F = pde_spec["parameters"]["F"]
    k = pde_spec["parameters"]["k"]

    # --- Initial condition ---
    u = np.ones((Nx, Ny), dtype=np.float64)
    v = np.zeros((Nx, Ny), dtype=np.float64)
    # Patch: small square at center
    patch_size = Nx // 10  # 10% of domain
    cx, cy = Nx // 2, Ny // 2
    px0, px1 = cx - patch_size//2, cx + patch_size//2
    py0, py1 = cy - patch_size//2, cy + patch_size//2
    u[px0:px1, py0:py1] = 0.5
    v[px0:px1, py0:py1] = 0.25

    # --- Laplacian operator with periodic BCs ---
    def laplacian(Z):
        # 2D, periodic, central difference
        return (
            (np.roll(Z, +1, axis=0) + np.roll(Z, -1, axis=0) - 2*Z) / dx**2 +
            (np.roll(Z, +1, axis=1) + np.roll(Z, -1, axis=1) - 2*Z) / dy**2
        )

    # --- Crank-Nicolson step for coupled system ---
    # For memory: only keep current u, v (no time history)
    # For each step: solve (I - dt/2*L) U^{n+1} = (I + dt/2*L) U^n + dt*RHS
    # Here, L is the diffusion operator, RHS is the nonlinear/reaction part

    # Precompute Laplacian operator in Fourier space for periodic BCs
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    lap_eig = - (KX**2 + KY**2)  # eigenvalues of Laplacian

    # For Crank-Nicolson, the linear part is solved implicitly, nonlinear part is explicit
    # (I - dt/2*D*L) u^{n+1} = (I + dt/2*D*L) u^n + dt * N(u^n, v^n)
    # Use spectral method for the linear part (diagonal in Fourier space)
    def crank_nicolson_step(u, v, dt):
        # Reaction terms at n
        uv2 = u * v**2
        Ru = -uv2 + F * (1 - u)
        Rv = uv2 - (F + k) * v

        # Linear (diffusion) part in Fourier space
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)
        # Right-hand side
        rhs_u = np.fft.fft2(u + dt/2 * (D_u * laplacian(u) + Ru))
        rhs_v = np.fft.fft2(v + dt/2 * (D_v * laplacian(v) + Rv))
        # Denominator for implicit solve
        denom_u = 1 - dt/2 * D_u * lap_eig
        denom_v = 1 - dt/2 * D_v * lap_eig
        # Solve for next step in Fourier space
        u1_hat = (rhs_u) / denom_u
        v1_hat = (rhs_v) / denom_v
        # Back to real space
        u1 = np.fft.ifft2(u1_hat).real
        v1 = np.fft.ifft2(v1_hat).real

        # Reaction terms at n+1 (for CN: average with previous)
        uv2_1 = u1 * v1**2
        Ru1 = -uv2_1 + F * (1 - u1)
        Rv1 = uv2_1 - (F + k) * v1

        # Final CN update: average nonlinear terms
        u_rhs = u + dt/2 * (D_u * laplacian(u) + Ru) + dt/2 * (D_u * laplacian(u1) + Ru1)
        v_rhs = v + dt/2 * (D_v * laplacian(v) + Rv) + dt/2 * (D_v * laplacian(v1) + Rv1)
        # Solve again for u^{n+1}
        u1_hat = np.fft.fft2(u_rhs) / denom_u
        v1_hat = np.fft.fft2(v_rhs) / denom_v
        u1 = np.fft.ifft2(u1_hat).real
        v1 = np.fft.ifft2(v1_hat).real
        return u1, v1

    # --- Time stepping ---
    for n in range(Nt):
        u, v = crank_nicolson_step(u, v, dt)

    # --- Residual computation ---
    # Compute pointwise residuals at final time
    # For u: res_u = u_t - D_u*(u_xx + u_yy) + u*v^2 - F*(1-u)
    # For v: res_v = v_t - D_v*(v_xx + v_yy) - u*v^2 + (F+k)*v
    # Estimate u_t, v_t by backward difference
    u_prev, v_prev = crank_nicolson_step(u, v, -dt)  # step backward
    u_t = (u - u_prev) / dt
    v_t = (v - v_prev) / dt
    lap_u = laplacian(u)
    lap_v = laplacian(v)
    uv2 = u * v**2
    res_u = u_t - (D_u * lap_u - uv2 + F * (1 - u))
    res_v = v_t - (D_v * lap_v + uv2 - (F + k) * v)
    # Stack residuals as (2, Nx, Ny)
    residual_grid = np.stack([res_u, res_v], axis=0)

    # --- Output ---
    # Only final state (memory safe)
    u_out = np.stack([u, v], axis=0)  # shape (2, Nx, Ny)
    return {
        "u": u_out,
        "coords": coords,
        "t": t_array[-1:],
        "residual": residual_grid
    }
```
**Notes:**
- Only the final state is returned (shape (2, Nx, Ny)), not the full time history.
- The residual is a grid (shape (2, Nx, Ny)), as required.
- Crank-Nicolson is implemented via spectral (FFT) diagonalization for the diffusion operator, which is efficient and exact for periodic BCs.
- The nonlinear reaction terms are handled with a predictor-corrector (two-stage CN) for accuracy.
- The time array `t` is returned as a 1-element array (final time only, since only final state is stored).
- All arrays are NumPy ndarrays, and the function is memory safe for large grids and long times.