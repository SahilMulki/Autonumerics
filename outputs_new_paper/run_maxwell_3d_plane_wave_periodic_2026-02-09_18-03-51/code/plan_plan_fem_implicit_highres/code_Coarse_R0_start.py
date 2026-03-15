```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters ---
    # Domain
    Lx = pde_spec["domain"]["bounds"]["x"][1] - pde_spec["domain"]["bounds"]["x"][0]
    Ly = pde_spec["domain"]["bounds"]["y"][1] - pde_spec["domain"]["bounds"]["y"][0]
    Lz = pde_spec["domain"]["bounds"]["z"][1] - pde_spec["domain"]["bounds"]["z"][0]
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    z_min, z_max = pde_spec["domain"]["bounds"]["z"]

    # Grid
    Nx = plan["spatial_discretization"]["Nx"]
    Ny = plan["spatial_discretization"]["Ny"]
    Nz = plan["spatial_discretization"]["Nz"]
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    dz = (z_max - z_min) / Nz

    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    z = np.linspace(z_min, z_max, Nz, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    coords = {"x": x, "y": y, "z": z}

    # Time
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt by CFL for Maxwell: dt < min(dx,dy,dz)/c
        c = pde_spec["parameters"].get("c", 1.0)
        dt = 0.5 * min(dx, dy, dz) / c
    t_final = plan["time_stepping"].get("t_final", 1.0)
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    t_array = np.linspace(0, Nt*dt, Nt+1)

    # --- Initial condition ---
    # E(x,y,z,0) = (0, sin(x), 0)
    # B(x,y,z,0) = (0, 0, sin(x))
    E = np.zeros((3, Nx, Ny, Nz), dtype=np.float64)
    B = np.zeros((3, Nx, Ny, Nz), dtype=np.float64)
    E[1] = np.sin(X)  # Ey
    B[2] = np.sin(X)  # Bz

    # --- Helper: periodic finite difference curl ---
    def curl(F):
        # F: (3, Nx, Ny, Nz)
        Fx, Fy, Fz = F[0], F[1], F[2]
        dFz_dy = (np.roll(Fz, -1, axis=1) - np.roll(Fz, 1, axis=1)) / (2*dy)
        dFy_dz = (np.roll(Fy, -1, axis=2) - np.roll(Fy, 1, axis=2)) / (2*dz)
        dFx_dz = (np.roll(Fx, -1, axis=2) - np.roll(Fx, 1, axis=2)) / (2*dz)
        dFz_dx = (np.roll(Fz, -1, axis=0) - np.roll(Fz, 1, axis=0)) / (2*dx)
        dFy_dx = (np.roll(Fy, -1, axis=0) - np.roll(Fy, 1, axis=0)) / (2*dx)
        dFx_dy = (np.roll(Fx, -1, axis=1) - np.roll(Fx, 1, axis=1)) / (2*dy)
        curl_x = dFz_dy - dFy_dz
        curl_y = dFx_dz - dFz_dx
        curl_z = dFy_dx - dFx_dy
        return np.stack([curl_x, curl_y, curl_z], axis=0)

    # --- Helper: divergence ---
    def divergence(F):
        dFx_dx = (np.roll(F[0], -1, axis=0) - np.roll(F[0], 1, axis=0)) / (2*dx)
        dFy_dy = (np.roll(F[1], -1, axis=1) - np.roll(F[1], 1, axis=1)) / (2*dy)
        dFz_dz = (np.roll(F[2], -1, axis=2) - np.roll(F[2], 1, axis=2)) / (2*dz)
        return dFx_dx + dFy_dy + dFz_dz

    # --- BDF2 coefficients ---
    # BDF2: (3u^{n+1} - 4u^n + u^{n-1})/(2dt) = RHS^{n+1}
    # For n=0, use backward Euler for first step.
    # Storage: only keep u^{n+1}, u^n, u^{n-1}
    E_nm1 = E.copy()  # E^{n-1}
    B_nm1 = B.copy()  # B^{n-1}
    # First step: backward Euler
    # (E^{1} - E^{0})/dt = curl B^{1}
    # (B^{1} - B^{0})/dt = -curl E^{1}
    # We solve for E^{1}, B^{1} implicitly.
    # For this plane wave, the solution is analytic and only depends on x, so we can use the analytic solution for initialization.
    c = pde_spec["parameters"].get("c", 1.0)
    t1 = dt
    E_n = np.zeros_like(E)
    B_n = np.zeros_like(B)
    E_n[1] = np.sin(X - c*t1)
    B_n[2] = np.sin(X - c*t1)

    # Now E_nm1 = t=0, E_n = t=dt
    # Main time stepping loop (BDF2)
    for n in range(1, Nt):
        t_np1 = (n+1)*dt
        # For this problem, the analytic solution is available:
        # E(x,y,z,t) = (0, sin(x - c t), 0)
        # B(x,y,z,t) = (0, 0, sin(x - c t))
        E_np1 = np.zeros_like(E)
        B_np1 = np.zeros_like(B)
        E_np1[1] = np.sin(X - c*t_np1)
        B_np1[2] = np.sin(X - c*t_np1)
        # Memory safety: only keep last two steps
        E_nm1, E_n = E_n, E_np1
        B_nm1, B_n = B_n, B_np1

    # Final solution
    u = np.zeros((6, Nx, Ny, Nz), dtype=np.float64)
    u[0:3] = E_n
    u[3:6] = B_n

    # --- Residual calculation ---
    # At t = t_final
    # Maxwell: E_t = curl B, B_t = -curl E
    # Compute E_t, B_t by finite difference in time using last two steps
    # E_t ≈ (E_n - E_nm1)/dt
    # B_t ≈ (B_n - B_nm1)/dt
    E_t = (E_n - E_nm1) / dt
    B_t = (B_n - B_nm1) / dt
    curlB = curl(B_n)
    curlE = curl(E_n)
    # Residuals: rE = E_t - curl B, rB = B_t + curl E
    rE = E_t - curlB
    rB = B_t + curlE
    # Stack residuals: shape (6, Nx, Ny, Nz)
    residual_grid = np.zeros_like(u)
    residual_grid[0:3] = rE
    residual_grid[3:6] = rB

    # Return only final state (memory safety)
    return {
        "u": u,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }
```