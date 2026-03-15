```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters from spec and plan ---
    # Domain
    Lx = pde_spec['domain']['bounds']['x'][1] - pde_spec['domain']['bounds']['x'][0]
    Ly = pde_spec['domain']['bounds']['y'][1] - pde_spec['domain']['bounds']['y'][0]
    Lz = pde_spec['domain']['bounds']['z'][1] - pde_spec['domain']['bounds']['z'][0]
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    z_min, z_max = pde_spec['domain']['bounds']['z']

    # Grid sizes
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    Nz = plan['spatial_discretization']['Nz']

    # Coordinates
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    z = np.linspace(z_min, z_max, Nz, endpoint=False)
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    dz = (z_max - z_min) / Nz

    # Time stepping
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    if dt is None:
        # Estimate dt by CFL: dt < min(dx,dy,dz)/c
        c = pde_spec['parameters'].get('c', 1.0)
        dt = 0.5 * min(dx, dy, dz) / c
    if t_final is None:
        Nt = plan['time_stepping'].get('Nt', 1000)
        t_final = Nt * dt
    else:
        Nt = int(np.ceil(t_final / dt))
    t_array = np.linspace(0, t_final, Nt+1)

    # --- 2. Spectral grid and dealiasing mask ---
    # Fourier wavenumbers
    kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
    kz = np.fft.fftfreq(Nz, d=dz) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    # Dealiasing mask (2/3 rule)
    def dealias_mask(N):
        cutoff = int(N // 3)
        mask = np.zeros(N, dtype=bool)
        mask[:cutoff] = True
        mask[-cutoff:] = True
        return mask
    mask_x = dealias_mask(Nx)
    mask_y = dealias_mask(Ny)
    mask_z = dealias_mask(Nz)
    dealias = np.outer(mask_x, np.ones(Ny, dtype=bool))[:, :, None] & \
              np.outer(np.ones(Nx, dtype=bool), mask_y)[None, :, :] & \
              mask_z[None, None, :]

    # --- 3. Initial condition ---
    # E(x,y,z,0) = (0, sin(x), 0)
    # B(x,y,z,0) = (0, 0, sin(x))
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    E0 = np.zeros((3, Nx, Ny, Nz), dtype=np.float64)
    B0 = np.zeros((3, Nx, Ny, Nz), dtype=np.float64)
    E0[1] = np.sin(X)
    B0[2] = np.sin(X)

    # --- 4. Transform to spectral space ---
    def fftn3(u):
        return np.fft.fftn(u, axes=(1,2,3))
    def ifftn3(u_hat):
        return np.fft.ifftn(u_hat, axes=(1,2,3)).real

    E_hat = fftn3(E0)
    B_hat = fftn3(B0)

    # --- 5. Spectral curl operator ---
    def curl_hat(F_hat):
        # F_hat: (3, Nx, Ny, Nz)
        Fx, Fy, Fz = F_hat[0], F_hat[1], F_hat[2]
        curl_x = 1j * (KY * Fz - KZ * Fy)
        curl_y = 1j * (KZ * Fx - KX * Fz)
        curl_z = 1j * (KX * Fy - KY * Fx)
        return np.stack([curl_x, curl_y, curl_z], axis=0)

    # --- 6. Time stepping: IMEX RK3 (linear terms implicit, nonlinear explicit) ---
    # Maxwell is linear, so IMEX is just RK3 for linear system.
    # For demonstration, use a strong-stability-preserving RK3 (SSP-RK3).
    # Since the system is linear, we can use the exact exponential for the linear part.
    # But for generality, implement a 3-stage IMEX RK:
    #   u^{n+1} = u^n + dt * L(u^n)
    # Here, L(u) = Maxwell operator in spectral space.

    c = pde_spec['parameters'].get('c', 1.0)

    def rhs_hat(E_hat, B_hat):
        # dE/dt = curl B
        # dB/dt = -curl E
        curlB_hat = curl_hat(B_hat)
        curlE_hat = curl_hat(E_hat)
        dE_hat = c * curlB_hat
        dB_hat = -c * curlE_hat
        return dE_hat, dB_hat

    # Memory safety: only store final state
    for n in range(Nt):
        # Stage 1
        dE1, dB1 = rhs_hat(E_hat, B_hat)
        E1_hat = E_hat + dt * dE1
        B1_hat = B_hat + dt * dB1

        # Stage 2
        dE2, dB2 = rhs_hat(E1_hat, B1_hat)
        E2_hat = 0.75 * E_hat + 0.25 * (E1_hat + dt * dE2)
        B2_hat = 0.75 * B_hat + 0.25 * (B1_hat + dt * dB2)

        # Stage 3
        dE3, dB3 = rhs_hat(E2_hat, B2_hat)
        E_hat = (1/3) * E_hat + (2/3) * (E2_hat + dt * dE3)
        B_hat = (1/3) * B_hat + (2/3) * (B2_hat + dt * dB3)

        # Dealiasing
        E_hat[:, ~dealias] = 0
        B_hat[:, ~dealias] = 0

    # --- 7. Transform back to physical space ---
    E = ifftn3(E_hat)
    B = ifftn3(B_hat)
    # Solution: shape (6, Nx, Ny, Nz)
    u = np.zeros((6, Nx, Ny, Nz), dtype=np.float64)
    u[0:3] = E
    u[3:6] = B

    # --- 8. Compute residual ---
    # PDE: E_t = c curl B, B_t = -c curl E
    # At final time, approximate E_t, B_t by backward difference
    #   E_t ≈ (E^n - E^{n-1})/dt
    #   B_t ≈ (B^n - B^{n-1})/dt
    # So, do one step backward to get E_prev, B_prev
    # (This is cheap: just one step of RK3 backward in time)
    # Actually, we can store E_hat_prev, B_hat_prev during last step
    # But since we didn't, do one step backward using -dt
    # (Note: this is not exact, but sufficient for residual estimation)

    # One step backward in time
    def rk3_step(E_hat, B_hat, dt):
        # Stage 1
        dE1, dB1 = rhs_hat(E_hat, B_hat)
        E1_hat = E_hat + dt * dE1
        B1_hat = B_hat + dt * dB1

        # Stage 2
        dE2, dB2 = rhs_hat(E1_hat, B1_hat)
        E2_hat = 0.75 * E_hat + 0.25 * (E1_hat + dt * dE2)
        B2_hat = 0.75 * B_hat + 0.25 * (B1_hat + dt * dB2)

        # Stage 3
        dE3, dB3 = rhs_hat(E2_hat, B2_hat)
        E_hat_new = (1/3) * E_hat + (2/3) * (E2_hat + dt * dE3)
        B_hat_new = (1/3) * B_hat + (2/3) * (B2_hat + dt * dB3)
        return E_hat_new, B_hat_new

    E_hat_prev, B_hat_prev = rk3_step(E_hat, B_hat, -dt)
    E_prev = ifftn3(E_hat_prev)
    B_prev = ifftn3(B_hat_prev)

    E_t = (E - E_prev) / dt
    B_t = (B - B_prev) / dt

    # Compute curl B, curl E in physical space
    # (Best to do in spectral space for accuracy)
    curlB_hat = curl_hat(B_hat)
    curlE_hat = curl_hat(E_hat)
    curlB = ifftn3(curlB_hat)
    curlE = ifftn3(curlE_hat)

    # Residuals: E_t - c curl B, B_t + c curl E
    res_E = E_t - c * curlB
    res_B = B_t + c * curlE

    # Stack to shape (6, Nx, Ny, Nz)
    residual_grid = np.zeros_like(u)
    residual_grid[0:3] = res_E
    residual_grid[3:6] = res_B

    # --- 9. Output ---
    coords = {'x': x, 'y': y, 'z': z}
    return {
        "u": u,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }
```
