import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
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

    # Coordinates
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    z = np.linspace(z_min, z_max, Nz, endpoint=False)
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz

    # Time
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    Nt = plan["time_stepping"].get("Nt", None)
    if dt is None:
        # Estimate dt by CFL: dt < min(dx,dy,dz)/c
        c = pde_spec["parameters"].get("c", 1.0)
        dt = 0.5 * min(dx, dy, dz) / c
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    t_array = np.arange(Nt+1) * dt
    t_final = t_array[-1]

    # --- 2. Spectral grid (Fourier) ---
    kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
    kz = np.fft.fftfreq(Nz, d=dz) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')  # shape (Nx,Ny,Nz)

    # --- 3. Dealiasing mask (2/3 rule) ---
    def dealias_mask(N):
        cutoff = int(N // 3)
        mask = np.ones(N, dtype=bool)
        if cutoff > 0:
            mask[cutoff:-cutoff] = False
        return mask
    mask_x = dealias_mask(Nx)
    mask_y = dealias_mask(Ny)
    mask_z = dealias_mask(Nz)
    # Build 3D mask using broadcasting
    dealias = mask_x[:, None, None] & mask_y[None, :, None] & mask_z[None, None, :]

    # --- 4. Initial condition ---
    # E(x,y,z,0) = (0, sin(x), 0)
    # B(x,y,z,0) = (0, 0, sin(x))
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    E0 = np.zeros((3, Nx, Ny, Nz), dtype=np.float64)
    B0 = np.zeros((3, Nx, Ny, Nz), dtype=np.float64)
    E0[1] = np.sin(X)
    B0[2] = np.sin(X)

    # --- 5. FFT of initial fields ---
    E_hat = np.fft.fftn(E0, axes=(1,2,3))
    B_hat = np.fft.fftn(B0, axes=(1,2,3))

    # --- 6. Spectral curl operator ---
    def curl_hat(F_hat):
        # F_hat: (3, Nx, Ny, Nz)
        curlF_hat = np.zeros_like(F_hat)
        # curl_x = dFz/dy - dFy/dz
        curlF_hat[0] = 1j * (KY * F_hat[2] - KZ * F_hat[1])
        # curl_y = dFx/dz - dFz/dx
        curlF_hat[1] = 1j * (KZ * F_hat[0] - KX * F_hat[2])
        # curl_z = dFy/dx - dFx/dy
        curlF_hat[2] = 1j * (KX * F_hat[1] - KY * F_hat[0])
        return curlF_hat

    # --- 7. Time stepping (RK4) ---
    c = pde_spec["parameters"].get("c", 1.0)
    def dealias_field(F_hat):
        F_hat = F_hat.copy()
        F_hat[:, ~dealias] = 0
        return F_hat

    for n in range(Nt):
        # RHS for Maxwell: dE/dt = c * curl B, dB/dt = -c * curl E
        def rhs(E_hat, B_hat):
            curlB_hat = curl_hat(B_hat)
            curlE_hat = curl_hat(E_hat)
            dE_hat = c * curlB_hat
            dB_hat = -c * curlE_hat
            # Dealias after each RHS
            dE_hat = dealias_field(dE_hat)
            dB_hat = dealias_field(dB_hat)
            return dE_hat, dB_hat

        # RK4 steps
        E1, B1 = E_hat, B_hat
        k1_E, k1_B = rhs(E1, B1)
        E2 = E_hat + 0.5 * dt * k1_E
        B2 = B_hat + 0.5 * dt * k1_B
        k2_E, k2_B = rhs(E2, B2)
        E3 = E_hat + 0.5 * dt * k2_E
        B3 = B_hat + 0.5 * dt * k2_B
        k3_E, k3_B = rhs(E3, B3)
        E4 = E_hat + dt * k3_E
        B4 = B_hat + dt * k3_B
        k4_E, k4_B = rhs(E4, B4)
        E_hat = E_hat + (dt/6.0) * (k1_E + 2*k2_E + 2*k3_E + k4_E)
        B_hat = B_hat + (dt/6.0) * (k1_B + 2*k2_B + 2*k3_B + k4_B)
        # Dealias after each step
        E_hat = dealias_field(E_hat)
        B_hat = dealias_field(B_hat)

    # --- 8. Transform back to real space ---
    E = np.fft.ifftn(E_hat, axes=(1,2,3)).real
    B = np.fft.ifftn(B_hat, axes=(1,2,3)).real

    # --- 9. Compute residuals ---
    # PDE: E_t = c curl B, B_t = -c curl E
    # Approximate time derivative by backward difference (since we only have final state)
    # Use analytic solution for E_prev, B_prev at t = t_final - dt
    def analytic_fields(t):
        # E(x,y,z,t) = (0, sin(x - c t), 0)
        # B(x,y,z,t) = (0, 0, sin(x - c t))
        E = np.zeros((3, Nx, Ny, Nz), dtype=np.float64)
        B = np.zeros((3, Nx, Ny, Nz), dtype=np.float64)
        E[1] = np.sin(X - c * t)
        B[2] = np.sin(X - c * t)
        return E, B

    E_prev, B_prev = analytic_fields(t_final - dt)
    E_curr, B_curr = E, B

    # Time derivatives (backward difference)
    E_t = (E_curr - E_prev) / dt
    B_t = (B_curr - B_prev) / dt

    # Compute curl B and curl E at t_final
    # FFT of B_curr, E_curr
    E_hat_curr = np.fft.fftn(E_curr, axes=(1,2,3))
    B_hat_curr = np.fft.fftn(B_curr, axes=(1,2,3))
    curlB_hat = curl_hat(B_hat_curr)
    curlE_hat = curl_hat(E_hat_curr)
    curlB = np.fft.ifftn(curlB_hat, axes=(1,2,3)).real
    curlE = np.fft.ifftn(curlE_hat, axes=(1,2,3)).real

    # Residuals: shape (6, Nx, Ny, Nz)
    # residual[0:3] = E_t - c curl B
    # residual[3:6] = B_t + c curl E
    residual = np.zeros((6, Nx, Ny, Nz), dtype=np.float64)
    residual[0:3] = E_t - c * curlB
    residual[3:6] = B_t + c * curlE

    # --- 10. Output ---
    # Only return final state (memory safe)
    u = np.zeros((6, Nx, Ny, Nz), dtype=np.float64)
    u[0:3] = E
    u[3:6] = B

    coords = {"x": x, "y": y, "z": z}
    return {
        "u": u,
        "coords": coords,
        "t": t_array
    }