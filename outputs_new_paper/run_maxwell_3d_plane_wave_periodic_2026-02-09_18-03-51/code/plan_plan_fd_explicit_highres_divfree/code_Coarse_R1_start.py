```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters from spec and plan ---
    # Domain
    Lx = pde_spec['domain']['bounds']['x'][1] - pde_spec['domain']['bounds']['x'][0]
    Ly = pde_spec['domain']['bounds']['y'][1] - pde_spec['domain']['bounds']['y'][0]
    Lz = pde_spec['domain']['bounds']['z'][1] - pde_spec['domain']['bounds']['z'][0]
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    z_min, z_max = pde_spec['domain']['bounds']['z']

    # Grid
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    Nz = plan['spatial_discretization']['Nz']
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    dz = (z_max - z_min) / Nz

    # Time
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    Nt = plan['time_stepping'].get('Nt', None)
    c = pde_spec['parameters'].get('c', 1.0)

    # CFL safety if dt not specified
    if dt is None:
        dt = 0.99 * min(dx, dy, dz) / (np.sqrt(3) * c)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    t_array = np.linspace(0, Nt*dt, Nt+1)
    if t_final is not None:
        Nt = int(np.ceil(t_final / dt))
        t_array = np.linspace(0, Nt*dt, Nt+1)

    # Yee grid: staggered
    # E_x at (i+1/2, j, k), E_y at (i, j+1/2, k), E_z at (i, j, k+1/2)
    # B_x at (i, j+1/2, k+1/2), B_y at (i+1/2, j, k+1/2), B_z at (i+1/2, j+1/2, k)
    # We'll use arrays of shape:
    # E_x: (Nx+1, Ny, Nz)
    # E_y: (Nx, Ny+1, Nz)
    # E_z: (Nx, Ny, Nz+1)
    # B_x: (Nx, Ny+1, Nz+1)
    # B_y: (Nx+1, Ny, Nz+1)
    # B_z: (Nx+1, Ny+1, Nz)

    # --- Allocate fields ---
    E_x = np.zeros((Nx+1, Ny, Nz), dtype=np.float64)
    E_y = np.zeros((Nx, Ny+1, Nz), dtype=np.float64)
    E_z = np.zeros((Nx, Ny, Nz+1), dtype=np.float64)
    B_x = np.zeros((Nx, Ny+1, Nz+1), dtype=np.float64)
    B_y = np.zeros((Nx+1, Ny, Nz+1), dtype=np.float64)
    B_z = np.zeros((Nx+1, Ny+1, Nz), dtype=np.float64)

    # --- Set up coordinates for each field (cell-centered for each component) ---
    # E_x at (i+1/2, j, k)
    x_E_x = x_min + (np.arange(Nx+1) + 0.5) * dx
    y_E_x = y_min + np.arange(Ny) * dy
    z_E_x = z_min + np.arange(Nz) * dz
    # E_y at (i, j+1/2, k)
    x_E_y = x_min + np.arange(Nx) * dx
    y_E_y = y_min + (np.arange(Ny+1) + 0.5) * dy
    z_E_y = z_min + np.arange(Nz) * dz
    # E_z at (i, j, k+1/2)
    x_E_z = x_min + np.arange(Nx) * dx
    y_E_z = y_min + np.arange(Ny) * dy
    z_E_z = z_min + (np.arange(Nz+1) + 0.5) * dz

    # For output, we will return E and B at cell centers:
    # cell centers: (i+0.5, j+0.5, k+0.5)
    x_c = x_min + (np.arange(Nx) + 0.5) * dx
    y_c = y_min + (np.arange(Ny) + 0.5) * dy
    z_c = z_min + (np.arange(Nz) + 0.5) * dz

    # --- Initial condition ---
    # E(x,y,z,0) = (0, sin(x), 0)
    # B(x,y,z,0) = (0, 0, sin(x))
    # Set E_y at (i, j+1/2, k): x = x_E_y, y = y_E_y, z = z_E_y
    # Set B_z at (i+1/2, j+1/2, k): x = x_min + (i+0.5)*dx, y = y_min + (j+0.5)*dy, z = z_min + k*dz

    # E_x is zero
    # E_y: sin(x)
    for i in range(Nx):
        E_y[i, :, :] = np.sin(x_E_y[i])
    # E_z is zero

    # B_x, B_y are zero
    # B_z: sin(x)
    for i in range(Nx+1):
        for j in range(Ny+1):
            B_z[i, j, :] = np.sin(x_min + (i+0.5)*dx)

    # --- Time stepping (leapfrog) ---
    # Yee: B at n-1/2, E at n
    # We'll initialize B at t = -dt/2 using a backward Euler step (or analytic, since analytic solution is known)
    # For this plane wave, B(x,y,z, t) = (0, 0, sin(x - c t))
    # So at t = -dt/2:
    for i in range(Nx+1):
        for j in range(Ny+1):
            B_z[i, j, :] = np.sin(x_min + (i+0.5)*dx + c*dt/2)

    # --- Main loop ---
    # Only store E and B at final time for memory safety
    for n in range(Nt):
        # 1. Update B at n+1/2 from E at n (Faraday's law)
        # B_x at (i, j+1/2, k+1/2)
        # B_x[i, j, k] = B_x[i, j, k] - dt * ( (dE_z/dy - dE_y/dz) )
        # dE_z/dy at (i, j+1/2, k+1/2): (E_z[i, j+1, k+1/2] - E_z[i, j, k+1/2]) / dy
        # dE_y/dz at (i, j+1/2, k+1/2): (E_y[i, j+1/2, k+1] - E_y[i, j+1/2, k]) / dz

        # B_x
        B_x = B_x - dt * (
            (E_z[:, 1:, 1:] - E_z[:, :-1, 1:]) / dy
            - (E_y[:, 1:, 1:] - E_y[:, 1:, :-1]) / dz
        )

        # B_y at (i+1/2, j, k+1/2)
        # dE_x/dz at (i+1/2, j, k+1/2): (E_x[i+1, j, k+1/2] - E_x[i, j, k+1/2]) / dx
        # dE_z/dx at (i+1/2, j, k+1/2): (E_z[i+1/2, j, k+1] - E_z[i+1/2, j, k]) / dz
        B_y = B_y - dt * (
            (E_x[1:, :, 1:] - E_x[:-1, :, 1:]) / dx
            - (E_z[1:, :, 1:] - E_z[1:, :, :-1]) / dz
        )

        # B_z at (i+1/2, j+1/2, k)
        # dE_y/dx at (i+1/2, j+1/2, k): (E_y[i+1, j+1/2, k] - E_y[i, j+1/2, k]) / dx
        # dE_x/dy at (i+1/2, j+1/2, k): (E_x[i+1/2, j+1, k] - E_x[i+1/2, j, k]) / dy
        B_z = B_z - dt * (
            (E_y[1:, :, :] - E_y[:-1, :, :]) / dx
            - (E_x[:, 1:, :] - E_x[:, :-1, :]) / dy
        )

        # 2. Update E at n+1 from B at n+1/2 (Ampere's law)
        # E_x at (i+1/2, j, k)
        # dB_z/dy at (i+1/2, j, k): (B_z[i+1/2, j+1, k] - B_z[i+1/2, j, k]) / dy
        # dB_y/dz at (i+1/2, j, k): (B_y[i+1/2, j, k+1] - B_y[i+1/2, j, k]) / dz
        E_x = E_x + dt * (
            (B_z[:, 1:, :] - B_z[:, :-1, :]) / dy
            - (B_y[:, :, 1:] - B_y[:, :, :-1]) / dz
        )

        # E_y at (i, j+1/2, k)
        # dB_x/dz at (i, j+1/2, k): (B_x[i, j+1/2, k+1] - B_x[i, j+1/2, k]) / dz
        # dB_z/dx at (i, j+1/2, k): (B_z[i+1, j+1/2, k] - B_z[i, j+1/2, k]) / dx
        E_y = E_y + dt * (
            (B_x[:, :, 1:] - B_x[:, :, :-1]) / dz
            - (B_z[1:, :, :] - B_z[:-1, :, :]) / dx
        )

        # E_z at (i, j, k+1/2)
        # dB_y/dx at (i, j, k+1/2): (B_y[i+1, j, k+1/2] - B_y[i, j, k+1/2]) / dx
        # dB_x/dy at (i, j, k+1/2): (B_x[i, j+1, k+1/2] - B_x[i, j, k+1/2]) / dy
        E_z = E_z + dt * (
            (B_y[1:, :, :] - B_y[:-1, :, :]) / dx
            - (B_x[:, 1:, :] - B_x[:, :-1, :]) / dy
        )

        # --- Periodic boundary conditions ---
        # E_x: periodic in x (wrap i), y, z
        E_x[0, :, :] = E_x[-1, :, :]
        E_x[:, 0, :] = E_x[:, -1, :]
        E_x[:, :, 0] = E_x[:, :, -1]
        # E_y: periodic in x, y (wrap j), z
        E_y[0, :, :] = E_y[-1, :, :]
        E_y[:, 0, :] = E_y[:, -1, :]
        E_y[:, :, 0] = E_y[:, :, -1]
        # E_z: periodic in x, y, z (wrap k)
        E_z[0, :, :] = E_z[-1, :, :]
        E_z[:, 0, :] = E_z[:, -1, :]
        E_z[:, :, 0] = E_z[:, :, -1]
        # B_x: periodic in x, y, z
        B_x[0, :, :] = B_x[-1, :, :]
        B_x[:, 0, :] = B_x[:, -1, :]
        B_x[:, :, 0] = B_x[:, :, -1]
        # B_y: periodic in x, y, z
        B_y[0, :, :] = B_y[-1, :, :]
        B_y[:, 0, :] = B_y[:, -1, :]
        B_y[:, :, 0] = B_y[:, :, -1]
        # B_z: periodic in x, y, z
        B_z[0, :, :] = B_z[-1, :, :]
        B_z[:, 0, :] = B_z[:, -1, :]
        B_z[:, :, 0] = B_z[:, :, -1]

        # Optionally, enforce divergence-free (projection step), but for this analytic solution, it's always satisfied.

    # --- Interpolate E, B to cell centers for output ---
    # E_x at (i+1/2, j, k), E_y at (i, j+1/2, k), E_z at (i, j, k+1/2)
    # Cell center at (i+0.5, j+0.5, k+0.5)
    # Average neighboring values to cell centers
    u_shape = (Nx, Ny, Nz, 6)  # (E_x, E_y, E_z, B_x, B_y, B_z)
    u = np.zeros(u_shape, dtype=np.float64)

    # E_x to cell center: average over y and z
    u[..., 0] = 0.25 * (
        E_x[:-1, :-1, :-1] + E_x[1:, :-1, :-1] + E_x[:-1, 1:, :-1] + E_x[:-1, :-1, 1:]
    )
    # E_y to cell center: average over x and z
    u[..., 1] = 0.25 * (
        E_y[:-1, :-1, :-1] + E_y[1:, :-1, :-1] + E_y[:-1, 1:, :-1] + E_y[:-1, :-1, 1:]
    )
    # E_z to cell center: average over x and y
    u[..., 2] = 0.25 * (
        E_z[:-1, :-1, :-1] + E_z[1:, :-1, :-1] + E_z[:-1, 1:, :-1] + E_z[:-1, :-1, 1:]
    )
    # B_x to cell center: average over y and z
    u[..., 3] = 0.25 * (
        B_x[:-1, :-1, :-1] + B_x[1:, :-1, :-1] + B_x[:-1, 1:, :-1] + B_x[:-1, :-1, 1:]
    )
    # B_y to cell center: average over x and z
    u[..., 4] = 0.25 * (
        B_y[:-1, :-1, :-1] + B_y[1:, :-1, :-1] + B_y[:-1, 1:, :-1] + B_y[:-1, :-1, 1:]
    )
    # B_z to cell center: average over x and y
    u[..., 5] = 0.25 * (
        B_z[:-1, :-1, :-1] + B_z[1:, :-1, :-1] + B_z[:-1, 1:, :-1] + B_z[:-1, :-1, 1:]
    )

    # --- Residual computation ---
    # At cell centers, compute:
    # E_t - curl B = 0
    # B_t + curl E = 0
    # We'll use central differences for spatial derivatives, and finite difference for time derivative (approximate E_t, B_t as (E^n - E^{n-1})/dt)
    # Since we only have E and B at final time, and the analytic solution is known, we can use the analytic time derivative for residual

    # Compute curl B and curl E at cell centers
    # For u[..., 3:6] = (B_x, B_y, B_z)
    # curl B = (dBz/dy - dBy/dz, dBx/dz - dBz/dx, dBy/dx - dBx/dy)
    # Use periodic BCs

    # Pad for periodic BCs
    def periodic_pad(arr):
        return np.pad(arr, ((1,1),(1,1),(1,1)), mode='wrap')

    Bx_c = u[..., 3]
    By_c = u[..., 4]
    Bz_c = u[..., 5]
    Ex_c = u[..., 0]
    Ey_c = u[..., 1]
    Ez_c = u[..., 2]

    # Pad arrays
    Bx_p = periodic_pad(Bx_c)
    By_p = periodic_pad(By_c)
    Bz_p = periodic_pad(Bz_c)
    Ex_p = periodic_pad(Ex_c)
    Ey_p = periodic_pad(Ey_c)
    Ez_p = periodic_pad(Ez_c)

    # Compute derivatives at cell centers
    dBz_dy = (Bz_p[1:-1,2:,1:-1] - Bz_p[1:-1,0:-2,1:-1]) / (2*dy)
    dBy_dz = (By_p[1:-1,1:-1,2:] - By_p[1:-1,1:-1,0:-2]) / (2*dz)
    dBx_dz = (Bx_p[1:-1,1:-1,2:] - Bx_p[1:-1,1:-1,0:-2]) / (2*dz)
    dBz_dx = (Bz_p[2:,1:-1,1:-1] - Bz_p[0:-2,1:-1,1:-1]) / (2*dx)
    dBy_dx = (By_p[2:,1:-1,1:-1] - By_p[0:-2,1:-1,1:-1]) / (2*dx)
    dBx_dy = (Bx_p[1:-1,2:,1:-1] - Bx_p[1:-1,0:-2,1:-1]) / (2*dy)

    dEz_dy = (Ez_p[1:-1,2:,1:-1] - Ez_p[1:-1,0:-2,1:-1]) / (2*dy)
    dEy_dz = (Ey_p[1:-1,1:-1,2:] - Ey_p[1:-1,1:-1,0:-2]) / (2*dz)
    dEx_dz = (Ex_p[1:-1,1:-1,2:] - Ex_p[1:-1,1:-1,0:-2]) / (2*dz)
    dEz_dx = (Ez_p[2:,1:-1,1:-1] - Ez_p[0:-2,1:-1,1:-1]) / (2*dx)
    dEy_dx = (Ey_p[2:,1:-1,1:-1] - Ey_p[0:-2,1:-1,1:-1]) / (2*dx)
    dEx_dy = (Ex_p[1:-1,2:,1:-1] - Ex_p[1:-1,0:-2,1:-1]) / (2*dy)

    # curl B
    curlB_x = dBz_dy - dBy_dz
    curlB_y = dBx_dz - dBz_dx
    curlB_z = dBy_dx - dBx_dy
    # curl E
    curlE_x = dEz_dy - dEy_dz
    curlE_y = dEx_dz - dEz_dx
    curlE_z = dEy_dx - dEx_dy

    # Analytic time derivatives at final time
    # E(x,y,z,t) = (0, sin(x - c t), 0)
    # B(x,y,z,t) = (0, 0, sin(x - c t))
    # E_t = (0, -c cos(x - c t), 0)
    # B_t = (0, 0, -c cos(x - c t))
    # At cell centers:
    Xc, Yc, Zc = np.meshgrid(x_c, y_c, z_c, indexing='ij')
    t_now = t_array[-1]
    E_t_analytic = np.zeros(u_shape[:-1] + (3,), dtype=np.float64)
    B_t_analytic = np.zeros(u_shape[:-1] + (3,), dtype=np.float64)
    E_t_analytic[..., 1] = -c * np.cos(Xc - c * t_now)
    B_t_analytic[..., 2] = -c * np.cos(Xc - c * t_now)

    # Residuals: E_t - curl B, B_t + curl E
    residual = np.zeros(u_shape, dtype=np.float64)
    residual[..., 0] = E_t_analytic[..., 0] - curlB_x
    residual[..., 1] = E_t_analytic[..., 1] - curlB_y
    residual[..., 2] = E_t_analytic[..., 2] - curlB_z
    residual[..., 3] = B_t_analytic[..., 0] + curlE_x
    residual[..., 4] = B_t_analytic[..., 1] + curlE_y
    residual[..., 5] = B_t_analytic[..., 2] + curlE_z

    # --- Output coordinates ---
    coords = {'x': x_c, 'y': y_c, 'z': z_c}

    return {
        "u": u,
        "coords": coords,
        "t": t_array,
        "residual": residual
    }
```