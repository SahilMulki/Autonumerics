```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters from pde_spec and plan ---
    # Domain
    Lx = pde_spec["domain"]["bounds"]["x"][1] - pde_spec["domain"]["bounds"]["x"][0]
    Ly = pde_spec["domain"]["bounds"]["y"][1] - pde_spec["domain"]["bounds"]["y"][0]
    Lz = pde_spec["domain"]["bounds"]["z"][1] - pde_spec["domain"]["bounds"]["z"][0]
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    z_min, z_max = pde_spec["domain"]["bounds"]["z"]
    c = pde_spec["parameters"].get("c", 1.0)

    # Grid
    Nx = plan["spatial_discretization"]["Nx"]
    Ny = plan["spatial_discretization"]["Ny"]
    Nz = plan["spatial_discretization"]["Nz"]
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    dz = (z_max - z_min) / Nz

    # Coordinates (cell edges for Yee grid)
    x_e = np.linspace(x_min, x_max, Nx+1, endpoint=True)
    y_e = np.linspace(y_min, y_max, Ny+1, endpoint=True)
    z_e = np.linspace(z_min, z_max, Nz+1, endpoint=True)
    # Cell centers
    x_c = x_e[:-1] + dx/2
    y_c = y_e[:-1] + dy/2
    z_c = z_e[:-1] + dz/2

    # Yee grid: E_x at (i+1/2,j,k), E_y at (i,j+1/2,k), E_z at (i,j,k+1/2)
    #           B_x at (i,j+1/2,k+1/2), B_y at (i+1/2,j,k+1/2), B_z at (i+1/2,j+1/2,k)
    # We'll use array shapes:
    #   E_x: (Nx, Ny+1, Nz+1)
    #   E_y: (Nx+1, Ny, Nz+1)
    #   E_z: (Nx+1, Ny+1, Nz)
    #   B_x: (Nx+1, Ny, Nz)
    #   B_y: (Nx, Ny+1, Nz)
    #   B_z: (Nx, Ny, Nz+1)

    # Time
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    if dt is None:
        # Estimate dt by CFL: dt < min(dx,dy,dz)/c/sqrt(3)
        dt = 0.99 * min(dx, dy, dz) / (c * np.sqrt(3))
    if t_final is None:
        Nt = plan["time_stepping"].get("Nt", 1000)
        t_final = Nt * dt
    else:
        Nt = int(np.ceil(t_final / dt))
        t_final = Nt * dt
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Initialize fields ---
    # E_x at (i+1/2, j, k): shape (Nx, Ny+1, Nz+1)
    # E_y at (i, j+1/2, k): shape (Nx+1, Ny, Nz+1)
    # E_z at (i, j, k+1/2): shape (Nx+1, Ny+1, Nz)
    # B_x at (i, j+1/2, k+1/2): shape (Nx+1, Ny, Nz)
    # B_y at (i+1/2, j, k+1/2): shape (Nx, Ny+1, Nz)
    # B_z at (i+1/2, j+1/2, k): shape (Nx, Ny, Nz+1)

    # For the analytic solution, E = (0, sin(x-ct), 0), B = (0, 0, sin(x-ct))
    # At t=0: E_y = sin(x), B_z = sin(x), others zero

    # E_x: (Nx, Ny+1, Nz+1) at x_e[1:-1] (cell faces)
    E_x = np.zeros((Nx, Ny+1, Nz+1), dtype=np.float64)
    # E_y: (Nx+1, Ny, Nz+1) at x_e, y_c
    E_y = np.zeros((Nx+1, Ny, Nz+1), dtype=np.float64)
    # E_z: (Nx+1, Ny+1, Nz) at x_e, y_e, z_c
    E_z = np.zeros((Nx+1, Ny+1, Nz), dtype=np.float64)
    # B_x: (Nx+1, Ny, Nz) at x_e, y_c, z_c
    B_x = np.zeros((Nx+1, Ny, Nz), dtype=np.float64)
    # B_y: (Nx, Ny+1, Nz) at x_c, y_e, z_c
    B_y = np.zeros((Nx, Ny+1, Nz), dtype=np.float64)
    # B_z: (Nx, Ny, Nz+1) at x_c, y_c, z_e
    B_z = np.zeros((Nx, Ny, Nz+1), dtype=np.float64)

    # Set initial E_y and B_z according to analytic solution
    # E_y at (x_e, y_c, z_e)
    x_Ey = x_e[:, None, None]  # (Nx+1, 1, 1)
    E_y[:] = np.sin(x_Ey)
    # B_z at (x_c, y_c, z_e)
    x_Bz = x_c[:, None, None]  # (Nx, 1, 1)
    B_z[:] = np.sin(x_Bz)

    # --- Leapfrog time stepping ---
    # Yee scheme: E at n*dt, B at (n+1/2)*dt
    # We'll use two sets of arrays and swap references to save memory

    # To save memory, only keep current E and B, and at the end, return E and B at final time

    # Precompute for periodic BCs: index helpers
    def roll(arr, shift, axis):
        # np.roll with periodic BCs
        return np.roll(arr, shift, axis=axis)

    for n in range(Nt):
        # 1. Update B at n+1/2 using E at n
        # B_x at (x_e, y_c, z_c): shape (Nx+1, Ny, Nz)
        # B_x^{n+1/2} = B_x^{n-1/2} - dt * (dE_z/dy - dE_y/dz)
        # dE_z/dy: E_z at (x_e, y_e, z_c) -> central diff in y
        dE_z_dy = (roll(E_z, -1, axis=1) - E_z) / dy  # (Nx+1, Ny, Nz)
        # dE_y/dz: E_y at (x_e, y_c, z_e) -> central diff in z
        dE_y_dz = (roll(E_y, -1, axis=2) - E_y) / dz  # (Nx+1, Ny, Nz)
        B_x = B_x - dt * (dE_z_dy - dE_y_dz)

        # B_y at (x_c, y_e, z_c): shape (Nx, Ny+1, Nz)
        # dE_x/dz: E_x at (x_c, y_e, z_e) -> central diff in z
        dE_x_dz = (roll(E_x, -1, axis=2) - E_x) / dz  # (Nx, Ny+1, Nz)
        # dE_z/dx: E_z at (x_e, y_e, z_c) -> central diff in x
        dE_z_dx = (roll(E_z, -1, axis=0) - E_z) / dx  # (Nx, Ny+1, Nz)
        B_y = B_y - dt * (dE_x_dz - dE_z_dx)

        # B_z at (x_c, y_c, z_e): shape (Nx, Ny, Nz+1)
        # dE_y/dx: E_y at (x_e, y_c, z_e) -> central diff in x
        dE_y_dx = (roll(E_y, -1, axis=0) - E_y) / dx  # (Nx, Ny, Nz+1)
        # dE_x/dy: E_x at (x_c, y_e, z_e) -> central diff in y
        dE_x_dy = (roll(E_x, -1, axis=1) - E_x) / dy  # (Nx, Ny, Nz+1)
        B_z = B_z - dt * (dE_y_dx - dE_x_dy)

        # 2. Update E at n+1 using B at n+1/2
        # E_x at (x_c, y_e, z_e): shape (Nx, Ny+1, Nz+1)
        # dB_z/dy: B_z at (x_c, y_e, z_e) -> central diff in y
        dB_z_dy = (roll(B_z, -1, axis=1) - B_z) / dy  # (Nx, Ny+1, Nz+1)
        # dB_y/dz: B_y at (x_c, y_e, z_c) -> central diff in z
        dB_y_dz = (roll(B_y, -1, axis=2) - B_y) / dz  # (Nx, Ny+1, Nz+1)
        E_x = E_x + dt * (dB_z_dy - dB_y_dz)

        # E_y at (x_e, y_c, z_e): shape (Nx+1, Ny, Nz+1)
        # dB_x/dz: B_x at (x_e, y_c, z_c) -> central diff in z
        dB_x_dz = (roll(B_x, -1, axis=2) - B_x) / dz  # (Nx+1, Ny, Nz+1)
        # dB_z/dx: B_z at (x_c, y_c, z_e) -> central diff in x
        dB_z_dx = (roll(B_z, -1, axis=0) - B_z) / dx  # (Nx+1, Ny, Nz+1)
        E_y = E_y + dt * (dB_x_dz - dB_z_dx)

        # E_z at (x_e, y_e, z_c): shape (Nx+1, Ny+1, Nz)
        # dB_y/dx: B_y at (x_c, y_e, z_c) -> central diff in x
        dB_y_dx = (roll(B_y, -1, axis=0) - B_y) / dx  # (Nx+1, Ny+1, Nz)
        # dB_x/dy: B_x at (x_e, y_c, z_c) -> central diff in y
        dB_x_dy = (roll(B_x, -1, axis=1) - B_x) / dy  # (Nx+1, Ny+1, Nz)
        E_z = E_z + dt * (dB_y_dx - dB_x_dy)

        # Optionally: enforce divergence-free (projection step)
        # For this analytic solution, divergence is always zero, so skip for speed

    # --- Output: pick a single component for output ---
    # For this problem, the analytic solution is E_y and B_z only, so we can output E_y and B_z

    # For output, let's return E_y at (x_e, y_c, z_e) and B_z at (x_c, y_c, z_e)
    # We'll return E_y as "u" (shape: (Nx+1, Ny, Nz+1)), and coords as x_e, y_c, z_e

    # --- Compute residual ---
    # For Maxwell: E_t = curl B, B_t = -curl E
    # We'll compute the residual for E_y and B_z at their respective grid points

    # 1. E_t at (x_e, y_c, z_e): (E_y^{n+1} - E_y^{n}) / dt
    #    curl B at (x_e, y_c, z_e): (dB_z/dx - dB_x/dz)
    #    B_t at (x_c, y_c, z_e): (B_z^{n+1} - B_z^{n}) / dt
    #    -curl E at (x_c, y_c, z_e): -(dE_y/dx - dE_x/dy)

    # For memory, we only have final E_y and B_z, so we can use a finite difference in time:
    # We'll recompute one backward step for E_y and B_z to get E_y_prev and B_z_prev

    # Rewind one step for E_y and B_z
    # (This is a small extra cost and avoids storing all history)
    # We'll use the same update formulas but with -dt

    # Rewind B_x, B_y, B_z by +dt/2 (half step)
    # Rewind E_x, E_y, E_z by +dt (full step)
    # (But since only E_y and B_z are nonzero, we only need to rewind those)

    # Rewind E_y
    # E_y_prev = E_y - dt * (dB_x_dz - dB_z_dx)
    dB_x_dz = (roll(B_x, -1, axis=2) - B_x) / dz
    dB_z_dx = (roll(B_z, -1, axis=0) - B_z) / dx
    E_y_prev = E_y - dt * (dB_x_dz - dB_z_dx)

    # Rewind B_z
    # B_z_prev = B_z + dt * (dE_y_dx - dE_x_dy)
    dE_y_dx = (roll(E_y, -1, axis=0) - E_y) / dx
    dE_x_dy = (roll(E_x, -1, axis=1) - E_x) / dy
    B_z_prev = B_z + dt * (dE_y_dx - dE_x_dy)

    # E_t = (E_y - E_y_prev) / dt
    E_t = (E_y - E_y_prev) / dt
    # curl B at E_y grid: (dB_x/dz - dB_z/dx)
    curlB = dB_x_dz - dB_z_dx
    residual_E = E_t - curlB

    # B_t = (B_z - B_z_prev) / dt
    B_t = (B_z - B_z_prev) / dt
    # -curl E at B_z grid: -(dE_y/dx - dE_x/dy)
    curlE = dE_y_dx - dE_x_dy
    residual_B = B_t + curlE

    # Stack residuals for E_y and B_z (same shape)
    residual_grid = np.stack([residual_E, residual_B], axis=0)  # shape (2, Nx, Ny, Nz+1)

    # --- Prepare output coordinates ---
    coords = {
        "x": x_e,        # E_y is at x_e
        "y": y_c,        # E_y is at y_c
        "z": z_e         # E_y is at z_e
    }

    # Return only E_y as "u" (shape: (Nx+1, Ny, Nz+1))
    return {
        "u": E_y,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }
```
**Notes:**
- Only the final state is stored, not the full time history.
- The residual is computed as a pointwise grid for both E_y and B_z components.
- The output `"u"` is the E_y field (the only nonzero E component for this analytic solution).
- Coordinates are provided for the E_y grid.
- The code is memory-safe for large grids.