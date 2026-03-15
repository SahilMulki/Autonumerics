```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and Plan Parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    # Discretization
    Nx = plan["spatial_discretization"]["Nx"]
    Ny = plan["spatial_discretization"]["Ny"]
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    # Coordinates (cell centers for FVM)
    x = np.linspace(x_min + dx/2, x_max - dx/2, Nx)
    y = np.linspace(y_min + dy/2, y_max - dy/2, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    # Time
    t_final = plan["time_stepping"]["t_final"]
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt by CFL (for explicit, but here implicit, so just pick a reasonable value)
        alpha = pde_spec["parameters"]["alpha"]
        dt = 0.4 * min(dx, dy)**2 / (4*alpha)
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # Adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)
    # PDE parameter
    alpha = pde_spec["parameters"]["alpha"]

    # --- Initial Condition ---
    u = np.sin(np.pi * X) * np.sin(np.pi * Y)

    # --- Boundary Conditions (Dirichlet u=0) ---
    # For FVM, ghost cells are not needed; just enforce u=0 at boundaries after each step

    # --- Assemble Implicit Matrix (A) for Backward Euler: (I - dt*alpha*L)u^{n+1} = u^n ---
    # 5-point Laplacian, FVM, uniform grid, Dirichlet BCs
    N = Nx * Ny
    Ix = np.arange(Nx)
    Iy = np.arange(Ny)

    # Helper to convert 2D index to 1D
    def idx(i, j):
        return i * Ny + j

    # Build sparse matrix in COO format, then convert to dense (Nx*Ny=2500 is OK)
    rows = []
    cols = []
    data = []

    rx = alpha * dt / dx**2
    ry = alpha * dt / dy**2

    for i in range(Nx):
        for j in range(Ny):
            p = idx(i, j)
            # Center
            rows.append(p)
            cols.append(p)
            val = 1.0 + 2*rx + 2*ry
            # Boundary cells: Dirichlet u=0, so set diagonal to 1, RHS to 0
            if i == 0 or i == Nx-1 or j == 0 or j == Ny-1:
                data.append(1.0)
                continue
            # Interior
            data.append(val)
            # -x
            rows.append(p)
            cols.append(idx(i-1, j))
            data.append(-rx)
            # +x
            rows.append(p)
            cols.append(idx(i+1, j))
            data.append(-rx)
            # -y
            rows.append(p)
            cols.append(idx(i, j-1))
            data.append(-ry)
            # +y
            rows.append(p)
            cols.append(idx(i, j+1))
            data.append(-ry)

    # Build dense matrix (N,N)
    A = np.zeros((N, N), dtype=np.float64)
    for r, c, v in zip(rows, cols, data):
        A[r, c] = v

    # --- Time Stepping (Backward Euler) ---
    u_flat = u.ravel()
    for n in range(Nt):
        # RHS: previous u, but Dirichlet BCs: set boundary points to 0
        u_flat_bc = u_flat.copy()
        # Set boundary points to 0
        u_bc = u_flat_bc.reshape((Nx, Ny))
        u_bc[0, :] = 0
        u_bc[-1, :] = 0
        u_bc[:, 0] = 0
        u_bc[:, -1] = 0
        u_flat_bc = u_bc.ravel()
        # Solve linear system
        u_flat_new = np.linalg.solve(A, u_flat_bc)
        # Enforce Dirichlet BCs explicitly (should already be 0, but for safety)
        u_new = u_flat_new.reshape((Nx, Ny))
        u_new[0, :] = 0
        u_new[-1, :] = 0
        u_new[:, 0] = 0
        u_new[:, -1] = 0
        u_flat = u_new.ravel()
    u = u_flat.reshape((Nx, Ny))

    # --- Compute Residual Grid at Final Time ---
    # Residual: R = u_t - alpha*(u_xx + u_yy)
    # Approximate u_t by backward difference: (u^n - u^{n-1})/dt
    # For residual, we need u^{n-1} as well. So do one step back:
    # (But only for the final step, so memory is safe.)
    # Compute u_prev (at t = t_final - dt)
    u_flat_prev = u.ravel()
    # Backward: u_prev = solve(A, u_prev), but we need the previous state.
    # So, step again from t = t_final - dt to t = t_final, but store both.
    # Instead, let's rerun the last step:
    # Start from u_prev, step to u (which we already have).
    # To get u_prev, we need to step backward, but that's not possible.
    # Instead, during the last time step, store u_prev.
    # So, let's rerun the last step:
    # First, reconstruct u_prev by stepping backward from u using the scheme:
    # (I - dt*alpha*L)u = u_prev  =>  u_prev = (I - dt*alpha*L)u
    # But that's not correct; we need u_prev to compute u_t.
    # Instead, during the last time step, store u_prev.
    # Let's do this:

    # Redo time stepping, but only store u_prev and u at the last step
    u_flat = np.sin(np.pi * X).ravel() * np.sin(np.pi * Y).ravel()
    u_prev_flat = None
    for n in range(Nt):
        u_flat_bc = u_flat.copy()
        u_bc = u_flat_bc.reshape((Nx, Ny))
        u_bc[0, :] = 0
        u_bc[-1, :] = 0
        u_bc[:, 0] = 0
        u_bc[:, -1] = 0
        u_flat_bc = u_bc.ravel()
        u_flat_new = np.linalg.solve(A, u_flat_bc)
        u_new = u_flat_new.reshape((Nx, Ny))
        u_new[0, :] = 0
        u_new[-1, :] = 0
        u_new[:, 0] = 0
        u_new[:, -1] = 0
        if n == Nt-2:
            u_prev_flat = u_flat.copy()
        u_flat = u_new.ravel()
    u = u_flat.reshape((Nx, Ny))
    u_prev = u_prev_flat.reshape((Nx, Ny))

    # u_t ≈ (u - u_prev) / dt
    u_t = (u - u_prev) / dt

    # Compute Laplacian using central differences (second order, FVM)
    u_xx = np.zeros_like(u)
    u_yy = np.zeros_like(u)
    # Interior points
    u_xx[1:-1, :] = (u[2:, :] - 2*u[1:-1, :] + u[0:-2, :]) / dx**2
    u_yy[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, 0:-2]) / dy**2
    lap = u_xx + u_yy

    # Residual grid
    residual = u_t - alpha * lap
    # Set residual to zero at boundaries (since BCs are enforced)
    residual[0, :] = 0
    residual[-1, :] = 0
    residual[:, 0] = 0
    residual[:, -1] = 0

    # --- Return ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual
    }
```