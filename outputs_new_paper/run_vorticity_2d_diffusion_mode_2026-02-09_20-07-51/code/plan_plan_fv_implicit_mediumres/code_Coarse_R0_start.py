```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and plan parameters ---
    # Domain
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    t_min, t_max = pde_spec['domain']['bounds']['t']
    nu = float(pde_spec['parameters']['nu'])

    # Grid
    Nx = int(plan['spatial_discretization']['Nx'])
    Ny = int(plan['spatial_discretization']['Ny'])
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny

    # Time
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', t_max)
    if dt is None:
        # Estimate dt by CFL for diffusion: dt <= 0.25 * min(dx,dy)^2 / nu
        dt = 0.25 * min(dx, dy)**2 / nu
    Nt = int(np.ceil((t_final - t_min) / dt))
    dt = (t_final - t_min) / Nt  # Adjust dt to land exactly on t_final
    t_array = np.linspace(t_min, t_final, Nt+1)

    # Coordinates (cell centers for finite volume)
    x = np.linspace(x_min + dx/2, x_max - dx/2, Nx)
    y = np.linspace(y_min + dy/2, y_max - dy/2, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial condition ---
    # omega(x, y, 0) = sin(2pi x) * sin(2pi y)
    u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    # --- Helper: periodic roll ---
    def periodic_roll(arr, shift, axis):
        return np.roll(arr, shift, axis=axis)

    # --- Assemble the implicit operator (A) for backward Euler ---
    # The update is: (I - dt*nu*L) u^{n+1} = u^n
    # L is the 2D Laplacian with periodic BCs, using 2nd order central diff (FV)
    # We'll use a matrix-free approach: define a function matvec(u) = (I - dt*nu*L)u

    alpha_x = nu * dt / dx**2
    alpha_y = nu * dt / dy**2

    def apply_A(u_vec):
        # u_vec: flattened (Nx*Ny,) array
        u_grid = u_vec.reshape((Nx, Ny))
        # 2D Laplacian with periodic BCs
        lap = (
            periodic_roll(u_grid, +1, axis=0) + periodic_roll(u_grid, -1, axis=0) - 2 * u_grid
        ) / dx**2 + (
            periodic_roll(u_grid, +1, axis=1) + periodic_roll(u_grid, -1, axis=1) - 2 * u_grid
        ) / dy**2
        res = u_grid - dt * nu * lap
        return res.ravel()

    # --- Linear solver: GMRES (matrix-free, Jacobi preconditioner) ---
    # For medium grid, Jacobi is sufficient and memory-safe
    def gmres(A_func, b, tol=1e-8, maxiter=200, restart=30, M_inv=None):
        # Simple restarted GMRES, matrix-free, for moderate size
        n = b.size
        x = np.zeros_like(b)
        r = b - A_func(x)
        if M_inv is not None:
            z = M_inv(r)
        else:
            z = r
        beta = np.linalg.norm(z)
        if beta < tol:
            return x
        V = np.zeros((n, restart+1))
        H = np.zeros((restart+1, restart))
        for outer in range(maxiter // restart + 1):
            V[:,0] = z / beta
            g = np.zeros(restart+1)
            g[0] = beta
            for j in range(restart):
                w = A_func(V[:,j])
                if M_inv is not None:
                    w = M_inv(w)
                for i in range(j+1):
                    H[i,j] = np.dot(V[:,i], w)
                    w = w - H[i,j]*V[:,i]
                H[j+1,j] = np.linalg.norm(w)
                if H[j+1,j] != 0 and j+1 < restart+1:
                    V[:,j+1] = w / H[j+1,j]
                # Solve least squares
                y, *_ = np.linalg.lstsq(H[:j+2,:j+1], g[:j+2], rcond=None)
                x_new = x + V[:,:j+1] @ y
                res = b - A_func(x_new)
                if np.linalg.norm(res) < tol:
                    return x_new
                if H[j+1,j] == 0:
                    break
            # Restart
            x = x_new
            r = b - A_func(x)
            if M_inv is not None:
                z = M_inv(r)
            else:
                z = r
            beta = np.linalg.norm(z)
            if beta < tol:
                return x
        return x

    # Jacobi preconditioner: diagonal of A
    diag_A = 1 + 2*alpha_x + 2*alpha_y
    def M_inv(v):
        return v / diag_A

    # --- Time stepping loop (only store final state) ---
    u_vec = u.ravel()
    for n in range(Nt):
        b = u_vec.copy()
        # Solve (I - dt*nu*L) u^{n+1} = u^n
        u_vec = gmres(apply_A, b, tol=1e-8, maxiter=100, restart=20, M_inv=M_inv)
    u = u_vec.reshape((Nx, Ny))

    # --- Compute residual grid at final time ---
    # Residual: R = u_t - nu*(u_xx + u_yy)
    # Approximate u_t by backward difference: (u^n - u^{n-1})/dt
    # For residual, we need u^{n-1} (previous step)
    # So, rerun one step backward to get u_prev
    # (This is memory-safe for a single step)
    # Step back:
    b_prev = u_vec.copy()
    u_prev_vec = gmres(apply_A, b_prev, tol=1e-8, maxiter=100, restart=20, M_inv=M_inv)
    u_prev = u_prev_vec.reshape((Nx, Ny))

    u_t = (u - u_prev) / dt
    # Laplacian at final time
    lap_u = (
        periodic_roll(u, +1, axis=0) + periodic_roll(u, -1, axis=0) - 2 * u
    ) / dx**2 + (
        periodic_roll(u, +1, axis=1) + periodic_roll(u, -1, axis=1) - 2 * u
    ) / dy**2
    residual = u_t - nu * lap_u

    # --- Return ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual
    }
```
