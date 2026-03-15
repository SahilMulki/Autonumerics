```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and plan parameters ---
    # Domain
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    # Grid
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    x = np.linspace(x_min, x_max - dx, Nx)  # periodic grid: last point = x_max - dx
    y = np.linspace(y_min, y_max - dy, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Time
    t_final = plan['time_stepping']['t_final']
    dt = plan['time_stepping'].get('dt', None)
    if dt is None:
        # Estimate dt by CFL for convection-diffusion (not needed here, but for robustness)
        c_x = pde_spec['parameters']['c_x']
        c_y = pde_spec['parameters']['c_y']
        nu = pde_spec['parameters']['nu']
        dt_adv = 0.5 * min(dx/abs(c_x) if c_x != 0 else np.inf, dy/abs(c_y) if c_y != 0 else np.inf)
        dt_diff = 0.25 * min(dx*dx, dy*dy) / nu
        dt = min(dt_adv, dt_diff)
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # PDE parameters
    c_x = pde_spec['parameters']['c_x']
    c_y = pde_spec['parameters']['c_y']
    nu = pde_spec['parameters']['nu']

    # --- Initial condition ---
    u = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)

    # --- Helper functions for periodic finite differences ---
    def periodic_roll(arr, shift, axis):
        return np.roll(arr, shift=shift, axis=axis)

    def diff_x(u):
        # Central difference, periodic, 2nd order
        return (periodic_roll(u, -1, axis=0) - periodic_roll(u, 1, axis=0)) / (2 * dx)

    def diff_y(u):
        return (periodic_roll(u, -1, axis=1) - periodic_roll(u, 1, axis=1)) / (2 * dy)

    def diff_xx(u):
        return (periodic_roll(u, -1, axis=0) - 2 * u + periodic_roll(u, 1, axis=0)) / (dx * dx)

    def diff_yy(u):
        return (periodic_roll(u, -1, axis=1) - 2 * u + periodic_roll(u, 1, axis=1)) / (dy * dy)

    # --- Build the implicit Crank-Nicolson operator ---
    # The equation: u_t + c_x u_x + c_y u_y = nu (u_xx + u_yy)
    # Crank-Nicolson: (I + dt/2 L) u^{n+1} = (I - dt/2 L) u^n
    # where L is the spatial operator (advection + diffusion)
    # We'll use matrix-free GMRES for the linear solve

    # Flattening: (i, j) -> k = i*Ny + j
    def L_op(u_grid):
        # L u = -c_x u_x - c_y u_y + nu (u_xx + u_yy)
        return (
            -c_x * diff_x(u_grid)
            -c_y * diff_y(u_grid)
            + nu * (diff_xx(u_grid) + diff_yy(u_grid))
        )

    def matvec_CN(u_vec):
        # (I + dt/2 L) u_vec, where u_vec is flattened
        u_grid = u_vec.reshape((Nx, Ny))
        return (u_grid + 0.5 * dt * L_op(u_grid)).ravel()

    def rhs_CN(u_grid):
        # (I - dt/2 L) u^n
        return (u_grid - 0.5 * dt * L_op(u_grid)).ravel()

    # --- GMRES solver (matrix-free, no preconditioning for memory safety) ---
    def gmres(A_mv, b, x0=None, tol=1e-8, maxiter=100):
        # Simple restarted GMRES, no preconditioning, for moderate grid sizes
        n = b.size
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
        r = b - A_mv(x)
        beta = np.linalg.norm(r)
        if beta < tol:
            return x
        V = np.zeros((n, maxiter+1), dtype=b.dtype)
        H = np.zeros((maxiter+1, maxiter), dtype=b.dtype)
        V[:,0] = r / beta
        for j in range(maxiter):
            w = A_mv(V[:,j])
            for i in range(j+1):
                H[i,j] = np.dot(V[:,i].conj(), w)
                w = w - H[i,j]*V[:,i]
            H[j+1,j] = np.linalg.norm(w)
            if H[j+1,j] != 0 and j+1 < maxiter:
                V[:,j+1] = w / H[j+1,j]
            # Solve least squares
            e1 = np.zeros(j+2, dtype=b.dtype)
            e1[0] = beta
            y, *_ = np.linalg.lstsq(H[:j+2,:j+1], e1, rcond=None)
            x_approx = x + V[:,:j+1] @ y
            res = np.linalg.norm(b - A_mv(x_approx))
            if res < tol:
                return x_approx
        return x_approx  # return best found

    # --- Time stepping loop ---
    u_flat = u.ravel()
    for n in range(Nt):
        b = rhs_CN(u_flat.reshape((Nx, Ny)))
        u_flat = gmres(matvec_CN, b, x0=u_flat, tol=1e-8, maxiter=30)
        # No storage of intermediate steps for memory safety

    u_final = u_flat.reshape((Nx, Ny))

    # --- Residual calculation ---
    # Compute pointwise residual: R = u_t + c_x u_x + c_y u_y - nu (u_xx + u_yy)
    # u_t ≈ (u_final - u_prev) / dt, but since we don't have u_prev, use backward difference
    # For residual, use the last time step: u_t ≈ (u_final - u_prev) / dt
    # We'll do one backward Euler step to get u_prev
    b_prev = u_final.ravel() - dt * L_op(u_final)
    u_prev = gmres(matvec_CN, b_prev, x0=u_final.ravel(), tol=1e-8, maxiter=30).reshape((Nx, Ny))
    u_t = (u_final - u_prev) / dt
    residual_grid = (
        u_t
        + c_x * diff_x(u_final)
        + c_y * diff_y(u_final)
        - nu * (diff_xx(u_final) + diff_yy(u_final))
    )

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual_grid
    }
```