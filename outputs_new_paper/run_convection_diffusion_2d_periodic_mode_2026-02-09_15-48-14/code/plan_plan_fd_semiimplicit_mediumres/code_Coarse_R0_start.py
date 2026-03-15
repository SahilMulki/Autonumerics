```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    Lx = pde_spec['domain']['bounds']['x'][1] - pde_spec['domain']['bounds']['x'][0]
    Ly = pde_spec['domain']['bounds']['y'][1] - pde_spec['domain']['bounds']['y'][0]
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    c_x = float(pde_spec['parameters']['c_x'])
    c_y = float(pde_spec['parameters']['c_y'])
    nu = float(pde_spec['parameters']['nu'])
    bc_type = pde_spec['boundary_conditions']['type']
    assert bc_type == 'periodic', "Only periodic BCs implemented"

    # --- Extract Plan parameters ---
    Nx = int(plan['spatial_discretization']['Nx'])
    Ny = int(plan['spatial_discretization']['Ny'])
    dx = Lx / Nx
    dy = Ly / Ny

    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    Nt = plan['time_stepping'].get('Nt', None)
    if dt is None:
        # Estimate dt by CFL (convection + diffusion)
        cfl = 0.4
        dt_conv = cfl * min(dx/abs(c_x) if c_x!=0 else np.inf, dy/abs(c_y) if c_y!=0 else np.inf)
        dt_diff = 0.25 * min(dx*dx, dy*dy) / nu
        dt = min(dt_conv, dt_diff)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt
    else:
        t_final = Nt * dt

    # --- Grids ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial Condition ---
    u0 = np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
    u = u0.copy()

    # --- Precompute Laplacian operator (for implicit diffusion) ---
    # 2D periodic Laplacian, Kronecker sum of 1D periodic Laplacians
    # We'll use FFT for implicit solve (since periodic, diagonal in Fourier space)
    # But since plan says CG, let's implement matrix-free CG for the implicit step

    # --- Helper functions ---
    def periodic_roll(arr, shift, axis):
        return np.roll(arr, shift, axis=axis)

    def laplacian(u):
        # 2nd order central difference, periodic
        return (
            (periodic_roll(u, -1, axis=0) - 2*u + periodic_roll(u, 1, axis=0)) / dx**2 +
            (periodic_roll(u, -1, axis=1) - 2*u + periodic_roll(u, 1, axis=1)) / dy**2
        )

    def convection(u):
        # 2nd order central difference, periodic
        ux = (periodic_roll(u, -1, axis=0) - periodic_roll(u, 1, axis=0)) / (2*dx)
        uy = (periodic_roll(u, -1, axis=1) - periodic_roll(u, 1, axis=1)) / (2*dy)
        return c_x * ux + c_y * uy

    # --- CG solver for (I - dt*nu*L) u = rhs ---
    def cg_solve(rhs, u_init, tol=1e-10, maxiter=100):
        u = u_init.copy()
        def A(v):
            return v - dt*nu*laplacian(v)
        r = rhs - A(u)
        p = r.copy()
        rsold = np.sum(r*r)
        for i in range(maxiter):
            Ap = A(p)
            alpha = rsold / np.sum(p*Ap)
            u += alpha * p
            r -= alpha * Ap
            rsnew = np.sum(r*r)
            if np.sqrt(rsnew) < tol:
                break
            p = r + (rsnew/rsold)*p
            rsold = rsnew
        return u

    # --- Time stepping ---
    t_array = np.linspace(0, t_final, Nt+1)
    u_curr = u.copy()
    u_new = np.empty_like(u)
    for n in range(Nt):
        # Explicit convection
        conv = convection(u_curr)
        rhs = u_curr - dt * conv
        # Implicit diffusion: (I - dt*nu*L) u^{n+1} = rhs
        u_new = cg_solve(rhs, u_curr, tol=1e-10, maxiter=100)
        u_curr, u_new = u_new, u_curr  # swap for next step

    u_final = u_curr

    # --- Residual calculation ---
    # Compute all terms at t_final
    # u_t ≈ (u_final - u_prev) / dt (backward difference)
    # For residual, use last time step
    # If Nt==0, set residual to zeros
    if Nt > 0:
        # Recompute u_prev (one step back)
        # We need u at t_{N-1}
        u_prev = u_final.copy()
        # Step backward: u_prev = (I - dt*nu*L)^{-1} (u_final + dt*convection(u_final))
        # But that's not trivial; instead, rerun the last step
        # So, rerun from t_{N-1} to t_N
        # To save memory, we can rerun the last step from u_{N-1} to u_N
        # But we already have u_{N-1} in the previous loop, so let's store it
        # To avoid OOM, only store u_prev for last step
        # So, rerun time loop, but only keep last two steps
        u_prev = u0.copy()
        u_tmp = np.empty_like(u0)
        for n in range(Nt-1):
            conv = convection(u_prev)
            rhs = u_prev - dt * conv
            u_tmp = cg_solve(rhs, u_prev, tol=1e-10, maxiter=100)
            u_prev, u_tmp = u_tmp, u_prev
        # Now u_prev is u^{N-1}
        u_t = (u_final - u_prev) / dt
    else:
        u_t = np.zeros_like(u_final)

    conv_term = convection(u_final)
    diff_term = laplacian(u_final)
    # PDE: u_t + c_x u_x + c_y u_y - nu(u_xx + u_yy) = 0
    residual = u_t + conv_term - nu * diff_term

    # --- Output ---
    result = {
        "u": u_final,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual
    }
    return result
```
