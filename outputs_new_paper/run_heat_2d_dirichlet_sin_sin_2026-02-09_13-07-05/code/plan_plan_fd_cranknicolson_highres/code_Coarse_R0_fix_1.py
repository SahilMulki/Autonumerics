import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and plan parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    # Grid
    Nx = plan["spatial_discretization"]["Nx"]
    Ny = plan["spatial_discretization"]["Ny"]
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    coords = {"x": x, "y": y}

    # Time
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    alpha = pde_spec["parameters"]["alpha"]
    if dt is None:
        # Estimate dt by CFL (for explicit, but here for robustness)
        dt = 0.25 * min(dx, dy)**2 / alpha
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt  # adjust dt to hit t_final exactly
    t_array = np.linspace(0, Nt*dt, Nt+1)

    # Initial condition
    u = np.sin(np.pi * X) * np.sin(np.pi * Y)
    u_new = np.empty_like(u)

    # Dirichlet BCs: u=0 on all boundaries
    def apply_bc(U):
        U[0, :] = 0
        U[-1, :] = 0
        U[:, 0] = 0
        U[:, -1] = 0

    apply_bc(u)

    # --- Crank-Nicolson Setup ---
    rx = alpha * dt / (2 * dx**2)
    ry = alpha * dt / (2 * dy**2)

    # --- Time stepping ---
    tol = plan["time_stepping"]["extra_parameters"].get("tolerance", 1e-8)
    max_iter = 5000

    for n in range(Nt):
        # Right-hand side: B * u^n + boundary terms
        u_n = u[1:-1, 1:-1]
        u_xx = (u[2:, 1:-1] - 2*u_n + u[:-2, 1:-1]) / dx**2
        u_yy = (u[1:-1, 2:] - 2*u_n + u[1:-1, :-2]) / dy**2
        rhs = u_n + 0.5 * alpha * dt * (u_xx + u_yy)

        # Jacobi iteration for implicit solve
        u_guess = u_n.copy()
        for it in range(max_iter):
            u_new_in = rhs.copy()
            # x-direction neighbors
            u_new_in += rx * (np.roll(u_guess, 1, axis=0) + np.roll(u_guess, -1, axis=0))
            # y-direction neighbors
            u_new_in += ry * (np.roll(u_guess, 1, axis=1) + np.roll(u_guess, -1, axis=1))
            # Subtract the diagonal contributions (since we added them above)
            u_new_in -= 2*rx*u_guess + 2*ry*u_guess
            # Divide by diagonal
            u_new_in /= (1 + 2*rx + 2*ry)
            # Enforce Dirichlet BCs on the guess (interior only, so not needed)
            if np.linalg.norm(u_new_in - u_guess, ord=np.inf) < tol:
                break
            u_guess[:] = u_new_in
        u[1:-1, 1:-1] = u_new_in
        apply_bc(u)

    # --- Compute residual at final time ---
    # Residual: R = u_t - alpha*(u_xx + u_yy)
    # Approximate u_t by backward difference
    # For residual, we need u at t_final and t_final-dt

    # Rewind one step to get u_prev
    u_prev = np.sin(np.pi * X) * np.sin(np.pi * Y)
    apply_bc(u_prev)
    for n in range(Nt-1):
        u_n = u_prev[1:-1, 1:-1]
        u_xx = (u_prev[2:, 1:-1] - 2*u_n + u_prev[:-2, 1:-1]) / dx**2
        u_yy = (u_prev[1:-1, 2:] - 2*u_n + u_prev[1:-1, :-2]) / dy**2
        rhs = u_n + 0.5 * alpha * dt * (u_xx + u_yy)
        u_guess = u_n.copy()
        for it in range(max_iter):
            u_new_in = rhs.copy()
            u_new_in += rx * (np.roll(u_guess, 1, axis=0) + np.roll(u_guess, -1, axis=0))
            u_new_in += ry * (np.roll(u_guess, 1, axis=1) + np.roll(u_guess, -1, axis=1))
            u_new_in -= 2*rx*u_guess + 2*ry*u_guess
            u_new_in /= (1 + 2*rx + 2*ry)
            if np.linalg.norm(u_new_in - u_guess, ord=np.inf) < tol:
                break
            u_guess[:] = u_new_in
        u_prev[1:-1, 1:-1] = u_new_in
        apply_bc(u_prev)

    # Now u_prev is at t = t_final - dt, u is at t_final
    # Compute residual at all grid points
    u_t = (u - u_prev) / dt
    u_xx = np.zeros_like(u)
    u_yy = np.zeros_like(u)
    # Second derivatives, central difference, interior
    u_xx[1:-1, :] = (u[2:, :] - 2*u[1:-1, :] + u[:-2, :]) / dx**2
    u_yy[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]) / dy**2
    residual = u_t - alpha * (u_xx + u_yy)

    return {
        "u": u.copy(),
        "coords": coords,
        "t": t_array
    }