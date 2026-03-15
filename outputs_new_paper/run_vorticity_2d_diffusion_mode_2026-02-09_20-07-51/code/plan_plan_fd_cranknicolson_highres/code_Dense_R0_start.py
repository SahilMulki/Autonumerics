import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and Plan Parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    t_min, t_max = pde_spec["domain"]["bounds"]["t"]
    nu = float(pde_spec["parameters"]["nu"])

    # Grid
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny

    # Time
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt by CFL for diffusion: dt <= 0.25 * min(dx,dy)^2 / nu
        dt = 0.25 * min(dx, dy)**2 / nu
    t_final = plan["time_stepping"].get("t_final", t_max)
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil((t_final - t_min) / dt))
        dt = (t_final - t_min) / Nt  # adjust dt to hit t_final exactly
    else:
        dt = (t_final - t_min) / Nt
    t_array = np.linspace(t_min, t_final, Nt+1)

    # Coordinates (periodic grid: last point == first point, so use endpoint=False)
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial Condition ---
    # omega(x, y, 0) = sin(2*pi*x) * sin(2*pi*y)
    u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    # --- Crank-Nicolson Setup ---
    rx = nu * dt / (2 * dx**2)
    ry = nu * dt / (2 * dy**2)

    # Helper: roll for periodic BCs
    def laplacian_periodic(U):
        return (
            (np.roll(U, -1, axis=0) - 2*U + np.roll(U, 1, axis=0)) / dx**2 +
            (np.roll(U, -1, axis=1) - 2*U + np.roll(U, 1, axis=1)) / dy**2
        )

    # --- Matrix-free iterative solver (Jacobi) for (I - rx*Lx - ry*Ly) u^{n+1} = rhs ---
    def crank_nicolson_step(u0, rhs, rx, ry, maxiter=100, tol=1e-8):
        u_new = u0.copy()
        for _ in range(maxiter):
            u_old = u_new.copy()
            # Jacobi update (matrix-free, periodic)
            u_new = (
                rhs
                + rx * (np.roll(u_old, 1, axis=0) + np.roll(u_old, -1, axis=0))
                + ry * (np.roll(u_old, 1, axis=1) + np.roll(u_old, -1, axis=1))
            ) / (1 + 2*rx + 2*ry)
            # Check convergence
            if np.linalg.norm(u_new - u_old, ord=np.inf) < tol:
                break
        return u_new

    # --- Time-stepping loop ---
    for n in range(Nt):
        # For Crank-Nicolson, the RHS is:
        # rhs = u + rx * (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0))
        #           + ry * (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1))
        rhs = (
            u
            + rx * (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0))
            + ry * (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1))
        )
        # Solve (I - rx*Lx - ry*Ly) u^{n+1} = rhs
        u = crank_nicolson_step(u, rhs, rx, ry, maxiter=200, tol=1e-10)

    # --- Compute Residual Grid ---
    # PDE: omega_t = nu * (omega_xx + omega_yy)
    # Approximate omega_t ≈ (u_final - u_prev) / dt
    # For residual, use backward difference for time derivative at final step
    u_final = u.copy()
    # One step backward in time (approximate u_prev)
    rhs_prev = (
        u_final
        - rx * (np.roll(u_final, -1, axis=0) - 2*u_final + np.roll(u_final, 1, axis=0))
        - ry * (np.roll(u_final, -1, axis=1) - 2*u_final + np.roll(u_final, 1, axis=1))
    )
    u_prev = crank_nicolson_step(u_final, rhs_prev, -rx, -ry, maxiter=200, tol=1e-10)

    omega_t = (u_final - u_prev) / dt
    omega_xx_yy = laplacian_periodic(u_final)
    residual_grid = omega_t - nu * omega_xx_yy

    # L2 norm of residual over the domain
    residual = np.sqrt(np.mean(residual_grid**2))

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual
    }