import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    D = float(pde_spec['parameters']['D'])
    r = float(pde_spec['parameters']['r'])
    domain = pde_spec['domain']
    x_min, x_max = domain['bounds']['x']
    y_min, y_max = domain['bounds']['y']
    Nx = int(plan['spatial_discretization']['Nx'])
    Ny = int(plan['spatial_discretization']['Ny'])
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)

    # --- Time stepping parameters ---
    t_final = float(plan['time_stepping']['t_final'])
    dt = plan['time_stepping'].get('dt', None)
    if dt is None:
        # Estimate dt by CFL for diffusion: dt <= 0.25 * min(dx^2, dy^2) / D
        dt = 0.25 * min(dx**2, dy**2) / D
    dt = float(dt)
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # Adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Grids ---
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial condition ---
    u = np.sin(np.pi * X) * np.sin(np.pi * Y)
    u_new = np.zeros_like(u)

    # --- Dirichlet BCs: u=0 on all boundaries ---
    def apply_bc(U):
        U[0, :] = 0
        U[-1, :] = 0
        U[:, 0] = 0
        U[:, -1] = 0

    apply_bc(u)

    # --- Crank-Nicolson coefficients ---
    rx = D * dt / (2 * dx**2)
    ry = D * dt / (2 * dy**2)
    rr = r * dt / 2

    # --- Jacobi iterative solver for the implicit system ---
    def jacobi_step(u_old, rhs, maxiter=100, tol=1e-6):
        u = u_old.copy()
        denom = (1 + 2*rx + 2*ry + rr)
        for it in range(maxiter):
            u_prev = u.copy()
            # Update interior points only
            u[1:-1,1:-1] = (
                rhs[1:-1,1:-1]
                + rx * (u[2:,1:-1] + u[:-2,1:-1])
                + ry * (u[1:-1,2:] + u[1:-1,:-2])
            ) / denom
            apply_bc(u)
            if np.linalg.norm(u-u_prev, ord=np.inf) < tol:
                break
        return u

    # --- Time stepping loop ---
    for n in range(Nt):
        # Right-hand side (explicit half-step)
        rhs = np.zeros_like(u)
        rhs[1:-1,1:-1] = (
            u[1:-1,1:-1]
            + rx * (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1])
            + ry * (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2])
            + rr * u[1:-1,1:-1]
        )
        # Jacobi solve for implicit half-step
        u_new = jacobi_step(u, rhs, maxiter=200, tol=1e-7)
        u = u_new

    # --- Analytic solution for residual ---
    # analytic: exp((r-2*D*pi^2)*t)*sin(pi*x)*sin(pi*y)
    t = t_final
    analytic = np.exp((r - 2*D*np.pi**2)*t) * np.sin(np.pi * X) * np.sin(np.pi * Y)
    # L2 norm of error (residual)
    error = u - analytic
    residual = np.sqrt(np.sum(error**2) * dx * dy)

    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual
    }