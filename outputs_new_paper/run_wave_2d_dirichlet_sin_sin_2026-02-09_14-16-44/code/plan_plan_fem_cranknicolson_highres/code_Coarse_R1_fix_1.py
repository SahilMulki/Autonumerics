import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and Plan Parameters ---
    # Domain
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    t_min, t_max = pde_spec['domain']['bounds']['t']
    c = float(pde_spec['parameters']['c'])

    # FEM mesh
    Nx = int(plan['spatial_discretization']['Nx'])
    Ny = int(plan['spatial_discretization']['Ny'])
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)

    # Time stepping
    dt = float(plan['time_stepping'].get('dt', None))
    t_final = float(plan['time_stepping'].get('t_final', t_max))
    Nt = plan['time_stepping'].get('Nt', None)
    if Nt is None:
        Nt = int(np.ceil((t_final - t_min) / dt))
    else:
        Nt = int(Nt)
        dt = (t_final - t_min) / Nt
    t_array = np.linspace(t_min, t_final, Nt + 1)

    # --- Generate Mesh ---
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial Conditions ---
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)  # displacement
    v0 = np.zeros_like(u0)                      # velocity

    # --- Helper: Dirichlet BC mask ---
    def apply_dirichlet(u, t=0.0):
        u[0, :] = 0.0
        u[-1, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        return u

    # --- Laplacian operator (5-point stencil) ---
    def laplace(u):
        lap = np.zeros_like(u)
        lap[1:-1,1:-1] = (
            (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]) / dx**2 +
            (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]) / dy**2
        )
        return lap

    # --- Allocate arrays ---
    u_nm1 = u0.copy()
    u_n = u0.copy()
    u_np1 = np.zeros_like(u0)

    # --- First step: u^1 using Taylor expansion ---
    lap_u0 = laplace(u0)
    u_n = apply_dirichlet(u_n)
    u_np1 = u0 + dt * v0 + 0.5 * dt**2 * c**2 * lap_u0
    u_np1 = apply_dirichlet(u_np1)

    # Store previous two time levels for residual computation
    u_hist_nm1 = u0.copy()
    u_hist_n = u_np1.copy()

    # --- Time stepping loop ---
    for n in range(1, Nt):
        rhs = (2.0 / dt**2) * u_n - (1.0 / dt**2) * u_nm1 + 0.5 * c**2 * laplace(u_nm1)
        u_guess = u_n.copy()
        for _ in range(8):  # Jacobi iterations
            lap_u_guess = laplace(u_guess)
            u_new = np.zeros_like(u_guess)
            u_new[1:-1,1:-1] = (
                rhs[1:-1,1:-1] + 0.5 * c**2 * lap_u_guess[1:-1,1:-1]
            ) / (1.0 / dt**2)
            u_new = apply_dirichlet(u_new)
            u_guess = u_new
        u_np1 = u_guess

        # Rotate time levels
        u_nm1, u_n = u_n, u_np1

        # For residual at final time, keep last three time levels
        if n == Nt - 2:
            u_hist_nm1 = u_nm1.copy()
            u_hist_n = u_n.copy()
        if n == Nt - 1:
            u_hist_np1 = u_np1.copy()

    u = u_np1.copy()  # Final solution at t = t_final

    # --- Compute Residual Grid ---
    # Residual: R = u_tt - c^2*(u_xx + u_yy)
    # Approximate u_tt with backward difference at final time:
    #   u_tt ≈ (u_np1 - 2u_n + u_nm1) / dt^2
    # u_xx + u_yy via laplace(u)
    if Nt >= 2:
        u_tt = (u_hist_np1 - 2*u_hist_n + u_hist_nm1) / dt**2
        lap_u = laplace(u)
        residual = u_tt - c**2 * lap_u
    else:
        residual = np.zeros_like(u)

    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": t_array
    }