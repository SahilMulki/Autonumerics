import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    Lx = pde_spec["domain"]["bounds"]["x"][1] - pde_spec["domain"]["bounds"]["x"][0]
    Ly = pde_spec["domain"]["bounds"]["y"][1] - pde_spec["domain"]["bounds"]["y"][0]
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    c_x = float(pde_spec["parameters"]["c_x"])
    c_y = float(pde_spec["parameters"]["c_y"])
    nu = float(pde_spec["parameters"]["nu"])
    initial_condition = pde_spec["initial_condition"]

    # --- Extract plan parameters ---
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])
    dt = float(plan["time_stepping"].get("dt", 0.0))
    t_final = float(plan["time_stepping"].get("t_final", 1.0))
    order = int(plan["spatial_discretization"].get("order", 2))

    # --- Set up grid ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    dx = Lx / Nx
    dy = Ly / Ny
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Time stepping ---
    if dt <= 0.0:
        # Estimate dt by CFL
        cfl = 0.5
        dt_adv = cfl * min(dx/abs(c_x) if c_x != 0 else np.inf,
                           dy/abs(c_y) if c_y != 0 else np.inf)
        dt_diff = cfl * min(dx**2, dy**2) / (4*nu)
        dt = min(dt_adv, dt_diff)
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Initial condition ---
    # u0 = sin(2*pi*x) * cos(2*pi*y)
    u = np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
    u_prev = u.copy()  # for BDF2

    # --- FEM matrices (mass, stiffness, convection) ---
    # Mass matrix (lumped): just dx*dy per node
    mass = dx * dy

    # Laplacian (diffusion) operator: 5-point stencil, periodic
    def laplacian(U):
        return (
            (np.roll(U, -1, axis=0) - 2*U + np.roll(U, 1, axis=0)) / dx**2 +
            (np.roll(U, -1, axis=1) - 2*U + np.roll(U, 1, axis=1)) / dy**2
        )

    # Convection operator: upwinded 1st order for stability (periodic)
    def convection(U):
        # Upwind in x
        if c_x >= 0:
            dUdx = (U - np.roll(U, 1, axis=0)) / dx
        else:
            dUdx = (np.roll(U, -1, axis=0) - U) / dx
        # Upwind in y
        if c_y >= 0:
            dUdy = (U - np.roll(U, 1, axis=1)) / dy
        else:
            dUdy = (np.roll(U, -1, axis=1) - U) / dy
        return c_x * dUdx + c_y * dUdy

    # --- IMEX BDF2 time stepping ---
    # Precompute Laplacian operator in Fourier space for implicit solve (periodic grid)
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    lapl_eig = -(KX**2 + KY**2)  # eigenvalues of Laplacian

    # For implicit solve: (3/2/dt - nu*lapl_eig) in Fourier space
    def implicit_solve(rhs):
        rhs_hat = np.fft.fft2(rhs)
        denom = (3/(2*dt)) - nu * lapl_eig
        u_hat = rhs_hat / denom
        u_new = np.fft.ifft2(u_hat).real
        return u_new

    # --- Time stepping loop ---
    # For storing previous two steps for BDF2
    u_nm1 = u_prev.copy()  # u^{n-1}
    u_n = u.copy()         # u^{n}
    for n in range(Nt):
        t = t_array[n]
        if n == 0:
            # Backward Euler: (u1 - u0)/dt = -convection(u0) + nu*laplacian(u1)
            rhs = u_n + dt * (-convection(u_n))
            denom = (1/dt) - nu * lapl_eig
            rhs_hat = np.fft.fft2(rhs/dt)
            u_hat = rhs_hat / denom
            u_np1 = np.fft.ifft2(u_hat).real
        else:
            # BDF2 IMEX:
            # (3u^{n+1} - 4u^n + u^{n-1})/(2dt) = -convection(u^n) + nu*laplacian(u^{n+1})
            rhs = (4*u_n - u_nm1)/2 + dt * (-convection(u_n))
            u_np1 = implicit_solve(rhs/dt)
        # Update for next step
        u_nm1, u_n = u_n, u_np1

    u = u_n
    u_prev = u_nm1

    # --- Compute residual at final time ---
    # Residual: R = u_t + c_x u_x + c_y u_y - nu (u_xx + u_yy)
    # Approximate u_t at final time using backward difference
    u_t = (u - u_prev) / dt

    # Central differences for residual (higher accuracy)
    def central_diff_x(U):
        return (np.roll(U, -1, axis=0) - np.roll(U, 1, axis=0)) / (2*dx)
    def central_diff_y(U):
        return (np.roll(U, -1, axis=1) - np.roll(U, 1, axis=1)) / (2*dy)
    def central_laplacian(U):
        return (
            (np.roll(U, -1, axis=0) - 2*U + np.roll(U, 1, axis=0)) / dx**2 +
            (np.roll(U, -1, axis=1) - 2*U + np.roll(U, 1, axis=1)) / dy**2
        )

    u_x = central_diff_x(u)
    u_y = central_diff_y(u)
    u_xx_yy = central_laplacian(u)
    residual_grid = u_t + c_x * u_x + c_y * u_y - nu * u_xx_yy

    # --- Output ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual_grid
    }